//! Terrain tiles over HTTP.
//!
//! A browser cannot wait for a tile, so everything here is a promise: a
//! request starts a fetch and the tile turns up in a queue when the fetch
//! resolves. Unpacking a tile takes a few milliseconds, which is done when
//! the renderer takes it from the queue, a few a frame, rather than whenever
//! the network happens to deliver.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};
use relief_render::camera::{GroundTiles, TileHeights};
use relief_render::renderer::clipmap::{derive_from_parent, derive_from_whole, Prepared, TileFeed, TileKey};
use relief_terrain::pack::{self, TileIndex};
use relief_terrain::terrain::TILED_LEVELS;
use relief_terrain::{Grid, Terrain, Tile};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::{spawn_local, JsFuture};

/// Tiles kept to derive the tiles below them from.
const PARENT_CACHE_TILES: usize = 48;
/// The camera's ground keeps tiles within this many of the newest one.
const GROUND_REACH_TILES: usize = 5;

fn js_error(value: JsValue) -> anyhow::Error {
    anyhow!("{}", value.as_string().unwrap_or_else(|| format!("{value:?}")))
}

/// The bytes at a URL.
pub(crate) async fn fetch(url: &str) -> Result<Vec<u8>> {
    let window = web_sys::window().ok_or_else(|| anyhow!("not running in a page"))?;
    let response: web_sys::Response =
        JsFuture::from(window.fetch_with_str(url)).await.map_err(js_error)?.dyn_into().map_err(js_error)?;
    ensure!(response.ok(), "{url}: HTTP {}", response.status());
    let buffer = JsFuture::from(response.array_buffer().map_err(js_error)?).await.map_err(js_error)?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// A tile that has arrived: still packed if it came off the network, or
/// made up where the terrain has none.
enum Arrived {
    Packed(Vec<u8>),
    Derived(Rc<Tile>),
}

/// What the feed, its fetches and the camera's ground share.
struct Shared {
    /// Where the packed terrain is served from, without a trailing slash.
    url: String,
    index: TileIndex,
    /// The first whole level of both pyramids, from which tiles without data
    /// are derived.
    whole: [Grid<f32>; 2],
    /// Cell size of level 0 in metres.
    cell: f32,
    /// The level whose tiles the camera's ground keeps.
    ground_level: usize,
    arrived: RefCell<VecDeque<(TileKey, Result<Arrived>)>>,
    parents: RefCell<HashMap<TileKey, Rc<Tile>>>,
    ground: RefCell<HashMap<[usize; 2], TileHeights>>,
    /// Bytes fetched so far, for the curious.
    fetched: Cell<u64>,
}

impl Shared {
    fn has(&self, level: usize, tile: [i32; 2]) -> bool {
        tile[0] >= 0 && tile[1] >= 0 && self.index.has(level, tile[0] as usize, tile[1] as usize)
    }

    async fn packed(&self, level: usize, tile: [i32; 2]) -> Result<Vec<u8>> {
        let url = format!("{}/tiles/{level}/{}_{}.bin", self.url, tile[0], tile[1]);
        // One more try rides out a dropped connection.
        let bytes = match fetch(&url).await {
            Ok(bytes) => bytes,
            Err(_) => fetch(&url).await?,
        };
        self.fetched.set(self.fetched.get() + bytes.len() as u64);
        Ok(bytes)
    }

    /// The tile as it arrives: off the network, or derived from the nearest
    /// level above that has the place.
    async fn load(&self, level: usize, tile: [i32; 2]) -> Result<Arrived> {
        if self.has(level, tile) {
            return self.packed(level, tile).await.map(Arrived::Packed);
        }
        // Climb until something is there, then come back down.
        let mut missing = vec![];
        let (mut above, mut at) = (level, tile);
        let mut found = loop {
            if let Some(known) = self.parents.borrow().get(&(above, at)) {
                break known.clone();
            }
            if self.has(above, at) {
                let parent = Rc::new(pack::decode_tile(&self.packed(above, at).await?, above)?);
                self.remember(above, at, &parent);
                break parent;
            }
            if above + 1 == TILED_LEVELS {
                let parent = Rc::new(derive_from_whole(&self.whole, at));
                self.remember(above, at, &parent);
                break parent;
            }
            missing.push((above, at));
            (above, at) = (above + 1, [at[0] >> 1, at[1] >> 1]);
        };
        for (level, tile) in missing.into_iter().rev() {
            found = Rc::new(derive_from_parent(&found, tile));
            self.remember(level, tile, &found);
        }
        Ok(Arrived::Derived(found))
    }

    /// A tile made up from the whole levels alone.
    fn made_up(&self, level: usize, tile: [i32; 2]) -> Tile {
        let ancestor = |of: usize| [tile[0] >> (of - level), tile[1] >> (of - level)];
        let mut data = derive_from_whole(&self.whole, ancestor(TILED_LEVELS - 1));
        for of in (level..TILED_LEVELS - 1).rev() {
            data = derive_from_parent(&data, ancestor(of));
        }
        data
    }

    fn remember(&self, level: usize, tile: [i32; 2], data: &Rc<Tile>) {
        let mut parents = self.parents.borrow_mut();
        if parents.len() >= PARENT_CACHE_TILES {
            parents.clear();
        }
        parents.insert((level, tile), data.clone());
    }

    /// Keeps a tile's heights for the camera, and lets go of those far from
    /// it: the camera has moved on.
    fn keep_for_ground(&self, tile: [i32; 2], data: &Tile) {
        let [x, y] = tile.map(|index| index.max(0) as usize);
        let mut ground = self.ground.borrow_mut();
        ground.retain(|&[kx, ky], _| kx.abs_diff(x) <= GROUND_REACH_TILES && ky.abs_diff(y) <= GROUND_REACH_TILES);
        ground.insert([x, y], Arc::new(data.plain.clone()));
    }
}

/// The feed of a packed terrain served over HTTP.
pub struct FetchFeed(Rc<Shared>);

impl FetchFeed {
    /// `url` is where the packed terrain is served from; `ground_level` the
    /// level whose tiles are also kept for the camera.
    pub fn new(terrain: &Terrain, url: &str, ground_level: usize) -> Result<Self> {
        let Some(index) = terrain.index.clone() else { bail!("the terrain is not packed for the web") };
        Ok(Self(Rc::new(Shared {
            url: url.trim_end_matches('/').to_owned(),
            index,
            whole: [terrain.coarse[0][0].clone(), terrain.coarse[1][0].clone()],
            cell: terrain.meta.cell_size,
            ground_level,
            arrived: RefCell::new(VecDeque::new()),
            parents: RefCell::new(HashMap::new()),
            ground: RefCell::new(HashMap::new()),
            fetched: Cell::new(0),
        })))
    }

    /// A handle for whoever wants to know how the feed is doing.
    pub fn monitor(&self) -> FeedMonitor {
        FeedMonitor(self.0.clone())
    }

    /// The camera's view of the tiles that have passed through.
    pub fn ground(&self) -> Box<dyn GroundTiles> {
        Box::new(FeedGround(self.0.clone()))
    }
}

impl TileFeed for FetchFeed {
    fn request(&mut self, level: usize, tile: [i32; 2]) {
        let shared = self.0.clone();
        spawn_local(async move {
            let arrived = shared.load(level, tile).await;
            shared.arrived.borrow_mut().push_back(((level, tile), arrived));
        });
    }

    fn ready(&mut self) -> Option<Result<Prepared>> {
        let shared = &self.0;
        let ((level, tile), arrived) = shared.arrived.borrow_mut().pop_front()?;
        let prepare = |data: &Tile| {
            if level == shared.ground_level {
                shared.keep_for_ground(tile, data);
            }
            Prepared::new(level, tile, data, shared.cell)
        };
        Some(match arrived {
            Ok(Arrived::Packed(bytes)) => pack::decode_tile(&bytes, level).map(|data| prepare(&data)),
            Ok(Arrived::Derived(data)) => Ok(prepare(&data)),
            // A tile that cannot be had is made up from the whole levels,
            // so the map stays complete; it is only soft there.
            Err(error) => {
                log::warn!("level {level} tile {tile:?}: {error:#}");
                Ok(prepare(&shared.made_up(level, tile)))
            }
        })
    }

    fn wait(&mut self) -> Result<Prepared> {
        bail!("a browser cannot wait for a tile")
    }
}

/// Reads how a feed is doing.
pub struct FeedMonitor(Rc<Shared>);

impl FeedMonitor {
    /// Whether tiles have arrived that the renderer has not taken yet.
    pub fn has_arrivals(&self) -> bool {
        !self.0.arrived.borrow().is_empty()
    }

    pub fn fetched_bytes(&self) -> u64 {
        self.0.fetched.get()
    }
}

struct FeedGround(Rc<Shared>);

impl GroundTiles for FeedGround {
    fn heights(&self, tile: [usize; 2]) -> Option<TileHeights> {
        self.0.ground.borrow().get(&tile).cloned()
    }
}
