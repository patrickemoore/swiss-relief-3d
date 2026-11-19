//! The streamed part of the terrain: a clipmap of tiles around the camera.
//!
//! A country at 10 m is far too large for one texture, but a frame never
//! needs all of it at full detail: the finest scale a pixel can show grows
//! with distance. So each of the finest pyramid levels keeps only a square
//! window of tiles centred on the camera, every window as many texels wide
//! as the last and therefore twice as far-reaching. The coarser levels are
//! small and stay whole.
//!
//! A window is only loaded while the camera is close enough to the ground
//! for its level to show, and only as far as it is asked to reach: all of
//! it from a disk, less in a browser, where every tile is a download.
//!
//! Windows move a tile at a time. A window gives up the tiles it leaves
//! before their slots are overwritten, and takes in a new row or column only
//! once all of it is loaded, so the part of each level the shader may sample
//! is always a rectangle of finished tiles. Where a level is not loaded, the
//! shader falls back to the next-coarser one.
//!
//! Tiles come from a [`TileFeed`]: read from disk on worker threads, or
//! fetched over HTTP in a browser. Either way the render thread only copies
//! finished tiles to the GPU, so flying never waits for them; detail simply
//! arrives a few frames later.

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex};

use anyhow::Result;
use glam::Vec2;
use half::f16;
use relief_terrain::terrain::{TILED_LEVELS, TILE_SIZE, TILE_SPAN};
#[cfg(not(target_arch = "wasm32"))]
use relief_terrain::TileSource;
use relief_terrain::{Grid, Terrain, Tile};

/// Tiles along one side of a window as a disk can afford it.
pub const FULL_REACH: usize = 8;

/// Texels along one side of a window that reaches `reach` tiles.
pub fn window_texels(reach: usize) -> u32 {
    (reach * TILE_SIZE) as u32
}
/// Layers of the height and gradient arrays: level 0, which both pyramids
/// share, then the other levels of the plain and of the accentuated pyramid.
const PYRAMID_LAYERS: u32 = 2 * TILED_LEVELS as u32 - 1;
/// How far, in tiles, the camera may stray from the centre of a window
/// before the window follows. More than half a tile, so that a camera
/// hovering over a tile boundary does not move the window back and forth.
const RECENTRE_DISTANCE: f32 = 0.65;
/// Synthesized tiles kept for the tiles below them to be derived from.
#[cfg(not(target_arch = "wasm32"))]
const PARENT_CACHE_TILES: usize = 64;
/// How far ahead of the camera a window is centred, as a share of its side,
/// when the camera looks toward the horizon. What a tilted camera sees lies
/// in front of it, so that is where the detail goes; a camera looking
/// straight down keeps its windows below it.
const LOOK_AHEAD: f32 = 0.22;

/// Farther than any terrain reaches, in metres.
const OUT_OF_REACH_M: f32 = 1e9;

/// A tile's level and its index in that level.
pub type TileKey = (usize, [i32; 2]);

/// A rectangle of tiles; `max` is exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRect {
    pub min: [i32; 2],
    pub max: [i32; 2],
}

impl TileRect {
    pub const EMPTY: Self = Self { min: [0, 0], max: [0, 0] };

    pub fn is_empty(&self) -> bool {
        self.min[0] >= self.max[0] || self.min[1] >= self.max[1]
    }

    fn intersection(&self, other: &Self) -> Self {
        let rect = Self {
            min: [self.min[0].max(other.min[0]), self.min[1].max(other.min[1])],
            max: [self.max[0].min(other.max[0]), self.max[1].min(other.max[1])],
        };
        if rect.is_empty() {
            Self::EMPTY
        } else {
            rect
        }
    }

    pub fn contains(&self, tile: [i32; 2]) -> bool {
        (self.min[0]..self.max[0]).contains(&tile[0]) && (self.min[1]..self.max[1]).contains(&tile[1])
    }

    pub fn tiles(&self) -> impl Iterator<Item = [i32; 2]> + '_ {
        (self.min[1]..self.max[1]).flat_map(move |y| (self.min[0]..self.max[0]).map(move |x| [x, y]))
    }
}

/// The window of one level.
struct Window {
    /// Tiles along one side of the part that is loaded.
    reach: i32,
    /// Where the window wants to be, before clipping to the terrain. `None`
    /// until the first update.
    origin: Option<[i32; 2]>,
    /// The tiles the window should hold.
    target: TileRect,
    /// The tiles that are uploaded and may be sampled.
    loaded: TileRect,
    /// Tiles still to upload before `loaded` can grow to `growth`.
    pending: Vec<[i32; 2]>,
    growth: TileRect,
}

impl Window {
    fn new(reach: i32) -> Self {
        Self {
            reach,
            origin: None,
            target: TileRect::EMPTY,
            loaded: TileRect::EMPTY,
            pending: vec![],
            growth: TileRect::EMPTY,
        }
    }

    /// Moves the target to follow the camera, given in tiles of this level.
    fn follow(&mut self, camera: Vec2, terrain: TileRect) {
        let reach = self.reach;
        let centred = |origin: [i32; 2]| {
            let centre = Vec2::new(origin[0] as f32, origin[1] as f32) + 0.5 * reach as f32;
            (camera - centre).abs().max_element() <= RECENTRE_DISTANCE
        };
        if self.origin.is_some_and(centred) {
            return;
        }
        let origin = [camera.x.round() as i32 - reach / 2, camera.y.round() as i32 - reach / 2];
        self.origin = Some(origin);
        let window = TileRect { min: origin, max: [origin[0] + reach, origin[1] + reach] };
        self.target = window.intersection(&terrain);
        // Tiles outside the new target are given up at once: their slots are
        // about to be reused.
        self.loaded = self.loaded.intersection(&self.target);
        self.pending.clear();
    }

    /// Chooses what to load next: the row or column that extends the loaded
    /// tiles toward the camera's side of the target, or a first tile.
    fn plan(&mut self, camera: Vec2) {
        if self.loaded.is_empty() {
            let clamp = |value: f32, axis: usize| {
                (value.floor() as i32).clamp(self.target.min[axis], self.target.max[axis] - 1)
            };
            let seed = [clamp(camera.x, 0), clamp(camera.y, 1)];
            self.growth = TileRect { min: seed, max: [seed[0] + 1, seed[1] + 1] };
            self.pending = vec![seed];
            return;
        }
        let (loaded, target) = (self.loaded, self.target);
        // Each side that can still grow, with the camera's distance to it.
        let sides = [
            (camera.x - loaded.min[0] as f32, loaded.min[0] > target.min[0], 0, -1),
            (loaded.max[0] as f32 - camera.x, loaded.max[0] < target.max[0], 0, 1),
            (camera.y - loaded.min[1] as f32, loaded.min[1] > target.min[1], 1, -1),
            (loaded.max[1] as f32 - camera.y, loaded.max[1] < target.max[1], 1, 1),
        ];
        let Some(&(_, _, axis, direction)) = sides.iter().filter(|side| side.1).min_by(|a, b| a.0.total_cmp(&b.0))
        else {
            return;
        };
        let mut strip = loaded;
        if direction < 0 {
            strip.max[axis] = loaded.min[axis];
            strip.min[axis] = loaded.min[axis] - 1;
            self.growth = TileRect { min: strip.min, max: loaded.max };
        } else {
            strip.min[axis] = loaded.max[axis];
            strip.max[axis] = loaded.max[axis] + 1;
            self.growth = TileRect { min: loaded.min, max: strip.max };
        }
        self.pending = strip.tiles().collect();
    }

    fn complete(&self) -> bool {
        self.loaded == self.target
    }

    /// Stops loading. What is loaded stays usable until the window follows
    /// the camera again.
    fn rest(&mut self) {
        self.origin = None;
        self.target = self.loaded;
        self.pending.clear();
    }

    /// Takes a finished tile off the list. False if the window has moved on
    /// and no longer waits for it.
    fn accept(&mut self, tile: [i32; 2]) -> bool {
        let Some(index) = self.pending.iter().position(|&pending| pending == tile) else {
            return false;
        };
        self.pending.swap_remove(index);
        if self.pending.is_empty() {
            self.loaded = self.growth;
        }
        true
    }
}

/// One pyramid layer of a tile, as the bytes its textures take.
struct Layer {
    index: usize,
    heights: Vec<u8>,
    gradients: Vec<u8>,
}

/// A tile ready to be copied to the GPU.
pub struct Prepared {
    level: usize,
    tile: [i32; 2],
    layers: Vec<Layer>,
    water: Vec<u8>,
}

impl Prepared {
    /// Turns a tile into what its textures take. `cell` is the cell size of
    /// level 0 in metres.
    pub fn new(level: usize, tile: [i32; 2], data: &Tile, cell: f32) -> Self {
        let cell = cell * (1u64 << level) as f32;
        let pyramids = [(pyramid_layer(level, false), &data.plain), (pyramid_layer(level, true), &data.accentuated)];
        // At level 0 both pyramids share a layer.
        let layers = pyramids
            .into_iter()
            .take(if level == 0 { 1 } else { 2 })
            .map(|(index, heights)| Layer {
                index,
                heights: bytemuck::cast_slice(&interior(heights)).to_vec(),
                gradients: gradient_bytes(heights, cell),
            })
            .collect();
        Self { level, tile, layers, water: data.water.clone() }
    }
}

/// Where a clipmap's tiles come from.
pub trait TileFeed {
    /// Starts preparing a tile. It turns up in [`ready`](Self::ready) later.
    fn request(&mut self, level: usize, tile: [i32; 2]);
    /// A tile that has been prepared, if one is waiting.
    fn ready(&mut self) -> Option<Result<Prepared>>;
    /// Waits for the next tile. Only offline rendering does, and a feed that
    /// cannot wait may refuse.
    fn wait(&mut self) -> Result<Prepared>;
}

/// Makes up the tile of the finest tiled level at `tile` by interpolating
/// the first whole level of both pyramids. Where there is no data that level
/// holds a smooth continuation of the terrain.
pub fn derive_from_whole(whole: &[Grid<f32>; 2], tile: [i32; 2]) -> Tile {
    let side = whole[0].width() as f32;
    let upsample = |grid: &Grid<f32>| -> Vec<f32> {
        (0..TILE_SPAN * TILE_SPAN)
            .map(|i| {
                let (x, y) = (coarser(i % TILE_SPAN, tile[0]), coarser(i / TILE_SPAN, tile[1]));
                grid.sample((x + 0.5) / side, (y + 0.5) / side)
            })
            .collect()
    };
    Tile { plain: upsample(&whole[0]), accentuated: upsample(&whole[1]), water: vec![0; TILE_SIZE * TILE_SIZE] }
}

/// Makes up a tile by interpolating `parent`, the tile of the next-coarser
/// level that covers it.
pub fn derive_from_parent(parent: &Tile, tile: [i32; 2]) -> Tile {
    let parent_tile = [tile[0] >> 1, tile[1] >> 1];
    // The parent's heights start one texel before its first texel.
    let (west, south) =
        ((parent_tile[0] * TILE_SIZE as i32 - 1) as f32, (parent_tile[1] * TILE_SIZE as i32 - 1) as f32);
    let upsample = |heights: &[f32]| -> Vec<f32> {
        (0..TILE_SPAN * TILE_SPAN)
            .map(|i| {
                let x = (coarser(i % TILE_SPAN, tile[0]) - west).clamp(0.0, TILE_SPAN as f32 - 1.001);
                let y = (coarser(i / TILE_SPAN, tile[1]) - south).clamp(0.0, TILE_SPAN as f32 - 1.001);
                let (column, row) = (x as usize, y as usize);
                let (fx, fy) = (x - column as f32, y - row as f32);
                let at = |column: usize, row: usize| heights[row * TILE_SPAN + column];
                let lower = at(column, row) * (1.0 - fx) + at(column + 1, row) * fx;
                let upper = at(column, row + 1) * (1.0 - fx) + at(column + 1, row + 1) * fx;
                lower * (1.0 - fy) + upper * fy
            })
            .collect()
    };
    Tile {
        plain: upsample(&parent.plain),
        accentuated: upsample(&parent.accentuated),
        water: vec![0; TILE_SIZE * TILE_SIZE],
    }
}

/// The centre of a tile's texel, apron included, in texels of the
/// next-coarser level.
fn coarser(index: usize, tile: i32) -> f32 {
    0.5 * ((tile * TILE_SIZE as i32) as f32 + index as f32 - 0.5) - 0.5
}

/// Reads tiles and prepares them for upload. Shared by the worker threads.
#[cfg(not(target_arch = "wasm32"))]
struct TileLoader {
    tiles: Arc<dyn TileSource>,
    /// The first whole level of both pyramids, from which tiles without data
    /// are derived.
    whole: [Grid<f32>; 2],
    /// Cell size of level 0 in metres.
    cell: f32,
    parents: Mutex<HashMap<TileKey, Arc<Tile>>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl TileLoader {
    fn new(terrain: &Terrain) -> Self {
        Self {
            tiles: terrain.tiles.clone(),
            whole: [terrain.coarse[0][0].clone(), terrain.coarse[1][0].clone()],
            cell: terrain.meta.cell_size,
            parents: Mutex::new(HashMap::new()),
        }
    }

    fn prepare(&self, level: usize, tile: [i32; 2]) -> Result<Prepared> {
        Ok(Prepared::new(level, tile, &*self.resolve(level, tile)?, self.cell))
    }

    /// The tile from the source, or derived from the level above where the
    /// terrain has no data.
    fn resolve(&self, level: usize, tile: [i32; 2]) -> Result<Arc<Tile>> {
        if let Some(found) = self.tiles.tile(level, tile[0] as usize, tile[1] as usize)? {
            return Ok(Arc::new(found));
        }
        let cached = |parents: &HashMap<TileKey, Arc<Tile>>| parents.get(&(level, tile)).cloned();
        if let Some(cached) = cached(&self.parents.lock().expect("no loader thread panics while holding the cache")) {
            return Ok(cached);
        }
        // Two threads may derive the same tile at once; both get the same
        // result, and the cache keeps one of them.
        let derived = Arc::new(self.derive(level, tile)?);
        let mut parents = self.parents.lock().expect("no loader thread panics while holding the cache");
        if parents.len() >= PARENT_CACHE_TILES {
            parents.clear();
        }
        parents.insert((level, tile), derived.clone());
        Ok(derived)
    }

    /// Makes up a tile from the next-coarser level.
    fn derive(&self, level: usize, tile: [i32; 2]) -> Result<Tile> {
        if level + 1 == TILED_LEVELS {
            return Ok(derive_from_whole(&self.whole, tile));
        }
        let parent = self.resolve(level + 1, [tile[0] >> 1, tile[1] >> 1])?;
        Ok(derive_from_parent(&parent, tile))
    }
}

/// The feed of a terrain on disk or in memory: worker threads read its
/// tiles and prepare them.
#[cfg(not(target_arch = "wasm32"))]
pub struct ThreadedFeed {
    loader: Arc<TileLoader>,
    /// Where the worker threads hand in what they have prepared.
    finished: Sender<Result<Prepared>>,
    prepared: Receiver<Result<Prepared>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ThreadedFeed {
    pub fn new(terrain: &Terrain) -> Self {
        let (finished, prepared) = mpsc::channel();
        Self { loader: Arc::new(TileLoader::new(terrain)), finished, prepared }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl TileFeed for ThreadedFeed {
    fn request(&mut self, level: usize, tile: [i32; 2]) {
        let (loader, finished) = (self.loader.clone(), self.finished.clone());
        rayon::spawn(move || {
            // If the feed is gone by now, nobody wants the tile.
            let _ = finished.send(loader.prepare(level, tile));
        });
    }

    fn ready(&mut self) -> Option<Result<Prepared>> {
        self.prepared.try_recv().ok()
    }

    fn wait(&mut self) -> Result<Prepared> {
        use anyhow::Context;
        self.prepared.recv().context("the tile loader stopped")?
    }
}

pub struct Clipmap {
    feed: Box<dyn TileFeed>,
    /// Cell size of level 0 in metres.
    cell: f32,
    /// Side length of the terrain square in level-0 cells.
    size: usize,
    /// The finest level the terrain has tiles for.
    finest: usize,
    /// Tiles along one side of a window.
    reach: i32,
    /// Set when tiles were uploaded or given up since it was last taken.
    changed: bool,
    heights: wgpu::Texture,
    gradients: wgpu::Texture,
    water: wgpu::Texture,
    pub height_view: wgpu::TextureView,
    pub gradient_view: wgpu::TextureView,
    pub water_view: wgpu::TextureView,
    windows: [Window; TILED_LEVELS],
}

impl Clipmap {
    /// `reach` is how many tiles along a side each window holds, an even
    /// number up to [`FULL_REACH`].
    pub fn new(device: &wgpu::Device, terrain: &Terrain, feed: Box<dyn TileFeed>, reach: usize) -> Self {
        assert!(reach.is_multiple_of(2) && (2..=FULL_REACH).contains(&reach), "a window cannot reach {reach} tiles");
        let side = window_texels(reach);
        let array = |label: &str, format, layers: u32| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: side, height: side, depth_or_array_layers: layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let heights = array("streamed heights", wgpu::TextureFormat::R32Float, PYRAMID_LAYERS);
        let gradients = array("streamed gradients", wgpu::TextureFormat::Rg16Float, PYRAMID_LAYERS);
        let water = array("streamed lake mask", wgpu::TextureFormat::R8Unorm, TILED_LEVELS as u32);
        Self {
            feed,
            cell: terrain.meta.cell_size,
            size: terrain.meta.size,
            finest: terrain.finest_level(),
            reach: reach as i32,
            changed: false,
            height_view: view(&heights),
            gradient_view: view(&gradients),
            water_view: view(&water),
            heights,
            gradients,
            water,
            windows: std::array::from_fn(|_| Window::new(reach as i32)),
        }
    }

    /// Side length of a tile of `level` in metres.
    fn tile_metres(&self, level: usize) -> f32 {
        self.cell * (TILE_SIZE << level) as f32
    }

    /// The tiles of `level` that lie in the terrain square.
    fn terrain_tiles(&self, level: usize) -> TileRect {
        let side = (self.size >> level).div_ceil(TILE_SIZE) as i32;
        TileRect { min: [0, 0], max: [side, side] }
    }

    /// Moves the windows after the camera, asks the feed for the tiles they
    /// are missing and uploads those it has prepared: at most `budget` of
    /// them. With `None` it waits until every window is complete.
    ///
    /// `ahead` is the horizontal part of the camera's unit view direction,
    /// and `finest` the finest level worth loading for the view.
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        camera: Vec2,
        ahead: Vec2,
        finest: usize,
        budget: Option<usize>,
    ) -> Result<()> {
        let mut uploads = 0;
        loop {
            // Coarse levels are asked for first, since everything finer falls
            // back on them.
            for level in (self.finest..TILED_LEVELS).rev() {
                if level < finest {
                    self.windows[level].rest();
                    continue;
                }
                let metres = self.tile_metres(level);
                let window = &mut self.windows[level];
                let focus = camera / metres + ahead * (LOOK_AHEAD * window.reach as f32);
                let terrain_tiles =
                    TileRect { min: [0, 0], max: [1, 1].map(|_| (self.size >> level).div_ceil(TILE_SIZE) as i32) };
                let held = window.loaded;
                window.follow(focus, terrain_tiles);
                self.changed |= window.loaded != held;
                if !window.complete() && window.pending.is_empty() {
                    window.plan(focus);
                    for tile in window.pending.clone() {
                        self.feed.request(level, tile);
                    }
                }
            }

            let prepared = match budget {
                Some(limit) if uploads >= limit => break,
                Some(_) => match self.feed.ready() {
                    Some(prepared) => prepared?,
                    None => break,
                },
                None if self.windows.iter().all(Window::complete) => break,
                None => self.feed.wait()?,
            };
            // A window that has moved on has no slot left for this tile.
            if self.windows[prepared.level].accept(prepared.tile) {
                self.upload(queue, &prepared);
                uploads += 1;
                self.changed = true;
            }
        }
        Ok(())
    }

    /// Whether what the shader may sample has changed since this was last
    /// asked. A frame that would otherwise repeat the last one has to be
    /// drawn again if so.
    pub fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    /// For every streamed level, the tiles its window holds or is loading.
    pub fn targets(&self) -> [TileRect; TILED_LEVELS] {
        std::array::from_fn(|level| self.windows[level].target)
    }

    /// Tiles still to be uploaded before every window is complete.
    pub fn missing_tiles(&self) -> usize {
        let area = |rect: &TileRect| ((rect.max[0] - rect.min[0]) * (rect.max[1] - rect.min[1])).max(0) as usize;
        self.windows.iter().map(|window| area(&window.target) - area(&window.loaded)).sum()
    }

    /// For every streamed level, the rectangle that may be sampled, as west,
    /// south, east and north in metres. Empty rectangles are inside out.
    ///
    /// A level fades into the next-coarser one toward the sides of its
    /// rectangle. Where a side is the edge of the terrain there is nothing
    /// to fade into, so that side is moved out of reach.
    pub fn loaded_regions(&self) -> [[f32; 4]; TILED_LEVELS] {
        std::array::from_fn(|level| {
            let loaded = self.windows[level].loaded;
            if loaded.is_empty() {
                return [1.0, 1.0, 0.0, 0.0];
            }
            let (metres, terrain) = (self.tile_metres(level), self.terrain_tiles(level));
            let side = |tile: i32, edge: i32, outward: f32| {
                if tile == edge {
                    outward * OUT_OF_REACH_M
                } else {
                    tile as f32 * metres
                }
            };
            [
                side(loaded.min[0], terrain.min[0], -1.0),
                side(loaded.min[1], terrain.min[1], -1.0),
                side(loaded.max[0], terrain.max[0], 1.0),
                side(loaded.max[1], terrain.max[1], 1.0),
            ]
        })
    }

    fn upload(&self, queue: &wgpu::Queue, prepared: &Prepared) {
        // Windows wrap around their textures, so a tile's slot depends only
        // on its own index.
        let slot = prepared.tile.map(|index| index.rem_euclid(self.reach) as u32 * TILE_SIZE as u32);
        let write = |texture: &wgpu::Texture, layer: usize, bytes: &[u8]| {
            let texel_bytes = bytes.len() / (TILE_SIZE * TILE_SIZE);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: slot[0], y: slot[1], z: layer as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some((TILE_SIZE * texel_bytes) as u32),
                    rows_per_image: Some(TILE_SIZE as u32),
                },
                wgpu::Extent3d { width: TILE_SIZE as u32, height: TILE_SIZE as u32, depth_or_array_layers: 1 },
            );
        };
        for layer in &prepared.layers {
            write(&self.heights, layer.index, &layer.heights);
            write(&self.gradients, layer.index, &layer.gradients);
        }
        write(&self.water, prepared.level, &prepared.water);
    }
}

/// The array layer that holds a level of the plain or accentuated pyramid.
/// Mirrored by `streamed_layer` in terrain.wgsl.
fn pyramid_layer(level: usize, accentuated: bool) -> usize {
    if accentuated && level > 0 {
        level + TILED_LEVELS - 1
    } else {
        level
    }
}

/// A tile's heights without their apron.
fn interior(heights: &[f32]) -> Vec<f32> {
    let (rows, _) = heights.as_chunks::<TILE_SPAN>();
    rows.iter().skip(1).take(TILE_SIZE).flat_map(|row| &row[1..=TILE_SIZE]).copied().collect()
}

/// Slopes `(dh/dx, dh/dy)` of a tile by central differences across its
/// apron, as half-float pairs.
fn gradient_bytes(heights: &[f32], cell: f32) -> Vec<u8> {
    let scale = 0.5 / cell;
    let at = |column: usize, row: usize| heights[row * TILE_SPAN + column];
    let mut bytes = Vec::with_capacity(TILE_SIZE * TILE_SIZE * 4);
    for row in 1..=TILE_SIZE {
        for column in 1..=TILE_SIZE {
            let slopes = [at(column + 1, row) - at(column - 1, row), at(column, row + 1) - at(column, row - 1)];
            for slope in slopes {
                bytes.extend_from_slice(&f16::from_f32(slope * scale).to_bits().to_le_bytes());
            }
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    const TERRAIN: TileRect = TileRect { min: [0, 0], max: [64, 64] };

    /// Loads everything a window has planned, as `Clipmap::update` would.
    fn load_all(window: &mut Window, camera: Vec2) -> usize {
        let mut uploads = 0;
        while !window.complete() {
            if window.pending.is_empty() {
                window.plan(camera);
            }
            for tile in window.pending.clone() {
                assert!(window.accept(tile));
                uploads += 1;
            }
        }
        uploads
    }

    #[test]
    fn a_window_fills_around_the_camera() {
        let mut window = Window::new(FULL_REACH as i32);
        let camera = Vec2::new(20.3, 30.8);
        window.follow(camera, TERRAIN);
        assert_eq!(window.target, TileRect { min: [16, 27], max: [24, 35] });
        assert!(window.loaded.is_empty());

        // The first tile is the one below the camera.
        window.plan(camera);
        assert_eq!(window.pending, vec![[20, 30]]);
        assert_eq!(load_all(&mut window, camera), 64);
        assert_eq!(window.loaded, window.target);
    }

    #[test]
    fn moving_keeps_what_overlaps_and_loads_one_strip() {
        let mut window = Window::new(FULL_REACH as i32);
        let mut camera = Vec2::new(20.0, 30.0);
        window.follow(camera, TERRAIN);
        load_all(&mut window, camera);

        // A small move stays within the window's tolerance.
        camera.x += 0.6;
        window.follow(camera, TERRAIN);
        assert!(window.complete());

        // A move past it shifts the window by one tile.
        camera.x += 0.2;
        window.follow(camera, TERRAIN);
        assert_eq!(window.target, TileRect { min: [17, 26], max: [25, 34] });
        assert_eq!(window.loaded, TileRect { min: [17, 26], max: [24, 34] });
        assert_eq!(load_all(&mut window, camera), 8);
    }

    #[test]
    fn windows_stop_at_the_edge_of_the_terrain() {
        let mut window = Window::new(FULL_REACH as i32);
        let camera = Vec2::new(-10.0, 1.0);
        window.follow(camera, TERRAIN);
        assert!(window.target.is_empty() && window.complete());

        let camera = Vec2::new(1.0, 1.0);
        window.follow(camera, TERRAIN);
        assert_eq!(window.target, TileRect { min: [0, 0], max: [5, 5] });
        assert_eq!(load_all(&mut window, camera), 25);
    }

    #[test]
    fn a_strip_becomes_usable_only_when_all_of_it_has_arrived() {
        let mut window = Window::new(FULL_REACH as i32);
        let mut camera = Vec2::new(20.0, 30.0);
        window.follow(camera, TERRAIN);
        load_all(&mut window, camera);
        camera.x += 1.0;
        window.follow(camera, TERRAIN);
        window.plan(camera);
        let (before, strip) = (window.loaded, window.pending.clone());

        // Tiles arrive in any order, and one the window never asked for is
        // turned away.
        assert!(!window.accept([3, 3]));
        for &tile in strip.iter().rev().skip(1) {
            assert!(window.accept(tile));
            assert_eq!(window.loaded, before);
        }
        assert!(window.accept(strip[strip.len() - 1]));
        assert!(window.complete());

        // What arrives after the window has moved on is turned away too.
        window.plan(camera);
        assert!(!window.accept(strip[0]));
    }

    #[test]
    fn the_loader_prepares_every_layer_of_a_tile() {
        let terrain = Terrain::from(relief_terrain::synthetic::cone_dataset(1024).unwrap());
        let loader = TileLoader::new(&terrain);
        let texels = TILE_SIZE * TILE_SIZE;

        let fine = loader.prepare(0, [1, 1]).unwrap();
        assert_eq!(fine.layers.iter().map(|layer| layer.index).collect::<Vec<_>>(), [0]);
        let coarse = loader.prepare(2, [0, 0]).unwrap();
        assert_eq!(coarse.layers.iter().map(|layer| layer.index).collect::<Vec<_>>(), [2, 6]);
        for layer in fine.layers.iter().chain(&coarse.layers) {
            assert_eq!((layer.heights.len(), layer.gradients.len()), (4 * texels, 4 * texels));
        }
        assert_eq!(coarse.water.len(), texels);

        // Beyond the terrain's own tiles, a tile is derived from the level
        // above: the plain around the cone continues at 500 m.
        let derived = loader.prepare(0, [5, 5]).unwrap();
        let heights: Vec<f32> = bytemuck::pod_collect_to_vec(&derived.layers[0].heights);
        assert!(heights.iter().all(|height| (height - 500.0).abs() < 1.0));
    }

    #[test]
    fn pyramids_share_level_zero_only() {
        let layers: Vec<usize> =
            (0..TILED_LEVELS).flat_map(|level| [pyramid_layer(level, false), pyramid_layer(level, true)]).collect();
        assert_eq!(layers, [0, 0, 1, 5, 2, 6, 3, 7, 4, 8]);
        assert_eq!(*layers.iter().max().unwrap() as u32 + 1, PYRAMID_LAYERS);
    }

    #[test]
    fn gradients_and_interiors_skip_the_apron() {
        // Heights rise 2 m per texel east and 6 m per texel north.
        let heights: Vec<f32> =
            (0..TILE_SPAN * TILE_SPAN).map(|i| 2.0 * (i % TILE_SPAN) as f32 + 6.0 * (i / TILE_SPAN) as f32).collect();
        let inner = interior(&heights);
        assert_eq!(inner.len(), TILE_SIZE * TILE_SIZE);
        assert_eq!(inner[0], 2.0 + 6.0);

        let bytes = gradient_bytes(&heights, 20.0);
        let slope =
            |index: usize| f16::from_bits(u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])).to_f32();
        assert_eq!(bytes.len(), TILE_SIZE * TILE_SIZE * 4);
        assert!((slope(0) - 0.1).abs() < 1e-3 && (slope(1) - 0.3).abs() < 1e-3);
    }
}
