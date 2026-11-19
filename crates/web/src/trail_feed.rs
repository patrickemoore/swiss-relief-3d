//! Trail tiles over HTTP, the way terrain tiles come: a request starts a
//! fetch, and the decoded polylines turn up in a queue when it resolves.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use anyhow::Result;
use relief_render::renderer::trails::{TrailFeed, TrailKey};
use relief_terrain::trails::{decode_tile, Polyline};
use wasm_bindgen_futures::spawn_local;

use crate::feed::fetch;

struct Shared {
    /// Where the trails are served from, without a trailing slash.
    url: String,
    /// Cell size of level 0 in metres.
    cell: f32,
    arrived: RefCell<VecDeque<(TrailKey, Result<Vec<Polyline>>)>>,
}

/// The feed of trails served next to a packed terrain, under `trails/`.
pub struct FetchTrailFeed(Rc<Shared>);

impl FetchTrailFeed {
    pub fn new(data_url: &str, cell: f32) -> Self {
        Self(Rc::new(Shared {
            url: format!("{}/trails", data_url.trim_end_matches('/')),
            cell,
            arrived: RefCell::new(VecDeque::new()),
        }))
    }
}

impl TrailFeed for FetchTrailFeed {
    fn request(&mut self, level: usize, tile: [i32; 2]) {
        let shared = self.0.clone();
        spawn_local(async move {
            let url = format!("{}/{level}/{}_{}.bin", shared.url, tile[0], tile[1]);
            let lines = match fetch(&url).await {
                Ok(bytes) => decode_tile(&bytes, level, [tile[0] as usize, tile[1] as usize], shared.cell),
                Err(error) => Err(error),
            };
            shared.arrived.borrow_mut().push_back(((level, tile), lines));
        });
    }

    fn ready(&mut self) -> Option<(TrailKey, Result<Vec<Polyline>>)> {
        self.0.arrived.borrow_mut().pop_front()
    }
}
