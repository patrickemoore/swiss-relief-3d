//! Names on the map: peaks, lakes, glaciers and towns, placed each frame the
//! way a cartographer would place them. A name is shown when the view is
//! close enough for something of its importance, when the terrain does not
//! hide the place, and when it does not collide with a more important name.
//! Names fade in and out rather than pop, and one that is shown holds its
//! place against a newcomer of similar standing, so nothing flickers.
//!
//! The gazetteer is built by `web/scripts/labels.mjs` from Wikidata and
//! embedded. Positions are LV95; the terrain's origin turns them into local
//! metres. The drawing is left to the host: the browser puts text over the
//! canvas, the desktop viewer paints it with egui. Both get the same boxes,
//! so that the layout is decided once, here.

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::camera::{Ground, Pose};

const GAZETTEER: &str = include_str!("../assets/labels.json");

/// Seconds a name takes to fade in or out.
const FADE_S: f32 = 0.3;
/// A shown name keeps showing until the view has backed off this much
/// beyond the scale it appeared at, and counts for this much more against
/// newcomers competing for its place.
const HYSTERESIS: f32 = 1.3;
/// Room left around every name, in CSS pixels, when testing for collisions.
const PADDING_PX: f32 = 3.0;
/// Average width of a character as a fraction of the font size, for a
/// humanist sans-serif, and the height of a line.
const CHAR_EM: f32 = 0.56;
const LINE_EM: f32 = 1.25;
/// A name may be this close to the edge of the view before it goes.
const EDGE_PX: f32 = 8.0;
/// Metres between the samples that test whether terrain hides a place, and
/// the most samples one place gets.
const OCCLUSION_STEP_M: f32 = 120.0;
const OCCLUSION_SAMPLES: usize = 48;
/// A place may be this many metres below the line of sight, plus this
/// fraction of its distance, before it counts as hidden: the heights on the
/// CPU are coarser than the ones drawn.
const OCCLUSION_SLACK_M: f32 = 12.0;
const OCCLUSION_SLACK_PER_M: f32 = 0.003;

/// What a name names. The order is the order names of equal standing are
/// placed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Peak,
    Lake,
    Town,
    Glacier,
}

impl Kind {
    /// Ground metres per CSS pixel at which the best-ranked name of the kind
    /// appears and at which the worst does; between them the scale falls
    /// geometrically with rank, so that names come in steadily as the view
    /// closes in.
    fn scale_range(self) -> (f32, f32) {
        match self {
            Kind::Peak => (700.0, 4.0),
            Kind::Lake => (900.0, 12.0),
            Kind::Town => (550.0, 8.0),
            Kind::Glacier => (160.0, 5.0),
        }
    }

    /// How a kind fares against the others when two names want one place:
    /// on a relief map the peaks and the waters come first.
    fn weight(self) -> f32 {
        match self {
            Kind::Peak => 1.0,
            Kind::Lake => 0.95,
            Kind::Glacier => 0.9,
            Kind::Town => 0.7,
        }
    }

    /// Font size in CSS pixels, from the moment the name appears to when the
    /// view is well past that.
    fn font_px(self) -> (f32, f32) {
        match self {
            Kind::Peak => (12.0, 14.5),
            Kind::Lake => (12.5, 16.0),
            Kind::Town => (12.0, 15.0),
            Kind::Glacier => (11.5, 13.5),
        }
    }
}

/// A place that can be named, in local metres.
#[derive(Clone, Debug)]
pub struct Entry {
    pub kind: Kind,
    pub name: String,
    /// Summit height in metres, for peaks.
    pub elevation: Option<i32>,
    pub position: Vec2,
    /// Ground metres per CSS pixel at the place below which it may be named.
    threshold: f32,
    /// Where it stands among its kind, 0 for the best and 1 for the last.
    rank: f32,
}

#[derive(Deserialize)]
struct Record {
    kind: Kind,
    name: String,
    east: f64,
    north: f64,
    #[serde(default)]
    elevation: Option<i32>,
}

/// Where the camera is and what it sees, for one frame.
pub struct View<'a> {
    pub pose: Pose,
    pub fov_y: f32,
    /// The size of the view in device pixels.
    pub viewport: Vec2,
    /// Device pixels per CSS pixel.
    pub pixel_ratio: f32,
    /// Vertical exaggeration of the drawn terrain.
    pub z_scale: f32,
    pub ground: &'a Ground,
    /// How much of the view the host's controls cover at the top, right,
    /// bottom and left, in device pixels. No name goes under them.
    pub insets: [f32; 4],
}

/// A name placed in the view, in device pixels from the top left.
#[derive(Clone, Debug, Serialize)]
pub struct Placed {
    /// Index of the entry in [`Labels::entries`].
    pub id: usize,
    pub kind: Kind,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elevation: Option<i32>,
    /// The place itself on the screen: the summit, the middle of the lake,
    /// the town.
    pub x: f32,
    pub y: f32,
    /// The box the text goes in: left, top, width and height.
    pub rect: [f32; 4],
    /// Font size in device pixels.
    pub font: f32,
    pub opacity: f32,
}

#[derive(Clone, Copy, Default)]
struct State {
    opacity: f32,
    shown: bool,
}

/// A candidate for this frame, before collisions are settled.
struct Candidate {
    id: usize,
    priority: f32,
    screen: Vec2,
    font: f32,
    rect: [f32; 4],
}

pub struct Labels {
    pub entries: Vec<Entry>,
    states: Vec<State>,
    placed: Vec<Placed>,
    animating: bool,
}

impl Labels {
    /// The built-in gazetteer for a terrain whose south-west corner is at
    /// `origin_lv95` and whose data covers `region` (west, south, east and
    /// north in local metres).
    pub fn builtin(origin_lv95: [f64; 2], region: [f32; 4]) -> Self {
        let records: Vec<Record> = serde_json::from_str(GAZETTEER).expect("the embedded gazetteer is valid");
        Self::new(records, origin_lv95, region)
    }

    fn new(records: Vec<Record>, origin_lv95: [f64; 2], region: [f32; 4]) -> Self {
        let [west, south, east, north] = region;
        let mut entries: Vec<Entry> = records
            .into_iter()
            .filter_map(|record| {
                let position = Vec2::new((record.east - origin_lv95[0]) as f32, (record.north - origin_lv95[1]) as f32);
                let inside = (west..=east).contains(&position.x) && (south..=north).contains(&position.y);
                inside.then_some(Entry {
                    kind: record.kind,
                    name: record.name,
                    elevation: record.elevation,
                    position,
                    threshold: 0.0,
                    rank: 0.0,
                })
            })
            .collect();
        // The gazetteer lists each kind best first; the rank sets the scale.
        for kind in [Kind::Peak, Kind::Lake, Kind::Town, Kind::Glacier] {
            let count = entries.iter().filter(|entry| entry.kind == kind).count().max(2);
            let (far, near) = kind.scale_range();
            for (rank, entry) in entries.iter_mut().filter(|entry| entry.kind == kind).enumerate() {
                let fraction = rank as f32 / (count - 1) as f32;
                entry.threshold = far * (near / far).powf(fraction);
                entry.rank = fraction;
            }
        }
        let states = vec![State::default(); entries.len()];
        Self { entries, states, placed: Vec::new(), animating: false }
    }

    /// The names placed by the last [`Labels::update`].
    pub fn placed(&self) -> &[Placed] {
        &self.placed
    }

    /// Whether names are still fading, so that another frame is needed.
    pub fn animating(&self) -> bool {
        self.animating
    }

    /// Lays the names out for a frame `dt` seconds after the last.
    pub fn update(&mut self, view: &View, dt: f32) {
        let aspect = view.viewport.x / view.viewport.y;
        let view_projection = view.pose.view_projection(view.fov_y, aspect);
        let css_height = view.viewport.y / view.pixel_ratio;
        let metres_per_css_pixel = 2.0 * (0.5 * view.fov_y).tan() / css_height;

        let mut candidates = Vec::new();
        for (id, entry) in self.entries.iter().enumerate() {
            let shown = self.states[id].shown;
            let ground = view.ground.height(entry.position);
            let height = match entry.elevation {
                // Summits are sharper than the heights on the CPU.
                Some(elevation) => ground.max(elevation as f32),
                None => ground,
            };
            let world = entry.position.extend(height * view.z_scale);
            let distance = world.distance(view.pose.position);
            let scale = distance * metres_per_css_pixel;
            let allowance = if shown { HYSTERESIS } else { 1.0 };
            if scale > entry.threshold * allowance {
                continue;
            }
            let clip = view_projection * world.extend(1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let ndc = clip.truncate() / clip.w;
            let screen = Vec2::new(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * view.viewport;
            if hidden(view, world, distance) {
                continue;
            }

            // How far past its own scale the view is: a name that has been
            // established for a while is bigger and harder to displace.
            let strength = (entry.threshold / scale.max(1e-3)).ln() / 8f32.ln();
            let (small, large) = entry.kind.font_px();
            let font = (small + (large - small) * strength.clamp(0.0, 1.0)) * view.pixel_ratio;
            let rect = text_box(entry, screen, font, view.pixel_ratio);
            if !within(rect, view) {
                continue;
            }
            // Standing decides who gets a contested place: the best of a kind
            // first, the kinds weighed against each other, and a name that is
            // already there ahead of one that is not.
            let priority = (1.0 - entry.rank) * entry.kind.weight() * if shown { HYSTERESIS } else { 1.0 };
            candidates.push(Candidate { id, priority, screen, font, rect });
        }

        candidates.sort_by(|a, b| {
            b.priority
                .total_cmp(&a.priority)
                .then_with(|| (self.entries[a.id].kind as u8).cmp(&(self.entries[b.id].kind as u8)))
        });
        let padding = PADDING_PX * view.pixel_ratio;
        let mut taken: Vec<[f32; 4]> = Vec::new();
        let mut shown_now = vec![false; self.entries.len()];
        let mut fonts = vec![0.0; self.entries.len()];
        let mut placements: Vec<(usize, Vec2, [f32; 4])> = Vec::new();
        for candidate in candidates {
            let padded = grow(candidate.rect, padding);
            if taken.iter().any(|other| overlaps(*other, padded)) {
                continue;
            }
            taken.push(padded);
            shown_now[candidate.id] = true;
            fonts[candidate.id] = candidate.font;
            placements.push((candidate.id, candidate.screen, candidate.rect));
        }

        // Fades, then the output: what is placed, and what is still going.
        let step = dt / FADE_S;
        self.animating = false;
        self.placed.clear();
        for (id, state) in self.states.iter_mut().enumerate() {
            state.shown = shown_now[id];
            let target = if state.shown { 1.0 } else { 0.0 };
            let before = state.opacity;
            state.opacity = if target > before { (before + step).min(1.0) } else { (before - step).max(0.0) };
            if state.opacity != target {
                self.animating = true;
            }
        }
        for (id, screen, rect) in placements {
            self.placed.push(self.output(id, screen, rect, fonts[id]));
        }
        // Names on their way out keep their last place.
        let placed_ids: Vec<usize> = self.placed.iter().map(|placed| placed.id).collect();
        let fading: Vec<usize> = (0..self.entries.len())
            .filter(|id| !self.states[*id].shown && self.states[*id].opacity > 0.0 && !placed_ids.contains(id))
            .collect();
        for id in fading {
            let entry = &self.entries[id];
            let height = view.ground.height(entry.position).max(entry.elevation.map_or(f32::MIN, |e| e as f32));
            let world = entry.position.extend(height * view.z_scale);
            let clip = view_projection * world.extend(1.0);
            if clip.w <= 0.0 {
                self.states[id].opacity = 0.0;
                continue;
            }
            let ndc = clip.truncate() / clip.w;
            let screen = Vec2::new(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y)) * view.viewport;
            let (small, _) = entry.kind.font_px();
            let font = small * view.pixel_ratio;
            let rect = text_box(entry, screen, font, view.pixel_ratio);
            self.placed.push(self.output(id, screen, rect, font));
        }
    }

    fn output(&self, id: usize, screen: Vec2, rect: [f32; 4], font: f32) -> Placed {
        let entry = &self.entries[id];
        Placed {
            id,
            kind: entry.kind,
            name: entry.name.clone(),
            elevation: entry.elevation,
            x: screen.x,
            y: screen.y,
            rect,
            font,
            opacity: self.states[id].opacity,
        }
    }
}

/// Whether a box on the screen lies in the part of the view that is free:
/// not over its edge and not under the host's controls.
fn within(rect: [f32; 4], view: &View) -> bool {
    let edge = EDGE_PX * view.pixel_ratio;
    let [top, right, bottom, left] = view.insets;
    rect[0] >= left + edge
        && rect[1] >= top + edge
        && rect[0] + rect[2] <= view.viewport.x - right - edge
        && rect[1] + rect[3] <= view.viewport.y - bottom - edge
}

/// Whether terrain between the camera and `world` hides it.
fn hidden(view: &View, world: Vec3, distance: f32) -> bool {
    let samples = ((distance / OCCLUSION_STEP_M) as usize).clamp(6, OCCLUSION_SAMPLES);
    let slack = (OCCLUSION_SLACK_M + OCCLUSION_SLACK_PER_M * distance) * view.z_scale;
    (1..samples).any(|i| {
        // Not right at the place, where the terrain is the place.
        let t = i as f32 / samples as f32 * 0.96;
        let point = view.pose.position.lerp(world, t);
        view.ground.height(point.truncate()) * view.z_scale > point.z + slack
    })
}

/// The box a name's text takes, placed against its point the way its kind
/// is set: a peak's name above the summit with its height below the name,
/// a town's name to the right of its dot, a lake's and a glacier's across
/// the water.
fn text_box(entry: &Entry, screen: Vec2, font: f32, pixel_ratio: f32) -> [f32; 4] {
    let width = entry.name.chars().count() as f32 * CHAR_EM * font;
    let line = LINE_EM * font;
    match entry.kind {
        Kind::Peak => {
            let height_line = if entry.elevation.is_some() { 0.85 * line } else { 0.0 };
            let gap = 7.0 * pixel_ratio;
            [screen.x - 0.5 * width, screen.y - gap - line - height_line, width, line + height_line]
        }
        Kind::Town => {
            let gap = 8.0 * pixel_ratio;
            [screen.x + gap, screen.y - 0.5 * line, width, line]
        }
        Kind::Lake | Kind::Glacier => [screen.x - 0.5 * width, screen.y - 0.5 * line, width, line],
    }
}

fn grow(rect: [f32; 4], by: f32) -> [f32; 4] {
    [rect[0] - by, rect[1] - by, rect[2] + 2.0 * by, rect[3] + 2.0 * by]
}

fn overlaps(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gazetteer_parses_and_ranks() {
        // Switzerland's square as the country dataset has it.
        let labels = Labels::builtin([2_480_000.0, 1_070_000.0], [0.0, 0.0, 360_000.0, 180_000.0]);
        assert!(labels.entries.len() > 500);
        let matterhorn = labels.entries.iter().find(|entry| entry.name == "Matterhorn").expect("the Matterhorn");
        assert_eq!(matterhorn.kind, Kind::Peak);
        assert_eq!(matterhorn.elevation, Some(4478));
        // The best peak shows from the farthest; later peaks need closer views.
        let peaks: Vec<f32> = labels.entries.iter().filter(|e| e.kind == Kind::Peak).map(|e| e.threshold).collect();
        assert!(peaks.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!((peaks[0] - 700.0).abs() < 1e-3);
    }

    #[test]
    fn boxes_overlap_only_when_they_do() {
        assert!(overlaps([0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 10.0, 10.0]));
        assert!(!overlaps([0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 10.0, 10.0]));
        assert!(overlaps(grow([0.0, 0.0, 10.0, 10.0], 1.0), [10.5, 0.0, 10.0, 10.0]));
    }

    #[test]
    fn peak_names_sit_above_the_summit() {
        let entry = Entry {
            kind: Kind::Peak,
            name: "Eiger".into(),
            elevation: Some(3967),
            position: Vec2::ZERO,
            threshold: 1.0,
            rank: 0.0,
        };
        let rect = text_box(&entry, Vec2::new(100.0, 100.0), 14.0, 1.0);
        assert!(rect[1] + rect[3] < 100.0, "the text ends above the point");
        assert!((rect[0] + 0.5 * rect[2] - 100.0).abs() < 1e-3, "and is centred on it");
    }
}
