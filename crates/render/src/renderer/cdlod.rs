//! Continuous distance-dependent level of detail (plan section 5).
//!
//! After Strugar (2009): one small grid patch is drawn instanced for every
//! selected quadtree node, each level of the tree covers a distance range
//! twice as far as the previous one, and in the outer part of each range the
//! vertices morph toward the next-coarser grid.
//!
//! The mesh only decides silhouettes and depth. Shading is sampled per pixel
//! from the pyramids, so it is the same at every mesh level.
//!
//! Level selection and morphing both use one distance metric: horizontal
//! distance to the camera combined with the camera's height above a reference
//! plane. Because the metric does not depend on a vertex's own height, the
//! CPU selection and the vertex shader agree exactly, which keeps the
//! boundaries between levels free of cracks.

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4, Vec4Swizzles};
use relief_terrain::terrain::LEAF_CELLS;
use relief_terrain::Grid;

/// Quads along one side of a patch. A patch covers one quadrant of a node,
/// and a level-0 node is a leaf of [`LEAF_CELLS`] height cells, so level `l`
/// has the vertex spacing of pyramid level `l`.
pub const PATCH_QUADS: u32 = (LEAF_CELLS / 2) as u32;
/// Fraction of a level's range at which its vertices start to morph.
pub const MORPH_START: f32 = 0.75;
/// Smallest allowed range of level 0, in units of the level-0 node diagonal.
///
/// A level-`l` node may extend one node diagonal past the previous range,
/// and its far corner must still be unmorphed where it borders level `l + 1`.
const MIN_RANGE_IN_DIAGONALS: f32 = 1.0 / (2.0 * MORPH_START - 1.0);

/// Per-instance data of one patch: where it is and at which level.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct PatchInstance {
    /// South-west corner in local metres.
    pub origin: [f32; 2],
    /// Side length in metres.
    pub size: f32,
    /// Quadtree level, equal to the pyramid level of its vertex spacing.
    pub level: f32,
}

/// Region edges, in the order skirt meshes and instances are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    West = 0,
    East = 1,
    South = 2,
    North = 3,
}

/// The patches to draw for one frame.
#[derive(Default)]
pub struct Selection {
    pub patches: Vec<PatchInstance>,
    /// Patches along each region edge, which also draw a skirt there.
    pub skirts: [Vec<PatchInstance>; 4],
}

impl Selection {
    fn clear(&mut self) {
        self.patches.clear();
        self.skirts.iter_mut().for_each(Vec::clear);
    }
}

/// View frustum as inward-facing planes.
pub struct Frustum {
    planes: [Vec4; 5],
}

impl Frustum {
    /// Extracts the side planes and the near plane from a reverse-Z view
    /// projection (the far plane is at infinity).
    pub fn from_view_projection(matrix: &Mat4) -> Self {
        let (x, y, z, w) = (matrix.row(0), matrix.row(1), matrix.row(2), matrix.row(3));
        Self { planes: [w + x, w - x, w + y, w - y, w - z] }
    }

    /// A frustum that contains everything.
    #[cfg(test)]
    pub fn everything() -> Self {
        Self { planes: [Vec4::W; 5] }
    }

    pub fn intersects_box(&self, min: Vec3, max: Vec3) -> bool {
        self.planes.iter().all(|plane| {
            // The box corner furthest along the plane normal.
            let corner = Vec3::select(plane.xyz().cmpge(Vec3::ZERO), max, min);
            plane.xyz().dot(corner) + plane.w >= 0.0
        })
    }
}

/// Everything level selection depends on.
pub struct LodView {
    pub camera: Vec3,
    /// World height of the plane distances are measured against, normally
    /// the ground below the camera.
    pub reference_z: f32,
    /// Range of level 0 in metres; level `l` reaches `range0 * 2^l`.
    pub range0: f32,
    pub frustum: Frustum,
    /// Vertical scale the terrain is drawn with.
    pub z_scale: f32,
}

impl LodView {
    /// The distance metric shared with the vertex shader, to the nearest
    /// point of a rectangle.
    fn distance_to(&self, min: Vec2, max: Vec2) -> f32 {
        let nearest = self.camera.truncate().clamp(min, max);
        (nearest - self.camera.truncate()).extend(self.camera.z - self.reference_z).length()
    }
}

/// Whether a height range belongs to a node that holds data. Nodes without
/// any, such as those beyond a country's border, carry an inverted range.
fn has_data(z_range: [f32; 2]) -> bool {
    z_range[0] <= z_range[1]
}

/// A quadtree over the terrain with the height range of every node.
pub struct Quadtree {
    /// `ranges[l]` holds `[min, max]` true heights for the nodes of level `l`,
    /// inverted for nodes without data.
    ranges: Vec<Grid<[f32; 2]>>,
    cell: f32,
    extent: f32,
}

impl Quadtree {
    /// Builds the tree over the height ranges of its leaves, each
    /// [`LEAF_CELLS`] cells of `cell` metres wide.
    pub fn new(leaf_ranges: &Grid<[f32; 2]>, cell: f32) -> Self {
        let leaves = leaf_ranges.width();
        assert!(
            leaves.is_power_of_two() && leaf_ranges.height() == leaves,
            "the leaves must form a power-of-two square"
        );

        let mut ranges = vec![leaf_ranges.clone()];
        while ranges.last().is_some_and(|level| level.width() > 1) {
            let children = ranges.last().expect("the quadtree has a leaf level");
            ranges.push(Grid::from_fn(children.width() / 2, children.height() / 2, |x, y| {
                [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .map(|(dx, dy)| children.get(2 * x + dx, 2 * y + dy))
                    .into_iter()
                    .filter(|&child| has_data(child))
                    .fold([f32::MAX, f32::MIN], |[low, high], [child_low, child_high]| {
                        [low.min(child_low), high.max(child_high)]
                    })
            }));
        }
        Self { ranges, cell, extent: (leaves * LEAF_CELLS) as f32 * cell }
    }

    pub fn top_level(&self) -> usize {
        self.ranges.len() - 1
    }

    fn node_size(&self, level: usize) -> f32 {
        LEAF_CELLS as f32 * self.cell * (1u32 << level) as f32
    }

    /// The level-0 range that puts about `pixels_per_quad` screen pixels on
    /// each mesh quad, never below the smallest crack-free range.
    pub fn range0(&self, pixels_per_quad: f32, pixel_angle: f32) -> f32 {
        let wanted = self.cell / (pixels_per_quad * pixel_angle);
        let diagonal = self.node_size(0) * std::f32::consts::SQRT_2;
        wanted.max(1.05 * MIN_RANGE_IN_DIAGONALS * diagonal)
    }

    /// Fills `selection` with the patches visible from `view`.
    pub fn select(&self, view: &LodView, selection: &mut Selection) {
        selection.clear();
        self.select_node(self.top_level(), 0, 0, view, selection);
    }

    /// Returns false if the node lies beyond its level's range, in which
    /// case the parent covers it at the coarser level.
    fn select_node(&self, level: usize, x: usize, y: usize, view: &LodView, selection: &mut Selection) -> bool {
        let size = self.node_size(level);
        let min = Vec2::new(x as f32, y as f32) * size;
        let distance = view.distance_to(min, min + size);
        let range = view.range0 * (1u32 << level) as f32;
        if level < self.top_level() && distance >= range {
            return false;
        }

        let z_range = self.ranges[level].get(x, y);
        if !has_data(z_range) || !self.visible(level, min, size, z_range, view) {
            return true;
        }

        let children = [(0, 0), (1, 0), (0, 1), (1, 1)];
        let whole_node = level == 0 || distance >= 0.5 * range;
        for (dx, dy) in children {
            let covered = !whole_node && self.select_node(level - 1, 2 * x + dx, 2 * y + dy, view, selection);
            if !covered {
                let origin = min + Vec2::new(dx as f32, dy as f32) * 0.5 * size;
                let quadrant_range = match level {
                    0 => z_range,
                    _ => self.ranges[level - 1].get(2 * x + dx, 2 * y + dy),
                };
                self.emit(level, origin, 0.5 * size, quadrant_range, view, selection);
            }
        }
        true
    }

    fn visible(&self, level: usize, min: Vec2, size: f32, z_range: [f32; 2], view: &LodView) -> bool {
        // Coarser pyramid levels average a little beyond the node, so pad
        // the height range by what a steep slope gains over that margin.
        let pad = 3.0 * self.cell * (1u32 << level) as f32;
        view.frustum.intersects_box(
            min.extend(z_range[0] * view.z_scale - pad),
            (min + size).extend(z_range[1] * view.z_scale + pad),
        )
    }

    fn emit(
        &self,
        level: usize,
        origin: Vec2,
        size: f32,
        z_range: [f32; 2],
        view: &LodView,
        selection: &mut Selection,
    ) {
        if !has_data(z_range) || !self.visible(level, origin, size, z_range, view) {
            return;
        }
        let patch = PatchInstance { origin: origin.into(), size, level: level as f32 };
        selection.patches.push(patch);

        let tolerance = 0.5 * self.cell;
        let touches = [
            (Edge::West, origin.x < tolerance),
            (Edge::East, origin.x + size > self.extent - tolerance),
            (Edge::South, origin.y < tolerance),
            (Edge::North, origin.y + size > self.extent - tolerance),
        ];
        for (edge, _) in touches.into_iter().filter(|(_, touching)| *touching) {
            selection.skirts[edge as usize].push(patch);
        }
    }
}

/// The shared patch grid plus one skirt strip per region edge.
pub struct PatchMesh {
    /// Grid coordinates `x`, `y` in quads; `z` is 1 on the bottom row of a
    /// skirt and 0 everywhere else.
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u16>,
    pub patch_indices: Range<u32>,
    /// Index ranges of the skirt strips, indexed by [`Edge`].
    pub skirt_indices: [Range<u32>; 4],
}

impl Default for PatchMesh {
    fn default() -> Self {
        Self::new()
    }
}

impl PatchMesh {
    pub fn new() -> Self {
        let quads = PATCH_QUADS as u16;
        let row = quads + 1;
        let mut vertices: Vec<[f32; 3]> =
            (0..row).flat_map(|y| (0..row).map(move |x| [x as f32, y as f32, 0.0])).collect();
        let mut indices: Vec<u16> = (0..quads)
            .flat_map(|y| (0..quads).map(move |x| y * row + x))
            .flat_map(|corner| [corner, corner + 1, corner + row + 1, corner, corner + row + 1, corner + row])
            .collect();
        let patch_indices = 0..indices.len() as u32;

        let far = quads as f32;
        let edges: [fn(f32, f32) -> [f32; 2]; 4] =
            [|_, t| [0.0, t], |far, t| [far, t], |_, t| [t, 0.0], |far, t| [t, far]];
        let skirt_indices = edges.map(|position| {
            let base = vertices.len() as u16;
            let start = indices.len() as u32;
            for step in 0..row {
                let [x, y] = position(far, step as f32);
                vertices.extend([[x, y, 0.0], [x, y, 1.0]]);
            }
            for step in 0..quads {
                let top = base + 2 * step;
                indices.extend([top, top + 1, top + 3, top, top + 3, top + 2]);
            }
            start..indices.len() as u32
        });
        Self { vertices, indices, patch_indices, skirt_indices }
    }
}

#[cfg(test)]
mod tests {
    use relief_terrain::terrain::leaf_ranges;

    use super::*;

    fn quadtree() -> Quadtree {
        // 1024 cells of 10 m: 16 leaves per side, five levels.
        let heights = Grid::from_fn(1024, 1024, |x, y| (x + y) as f32);
        Quadtree::new(&leaf_ranges(&heights, 16, |_, _| true), 10.0)
    }

    fn view(camera: Vec3, range0: f32) -> LodView {
        LodView { camera, reference_z: 0.0, range0, frustum: Frustum::everything(), z_scale: 1.0 }
    }

    #[test]
    fn height_ranges_merge_upward() {
        let tree = quadtree();
        assert_eq!(tree.top_level(), 4);
        assert_eq!(tree.ranges[4].get(0, 0), [0.0, 2046.0]);
    }

    #[test]
    fn selection_tiles_the_terrain_exactly_once() {
        let tree = quadtree();
        let mut selection = Selection::default();
        for camera in
            [Vec3::new(5000.0, 5000.0, 300.0), Vec3::new(-2000.0, 12_000.0, 4000.0), Vec3::new(100.0, 100.0, 50_000.0)]
        {
            tree.select(&view(camera, 2000.0), &mut selection);
            let area: f64 = selection.patches.iter().map(|patch| (patch.size as f64).powi(2)).sum();
            assert!((area - 10_240.0f64.powi(2)).abs() < 1.0, "patches cover {area} m2");

            let overlaps = |a: &PatchInstance, b: &PatchInstance| {
                (0..2).all(|i| a.origin[i] < b.origin[i] + b.size && b.origin[i] < a.origin[i] + a.size)
            };
            for (i, a) in selection.patches.iter().enumerate() {
                assert!(selection.patches[i + 1..].iter().all(|b| !overlaps(a, b)));
            }
        }
    }

    #[test]
    fn nodes_without_data_are_not_drawn() {
        // Only the western half of the terrain holds data.
        let heights = Grid::from_fn(1024, 1024, |x, y| (x + y) as f32);
        let tree = Quadtree::new(&leaf_ranges(&heights, 16, |x, _| x < 8), 10.0);
        let mut selection = Selection::default();
        for camera in [Vec3::new(2000.0, 5000.0, 300.0), Vec3::new(9000.0, 5000.0, 30_000.0)] {
            tree.select(&view(camera, 2000.0), &mut selection);
            let area: f64 = selection.patches.iter().map(|patch| (patch.size as f64).powi(2)).sum();
            assert!((area - 0.5 * 10_240.0f64.powi(2)).abs() < 1.0, "patches cover {area} m2");
            assert!(selection.patches.iter().all(|patch| patch.origin[0] + patch.size <= 5120.5));
        }
    }

    #[test]
    fn detail_follows_the_camera() {
        let tree = quadtree();
        let mut selection = Selection::default();
        let camera = Vec3::new(2000.0, 2000.0, 100.0);
        tree.select(&view(camera, 2000.0), &mut selection);

        let level_at = |point: Vec2| {
            let inside = |patch: &&PatchInstance| {
                let origin = Vec2::from(patch.origin);
                point.cmpge(origin).all() && point.cmplt(origin + patch.size).all()
            };
            selection.patches.iter().find(inside).expect("the terrain is covered").level
        };
        assert_eq!(level_at(Vec2::new(2000.0, 2000.0)), 0.0);
        assert!(level_at(Vec2::new(9000.0, 9000.0)) >= 2.0);
    }

    #[test]
    fn every_patch_stays_within_its_morph_range() {
        // No vertex of a level-l patch may lie beyond that level's range,
        // or it would be morphed past the grid its coarser neighbours use.
        let tree = quadtree();
        let mut selection = Selection::default();
        let lod_view = view(Vec3::new(3000.0, 7000.0, 800.0), tree.range0(4.0, 0.0008));
        tree.select(&lod_view, &mut selection);
        for patch in &selection.patches {
            let level = patch.level as usize;
            if level == tree.top_level() {
                continue;
            }
            let origin = Vec2::from(patch.origin);
            let corners = [origin, origin + Vec2::X * patch.size, origin + Vec2::Y * patch.size, origin + patch.size];
            let farthest = corners.iter().map(|&c| lod_view.distance_to(c, c)).fold(0.0, f32::max);
            let next_morph_start = MORPH_START * lod_view.range0 * (2u32 << level) as f32;
            assert!(farthest <= next_morph_start, "level {level} patch reaches {farthest} m");
        }
    }

    #[test]
    fn skirts_line_the_region_border() {
        let tree = quadtree();
        let mut selection = Selection::default();
        tree.select(&view(Vec3::new(5000.0, 5000.0, 60_000.0), 2000.0), &mut selection);
        for edge in &selection.skirts {
            let length: f32 = edge.iter().map(|patch| patch.size).sum();
            assert!((length - 10_240.0).abs() < 1.0);
        }
    }

    #[test]
    fn frustum_rejects_boxes_behind_the_camera() {
        let view = Mat4::look_to_rh(Vec3::ZERO, Vec3::Y, Vec3::Z);
        let projection = Mat4::perspective_infinite_reverse_rh(1.0, 1.5, 1.0);
        let frustum = Frustum::from_view_projection(&(projection * view));
        assert!(frustum.intersects_box(Vec3::new(-10.0, 50.0, -10.0), Vec3::new(10.0, 60.0, 10.0)));
        assert!(!frustum.intersects_box(Vec3::new(-10.0, -60.0, -10.0), Vec3::new(10.0, -50.0, 10.0)));
        assert!(!frustum.intersects_box(Vec3::new(500.0, 50.0, -10.0), Vec3::new(520.0, 60.0, 10.0)));
    }

    #[test]
    fn mesh_has_a_grid_and_four_strips() {
        let mesh = PatchMesh::new();
        let grid_vertices = (PATCH_QUADS as usize + 1).pow(2);
        assert_eq!(mesh.vertices.len(), grid_vertices + 4 * 2 * (PATCH_QUADS as usize + 1));
        assert_eq!(mesh.patch_indices.len(), (PATCH_QUADS * PATCH_QUADS * 6) as usize);
        assert!(mesh.skirt_indices.iter().all(|range| range.len() == (PATCH_QUADS * 6) as usize));
        assert!(mesh.indices.iter().all(|&index| (index as usize) < mesh.vertices.len()));
    }
}
