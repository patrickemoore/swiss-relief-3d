//! Per-pass GPU timings from timestamp queries (plan section 8).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// The passes that are timed, in frame order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    LocalLight = 0,
    Terrain = 1,
    DepthEdges = 2,
    Composite = 3,
    Ui = 4,
}

impl Pass {
    pub const ALL: [Self; 5] = [Self::LocalLight, Self::Terrain, Self::DepthEdges, Self::Composite, Self::Ui];

    pub fn label(self) -> &'static str {
        match self {
            Self::LocalLight => "Local light field",
            Self::Terrain => "Terrain",
            Self::DepthEdges => "Depth-edge blur",
            Self::Composite => "Sky and output",
            Self::Ui => "Interface",
        }
    }

    /// Column name in evaluation tables.
    pub fn key(self) -> &'static str {
        match self {
            Self::LocalLight => "local_light",
            Self::Terrain => "terrain",
            Self::DepthEdges => "depth_edges",
            Self::Composite => "composite",
            Self::Ui => "interface",
        }
    }

    fn begin_query(self) -> u32 {
        2 * self as u32
    }

    fn end_query(self) -> u32 {
        2 * self as u32 + 1
    }
}

const PASS_COUNT: usize = Pass::ALL.len();
const QUERY_COUNT: u32 = 2 * PASS_COUNT as u32;
/// Each pass resolves into its own slot of the buffer, because resolve
/// offsets must be multiples of `QUERY_RESOLVE_BUFFER_ALIGNMENT`.
const RESOLVE_STRIDE: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;
const RESOLVE_BYTES: u64 = PASS_COUNT as u64 * RESOLVE_STRIDE;
/// Readbacks in flight at once; results arrive a frame or two late.
const READBACK_SLOTS: usize = 4;

struct Readback {
    buffer: wgpu::Buffer,
    /// Set by the map callback once the buffer can be read.
    ready: Arc<AtomicBool>,
    in_flight: bool,
    /// Which passes wrote their queries in the frame this slot holds.
    written: [bool; PASS_COUNT],
}

struct Queries {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readbacks: Vec<Readback>,
    /// Nanoseconds per timestamp tick.
    period_ns: f32,
}

/// Measures how long each pass takes on the GPU.
pub struct GpuTimers {
    queries: Option<Queries>,
    written: [bool; PASS_COUNT],
    /// The slot this frame's queries were copied into, until it is mapped.
    pending: Option<usize>,
    /// Latest measurement per pass in milliseconds, indexed by [`Pass`].
    latest_ms: [f32; PASS_COUNT],
    /// Exponentially smoothed measurement, for display.
    smoothed_ms: [f32; PASS_COUNT],
}

impl GpuTimers {
    /// Timers are inert when the device lacks timestamp queries.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, supported: bool) -> Self {
        let queries = supported.then(|| Queries {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timers"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERY_COUNT,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pass timers resolve"),
                size: RESOLVE_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readbacks: (0..READBACK_SLOTS)
                .map(|_| Readback {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("pass timers readback"),
                        size: RESOLVE_BYTES,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    ready: Arc::new(AtomicBool::new(false)),
                    in_flight: false,
                    written: [false; PASS_COUNT],
                })
                .collect(),
            period_ns: queue.get_timestamp_period(),
        });
        Self {
            queries,
            written: [false; PASS_COUNT],
            pending: None,
            latest_ms: [0.0; PASS_COUNT],
            smoothed_ms: [0.0; PASS_COUNT],
        }
    }

    pub fn supported(&self) -> bool {
        self.queries.is_some()
    }

    pub fn latest_ms(&self, pass: Pass) -> f32 {
        self.latest_ms[pass as usize]
    }

    pub fn smoothed_ms(&self, pass: Pass) -> f32 {
        self.smoothed_ms[pass as usize]
    }

    /// Declares that `pass` runs this frame. Passes that are skipped report
    /// zero instead of a stale measurement.
    pub fn mark(&mut self, pass: Pass) {
        self.written[pass as usize] = true;
    }

    /// Timestamp writes for one render pass of `pass`. `opens` and `closes`
    /// say whether it is the first and the last pass of the measured span.
    pub fn render_span(&self, pass: Pass, opens: bool, closes: bool) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.queries.as_ref().filter(|_| opens || closes).map(|queries| wgpu::RenderPassTimestampWrites {
            query_set: &queries.set,
            beginning_of_pass_write_index: opens.then(|| pass.begin_query()),
            end_of_pass_write_index: closes.then(|| pass.end_query()),
        })
    }

    /// As [`render_span`](Self::render_span), for a compute pass.
    pub fn compute_span(&self, pass: Pass, opens: bool, closes: bool) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.queries.as_ref().filter(|_| opens || closes).map(|queries| wgpu::ComputePassTimestampWrites {
            query_set: &queries.set,
            beginning_of_pass_write_index: opens.then(|| pass.begin_query()),
            end_of_pass_write_index: closes.then(|| pass.end_query()),
        })
    }

    /// Call after the frame's last pass, before submitting the encoder.
    pub fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let written = std::mem::take(&mut self.written);
        let Some(queries) = &mut self.queries else { return };
        let Some(slot) = queries.readbacks.iter().position(|readback| !readback.in_flight) else { return };

        // Only queries written this frame hold a result, so each pass that
        // ran is resolved on its own.
        for pass in Pass::ALL.into_iter().filter(|&pass| written[pass as usize]) {
            let offset = pass as u64 * RESOLVE_STRIDE;
            encoder.resolve_query_set(&queries.set, pass.begin_query()..pass.end_query() + 1, &queries.resolve, offset);
        }
        encoder.copy_buffer_to_buffer(&queries.resolve, 0, &queries.readbacks[slot].buffer, 0, RESOLVE_BYTES);
        queries.readbacks[slot].written = written;
        queries.readbacks[slot].in_flight = true;
        self.pending = Some(slot);
    }

    /// Call after submitting the encoder that [`resolve`](Self::resolve) wrote to.
    pub fn after_submit(&mut self) {
        let (Some(queries), Some(slot)) = (&self.queries, self.pending.take()) else { return };
        let ready = queries.readbacks[slot].ready.clone();
        queries.readbacks[slot].buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                ready.store(true, Ordering::Release);
            }
        });
    }

    /// Collects finished measurements. Call once per frame after polling
    /// the device.
    pub fn collect(&mut self) {
        let Some(queries) = &mut self.queries else { return };
        for readback in queries.readbacks.iter_mut().filter(|readback| readback.ready.load(Ordering::Acquire)) {
            {
                let bytes = readback.buffer.slice(..).get_mapped_range();
                let tick = |offset: usize| u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("8 bytes"));
                for pass in Pass::ALL {
                    let index = pass as usize;
                    let slot = index * RESOLVE_STRIDE as usize;
                    let elapsed = tick(slot + 8).saturating_sub(tick(slot));
                    let ms = if readback.written[index] { elapsed as f32 * queries.period_ns * 1e-6 } else { 0.0 };
                    self.latest_ms[index] = ms;
                    self.smoothed_ms[index] += 0.1 * (ms - self.smoothed_ms[index]);
                }
            }
            readback.buffer.unmap();
            readback.ready.store(false, Ordering::Release);
            readback.in_flight = false;
        }
    }
}
