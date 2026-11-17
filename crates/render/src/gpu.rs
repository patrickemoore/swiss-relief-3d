//! Device setup shared by the window and the headless tools.

use anyhow::{Context, Result};

/// The GPU device and what it supports.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
    /// Whether per-pass GPU timings are available.
    pub timestamps: bool,
    /// Whether 32-bit float textures can be sampled with filtering. Most
    /// phones cannot; the shaders then filter the terrain heights themselves.
    pub float_filtering: bool,
}

impl Gpu {
    /// An instance on the platform's primary backends, or those named by
    /// the `WGPU_BACKEND` environment variable.
    pub fn instance() -> wgpu::Instance {
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            ..Default::default()
        })
    }

    /// Opens the most capable adapter. Pass the window's surface to make
    /// sure the adapter can present to it; `None` renders offscreen only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        pollster::block_on(Self::open(instance, surface))
    }

    /// [`new`](Self::new) without blocking, as a browser requires.
    pub async fn open(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
            })
            .await
            .context("no compatible graphics adapter found")?;

        // The height pyramid is R32Float, sampled with trilinear filtering
        // where the adapter can. Setting RELIEF_NO_FLOAT_FILTERING tries the
        // other path on an adapter that can.
        let float_filtering = adapter.features().contains(wgpu::Features::FLOAT32_FILTERABLE)
            && std::env::var_os("RELIEF_NO_FLOAT_FILTERING").is_none();
        let filtering = if float_filtering { wgpu::Features::FLOAT32_FILTERABLE } else { wgpu::Features::empty() };
        // A browser offers timestamps too, but reading them back takes a
        // device poll that a page does not have.
        let timestamps =
            cfg!(not(target_arch = "wasm32")) && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let optional = if timestamps { wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("relief device"),
                    required_features: filtering | optional,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("opening the graphics device")?;

        let info = adapter.get_info();
        log::info!("rendering on {} ({:?})", info.name, info.backend);
        Ok(Self { device, queue, adapter, timestamps, float_filtering })
    }

    /// Human-readable adapter name for reports.
    pub fn adapter_name(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}
