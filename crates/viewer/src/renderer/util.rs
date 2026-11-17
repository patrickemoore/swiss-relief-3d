//! Small helpers that keep bind group setup readable.

/// A uniform buffer binding.
pub const UNIFORM: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Uniform,
    has_dynamic_offset: false,
    min_binding_size: None,
};

/// A filterable 2D float texture binding.
pub const TEXTURE: wgpu::BindingType = wgpu::BindingType::Texture {
    sample_type: wgpu::TextureSampleType::Float { filterable: true },
    view_dimension: wgpu::TextureViewDimension::D2,
    multisampled: false,
};

/// A filtering sampler binding.
pub const SAMPLER: wgpu::BindingType = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);

/// A layout whose bindings are numbered in the order given.
pub fn bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    visibility: wgpu::ShaderStages,
    types: &[wgpu::BindingType],
) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = types
        .iter()
        .enumerate()
        .map(|(binding, &ty)| wgpu::BindGroupLayoutEntry { binding: binding as u32, visibility, ty, count: None })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries })
}

/// A bind group whose resources are numbered in the order given.
pub fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource],
) -> wgpu::BindGroup {
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry { binding: binding as u32, resource: resource.clone() })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
}

pub fn texture(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

/// An uninitialized uniform buffer sized for `T`.
pub fn uniform_buffer<T>(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// A filterable 2D float texture array binding.
pub const TEXTURE_ARRAY: wgpu::BindingType = wgpu::BindingType::Texture {
    sample_type: wgpu::TextureSampleType::Float { filterable: true },
    view_dimension: wgpu::TextureViewDimension::D2Array,
    multisampled: false,
};

/// A 2D float texture array binding for a format that cannot be filtered.
/// Shaders load its texels instead of sampling it.
pub const PLAIN_TEXTURE_ARRAY: wgpu::BindingType = wgpu::BindingType::Texture {
    sample_type: wgpu::TextureSampleType::Float { filterable: false },
    view_dimension: wgpu::TextureViewDimension::D2Array,
    multisampled: false,
};
