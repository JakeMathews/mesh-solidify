//! Winding numbers on the GPU (wgpu: Metal, Vulkan or DX12). The same tree walk as
//! [`Bvh::winding`](crate::bvh::Bvh::winding), in single precision, for batches of points.
//!
//! Single precision is enough here: the result is only compared with the iso value, and a point
//! whose winding number is within rounding of it lies in a hole, where either side is as good.

use std::sync::{mpsc, Mutex};

use crate::bvh::Bvh;
use crate::mesh::V3;

/// Points per dispatch.
pub const BATCH: usize = 1 << 19;

const SHADER: &str = r#"
struct Node {
    c: vec3<f32>,
    r2: f32,
    n: vec3<f32>,
    first: u32,
    count: u32,
    p0: u32,
    p1: u32,
    p2: u32,
}
struct Params {
    count: u32,
    p0: u32,
    p1: u32,
    p2: u32,
}
@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> tris: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn solid_angle(t: u32, q: vec3<f32>) -> f32 {
    let a = tris[3u * t].xyz - q;
    let b = tris[3u * t + 1u].xyz - q;
    let c = tris[3u * t + 2u].xyz - q;
    let la = length(a);
    let lb = length(b);
    let lc = length(c);
    let num = dot(a, cross(b, c));
    let den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    return 2.0 * atan2(num, den);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let q = points[i].xyz;
    var w = 0.0;
    var stack: array<u32, 64>;
    var sp = 1;
    stack[0] = 0u;
    while (sp > 0) {
        sp -= 1;
        let nd = nodes[stack[sp]];
        let d = nd.c - q;
        let d2 = dot(d, d);
        if (d2 > nd.r2) {
            w += dot(nd.n, d) / (d2 * sqrt(d2));
            continue;
        }
        if (nd.count > 0u) {
            for (var t = nd.first; t < nd.first + nd.count; t += 1u) {
                w += solid_angle(t, q);
            }
        } else if (sp < 62) {
            stack[sp] = nd.first;
            stack[sp + 1] = nd.first + 1u;
            sp += 2;
        }
    }
    result[i] = w * 0.07957747154594767;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNode {
    c: [f32; 3],
    r2: f32,
    n: [f32; 3],
    first: u32,
    count: u32,
    pad: [u32; 3],
}

struct Set {
    points: wgpu::Buffer,
    result: wgpu::Buffer,
    read: wgpu::Buffer,
    params: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

pub struct Winding {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    nodes: wgpu::Buffer,
    tris: wgpu::Buffer,
    /// Positions are sent relative to this point, to keep single precision where it matters.
    centre: V3,
    pool: Mutex<Vec<Set>>,
    pub name: String,
}

impl Winding {
    /// Upload `bvh` to the best GPU, or say why there is none to use.
    pub fn new(bvh: &Bvh, beta: f64, centre: V3) -> Result<Winding, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("{} is a software renderer", info.name));
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mesh-solidify"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;

        let (bn, bt) = bvh.raw();
        let rel = |p: V3| {
            [
                (p[0] - centre[0]) as f32,
                (p[1] - centre[1]) as f32,
                (p[2] - centre[2]) as f32,
            ]
        };
        let nodes: Vec<GpuNode> = bn
            .iter()
            .map(|n| GpuNode {
                c: rel(n.0),
                r2: (beta * beta * n.1 * n.1) as f32,
                n: [n.2[0] as f32, n.2[1] as f32, n.2[2] as f32],
                first: n.3,
                count: n.4,
                pad: [0; 3],
            })
            .collect();
        let mut tris: Vec<[f32; 4]> = Vec::with_capacity(bt.len() * 3);
        for t in bt {
            for p in t {
                let r = rel(*p);
                tris.push([r[0], r[1], r[2], 0.0]);
            }
        }
        if tris.is_empty() {
            return Err("no triangles".into());
        }
        let max = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size as _) as u64;
        let (nb, tb): (&[u8], &[u8]) = (bytemuck::cast_slice(&nodes), bytemuck::cast_slice(&tris));
        if nb.len() as u64 > max || tb.len() as u64 > max {
            return Err(format!(
                "the mesh needs {} MB on the GPU, more than its {} MB per buffer",
                nb.len().max(tb.len()) >> 20,
                max >> 20
            ));
        }
        let upload = |label, bytes: &[u8]| {
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&b, 0, bytes);
            b
        };
        let nodes = upload("nodes", nb);
        let tris = upload("tris", tb);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("winding"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("winding"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Winding {
            device,
            queue,
            pipeline,
            nodes,
            tris,
            centre,
            pool: Mutex::new(Vec::new()),
            name: format!("{} ({:?})", info.name, info.backend),
        })
    }

    fn set(&self) -> Set {
        if let Some(s) = self.pool.lock().unwrap().pop() {
            return s;
        }
        let buf = |label, size: usize, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        let points = buf("points", BATCH * 16, U::STORAGE | U::COPY_DST);
        let result = buf("result", BATCH * 4, U::STORAGE | U::COPY_SRC);
        let read = buf("read", BATCH * 4, U::MAP_READ | U::COPY_DST);
        let params = buf("params", 16, U::UNIFORM | U::COPY_DST);
        fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            }
        }
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &self.nodes),
                entry(1, &self.tris),
                entry(2, &points),
                entry(3, &result),
                entry(4, &params),
            ],
        });
        Set {
            points,
            result,
            read,
            params,
            bind,
        }
    }

    /// Winding numbers at `pts`, appended to `out`. Blocks; callable from many threads.
    pub fn eval(&self, pts: &[V3], out: &mut Vec<f32>) -> Result<(), String> {
        let mut staged: Vec<[f32; 4]> = Vec::with_capacity(pts.len().min(BATCH));
        for chunk in pts.chunks(BATCH) {
            staged.clear();
            staged.extend(chunk.iter().map(|p| {
                [
                    (p[0] - self.centre[0]) as f32,
                    (p[1] - self.centre[1]) as f32,
                    (p[2] - self.centre[2]) as f32,
                    0.0,
                ]
            }));
            let n = chunk.len();
            let s = self.set();
            self.queue
                .write_buffer(&s.points, 0, bytemuck::cast_slice(&staged));
            self.queue
                .write_buffer(&s.params, 0, bytemuck::cast_slice(&[n as u32, 0, 0, 0]));
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &s.bind, &[]);
                pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
            }
            enc.copy_buffer_to_buffer(&s.result, 0, &s.read, 0, (n * 4) as u64);
            let index = self.queue.submit([enc.finish()]);
            let (tx, rx) = mpsc::channel();
            s.read
                .slice(..(n * 4) as u64)
                .map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: None,
                })
                .map_err(|e| e.to_string())?;
            // The callback may run on another thread's poll.
            loop {
                match rx.try_recv() {
                    Ok(r) => {
                        r.map_err(|e| e.to_string())?;
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        self.device
                            .poll(wgpu::PollType::Poll)
                            .map_err(|e| e.to_string())?;
                        std::thread::yield_now();
                    }
                    Err(e) => return Err(e.to_string()),
                }
            }
            {
                let view = s
                    .read
                    .slice(..(n * 4) as u64)
                    .get_mapped_range()
                    .map_err(|e| format!("{e:?}"))?;
                out.extend_from_slice(bytemuck::cast_slice(&view));
            }
            s.read.unmap();
            self.pool.lock().unwrap().push(s);
        }
        Ok(())
    }
}
