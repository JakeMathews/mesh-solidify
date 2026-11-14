//! The whole pipeline, parameters and report.

use serde::Serialize;

use crate::bvh::Bvh;
use crate::check::{verify, Verification};
use crate::fair::Adjacency;
use crate::grid::{self, Grid};
use crate::mesh::{sub, Mesh, V3};
use crate::edit::{self, Op, Tool};
use crate::{contour, decimate, fair, par};

/// Tuning knobs. Lengths are in output millimetres (after `scale`).
#[derive(Clone, Debug, Serialize)]
pub struct Params {
    /// Uniform scale applied to the input first (0.1 = 10%).
    pub scale: f64,
    /// Grid spacing in mm. 0 or less picks the finest spacing that fits `max_voxels`.
    pub voxel: f64,
    /// Offset the surface outward by this many mm so thin parts survive slicing.
    pub min_feature: f64,
    /// Winding-number threshold for "inside" (0..1). Lower fills more.
    pub iso: f64,
    /// Triangle budget of the output.
    pub target_faces: usize,
    /// Vertices farther than `fair_tau` voxels from the input count as hole patches and are
    /// faired. 0 turns fairing off.
    pub fair_tau: f64,
    /// Extra rings of vertices faired around each patch.
    pub fair_grow: usize,
    /// Refuse grids with more nodes than this.
    pub max_voxels: usize,
    /// Fair hole patches as a thin plate (curvature-continuous: right for organic, curved
    /// shapes) instead of a membrane (spans flat between the edges around it and meets them at a
    /// crease: right for boxy, machined parts). Off by default.
    pub smooth_fill: bool,
    /// Evaluate winding numbers on the GPU when the crate is built with its `gpu` feature and
    /// there is one. On by default.
    pub gpu: bool,
    /// Keep every separate solid part of at least this volume (mm³), not only the largest.
    /// 0 keeps the largest only.
    pub min_part_mm3: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            scale: 1.0,
            voxel: 0.125,
            min_feature: 0.0,
            iso: 0.5,
            target_faces: 400_000,
            fair_tau: 0.5,
            fair_grow: 2,
            max_voxels: 400_000_000,
            smooth_fill: false,
            gpu: true,
            min_part_mm3: 0.0,
        }
    }
}

/// Progress, logging and cancellation. Calls may come from worker threads.
pub trait Monitor: Sync {
    fn progress(&self, _fraction: f32, _stage: &str) {}
    fn log(&self, _message: &str) {}
    fn cancelled(&self) -> bool {
        false
    }
}

/// Passes `mon` through with its progress mapped onto one stage of the whole bar.
struct Scaled<'a> {
    mon: &'a dyn Monitor,
    bar: &'a (dyn Fn(f32) + Sync),
}
impl Monitor for Scaled<'_> {
    fn progress(&self, fraction: f32, _stage: &str) {
        (self.bar)(fraction)
    }
    fn log(&self, message: &str) {
        self.mon.log(message)
    }
    fn cancelled(&self) -> bool {
        self.mon.cancelled()
    }
}

pub struct NoMonitor;
impl Monitor for NoMonitor {}

#[derive(Debug, Clone)]
pub enum Error {
    InvalidParam(String),
    GridTooLarge {
        dims: [usize; 3],
        voxels: usize,
        max: usize,
        suggest: f64,
    },
    Cancelled,
    Failed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::InvalidParam(s) => write!(f, "{s}"),
            Error::GridTooLarge { dims, voxels, max, suggest } => write!(
                f,
                "grid {dims:?} = {}M voxels exceeds the limit of {}M; use a voxel of {suggest:.3} mm or more, or raise the limit",
                voxels / 1_000_000,
                max / 1_000_000
            ),
            Error::Cancelled => write!(f, "cancelled"),
            Error::Failed(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub ok: bool,
    pub triangles: usize,
    pub watertight: bool,
    pub is_volume: bool,
    pub bodies: usize,
    pub size_mm: [f64; 3],
    pub volume_mm3: f64,
    pub grid: [usize; 3],
    pub voxel_mm: f64,
    pub winding_evaluations: usize,
    pub components_found: usize,
    pub components_dropped_voxels: usize,
    pub voids_filled_voxels: usize,
    pub contour_triangles: usize,
    pub faired_vertices: usize,
    /// Edit tools applied (cuts, boxes, meshes combined).
    pub tools: usize,
    /// The solid parts left out, largest first (at most 20): volume and bounds in output mm.
    pub dropped_parts: Vec<DroppedPart>,
    pub params: Params,
}

/// A separate solid part that was left out of the result.
#[derive(Clone, Debug, Serialize)]
pub struct DroppedPart {
    pub volume_mm3: f64,
    pub min_mm: [f64; 3],
    pub max_mm: [f64; 3],
}

fn band(h: f64, min_feature: f64) -> usize {
    ((min_feature / h).ceil() as usize).min(1 << 20) + 2
}

/// Grid size for a model of `size` mm at spacing `h`.
pub fn grid_dims(size: V3, h: f64, min_feature: f64) -> [usize; 3] {
    let pad = band(h, min_feature) + 2;
    size.map(|s| ((s / h).ceil() as usize).saturating_add(2 * pad + 1))
}

fn nodes(d: [usize; 3]) -> f64 {
    d[0] as f64 * d[1] as f64 * d[2] as f64
}

/// The finest spacing whose grid has at most `budget` nodes.
pub fn auto_voxel(size: V3, budget: usize, min_feature: f64) -> f64 {
    let vol = size[0].max(1e-9) * size[1].max(1e-9) * size[2].max(1e-9);
    let mut h = (vol / budget as f64).cbrt().max(1e-6);
    while nodes(grid_dims(size, h, min_feature)) > budget as f64 {
        h *= 1.02;
    }
    (h * 1000.0).ceil() / 1000.0
}

fn check(p: &Params) -> Result<(), Error> {
    let bad = |s: &str| Err(Error::InvalidParam(s.into()));
    if !(p.scale.is_finite() && p.scale > 0.0) {
        return bad("scale must be a number greater than 0");
    }
    if !p.voxel.is_finite() {
        return bad("voxel must be a finite number (0 for automatic)");
    }
    if !(p.min_part_mm3.is_finite() && p.min_part_mm3 >= 0.0) {
        return bad("min_part_mm3 must be 0 or more");
    }
    if !(p.min_feature.is_finite() && p.min_feature >= 0.0) {
        return bad("min_feature must be 0 or more");
    }
    if !(p.iso > 0.0 && p.iso < 1.0) {
        return bad("iso must be between 0 and 1");
    }
    if p.target_faces < 100 {
        return bad("target_faces must be at least 100");
    }
    if !(p.fair_tau.is_finite() && p.fair_tau >= 0.0) {
        return bad("fair_tau must be 0 or more");
    }
    if p.max_voxels < 1000 {
        return bad("max_voxels must be at least 1000");
    }
    Ok(())
}

/// Rebuild `input` as one watertight solid.
pub fn solidify(input: &Mesh, p: &Params, mon: &dyn Monitor) -> Result<(Mesh, Report), Error> {
    solidify_with(input, &[], p, mon)
}

/// Rebuild `input` as one watertight solid, edited by `tools` in the order given (cuts, boxes
/// kept or removed, meshes added, subtracted or intersected). Tools are in the input's
/// coordinates, before the scale.
pub fn solidify_with(
    input: &Mesh,
    tools: &[Tool],
    p: &Params,
    mon: &dyn Monitor,
) -> Result<(Mesh, Report), Error> {
    check(p)?;
    if input.f.is_empty() {
        return Err(Error::Failed("the mesh has no triangles".into()));
    }
    if let Some(e) = tools.iter().find_map(Tool::check) {
        return Err(Error::InvalidParam(e));
    }
    let tools: Vec<Tool> = tools.iter().map(|t| t.scaled(p.scale)).collect();
    let mut m = input.clone();
    m.scale(p.scale);
    // The grid covers the input and what is added to it, cut down to what an intersection
    // after the last addition keeps.
    let (mut lo, mut hi) = m.bounds();
    let last_union = tools.iter().rposition(|t| t.op == Op::Union);
    for (i, t) in tools.iter().enumerate() {
        let Some((a, b)) = t.bounds() else { continue };
        match t.op {
            Op::Union => {
                for k in 0..3 {
                    lo[k] = lo[k].min(a[k]);
                    hi[k] = hi[k].max(b[k]);
                }
            }
            Op::Intersect if last_union.is_none_or(|u| i > u) => {
                for k in 0..3 {
                    lo[k] = lo[k].max(a[k]);
                    hi[k] = hi[k].min(b[k]);
                }
            }
            _ => {}
        }
    }
    // A cut square to an axis bounds the grid too, so what is kept gets the finer voxel.
    for (i, t) in tools.iter().enumerate() {
        if t.op != Op::Intersect || last_union.is_some_and(|u| i < u) {
            continue;
        }
        if let edit::Shape::HalfSpace { point, normal } = &t.shape {
            let l = crate::mesh::norm(*normal);
            for k in 0..3 {
                let c = normal[k] / l;
                if c > 1.0 - 1e-12 {
                    lo[k] = lo[k].max(point[k]);
                } else if c < -1.0 + 1e-12 {
                    hi[k] = hi[k].min(point[k]);
                }
            }
        }
    }
    if (0..3).any(|k| hi[k] <= lo[k]) {
        return Err(Error::Failed("nothing is left: the regions kept miss the mesh".into()));
    }
    let size = sub(hi, lo);
    let h = if p.voxel > 0.0 {
        p.voxel
    } else {
        auto_voxel(size, p.max_voxels, p.min_feature)
    };
    let k_band = band(h, p.min_feature);
    let pad = (k_band + 2) as f64;
    let dims = grid_dims(size, h, p.min_feature);
    // In f64: the product overflows usize on 32-bit WebAssembly for absurd voxel sizes.
    let count = nodes(dims);
    if count > p.max_voxels as f64 {
        let voxels = count.min(usize::MAX as f64) as usize;
        let suggest = auto_voxel(size, p.max_voxels, p.min_feature);
        return Err(Error::GridTooLarge {
            dims,
            voxels,
            max: p.max_voxels,
            suggest,
        });
    }
    let voxels = count as usize;
    let mut g = Grid {
        n: dims,
        origin: [lo[0] - pad * h, lo[1] - pad * h, lo[2] - pad * h],
        h,
    };
    // A flat face or cut that lies on a layer of grid nodes has a distance of exactly 0 there,
    // which makes slivers that cannot be simplified away: move the grid off every such level
    // (the input's own large axis-aligned faces, its flat bottom first among them, and the
    // tools' planes), by the offset that keeps them all farthest from a layer.
    for k in 0..3 {
        let mut levels: Vec<f64> = tools.iter().flat_map(|t| t.levels(k)).collect();
        levels.extend(flat_levels(&m, k, 4.0 * h * h));
        if levels.is_empty() {
            continue;
        }
        let clear = |shift: f64| {
            levels
                .iter()
                .map(|c| {
                    let f = ((c - (g.origin[k] - shift * h)) / h).rem_euclid(1.0);
                    f.min(1.0 - f)
                })
                .fold(f64::INFINITY, f64::min)
        };
        let best = [0.0, 0.5, 0.25, 0.75, 0.375, 0.625, 0.125, 0.875]
            .into_iter()
            .map(|s| (s, clear(s)))
            .fold((0.0, -1.0), |a, b| if b.1 > a.1 + 1e-9 { b } else { a });
        if best.0 > 0.0 {
            g.origin[k] -= best.0 * h;
            g.n[k] += 1;
        }
    }
    let g = g;
    mon.log(&format!(
        "input {} triangles; grid {:?} = {:.1}M nodes at {:.3} mm",
        m.f.len(),
        dims,
        voxels as f64 / 1e6,
        h
    ));

    // One bar over the whole run: each stage fills its share.
    let shares: &[(&str, f32)] = &[
        ("winding number", 0.25),
        ("signed distance", 0.07),
        ("contouring", 0.08),
        ("fairing hole patches", 0.20),
        ("decimating", 0.40),
    ];
    let stage = |n: usize| {
        let lo: f32 = shares[..n].iter().map(|s| s.1).sum();
        let (name, w) = shares[n];
        move |f: f32| mon.progress(lo + w * f.clamp(0.0, 1.0), name)
    };
    let mut at = 0;
    let mut next = || {
        at += 1;
        stage(at - 1)
    };

    let bar = next();
    bar(0.0);
    let bvh = Bvh::new(&m);
    let scaled = Scaled { mon, bar: &bar };
    #[allow(unused_mut)]
    let mut on_gpu: Option<(Vec<u8>, usize)> = None;
    #[cfg(feature = "gpu")]
    let mut gpu: Option<crate::gpu::Winding> = None;
    #[cfg(feature = "gpu")]
    if p.gpu {
        let centre = g.pos(dims[0] / 2, dims[1] / 2, dims[2] / 2);
        match crate::gpu::Winding::new(&bvh, 2.0, centre) {
            Ok(w) => {
                mon.log(&format!("using the GPU: {}", w.name));
                match grid::inside_mask_gpu(&bvh, &w, &g, p.iso, 0.5 * h, &scaled) {
                    Ok(r) => {
                        on_gpu = Some(r);
                        gpu = Some(w);
                    }
                    Err(grid::GpuStop::Cancelled) => return Err(Error::Cancelled),
                    Err(grid::GpuStop::Failed(e)) => {
                        mon.log(&format!("the GPU pass failed ({e}); using the CPU"))
                    }
                }
            }
            Err(e) => mon.log(&format!("no GPU to use ({e}); using the CPU")),
        }
    }
    let (mut mask, evals) = match on_gpu {
        Some(r) => r,
        None => {
            grid::inside_mask(&bvh, &g, p.iso, 0.5 * h, &scaled).map_err(|_| Error::Cancelled)?
        }
    };
    mon.log(&format!(
        "winding number: {:.1}M evaluations ({:.0}% of the grid)",
        evals as f64 / 1e6,
        100.0 * evals as f64 / voxels as f64
    ));

    let min_voxels = if p.min_part_mm3 > 0.0 { ((p.min_part_mm3 / (h * h * h)).ceil() as usize).max(1) } else { 0 };
    let first = grid::clean(&mut mask, &g, min_voxels);
    let (mut dropped, filled, comps) = (first.dropped, first.filled, first.comps);
    let mut parts = first.parts;
    mon.log(&format!(
        "{comps} solid components; kept {}, dropped {dropped} voxels, filled {filled} void voxels",
        if min_voxels > 0 { "those of at least min_part_mm3" } else { "the largest" }
    ));
    if mask.iter().all(|&x| x == 0) {
        return Err(Error::Failed(
            "nothing is inside; the mesh may be too small for this voxel size".into(),
        ));
    }
    if mon.cancelled() {
        return Err(Error::Cancelled);
    }

    let ready = edit::Ready::new(&tools, &g, p.iso, &|| mon.cancelled()).map_err(|_| Error::Cancelled)?;
    if !tools.is_empty() {
        mon.log(&format!("{} edit tools applied", tools.len()));
    }
    let far = (k_band + 1) as f64 * h;
    let bar = next();
    let (field, last) = contour::field(
        &bvh,
        &g,
        &mask,
        far,
        p.min_feature,
        &ready,
        min_voxels,
        #[cfg(feature = "gpu")]
        gpu.as_ref(),
        &bar,
    );
    #[cfg(feature = "gpu")]
    drop(gpu);
    if let Some(last) = last {
        dropped += last.dropped;
        parts.extend(last.parts);
    }
    parts.sort_by(|a, b| b.voxels.cmp(&a.voxels));
    let dropped_parts: Vec<DroppedPart> = parts
        .iter()
        .take(20)
        .map(|q| DroppedPart {
            volume_mm3: q.voxels as f64 * h * h * h,
            min_mm: g.pos(q.lo[0], q.lo[1], q.lo[2]),
            max_mm: g.pos(q.hi[0], q.hi[1], q.hi[2]),
        })
        .collect();
    let bar = next();
    let mut out = contour::march(&field, &bar);
    drop(field);
    drop(mask);
    let contour_triangles = out.f.len();
    mon.log(&format!(
        "marching tetrahedra: {contour_triangles} triangles"
    ));
    if mon.cancelled() {
        return Err(Error::Cancelled);
    }

    let mut faired = 0;
    let bar = next();
    if p.fair_tau > 0.0 {
        bar(0.0);
        let thr = p.min_feature + p.fair_tau * h;
        // A vertex on a tool's surface was cut there, not invented over a hole.
        let near_tool = p.fair_tau * h;
        let farv: Vec<bool> = par::map(out.v.len(), |i| {
            bvh.dist2(out.v[i], thr * thr) >= thr * thr && !ready.iter().any(|t| t.near(out.v[i], near_tool))
        });
        let adj = Adjacency::new(&out);
        faired = fair::fair(
            &mut out,
            &adj,
            &farv,
            p.fair_grow,
            !p.smooth_fill,
            400,
            &bar,
        );
        mon.log(&format!(
            "faired {faired} hole-patch vertices ({:.1}%)",
            100.0 * faired as f64 / out.v.len().max(1) as f64
        ));
    }
    if mon.cancelled() {
        return Err(Error::Cancelled);
    }

    let bar = next();
    bar(0.0);
    decimate::decimate(&mut out, p.target_faces, &|| mon.cancelled(), &bar)
        .map_err(|_| Error::Cancelled)?;
    bar(1.0);
    if !ready.is_empty() {
        let eps = 0.05 * h;
        let snapped: Vec<V3> = par::map(out.v.len(), |i| ready.iter().fold(out.v[i], |p, t| t.snap(p, eps)));
        out.v = snapped;
    }
    let v: Verification = verify(&out);
    let s = out.size();
    mon.log(&format!(
        "output {} triangles, watertight={}, bodies={}, size={:.2} x {:.2} x {:.2} mm",
        out.f.len(),
        v.watertight,
        v.bodies,
        s[0],
        s[1],
        s[2]
    ));
    let report = Report {
        ok: v.is_volume && v.bodies == 1,
        triangles: out.f.len(),
        watertight: v.watertight,
        is_volume: v.is_volume,
        bodies: v.bodies,
        size_mm: s.map(|x| (x * 1000.0).round() / 1000.0),
        volume_mm3: v.volume_mm3,
        grid: dims,
        voxel_mm: h,
        winding_evaluations: evals,
        components_found: comps,
        components_dropped_voxels: dropped,
        voids_filled_voxels: filled,
        contour_triangles,
        faired_vertices: faired,
        tools: tools.len(),
        dropped_parts,
        params: Params {
            voxel: h,
            ..p.clone()
        },
    };
    Ok((out, report))
}

/// The coordinates along axis `k` of the mesh's flat faces square to that axis, each of at
/// least `min_area` (faces at the same level added up).
fn flat_levels(m: &Mesh, k: usize, min_area: f64) -> Vec<f64> {
    let (lo, hi) = m.bounds();
    let tol = 1e-9 * (0..3).map(|i| hi[i] - lo[i]).fold(0.0, f64::max).max(1e-12);
    let mut at: Vec<(f64, f64)> = Vec::new();
    for t in &m.f {
        let p = t.map(|i| m.v[i as usize]);
        let c = p[0][k];
        if (p[1][k] - c).abs() > tol || (p[2][k] - c).abs() > tol {
            continue;
        }
        let n = crate::mesh::cross(sub(p[1], p[0]), sub(p[2], p[0]));
        at.push((c, crate::mesh::norm(n) / 2.0));
    }
    at.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::new();
    let mut i = 0;
    while i < at.len() {
        let (c, mut a) = at[i];
        let mut j = i + 1;
        while j < at.len() && at[j].0 - c <= 1e3 * tol {
            a += at[j].1;
            j += 1;
        }
        if a >= min_area {
            out.push(c);
        }
        i = j;
    }
    out
}
