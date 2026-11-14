//! Narrow-band signed distance around the mask boundary, contoured with marching
//! tetrahedra (Kuhn subdivision: crack-free and manifold for a field without zeros).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::bvh::Bvh;
use crate::grid::Grid;
use crate::mesh::{cross, dot, sub, Mesh, V3};
use crate::par;

/// Field block edge in nodes.
const S: usize = 8;

/// Fx-style hasher for integer keys (std's SipHash is several times slower here).
#[derive(Default)]
pub struct FxHasher(u64);
impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}
pub type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

pub struct Field<'a> {
    g: Grid,
    mask: std::borrow::Cow<'a, [u8]>,
    nb: [usize; 3],
    slot: Vec<u32>,
    pool: Vec<f32>,
    far_in: f32,
    far_out: f32,
}

impl Field<'_> {
    #[inline]
    fn get(&self, i: usize, j: usize, k: usize) -> f32 {
        let b = i / S + self.nb[0] * (j / S + self.nb[1] * (k / S));
        let s = self.slot[b];
        if s != u32::MAX {
            self.pool[s as usize * S * S * S + i % S + S * (j % S + S * (k % S))]
        } else if self.mask[self.g.idx(i, j, k)] != 0 {
            self.far_in
        } else {
            self.far_out
        }
    }
}

/// Signed distance (negative inside), clamped to `far`, offset outward by `offset`, stored only
/// in blocks near the mask boundary. Elsewhere the field is +-far.
pub fn field<'a>(
    bvh: &Bvh,
    g: &Grid,
    mask: &'a [u8],
    far: f64,
    offset: f64,
    #[cfg(feature = "gpu")] gpu: Option<&crate::gpu::Winding>,
    progress: &(dyn Fn(f32) + Sync),
) -> Field<'a> {
    let nb = [g.n[0].div_ceil(S), g.n[1].div_ceil(S), g.n[2].div_ceil(S)];
    let nblocks = nb[0] * nb[1] * nb[2];
    // Blocks holding a node whose +x, +y or +z neighbour is on the other side.
    let boundary: Vec<bool> = par::map(nblocks, |b| {
        let (bi, bj, bk) = (b % nb[0], (b / nb[0]) % nb[1], b / (nb[0] * nb[1]));
        for k in bk * S..((bk + 1) * S).min(g.n[2]) {
            for j in bj * S..((bj + 1) * S).min(g.n[1]) {
                for i in bi * S..((bi + 1) * S).min(g.n[0]) {
                    let m = mask[g.idx(i, j, k)];
                    if (i + 1 < g.n[0] && mask[g.idx(i + 1, j, k)] != m)
                        || (j + 1 < g.n[1] && mask[g.idx(i, j + 1, k)] != m)
                        || (k + 1 < g.n[2] && mask[g.idx(i, j, k + 1)] != m)
                    {
                        return true;
                    }
                }
            }
        }
        false
    });
    // Dilate so every cell the offset surface can cross has stored corners.
    let reach = 1 + ((offset / g.h).ceil() as usize + 2).div_ceil(S);
    let mut surf = boundary.clone();
    for (b, _) in boundary.iter().enumerate().filter(|(_, &x)| x) {
        let (bi, bj, bk) = (b % nb[0], (b / nb[0]) % nb[1], b / (nb[0] * nb[1]));
        for k in bk.saturating_sub(reach)..(bk + reach + 1).min(nb[2]) {
            for j in bj.saturating_sub(reach)..(bj + reach + 1).min(nb[1]) {
                for i in bi.saturating_sub(reach)..(bi + reach + 1).min(nb[0]) {
                    surf[i + nb[0] * (j + nb[1] * k)] = true;
                }
            }
        }
    }
    let ids: Vec<usize> = (0..nblocks).filter(|&b| surf[b]).collect();
    let mut slot = vec![u32::MAX; nblocks];
    for (s, &b) in ids.iter().enumerate() {
        slot[b] = s as u32;
    }
    let tiny = (1e-6 * g.h) as f32;
    let mut pool = vec![0f32; ids.len() * S * S * S];
    let done = AtomicUsize::new(0);
    // Signed and offset value of a node from its distance to the input.
    let value = |inside: bool, d: f64| {
        let mut f = (if inside { -d } else { d } - offset) as f32;
        if f.abs() < tiny {
            f = if inside { -tiny } else { tiny };
        }
        f
    };
    let node = |b: usize, c: usize| {
        let (bi, bj, bk) = (b % nb[0], (b / nb[0]) % nb[1], b / (nb[0] * nb[1]));
        (bi * S + c % S, bj * S + (c / S) % S, bk * S + c / (S * S))
    };
    let within = |(i, j, k): (usize, usize, usize)| i < g.n[0] && j < g.n[1] && k < g.n[2];

    // On the GPU: the stored blocks in groups of one batch of points.
    #[cfg(feature = "gpu")]
    let on_gpu = gpu.is_some_and(|gpu| {
        const GROUP: usize = crate::gpu::BATCH / (S * S * S);
        let failed = std::sync::atomic::AtomicBool::new(false);
        let groups = ids.len().div_ceil(GROUP);
        par::chunks_mut(&mut pool, GROUP * S * S * S, |gi, chunk| {
            if failed.load(Ordering::Relaxed) {
                return;
            }
            let mut pts = Vec::with_capacity(chunk.len());
            for c in 0..chunk.len() {
                let (i, j, k) = node(ids[gi * GROUP + c / (S * S * S)], c % (S * S * S));
                pts.push(g.pos(i, j, k));
            }
            let mut d = Vec::with_capacity(chunk.len());
            if gpu.dist(&pts, far, &mut d).is_err() {
                failed.store(true, Ordering::Relaxed);
                return;
            }
            for (c, out) in chunk.iter_mut().enumerate() {
                let n = node(ids[gi * GROUP + c / (S * S * S)], c % (S * S * S));
                if within(n) {
                    *out = value(mask[g.idx(n.0, n.1, n.2)] != 0, d[c] as f64);
                }
            }
            progress((done.fetch_add(1, Ordering::Relaxed) + 1) as f32 / groups as f32);
        });
        !failed.load(Ordering::Relaxed)
    });
    #[cfg(not(feature = "gpu"))]
    let on_gpu = false;

    if !on_gpu {
        let every = (ids.len() / 200).max(1);
        par::chunks_mut(&mut pool, S * S * S, |s, chunk| {
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % every == 0 {
                progress(d as f32 / ids.len() as f32);
            }
            for (c, out) in chunk.iter_mut().enumerate() {
                let n = node(ids[s], c);
                if within(n) {
                    let d = bvh.dist2(g.pos(n.0, n.1, n.2), far * far).sqrt();
                    *out = value(mask[g.idx(n.0, n.1, n.2)] != 0, d);
                }
            }
        });
    }
    // An offset surface is not the mask's: it also wraps input that lies outside the solid
    // (the loose parts dropped earlier) and seals gaps narrower than twice the offset into
    // closed pockets. Clean the offset inside/outside the same way, so one shell comes out.
    let mut mask = std::borrow::Cow::Borrowed(mask);
    if offset > 0.0 {
        let mut after = mask.to_vec();
        for (s, &b) in ids.iter().enumerate() {
            for c in 0..S * S * S {
                let n = node(b, c);
                if within(n) {
                    after[g.idx(n.0, n.1, n.2)] = (pool[s * S * S * S + c] < 0.0) as u8;
                }
            }
        }
        crate::grid::clean(&mut after, g);
        let step = (0.5 * g.h) as f32;
        par::chunks_mut(&mut pool, S * S * S, |s, chunk| {
            for (c, out) in chunk.iter_mut().enumerate() {
                let n = node(ids[s], c);
                if within(n) {
                    let inside = after[g.idx(n.0, n.1, n.2)] != 0;
                    if inside != (*out < 0.0) {
                        *out = if inside { -step } else { step };
                    }
                }
            }
        });
        mask = std::borrow::Cow::Owned(after);
    }
    Field {
        g: *g,
        mask,
        nb,
        slot,
        pool,
        far_in: (-far - offset) as f32,
        far_out: (far - offset) as f32,
    }
}

/// Kuhn's six tetrahedra of a cube; corners are bit sets (x = 1, y = 2, z = 4).
const TETS: [[u8; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

/// Contour the zero level of `f` (outward = positive side).
///
/// Blocks are contoured in parallel in groups, each with its own vertices; the vertices are
/// then merged by the grid edge they sit on.
pub fn march(f: &Field, progress: &(dyn Fn(f32) + Sync)) -> Mesh {
    let blocks: Vec<usize> = (0..f.slot.len())
        .filter(|&b| f.slot[b] != u32::MAX)
        .collect();
    const GROUP: usize = 128;
    let ngroups = blocks.len().div_ceil(GROUP);
    let done = AtomicUsize::new(0);
    let every = (ngroups / 200).max(1);
    let parts: Vec<(Vec<u64>, Mesh)> = par::map(ngroups, |gi| {
        let part = march_blocks(f, &blocks[gi * GROUP..((gi + 1) * GROUP).min(blocks.len())]);
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % every == 0 {
            progress(0.8 * d as f32 / ngroups as f32);
        }
        part
    });
    // Merge: sort every part's vertices by edge key; equal keys are one vertex.
    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let mut all: Vec<(u64, u32, u32)> = Vec::with_capacity(total);
    for (gi, (keys, _)) in parts.iter().enumerate() {
        all.extend(
            keys.iter()
                .enumerate()
                .map(|(l, &k)| (k, gi as u32, l as u32)),
        );
    }
    par::sort_unstable(&mut all);
    let mut mesh = Mesh::default();
    let mut remap: Vec<Vec<u32>> = parts.iter().map(|p| vec![0u32; p.0.len()]).collect();
    let mut last = u64::MAX;
    for &(k, gi, l) in &all {
        if k != last || mesh.v.is_empty() {
            mesh.v.push(parts[gi as usize].1.v[l as usize]);
            last = k;
        }
        remap[gi as usize][l as usize] = (mesh.v.len() - 1) as u32;
    }
    drop(all);
    progress(0.9);
    mesh.f.reserve(parts.iter().map(|p| p.1.f.len()).sum());
    for (gi, (_, part)) in parts.iter().enumerate() {
        let r = &remap[gi];
        mesh.f.extend(
            part.f
                .iter()
                .map(|t| [r[t[0] as usize], r[t[1] as usize], r[t[2] as usize]]),
        );
    }
    mesh
}

/// Contour some blocks: the mesh, and for each of its vertices the key of its grid edge.
fn march_blocks(f: &Field, blocks: &[usize]) -> (Vec<u64>, Mesh) {
    let g = &f.g;
    let mut mesh = Mesh::default();
    let mut keys: Vec<u64> = Vec::new();
    let mut verts: FxMap<u64, u32> = FxMap::default();
    for &b in blocks {
        let nb = f.nb;
        let (bi, bj, bk) = (b % nb[0], (b / nb[0]) % nb[1], b / (nb[0] * nb[1]));
        for k in bk * S..((bk + 1) * S).min(g.n[2] - 1) {
            for j in bj * S..((bj + 1) * S).min(g.n[1] - 1) {
                for i in bi * S..((bi + 1) * S).min(g.n[0] - 1) {
                    let mut v = [0f32; 8];
                    for (c, val) in v.iter_mut().enumerate() {
                        *val = f.get(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                    }
                    let npos = v.iter().filter(|&&x| x > 0.0).count();
                    if npos == 0 || npos == 8 {
                        continue;
                    }
                    let corner = |c: u8| -> V3 {
                        g.pos(
                            i + (c & 1) as usize,
                            j + ((c >> 1) & 1) as usize,
                            k + ((c >> 2) & 1) as usize,
                        )
                    };
                    let mut vert = |a: u8, b: u8, mesh: &mut Mesh| -> u32 {
                        let (lo, hi) = if a & b == a { (a, b) } else { (b, a) };
                        let node = g.idx(
                            i + (lo & 1) as usize,
                            j + ((lo >> 1) & 1) as usize,
                            k + ((lo >> 2) & 1) as usize,
                        );
                        let key = node as u64 * 8 + (hi ^ lo) as u64;
                        *verts.entry(key).or_insert_with(|| {
                            keys.push(key);
                            let (fa, fb) = (v[lo as usize] as f64, v[hi as usize] as f64);
                            let t = (fa / (fa - fb)).clamp(0.01, 0.99);
                            let (pa, pb) = (corner(lo), corner(hi));
                            mesh.v.push([
                                pa[0] + t * (pb[0] - pa[0]),
                                pa[1] + t * (pb[1] - pa[1]),
                                pa[2] + t * (pb[2] - pa[2]),
                            ]);
                            (mesh.v.len() - 1) as u32
                        })
                    };
                    for tet in TETS {
                        let pos: Vec<u8> = tet
                            .iter()
                            .copied()
                            .filter(|&c| v[c as usize] > 0.0)
                            .collect();
                        let neg: Vec<u8> = tet
                            .iter()
                            .copied()
                            .filter(|&c| v[c as usize] <= 0.0)
                            .collect();
                        let tris: Vec<[u32; 3]> = match pos.len() {
                            1 => vec![[
                                vert(pos[0], neg[0], &mut mesh),
                                vert(pos[0], neg[1], &mut mesh),
                                vert(pos[0], neg[2], &mut mesh),
                            ]],
                            3 => vec![[
                                vert(neg[0], pos[0], &mut mesh),
                                vert(neg[0], pos[1], &mut mesh),
                                vert(neg[0], pos[2], &mut mesh),
                            ]],
                            2 => {
                                let ac = vert(pos[0], neg[0], &mut mesh);
                                let ad = vert(pos[0], neg[1], &mut mesh);
                                let bd = vert(pos[1], neg[1], &mut mesh);
                                let bc = vert(pos[1], neg[0], &mut mesh);
                                vec![[ac, ad, bd], [ac, bd, bc]]
                            }
                            _ => continue,
                        };
                        // Orient each triangle so its normal points to the positive (outside) side.
                        let avg = |cs: &[u8]| {
                            let mut s = [0.0; 3];
                            for &c in cs {
                                let p = corner(c);
                                s = [s[0] + p[0], s[1] + p[1], s[2] + p[2]];
                            }
                            let n = cs.len() as f64;
                            [s[0] / n, s[1] / n, s[2] / n]
                        };
                        let dir = sub(avg(&pos), avg(&neg));
                        for t in tris {
                            let (a, b, c) = (
                                mesh.v[t[0] as usize],
                                mesh.v[t[1] as usize],
                                mesh.v[t[2] as usize],
                            );
                            if dot(cross(sub(b, a), sub(c, a)), dir) < 0.0 {
                                mesh.f.push([t[0], t[2], t[1]]);
                            } else {
                                mesh.f.push(t);
                            }
                        }
                    }
                }
            }
        }
    }
    (keys, mesh)
}
