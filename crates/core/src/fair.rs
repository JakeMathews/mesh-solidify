//! Fairing of hole patches: vertices that are far from every input triangle were invented to
//! close a hole, so they are moved to smooth out the voxel steps of the inside/outside mask,
//! with the rest of the surface held fixed.
//!
//! - Flat (membrane, Laplacian energy): the patch spans tight between the edges around it, so
//!   it stays flat where the gap is flat and meets the faces around it at a crease, the way a
//!   missing underside or bottom would.
//! - Smooth (thin plate, bi-Laplacian energy): the patch continues the curvature around it.
//!   Right for curved surfaces; on boxy parts it bulges and rounds corners over.

use crate::mesh::Mesh;
use crate::par;

/// Compressed adjacency.
pub struct Adjacency {
    pub start: Vec<u32>,
    pub nb: Vec<u32>,
}

impl Adjacency {
    pub fn new(m: &Mesh) -> Adjacency {
        let nv = m.v.len();
        let mut deg = vec![0u32; nv + 1];
        for t in &m.f {
            for k in 0..3 {
                deg[t[k] as usize] += 2;
            }
        }
        let mut start = vec![0u32; nv + 1];
        for i in 0..nv {
            start[i + 1] = start[i] + deg[i];
        }
        let mut fill = start.clone();
        let mut nb = vec![0u32; start[nv] as usize];
        for t in &m.f {
            for k in 0..3 {
                let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
                nb[fill[a as usize] as usize] = b;
                nb[fill[a as usize] as usize + 1] = c;
                fill[a as usize] += 2;
            }
        }
        // Sort and dedup each list in place, then compact.
        let mut out_start = vec![0u32; nv + 1];
        let mut w = 0usize;
        for i in 0..nv {
            let (s, e) = (start[i] as usize, start[i + 1] as usize);
            let mut l: Vec<u32> = nb[s..e].to_vec();
            l.sort_unstable();
            l.dedup();
            nb[w..w + l.len()].copy_from_slice(&l);
            w += l.len();
            out_start[i + 1] = w as u32;
        }
        nb.truncate(w);
        Adjacency {
            start: out_start,
            nb,
        }
    }
    #[inline]
    pub fn of(&self, i: usize) -> &[u32] {
        &self.nb[self.start[i] as usize..self.start[i + 1] as usize]
    }
}

/// Fair the vertices marked `far` (far from the input: they patch a hole), grown by `grow`
/// rings, as a membrane (`flat`) or a thin plate. Returns how many vertices moved.
pub fn fair(
    m: &mut Mesh,
    adj: &Adjacency,
    far: &[bool],
    grow: usize,
    flat: bool,
    max_iter: usize,
    progress: &dyn Fn(f32),
) -> usize {
    let nv = m.v.len();
    // 0 = fixed and untouched, 1 = free, 2 = first fixed ring, 3 = second fixed ring.
    let mut role = vec![0u8; nv];
    let mut front: Vec<u32> = (0..nv as u32).filter(|&i| far[i as usize]).collect();
    if front.is_empty() {
        return 0;
    }
    for &i in &front {
        role[i as usize] = 1;
    }
    for _ in 0..grow {
        let mut next = Vec::new();
        for &i in &front {
            for &j in adj.of(i as usize) {
                if role[j as usize] == 0 {
                    role[j as usize] = 1;
                    next.push(j);
                }
            }
        }
        front = next;
    }
    for ring in [2u8, 3] {
        let src: Vec<u32> = (0..nv as u32)
            .filter(|&i| role[i as usize] == ring - 1)
            .collect();
        for i in src {
            for &j in adj.of(i as usize) {
                if role[j as usize] == 0 {
                    role[j as usize] = ring;
                }
            }
        }
    }
    // Local numbering: free vertices first, then rings.
    let mut order: Vec<u32> = (0..nv as u32).filter(|&i| role[i as usize] == 1).collect();
    let nfree = order.len();
    order.extend((0..nv as u32).filter(|&i| role[i as usize] == 2));
    let nkr = order.len(); // vertices where K x is needed
    order.extend((0..nv as u32).filter(|&i| role[i as usize] == 3));
    let mut local = vec![u32::MAX; nv];
    for (l, &g) in order.iter().enumerate() {
        local[g as usize] = l as u32;
    }
    // Neighbours in local numbering, compressed.
    let mut nstart = Vec::with_capacity(nkr + 1);
    let mut nlist: Vec<u32> = Vec::new();
    nstart.push(0u32);
    for &g in &order[..nkr] {
        nlist.extend(adj.of(g as usize).iter().map(|&j| local[j as usize]));
        nstart.push(nlist.len() as u32);
    }
    let nbr = |i: usize| &nlist[nstart[i] as usize..nstart[i + 1] as usize];
    let n = order.len();
    let x0: Vec<[f64; 3]> = order.iter().map(|&g| m.v[g as usize]).collect();

    const C: usize = 1 << 14;
    // out[i] = (K x)[i] for i < out.len(), for x given on all `n`; neighbours outside the work
    // set never occur for these vertices.
    let lap = |x: &[[f64; 3]], out: &mut [[f64; 3]]| {
        par::chunks_mut(out, C, |ci, chunk| {
            for (j, o) in chunk.iter_mut().enumerate() {
                let i = ci * C + j;
                let l = nbr(i);
                let d = l.len() as f64;
                let mut s = [d * x[i][0], d * x[i][1], d * x[i][2]];
                for &j in l {
                    let xj = x[j as usize];
                    s[0] -= xj[0];
                    s[1] -= xj[1];
                    s[2] -= xj[2];
                }
                *o = s;
            }
        });
    };
    // out = K x (membrane) or K K x (thin plate) on the free vertices; `tmp` is scratch.
    let apply = |x: &[[f64; 3]], tmp: &mut [[f64; 3]], out: &mut [[f64; 3]]| {
        if flat {
            lap(x, out);
        } else {
            lap(x, &mut tmp[..nkr]);
            lap(tmp, out);
        }
    };
    // Chunked so the sums come out the same on every run.
    let dot3 = |a: &[[f64; 3]], b: &[[f64; 3]]| -> [f64; 3] {
        let len = a.len().min(b.len());
        let parts = par::map(len.div_ceil(C), |ci| {
            let mut s = [0.0; 3];
            for i in ci * C..((ci + 1) * C).min(len) {
                for k in 0..3 {
                    s[k] += a[i][k] * b[i][k];
                }
            }
            s
        });
        let mut s = [0.0; 3];
        for q in parts {
            for k in 0..3 {
                s[k] += q[k];
            }
        }
        s
    };
    let mut tmp = vec![[0.0; 3]; n];
    let mut ap = vec![[0.0; 3]; nfree];
    apply(&x0, &mut tmp, &mut ap);
    let b: Vec<[f64; 3]> = ap.iter().map(|v| [-v[0], -v[1], -v[2]]).collect();
    let dinv: Vec<f64> = (0..nfree)
        .map(|i| {
            let d = nbr(i).len() as f64;
            if flat {
                1.0 / d
            } else {
                1.0 / (d * d + d)
            }
        })
        .collect();

    // Preconditioned conjugate gradient, three right-hand sides at once. `p` spans the whole
    // work set and stays zero outside the free vertices.
    let mut x = vec![[0.0; 3]; nfree];
    let mut r = b.clone();
    let mut p = vec![[0.0; 3]; n];
    for i in 0..nfree {
        p[i] = [r[i][0] * dinv[i], r[i][1] * dinv[i], r[i][2] * dinv[i]];
    }
    let bnorm = dot3(&b, &b);
    let mut rz = dot3(&r, &p);
    let goal = 1e-10f64.ln();
    let debug = std::env::var_os("MESH_SOLIDIFY_DEBUG").is_some();
    // The residual isn't monotonic; the bar is.
    let mut shown = 0.0f64;
    for it in 0..max_iter {
        apply(&p, &mut tmp, &mut ap);
        let pap = dot3(&p, &ap);
        let mut alpha = [0.0; 3];
        for k in 0..3 {
            alpha[k] = if pap[k] > 0.0 { rz[k] / pap[k] } else { 0.0 };
        }
        par::chunks_mut(&mut x, C, |ci, chunk| {
            for (j, v) in chunk.iter_mut().enumerate() {
                for k in 0..3 {
                    v[k] += alpha[k] * p[ci * C + j][k];
                }
            }
        });
        par::chunks_mut(&mut r, C, |ci, chunk| {
            for (j, v) in chunk.iter_mut().enumerate() {
                for k in 0..3 {
                    v[k] -= alpha[k] * ap[ci * C + j][k];
                }
            }
        });
        let rr = dot3(&r, &r);
        if debug && (it % 50 == 0 || it + 1 == max_iter) {
            eprintln!(
                "fair it {it}: residual {:.2e} (free {nfree})",
                (0..3)
                    .map(|k| (rr[k] / bnorm[k].max(1e-30)).sqrt())
                    .fold(0.0, f64::max)
            );
        }
        if (0..3).all(|k| rr[k] <= 1e-10 * bnorm[k].max(1e-30)) {
            break;
        }
        let reached = (0..3)
            .map(|k| (rr[k] / bnorm[k].max(1e-30)).max(1e-30).ln() / goal)
            .fold(f64::INFINITY, f64::min);
        shown = shown.max((it as f64 / max_iter as f64).max(reached).clamp(0.0, 1.0));
        progress(shown as f32);
        // r . z with z = r / diag, then p = z + beta p.
        let parts = par::map(nfree.div_ceil(C), |ci| {
            let mut s = [0.0; 3];
            for i in ci * C..((ci + 1) * C).min(nfree) {
                for k in 0..3 {
                    s[k] += r[i][k] * r[i][k] * dinv[i];
                }
            }
            s
        });
        let mut rz_new = [0.0; 3];
        for q in parts {
            for k in 0..3 {
                rz_new[k] += q[k];
            }
        }
        let mut beta = [0.0; 3];
        for k in 0..3 {
            beta[k] = if rz[k] > 0.0 { rz_new[k] / rz[k] } else { 0.0 };
        }
        par::chunks_mut(&mut p[..nfree], C, |ci, chunk| {
            for (j, v) in chunk.iter_mut().enumerate() {
                let i = ci * C + j;
                for k in 0..3 {
                    v[k] = r[i][k] * dinv[i] + beta[k] * v[k];
                }
            }
        });
        rz = rz_new;
    }
    for i in 0..nfree {
        let g = order[i] as usize;
        for k in 0..3 {
            m.v[g][k] += x[i][k];
        }
    }
    nfree
}
