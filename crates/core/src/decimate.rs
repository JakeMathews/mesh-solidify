//! Quadric error metric edge collapse (Garland & Heckbert 1997) that keeps a closed manifold
//! closed and manifold: collapses must pass the link condition and may not flip or flatten faces.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering as Atomic};

use crate::mesh::{add, cross, dot, mul, norm, norm2, sub, Mesh, V3};
use crate::par;

type Quadric = [f64; 10];

/// Triangles worse than this shape quality make a collapse proportionally more expensive.
const QUALITY: f64 = 0.3; // a2 ab ac ad b2 bc bd c2 cd d2

fn plane_quadric(n: V3, d: f64, w: f64) -> Quadric {
    let [a, b, c] = n;
    [
        a * a * w,
        a * b * w,
        a * c * w,
        a * d * w,
        b * b * w,
        b * c * w,
        b * d * w,
        c * c * w,
        c * d * w,
        d * d * w,
    ]
}

fn qadd(a: &Quadric, b: &Quadric) -> Quadric {
    let mut r = [0.0; 10];
    for i in 0..10 {
        r[i] = a[i] + b[i];
    }
    r
}

fn qeval(q: &Quadric, p: V3) -> f64 {
    let [x, y, z] = p;
    q[0] * x * x
        + 2.0 * q[1] * x * y
        + 2.0 * q[2] * x * z
        + 2.0 * q[3] * x
        + q[4] * y * y
        + 2.0 * q[5] * y * z
        + 2.0 * q[6] * y
        + q[7] * z * z
        + 2.0 * q[8] * z
        + q[9]
}

/// Point minimising `q`, if the system is well conditioned.
fn qsolve(q: &Quadric) -> Option<V3> {
    let a = [[q[0], q[1], q[2]], [q[1], q[4], q[5]], [q[2], q[5], q[7]]];
    let b = [-q[3], -q[6], -q[8]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let scale = (q[0] + q[4] + q[7]).powi(3);
    if det.abs() <= 1e-9 * scale.abs().max(1e-300) {
        return None;
    }
    let inv = 1.0 / det;
    let col = |c: usize| {
        let mut m = a;
        for r in 0..3 {
            m[r][c] = b[r];
        }
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    Some([col(0) * inv, col(1) * inv, col(2) * inv])
}

struct Entry {
    cost: f64,
    u: u32,
    v: u32,
    su: u32,
    sv: u32,
}
impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.cost == o.cost
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        o.cost.total_cmp(&self.cost) // min-heap
    }
}

/// An array whose elements different threads may change at once, as long as no two touch the
/// same element. The parallel sweeps keep to that by region: a thread only changes vertices of
/// its own region whose neighbours are all in that region, and the faces around them.
struct Sh<T> {
    ptr: *mut T,
    len: usize,
    own: Vec<T>,
}

unsafe impl<T: Send> Sync for Sh<T> {}
unsafe impl<T: Send> Send for Sh<T> {}

impl<T> Sh<T> {
    fn new(mut own: Vec<T>) -> Sh<T> {
        Sh {
            ptr: own.as_mut_ptr(),
            len: own.len(),
            own,
        }
    }
    /// The element to change. The caller must be the only one using element `i`.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    fn at(&self, i: usize) -> &mut T {
        assert!(i < self.len);
        unsafe { &mut *self.ptr.add(i) }
    }
    fn into_vec(self) -> Vec<T> {
        self.own
    }
}

impl<T> std::ops::Index<usize> for Sh<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        assert!(i < self.len);
        unsafe { &*self.ptr.add(i) }
    }
}

struct State {
    pos: Sh<V3>,
    faces: Sh<[u32; 3]>,
    alive: Sh<bool>,
    vf: Sh<Vec<u32>>,
    q: Sh<Quadric>,
    stamp: Sh<u32>,
    vdead: Sh<bool>,
    lambda: f64,
    /// Faces still alive.
    left: AtomicUsize,
}

impl State {
    fn neighbors(&self, u: u32, out: &mut Vec<u32>) {
        out.clear();
        for &f in &self.vf[u as usize] {
            for &w in &self.faces[f as usize] {
                if w != u {
                    out.push(w);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    fn candidate(&self, u: u32, v: u32) -> (f64, V3) {
        let q = qadd(&self.q[u as usize], &self.q[v as usize]);
        let (pu, pv) = (self.pos[u as usize], self.pos[v as usize]);
        let mid = mul(add(pu, pv), 0.5);
        let len2 = norm2(sub(pu, pv));
        let mut best = (qeval(&q, mid), mid);
        if let Some(x) = qsolve(&q) {
            // Stay near the edge: far optima come from nearly flat quadrics.
            if norm2(sub(x, mid)) <= len2 {
                let c = qeval(&q, x);
                if c < best.0 {
                    best = (c, x);
                }
            }
        }
        for p in [pu, pv] {
            let c = qeval(&q, p);
            if c < best.0 {
                best = (c, p);
            }
        }
        (best.0.max(0.0) + self.lambda * len2, best.1)
    }

    fn push(
        &self,
        heap: &mut BinaryHeap<Entry>,
        u: u32,
        v: u32,
        nu: &mut Vec<u32>,
        nv: &mut Vec<u32>,
    ) {
        if let Some((cost, _)) = self.evaluate(u, v, nu, nv) {
            heap.push(Entry {
                cost,
                u,
                v,
                su: self.stamp[u as usize],
                sv: self.stamp[v as usize],
            });
        }
    }

    /// Collapse v into u at x. `nu` holds u's neighbours from `valid`.
    fn collapse(&self, u: u32, v: u32, x: V3, nu: &[u32]) {
        let vfaces = std::mem::take(self.vf.at(v as usize));
        let mut gone = 0;
        for f in vfaces {
            let t = self.faces.at(f as usize);
            if t.contains(&u) {
                *self.alive.at(f as usize) = false;
                gone += 1;
            } else {
                for k in t.iter_mut() {
                    if *k == v {
                        *k = u;
                    }
                }
                self.vf.at(u as usize).push(f);
            }
        }
        self.left.fetch_sub(gone, Atomic::Relaxed);
        let alive = &self.alive;
        self.vf.at(u as usize).retain(|&f| alive[f as usize]);
        // The two opposite vertices lost a face each; they are among u's neighbours.
        for &w in nu {
            let l = self.vf.at(w as usize);
            if l.iter().any(|&f| !alive[f as usize]) {
                l.retain(|&f| alive[f as usize]);
            }
        }
        *self.pos.at(u as usize) = x;
        *self.q.at(u as usize) = qadd(&self.q[u as usize], &self.q[v as usize]);
        *self.vdead.at(v as usize) = true;
        *self.stamp.at(u as usize) += 1;
    }

    /// Whether collapsing v into u at `x` keeps the mesh manifold and unflipped.
    fn valid(
        &self,
        u: u32,
        v: u32,
        x: V3,
        nu: &mut Vec<u32>,
        nv: &mut Vec<u32>,
    ) -> Option<(f64, f64)> {
        self.neighbors(u, nu);
        self.neighbors(v, nv);
        let shared = self.vf[u as usize]
            .iter()
            .filter(|&&f| self.faces[f as usize].contains(&v))
            .count();
        if shared != 2 {
            return None;
        }
        // Link condition: common neighbours are exactly the two opposite vertices.
        let mut common = 0;
        let (mut i, mut j) = (0, 0);
        while i < nu.len() && j < nv.len() {
            match nu[i].cmp(&nv[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    common += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        if common != 2 || nu.len() + nv.len() < 8 {
            return None;
        }
        // `minq`: worst shape after the collapse; `oldq`: worst shape before, among the same
        // triangles and the two that go away.
        let (mut minq, mut oldq): (f64, f64) = (1.0, 1.0);
        let shape = |p: &[V3; 3], l: f64| {
            let e2 = norm2(sub(p[1], p[0])) + norm2(sub(p[2], p[1])) + norm2(sub(p[0], p[2]));
            2.0 * 3f64.sqrt() * l / e2.max(1e-300)
        };
        for (a, other) in [(u, v), (v, u)] {
            for &f in &self.vf[a as usize] {
                let t = self.faces[f as usize];
                let p = [
                    self.pos[t[0] as usize],
                    self.pos[t[1] as usize],
                    self.pos[t[2] as usize],
                ];
                let n0 = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                let l0 = norm(n0);
                let q0 = shape(&p, l0);
                oldq = oldq.min(q0);
                if t.contains(&other) {
                    continue;
                }
                let mut q = p;
                for k in 0..3 {
                    if t[k] == a {
                        q[k] = x;
                    }
                }
                let n1 = cross(sub(q[1], q[0]), sub(q[2], q[0]));
                let l1 = norm(n1);
                if l1 <= 1e-12 * l0.max(1e-300) {
                    return None;
                }
                // No flips. A sliver has no dependable normal, so it can't veto a collapse
                // (or slivers would lock the mesh around them).
                if q0 > 1e-3 && dot(n0, n1) < 0.5 * l0 * l1 {
                    return None;
                }
                minq = minq.min(shape(&q, l1));
            }
        }
        Some((minq, oldq))
    }

    /// Full cost of collapsing (u, v): quadric error plus a length term, raised for collapses
    /// that would make poorly shaped triangles. None when the collapse is not allowed.
    fn evaluate(&self, u: u32, v: u32, nu: &mut Vec<u32>, nv: &mut Vec<u32>) -> Option<(f64, V3)> {
        let (cost, x) = self.candidate(u, v);
        let (q, was) = self.valid(u, v, x, nu, nv)?;
        // Refuse making slivers, but always allow a collapse that leaves shapes better than
        // they were: that is how existing slivers get removed.
        if q < 0.02 && q <= was {
            return None;
        }
        let penalty = if q < QUALITY && q <= was {
            QUALITY / q.max(1e-6)
        } else {
            1.0
        };
        Some((cost * penalty, x))
    }
}

/// Reduce `m` (closed, manifold) to about `target` faces.
pub fn decimate(
    m: &mut Mesh,
    target: usize,
    cancelled: &(dyn Fn() -> bool + Sync),
    progress: &dyn Fn(f32),
) -> Result<(), ()> {
    if m.f.len() <= target {
        return Ok(());
    }
    let nv = m.v.len();
    let mut vf: Vec<Vec<u32>> = vec![Vec::with_capacity(6); nv];
    let mut q = vec![[0.0; 10]; nv];
    let mut area_sum = 0.0;
    for (i, t) in m.f.iter().enumerate() {
        let (a, b, c) = (m.v[t[0] as usize], m.v[t[1] as usize], m.v[t[2] as usize]);
        let n = cross(sub(b, a), sub(c, a));
        let l = norm(n);
        area_sum += 0.5 * l;
        if l > 0.0 {
            let nn = mul(n, 1.0 / l);
            let pq = plane_quadric(nn, -dot(nn, a), 0.5 * l);
            for &k in t {
                q[k as usize] = qadd(&q[k as usize], &pq);
            }
        }
        for &k in t {
            vf[k as usize].push(i as u32);
        }
    }
    let lambda = 1e-3 * area_sum / m.f.len() as f64;
    let nf = m.f.len();
    let s = State {
        pos: Sh::new(std::mem::take(&mut m.v)),
        faces: Sh::new(std::mem::take(&mut m.f)),
        alive: Sh::new(vec![true; nf]),
        vf: Sh::new(vf),
        q: Sh::new(q),
        stamp: Sh::new(vec![0; nv]),
        vdead: Sh::new(vec![false; nv]),
        lambda,
        left: AtomicUsize::new(nf),
    };
    let left = || s.left.load(Atomic::Relaxed);
    let start = nf;
    let report = |left: usize| progress((start - left) as f32 / (start - target).max(1) as f32);
    let stop = std::sync::atomic::AtomicBool::new(false);

    // Phase 1, threshold sweeps: walk the faces in memory order (spatially coherent) and
    // collapse each face's cheapest edge when its cost is below a threshold picked from a
    // sample of current costs. Close to greedy order, without a queue over millions of edges.
    //
    // The sweeps run in parallel by region of space. A thread collapses an edge only when both
    // ends are in its region and so are all their neighbours, so no two threads touch the same
    // vertex or face. The regions move every sweep, so no edge stays on a border.
    let switch = target.saturating_mul(2).max(target + 1000);
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let cheapest = |t: [u32; 3], ok: &dyn Fn(u32, u32) -> bool| -> (f64, u32, u32) {
        let mut best = (f64::INFINITY, 0, 0);
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if !ok(a, b) {
                continue;
            }
            let (c, _) = s.candidate(a, b);
            if c < best.0 {
                best = (c, a, b);
            }
        }
        best
    };
    const REGIONS: usize = 1024;
    let mut round = 0u64;
    let mut stalled = 0;
    while left() > switch {
        let faces_left = left();
        let mut sample: Vec<f64> = Vec::with_capacity(4096);
        let mut tries = 0;
        while sample.len() < 4096 && tries < 1 << 20 {
            tries += 1;
            let f = (next() % nf as u64) as usize;
            if s.alive[f] {
                sample.push(cheapest(s.faces[f], &|_, _| true).0);
            }
        }
        if sample.is_empty() {
            break;
        }
        sample.sort_by(f64::total_cmp);
        let frac = ((faces_left - switch) as f64 / faces_left as f64).min(0.3);
        let tau = sample[((sample.len() - 1) as f64 * frac) as usize];

        // Regions: cells of a grid shifted every round, about 50,000 faces each.
        round += 1;
        let cell = (area_sum / (faces_left as f64 / 50_000.0).max(8.0)).sqrt();
        let shift = (round as f64 * 0.381_966) % 1.0 * cell;
        let label: Vec<u16> = par::map(nv, |v| {
            let p = s.pos[v];
            let mut h = 0u64;
            for k in 0..3 {
                let c = ((p[k] + shift) / cell).floor() as i64 as u64;
                h = (h.rotate_left(21) ^ c).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            }
            ((h >> 32) % REGIONS as u64) as u16
        });
        // A vertex is free when every neighbour shares its region.
        let mut free = vec![true; nv];
        let mut count = vec![0usize; REGIONS + 1];
        for f in 0..nf {
            if !s.alive[f] {
                continue;
            }
            let t = s.faces[f];
            let l = label[t[0] as usize];
            if label[t[1] as usize] != l || label[t[2] as usize] != l {
                for k in t {
                    free[k as usize] = false;
                }
            }
            count[l as usize + 1] += 1;
        }
        for r in 0..REGIONS {
            count[r + 1] += count[r];
        }
        let mut order = vec![0u32; faces_left];
        let mut fill = count.clone();
        for f in 0..nf {
            if s.alive[f] {
                let l = label[s.faces[f][0] as usize] as usize;
                order[fill[l]] = f as u32;
                fill[l] += 1;
            }
        }
        par::each(REGIONS, |r| {
            let (mut nu, mut nvb) = (Vec::new(), Vec::new());
            let mine = |a: u32, b: u32| {
                free[a as usize] && free[b as usize] && label[a as usize] as usize == r
            };
            for (n, &f) in order[count[r]..count[r + 1]].iter().enumerate() {
                if n % 4096 == 0 {
                    if stop.load(Atomic::Relaxed) || left() <= switch {
                        break;
                    }
                    if cancelled() {
                        stop.store(true, Atomic::Relaxed);
                        break;
                    }
                }
                let f = f as usize;
                if !s.alive[f] {
                    continue;
                }
                let (c, u, v) = cheapest(s.faces[f], &mine);
                if c > tau {
                    continue;
                }
                if let Some((c, x)) = s.evaluate(u, v, &mut nu, &mut nvb) {
                    if c <= tau {
                        s.collapse(u, v, x, &nu);
                    }
                }
            }
        });
        if stop.load(Atomic::Relaxed) {
            return Err(());
        }
        report(left());
        // A sweep that removes almost nothing twice in a row is as far as sweeps go.
        if left() * 100 > faces_left * 99 {
            stalled += 1;
            if stalled == 2 {
                break;
            }
        } else {
            stalled = 0;
        }
    }

    // Phase 2: exact greedy order with a heap.
    let (mut nu, mut nvb) = (Vec::new(), Vec::new());
    let mut steps = 0u64;
    let mut heap = BinaryHeap::with_capacity(left() * 3 / 2);
    for i in 0..nf {
        if !s.alive[i] {
            continue;
        }
        let t = s.faces[i];
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if a < b {
                s.push(&mut heap, a, b, &mut nu, &mut nvb);
            }
        }
    }
    let mut around = Vec::new();
    let (mut tu, mut tv) = (Vec::new(), Vec::new());
    while left() > target {
        let Some(e) = heap.pop() else { break };
        steps += 1;
        if steps % 65536 == 0 {
            if cancelled() {
                return Err(());
            }
            report(left());
        }
        let (u, v) = (e.u, e.v);
        if s.vdead[u as usize]
            || s.vdead[v as usize]
            || s.stamp[u as usize] != e.su
            || s.stamp[v as usize] != e.sv
        {
            continue;
        }
        let Some((_, x)) = s.evaluate(u, v, &mut nu, &mut nvb) else {
            continue;
        };
        s.collapse(u, v, x, &nu);
        s.neighbors(u, &mut around);
        for &w in &around {
            s.push(&mut heap, u, w, &mut tu, &mut tv);
        }
    }
    let faces_left = left();
    let (faces, alive) = (s.faces.into_vec(), s.alive.into_vec());
    let mut out = Mesh {
        v: s.pos.into_vec(),
        f: Vec::with_capacity(faces_left),
    };
    for (i, t) in faces.iter().enumerate() {
        if alive[i] {
            out.f.push(*t);
        }
    }
    out.compact();
    *m = out;
    Ok(())
}
