//! Cut Mesh: a triangle mesh cut by planes and boxes directly, without rebuilding it. Every
//! triangle the cut misses is kept as it is; the triangles it crosses are split along it, and
//! the openings the cut leaves in a closed part of the mesh are capped flat.
//!
//! A cut through an open part of the mesh (a hole in a scan) leaves that part of the cut open:
//! the report counts those loops; Solidify Mesh closes them.

use std::collections::{HashMap, HashSet};

use crate::bvh::Bvh;
use crate::mesh::{add, cross, dot, mul, norm, sub, Mesh, V3};

/// A region of space a cut keeps or removes.
#[derive(Clone, Debug)]
pub enum Region {
    /// The side of the plane `normal` points to.
    Plane { point: V3, normal: V3 },
    /// A box: its center, half sizes and three orthonormal axes.
    Box { center: V3, half: V3, axes: [V3; 3] },
}

impl Region {
    /// An axis-aligned box from its corners.
    pub fn aabb(lo: V3, hi: V3) -> Region {
        Region::Box {
            center: [
                (lo[0] + hi[0]) / 2.0,
                (lo[1] + hi[1]) / 2.0,
                (lo[2] + hi[2]) / 2.0,
            ],
            half: [
                (hi[0] - lo[0]).abs() / 2.0,
                (hi[1] - lo[1]).abs() / 2.0,
                (hi[2] - lo[2]).abs() / 2.0,
            ],
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// The half-spaces whose intersection is the region: a point on each plane and its normal
    /// pointing into the region.
    fn planes(&self) -> Result<Vec<(V3, V3)>, String> {
        match self {
            Region::Plane { point, normal } => {
                let l = norm(*normal);
                if !(l > 0.0 && l.is_finite()) || !point.iter().all(|x| x.is_finite()) {
                    return Err("a plane needs a point and a non-zero normal".into());
                }
                Ok(vec![(*point, mul(*normal, 1.0 / l))])
            }
            Region::Box { center, half, axes } => {
                if !half.iter().all(|h| *h > 0.0 && h.is_finite()) {
                    return Err("a box needs a size greater than 0 on every axis".into());
                }
                let mut out = Vec::new();
                for k in 0..3 {
                    let l = norm(axes[k]);
                    if !(l > 0.0 && l.is_finite()) {
                        return Err("a box's axes must be non-zero".into());
                    }
                    let a = mul(axes[k], 1.0 / l);
                    out.push((add(*center, mul(a, -half[k])), a));
                    out.push((add(*center, mul(a, half[k])), mul(a, -1.0)));
                }
                Ok(out)
            }
        }
    }
}

/// One step of a cut: keep what is inside the region, or what is outside it.
#[derive(Clone, Debug)]
pub struct Step {
    pub region: Region,
    pub keep: bool,
}

/// What a cut did.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CutReport {
    /// Triangles the cuts crossed (split along them).
    pub triangles_split: usize,
    /// Triangles removed (whole or the part of a split one).
    pub triangles_removed: usize,
    /// Flat caps added where the cuts opened a closed part of the mesh.
    pub caps: usize,
    /// Loops of the cut left open: the cut crosses a hole in the mesh there.
    pub open_loops: usize,
}

/// `m` cut by each step in turn; caps the openings when `cap`.
pub fn cut(m: &Mesh, steps: &[Step], cap: bool) -> Result<(Mesh, CutReport), String> {
    let mut cur = m.clone();
    let mut report = CutReport::default();
    for s in steps {
        let planes = s.region.planes()?;
        let (next, r) = one(&cur, &planes, s.keep, cap);
        report.triangles_split += r.triangles_split;
        report.triangles_removed += r.triangles_removed;
        report.caps += r.caps;
        report.open_loops += r.open_loops;
        cur = next;
    }
    Ok((cur, report))
}

/// `m` split by one region into what is inside it and what is outside it, each capped when
/// `cap`.
pub fn split(m: &Mesh, region: &Region, cap: bool) -> Result<[(Mesh, CutReport); 2], String> {
    let planes = region.planes()?;
    Ok([one(m, &planes, true, cap), one(m, &planes, false, cap)])
}

fn one(m: &Mesh, planes: &[(V3, V3)], keep_inside: bool, cap: bool) -> (Mesh, CutReport) {
    let mut report = CutReport::default();
    if m.f.is_empty() {
        return (m.clone(), report);
    }
    let size = norm(m.size()).max(1e-12);
    let eps = 1e-9 * size;
    let dist = |p: V3, j: usize| {
        let d = dot(sub(p, planes[j].0), planes[j].1);
        if d.abs() < eps {
            0.0
        } else {
            d
        }
    };
    let mut v = m.v.clone();
    let mut tris = m.f.clone();
    let mut split: HashSet<usize> = HashSet::new();
    for j in 0..planes.len() {
        let mut d: Vec<f64> = v.iter().map(|p| dist(*p, j)).collect();
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut next = Vec::with_capacity(tris.len());
        for (ti, t) in tris.iter().enumerate() {
            let s = t.map(|i| d[i as usize]);
            if !(s.iter().any(|x| *x > 0.0) && s.iter().any(|x| *x < 0.0)) {
                next.push(*t);
                continue;
            }
            split.insert(ti);
            let (mut pos, mut neg) = (Vec::with_capacity(4), Vec::with_capacity(4));
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                let (da, db) = (d[a as usize], d[b as usize]);
                if da >= 0.0 {
                    pos.push(a);
                }
                if da <= 0.0 {
                    neg.push(a);
                }
                if da * db < 0.0 {
                    let key = (a.min(b), a.max(b));
                    let x = *cache.entry(key).or_insert_with(|| {
                        let (p, q) = (v[key.0 as usize], v[key.1 as usize]);
                        let (dp, dq) = (d[key.0 as usize], d[key.1 as usize]);
                        let w = dp / (dp - dq);
                        v.push(add(p, mul(sub(q, p), w)));
                        d.push(0.0);
                        (v.len() - 1) as u32
                    });
                    pos.push(x);
                    neg.push(x);
                }
            }
            for poly in [pos, neg] {
                for k in 1..poly.len().saturating_sub(1) {
                    let f = [poly[0], poly[k], poly[k + 1]];
                    if f[0] != f[1] && f[1] != f[2] && f[0] != f[2] {
                        next.push(f);
                    }
                }
            }
        }
        tris = next;
    }
    report.triangles_split = split.len();

    // Which pieces stay: a piece lying in a plane is judged just behind it, inside the solid.
    let inside = |q: V3| (0..planes.len()).all(|j| dot(sub(q, planes[j].0), planes[j].1) >= 0.0);
    let mut kept = Vec::with_capacity(tris.len());
    let mut removed = 0usize;
    for t in &tris {
        let p = t.map(|i| v[i as usize]);
        let c = mul(add(add(p[0], p[1]), p[2]), 1.0 / 3.0);
        // A piece lying in a cut plane (a sliver of the split, or a face of the mesh there):
        // the cap covers it when there is one.
        if cap && (0..planes.len()).any(|j| p.iter().all(|q| dist(*q, j) == 0.0)) {
            removed += 1;
            continue;
        }
        let on = (0..planes.len()).any(|j| dist(c, j) == 0.0);
        let q = if on {
            let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            let l = norm(n);
            if l > 0.0 {
                add(c, mul(n, -1e-6 * size / l))
            } else {
                c
            }
        } else {
            c
        };
        if inside(q) == keep_inside {
            kept.push(*t);
        } else {
            removed += 1;
        }
    }
    report.triangles_removed = removed;

    if cap {
        let bvh = (planes.len() > 1).then(|| Bvh::new(m));
        let caps = caps(
            &mut v,
            &kept,
            planes,
            keep_inside,
            eps,
            bvh.as_ref(),
            &mut report,
        );
        kept.extend(caps);
    }
    let mut out = Mesh { v, f: kept };
    out.compact();
    (out, report)
}

/// The cap triangles of each plane's face of the region: the openings the kept triangles
/// leave in it, plus (for a box) the face's edges where they run inside the mesh.
fn caps(
    v: &mut Vec<V3>,
    kept: &[[u32; 3]],
    planes: &[(V3, V3)],
    keep_inside: bool,
    eps: f64,
    bvh: Option<&Bvh>,
    report: &mut CutReport,
) -> Vec<[u32; 3]> {
    let d = |p: V3, j: usize| dot(sub(p, planes[j].0), planes[j].1);
    let on = |p: V3, j: usize| d(p, j).abs() < 4.0 * eps;
    let within = |p: V3, j: usize| (0..planes.len()).all(|k| k == j || d(p, k) >= -4.0 * eps);
    // Directed boundary edges of what is kept.
    let mut directed: HashSet<(u32, u32)> = HashSet::with_capacity(kept.len() * 3);
    for t in kept {
        for k in 0..3 {
            directed.insert((t[k], t[(k + 1) % 3]));
        }
    }
    let boundary: Vec<(u32, u32)> = directed
        .iter()
        .filter(|(a, b)| !directed.contains(&(*b, *a)))
        .copied()
        .collect();
    let mut corners: HashMap<[usize; 3], u32> = HashMap::new();
    let mut out = Vec::new();
    for j in 0..planes.len() {
        let n = planes[j].1;
        // The cap faces out of what is kept.
        let cap_n = if keep_inside { mul(n, -1.0) } else { n };
        // Its boundary: the kept boundary on this face, reversed.
        let edges: Vec<(u32, u32)> = boundary
            .iter()
            .filter(|(a, b)| {
                let (p, q) = (v[*a as usize], v[*b as usize]);
                on(p, j) && on(q, j) && within(p, j) && within(q, j)
            })
            .map(|(a, b)| (*b, *a))
            .collect();
        // A box face's edges, where they run inside the mesh.
        let mut ridge: Vec<(u32, u32)> = Vec::new();
        if let Some(bvh) = bvh {
            let verts: HashSet<u32> = edges.iter().flat_map(|(a, b)| [*a, *b]).collect();
            for k in 0..planes.len() {
                if k == j || dot(n, planes[k].1).abs() > 0.5 {
                    continue;
                }
                let dir = cross(n, planes[k].1);
                let l = norm(dir);
                let dir = mul(dir, 1.0 / l);
                let mut pts: Vec<(f64, u32)> = verts
                    .iter()
                    .filter(|&&i| on(v[i as usize], k))
                    .map(|&i| (dot(v[i as usize], dir), i))
                    .collect();
                // The face's corners on this edge: where a third plane meets it.
                for m in 0..planes.len() {
                    if m == j
                        || m == k
                        || dot(n, planes[m].1).abs() > 0.5
                        || dot(planes[k].1, planes[m].1).abs() > 0.5
                    {
                        continue;
                    }
                    let Some(p) = meet([planes[j], planes[k], planes[m]]) else {
                        continue;
                    };
                    let mut key = [j, k, m];
                    key.sort();
                    let i = *corners.entry(key).or_insert_with(|| {
                        v.push(p);
                        (v.len() - 1) as u32
                    });
                    pts.push((dot(p, dir), i));
                }
                pts.sort_by(|a, b| a.0.total_cmp(&b.0));
                pts.dedup_by_key(|p| p.1);
                for w in pts.windows(2) {
                    let mid = mul(add(v[w[0].1 as usize], v[w[1].1 as usize]), 0.5);
                    if bvh.winding(mid, 2.0) >= 0.5 {
                        ridge.push((w[0].1, w[1].1));
                    }
                }
            }
        }
        if edges.is_empty() && ridge.is_empty() {
            continue;
        }
        let (loops, open) = chain(&edges, &ridge);
        report.open_loops += open;
        // A 2D frame in the plane, counter-clockwise seen from the cap's outside.
        let u = {
            let a = if cap_n[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let u = cross(a, cap_n);
            mul(u, 1.0 / norm(u))
        };
        let w = cross(cap_n, u);
        let to2 = |i: u32| {
            let p = v[i as usize];
            [dot(p, u), dot(p, w)]
        };
        let rings: Vec<(Vec<u32>, Vec<[f64; 2]>, f64)> = loops
            .into_iter()
            .map(|(mut l, oriented)| {
                let mut pts: Vec<[f64; 2]> = l.iter().map(|&i| to2(i)).collect();
                let mut a = area(&pts);
                // A loop of a box face's edges only (the face wholly inside the mesh) has no
                // direction of its own: it is an outer ring.
                if !oriented && a < 0.0 {
                    l.reverse();
                    pts.reverse();
                    a = -a;
                }
                (l, pts, a)
            })
            .filter(|r| r.2.abs() > 0.0)
            .collect();
        // Outer rings run counter-clockwise; each hole goes in the smallest outer ring around it.
        let outers: Vec<usize> = (0..rings.len()).filter(|&i| rings[i].2 > 0.0).collect();
        let mut holes: Vec<Vec<usize>> = vec![Vec::new(); rings.len()];
        for h in (0..rings.len()).filter(|&i| rings[i].2 < 0.0) {
            let p = rings[h].1[0];
            if let Some(&o) = outers
                .iter()
                .filter(|&&o| contains(&rings[o].1, p))
                .min_by(|&&a, &&b| rings[a].2.total_cmp(&rings[b].2))
            {
                holes[o].push(h);
            } else {
                report.open_loops += 1;
            }
        }
        for &o in &outers {
            let mut flat: Vec<f64> = Vec::new();
            let mut ids: Vec<u32> = Vec::new();
            let mut starts = Vec::new();
            for &r in std::iter::once(&o).chain(holes[o].iter()) {
                if r != o {
                    starts.push(ids.len());
                }
                for (k, p) in rings[r].1.iter().enumerate() {
                    flat.extend_from_slice(p);
                    ids.push(rings[r].0[k]);
                }
            }
            let Ok(tri) = earcutr::earcut(&flat, &starts, 2) else {
                report.open_loops += 1;
                continue;
            };
            // Earcut winds every triangle of a polygon the same way: orient them all by the
            // largest one (a sliver's own normal is noise).
            let tris: Vec<[u32; 3]> = tri
                .chunks_exact(3)
                .map(|t| [ids[t[0]], ids[t[1]], ids[t[2]]])
                .filter(|f| f[0] != f[1] && f[1] != f[2] && f[0] != f[2])
                .collect();
            let a2 = |f: &[u32; 3]| {
                let p = f.map(&to2);
                (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                    - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])
            };
            let flip = tris
                .iter()
                .map(a2)
                .max_by(|a, b| a.abs().total_cmp(&b.abs()))
                .is_some_and(|a| a < 0.0);
            out.extend(
                tris.into_iter()
                    .map(|f| if flip { [f[0], f[2], f[1]] } else { f }),
            );
            report.caps += 1;
        }
    }
    out
}

/// Closed loops from directed edges and undirected ones (a box face's edges); returns them and
/// how many chains did not close.
fn chain(directed: &[(u32, u32)], free: &[(u32, u32)]) -> (Vec<(Vec<u32>, bool)>, usize) {
    let mut out_of: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, (a, _)) in directed.iter().enumerate() {
        out_of.entry(*a).or_default().push(i);
    }
    let mut at: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, (a, b)) in free.iter().enumerate() {
        at.entry(*a).or_default().push(i);
        at.entry(*b).or_default().push(i);
    }
    let mut used_d = vec![false; directed.len()];
    let mut used_f = vec![false; free.len()];
    let mut loops = Vec::new();
    let mut open = 0;
    // Start at a directed edge where there is one (its direction orients the loop), else at a
    // free one.
    let starts: Vec<(bool, usize)> = (0..directed.len())
        .map(|i| (true, i))
        .chain((0..free.len()).map(|i| (false, i)))
        .collect();
    for (is_d, i) in starts {
        if (is_d && used_d[i]) || (!is_d && used_f[i]) {
            continue;
        }
        let (start, mut cur) = if is_d {
            used_d[i] = true;
            directed[i]
        } else {
            used_f[i] = true;
            free[i]
        };
        let mut l = vec![start];
        let mut oriented = is_d;
        let closed = loop {
            if cur == start {
                break true;
            }
            l.push(cur);
            if let Some(&e) = out_of
                .get(&cur)
                .and_then(|es| es.iter().find(|&&e| !used_d[e]))
            {
                used_d[e] = true;
                oriented = true;
                cur = directed[e].1;
            } else if let Some(&e) = at.get(&cur).and_then(|es| es.iter().find(|&&e| !used_f[e])) {
                used_f[e] = true;
                cur = if free[e].0 == cur {
                    free[e].1
                } else {
                    free[e].0
                };
            } else {
                break false;
            }
        };
        if closed && l.len() >= 3 {
            loops.push((l, oriented));
        } else {
            open += 1;
        }
    }
    (loops, open)
}

/// Signed area of a 2D ring.
fn area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    (0..n)
        .map(|i| p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1])
        .sum::<f64>()
        / 2.0
}

/// Whether `q` is inside a 2D ring (even-odd).
fn contains(ring: &[[f64; 2]], q: [f64; 2]) -> bool {
    let n = ring.len();
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        if (a[1] > q[1]) != (b[1] > q[1]) {
            let x = a[0] + (q[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if x > q[0] {
                inside = !inside;
            }
        }
    }
    inside
}

/// The point where three planes meet.
fn meet(p: [(V3, V3); 3]) -> Option<V3> {
    let (n0, n1, n2) = (p[0].1, p[1].1, p[2].1);
    let det = dot(n0, cross(n1, n2));
    if det.abs() < 1e-12 {
        return None;
    }
    let (d0, d1, d2) = (dot(n0, p[0].0), dot(n1, p[1].0), dot(n2, p[2].0));
    let r = add(
        add(mul(cross(n1, n2), d0), mul(cross(n2, n0), d1)),
        mul(cross(n0, n1), d2),
    );
    Some(mul(r, 1.0 / det))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::verify;

    fn cube(lo: V3, hi: V3) -> Mesh {
        let c = |i: usize| {
            [
                if i & 1 == 0 { lo[0] } else { hi[0] },
                if i & 2 == 0 { lo[1] } else { hi[1] },
                if i & 4 == 0 { lo[2] } else { hi[2] },
            ]
        };
        let f = [
            [0, 2, 3],
            [0, 3, 1],
            [4, 5, 7],
            [4, 7, 6],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
        ];
        Mesh::from_soup(f.iter().map(|t: &[usize; 3]| t.map(c)))
    }

    /// A finer closed mesh: a sphere.
    fn sphere(r: f64, n: usize) -> Mesh {
        let p = |i: usize, j: usize| {
            let (th, ph) = (
                std::f64::consts::PI * i as f64 / n as f64,
                2.0 * std::f64::consts::PI * j as f64 / (2 * n) as f64,
            );
            [
                r * th.sin() * ph.cos(),
                r * th.sin() * ph.sin(),
                r * th.cos(),
            ]
        };
        let mut soup = Vec::new();
        for i in 0..n {
            for j in 0..2 * n {
                let (a, b, c, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
                if i != 0 {
                    soup.push([a, b, d]);
                }
                if i != n - 1 {
                    soup.push([b, c, d]);
                }
            }
        }
        // Poles: snap the near-zero coordinates so the fan closes.
        let snap = |x: f64| if x.abs() < 1e-12 { 0.0 } else { x };
        Mesh::from_soup(soup.into_iter().map(|t| t.map(|q| q.map(snap))))
    }

    fn closed(m: &Mesh) -> bool {
        let v = verify(m);
        v.watertight
    }

    #[test]
    fn a_plane_cut_is_capped_and_closed() {
        let s = sphere(10.0, 24);
        assert!(closed(&s));
        let (m, r) = cut(
            &s,
            &[Step {
                region: Region::Plane {
                    point: [0.0, 0.0, 3.3],
                    normal: [0.0, 0.0, 1.0],
                },
                keep: true,
            }],
            true,
        )
        .unwrap();
        assert!(closed(&m), "{r:?}");
        assert_eq!(r.caps, 1);
        assert_eq!(r.open_loops, 0);
        let (lo, _) = m.bounds();
        assert!((lo[2] - 3.3).abs() < 1e-9);
        // The cap volume: a spherical cap of height 6.7.
        let h: f64 = 6.7;
        let want = std::f64::consts::PI * h * h * (3.0 * 10.0 - h) / 3.0;
        assert!(
            (m.volume() - want).abs() / want < 0.03,
            "{} vs {want}",
            m.volume()
        );
    }

    #[test]
    fn split_gives_both_halves() {
        let s = sphere(10.0, 20);
        let [(a, _), (b, _)] = split(
            &s,
            &Region::Plane {
                point: [0.0, 0.0, 0.0],
                normal: [0.3, 0.1, 1.0],
            },
            true,
        )
        .unwrap();
        assert!(closed(&a) && closed(&b));
        assert!(((a.volume() + b.volume()) - s.volume()).abs() < 1e-6 * s.volume());
    }

    #[test]
    fn a_box_is_removed_and_kept() {
        let c = cube([0.0; 3], [10.0; 3]);
        // A notch out of a corner: the box pokes out of the cube.
        let (m, r) = cut(
            &c,
            &[Step {
                region: Region::aabb([6.0, 6.0, 6.0], [12.0, 12.0, 12.0]),
                keep: false,
            }],
            true,
        )
        .unwrap();
        assert!(closed(&m), "{r:?}");
        assert!(
            (m.volume() - (1000.0 - 64.0)).abs() < 1e-6,
            "{}",
            m.volume()
        );
        let (k, r) = cut(
            &c,
            &[Step {
                region: Region::aabb([6.0, 6.0, 6.0], [12.0, 12.0, 12.0]),
                keep: true,
            }],
            true,
        )
        .unwrap();
        assert!(closed(&k), "{r:?}");
        assert!((k.volume() - 64.0).abs() < 1e-6, "{}", k.volume());
        // A box through the sphere, rotated.
        let s = sphere(10.0, 24);
        let (a, b) = (std::f64::consts::FRAC_1_SQRT_2, 0.0);
        let rb = Region::Box {
            center: [2.0, 1.0, 0.5],
            half: [4.0, 30.0, 3.0],
            axes: [[a, a, b], [-a, a, b], [0.0, 0.0, 1.0]],
        };
        let (m, r) = cut(
            &s,
            &[Step {
                region: rb.clone(),
                keep: false,
            }],
            true,
        )
        .unwrap();
        assert!(closed(&m), "{r:?}");
        let (k, r) = cut(
            &s,
            &[Step {
                region: rb,
                keep: true,
            }],
            true,
        )
        .unwrap();
        assert!(closed(&k), "{r:?}");
        assert!(((m.volume() + k.volume()) - s.volume()).abs() < 1e-6 * s.volume());
    }

    #[test]
    fn a_box_inside_leaves_a_closed_cavity() {
        let c = cube([0.0; 3], [10.0; 3]);
        let (m, r) = cut(
            &c,
            &[Step {
                region: Region::aabb([3.0; 3], [5.0; 3]),
                keep: false,
            }],
            true,
        )
        .unwrap();
        assert!(closed(&m), "{r:?}");
        assert!((m.volume() - 992.0).abs() < 1e-6, "{}", m.volume());
    }

    #[test]
    fn an_open_mesh_is_cut_and_its_hole_left_open() {
        let mut s = sphere(10.0, 20);
        // A hole through the cut.
        let keep: Vec<[u32; 3]> =
            s.f.iter()
                .copied()
                .filter(|t| {
                    let c = t.iter().map(|&i| s.v[i as usize]).fold([0.0; 3], add);
                    !(c[0] > 20.0 && c[2].abs() < 6.0)
                })
                .collect();
        s.f = keep;
        let (m, r) = cut(
            &s,
            &[Step {
                region: Region::Plane {
                    point: [0.0; 3],
                    normal: [0.0, 0.0, 1.0],
                },
                keep: true,
            }],
            true,
        )
        .unwrap();
        assert!(r.open_loops >= 1);
        assert!(!m.f.is_empty());
    }
}
