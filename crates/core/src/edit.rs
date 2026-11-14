//! Edits made in the voxel grid while solidifying: keep or cut away a region (a half-space or a
//! box), or combine the solid with other closed meshes (union, subtract, intersect).
//!
//! Each tool is a signed distance (negative inside) combined with the solid's own: union is the
//! smaller of the two, subtract the larger of the solid's and the tool's negated, intersect the
//! larger. The surface that comes out is cut at full grid resolution, flat where a plane or box
//! cuts, and closed by construction.

use crate::bvh::Bvh;
use crate::grid::{self, Grid};
use crate::mesh::{dot, norm, sub, Mesh, V3};
use crate::pipeline::NoMonitor;

/// A region a tool covers.
#[derive(Clone, Debug)]
pub enum Shape {
    /// A closed mesh (its winding number says what is inside).
    Mesh(Mesh),
    /// The side of the plane through `point` that `normal` points to.
    HalfSpace { point: V3, normal: V3 },
    /// A box: `center`, half its size along each axis, and its axes (unit, perpendicular).
    Box { center: V3, half: V3, axes: [V3; 3] },
}

/// How a tool combines with the solid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// Add the tool's region.
    Union,
    /// Remove the tool's region.
    Subtract,
    /// Keep only what is inside the tool's region.
    Intersect,
}

/// One edit, applied in the order given.
#[derive(Clone, Debug)]
pub struct Tool {
    pub shape: Shape,
    pub op: Op,
}

impl Tool {
    /// Keep the side of the plane through `point` that `normal` points to.
    pub fn keep_side(point: V3, normal: V3) -> Tool {
        Tool { shape: Shape::HalfSpace { point, normal }, op: Op::Intersect }
    }
    /// An axis-aligned box from two corners.
    pub fn aabb(a: V3, b: V3, op: Op) -> Tool {
        let center = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0];
        let half = [(a[0] - b[0]).abs() / 2.0, (a[1] - b[1]).abs() / 2.0, (a[2] - b[2]).abs() / 2.0];
        Tool { shape: Shape::Box { center, half, axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }, op }
    }

    /// A description of what is wrong with the tool, if anything.
    pub(crate) fn check(&self) -> Option<String> {
        let finite = |v: &V3| v.iter().all(|x| x.is_finite());
        match &self.shape {
            Shape::Mesh(m) if m.f.is_empty() => Some("a tool mesh has no triangles".into()),
            Shape::Mesh(m) if !m.v.iter().all(finite) => Some("a tool mesh has a coordinate that is not a number".into()),
            Shape::HalfSpace { point, normal } if !finite(point) || !finite(normal) || norm(*normal) < 1e-12 => {
                Some("a plane needs a point and a non-zero normal".into())
            }
            Shape::Box { center, half, axes } => {
                if !finite(center) || !half.iter().all(|x| x.is_finite() && *x > 0.0) {
                    return Some("a box needs a centre and a size greater than 0 on every axis".into());
                }
                let u = axes.map(|a| norm(a));
                if u.iter().any(|&l| (l - 1.0).abs() > 1e-6)
                    || dot(axes[0], axes[1]).abs() > 1e-6
                    || dot(axes[1], axes[2]).abs() > 1e-6
                    || dot(axes[0], axes[2]).abs() > 1e-6
                {
                    return Some("a box's axes must be unit length and perpendicular".into());
                }
                None
            }
            _ => None,
        }
    }

    /// The tool at the scale the input is solidified at (scaled about the origin, as the input is).
    pub(crate) fn scaled(&self, s: f64) -> Tool {
        let shape = match &self.shape {
            Shape::Mesh(m) => {
                let mut m = m.clone();
                m.scale(s);
                Shape::Mesh(m)
            }
            Shape::HalfSpace { point, normal } => Shape::HalfSpace { point: point.map(|x| x * s), normal: *normal },
            Shape::Box { center, half, axes } => Shape::Box { center: center.map(|x| x * s), half: half.map(|x| x * s), axes: *axes },
        };
        Tool { shape, op: self.op }
    }

    /// Where the tool's flat faces square to axis `k` cross it.
    pub(crate) fn levels(&self, k: usize) -> Vec<f64> {
        let along = |d: &V3| (d[k].abs() / norm(*d).max(1e-300)) > 1.0 - 1e-9;
        match &self.shape {
            Shape::HalfSpace { point, normal } if along(normal) => vec![point[k]],
            Shape::Box { center, half, axes } => {
                (0..3).filter(|&a| along(&axes[a])).flat_map(|a| [center[k] - half[a], center[k] + half[a]]).collect()
            }
            _ => Vec::new(),
        }
    }

    /// The box the tool's region fits in, if it is bounded.
    pub(crate) fn bounds(&self) -> Option<(V3, V3)> {
        match &self.shape {
            Shape::Mesh(m) => Some(m.bounds()),
            Shape::HalfSpace { .. } => None,
            Shape::Box { center, half, axes } => {
                let r: V3 = std::array::from_fn(|k| (0..3).map(|a| half[a] * axes[a][k].abs()).sum());
                Some((std::array::from_fn(|k| center[k] - r[k]), std::array::from_fn(|k| center[k] + r[k])))
            }
        }
    }
}

/// A tool ready to be evaluated on the grid.
pub(crate) struct Ready {
    pub op: Op,
    kind: Kind,
}

enum Kind {
    Mesh { bvh: Bvh, mask: Vec<u8> },
    Plane { point: V3, normal: V3 },
    Box { center: V3, half: V3, axes: [V3; 3] },
}

impl Ready {
    /// Build the tools' masks on `g`. Fails only when cancelled.
    pub fn new(tools: &[Tool], g: &Grid, iso: f64, cancelled: &dyn Fn() -> bool) -> Result<Vec<Ready>, ()> {
        let mut out = Vec::with_capacity(tools.len());
        for t in tools {
            if cancelled() {
                return Err(());
            }
            let kind = match &t.shape {
                Shape::Mesh(m) => {
                    let bvh = Bvh::new(m);
                    let (mut mask, _) = grid::inside_mask(&bvh, g, iso, 0.5 * g.h, &NoMonitor)?;
                    // Enclosed pockets of a tool are part of it.
                    grid::clean(&mut mask, g, 1);
                    Kind::Mesh { bvh, mask }
                }
                Shape::HalfSpace { point, normal } => {
                    let l = norm(*normal);
                    Kind::Plane { point: *point, normal: normal.map(|x| x / l) }
                }
                Shape::Box { center, half, axes } => Kind::Box { center: *center, half: *half, axes: *axes },
            };
            out.push(Ready { op: t.op, kind });
        }
        Ok(out)
    }

    /// Is grid node `idx` at `p` inside the tool?
    #[inline]
    pub fn inside(&self, idx: usize, p: V3) -> bool {
        match &self.kind {
            Kind::Mesh { mask, .. } => mask[idx] != 0,
            _ => self.analytic(p) < 0.0,
        }
    }

    fn analytic(&self, p: V3) -> f64 {
        match &self.kind {
            Kind::Plane { point, normal } => -dot(sub(p, *point), *normal),
            Kind::Box { center, half, axes } => {
                let d = sub(p, *center);
                let q: V3 = std::array::from_fn(|a| dot(d, axes[a]).abs() - half[a]);
                let out = q.map(|x| x.max(0.0));
                norm(out) + q[0].max(q[1]).max(q[2]).min(0.0)
            }
            Kind::Mesh { .. } => unreachable!(),
        }
    }

    /// Signed distance (negative inside) at grid node `idx`, at `p`, clamped to `far`.
    #[inline]
    pub fn sdf(&self, idx: usize, p: V3, far: f64) -> f64 {
        match &self.kind {
            Kind::Mesh { bvh, mask } => {
                let d = bvh.dist2(p, far * far).sqrt().min(far);
                if mask[idx] != 0 {
                    -d
                } else {
                    d
                }
            }
            _ => self.analytic(p).clamp(-far, far),
        }
    }

    /// Is `p` within `r` of the tool's surface?
    #[inline]
    pub fn near(&self, p: V3, r: f64) -> bool {
        match &self.kind {
            Kind::Mesh { bvh, .. } => bvh.dist2(p, r * r) < r * r,
            _ => self.analytic(p).abs() < r,
        }
    }
}

impl Ready {
    /// `p` moved onto the tool's flat faces when it is within `eps` of one, so a cut face comes
    /// out exactly flat (the grid stores distances in single precision).
    pub fn snap(&self, p: V3, eps: f64) -> V3 {
        match &self.kind {
            Kind::Plane { point, normal } => {
                let d = dot(sub(p, *point), *normal);
                if d.abs() < eps {
                    std::array::from_fn(|k| p[k] - d * normal[k])
                } else {
                    p
                }
            }
            Kind::Box { center, half, axes } => {
                let d = sub(p, *center);
                let mut q: V3 = std::array::from_fn(|a| dot(d, axes[a]));
                let mut moved = false;
                for a in 0..3 {
                    if (q[a].abs() - half[a]).abs() < eps {
                        q[a] = half[a].copysign(q[a]);
                        moved = true;
                    }
                }
                if !moved {
                    return p;
                }
                std::array::from_fn(|k| center[k] + (0..3).map(|a| q[a] * axes[a][k]).sum::<f64>())
            }
            Kind::Mesh { .. } => p,
        }
    }
}

/// `a` combined with a tool's value `b`.
#[inline]
pub fn combine(op: Op, a: f64, b: f64) -> f64 {
    match op {
        Op::Union => a.min(b),
        Op::Subtract => a.max(-b),
        Op::Intersect => a.max(b),
    }
}

/// `a` combined with a tool's inside test `b`.
#[inline]
pub fn combine_inside(op: Op, a: bool, b: bool) -> bool {
    match op {
        Op::Union => a || b,
        Op::Subtract => a && !b,
        Op::Intersect => a && b,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{solidify_with, NoMonitor, Params};

    /// A closed axis-aligned box from `a` to `b`.
    fn cube(a: V3, b: V3) -> Mesh {
        let c = |i: usize| [if i & 1 == 0 { a[0] } else { b[0] }, if i & 2 == 0 { a[1] } else { b[1] }, if i & 4 == 0 { a[2] } else { b[2] }];
        let quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        Mesh::from_soup(quads.iter().flat_map(|q| [[c(q[0]), c(q[1]), c(q[2])], [c(q[0]), c(q[2]), c(q[3])]]))
    }

    fn run(tools: &[Tool], min_part: f64) -> crate::Report {
        let p = Params { voxel: 0.25, gpu: false, fair_tau: 0.0, min_part_mm3: min_part, ..Params::default() };
        solidify_with(&cube([0.0; 3], [10.0; 3]), tools, &p, &NoMonitor).unwrap().1
    }

    #[test]
    fn a_plane_cut_keeps_one_side_flat() {
        let r = run(&[Tool::keep_side([0.0, 0.0, 4.0], [0.0, 0.0, 1.0])], 0.0);
        assert!(r.ok, "{r:?}");
        assert!((r.size_mm[2] - 6.0).abs() < 0.05, "{:?}", r.size_mm);
        assert!((r.volume_mm3 - 600.0).abs() < 6.0, "{}", r.volume_mm3);
    }

    #[test]
    fn a_box_is_removed_and_a_mesh_added() {
        let r = run(&[Tool::aabb([5.0, 5.0, 5.0], [20.0, 20.0, 20.0], Op::Subtract)], 0.0);
        assert!(r.ok && (r.volume_mm3 - 875.0).abs() < 9.0, "{}", r.volume_mm3);
        let add = Tool { shape: Shape::Mesh(cube([8.0, 0.0, 0.0], [14.0, 2.0, 2.0])), op: Op::Union };
        let r = run(&[add], 0.0);
        assert!(r.ok && (r.size_mm[0] - 14.0).abs() < 0.05 && (r.volume_mm3 - 1016.0).abs() < 10.0, "{r:?}");
    }

    #[test]
    fn a_part_cut_free_is_dropped_or_kept() {
        // A slab through the middle leaves two halves of 400 mm³.
        let slab = Tool::aabb([-1.0, -1.0, 4.0], [11.0, 11.0, 6.0], Op::Subtract);
        let r = run(&[slab.clone()], 0.0);
        assert!(r.ok && (r.volume_mm3 - 400.0).abs() < 5.0, "{r:?}");
        // Counted in grid nodes, so a little under the half's volume.
        assert!((r.dropped_parts[0].volume_mm3 - 400.0).abs() < 30.0, "{:?}", r.dropped_parts);
        let r = run(&[slab], 100.0);
        assert!(r.watertight && r.bodies == 2 && (r.volume_mm3 - 800.0).abs() < 9.0, "{r:?}");
    }
}
