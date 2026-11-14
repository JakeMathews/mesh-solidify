use mesh_solidify::{check::verify, inspect, io, solidify, Mesh, NoMonitor, Params};

/// Unit sphere from a subdivided octahedron, outward-oriented.
fn sphere(levels: usize) -> Mesh {
    let mut v: Vec<[f64; 3]> = vec![
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
    ];
    let mut f: Vec<[u32; 3]> = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    for _ in 0..levels {
        let mut mid = std::collections::HashMap::new();
        let mut m = |a: u32, b: u32, v: &mut Vec<[f64; 3]>| -> u32 {
            *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
                let (p, q) = (v[a as usize], v[b as usize]);
                let c = [(p[0] + q[0]) / 2., (p[1] + q[1]) / 2., (p[2] + q[2]) / 2.];
                let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                v.push([c[0] / l, c[1] / l, c[2] / l]);
                (v.len() - 1) as u32
            })
        };
        let mut nf = Vec::new();
        for t in &f {
            let (a, b, c) = (t[0], t[1], t[2]);
            let (ab, bc, ca) = (m(a, b, &mut v), m(b, c, &mut v), m(c, a, &mut v));
            nf.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
        }
        f = nf;
    }
    Mesh { v, f }
}

#[test]
fn closed_sphere_stays_a_sphere() {
    let s = sphere(4);
    assert!(inspect(&s).watertight);
    let p = Params {
        voxel: 0.04,
        target_faces: 3000,
        ..Params::default()
    };
    let (out, r) = solidify(&s, &p, &NoMonitor).unwrap();
    assert!(r.ok && r.watertight && r.bodies == 1, "{r:?}");
    let vol = out.volume();
    let exact = 4.0 / 3.0 * std::f64::consts::PI;
    assert!((vol - exact).abs() < 0.05 * exact, "volume {vol}");
}

#[test]
fn holes_get_closed() {
    let mut s = sphere(4);
    // Cut a big hole (the cap above z = 0.6) and a few small ones.
    let tri = |m: &Mesh, i: usize| m.tri(i);
    let keep: Vec<bool> = (0..s.f.len())
        .map(|i| {
            let c = tri(&s, i).iter().fold([0.0; 3], |a, p| {
                [a[0] + p[0] / 3., a[1] + p[1] / 3., a[2] + p[2] / 3.]
            });
            c[2] < 0.6 && !(c[0] > 0.9 && c[1].abs() < 0.1)
        })
        .collect();
    s.f =
        s.f.iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(t, _)| *t)
            .collect();
    let before = inspect(&s);
    assert!(!before.watertight && before.holes >= 2, "{before:?}");
    let p = Params {
        voxel: 0.04,
        target_faces: 3000,
        ..Params::default()
    };
    let (out, r) = solidify(&s, &p, &NoMonitor).unwrap();
    assert!(r.ok, "{r:?}");
    let v = verify(&out);
    assert!(v.watertight && v.is_volume && v.bodies == 1);
    assert!(out.f.len() <= 3000);
    // The cap is filled, so the volume is near the full sphere's, not a hollow shell's.
    let exact = 4.0 / 3.0 * std::f64::consts::PI;
    assert!(out.volume() > 0.8 * exact, "volume {}", out.volume());
}

#[test]
fn loose_parts_are_dropped_and_scale_applies() {
    let mut s = sphere(3);
    let mut small = sphere(2);
    small.scale(0.2);
    small.translate([3.0, 0.0, 0.0]);
    let off = s.v.len() as u32;
    s.v.extend(small.v);
    s.f.extend(small.f.iter().map(|t| [t[0] + off, t[1] + off, t[2] + off]));
    let p = Params {
        scale: 10.0,
        voxel: 0.4,
        target_faces: 2000,
        ..Params::default()
    };
    let (out, r) = solidify(&s, &p, &NoMonitor).unwrap();
    assert!(
        r.ok && r.bodies == 1 && r.components_dropped_voxels > 0,
        "{r:?}"
    );
    let size = out.size();
    assert!((size[0] - 20.0).abs() < 1.0, "{size:?}");
}

#[test]
fn grid_limit_is_reported_with_a_working_suggestion() {
    let s = sphere(2);
    let p = Params {
        voxel: 0.001,
        max_voxels: 1_000_000,
        ..Params::default()
    };
    match solidify(&s, &p, &NoMonitor) {
        Err(mesh_solidify::Error::GridTooLarge { suggest, .. }) => {
            let p2 = Params {
                voxel: suggest,
                max_voxels: 1_000_000,
                target_faces: 1000,
                ..Params::default()
            };
            assert!(
                solidify(&s, &p2, &NoMonitor).is_ok(),
                "suggested voxel {suggest} must fit"
            );
        }
        other => panic!("expected GridTooLarge, got {:?}", other.map(|x| x.1)),
    }
}

#[test]
fn file_formats_round_trip() {
    let s = sphere(2);
    let stl = io::write_stl(&s);
    let back = io::read("x.stl", &stl).unwrap();
    assert_eq!(back.f.len(), s.f.len());
    assert_eq!(back.v.len(), s.v.len());
    let obj = io::write_obj(&s);
    let back = io::read("x.obj", &obj).unwrap();
    assert_eq!(back.f.len(), s.f.len());
    let ply = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n3 0 2 3\n";
    let back = io::read("x.ply", ply.as_bytes()).unwrap();
    assert_eq!(back.f.len(), 3);
}

/// A 2 x 2 x 1 box (z from 0 to 1) without its top face.
fn open_box() -> Mesh {
    let (w, d, hgt) = (2.0, 2.0, 1.0);
    let n = 20;
    let mut tris: Vec<[[f64; 3]; 3]> = Vec::new();
    // Tile a rectangle given by an origin and two edge vectors, outward by right-hand rule.
    let mut quad = |o: [f64; 3], u: [f64; 3], v: [f64; 3]| {
        for i in 0..n {
            for j in 0..n {
                let p = |a: usize, b: usize| {
                    let (s, t) = (a as f64 / n as f64, b as f64 / n as f64);
                    [
                        o[0] + s * u[0] + t * v[0],
                        o[1] + s * u[1] + t * v[1],
                        o[2] + s * u[2] + t * v[2],
                    ]
                };
                tris.push([p(i, j), p(i + 1, j), p(i + 1, j + 1)]);
                tris.push([p(i, j), p(i + 1, j + 1), p(i, j + 1)]);
            }
        }
    };
    quad([0., 0., 0.], [0., d, 0.], [w, 0., 0.]); // bottom, facing -z
    quad([0., 0., 0.], [w, 0., 0.], [0., 0., hgt]); // y = 0
    quad([w, d, 0.], [-w, 0., 0.], [0., 0., hgt]); // y = d
    quad([0., d, 0.], [0., -d, 0.], [0., 0., hgt]); // x = 0
    quad([w, 0., 0.], [0., d, 0.], [0., 0., hgt]); // x = w
    Mesh::from_soup(tris)
}

/// How far the top patch strays from the plane z = 1 (the missing face), worst case, among
/// vertices more than `lo` from the walls.
fn top_bulge(m: &Mesh, lo: f64) -> f64 {
    m.v.iter()
        .filter(|p| p[2] > 0.5 && p[0] > lo && p[0] < 2.0 - lo && p[1] > lo && p[1] < 2.0 - lo)
        .map(|p| (p[2] - 1.0).abs())
        .fold(0.0, f64::max)
}

#[test]
fn flat_fill_keeps_a_missing_face_flat() {
    let b = open_box();
    assert!(!inspect(&b).watertight);
    let p = Params {
        voxel: 0.04,
        target_faces: 4000,
        ..Params::default()
    };
    let (flat, r) = solidify(&b, &p, &NoMonitor).unwrap();
    assert!(r.ok, "{r:?}");
    let f = top_bulge(&flat, 0.1);
    // Within two voxels of the missing face (the patch sits where the grid's inside/outside
    // flips, up to a voxel off).
    assert!(f < 0.08, "flat fill strays {f} from the missing face");
    let v = flat.volume();
    assert!((v - 4.0).abs() < 0.2, "volume {v}");
}

#[test]
fn progress_runs_from_start_to_end_in_order() {
    use std::sync::Mutex;
    struct Rec(Mutex<Vec<f32>>);
    impl mesh_solidify::Monitor for Rec {
        fn progress(&self, f: f32, _stage: &str) {
            self.0.lock().unwrap().push(f);
        }
    }
    let rec = Rec(Mutex::new(Vec::new()));
    let p = Params {
        voxel: 0.04,
        target_faces: 1000,
        ..Params::default()
    };
    solidify(&open_box(), &p, &rec).unwrap();
    let v = rec.0.into_inner().unwrap();
    assert!(v.len() > 10, "only {} progress reports", v.len());
    assert!(v.iter().all(|f| (0.0..=1.0).contains(f)));
    // Stages run one after another; the winding and distance passes report from worker
    // threads, so only the stages after them are checked for order.
    let after_winding: Vec<f32> = v.iter().copied().filter(|&f| f >= 0.33).collect();
    assert!(
        after_winding.windows(2).all(|w| w[1] >= w[0] - 1e-6),
        "{after_winding:?}"
    );
    assert!(*v.last().unwrap() > 0.99);
}
