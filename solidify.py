#!/usr/bin/env python3
"""Turn a broken mesh (holes, open shells, self-intersections) into one watertight,
printable solid.

Pipeline:
  1. load + scale to print size (so all tolerances are in print mm)
  2. generalized winding number on a voxel grid -> inside/outside field
     (robust to holes and overlapping shells; no hole patching)
  3. keep the largest connected solid, drop specks
  4. narrow-band signed distance (sign from winding number) for a smooth surface
  5. marching cubes -> repair
  6. biharmonic fairing of hole-patch vertices (far from any input triangle) -> no terraces
  7. quadric decimation -> verify watertight/manifold

Usage:
  python solidify.py in.stl out.3mf --scale 0.1 --voxel 0.125 --min-feature 0
Deps: pip install trimesh numpy scipy scikit-image libigl pymeshlab manifold3d
"""
import argparse, time
import numpy as np, trimesh, igl, pymeshlab as ml, manifold3d as mf
from scipy import ndimage
from skimage.measure import marching_cubes

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--voxel", type=float, default=0.125, help="grid size in print mm")
ap.add_argument("--min-feature", type=float, default=0.0, help="optional outward offset in print mm (0=off)")
ap.add_argument("--iso", type=float, default=0.5, help="winding-number threshold")
ap.add_argument("--target-faces", type=int, default=400_000)
ap.add_argument("--fair-tau", type=float, default=0.5, help="vertices farther than this many voxels from the input are hole patch (0=no fairing)")
ap.add_argument("--fair-grow", type=int, default=2, help="extra rings around each patch to refair")
a = ap.parse_args()
T = time.time()
log = lambda *s: print(f"[{time.time()-T:6.1f}s]", *s, flush=True)

m = trimesh.load(a.src, force="mesh")
m.apply_scale(a.scale)
V = np.ascontiguousarray(m.vertices, float); F = np.ascontiguousarray(m.faces, np.int64)
h, pad = a.voxel, 4
lo = V.min(0) - pad * h
dims = np.ceil((V.max(0) + pad * h - lo) / h).astype(int) + 1
log(f"input {len(F)} tris, watertight={m.is_watertight}; grid {dims.tolist()} = {dims.prod()/1e6:.0f}M")

# 2. winding number field, slab by slab
xs, ys, zs = [lo[i] + h * np.arange(dims[i]) for i in range(3)]
X, Y = np.meshgrid(xs, ys, indexing="ij"); XY = np.c_[X.ravel(), Y.ravel()]
W = np.empty(dims, np.float32); step = max(1, int(4e6 // len(XY)))
for k in range(0, dims[2], step):
    zz = zs[k:k + step]
    Q = np.vstack([np.c_[XY, np.full(len(XY), z)] for z in zz])
    W[:, :, k:k + len(zz)] = igl.fast_winding_number(V, F, Q).reshape(len(zz), dims[0], dims[1]).transpose(1, 2, 0)
log("winding number done")

# 3. largest component only
B = W > a.iso
lab, n = ndimage.label(B)
sz = np.bincount(lab.ravel()); sz[0] = 0
B = lab == sz.argmax(); del lab
log(f"{n} components, kept largest ({sz.max()} vox), dropped {sz.sum()-sz.max()} vox")
del W

# 4. signed distance in a narrow band: sign from the winding number (robust to holes),
#    magnitude = exact distance to the original triangles (smooth, sub-voxel accurate).
#    A pure 0/1 winding field makes marching cubes produce voxel terraces.
band = ndimage.binary_dilation(B, iterations=2) & ~ndimage.binary_erosion(B, iterations=2)
idx = np.nonzero(band)
P = lo + h * np.c_[idx]
d = np.sqrt(igl.point_mesh_squared_distance(P, V, F)[0])
d = np.maximum(d, 1e-4 * h)                        # keep the sign at points that sit on the surface
field = np.where(B, 3 * h, -3 * h).astype(np.float32)  # +inside / -outside
field[idx] = np.where(B[idx], d, -d)
if a.min_feature > 0:                              # optional uniform offset (outward), no blobs
    field += a.min_feature
log(f"distance band: {len(d)/1e6:.1f}M voxels")
del band, idx, P, d, B

# 5. surface, repair, decimate, verify
v, f, _, _ = marching_cubes(field, 0.0, spacing=(h, h, h)); v += lo; del field
ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(v, f[:, ::-1]))
ms.meshing_remove_duplicate_vertices(); ms.meshing_repair_non_manifold_edges(); ms.meshing_repair_non_manifold_vertices()
ms.meshing_remove_connected_component_by_face_number(mincomponentsize=1000)
log(f"marching cubes: {ms.current_mesh().face_number()} tris")

# 6. fair hole patches: vertices far from every original triangle were invented to close a hole
#    and carry voxel terraces. Re-solve them as a biharmonic (minimum-curvature-variation) surface
#    with everything else held fixed, so they continue the curvature of the real surface around them.
if a.fair_tau > 0:
    cm = ms.current_mesh(); pv = cm.vertex_matrix().copy(); pf = cm.face_matrix().astype(np.int64)
    dist = np.sqrt(igl.point_mesh_squared_distance(pv, V, F)[0])
    free = dist > a.fair_tau * h
    # grow by a couple of rings so the step at the patch rim is also refaired
    import scipy.sparse as sp, scipy.sparse.linalg as spl
    n = len(pv); e = np.r_[pf[:, [0, 1]], pf[:, [1, 2]], pf[:, [2, 0]]]
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(float)
    for _ in range(a.fair_grow):
        free = free | (A @ free.astype(float) > 0)
    nf = int(free.sum())
    if nf:
        deg = np.asarray(A.sum(1)).ravel()
        L = sp.eye(n) - sp.diags(1 / deg) @ A          # uniform Laplacian
        Mb = (L.T @ L).tocsr()                          # bi-Laplacian (SPD restricted to free)
        fi, ci = np.nonzero(free)[0], np.nonzero(~free)[0]
        Mff = Mb[fi][:, fi].tocsc(); rhs = -(Mb[fi][:, ci] @ pv[ci])
        solve = spl.factorized(Mff)
        pv[fi] = np.column_stack([solve(rhs[:, k]) for k in range(3)])
        ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(pv, pf.astype(np.int32)))
    log(f"faired {nf} hole-patch vertices ({100*nf/n:.1f}%)")

if ms.current_mesh().face_number() > a.target_faces:
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=a.target_faces, preservetopology=True,
                                                preserveboundary=True, qualitythr=0.3, planarquadric=True)
ms.meshing_remove_duplicate_vertices(); ms.meshing_repair_non_manifold_edges()
cm = ms.current_mesh()
r = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
trimesh.repair.fix_normals(r)
M = mf.Manifold(mf.Mesh(vert_properties=r.vertices.astype(np.float32), tri_verts=r.faces.astype(np.uint32)))
r.export(a.dst)
log(f"out {len(r.faces)} tris, watertight={r.is_watertight}, volume={r.is_volume}, bodies={len(r.split())}, "
    f"manifold3d={M.status().name}, size={np.round(r.extents,1).tolist()} mm -> {a.dst}")
