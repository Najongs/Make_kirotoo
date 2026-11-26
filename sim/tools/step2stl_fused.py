"""print_tests/tools/step2stl.py with one change: the solids of a part are fused (OCC boolean union) before meshing,
so parts modelled as several touching / overlapping solids give one closed shell. Same meshing options and checks.
usage: LD_LIBRARY_PATH=<libGLU folder> uv run --no-project --python 3.12 --with gmsh --with numpy \
           python -I step2stl_fused.py in.stp out.stl [max_edge_mm]"""
import sys, numpy as np, gmsh
from collections import Counter

src, dst = sys.argv[1], sys.argv[2]
hmax = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
vols = gmsh.model.occ.importShapes(src)
gmsh.model.occ.synchronize()
v3 = [(d, t) for d, t in vols if d == 3]
n_in = len(v3)
sum_vol = sum(gmsh.model.occ.getMass(d, t) for d, t in v3)
if len(v3) > 1:
    out, _ = gmsh.model.occ.fuse([v3[0]], v3[1:])
    gmsh.model.occ.synchronize()
    v3 = [(d, t) for d, t in out if d == 3]
occ_vol = sum(gmsh.model.occ.getMass(d, t) for d, t in v3)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
gmsh.model.mesh.generate(2)
gmsh.option.setNumber("Mesh.Binary", 1)
gmsh.write(dst)
tags, coords, _ = gmsh.model.mesh.getNodes()
P = dict(zip(tags, coords.reshape(-1, 3)))
types, _, nodes = gmsh.model.mesh.getElements(2)
tri = np.array([n for t, n in zip(types, nodes) if t == 2][0]).reshape(-1, 3)
gmsh.finalize()
V = np.array([[P[a], P[b], P[c]] for a, b, c in tri])
vol = abs(np.einsum("ij,ij->i", V[:, 0], np.cross(V[:, 1], V[:, 2])).sum() / 6)
e = Counter()
for a, b, c in tri:
    for u, v in ((a, b), (b, c), (c, a)):
        e[(min(u, v), max(u, v))] += 1
bad = sum(1 for k in e.values() if k != 2)
print(f"{dst}: solids_in {n_in} solids_out {len(v3)} tris {len(tri)} open_edges {bad} vol_stl {vol/1000:.3f} cm3 "
      f"vol_fused {occ_vol/1000:.3f} cm3 vol_sum_in {sum_vol/1000:.3f} cm3 diff {100*(vol-occ_vol)/occ_vol:+.2f}%")
