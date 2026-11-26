"""STEP -> STL (gmsh/OpenCASCADE). 닫힌 메시 · 부피를 확인해 출력한다.
usage: LD_LIBRARY_PATH=<libGLU 폴더> uv run --no-project --python 3.12 --with gmsh --with numpy \
           python -I step2stl.py in.stp out.stl [max_edge_mm]
DGX_1 에는 시스템 libGLU.so.1 이 없어 gmsh 가 안 뜬다. blender 번들
/raid/kiro-ai/tools/blender-4.2.23-linux-x64/lib/mesa/libGLU.so.1.3.1 을 빈 폴더에 libGLU.so.1 로 복사해
그 폴더만 LD_LIBRARY_PATH 에 둔다(blender 의 다른 mesa 라이브러리는 섞지 않는다).
큐비콘 Style Neo 슬라이서는 STEP 을 못 읽어 STL 이 필요하다 (소유자 2026-11-26).
"""
import sys, numpy as np, gmsh

src, dst = sys.argv[1], sys.argv[2]
hmax = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
vols = gmsh.model.occ.importShapes(src)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)     # 원 한 바퀴 24 분할 이상
gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
gmsh.model.mesh.generate(2)
gmsh.option.setNumber("Mesh.Binary", 1)
gmsh.write(dst)
# 확인: 삼각형 모음으로 닫힘 · 부피
tags, coords, _ = gmsh.model.mesh.getNodes()
P = dict(zip(tags, coords.reshape(-1, 3)))
types, _, nodes = gmsh.model.mesh.getElements(2)
tri = np.array([n for t, n in zip(types, nodes) if t == 2][0]).reshape(-1, 3)
occ_vol = sum(gmsh.model.occ.getMass(d, t) for d, t in vols if d == 3)
gmsh.finalize()
V = np.array([[P[a], P[b], P[c]] for a, b, c in tri])
vol = abs(np.einsum("ij,ij->i", V[:, 0], np.cross(V[:, 1], V[:, 2])).sum() / 6)
from collections import Counter
e = Counter()
for a, b, c in tri:
    for u, v in ((a, b), (b, c), (c, a)):
        e[(min(u, v), max(u, v))] += 1
bad = sum(1 for k in e.values() if k != 2)
print(f"{dst}: solids {sum(1 for d,_ in vols if d==3)} tris {len(tri)} open_edges {bad} "
      f"vol_stl {vol/1000:.2f} cm3 vol_step {occ_vol/1000:.2f} cm3 diff {100*(vol-occ_vol)/occ_vol:+.2f}%")
