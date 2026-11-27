"""베어링 대신 끼우는 출력 부시(미끄럼 고리). 바깥지름 = 베어링 바깥지름(같은 자리에 그대로 들어감), 안지름 = 축 + 돌아갈 틈.
usage: LD_LIBRARY_PATH=sim/tmp/glu uv run --no-project --python 3.12 --with gmsh python -I print_tests/tools/bushings.py out20/bushing_stl"""
import sys, os, gmsh
out = sys.argv[1]
# 이름: (바깥지름, 안지름, 폭) mm — 6805 자리는 요 허브 Ø24.8(build20 hub r 12.4) + 한쪽 0.25, 688 축 Ø8 + 0.2, 684 축 Ø4 + 0.15
RINGS = {"bushing_6805_hip_yaw": (37.0, 25.3, 7.0), "bushing_688_neck": (16.0, 8.4, 5.0), "bushing_684_ear": (9.0, 4.3, 4.0)}
for name, (do, di, w) in RINGS.items():
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    a = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, w, do / 2); b = gmsh.model.occ.addCylinder(0, 0, -1, 0, 0, w + 2, di / 2)
    gmsh.model.occ.cut([(3, a)], [(3, b)]); gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 48); gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
    gmsh.model.mesh.generate(2); gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(os.path.join(out, name + ".stl")); gmsh.finalize(); print(name, do, di, w)
