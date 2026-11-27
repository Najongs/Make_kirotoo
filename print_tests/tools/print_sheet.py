"""PRINT_v2 폴더마다: 부품 모아 보기 PNG(앞 위 한 방향) + 메시 점검표 CHECK.md
usage: CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 sim/.venv/bin/python print_tests/tools/print_sheet.py out20/PRINT_v2"""
import sys, os, glob, struct, numpy as np, mujoco, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_F = os.path.expanduser("~/.local/share/fonts/NotoSansMonoCJKkr.otf")
fm.fontManager.addfont(_F); plt.rcParams["font.family"] = fm.FontProperties(fname=_F).get_name()
root = sys.argv[1]
def load(p):
    b = open(p, "rb").read(); n = struct.unpack("<I", b[80:84])[0]
    return np.frombuffer(b[84:84+n*50], dtype=np.dtype([("n","<f4",3),("v","<f4",(3,3)),("a","<u2")]))["v"].astype(np.float64)
def check(T):
    V = T.reshape(-1,3); u, inv = np.unique(np.round(V,4), axis=0, return_inverse=True); F = inv.reshape(-1,3)
    E = np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),1); _, c = np.unique(E, axis=0, return_counts=True)
    vol = np.einsum("ij,ij->i", T[:,0], np.cross(T[:,1], T[:,2])).sum()/6/1000
    return len(T), int((c==1).sum()), vol
rows_all = []
for d in sorted(glob.glob(os.path.join(root, "*/"))):
    fs = sorted(glob.glob(d+"*.stl")); name = os.path.basename(d.rstrip("/"))
    k = len(fs); cols = 4 if k <= 16 else 6; rws = -(-k//cols)
    fig, axs = plt.subplots(rws, cols, figsize=(cols*3.2, rws*3.9), dpi=110)
    rows = []
    for ax, f in zip(axs.flat, fs):
        T = load(f); lo, hi = T.reshape(-1,3).min(0), T.reshape(-1,3).max(0)
        nt, op, vol = check(T); sz = hi-lo; rows.append((os.path.basename(f)[:-4], sz, nt, op, vol))
        xml = f'<mujoco><compiler meshdir="{d}"/><asset><mesh name="p" file="{os.path.basename(f)}" scale="0.001 0.001 0.001"/></asset><visual><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/><global offwidth="500" offheight="500"/></visual><worldbody><light pos="0.3 -0.3 0.6" dir="-0.4 0.4 -0.8"/><geom type="mesh" mesh="p" rgba="0.35 0.55 0.85 1"/></worldbody></mujoco>'
        m = mujoco.MjModel.from_xml_string(xml); dd = mujoco.MjData(m); mujoco.mj_forward(m, dd)
        R = mujoco.Renderer(m, 500, 500); cam = mujoco.MjvCamera(); cam.lookat[:] = (lo+hi)/2000
        cam.distance = np.linalg.norm(sz)/2000*3.0; cam.elevation = -25; cam.azimuth = 225
        R.update_scene(dd, camera=cam); ax.imshow(R.render()); R.close()
        ax.set_title(f"{os.path.basename(f)[:-4]}\n{sz[0]:.0f}×{sz[1]:.0f}×{sz[2]:.0f} mm", fontsize=9)
    for ax in axs.flat: ax.axis("off")
    fig.suptitle(f"{name} — 부품 {k}개, 출력 방향 그대로(바닥이 판), 크기는 각자 다름", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.5); out = os.path.join(d, f"_모아보기_{name}.png"); fig.savefig(out); plt.close(fig); print(out)
    rows_all.append((name, rows))
L = ["# PRINT_v2 메시 점검 (2026-11-28, 출력 묶음 STL 52개)", "",
     "- 기준: 열린 모서리 0 = 닫힌 메시. 부피는 STL 에서 잰 값(cm³). 크기는 출력 방향 그대로 x×y×z mm, 판 220×220×220 안이어야 함.",
     "- 만든 스크립트: `print_tests/tools/print_sheet.py out20/PRINT_v2`(꼭짓점 0.0001 mm 반올림으로 모서리 짝 셈).", ""]
for name, rows in rows_all:
    L += [f"## {name}", "", "| 부품 | 크기 (mm) | 삼각형 | 열린 모서리 | 부피 (cm³) | 판 220 안 |", "|---|---|---|---|---|---|"]
    for n, sz, nt, op, vol in rows:
        L.append(f"| {n} | {sz[0]:.0f}×{sz[1]:.0f}×{sz[2]:.0f} | {nt} | {op} | {vol:.2f} | {'예' if (sz<=220).all() else '아니오'} |")
    L.append("")
open(os.path.join(root, "CHECK.md"), "w").write("\n".join(L)); print("CHECK.md")
