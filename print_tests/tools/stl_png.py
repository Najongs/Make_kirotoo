"""STL 한 개를 네 방향에서 그린 PNG (MuJoCo 화면 밖 렌더, EGL). 출력 방향 그대로(z 위).
usage: CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 sim/.venv/bin/python print_tests/tools/stl_png.py <in.stl> <out.png> "<제목>" """
import sys, os, struct, numpy as np, mujoco, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_F = os.path.expanduser("~/.local/share/fonts/NotoSansMonoCJKkr.otf")
if os.path.exists(_F):
    fm.fontManager.addfont(_F); plt.rcParams["font.family"] = fm.FontProperties(fname=_F).get_name()
src, dst, title = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
b = open(src, "rb").read(); n = struct.unpack("<I", b[80:84])[0]
V = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"].reshape(-1, 3)
lo, hi = V.min(0), V.max(0); c = (lo + hi) / 2 / 1000; r = np.linalg.norm(hi - lo) / 2 / 1000
xml = f"""<mujoco><compiler meshdir="{os.path.dirname(src)}"/><asset><mesh name="p" file="{os.path.basename(src)}" scale="0.001 0.001 0.001"/></asset>
<visual><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/><global offwidth="900" offheight="900"/><quality shadowsize="0"/></visual>
<worldbody><light pos="0.3 -0.3 0.6" dir="-0.4 0.4 -0.8" diffuse="0.5 0.5 0.5"/>
<geom type="mesh" mesh="p" rgba="0.35 0.55 0.85 1"/></worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
R = mujoco.Renderer(m, 900, 900)
views = [(-25, 225, "앞 위"), (-25, 45, "뒤 위"), (0, 270, "옆 (−y 에서)"), (35, 135, "아래에서")]
fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=100)
for ax, (el, az, nm) in zip(axs.flat, views):
    cam = mujoco.MjvCamera(); cam.lookat[:] = c; cam.distance = r * 3.0; cam.elevation = el; cam.azimuth = az
    R.update_scene(d, camera=cam); ax.imshow(R.render()); ax.set_title(nm, fontsize=14); ax.axis("off")
fig.suptitle(f"{title}\n크기 {hi[0]-lo[0]:.0f}×{hi[1]-lo[1]:.0f}×{hi[2]-lo[2]:.0f} mm · 출력 방향 그대로(바닥이 판)", fontsize=16)
plt.tight_layout(); fig.savefig(dst); print(dst)
