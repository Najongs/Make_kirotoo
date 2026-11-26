"""Pi Camera v2 시점 렌더: 렌즈가 왼쪽 눈 뒤(설계 build18 camera_pi_v2 렌즈 블록 x 65-70, y 33.2-41.8, z 448.8-457.2)에 있다.
렌즈 앞면 중심 (70, 37.5, 453) mm, 정면(+x)을 본다. 세로 화각 48.8° (Pi Camera v2 사양), 640x480.
머리 끄덕임 -10 / 0 / +20° 에서 로봇 눈 영상을 찍어, 얼굴판 · 눈이 시야를 가리는지 본다.
usage: cd sim && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<gpu> uv run python tests/camera_view.py [model.xml]
"""
import sys, os, json, numpy as np, mujoco
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
xml = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "model", "kitoru_v18_fixed.xml")
OUT = os.path.join(HERE, "..", "results_camera")
spec = mujoco.MjSpec.from_file(xml)
head = spec.body("head")
LENS = np.array([70.0, 37.5, 453.0]) / 1000 - np.array([0, 0, 0.3485])
for nm, dx in (("cam_lens", 0.0), ("cam_front", 0.0015)):      # 렌즈 면 / 눈판 바로 앞(1.5 mm 앞)
    c = head.add_camera(); c.name = nm; c.pos = LENS + [dx + (0.0012 if nm == "cam_front" else 0), 0, 0]
    c.fovy = 48.8; c.quat = [0.5, 0.5, -0.5, -0.5]            # 카메라 -z 가 +x(정면), 영상 위쪽이 +z
# 앞에 표지물: 바닥 위 상자 셋, 벽 위 체커 판
wb = spec.worldbody
for i, (x, y, rgba) in enumerate(((0.6, 0.0, [0.9, 0.2, 0.2, 1]), (0.9, 0.25, [0.2, 0.7, 0.2, 1]), (1.2, -0.3, [0.2, 0.3, 0.9, 1]))):
    g = wb.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.05, 0.05, 0.05]; g.pos = [x, y, 0.05]; g.rgba = rgba
    g.contype = 0; g.conaffinity = 0
g = wb.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.01, 0.8, 0.4]; g.pos = [1.6, 0, 0.4]; g.rgba = [0.95, 0.9, 0.6, 1]
g.contype = 0; g.conaffinity = 0
spec.visual.map.znear = 0.0005
m = spec.compile(); d = mujoco.MjData(m)
r = mujoco.Renderer(m, 480, 640)
crank = m.actuator("nod_crank").id if "nod_crank" in [m.actuator(i).name for i in range(m.nu)] else None
fit = json.load(open(os.path.join(HERE, "..", "model", "closed_loop_fit_kitoru_v18.json"))) if os.path.exists(
    os.path.join(HERE, "..", "model", "closed_loop_fit_kitoru_v18.json")) else None
res = {}
tiles = []
for beta in (-10, 0, 20):
    mujoco.mj_resetData(m, d)
    # 머리 각 beta 에 맞는 크랭크 목표: 1차 근사(dβ/dα≈0.94) 후 시뮬로 맞춤
    tgt = np.radians(beta) / 0.94
    for k in range(2000):
        if crank is not None:
            d.ctrl[crank] = tgt
        mujoco.mj_step(m, d)
        if k % 400 == 399:
            err = np.radians(beta) - d.joint("head_nod").qpos[0]; tgt += err / 0.94
    got = float(np.degrees(d.joint("head_nod").qpos[0]))
    row = []
    for cam in ("cam_lens", "cam_front"):
        r.update_scene(d, camera=cam); img = r.render()
        # 가려짐 추정: 화면이 거의 한 색(얼굴판 · 눈 안쪽)인 비율
        gray = img.mean(2); flat = float((np.abs(gray - np.median(gray)) < 4).mean())
        res[f"{cam}_beta{beta}"] = {"head_deg": round(got, 2), "uniform_fraction": round(flat, 3)}
        im = Image.fromarray(img); ImageDraw.Draw(im).text((8, 8), f"{cam}  head {got:+.1f} deg", fill=(255, 255, 0))
        row.append(im)
    tiles.append(row)
W, H = 640, 480
sheet = Image.new("RGB", (2 * W, 3 * H), "black")
for i, row in enumerate(tiles):
    for j, im in enumerate(row):
        sheet.paste(im, (j * W, i * H))
sheet.save(os.path.join(OUT, "camera_view_v18.png"))
json.dump(res, open(os.path.join(OUT, "camera_view_v18.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
