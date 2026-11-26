"""v21 elbow arm figures (render6 toon renderer, same as render20) -> out21_arm/.

usage (code_v17/):  ../sim/.venv/bin/python render21_arm.py
  arm21_range.png  : right side view, straight / full forward fold / full backward fold / walking fold
  arm21_pushup.png : getup push-up pose (sim/getup key 2 legs + elbow), hands under the shoulders on a common floor
Elbow limits are read from out21_arm/sweep_arm21.json (run arm21.py sweep first).
"""
import os, sys, json, math
import numpy as np
os.environ["BUILD"] = "build21_arm"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render6 as r
import build21_arm as BM
import head21 as H21

OUT = os.path.abspath(BM.OUT)
r.CHAINS.pop("neck", None); r.CHAINS["head"] = ["head_nod"]
r.CHAINS.update({"R_arm": ["R_arm_swing"], "L_arm": ["L_arm_swing"], "R_forearm": ["R_arm_swing", "R_elbow"],
                 "L_forearm": ["L_arm_swing", "L_elbow"], "nod_crank": ["head_nod", "nod_crank"], "nod_rod": ["nod_rod"],
                 "ears": ["head_nod", "ear_axle"], "ear_rod": ["head_nod", "nod_crank", "ear_rod"]})
for S_ in "RL":
    nm = [f"{S_}_hip_yaw", f"{S_}_hip_roll", f"{S_}_hip_pitch", f"{S_}_knee", f"{S_}_ankle_pitch", f"{S_}_ankle_roll"]
    for i, ln in enumerate([f"{S_}_hip_yaw_link", f"{S_}_hip_roll_link", f"{S_}_thigh", f"{S_}_shank", f"{S_}_ankle_link", f"{S_}_foot"]):
        r.CHAINS[ln] = nm[:i + 1]
# whole-body pitch (push-up figure): a virtual joint in front of every chain
r.J["base"] = {"name": "base", "axis": "y", "point_mm": (0.0, 0.0, 0.0)}
for ln in list(r.CHAINS):
    r.CHAINS[ln] = ["base"] + r.CHAINS[ln]
missing = set(r.links) - set(r.CHAINS)
assert not missing, missing


def nod_pose(beta):
    al = H21.nod(beta)["alpha"]; ps = H21.ear(al)["psi"]
    pw = H21.rot2(H21.rot2(H21.S + np.array([H21.NOD_R, 0.0]), al, H21.S), beta, H21.P)
    A_ = np.array(H21.NOD_A); pin0 = H21.S + np.array([H21.NOD_R, 0.0])
    d = -math.degrees(math.atan2(*(pw - A_)[::-1]) - math.atan2(*(pin0 - A_)[::-1]))
    P1, P2, LE = H21.ear_geom()
    q1 = H21.rot2(P1, al, H21.S); q2 = H21.rot2(P2, ps, H21.E0)
    ang = lambda u: math.degrees(math.atan2(u[1], u[0]))
    rod = -(ang(q2 - q1) - ang(P2 - P1))
    return {"head_nod": beta, "nod_crank": al, "nod_rod": d, "ear_axle": ps, "ear_rod": rod - al}


def M(v):
    """minus sign glyph for figure text"""
    return str(v).replace("-", "\u2212")


def arms(a, e):
    return {"R_arm_swing": a, "L_arm_swing": a, "R_elbow": e, "L_elbow": e}


def font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ("/home/kiro-ai/.local/share/fonts/NotoSansMonoCJKkr.otf", "/home/kiro-ai/.local/share/fonts/NotoSansKR.ttf"):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    return plt


def compose(imgs, captions, title, path, sub=None):
    plt = font()
    H = max(i.shape[0] for i in imgs)
    ws = [i.shape[1] for i in imgs]
    fig = plt.figure(figsize=(sum(ws) / 110 + 0.4 * len(imgs), H / 110 + 1.6), dpi=110)
    fig.patch.set_facecolor("white")
    x = 0.01
    tot = sum(ws) * 1.04
    for im, cap, w in zip(imgs, captions, ws):
        ax = fig.add_axes([x, 0.10, w / tot, 0.74 * im.shape[0] / H])
        ax.imshow(im); ax.axis("off")
        ax.set_title(cap, fontsize=11, color="#191919", y=-0.12 if False else 1.0, pad=6)
        x += w / tot + 0.01
    fig.suptitle(title, fontsize=13.5, color="#191919", x=0.01, ha="left", y=0.985)
    if sub:
        import textwrap
        fig.text(0.01, 0.015, textwrap.fill(sub, max(60, int(fig.get_figwidth() * 110 / 10.5))), fontsize=9, color="#555555",
                 ha="left", va="bottom", linespacing=1.4)
    fig.savefig(path, dpi=110, facecolor="white"); plt.close(fig)


def link_pts(ln, ang):
    M = r.link_pose(ln, ang)
    P = np.vstack([T.reshape(-1, 3) for l, pn, pid, T, rgb, caps in r.GEOM if l == ln and len(T)])
    return P @ M[:3, :3].T + M[:3, 3]


if __name__ == "__main__":
    sw = json.load(open(os.path.join(OUT, "sweep_arm21.json")))
    lo, hi = sw["elbow_self"]["R"]["free"]
    jl = json.load(open(os.path.join(OUT, "arm21_joints_mass.json")))
    lo_t = next(j for j in jl["joints"] if j["name"] == "R_elbow")
    # ---- range figure: right side, head at 0
    head0 = nod_pose(0)
    poses = [("펴짐\n어깨 0° · 팔꿈치 0°", arms(0, 0)),
             (f"앞으로 최대 접힘\n어깨 0° · 팔꿈치 {M(lo)}°", arms(0, lo)),
             (f"뒤로 최대 접힘\n어깨 0° · 팔꿈치 +{hi}°", arms(0, hi)),
             ("과제 걷기 접힘\n어깨 −15° · 팔꿈치 −90°", arms(-15, -90)),
             ("추천 걷기 접힘\n어깨 −25° · 팔꿈치 −90°", arms(-25, -90))]
    imgs = []
    for cap, p in poses:
        p = dict(p); p.update(head0)
        imgs.append(r.render(0, -90, p, scale=1.45))
    wf = [f for f in sw["fold"] if f["arm"] == "R" and f["pose"] == "walk"][0]["by_roll"]
    wr = [f for f in sw["fold"] if f["arm"] == "R" and f["pose"] == "walk_s-25"][0]["by_roll"]
    title = (f"팔꿈치는 형상으로 앞 {M(lo)}° · 뒤 +{hi}° 까지 접히고(목표 ±130°), 걷기 접힘은 어깨 −25° 에서 다리와 "
             f"{wr['12']['min_mm']:.0f} mm 떨어집니다(과제 자세 −15° 는 {wf['12']['min_mm']:.1f} mm, 고관절 롤 바깥 12°)")
    sub = ("오른쪽에서 본 모습(앞이 오른쪽), 머리 끄덕임 0°, 다리 0°. 팔꿈치 각은 설계 형상 충돌 점검(표면 점 0.6 mm, 1° 간격, arm21.py) 값이고, "
           "틈은 오른 다리를 고관절 롤 바깥으로 12° 벌리고 고관절 피치 −40 · 0 · +20° 중 가장 가까운 형상 최단 거리입니다. 출처: out21_arm/sweep_arm21.json")
    compose(imgs, [c for c, _ in poses], title, os.path.join(OUT, "arm21_range.png"), sub)
    print("range figure"); sys.stdout.flush()
    # ---- push-up figure: pose solved in arm21.py (hands under the shoulder axis, hands / toes on one floor)
    pu = sw["pushup"]
    phi, a, E = pu["body_pitch_deg"], pu["shoulder_deg"], pu["elbow_deg"]
    legs = {f"{S}_{k}": v for S in "RL" for k, v in pu["legs"].items()}
    ang = dict(legs); ang.update(arms(a, E)); ang["base"] = phi; ang.update(nod_pose(0))
    hp, fz = np.array(pu["hand_low"]), pu["floor_z"]
    # floor slab under the robot
    xs = [p[0] for p in (hp,)] + [link_pts(f"R_foot", ang)[:, 0].min(), link_pts(f"R_foot", ang)[:, 0].max()]
    x0, x1 = min(xs) - 60, max(xs) + 60
    T = np.array([[[x0, -140, fz - 0.6], [x1, -140, fz - 0.6], [x1, 140, fz - 0.6]], [[x0, -140, fz - 0.6], [x1, 140, fz - 0.6], [x0, 140, fz - 0.6]],
                  [[x0, -140, fz - 4.0], [x1, -140, fz - 4.0], [x1, -140, fz - 0.6]], [[x0, -140, fz - 4.0], [x1, -140, fz - 0.6], [x0, -140, fz - 0.6]]])
    r.GEOM.append(("floor", "floor", 99999, T, np.array([0.86, 0.86, 0.86]), []))
    r.CHAINS["floor"] = []
    imgs = [r.render(0, -90, ang, scale=1.25), r.render(25, -50, ang, scale=1.0)]
    ck = pu["check"]["R"]
    title = (f"팔굽혀펴기 자세(몸 {phi:.0f}° 숙임 · 어깨 {M(round(a))}° · 팔꿈치 {M(round(E))}°)에서 손이 어깨 바로 밑 바닥에 닿고 팔은 몸 어디에도 닿지 않습니다"
             if not ck["hits"] else f"팔굽혀펴기 자세(몸 {phi:.0f}° 숙임 · 어깨 {M(round(a))}° · 팔꿈치 {M(round(E))}°)에서 팔이 몸에 {len(ck['hits'])} 곳 닿습니다")
    sub = (f"다리는 일어나기 성공 동작 키 2(고관절 피치 −72° · 무릎 1° · 발목 피치 −25°, sim/getup 팔꿈치 100+100 기준), 팔꿈치 −58° 그대로. "
           f"110+110 팔에서 손 가장 낮은 점이 어깨 축 바로 밑, 발끝과 같은 바닥에 오도록 몸 기울기 · 어깨 각을 풀었습니다(계산). "
           f"가장 가까운 거리: 팔 ↔ 머리 {ck['d_head']} mm · 몸통 {ck['d_torso']} mm · 다리 {ck['d_legs']} mm (머리 끄덕임 −20 · 0 · +28° 중 최소)")
    compose(imgs, ["오른쪽에서", "앞 비스듬히"], title, os.path.join(OUT, "arm21_pushup.png"), sub)
    print("ok")
