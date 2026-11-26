"""새 배터리(60 × 40 × 70 mm) 자리 찾기 — v19 설계 형상(code_v17/build19.py, 읽기만).

방법: 몸통 bbox 를 2 mm 격자로 나눠 「몸통 껍데기(앞 · 뒤 덮개 · 뚜껑 · 반바지) 볼록 껍질 안」이면서
「어느 부품 solid 에도 안 든」 칸을 빈칸으로 본다. 빈칸으로만 이루어진 상자(팩 + 사방 2 mm 여유)를 6가지 방향으로
적분 부피(3차원 누적합)로 찾는다. 시나리오마다 빼낼 수 있는 부품을 다르게 둔다.
등 뒤 배낭형 · 골반 아래 해치는 몸통 밖이라 따로: 머리 끄덕임 · 팔 흔들기 · 다리 요(±20°) · 피치 자세의 설계 점과 겹치는지 본다.

usage: cd code_v17 && ../sim/.venv/bin/python ../sim/walk/battery6000/placement.py → sim/walk/battery6000/results/placement.json, placement.png
"""
import os, sys, json, math, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "..", "code_v17"); sys.path.insert(0, CODE)
import build19, build4, geom4
from geom4 import rot, apply
from scipy.spatial import Delaunay
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)

PACK = (60.0, 40.0, 70.0)
CLR = 2.0
VX = 2.0
LINKS, JOINTS, _ = build19.build_model()
J = {j["name"]: j for j in JOINTS}
T = LINKS["torso"]["parts"]
SHELL = ["torso_cover_front", "torso_cover_back", "lid", "shorts"]

lo = np.array([-96.0, -78.0, 200.0]); hi = np.array([72.0, 78.0, 340.0])
ax = [np.arange(lo[i] + VX / 2, hi[i], VX) for i in range(3)]
G = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
shape = G.shape[:3]
P = G.reshape(-1, 3)


def occ_of(parts):
    m = np.zeros(len(P), bool)
    for nm in parts:
        for s in T[nm]:
            b = build4.aabb(build4.sampled(s))
            sel = np.all((P >= [x[0] for x in b]) & (P <= [x[1] for x in b]), axis=1)
            if sel.any():
                idx = np.where(sel)[0]
                m[idx] |= geom4.inside(s, P[idx], 0.0)
    return m


shell_pts = np.vstack([geom4.samples(s, 3.0) for nm in SHELL for s in T[nm]])
hull = Delaunay(shell_pts)
inside_env = hull.find_simplex(P) >= 0
OCC = {nm: occ_of([nm]) for nm in T}


def fits(free3, dims):
    """free3: bool (nx, ny, nz). dims: 상자 크기 mm(여유 포함). 들어가는 모든 자리의 최소 모서리 index"""
    n = [int(math.ceil(d / VX)) for d in dims]
    bad = (~free3).astype(np.int32)
    S = np.zeros(tuple(s + 1 for s in bad.shape), np.int64)
    S[1:, 1:, 1:] = bad.cumsum(0).cumsum(1).cumsum(2)
    a, b, c = n
    if a > bad.shape[0] or b > bad.shape[1] or c > bad.shape[2]:
        return []
    tot = (S[a:, b:, c:] - S[:-a, b:, c:] - S[a:, :-b, c:] - S[a:, b:, :-c] + S[:-a, :-b, c:] + S[:-a, b:, :-c] + S[a:, :-b, :-c] - S[:-a, :-b, :-c])
    return np.argwhere(tot == 0), n


SCEN = {
    "몸통 안: 지금 배치에서 두 배터리만 뺌": ["battery_LB012", "battery_jetson_2S_1500"],
    "몸통 안: 두 배터리 + 해치 · 자석 뺌(바닥을 새로)": ["battery_LB012", "battery_jetson_2S_1500", "pelvis_hatch", "hatch_magnets_6x3"],
    "몸통 안: 두 배터리 + Jetson · 팬 · 강압 모듈 다시 배치": ["battery_LB012", "battery_jetson_2S_1500", "jetson_nano_2gb_devkit", "jetson_fan_40mm_5V", "buck_5V_3A_jetson"],
    "몸통 안: 두 배터리 + 선반 · 핀 · Jetson · 팬 · 강압 모듈 다시 배치": ["battery_LB012", "battery_jetson_2S_1500", "jetson_nano_2gb_devkit", "jetson_fan_40mm_5V", "buck_5V_3A_jetson", "electronics_shelf", "shelf_pins"],
}
out = {"pack_mm": PACK, "clearance_mm": CLR, "voxel_mm": VX, "inside": {}}
for sn, removable in SCEN.items():
    occ = np.zeros(len(P), bool)
    for nm, m in OCC.items():
        if nm not in removable:
            occ |= m
    free = (inside_env & ~occ).reshape(shape)
    res = []
    for perm in set(itertools.permutations(PACK)):
        dims = [d + 2 * CLR for d in perm]
        r = fits(free, dims)
        if not len(r) or not len(r[0]):
            continue
        idx, n = r
        # 후보 중 무게중심이 몸 가운데(y 0)에 가깝고 낮은 것 하나
        cent = np.array([[ax[i][k[i]] - VX / 2 + n[i] * VX / 2 for i in range(3)] for k in idx])
        score = np.abs(cent[:, 1]) * 2 + (cent[:, 2] - 230) * 0.5 + np.abs(cent[:, 0] + 10) * 0.3
        best = cent[np.argmin(score)]
        res.append(dict(orient_xyz=perm, n_positions=int(len(idx)), best_center_mm=best.round(1).tolist(),
                        x_range=[float(cent[:, 0].min()), float(cent[:, 0].max())], z_range=[float(cent[:, 2].min()), float(cent[:, 2].max())]))
    # 가장 큰 빈 상자 크기(참고): 방향마다 줄여 가며
    out["inside"][sn] = dict(removed=removable, fits=res, free_volume_cm3=float(free.sum() * VX ** 3 / 1000))
    print(sn, "fits:", [(r["orient_xyz"], r["n_positions"], r["best_center_mm"]) for r in res], flush=True)

# 참고: 두 배터리만 뺐을 때 들어가는 가장 큰 「40 mm 높이」 상자
occ = np.zeros(len(P), bool)
for nm, m in OCC.items():
    if nm not in SCEN["몸통 안: 지금 배치에서 두 배터리만 뺌"]:
        occ |= m
free = (inside_env & ~occ).reshape(shape)
best = None
for a in range(40, 121, 4):
    for b in range(40, 121, 4):
        for c in (40, 44):
            r = fits(free, (a + 2 * CLR, b + 2 * CLR, c + 2 * CLR))
            if len(r) and len(r[0]):
                v = a * b * c
                if best is None or v > best[0]:
                    best = (v, a, b, c)
out["inside_largest_box_after_removing_batteries_mm"] = best[1:] if best else None
print("largest box", best)


# ---------------------------------------------------------------- 몸 밖 후보
def link_pts(ln, step=2.0):
    return np.vstack([geom4.samples(s, step) for ss in LINKS[ln]["parts"].values() for s in ss])


def box_hits(c, d, pts):
    c = np.asarray(c); d = np.asarray(d) / 2 + CLR
    return int(np.all(np.abs(pts - c) <= d, axis=1).sum())


head = link_pts("head"); ears = link_pts("ears")
arms = {s: link_pts(f"{s}_arm") for s in "RL"}
torso_all = np.vstack([geom4.samples(s, 2.0) for nm, ss in T.items() for s in ss])
outside = {}
# (가) 등 뒤 배낭형: 두께 40 을 x 로, 뒤 덮개(x −93) 바깥 2 mm 판 뒤에
for nm, dims in (("등 뒤 배낭형(세로 70)", (40.0, 60.0, 70.0)), ("등 뒤 배낭형(가로 70)", (40.0, 70.0, 60.0))):
    best = None
    for zc in np.arange(240, 330, 2.0):
        c = (-93.0 - 2.0 - CLR - dims[0] / 2, 0.0, zc)
        h_head = max(box_hits(c, dims, apply(rot("y", b, J["head_nod"]["point_mm"]), np.vstack([head, ears]))) for b in (-25, -10, 0, 20, 33))
        h_arm = max(box_hits(c, dims, apply(rot("y", a, J[f"{s}_arm_swing"]["point_mm"]), arms[s])) for s in "RL" for a in range(-180, 181, 10))
        h_tor = box_hits(c, dims, torso_all)
        if h_head == 0 and h_arm == 0 and h_tor == 0:
            best = c if best is None or abs(zc - 280) < abs(best[2] - 280) else best
    outside[nm] = dict(dims_xyz=dims, center_mm=best, ok=best is not None,
                       note="머리 끄덕임 −25 ~ +33°(형상 한계까지), 팔 ±180° 와 안 겹침. OpenCR(뒤 x −83 ~ −63)의 포트를 가림")
# (나) 골반 아래 해치(두 다리 사이): 몸통 아래(z ≤ 221)에 매닮. 다리는 요 ±20° · 피치 −80 ~ 30° · (v20) 롤 B 의 롤 서보 상자
legs = {s: {ln: link_pts(f"{s}_{ln}") for ln in ("hip_yaw_link", "thigh", "shank")} for s in "RL"}
roll_box = dict(lo=np.array([-58.6, -54.25, 149.7]), hi=np.array([-17.7, -25.75, 196.2]))   # ../hiproll 안 B 롤 서보(오른쪽) [설계 형상 근사]


def leg_points(s, yaw, pitch, roll=0.0):
    jy, jp = J[f"{s}_hip_yaw"], J[f"{s}_hip_pitch"]
    My = rot("z", yaw, jy["point_mm"])
    Mp = My @ rot("x", roll, jp["point_mm"]) @ rot("y", pitch, jp["point_mm"])
    pts = [apply(My, legs[s]["hip_yaw_link"]), apply(Mp, legs[s]["thigh"]), apply(Mp, legs[s]["shank"])]
    rb = roll_box if s == "R" else dict(lo=roll_box["lo"] * [1, -1, 1], hi=roll_box["hi"] * [1, -1, 1])
    g = np.stack(np.meshgrid(*[np.arange(min(rb["lo"][i], rb["hi"][i]), max(rb["lo"][i], rb["hi"][i]), 3.0) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    pts.append(apply(My, g))
    return np.vstack(pts)


pel = {}
for dims in ((40.0, 40.0, 60.0), (60.0, 40.0, 70.0), (70.0, 40.0, 60.0), (40.0, 60.0, 70.0)):
    # dims = (x, y, z). 윗면을 z 221(해치 아래)에 붙임
    c = (-10.0, 0.0, 221.0 - CLR - dims[2] / 2)
    worst = 0; first = None
    for s in "RL":
        for yaw in range(-20, 21, 5):
            for pitch in range(-80, 31, 10):
                for roll in (-11.5, 0.0, 11.5):
                    rr = roll if s == "R" else -roll
                    n = box_hits(c, dims, leg_points(s, yaw, pitch, rr))
                    if n and first is None:
                        first = (s, yaw, pitch, rr, n)
                    worst = max(worst, n)
    pel[str(dims)] = dict(center=c, hits=worst, first=first)
    print("pelvis", dims, worst, first, flush=True)
outside["골반 아래 해치(두 다리 사이)"] = pel
out["outside"] = outside
json.dump(out, open(os.path.join(RES, "placement.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(outside, ensure_ascii=False, default=float)[:1500])
