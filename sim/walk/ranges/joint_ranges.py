"""관절별 최대 이동 범위(형상 한계) · 짝 관절 충돌 지도 · 바닥/자세 범위 · 안전 범위 초안.

판정: 설계 형상 정밀 검사 — code_v17 의 build 모듈 build_model() 부품 solid, 움직이는 쪽은 격자 점(geom4.samples),
      고정 쪽은 geom4.inside(margin 0.05 mm)  (sweep17.py 와 같은 방식).
후보 고르기: MuJoCo MJCF 의 부품 볼록 껍질 사이 거리(mj_geomDistance). 껍질이 1 mm 보다 멀면 그 링크 쌍은 정밀 검사를 건너뜀
      (볼록 껍질은 부품을 감싸므로 안전한 거름).
영점 자세에서 이미 겹치는 부품 쌍(압입 · 축 · 핀)은 처음에 한 번 찾아 그 뒤로 뺌(목록을 결과에 남김).

usage (v19):  cd sim/walk && ../.venv/bin/python ranges/joint_ranges.py --build build19 --mjcf ../model/kitoru_v19.xml \
                  --joints-csv ../../out19/joints.csv --tag v19
v20 이 나오면:  --build build20 --mjcf ../model/kitoru_v20.xml --joints-csv ../../out20/joints.csv --tag v20
              (R/L_hip_roll 은 joints 목록에서 자동으로 잡힘; --joints 로 일부만 고를 수 있음)
결과: ranges/results/<tag>/single.json, pairs.npz, pairs_*.png, posture.json, joint_ranges_<tag>.csv, mjcf_ranges_<tag>.txt
"""
import os, sys, json, math, argparse, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CODE = os.path.join(ROOT, "code_v17")
sys.path.insert(0, CODE)
os.environ.setdefault("MUJOCO_GL", "egl")

ap = argparse.ArgumentParser()
ap.add_argument("--build", default="build19")
ap.add_argument("--mjcf", default=os.path.join(ROOT, "sim", "model", "kitoru_v19.xml"))
ap.add_argument("--joints-csv", default=os.path.join(ROOT, "out19", "joints.csv"))
ap.add_argument("--tag", default="v19")
ap.add_argument("--step", type=float, default=1.5, help="움직이는 부품 격자 간격 mm")
ap.add_argument("--joints", default="", help="쉼표로 고른 관절만(비우면 액추에이터 관절 전부 + head_nod)")
ap.add_argument("--procs", type=int, default=40)
ap.add_argument("--skip", default="", help="건너뛸 단계: single,pairs,posture")
ARGS = ap.parse_args()

import importlib
import geom4, build4
from geom4 import rot, apply
import mujoco

BM = importlib.import_module(ARGS.build)
LINKS, JOINTS, _ = BM.build_model()
J = {j["name"]: j for j in JOINTS}
OUT = os.path.join(HERE, "results", ARGS.tag); os.makedirs(OUT, exist_ok=True)
MARGIN = 0.05
# 닫힌 고리 막대는 고리 검사(sweep17)에서 따로 봄: 여기서는 뺌. 귀 · 크랭크는 머리와 함께 움직인다고 봄(근사).
EXCLUDE_LINKS = {"nod_rod", "ear_rod"}
RIGID_WITH = {"nod_crank": "head", "ears": "head"}
# 기구로 원래 맞닿아 움직이는 부품 쌍(sweep17.py 와 같은 예외): 팔 캠 막대 ↔ 혼 어댑터, 손 날개 ↔ 핀, 귀 축 ↔ 베어링
MECH_PAIRS = [("finger_rod", "horn_adapter"), ("hand_flap", "pin"), ("axle", "bearing")]


def mech_ok(pn, fn):
    return any((a in pn and b in fn) or (a in fn and b in pn) for a, b in MECH_PAIRS)
FLAP = getattr(BM, "flap_angle", None)

# ---------------------------------------------------------------- 운동학
CHILD_JOINT = {j["child"]: j for j in JOINTS}


def link_path(link):
    p = []
    while link in CHILD_JOINT:
        j = CHILD_JOINT[link]; p.append(j["name"]); link = j["parent"]
    return p[::-1]


def poses(q):
    """q: {관절 이름: 도} → {링크: 4x4}. 손 날개는 팔 각도를 따름(flap_angle), 귀 · 크랭크는 머리에 고정."""
    out = {}
    for ln in LINKS:
        if ln in EXCLUDE_LINKS:
            continue
        M = np.eye(4)
        for jn in link_path(ln):
            if jn in ("nod_crank", "ear_axle", "ear_rod", "nod_rod"):
                continue
            a = q.get(jn, 0.0)
            if jn.endswith("_hand_flap") and FLAP is not None:
                a = float(FLAP(q.get(jn.replace("hand_flap", "arm_swing"), 0.0)))
            jj = J[jn]
            M = M @ rot(jj["axis"], a, jj["point_mm"])
        out[ln] = M
    return out


def subtree(jname):
    root = J[jname]["child"]; s = {root}
    changed = True
    while changed:
        changed = False
        for j in JOINTS:
            if j["parent"] in s and j["child"] not in s:
                s.add(j["child"]); changed = True
    return s - EXCLUDE_LINKS


# ---------------------------------------------------------------- 형상
def solids(link):
    return [(pn, s) for pn, ss in LINKS[link]["parts"].items() if "joint_disc" not in pn for s in ss
            if not (s[0] == "cyl" and s[2] == build4.DISC)]


FIX = {ln: solids(ln) for ln in LINKS if ln not in EXCLUDE_LINKS}
BB = {id(s): build4.aabb(build4.sampled(s)) for ss in FIX.values() for _, s in ss}
PTS = {}
for ln in FIX:
    PTS[ln] = [(pn, geom4.samples(s, ARGS.step)) for pn, s in FIX[ln]]


def hits(moving, M, fixed, Mf):
    """moving 링크(자세 M)의 격자 점이 fixed 링크(자세 Mf) solid 안에 든 것: [(부품, 부품, 점 수, 점 평균(world))]"""
    out = []
    Mi = np.linalg.inv(Mf) @ M
    for pn, P in PTS[moving]:
        if not len(P):
            continue
        Q = apply(Mi, P)
        qlo, qhi = Q.min(0), Q.max(0)
        for fn, f in FIX[fixed]:
            b = BB[id(f)]
            if any(qhi[i] < b[i][0] or qlo[i] > b[i][1] for i in range(3)):
                continue
            sel = np.all((Q >= [x[0] for x in b]) & (Q <= [x[1] for x in b]), axis=1)
            if sel.any():
                ins = geom4.inside(f, Q[sel], MARGIN)
                n = int(ins.sum())
                if n:
                    Pw = apply(M, P[sel][ins])
                    out.append((pn, fn, n, Pw.mean(0)))
    return out


# ---------------------------------------------------------------- MuJoCo 볼록 껍질 거름
MJ = mujoco.MjModel.from_xml_path(ARGS.mjcf); MD = mujoco.MjData(MJ)
BODY_OF = {"R_hand_flap": "R_arm", "L_hand_flap": "L_arm", "nod_rod": "torso", "ear_rod": "head"}
COLG = {}
for g in range(MJ.ngeom):
    if MJ.geom_contype[g] == 0 and MJ.geom_conaffinity[g] == 0:
        continue
    bn = mujoco.mj_id2name(MJ, mujoco.mjtObj.mjOBJ_BODY, MJ.geom_bodyid[g])
    if bn and bn != "world":
        COLG.setdefault(bn, []).append(g)
MJ_JOINTS = {mujoco.mj_id2name(MJ, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(MJ.njnt)}
FROM = np.zeros(6)


def mj_set(q, base=None):
    MD.qpos[:] = MJ.qpos0
    if MJ.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        MD.qpos[3:7] = [1, 0, 0, 0] if base is None else base
    for jn, a in q.items():
        if jn in MJ_JOINTS:
            MD.qpos[MJ.jnt_qposadr[mujoco.mj_name2id(MJ, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = math.radians(a)
    mujoco.mj_kinematics(MJ, MD)


def hull_near(a, b, dmax=0.001):
    ba, bb = BODY_OF.get(a, a), BODY_OF.get(b, b)
    if ba == bb or ba not in COLG or bb not in COLG:
        return True
    for g1 in COLG[ba]:
        for g2 in COLG[bb]:
            if mujoco.mj_geomDistance(MJ, MD, g1, g2, dmax, FROM) < dmax:
                return True
    return False


# ---------------------------------------------------------------- 충돌 판정
BASE = None


def baseline():
    """영점 자세에서 이미 겹치는 (링크, 부품, 링크, 부품) — 압입 · 축 · 핀"""
    P0 = poses({})
    mj_set({})
    out = set()
    names = list(FIX)
    for i, a in enumerate(names):
        for b in names:
            if a == b or not hull_near(a, b):
                continue
            for pn, fn, n, c in hits(a, P0[a], b, P0[b]):
                out.add((a, pn, b, fn))
    return out


def collide(q, moving, others=None, use_mj=True):
    """moving 링크 집합과 나머지 사이 충돌: [(움직이는 링크, 부품, 고정 링크, 부품, 점 수, 위치)]"""
    P = poses(q)
    if use_mj:
        mj_set(q)
    others = others if others is not None else [l for l in FIX if l not in moving]
    res = []
    for a in moving:
        for b in others:
            if use_mj and not hull_near(a, b):
                continue
            for pn, fn, n, c in hits(a, P[a], b, P[b]):
                if (a, pn, b, fn) in BASE or (b, fn, a, pn) in BASE or mech_ok(pn, fn):
                    continue
                res.append((a, pn, b, fn, n, c))
    return res


def collide_all(q, use_mj=True):
    """자세 q 에서 서로 상대 자세가 바뀐 모든 링크 쌍을 양쪽 방향으로 검사(짝 관절 · 자세 검사용)"""
    P = poses(q)
    if use_mj:
        mj_set(q)
    names = list(P)
    res = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rel = np.linalg.inv(P[a]) @ P[b]
            if np.allclose(rel, np.eye(4), atol=1e-9):
                continue
            if use_mj and not hull_near(a, b):
                continue
            for x, y in ((a, b), (b, a)):
                for pn, fn, n, c in hits(x, P[x], y, P[y]):
                    if (x, pn, y, fn) in BASE or (y, fn, x, pn) in BASE or mech_ok(pn, fn):
                        continue
                    res.append((x, pn, y, fn, n, c))
    return res


def ok_pose(q, moving):
    return not collide(q, moving)


# ---------------------------------------------------------------- 1. 관절 하나씩
def axis_radius(jn, c):
    jj = J[jn]; p = np.array(jj["point_mm"], float); ax = {"x": 0, "y": 1, "z": 2}[jj["axis"]]
    d = np.array(c, float) - p; d[ax] = 0
    return float(np.linalg.norm(d))


def search_dir(args):
    jn, sgn, step, amax = args
    mov = subtree(jn)
    prev = 0.0
    a = step
    while a <= amax + 1e-9:
        h = collide_all({jn: sgn * a})
        if h:
            lo, hi = prev, a
            for _ in range(4):
                mid = 0.5 * (lo + hi)
                if collide_all({jn: sgn * mid}):
                    hi = mid
                else:
                    lo = mid
            hh = collide_all({jn: sgn * hi})
            worst = max(hh, key=lambda x: x[4])
            r = axis_radius(jn, worst[5])
            return dict(joint=jn, dir=sgn, limit=sgn * lo, first_hit=sgn * hi, pair=f"{worst[0]}:{worst[1]} ↔ {worst[2]}:{worst[3]}",
                        n_pts=worst[4], contact_mm=np.round(worst[5], 1).tolist(), radius_mm=r)
        prev = a; a += step
    return dict(joint=jn, dir=sgn, limit=sgn * amax, first_hit=None, pair="(±180° 까지 충돌 없음)", n_pts=0, contact_mm=None, radius_mm=None)


def init_worker():
    pass


# ---------------------------------------------------------------- 2. 짝 관절 지도
def pair_specs():
    S = []
    have = set(J)
    for s in "RL":
        if f"{s}_knee" in have:
            S.append((f"{s} 고관절 피치 × 무릎", f"{s}_hip_pitch", (-110, 60, 5), f"{s}_knee", (-30, 160, 5), {}))
            S.append((f"{s} 무릎 × 발목 피치", f"{s}_knee", (-30, 160, 5), f"{s}_ankle_pitch", (-100, 70, 5), {}))
            S.append((f"{s} 발목 피치 × 발목 롤", f"{s}_ankle_pitch", (-100, 70, 5), f"{s}_ankle_roll", (-50, 50, 5), {}))
            if f"{s}_hip_roll" in have:
                S.append((f"{s} 고관절 롤 × 고관절 피치", f"{s}_hip_roll", (-45, 45, 5), f"{s}_hip_pitch", (-110, 60, 5), {}))
        S.append((f"{s} 팔 × 머리 끄덕임", f"{s}_arm_swing", (-180, 180, 10), "head_nod", (-20, 30, 5), {}))
        S.append((f"{s} 팔 × 고관절 피치", f"{s}_arm_swing", (-180, 180, 10), f"{s}_hip_pitch", (-110, 60, 10), {}))
    for pk, (hp, kn) in {"다리 편 자세": (0, 0), "웅크림(피치 −30, 무릎 60)": (-30, 60), "무릎만 60": (0, 60)}.items():
        ctx = {f"{s}_hip_pitch": hp for s in "RL"}; ctx.update({f"{s}_knee": kn for s in "RL"}); ctx.update({f"{s}_ankle_pitch": -(hp + kn) for s in "RL"})
        S.append((f"두 다리 고관절 요 ({pk})", "R_hip_yaw", (-45, 45, 3), "L_hip_yaw", (-45, 45, 3), ctx))
    return S


def pair_cell(args):
    name, ja, a, jb, b, ctx = args
    q = dict(ctx); q[ja] = a; q[jb] = b
    return name, a, b, len(collide_all(q)) == 0


# ---------------------------------------------------------------- 3. 바닥 · 자세
def all_points(q, base_R=np.eye(3)):
    P = poses(q)
    out = {}
    for ln, M in P.items():
        Pw = np.vstack([apply(M, p) for _, p in PTS[ln] if len(p)])
        out[ln] = Pw @ base_R.T
    return out


def posture():
    res = {}
    leg = [l for l in FIX if l[:2] in ("R_", "L_") and "arm" not in l and "flap" not in l]
    # (a) 두 발 평평히 서서 웅크리기: 고관절 피치 = −무릎/2, 발목 피치 = −무릎/2 (몸통 세움, 발목이 고관절 바로 아래)
    crouch = []
    for kn in np.arange(0, 161, 2.0):
        q = {}
        for s in "RL":
            q.update({f"{s}_hip_pitch": -kn / 2, f"{s}_knee": kn, f"{s}_ankle_pitch": -kn / 2})
        mov = {l for l in FIX if l.startswith(("R_thigh", "R_shank", "R_ankle", "R_foot", "L_thigh", "L_shank", "L_ankle", "L_foot"))}
        h = collide_all(q)
        L = 2 * 75 * math.cos(math.radians(kn / 2))
        crouch.append(dict(knee=kn, hip_ankle_mm=L, crouch_mm=150 - L, ok=not h, hit=(f"{h[0][0]}:{h[0][1]} ↔ {h[0][2]}:{h[0][3]}" if h else "")))
    first_bad = next((c for c in crouch if not c["ok"]), None)
    lim_j = {s: (J[f"R_{s}"]["lower_deg"], J[f"R_{s}"]["upper_deg"]) for s in ("hip_pitch", "knee", "ankle_pitch")}
    kmax_joint = min(lim_j["knee"][1], -2 * lim_j["hip_pitch"][0], -2 * lim_j["ankle_pitch"][0])
    res["crouch"] = dict(table=crouch, first_collision=first_bad, knee_max_by_joints_csv=kmax_joint,
                         crouch_max_by_joints_csv_mm=150 - 150 * math.cos(math.radians(kmax_joint / 2)))
    # (b) 발목 롤로 몸 기울이기: 두 발 평평이면 고관절 롤이 없을 때 몸통 롤 0 으로 묶임(운동학). 한 발로 서면 발목 롤 한계까지.
    res["ankle_roll_tilt"] = dict(double_support_deg=(0.0 if not any(k.endswith("hip_roll") for k in J) else None),
                                  note="두 발이 평평하고 평행이면 고관절 롤이 없을 때 몸통 롤은 0 으로 묶임(cos φ = 1). 다리 길이 차이를 쓰면 약 0.08/cos φ 의 발 간격이 필요해 수 mm 어긋남을 패드 · 서보 휨이 받음.")
    # (c) 엎드림 · 누움: 몸을 통째로 눕힌 뒤 바닥 = 영점 자세의 가장 낮은 점. 팔 · 다리를 돌릴 때 바닥에 닿는 각도.
    lying = {}
    for nm, deg in (("엎드림(앞이 바닥)", 90.0), ("누움(등이 바닥)", -90.0)):
        Rb = np.array([[math.cos(math.radians(deg)), 0, math.sin(math.radians(deg))], [0, 1, 0], [-math.sin(math.radians(deg)), 0, math.cos(math.radians(deg))]])
        base = all_points({}, Rb)
        floor = min(P[:, 2].min() for P in base.values())
        rest = sorted(((P[:, 2].min() - floor, ln) for ln, P in base.items()))[:4]
        out = {"floor_mm": floor, "resting_links": [(ln, round(float(d), 1)) for d, ln in rest]}
        for jn, grid in (("R_arm_swing", np.arange(-180, 181, 5)), ("L_arm_swing", np.arange(-180, 181, 5)),
                         ("R_hip_pitch", np.arange(-110, 61, 5)), ("R_knee", np.arange(-30, 161, 5)), ("R_ankle_pitch", np.arange(-100, 71, 5))):
            if jn not in J:
                continue
            mov = subtree(jn)
            rows = []
            for a in grid:
                q = {jn: float(a)}
                Pp = poses(q)
                zmin = min((apply(Pp[l], np.vstack([p for _, p in PTS[l] if len(p)])) @ Rb.T)[:, 2].min() for l in mov)
                self_ok = len(collide_all(q)) == 0
                rows.append((float(a), float(zmin - floor), self_ok))
            out[jn] = rows
        lying[nm] = out
    res["lying"] = lying
    return res


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    t0 = time.time()
    BASE = baseline()
    print(f"[{ARGS.tag}] baseline overlaps {len(BASE)} ({time.time() - t0:.0f}s)", flush=True)
    skip = set(ARGS.skip.split(",")) if ARGS.skip else set()
    act = [j["name"] for j in JOINTS if j.get("actuator") and not str(j["actuator"]).startswith("passive")]
    jl = [x for x in ARGS.joints.split(",") if x] or (act + (["head_nod"] if "head_nod" in J else []))
    jl = [j for j in jl if j != "nod_crank"]
    from multiprocessing import Pool
    json.dump({"baseline_overlaps": sorted([list(x) for x in BASE]), "joints": jl, "build": ARGS.build, "mjcf": ARGS.mjcf, "step_mm": ARGS.step},
              open(os.path.join(OUT, "meta.json"), "w"), indent=1, ensure_ascii=False)
    if "single" not in skip:
        tasks = [(jn, s, 2.0, 180.0) for jn in jl for s in (-1, 1)]
        with Pool(min(ARGS.procs, len(tasks))) as p:
            single = p.map(search_dir, tasks)
        json.dump(single, open(os.path.join(OUT, "single.json"), "w"), indent=1, ensure_ascii=False, default=float)
        print(f"single done ({time.time() - t0:.0f}s)", flush=True)
    if "pairs" not in skip:
        specs = pair_specs()
        cells = []
        for name, ja, (a0, a1, da), jb, (b0, b1, db), ctx in specs:
            for a in np.arange(a0, a1 + 1e-9, da):
                for b in np.arange(b0, b1 + 1e-9, db):
                    cells.append((name, ja, float(a), jb, float(b), ctx))
        with Pool(ARGS.procs) as p:
            res = p.map(pair_cell, cells, chunksize=8)
        out = {}
        for name, ja, (a0, a1, da), jb, (b0, b1, db), ctx in specs:
            A = np.arange(a0, a1 + 1e-9, da); B = np.arange(b0, b1 + 1e-9, db)
            out[name] = dict(ja=ja, jb=jb, A=A, B=B, free=np.zeros((len(A), len(B)), bool))
        for name, a, b, okk in res:
            o = out[name]; i = int(round((a - o["A"][0]) / (o["A"][1] - o["A"][0]))); k = int(round((b - o["B"][0]) / (o["B"][1] - o["B"][0])))
            o["free"][i, k] = okk
        np.savez(os.path.join(OUT, "pairs.npz"), **{f"{k}|{f}": v[f] for k, v in out.items() for f in ("A", "B", "free")},
                 **{f"{k}|joints": np.array([v["ja"], v["jb"]]) for k, v in out.items()})
        print(f"pairs done {len(cells)} cells ({time.time() - t0:.0f}s)", flush=True)
    if "posture" not in skip:
        pos = posture()
        json.dump(pos, open(os.path.join(OUT, "posture.json"), "w"), indent=1, ensure_ascii=False, default=float)
        print(f"posture done ({time.time() - t0:.0f}s)", flush=True)
