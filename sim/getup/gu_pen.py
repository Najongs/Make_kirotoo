"""팔이 몸을 뚫고 지나가는지(시뮬은 팔과 몸 · 다리의 충돌을 계산하지 않음): 성공한 동작을 다시 돌리며 50 Hz 로
팔 geom(어깨 · 아래팔)과 머리 · 몸통 · 다리 geom 사이 가장 짧은 거리(음수 = 파고듦 깊이)를 잰다. 쉬는 자세(qpos0)의 값을 기준으로 함께 적는다.
usage: ../.venv/bin/python gu_pen.py <tag> <팔 이름> <start> → results/pen_<tag>.json"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import mujoco
import gu_search as G
from gu_model import RES, jdump

OTHER = ("head", "torso", "R_hip_yaw_link", "L_hip_yaw_link", "R_thigh", "L_thigh", "R_shank", "L_shank")


def groups(m):
    nm = lambda i: m.body(m.geom_bodyid[i]).name
    arm = {s: [i for i in range(m.ngeom) if m.geom_contype[i] and nm(i) in (f"{s}_arm", f"{s}_forearm")] for s in "RL"}
    # 몸통의 어깨 혼 · 아이들러 · 어깨 축은 팔이 붙는 자리라 빼고 잰다(원래 팔과도 늘 겹침)
    skip = lambda i: any(k in (m.geom(i).name or "") for k in ("arm_", "shoulder_shaft"))
    oth = {b: [i for i in range(m.ngeom) if nm(i) == b and (m.geom_contype[i] or m.geom_conaffinity[i]) and not skip(i)] for b in OTHER}
    return arm, oth


LEGB = ("R_thigh", "L_thigh", "R_shank", "L_shank", "R_ankle_link", "L_ankle_link", "R_foot", "L_foot")


def dists(m, d, arm, oth, ft):
    out = {}
    # 다리(허벅지 · 정강이 · 발목 · 발) ↔ 머리 · 몸통(어깨 부품 뺌)도 시뮬이 충돌을 계산하지 않으므로 함께 잰다
    if not hasattr(dists, "leg"):
        nm = lambda i: m.body(m.geom_bodyid[i]).name
        dists.leg = [i for i in range(m.ngeom) if nm(i) in LEGB and (m.geom_contype[i] or m.geom_conaffinity[i]) and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
    for b in ("head", "torso"):
        out["다리↔" + b] = min(mujoco.mj_geomDistance(m, d, a, g, 0.2, ft) for a in dists.leg for g in oth[b]) * 1e3
    for b, gs in oth.items():
        out[b] = min(mujoco.mj_geomDistance(m, d, a, g, 0.2, ft) for s in "RL" for a in arm[s] for g in gs) * 1e3
    return out


class NullRen:
    def update_scene(self, d, cam): pass
    def render(self): return np.zeros(1)


if __name__ == "__main__":
    tag, armname, start = sys.argv[1:4]
    r = next(x for x in json.load(open(os.path.join(RES, f"{tag}.json"))) if x["arm"] == armname and x["start"] == start)
    S0, _, _, _ = G.replay(r)
    m = S0.m; arm, oth = groups(m); ft = np.zeros(6)
    d0 = mujoco.MjData(m); d0.qpos[:] = m.qpos0; mujoco.mj_forward(m, d0); base = dists(m, d0, arm, oth, ft)
    rec = []

    def camf(d):
        mujoco.mj_forward(m, d) if False else None
        rec.append((d.time, dists(m, d, arm, oth, ft))); return None
    # replay 는 새 Sim 을 만들므로 그 모델로 다시 그룹을 잡는다
    import gu_model
    orig = gu_model.Sim.run

    def run_hook(self, keys, start="prone", T_hold=1.5, render=None, settle=0.8):
        global m, arm, oth
        m = self.m; arm, oth = groups(m)
        return orig(self, keys, start=start, T_hold=T_hold, render=(NullRen(), camf, 50), settle=settle)
    gu_model.Sim.run = run_hook
    S, res, H, _ = G.replay(r)
    KEYS = list(OTHER) + ["다리↔head", "다리↔torso"]
    worst = {b: min((x[1][b], x[0]) for x in rec) for b in KEYS}
    out = dict(tag=tag, arm=armname, start=start, ok=res["ok"], base_mm=base, worst_mm={b: dict(dist=w[0], t=w[1]) for b, w in worst.items()},
               below_base_minus5_s={b: 0.02 * sum(1 for x in rec if x[1][b] < min(base[b], 0) - 5) for b in KEYS})
    path = os.path.join(RES, f"pen_{tag}.json"); old = json.load(open(path)) if os.path.exists(path) else {}
    old[f"{armname}/{start}"] = out; jdump(old, path)
    print("ok", res["ok"]); print("쉬는 자세", {b: round(v, 1) for b, v in base.items()})
    print("최소", {b: (round(w[0], 1), round(w[1], 2)) for b, w in worst.items()})
    print("기준보다 5 mm 넘게 파고든 시간 s", out["below_base_minus5_s"])
