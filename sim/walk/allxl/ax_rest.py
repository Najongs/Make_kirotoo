"""추가 조사(소유자 「서 있을 때 굳이 구부리고 있어야 하나?」): 편 무릎으로 쉬기 + 얕은 웅크림 걷기.
 rest : 두 발 서기 20 s, 웅크림 1 · 3 · 35 mm(무릎 약 12 · 20 · 63°) → 쉬는 동안 무릎 부하 RMS · 발열.
        웅크림은 다리를 다 편 몸통 높이에서 내려간 양(kw_ctrl). 제어기는 무릎 0.2 rad(11.5°)에서 시작한다(편 무릎은 역기구학 특이점).
 walk : 웅크림 15 · 20 · 25 mm × 0.04 · 0.05 · 0.077 m/s(한 걸음 0.65 s, 보폭 26 · 32.5 · 50 mm) × 마찰 0.7 · 0.8 · 0.9.
 trans: 편 무릎 쉬기(2 mm) 4 s → 1 s 동안 웅크려 걷기 → 18걸음 → 1 s 동안 펴기 → 4 s 쉬기(마찰 0.8). 제어기 웅크림 값만 시간에 따라 바꾼다.
usage: cd sim/walk/allxl && ../../.venv/bin/python ax_rest.py [arm2] → results/rest.json (arm2: rest_arm2.json, 관성 × 2)"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import ax_common as A
import kw_sim
from kw_ctrl import Controller
from kw_common import jdump
from run_tests import BASE

K2 = "2xXL430e"
LIGHT = {"torso": 0.8, "head": 0.8}
CFGS = {"17개: 모두 XL 그대로": A.cfg(),
        "17개: 모두 XL, −300 g": A.cfg(mass_scale=LIGHT),
        "19개: 무릎 XL 둘": A.cfg(dict(knee=K2)),
        "19개: 무릎 XL 둘, −300 g": A.cfg(dict(knee=K2), mass_scale=LIGHT),
        "21개: D6 (무릎 · 롤 XL 둘, −300 g)": A.cfg(dict(knee=K2, hip_roll=K2), mass_scale=LIGHT)}
SCHED = {"pts": None}


class SchedController(Controller):
    """웅크림 값을 시간표대로 바꾼다(SCHED["pts"] = [(t, crouch), ...] 직선 보간). 없으면 원래대로."""
    def step(self, t, sen):
        if SCHED["pts"]:
            ts, cs = zip(*SCHED["pts"]); self.crouch = float(np.interp(t, ts, cs))
        return super().step(t, sen)


kw_sim.Controller = SchedController


def one(job):
    meta, c = job
    SCHED["pts"] = c.pop("sched", None)
    try:
        return A.one((meta, c))
    finally:
        SCHED["pts"] = None


def jobs():
    J = []
    for n, c in CFGS.items():
        for cr in (0.001, 0.003, 0.035):
            J.append((dict(cfg=n, test="rest", crouch=cr), dict(c, crouch=cr, t_walk=99, T_end=20.0)))
        for cr in (0.015, 0.02, 0.025):
            for s in (0.026, 0.0325, 0.05):
                for f in A.FRIC:
                    J.append((dict(cfg=n, test="walk", crouch=cr, stride=s, fr=f), dict(c, crouch=cr, stride=s, friction=f)))
        for cr in (0.02, 0.035):
            tw = 5.0; T = BASE["Ts"] + BASE["Td"]; t_end_walk = tw + 0.8 + (BASE["n_steps"] - 1) * T + 0.8
            pts = [(0.0, 0.002), (tw - 1.5, 0.002), (tw - 0.5, cr), (t_end_walk + 0.3, cr), (t_end_walk + 1.3, 0.002), (99.0, 0.002)]
            J.append((dict(cfg=n, test="trans", crouch=cr, stride=0.0325, fr=0.8),
                      dict(c, crouch=cr, stride=0.0325, t_walk=tw, T_end=t_end_walk + 5.5, sched=pts)))
    return J


def jobs_arm2():
    """회전자 관성 × 2 민감도(XL 은 코어 모터): 17개 −300 g 과 D6, 웅크림 20 mm, 0.04 · 0.05 m/s, 마찰 세 판"""
    import kw_common as C
    for k in ("XL430e", "2xXL430e"):
        v = C.SERVO_SIM[k]; C.SERVO_SIM[k + "_a2"] = v[:3] + (2 * v[3],); C.SERVO_G[k + "_a2"] = C.SERVO_G[k]; A.HEAT[k + "_a2"] = A.HEAT[k]
    a2 = {k: "XL430e_a2" for k in A.ALLXL}
    cf = {"17개: 모두 XL, −300 g, 관성 × 2": A.cfg(a2, mass_scale=LIGHT),
          "21개: D6, 관성 × 2": A.cfg(dict(a2, knee=K2 + "_a2", hip_roll=K2 + "_a2"), mass_scale=LIGHT)}
    return [(dict(cfg=n, test="walk", crouch=0.02, stride=s, fr=f), dict(c, crouch=0.02, stride=s, friction=f))
            for n, c in cf.items() for s in (0.026, 0.0325) for f in A.FRIC]


if __name__ == "__main__":
    t = time.time()
    from multiprocessing import Pool
    arm2 = "arm2" in sys.argv
    J = jobs_arm2() if arm2 else jobs()
    with Pool(min(76, len(J))) as p:
        out = p.map(one, J, chunksize=1)
    rows = [dict(meta=m_, cfg={k: v for k, v in c.items() if k != "extra_mass"}, res=r) for m_, c, r in out]
    jdump(dict(note=__doc__, rows=rows), os.path.join(A.RES, "rest_arm2.json" if arm2 else "rest.json"))
    print(f"rest: {len(rows)} runs, {sum(1 for r in rows if r['res'].get('fallen'))} fell, {sum(1 for r in rows if 'error' in r['res'])} errors, {time.time() - t:.0f}s")
