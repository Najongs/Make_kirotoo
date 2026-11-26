"""새 배터리(12 V 6000 mAh, 60 × 40 × 70 mm, 무게 250 g = 소유자 지정 가정값(민감도 300 g)) 자리 후보별 피드백 걷기 · 밀기 · 전류(MuJoCo).

모델: v18 MJCF + 고관절 롤 안 B(XM430, ../hiproll/hr_cfg.py) + v20 서보 배치(XM: 롤 · 피치 · 무릎 · 발목 롤, XC: 발목 피치,
      XL: 고관절 요 · 팔 · 머리 크랭크) — kw_common servo_map 으로 kp · 토크 · 감쇠 · 관성 · 서보 질량 차이를 바꿈.
배터리: 지금 LB-012(106 g, 골반 위 (−11, 0, 243) mm)를 빼고 새 팩을 후보 자리에 점 질량으로. 한 팩 안은 Jetson 배터리
      (85 g, (42.5, 0, 253) mm)도 빼고 강압 모듈 15 g [추정]을 OpenCR 옆 (−55, 0, 290) mm 에 더함.
python battery6000/sim_runs.py → battery6000/results/sim.json, current.json
"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hiproll"))
from run_tests import BASE, push_t, run_jobs
from hr_cfg import hr
from kw_common import jdump, build_model, Idx
import numpy as np, mujoco

V20 = {"hip_yaw": "XL430-W250", "hip_pitch": "XM430-W350", "knee": "XM430-W350", "ankle_pitch": "XC430-W240",
       "ankle_roll": "XM430-W350", "arm_swing": "XL430-W250"}
MODEL = dict(hip_roll=hr("B", "XM430-W350"), servo_map=V20)
LB = ("torso", -0.106, (-0.011, 0.0, 0.2431))
JB = ("torso", -0.085, (0.0425, 0.0, 0.253))
BUCK = ("torso", 0.015, (-0.055, 0.0, 0.290))
BACK = (-0.117, 0.0, 0.280)          # placement.py: 등 뒤 배낭형 가운데 [설계 형상]
INSIDE = (0.005, 0.0, 0.302)         # placement.py: 몸통 위칸(Jetson · 팬 · 강압 모듈을 다른 데로 옮긴 뒤) [설계 형상]


def pack(pos, m, merge=True):
    em = [LB, ("torso", m, pos)]
    if merge:
        em += [JB, BUCK]
    return tuple(em)


CONFIGS = {
    "지금 배터리(LB-012 + Jetson 2S)": dict(MODEL),
    "등 뒤 250 g (한 팩)": dict(MODEL, extra_mass=pack(BACK, 0.250)),
    "등 뒤 300 g (한 팩)": dict(MODEL, extra_mass=pack(BACK, 0.300)),
    "등 뒤 250 g (두 팩: Jetson 2S 유지)": dict(MODEL, extra_mass=pack(BACK, 0.250, merge=False)),
    "몸통 위칸 250 g (한 팩)": dict(MODEL, extra_mass=pack(INSIDE, 0.250)),
    "몸통 위칸 300 g (한 팩)": dict(MODEL, extra_mass=pack(INSIDE, 0.300)),
}


def jobs():
    J = []
    t0 = push_t(6, "ss")
    for cn, cc in CONFIGS.items():
        for s in (0.05, 0.055, 0.06):
            J.append((dict(config=cn, test="speed", stride=s), dict(cc, stride=s)))
        J.append((dict(config=cn, test="stand"), dict(cc, t_walk=99, T_end=12.0)))
        for dr in ("+y", "-y", "+x", "-x"):
            for F in (2, 4, 6, 8, 10):
                J.append((dict(config=cn, test="push", dir=dr, F=F), dict(cc, push=(t0, 0.2, F * (1 if dr[0] == "+" else -1), dr[1]), T_end=t0 + 4.0)))
    return J


def com_table():
    out = {}
    for cn, cc in CONFIGS.items():
        m, _ = build_model(hip_roll=cc["hip_roll"], servo_map=cc["servo_map"], extra_mass=cc.get("extra_mass", ()))
        d = mujoco.MjData(m); d.qpos[:] = m.qpos0; mujoco.mj_forward(m, d)
        out[cn] = dict(mass_g=float(m.body_subtreemass[1] * 1e3), com_mm=(d.subtree_com[1] * 1e3 - [0, 0, m.qpos0[2] * 1e3]).round(1).tolist())
    return out


if __name__ == "__main__":
    t = time.time()
    com = com_table(); print(json.dumps(com, ensure_ascii=False))
    out = run_jobs(jobs(), procs=48)
    rows = [dict(meta=m_, res=r) for m_, c, r in out]
    jdump(dict(com=com, rows=rows), os.path.join(HERE, "results", "sim.json"))
    print(len(rows), "runs", f"{time.time() - t:.0f}s", flush=True)
    # 전류(../hiproll/current_budget.py 의 계산을 그대로)
    import current_budget as CB
    t0 = push_t(6, "ss")
    sc = []
    for cn in ("지금 배터리(LB-012 + Jetson 2S)", "등 뒤 250 g (한 팩)"):
        cc = CONFIGS[cn]
        sc += [(f"{cn} · 서 있기", dict(cc, t_walk=99, T_end=12.0)), (f"{cn} · 걷기 0.076", dict(cc, stride=0.05)),
               (f"{cn} · 걷기 0.083", dict(cc, stride=0.055)), (f"{cn} · 끄덕이며 걷기 0.076", dict(cc, stride=0.05, nod=(1.0, 3.0))),
               (f"{cn} · 넘어짐(걷다 앞 15 N)", dict(cc, stride=0.03, push=(t0, 0.2, 15.0, "x"), T_end=t0 + 4, record_fall=True, after_fall_s=2.0))]
    from multiprocessing import Pool
    with Pool(len(sc)) as p:
        cur = p.map(CB.one, sc)
    for r in cur:
        r.pop("series", None)
    jdump(cur, os.path.join(HERE, "results", "current.json"))
    for r in cur:
        print(r["name"], "fall" if r["fallen"] else "", {k: round(v, 2) for k, v in r["total"].items()})
