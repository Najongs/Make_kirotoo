"""서보 전류 · 전력 예산(시뮬): 시험마다 서보 부하 토크 · 속도 → 전류, 버스별 · 전체 합, 배터리 사용 시간.

전류(스펙 방식, 상한): I = I_대기 + |τ|/Kt, 정지 전류에서 자름.  τ = 관절 부하 토크 = 액추에이터 힘 − 감쇠 b·ω(역기전력 몫).
  Kt · 정지 전류 · 대기 전류는 ROBOTIS e-manual 11.1 V 값 [e-manual]:
    XM430-W350 3.8 N·m / 2.1 A → 1.810 N·m/A, 대기 40 mA · XC430-W240 1.7 N·m / 1.3 A → 1.308, 대기 46 mA
    XL430-W250 1.4 N·m / 1.3 A → 1.077, 대기 52 mA.  무부하 전류는 e-manual 표에 없어 대기 전류로 대신 [추정].
  PWM 서보는 버티기만 할 때 듀티가 낮아 배터리 쪽 전류가 이 값보다 작다 → 이 값은 배터리 전류의 상한.
전력 방식(배터리 쪽 추정): P = V·I_대기 + (τ/Kt)²·R + max(0, τ·ω), R = V / I_정지(정지 때 전압이 모두 권선 저항에 걸린다고 봄) [추정 모델].
  배터리 전류 = ΣP / V.  V = 11.1 V.

python hiproll/current_budget.py  → hiproll/results/current_budget.json, current_budget.md, current_bus.png
"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from multiprocessing import Pool
from run_tests import BASE, push_t
from hr_cfg import hr
from kw_common import jdump
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
V = 11.1
EM = {"XM430-W350": (1.810, 2.1, 0.040), "XC430-W240": (1.308, 1.3, 0.046), "XL430-W250": (1.077, 1.3, 0.052)}   # Kt, I_stall, I_sb
LB012 = dict(mass=0.106, pos=(-0.011, 0.0, 0.2431))          # 지금 모터 배터리: 골반 해치 위(설계 상자 가운데) [설계 파일]
BACK = (-0.1085, 0.0, 0.280)                                  # 등 뒤 덮개(x −93) 바깥에 세워 붙인 팩 가운데 [추정 배치]


def batt_move(new_mass, pos=BACK):
    return ((("torso", -LB012["mass"], LB012["pos"]), ("torso", new_mass, pos)))


def scenarios():
    t0 = push_t(6, "ss")
    S = [
        ("서 있기", dict(t_walk=99, T_end=12.0)),
        ("제자리 걸음", dict(stride=0.0)),
        ("걷기 0.076 m/s", dict(stride=0.05)),
        ("걷기 0.083 m/s", dict(stride=0.055)),
        ("끄덕이며 걷기 0.046 m/s · 1 Hz", dict(stride=0.03, nod=(1.0, 3.0))),
        ("끄덕이며 걷기 0.076 m/s · 1 Hz", dict(stride=0.05, nod=(1.0, 3.0))),
        ("고관절 롤 B(XM) 서 있기", dict(t_walk=99, T_end=12.0, hip_roll=hr("B"))),
        ("고관절 롤 B(XM) 걷기 0.076 m/s", dict(stride=0.05, hip_roll=hr("B"))),
        ("고관절 롤 B(XC) 걷기 0.076 m/s", dict(stride=0.05, hip_roll=hr("B", "XC430-W240"))),
        ("고관절 롤 A(XM) 걷기 0.076 m/s", dict(stride=0.05, hip_roll=hr("A"))),
        ("고관절 롤 B(XM) 끄덕이며 걷기 0.076 m/s · 1 Hz", dict(stride=0.05, nod=(1.0, 3.0), hip_roll=hr("B"))),
        ("넘어짐(서 있다 옆 25 N 밀기)", dict(t_walk=99, T_end=10.0, push=(4.0, 0.2, 25.0, "y"), record_fall=True, after_fall_s=2.0)),
        ("넘어짐(걷다 앞 15 N 밀기)", dict(stride=0.03, push=(t0, 0.2, 15.0, "x"), T_end=t0 + 4, record_fall=True, after_fall_s=2.0)),
        ("고관절 롤 B(XM) 넘어짐(서 있다 옆 25 N)", dict(t_walk=99, T_end=10.0, push=(4.0, 0.2, 25.0, "y"), record_fall=True, after_fall_s=2.0, hip_roll=hr("B"))),
    ]
    return S


def one(job):
    from kw_sim import run
    name, cfg = job
    c = dict(BASE); c.update(cfg)
    o, R, m = run(**c)
    names = [m.actuator(i).name for i in range(m.nu)]
    b = np.array([m.dof_damping[m.jnt_dofadr[m.actuator_trnid[i][0]]] for i in range(m.nu)])
    servo = [{3.8: "XM430-W350", 1.75: "XC430-W240", 1.4: "XL430-W250"}[round(float(m.actuator_forcerange[i, 1]), 2)] for i in range(m.nu)]
    tau = R["tau"]; w = R["w_act"]
    load = tau - b * w
    Kt = np.array([EM[s][0] for s in servo]); Ist = np.array([EM[s][1] for s in servo]); Isb = np.array([EM[s][2] for s in servo])
    I = np.minimum(Isb + np.abs(load) / Kt, Ist)
    Rw = V / Ist
    P = V * Isb + (load / Kt) ** 2 * Rw + np.maximum(0.0, load * w)
    t5 = np.arange(len(tau)) * 5e-3
    t_from = 2.3 if c.get("t_walk", 1.5) < 50 else 1.5          # 웅크림 · 첫 무게 옮김 뒤
    if c.get("record_fall"):
        t_from = (o["fall_t"] or 4.0) - 1.0
    sel = t5 >= t_from
    bus = {"TTL1 오른다리": [i for i, n in enumerate(names) if n.startswith("R_") and not n.endswith("arm_swing")],
           "TTL2 왼다리": [i for i, n in enumerate(names) if n.startswith("L_") and not n.endswith("arm_swing")],
           "TTL3 팔 + 머리": [i for i, n in enumerate(names) if n.endswith("arm_swing") or n == "nod_crank"]}
    def mov(x, n):      # n 칸 이동 평균의 최대
        if len(x) < n:
            return float(np.mean(x))
        c_ = np.cumsum(np.insert(x, 0, 0)); return float(((c_[n:] - c_[:-n]) / n).max())
    out = {"name": name, "fallen": o["fallen"], "fall_t": o["fall_t"], "window_s": [float(t_from), float(t5[-1])],
           "servo": {n: {"type": servo[i], "mean_A": float(I[sel, i].mean()), "max_A": float(I[sel, i].max()),
                         "u_p99": float(np.percentile(R["u"][sel, i], 99)), "sat_pct": float(100 * R["sat"][sel, i].mean())} for i, n in enumerate(names)},
           "bus": {}}
    for bn, idx in bus.items():
        s_ = I[sel][:, idx].sum(1)
        out["bus"][bn] = {"mean_A": float(s_.mean()), "max_A": float(s_.max()), "max_1s_A": mov(s_, 200)}
    tot = I[sel].sum(1); Pt = P[sel].sum(1)
    out["total"] = {"mean_A": float(tot.mean()), "max_A": float(tot.max()), "max_1s_A": mov(tot, 200), "max_20ms_A": mov(tot, 4),
                    "power_mean_W": float(Pt.mean()), "power_based_mean_A": float(Pt.mean() / V), "power_based_max_1s_A": mov(Pt / V, 200)}
    out["series"] = {"t": t5[sel][::4].tolist(), "bus": {bn: I[sel][:, idx].sum(1)[::4].tolist() for bn, idx in bus.items()}}
    return out


if __name__ == "__main__":
    with Pool(16) as p:
        rows = p.map(one, scenarios())
    jdump(rows, os.path.join(RES, "current_budget.json"))
    for r in rows:
        print(r["name"], "fall" if r["fallen"] else "", {k: (round(v["mean_A"], 2), round(v["max_A"], 2)) for k, v in r["bus"].items()},
              "total", {k: round(v, 2) for k, v in r["total"].items()})
