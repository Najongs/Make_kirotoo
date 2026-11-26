"""Final utilisation table: handover / final_v17.json vs MuJoCo (tests 1, 2). Envelope = walk + fast + static
(crouch, toe, heel), as servo_opt.score (jog excluded, shown apart). servo_study-only hip_strategy is listed apart.
Run after test1 and test2: cd sim && uv run python tests/summary_table.py -> results/summary_utilisation.json, .md
"""
import json, numpy as np
from common import *
import servo_opt as O

t1 = json.load(open(os.path.join(RES, "test1_static_hold.json")))["cases"]
t2_all = json.load(open(os.path.join(RES, "test2_gait_torque.json"))); t2 = t2_all["cases"]
final = json.load(open(os.path.join(CODE, "final_v17.json")))
HANDOVER_TAU = {"hip_yaw": 0.07, "hip_pitch": 1.51, "knee": 1.32, "ankle_pitch": 1.43, "ankle_roll": 1.21}
HANDOVER_U = {"hip_yaw": 0.04, "hip_pitch": 0.67, "knee": 0.76, "ankle_pitch": 0.81, "ankle_roll": 0.46}
cases, _ = tuned_cases()
for nm in S.CASES:
    S.CASES[nm]["psi_amp"] = cases[nm]["psi_amp"]
Mopt = ss_model(True); res, rms, pc = O.collect(Mopt)
assign = {j: SERVO_OF[j] for j in LEGJ}
u_ss, _ = O.score(res, rms, assign)
u_ss_nohip = {}
for j in LEGJ:
    ts, w0 = S.SPEC[assign[j]]
    u_ss_nohip[j] = max(float(np.max(np.abs(t) / ts + np.abs(w) / w0)) for t, w, nm in res[j] if nm not in ("jog", "hip_strategy"))
hs = t2_all["hip_strategy"]
Z = np.load(os.path.join(RES, "test2_series.npz"))
m0, _ = load(); ARM = {j: float(m0.dof_armature[va(m0, f"R_{j}")]) for j in LEGJ}


def u_scale(j, sc, cases=("walk", "fast")):
    """u of joint j over walk/fast (stance R, swing L) with the armature scaled by sc (u is linear in it)"""
    st, w0 = SPEC_MODEL[SERVO_OF[j]]; u = 0.0
    for c in cases:
        for side in "RL":
            dof = va(m0, f"{side}_{j}")
            tr, ta, qv = Z[f"{c}_tau_rigid"][:, dof], Z[f"{c}_tau_armature"][:, dof], Z[f"{c}_qvel"][:, dof]
            u = max(u, float(np.max(np.abs(tr + sc * (ta - tr)) / st + np.abs(qv) / w0)))
    if j == "hip_pitch":
        tr, ta, qv = Z["hip_strategy_tau_rigid"][:, 0], Z["hip_strategy_tau_armature"][:, 0], Z["hip_strategy_qvel"][:, 0]
        u = max(u, float(np.max(np.abs(tr + sc * (ta - tr)) / st + np.abs(qv) / w0)))
    return u


def breakeven(j, cases=("walk", "fast")):
    if u_scale(j, 0, cases) >= 1: return 0.0
    if u_scale(j, 100, cases) < 1: return None
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if u_scale(j, mid, cases) < 1 else (lo, mid)
    return lo * ARM[j]


rows = {}
for j in LEGJ:
    st, w0 = SPEC_MODEL[SERVO_OF[j]]; ste, w0e = SPEC_EMAN[SERVO_OF[j]]
    g = [t2[c]["joints"][f"{role}_{j}"] for c in ("walk", "fast") for role in ("stance", "swing")]
    jg = [t2["jog"]["joints"][f"{role}_{j}"] for role in ("stance", "swing")]
    stat = [t1[c]["joint_torque"][j]["mj"] for c in ("crouch", "toe", "heel")]
    if j == "hip_pitch":
        g = g + [hs]
    tau = max([x["mj_tau_max"] for x in g] + stat); w = max(x["mj_w_max"] for x in g)
    rows[j] = {"servo": SERVO_OF[j],
               "handover_tau_max": HANDOVER_TAU[j], "final_v17_u": final["u"][j], "servo_study_rerun_u": u_ss[j],
               "servo_study_rerun_u_without_hip_strategy": u_ss_nohip[j],
               "mj_tau_max": tau, "mj_w_max": w,
               "mj_u": max([x["mj_u"] for x in g] + [s / st for s in stat]),
               "mj_u_emanual": max([x["mj_u_emanual"] for x in g] + [s / ste for s in stat]),
               "mj_u_with_armature": max([x["mj_u_with_armature"] for x in g] + [s / st for s in stat]),
               "mj_u_with_armature_emanual": max([x["mj_u_with_armature_emanual"] for x in g] + [s / ste for s in stat]),
               "mj_jog_u": max(x["mj_u"] for x in jg), "mj_jog_u_with_armature": max(x["mj_u_with_armature"] for x in jg),
               "hip_strategy_tau_servo_study": pc["hip_strategy"]["stance_hip_pitch"][0] if j == "hip_pitch" else None,
               "armature_model": ARM[j], "u_armature_scale": {str(sc): u_scale(j, sc) for sc in (0, 0.25, 0.5, 1.0)},
               "armature_for_u1_walk_fast": breakeven(j), "armature_for_u1_walk_only": breakeven(j, ("walk",))}
jdump(rows, "summary_utilisation.json")
KR = {"hip_yaw": "고관절 요", "hip_pitch": "고관절 피치", "knee": "무릎", "ankle_pitch": "발목 피치", "ankle_roll": "발목 롤"}
lines = ["| 관절 (서보) | 인계 최대 토크 | final_v17 u | MuJoCo 최대 토크 | MuJoCo 최대 속도 | MuJoCo u (인계 표 값) | MuJoCo u (e-manual 값) | + 회전자 관성 u (인계 표) | + 회전자 관성 u (e-manual) | 빠른 종종걸음(jog) u / + 관성 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for j, r in rows.items():
    lines.append(f"| {KR[j]} ({r['servo'].split('-')[0]}) | {r['handover_tau_max']:.2f} | {r['final_v17_u']:.2f} | {r['mj_tau_max']:.2f} | {r['mj_w_max']:.2f} | "
                 f"{r['mj_u']:.2f} | {r['mj_u_emanual']:.2f} | {r['mj_u_with_armature']:.2f} | {r['mj_u_with_armature_emanual']:.2f} | {r['mj_jog_u']:.2f} / {r['mj_jog_u_with_armature']:.2f} |")
lines += ["", "| 관절 | 모델 armature (kg·m²) | u: 관성 0 / ×0.25 / ×0.5 / ×1 | u=1 이 되는 armature (walk+fast) | u=1 이 되는 armature (walk 만) |", "|---|---|---|---|---|"]
for j, r in rows.items():
    us = r["u_armature_scale"]; f = lambda v: "넘지 않음" if v is None else f"{v:.4f}"
    lines.append(f"| {KR[j]} | {r['armature_model']:.4f} | {us['0']:.2f} / {us['0.25']:.2f} / {us['0.5']:.2f} / {us['1.0']:.2f} | {f(r['armature_for_u1_walk_fast'])} | {f(r['armature_for_u1_walk_only'])} |")
open(os.path.join(RES, "summary_utilisation.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print({j: (round(r["servo_study_rerun_u"], 3), round(r["servo_study_rerun_u_without_hip_strategy"], 3)) for j, r in rows.items()})
