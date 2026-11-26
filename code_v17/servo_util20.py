"""servo use (u) for the v20 servo set with servo_study / servo_opt (read only).
servo_study has no hip roll: the hip roll link is folded into the hip yaw link (rigid at roll 0).
XL430 hip yaw, XM hip pitch / knee / ankle roll, XC ankle pitch. Output: final_v20.json"""
import json, sys, numpy as np
import servo_study as S, servo_opt as O
mass = json.load(open("mass_build20_light.json"))
for side in "RL":
    if f"{side}_hip_roll_link" in mass["links"]:
        S.Model._merge(mass["links"], f"{side}_hip_yaw_link", f"{side}_hip_roll_link")
M = S.Model(mass)
for nm, c in S.CASES.items():
    deg, err = S.tune_psi(M, c); S.CASES[nm]["psi_amp"] = np.radians(deg)
assign = {"hip_yaw": "XL430-W250", "hip_pitch": "XM430-W350", "knee": "XM430-W350", "ankle_pitch": "XC430-W240", "ankle_roll": "XM430-W350"}
res, rms, pc = O.collect(M)
u, heat = O.score(res, rms, assign)
uj, _ = O.score(res, rms, assign, ["jog"])
out = dict(mass=M.total(), assign=assign, u=u, u_jog=uj, heat=heat, psi_deg={k: float(np.degrees(v["psi_amp"])) for k, v in S.CASES.items()},
           note="servo_study model, hip roll folded into the hip yaw link; XL430 no-load 52 rpm in servo_study SPEC")
json.dump(out, open("final_v20.json", "w"), indent=1)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in u.items()}), "jog", {k: round(v, 3) for k, v in uj.items()}, "mass", round(M.total(), 1))
