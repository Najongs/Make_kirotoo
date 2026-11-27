"""Nod crank servo torque again with the v22 masses (2026-11-28: head shell walls 2.0, screws, crank on the XL430
horn -> head 390.6 -> 509.8 g). Same calculation as head21.py (virtual work on the closed loop: head + ears + crank,
plus the 1 Hz -20 .. +28 deg nod inertia term); head21.py and its constants stay as they are (the v21 doc uses them).

Run:  cd code_v17 && ../sim/.venv/bin/python head22_torque.py
Out:  stdout + out_head21/head22_torque.json  (written into out_head21/HEAD_CRANK_v21.md, section 2026-11-28)
Masses: before = out21_arm/mass_build21_arm_light.json (= out20/link_mass_v20.csv head / ears / crank, the v21 model),
        after  = sim/model/mass_kitoru_v22_light.json (sim/tools/mass_v22.py). CoM from the link origin (head: nod axis,
        ears: ear axle, crank: horn axis), x / z in the head plane, Iyy about the CoM.
"""
import os, json, math
import numpy as np
import head21 as H

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {"before (v21 model)": os.path.join(HERE, "..", "out21_arm", "mass_build21_arm_light.json"),
       "after (v22 model)": os.path.join(HERE, "..", "sim", "model", "mass_kitoru_v22_light.json")}
BETAS = (H.NEW_RANGE[0], -10.0, 0.0, 10.0, 20.0, H.NEW_RANGE[1])


def run(path):
    L = json.load(open(path))["links"]
    c = lambda k: (L[k]["com_local"][0], L[k]["com_local"][2])
    H.M_HEAD, H.C_HEAD, H.IYY_HEAD = L["head"]["mass_g"] * 1e-3, c("head"), L["head"]["inertia_gmm2"][1][1] * 1e-9
    H.M_EARS, H.C_EARS, H.IYY_EARS = L["ears"]["mass_g"] * 1e-3, c("ears"), L["ears"]["inertia_gmm2"][1][1] * 1e-9
    H.M_CRANK, H.C_CRANK = L["nod_crank"]["mass_g"] * 1e-3, c("nod_crank")
    tq = {b: H.torque(b) for b in BETAS}
    I = H.inertia_about_axis()
    amp = math.radians((H.NEW_RANGE[1] - H.NEW_RANGE[0]) / 2); acc = amp * (2 * math.pi * H.NOD_HZ) ** 2
    dyn = {b: abs(t[0]) + I * acc * abs(t[1]) for b, t in tq.items() if b in H.NEW_RANGE}
    return dict(masses_g=dict(head=round(H.M_HEAD * 1e3, 1), ears=round(H.M_EARS * 1e3, 1), crank=round(H.M_CRANK * 1e3, 1)),
                com_head_mm=[round(x, 2) for x in H.C_HEAD], com_ears_mm=[round(x, 2) for x in H.C_EARS],
                inertia_axis_kgm2=I, peak_acc_rad_s2=acc,
                static={str(b): dict(Nm=abs(t[0]), of_stall=abs(t[0]) / H.STALL, dbeta_dalpha=t[1]) for b, t in tq.items()},
                nod_1hz={str(b): dict(Nm=v, of_stall=v / H.STALL, margin=H.STALL / v) for b, v in dyn.items()})


if __name__ == "__main__":
    out = {k: run(p) for k, p in SRC.items()}
    out["stall_Nm"] = H.STALL
    json.dump(out, open(os.path.join(H.OUT, "head22_torque.json"), "w"), indent=1)
    for k in SRC:
        r = out[k]
        print(k, r["masses_g"], "head CoM", r["com_head_mm"], f"I_axis {r['inertia_axis_kgm2']:.5f}")
        for b, t in r["static"].items():
            print(f"   static beta {b:>6s}: {t['Nm']:.3f} N m ({100 * t['of_stall']:.1f} % of {H.STALL})")
        for b, t in r["nod_1hz"].items():
            print(f"   1 Hz nod at {b:>6s}: {t['Nm']:.3f} N m ({100 * t['of_stall']:.1f} %, stall / torque {t['margin']:.1f}x)")
