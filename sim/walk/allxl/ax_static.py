"""한 발 정지 토크 [정적 계산]: speed/static.py 의 자세 · 토크 계산(pose · solve_H · knee_torque) 그대로, 모델만 ax_common 구성.
웅크림 c 의 대칭 자세에서 몸무게 전부를 한 발이 받칠 때 고관절 피치 · 무릎 · 발목 피치 토크(시상면). 서보 한 개 몫 = 관절 토크/(개수 × 감속비 × 효율).
usage: cd sim/walk/allxl && ../../.venv/bin/python ax_static.py → results/static.json"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ax_common as A
import mujoco
from sp_common import SEG, build_model2
from kw_ctrl import WholeBodyIK
from kw_common import jdump, Idx
from static import pose, solve_H, knee_torque


def model(c):
    seg = c["seg"]; SEG.update(thigh=seg[0], shank=seg[1], mass=True, servo=c["servo_map"])
    kw = {k: c[k] for k in ("hip_roll", "servo_map", "extra_mass", "mass_scale") if k in c}
    m, _ = build_model2(**kw)
    SEG.update(thigh=0.0, shank=0.0, mass=False)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    return m


CASES = {"지금 v20A (XM · XC)": (A.cfg(A.V20A), 0.02), "모두 XL": (A.cfg(), 0.02), "모두 XL, 늘림 +10": (A.cfg(seg=0.01), 0.01),
         "모두 XL, 늘림 0": (A.cfg(seg=0.0), 0.0), "모두 XL, 몸무게 −20 %": (A.cfg(mass_f=0.8), 0.02),
         "모두 XL, 배낭 250 g": (A.cfg(pack_=True), 0.02), "무릎 XL 두 개": (A.cfg(dict(knee="2xXL430e")), 0.02),
         "추천 D6 (무릎 · 롤 XL 두 개, 몸통 · 머리 −300 g)": (A.cfg(dict(knee="2xXL430e", hip_roll="2xXL430e"), mass_scale={"torso": 0.8, "head": 0.8}), 0.02)}

if __name__ == "__main__":
    out = {}
    for name, (c, ext) in CASES.items():
        m = model(c); ix = Idx(m); ik = WholeBodyIK(m); lk = 0.075 + ext
        row = dict(mass_kg=ix.mass)
        for cr in (0.015, 0.025, 0.035):
            a = math.acos((2 * lk - cr) / (2 * lk))
            H = solve_H(m, ix, ik, 2 * a, -a) or -a
            q, com, foot, hh = pose(m, ix, ik, H, 2 * a, -a)
            t = knee_torque(m, ix, ik, q, com, two=False)
            row[f"crouch{int(cr * 1e3)}"] = dict(knee_deg=math.degrees(2 * a), **{k + "_Nm": v for k, v in t.items()})
        out[name] = row
        print(name, f"{ix.mass:.3f} kg", " | ".join(f"{k}: 무릎 {v['knee_Nm']:.2f} 피치 {v['hip_pitch_Nm']:.2f} 발목 {v['ankle_pitch_Nm']:.2f}"
                                                  for k, v in row.items() if k.startswith("crouch")))
    jdump(out, os.path.join(A.RES, "static.json"))
