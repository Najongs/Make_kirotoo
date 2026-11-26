"""모든 관절 XL430 모델로 피드백 걷기(MuJoCo, 제어기는 sim/walk 그대로). kp 는 같은 환산식(XL 6.4 N·m/rad × 이득 배수),
forcerange 1.4, damping 1.4/5.45, armature 0.0241 [추정], 서보 질량 차이 반영(kw_common servo_map).
python xl430/walk_xl.py → xl430/results/walk_xl.json"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hiproll"))
from run_tests import BASE, run_jobs
from hr_cfg import hr
from kw_common import jdump
LEG = ("hip_yaw", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
XL = {k: "XL430-W250" for k in LEG + ("arm_swing",)}


def xl(**over):
    d = dict(XL); d.update(over); return d


MODELS = {
    "v18 지금(XM 3 + XC 2)": {},
    "모두 XL430": dict(servo_map=XL),
    "모두 XL430, 몸 −10 %": dict(servo_map=XL, mass_scale={"": 0.9}),
    "모두 XL430, 몸 −20 %": dict(servo_map=XL, mass_scale={"": 0.8}),
    "모두 XL430, 몸 −30 %": dict(servo_map=XL, mass_scale={"": 0.7}),
    "XL + 무릎 XC": dict(servo_map=xl(knee="XC430-W240")),
    "XL + 무릎 · 발목 피치 XC": dict(servo_map=xl(knee="XC430-W240", ankle_pitch="XC430-W240")),
    "XL + 무릎 · 발목 피치 · 발목 롤 XC": dict(servo_map=xl(knee="XC430-W240", ankle_pitch="XC430-W240", ankle_roll="XC430-W240")),
    "XL + 무릎 XM": dict(servo_map=xl(knee="XM430-W350")),
    "XL + 무릎 · 발목 롤 XM": dict(servo_map=xl(knee="XM430-W350", ankle_roll="XM430-W350")),
    "XL, 무릎 XL 두 개": dict(servo_map=xl(knee="2xXL430")),
    "XL, 무릎 · 발목 피치 XL 두 개": dict(servo_map=xl(knee="2xXL430", ankle_pitch="2xXL430")),
    "XL, 무릎 · 발목 피치 · 발목 롤 XL 두 개": dict(servo_map=xl(knee="2xXL430", ankle_pitch="2xXL430", ankle_roll="2xXL430")),
    "모두 XL + 고관절 롤 B(XL)": dict(servo_map=XL, hip_roll=hr("B", "XL430-W250")),
}
TESTS = [("서기", dict(t_walk=99, T_end=12.0)), ("제자리", dict(stride=0.0))] + \
        [(f"보폭 {int(s * 1000)}", dict(stride=s)) for s in (0.01, 0.02, 0.03, 0.04, 0.05)]


def jobs():
    J = []
    for mn, mc in MODELS.items():
        for kp in (2.0, 4.0):
            for crouch in (0.025, 0.015):
                if crouch == 0.015 and not mn.startswith("모두 XL430") and mn != "v18 지금(XM 3 + XC 2)":
                    continue
                for tn, tc in TESTS:
                    J.append((dict(model=mn, kp=kp, crouch=crouch, test=tn), dict(mc, kp_scale=kp, crouch=crouch, **tc)))
    return J


if __name__ == "__main__":
    t = time.time()
    out = run_jobs(jobs(), procs=48)
    rows = [dict(meta=m_, res=r) for m_, c, r in out]
    jdump(rows, os.path.join(HERE, "results", "walk_xl.json"))
    print(len(rows), "runs", f"{time.time() - t:.0f}s")
