"""질문 2 · 3: 다리 늘림 길이 · 서보 변형별로 걸음 빠르기(한 걸음 시간) × 보폭을 넓게 돌려 넘어지지 않는 최고 속도 · 최대 보폭,
압력중심 여유, 밀기 버팀, 관절 사용률(토크 몫 · 속도 몫)을 잰다. 제어기는 sim/walk 그대로(이득 2배 · 200 Hz · 지연 10 ms).

usage: cd sim/walk/speed && ../../.venv/bin/python walk_sweep.py [legs servo push] → results/<묶음>.json"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sp_common import CUR, hrA, run_jobs
from kw_common import jdump
from run_tests import push_t, BASE

CAD = [(0.45, 0.20), (0.38, 0.15), (0.32, 0.12), (0.28, 0.10), (0.24, 0.08)]       # (한 발 지지 s, 두 발 지지 s)
STRIDES = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
LEGS = {"늘림 0 mm": dict(hip_roll=hrA(0.0), servo_map=CUR),
        "늘림 20 mm": dict(hip_roll=hrA(0.020), servo_map=CUR),
        "늘림 40.5 mm (안 A)": dict(hip_roll=hrA(0.0405), servo_map=CUR),
        "참고: 늘림 0 + 허벅지 · 정강이 각 20 mm": dict(hip_roll=hrA(0.0), servo_map=CUR, seg=(0.02, 0.02))}
W210 = "XM430-W210"
SERVOS = {"무릎만 W210": dict(knee=W210),
          "고관절 피치 · 무릎 W210": dict(hip_pitch=W210, knee=W210),
          "다리 XM 모두 W210": dict(hip_pitch=W210, knee=W210, ankle_roll=W210, _roll=W210),
          "발목 피치 XC430-W150": dict(ankle_pitch="XC430-W150"),
          "피치 · 무릎 W210 + 발목 피치 W150": dict(hip_pitch=W210, knee=W210, ankle_pitch="XC430-W150"),
          "발목 피치 XM430-W350": dict(ankle_pitch="XM430-W350"),
          "피치 · 무릎 W210 + 발목 피치 XM430-W210": dict(hip_pitch=W210, knee=W210, ankle_pitch=W210)}


def servo_cfg(sv, L=0.0405):
    sv = dict(sv); roll = sv.pop("_roll", "XM430-W350")
    return dict(hip_roll=hrA(L, roll), servo_map=dict(CUR, **sv))


def g_legs():
    return [(dict(cfg=n, Ts=a, Td=b, stride=s), dict(c, Ts=a, Td=b, stride=s)) for n, c in LEGS.items() for a, b in CAD for s in STRIDES]


def g_servo():
    return [(dict(cfg=n, Ts=a, Td=b, stride=s), dict(servo_cfg(sv), Ts=a, Td=b, stride=s))
            for n, sv in SERVOS.items() for a, b in CAD for s in STRIDES]


def g_push():
    J = []
    for n, c in LEGS.items():
        for dr in ("+y", "-y", "+x", "-x"):
            sg = 1 if dr[0] == "+" else -1
            for F in (2, 4, 6, 8, 10, 12, 15):
                t0 = push_t(6, "ss")
                J.append((dict(cfg=n, where="ss", dir=dr, F=F), dict(c, push=(t0, 0.2, F * sg, dr[1]), T_end=t0 + 4.0)))
                J.append((dict(cfg=n, where="stand", dir=dr, F=F), dict(c, t_walk=99, T_end=10.5, push=(4.0, 0.2, F * sg, dr[1]))))
        J.append((dict(cfg=n, where="stand", dir="none", F=0), dict(c, t_walk=99, T_end=10.0)))
    return J


# 경계 다시 재기: 넘어짐 경계가 들쭉날쭉해서(한 점 한 번) 경계 근처를 촘촘히, 마찰 0.7 · 0.8 · 0.9 세 번씩 돌려 통과 비율로 본다
CAD_R = [(0.45, 0.20), (0.38, 0.15), (0.32, 0.12)]
STR_R = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
FRIC = (0.7, 0.8, 0.9)


def g_legs_rep():
    return [(dict(cfg=n, Ts=a, Td=b, stride=s, fr=f), dict(c, Ts=a, Td=b, stride=s, friction=f))
            for n, c in LEGS.items() for a, b in CAD_R for s in STR_R for f in FRIC]


def g_servo_rep():
    cfgs = dict({"지금 (늘림 40.5 mm)": dict(hip_roll=hrA(0.0405), servo_map=CUR)}, **{n: servo_cfg(sv) for n, sv in SERVOS.items()})
    return [(dict(cfg=n, Ts=a, Td=b, stride=s, fr=f), dict(c, Ts=a, Td=b, stride=s, friction=f))
            for n, c in cfgs.items() for a, b in CAD_R + [(0.28, 0.10)] for s in STR_R for f in FRIC]


# 허벅지 · 정강이 늘림(질량 반영) × 롤-피치 늘림 0 · 20 mm × 웅크림 15 · 25 · 35 mm (코디네이터 추가 요청)
SEGS = (0.0, 0.01, 0.02, 0.03)
STR_S = [0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10]


def seg_cfgs():
    return {(L, e, cr): dict(hip_roll=hrA(L), servo_map=CUR, seg=(e, e, True), crouch=cr) for L in (0.0, 0.02) for e in SEGS for cr in (0.015, 0.025, 0.035)}


def g_seg():
    return [(dict(L=L, ext=e, crouch=cr, Ts=a, Td=b, stride=s, fr=f), dict(c, Ts=a, Td=b, stride=s, friction=f))
            for (L, e, cr), c in seg_cfgs().items() for a, b in [(0.45, 0.20), (0.38, 0.15)] for s in STR_S for f in FRIC]


def g_seg_push():
    J = []
    for (L, e, cr), c in seg_cfgs().items():
        for dr in ("+y", "+x"):
            for F in (4, 6, 8, 10, 12):
                t0 = push_t(6, "ss")
                J.append((dict(L=L, ext=e, crouch=cr, where="ss", dir=dr, F=F), dict(c, push=(t0, 0.2, F, dr[1]), T_end=t0 + 4.0)))
                J.append((dict(L=L, ext=e, crouch=cr, where="stand", dir=dr, F=F), dict(c, t_walk=99, T_end=10.5, push=(4.0, 0.2, F, dr[1]))))
    return J


def g_combo():
    """추천 조합을 함께: 고관절 피치 · 무릎 W210 + 발목 피치 W150 + 허벅지 · 정강이 각 +20 mm(질량 반영) + 웅크림 35 mm, 롤-피치 20 mm.
    비교로 같은 다리에 지금 서보."""
    sv = dict(CUR, hip_pitch=W210, knee=W210, ankle_pitch="XC430-W150")
    cf = {"추천 조합 (W210 · W150 + 다리 +20/+20 + 웅크림 35)": dict(hip_roll=hrA(0.02), servo_map=sv, seg=(0.02, 0.02, True), crouch=0.035),
          "같은 다리 + 지금 서보": dict(hip_roll=hrA(0.02), servo_map=CUR, seg=(0.02, 0.02, True), crouch=0.035)}
    return [(dict(cfg=n, Ts=a, Td=b, stride=s, fr=f), dict(c, Ts=a, Td=b, stride=s, friction=f))
            for n, c in cf.items() for a, b in CAD_R + [(0.28, 0.10)] for s in STR_S for f in FRIC]


G = dict(combo=g_combo, seg=g_seg, seg_push=g_seg_push, legs=g_legs, servo=g_servo, push=g_push, legs_rep=g_legs_rep, servo_rep=g_servo_rep)
if __name__ == "__main__":
    for g in sys.argv[1:] or list(G):
        t = time.time(); jobs = G[g]()
        out = run_jobs(jobs)
        rows = [dict(meta=m_, cfg={k: v for k, v in c.items() if k not in ("video", "eye_video")}, res=r) for m_, c, r in out]
        jdump(dict(base=BASE, rows=rows), os.path.join(HERE, "results", f"{g}.json"))
        print(f"{g}: {len(rows)} runs, {sum(1 for r in rows if r['res'].get('fallen'))} fell, "
              f"{sum(1 for r in rows if 'error' in r['res'])} errors, {time.time() - t:.0f}s", flush=True)
