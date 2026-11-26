"""시험 묶음 실행: python run_tests.py [speed nod push turn variants ...]  → results/<묶음>.json
기본 제어기 설정은 BASE. 각 실행은 독립(병렬)."""
import sys, json, math, os, time
from multiprocessing import Pool
import numpy as np
from kw_common import RES, jdump

BASE = dict(n_steps=18, Ts=0.45, Td=0.2, crouch=0.025, lift=0.012, k_dcm=1.0, kp_scale=2.0, ctrl_hz=200, delay_ms=10.0)
T_STEP = BASE["Ts"] + BASE["Td"]
T_GAIT0 = 1.5 + 0.8          # t_walk + Tinit: 첫 한 발 지지 시작


def one(job):
    from kw_sim import run
    meta, cfg = job
    c = dict(BASE); c.update(cfg)
    try:
        o, R, m = run(**c)
    except Exception as e:
        return meta, c, {"error": repr(e)}
    return meta, c, o


def run_jobs(jobs, procs=40):
    with Pool(min(procs, len(jobs))) as p:
        return p.map(one, jobs)


def push_t(k=6, where="ss"):
    base = T_GAIT0 + k * T_STEP
    return base + (0.125 if where == "ss" else BASE["Ts"] - 0.05)


def g_speed():
    jobs = []
    for s in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.055, 0.06):
        for kp in (1.0, 2.0, 4.0):
            for hz, dl in ((200, 0.0), (200, 10.0), (100, 0.0), (100, 10.0)):
                jobs.append((dict(stride=s, kp=kp, hz=hz, delay=dl), dict(stride=s, kp_scale=kp, ctrl_hz=hz, delay_ms=dl)))
    return jobs


def g_nod():
    jobs = []
    for s in (0.0, 0.03, 0.05):
        for f in (None, 0.5, 1.0):
            jobs.append((dict(stride=s, nod=f), dict(stride=s, nod=(f, 3.0) if f else None)))
    return jobs


MAGS = (2, 4, 6, 8, 10, 12, 15)


def g_push():
    jobs = []
    for fr in (0.5, 0.8, 1.0):
        for dr in ("+y", "-y", "+x", "-x"):
            for wh in ("ss", "ds"):
                for F in MAGS:
                    t0 = push_t(6, wh)
                    jobs.append((dict(friction=fr, dir=dr, where=wh, F=F),
                                 dict(friction=fr, push=(t0, 0.2, F * (1 if dr[0] == "+" else -1), dr[1]), T_end=t0 + 4.0)))
    # 서 있기(걷지 않음)
    for dr in ("+y", "-y", "+x", "-x"):
        for F in MAGS:
            jobs.append((dict(friction=0.8, dir=dr, where="stand", F=F),
                         dict(t_walk=99, T_end=10.5, push=(4.0, 0.2, F * (1 if dr[0] == "+" else -1), dr[1]))))
    # 마찰만 바꿔 걷기
    for fr in (0.5, 0.8, 1.0):
        for s in (0.03, 0.05):
            jobs.append((dict(friction=fr, stride=s, walk=True), dict(friction=fr, stride=s)))
    return jobs


def g_turn():
    return [(dict(yaw_step_deg=a), dict(yaw_step=math.radians(a))) for a in (10, 15, 20, 25, 30)]


VARIANTS = {
    "기준": {},
    "머리 질량 −20%": dict(head_mass=0.8),
    "바깥 패드 6 mm 바깥으로": dict(pad_dy_out=0.006),
    "바깥 패드 10 mm 바깥으로": dict(pad_dy_out=0.010),
    "앞뒤 패드 10 mm 씩 벌림": dict(pad_dx=0.010),
    "회전자 관성 ×0.5": dict(armature=0.5),
    "위치 이득 ×1": dict(kp_scale=1.0),
    "위치 이득 ×4": dict(kp_scale=4.0),
    "고관절 롤 추가": dict(hip_roll=True),
}


def g_variants():
    jobs = []
    for vn, vc in VARIANTS.items():
        for s in (0.04, 0.05, 0.055, 0.06, 0.07):
            jobs.append((dict(variant=vn, test="speed", stride=s), dict(vc, stride=s)))
        for dr in ("+y", "-y"):
            for F in (2, 4, 6, 8, 10, 12):
                t0 = push_t(6, "ss")
                jobs.append((dict(variant=vn, test="push", dir=dr, F=F),
                             dict(vc, push=(t0, 0.2, F * (1 if dr[0] == "+" else -1), "y"), T_end=t0 + 4.0)))
        jobs.append((dict(variant=vn, test="nod"), dict(vc, stride=0.03, nod=(1.0, 3.0))))
    return jobs


def g_push_kdcm():
    """DCM 되먹임 이득별 밀기(한 발 지지 중, 마찰 0.8)"""
    jobs = []
    for k in (0.0, 1.0, 2.0, 3.0):
        for dr in ("+y", "-y", "+x", "-x"):
            for F in (2, 4, 6, 8, 10, 12):
                t0 = push_t(6, "ss")
                jobs.append((dict(k_dcm=k, dir=dr, F=F), dict(k_dcm=k, push=(t0, 0.2, F * (1 if dr[0] == "+" else -1), dr[1]), T_end=t0 + 4.0)))
    return jobs


GROUPS = dict(speed=g_speed, nod=g_nod, push=g_push, turn=g_turn, variants=g_variants, push_kdcm=g_push_kdcm)

if __name__ == "__main__":
    names = sys.argv[1:] or list(GROUPS)
    for g in names:
        t = time.time()
        jobs = GROUPS[g]()
        out = run_jobs(jobs)
        rows = [dict(meta=m_, cfg={k: v for k, v in c.items() if k not in ("video", "eye_video")}, res=r) for m_, c, r in out]
        jdump(dict(base=BASE, rows=rows), os.path.join(RES, f"{g}.json"))
        nf = sum(1 for r in rows if r["res"].get("fallen"))
        ne = sum(1 for r in rows if "error" in r["res"])
        print(f"{g}: {len(rows)} runs, {nf} fell, {ne} errors, {time.time() - t:.0f}s", flush=True)
