"""걷기 영향: 팔 질량 · 무게중심이 바뀐 만큼을 팔 body 에 점 질량으로 더해(팔꿈치는 0° 로 고정했다고 봄) sim/walk 제어기 그대로 걷기 · 밀기 시험.
다리: 고관절 롤 안 A(40.5 mm) + 허벅지 · 정강이 각 +20 mm(질량 포함), 서보 지금 배치. 제어기 설정은 run_tests.BASE(이득 2배 · 200 Hz · 지연 10 ms).
- 걷기: 한 걸음 (0.45 + 0.20) s · (0.38 + 0.15) s × 보폭 50–110 mm, 18 걸음 끝까지 가면 「안 넘어짐」.
- 밀기: 6번째 걸음 한 발 지지 중 0.2 s 동안 앞(+x) · 옆(+y) 힘 4–14 N.
usage: cd sim/getup && ../.venv/bin/python gu_walk.py → results/walk.json"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import mujoco
from gu_model import ARMS, build, RES, jdump, arm_mass_g
from sp_common import CUR, hrA, run_jobs
from run_tests import push_t

CASES = ["곧은 131 (지금)", "곧은 220", "팔꿈치 110+110", "팔꿈치 140+140"]


def arm_delta(name):
    """원래 팔(v18 메시, XL 로 바꾼 질량) 대비 새 팔의 질량 차이와 그 자리(팔 body 좌표, 오른쪽): 점 질량 하나로 무게중심을 맞춤"""
    def sub(m):
        d = mujoco.MjData(m); d.qpos[:] = m.qpos0; mujoco.mj_forward(m, d)
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "R_arm")
        R = d.xmat[b].reshape(3, 3); return float(m.body_subtreemass[b]), R.T @ (d.subtree_com[b] - d.xpos[b])
    m0, c0 = sub(build(None, kp=0)[0]); m1, c1 = sub(build(ARMS[name], kp=0)[0])
    dm = m1 - m0
    return dm, ((m1 * c1 - m0 * c0) / dm if abs(dm) > 1e-3 else np.zeros(3)), m0, m1


def cfg(name):
    dm, p, _, _ = arm_delta(name)
    em = []
    if abs(dm) > 1e-3:          # 1 g 보다 작은 차이는 무시(지금 팔)
        em = [("R_arm", dm, tuple(p)), ("L_arm", dm, (p[0], -p[1], p[2]))]
    return dict(hip_roll=hrA(0.0405), servo_map=CUR, seg=(0.02, 0.02, True), extra_mass=em)


if __name__ == "__main__":
    jobs = []
    info = {}
    for n in CASES:
        dm, p, m0, m1 = arm_delta(n); info[n] = dict(dm_g=dm * 1e3, pos_mm=(p * 1e3).tolist(), arm_g=m1 * 1e3, base_arm_g=m0 * 1e3)
        c = cfg(n)
        for Ts, Td in ((0.45, 0.20), (0.38, 0.15)):
            for s in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11):
                jobs.append((dict(arm=n, test="walk", Ts=Ts, Td=Td, stride=s), dict(c, Ts=Ts, Td=Td, stride=s)))
        for dr in ("+x", "+y"):
            for F in (4, 6, 8, 10, 12, 14):
                t0 = push_t(6, "ss")
                jobs.append((dict(arm=n, test="push", dir=dr, F=F), dict(c, push=(t0, 0.2, F, dr[1]), T_end=t0 + 4.0)))
    print(info, len(jobs), flush=True)
    rows = run_jobs(jobs, procs=32)
    out = [dict(meta=mt, fallen=o.get("fallen"), error=o.get("error"), cop_min=(o.get("cop_margin_mm") or {}).get("min"), mass_kg=o.get("mass_kg"))
           for mt, c, o in rows]
    jdump(dict(info=info, rows=out), os.path.join(RES, "walk.json"))
    for n in CASES:
        ok_w = {(r["meta"]["Ts"], r["meta"]["stride"]): not r["fallen"] for r in out if r["meta"]["arm"] == n and r["meta"]["test"] == "walk"}
        mx = {ts: max([s for (t, s), v in ok_w.items() if t == ts and v], default=0) for ts in (0.45, 0.38)}
        pp = {dr: max([r["meta"]["F"] for r in out if r["meta"]["arm"] == n and r["meta"]["test"] == "push" and r["meta"]["dir"] == dr and not r["fallen"]], default=0) for dr in ("+x", "+y")}
        print(n, "최대 보폭", mx, "버틴 밀기 N", pp, "err", sum(1 for r in out if r["meta"]["arm"] == n and r["error"]))
