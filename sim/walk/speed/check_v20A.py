"""v20 안 A 키트 형상 그대로 걷기 확인: 롤-피치 53.58 mm, 다리 늘림 40.13 mm(롤-피치, U 바닥판 3 mm 포함), 허벅지 · 정강이 각 +20 mm(질량 넣음),
웅크림 35 mm, 롤 서보 무게중심 요 링크 원점 아래 26.55 mm. 제어기 · 시험은 walk_sweep.py 와 같음.
usage: cd sim/walk/speed && ../../.venv/bin/python check_v20A.py -> results/check_v20A.json"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sp_common import CUR, run_jobs
from kw_common import jdump
from run_tests import push_t
from walk_sweep import CAD, STRIDES
HR = dict(servo="XM430-W350", d_axis=0.05358, lengthen=0.04013, mass=0.082, mass_pos=(0.0, 0.0, -0.02655))
C0 = dict(hip_roll=HR, servo_map=CUR, seg=(0.02, 0.02, True), crouch=0.035)
J = [(dict(test="walk", Ts=a, Td=b, stride=s), dict(C0, Ts=a, Td=b, stride=s)) for a, b in CAD for s in STRIDES]
for dr in ("+y", "-y", "+x", "-x"):
    for F in (2, 4, 6, 8, 10, 12):
        t0 = push_t(6, "ss")
        J.append((dict(test="push", dir=dr, F=F), dict(C0, push=(t0, 0.2, F * (1 if dr[0] == "+" else -1), dr[1]), T_end=t0 + 4.0)))
if __name__ == "__main__":
    t = time.time()
    out = run_jobs(J, procs=30)
    rows = [dict(meta=m, res=r) for m, c, r in out]
    jdump(rows, os.path.join(HERE, "results", "check_v20A.json"))
    ok = [r for r in rows if r["meta"]["test"] == "walk" and "error" not in r["res"] and not r["res"]["fallen"]]
    best = max(ok, key=lambda r: r["meta"]["stride"] / (r["meta"]["Ts"] + r["meta"]["Td"])) if ok else None
    if best:
        m = best["meta"]; print("max speed", round(m["stride"] / (m["Ts"] + m["Td"]), 3), m, "cop p5", best["res"]["cop_margin_mm"]["p5"],
                                 "knee u99", best["res"]["split"].get("knee", {}).get("u_p99"), "roll u99", best["res"]["split"].get("hip_roll", {}).get("u_p99"))
    for dr in ("-y", "+y", "+x", "-x"):
        pr = sorted([r for r in rows if r["meta"]["test"] == "push" and r["meta"]["dir"] == dr], key=lambda r: r["meta"]["F"])
        print("push", dr, max([r["meta"]["F"] for r in pr if "error" not in r["res"] and not r["res"]["fallen"]] or [0]))
    print(len(rows), "runs", sum(1 for r in rows if "error" in r["res"]), "errors", f"{time.time() - t:.0f}s")
