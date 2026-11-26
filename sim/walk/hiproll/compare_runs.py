"""고관절 롤 배치안 · 배터리 위치별 걷기 · 밀기 비교 (같은 제어기, 이득 2배 · 200 Hz · 지연 10 ms).
python hiproll/compare_runs.py → hiproll/results/compare.json"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from run_tests import BASE, push_t, run_jobs
from hr_cfg import hr
from current_budget import batt_move
from kw_common import jdump
RES = os.path.join(HERE, "results")

MODELS = {"현재(롤 없음)": {}, "A 직렬 XM": dict(hip_roll=hr("A")), "B 축 교차 XM": dict(hip_roll=hr("B")),
          "B 축 교차 XC": dict(hip_roll=hr("B", "XC430-W240"))}
PACKS = {"LB-012 그대로": (), "3S 2200 mAh 180 g 등": batt_move(0.180), "3S 3000 mAh 240 g 등": batt_move(0.240)}


def jobs():
    J = []
    t0 = push_t(6, "ss")
    for mn, mc in MODELS.items():
        for kl in (0.0, 1.0):
            for s in (0.04, 0.05, 0.055, 0.06):
                J.append((dict(model=mn, k_lat=kl, pack="LB-012 그대로", test="speed", stride=s), dict(mc, k_lat=kl, stride=s)))
            for dr in ("+y", "-y", "+x", "-x"):
                for F in (2, 4, 6, 8, 10, 12):
                    J.append((dict(model=mn, k_lat=kl, pack="LB-012 그대로", test="push", dir=dr, F=F),
                              dict(mc, k_lat=kl, push=(t0, 0.2, F * (1 if dr[0] == "+" else -1), dr[1]), T_end=t0 + 4.0)))
    for mn in ("현재(롤 없음)", "B 축 교차 XM"):
        kl = 0.0 if mn.startswith("현재") else 1.0
        for pn, em in PACKS.items():
            if pn.startswith("LB"):
                continue
            mc = dict(MODELS[mn], extra_mass=em, k_lat=kl)
            for s in (0.05, 0.055, 0.06):
                J.append((dict(model=mn, k_lat=kl, pack=pn, test="speed", stride=s), dict(mc, stride=s)))
            for dr in ("+y", "-y", "+x", "-x"):
                for F in (2, 4, 6, 8, 10):
                    J.append((dict(model=mn, k_lat=kl, pack=pn, test="push", dir=dr, F=F),
                              dict(mc, push=(t0, 0.2, F * (1 if dr[0] == "+" else -1), dr[1]), T_end=t0 + 4.0)))
    return J


if __name__ == "__main__":
    t = time.time()
    out = run_jobs(jobs(), procs=40)
    rows = [dict(meta=m_, res={k: v for k, v in r.items()}) for m_, c, r in out]
    jdump(rows, os.path.join(RES, "compare.json"))
    print(len(rows), "runs", sum(1 for r in rows if "error" in r["res"]), "errors", f"{time.time() - t:.0f}s")
