"""v20 안 A 키트(check_v20A.py 와 같은 형상)에서 발판을 바깥으로 e 만큼 옮겼을 때(패드 다섯 개 모두 같이 옮김) 걷기 · 밀기.
고관절 롤 안쪽 범위를 넓히려는 형상 후보의 균형 영향 확인. usage: ../../.venv/bin/python check_v20A_foot.py -> results/check_v20A_foot.json"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import sp_common as SP
import kw_sim
from kw_common import jdump
from run_tests import push_t
from walk_sweep import CAD, STRIDES
from check_v20A import C0
_bm = kw_sim.build_model
FOOT = {"e": 0.0}


def build_model3(*a, **kw):
    e = FOOT["e"]
    if e:
        prev = kw.get("extra_bodies")
        def shift(spec):
            prev and prev(spec)
            for s, sg in (("R", -1.0), ("L", 1.0)):
                for g in spec.geoms:
                    if g.name and g.name.startswith(f"{s}_foot_pad"):
                        p = np.array(g.pos); p[1] += sg * e; g.pos = p
        kw["extra_bodies"] = shift
    return _bm(*a, **kw)


kw_sim.build_model = build_model3


def one(job):
    meta, cfg = job
    FOOT["e"] = cfg.pop("foot_out", 0.0)
    try:
        return SP.one((meta, cfg))
    finally:
        FOOT["e"] = 0.0


J = []
for e in (0.0, 0.010, 0.015, 0.020):
    C = dict(C0, foot_out=e)
    J += [(dict(e=e, test="walk", Ts=a, Td=b, stride=s), dict(C, Ts=a, Td=b, stride=s)) for a, b in CAD for s in STRIDES]
    for dr in ("+y", "-y", "+x", "-x"):
        for F in (2, 4, 6, 8, 10, 12):
            t0 = push_t(6, "ss")
            J.append((dict(e=e, test="push", dir=dr, F=F), dict(C, push=(t0, 0.2, F * (1 if dr[0] == "+" else -1), dr[1]), T_end=t0 + 4.0)))
if __name__ == "__main__":
    from multiprocessing import Pool
    t = time.time()
    with Pool(40) as p:
        out = p.map(one, J)
    rows = [dict(meta=m, res=r) for m, c, r in out]
    jdump(rows, os.path.join(HERE, "results", "check_v20A_foot.json"))
    for e in (0.0, 0.010, 0.015, 0.020):
        ok = [r for r in rows if r["meta"]["e"] == e and r["meta"]["test"] == "walk" and "error" not in r["res"] and not r["res"]["fallen"]]
        good = [r for r in ok if r["res"]["cop_margin_mm"]["p5"] >= 5 and r["res"]["split"].get("knee", {}).get("sat_frac", 1) < 0.10]
        b = max(good, key=lambda r: r["meta"]["stride"] / (r["meta"]["Ts"] + r["meta"]["Td"])) if good else None
        pu = {}
        for dr in ("-y", "+y", "+x", "-x"):
            pr = sorted([r for r in rows if r["meta"]["e"] == e and r["meta"]["test"] == "push" and r["meta"]["dir"] == dr], key=lambda r: r["meta"]["F"])
            ff = next((r["meta"]["F"] for r in pr if "error" in r["res"] or r["res"]["fallen"]), None)
            pu[dr] = max([r["meta"]["F"] for r in pr if "error" not in r["res"] and not r["res"]["fallen"] and (ff is None or r["meta"]["F"] < ff)] or [0])
        print("e", e, "usable", round(b["meta"]["stride"] / (b["meta"]["Ts"] + b["meta"]["Td"]), 3) if b else None,
              "margin", round(b["res"]["cop_margin_mm"]["p5"], 1) if b else None, "push", pu, "errors", sum(1 for r in rows if r["meta"]["e"] == e and "error" in r["res"]))
    print(f"{time.time() - t:.0f}s")
