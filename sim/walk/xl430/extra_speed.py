"""보폭 55 · 60 mm (0.085 · 0.092 m/s) 추가: 주요 모델만. → walk_xl.json 에 덧붙임"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import walk_xl as W
from run_tests import run_jobs
from kw_common import jdump
rows = json.load(open(os.path.join(HERE, "results", "walk_xl.json")))
J = []
for mn in ("v18 지금(XM 3 + XC 2)", "모두 XL430", "모두 XL430, 몸 −10 %", "모두 XL430, 몸 −20 %", "XL + 무릎 XC", "XL + 무릎 · 발목 피치 XC",
           "XL + 무릎 XM", "XL, 무릎 XL 두 개", "모두 XL + 고관절 롤 B(XL)"):
    for kp in (2.0, 4.0):
        for s in (0.055, 0.06):
            J.append((dict(model=mn, kp=kp, crouch=0.025, test=f"보폭 {int(round(s * 1000))}"), dict(W.MODELS[mn], kp_scale=kp, crouch=0.025, stride=s)))
out = run_jobs(J, procs=40)
rows += [dict(meta=m_, res=r) for m_, c, r in out]
jdump(rows, os.path.join(HERE, "results", "walk_xl.json"))
print(len(rows))
