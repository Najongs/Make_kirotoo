"""묶음 결과에서 오류 난 줄만 다시 돌려 덮어쓴다: python robust/rerun_errors.py oat"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rb_common import RES, ROB0, run_jobs, brief, jdump
import rb_sweep as sweep

g = sys.argv[1]
p = os.path.join(RES, f"{g}.json"); D = json.load(open(p))
jobs = {json.dumps(m, ensure_ascii=False, sort_keys=True): (m, c) for m, c in sweep.GROUPS[g]()}
bad = [i for i, r in enumerate(D["rows"]) if "error" in r["res"]]
out = run_jobs([jobs[json.dumps(D["rows"][i]["meta"], ensure_ascii=False, sort_keys=True)] for i in bad], procs=len(bad) or 1) if bad else []
for i, (m_, c, r) in zip(bad, out):
    D["rows"][i]["res"] = r
    print(json.dumps(m_, ensure_ascii=False), brief(r))
jdump(D, p)
