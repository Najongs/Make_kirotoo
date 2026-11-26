import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import walk_xl as W
from run_tests import run_jobs
from kw_common import jdump
rows = json.load(open(os.path.join(HERE, "results", "walk_xl.json")))
keep = [r for r in rows if "error" not in r["res"]]
J = [j for j in W.jobs() if j[0]["model"] in ("모두 XL430, 몸 −20 %", "모두 XL430, 몸 −30 %")]
out = run_jobs(J, procs=48)
keep += [dict(meta=m_, res=r) for m_, c, r in out]
jdump(keep, os.path.join(HERE, "results", "walk_xl.json"))
print(len(keep), sum(1 for r in keep if "error" in r["res"]))
