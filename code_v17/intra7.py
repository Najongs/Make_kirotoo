"""Intra-link check: overlaps between different parts of the same link (snap joints, rails, inserts)."""
import sys, itertools, collections
sys.argv = ["x", sys.argv[1] if len(sys.argv) > 1 else "build7"]
import exact_check4 as ec
res = collections.Counter()
for ln, L in ec.links.items():
    sol = [(pn, s) for pn, ss in L["parts"].items() for s in ss]
    for (p1, s1), (p2, s2) in itertools.combinations(sol, 2):
        if p1 != p2 and ec.collide(s1, s2):
            res[(ln, p1, p2)] += 1
for k, v in sorted(res.items()): print(k, v)
