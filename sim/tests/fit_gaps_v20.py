"""printed-part to printed-part gaps inside each link (sliding fits, latches, rails): pairs whose closest distance is
between 0 (touching) and 0.6 mm, measured on volume samples (0.2 mm) near the closest spot; overlaps (press / glued
interfaces, < 0) are listed apart. Run: cd sim && uv run python tests/fit_gaps_v20.py [build20]"""
import os, sys, json, itertools, numpy as np
from scipy.spatial import cKDTree
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(SIM, "..", "code_v17"))
import importlib, build4, geom4
mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "build20")
links, joints, _ = mod.build_model()
mp = {p["part"]: p["kind"] for p in json.load(open(os.path.join(SIM, "..", "code_v17", "mass_build19_light.json")))["parts"]}
def printed(pn):
    return mp.get(pn, "print") == "print" and not any(k in pn for k in ("_XM430", "_XC430", "_XL430", "horn", "idler", "pin_4mm"))
out = []
for ln, L in links.items():
    parts = [pn for pn in L["parts"] if printed(pn)]
    bb = {pn: np.array([build4.aabb(build4.sampled(s)) for s in L["parts"][pn]]) for pn in parts}
    for a, b in itertools.combinations(parts, 2):
        A, Bb = bb[a], bb[b]
        lo = np.maximum(A[:, :, 0].min(0), Bb[:, :, 0].min(0)) - 0.7; hi = np.minimum(A[:, :, 1].max(0), Bb[:, :, 1].max(0)) + 0.7
        if np.any(lo > hi):
            continue
        def samp(pn, step):
            P = []
            for s in L["parts"][pn]:
                b_ = np.array(build4.aabb(build4.sampled(s)))
                l2 = np.maximum(b_[:, 0], lo); h2 = np.minimum(b_[:, 1], hi)
                if np.any(l2 >= h2): continue
                ax = [np.arange(l2[i] + step / 2, h2[i], step) for i in range(3)]
                if any(len(x) == 0 for x in ax): continue
                G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
                if len(G) > 400000: G = G[np.random.default_rng(0).choice(len(G), 400000, replace=False)]
                P.append(G[geom4.inside(s, G)])
            return np.vstack(P) if P else np.zeros((0, 3))
        Pa, Pb = samp(a, 0.25), samp(b, 0.25)
        if not len(Pa) or not len(Pb):
            continue
        d, i = cKDTree(Pb).query(Pa, k=1, distance_upper_bound=1.0)
        dmin = float(d.min())
        if dmin > 0.85:
            continue
        # overlap test: points of A inside B's solids
        ov = sum(int(geom4.inside(s, Pa).sum()) for s in L["parts"][b])
        j = int(np.argmin(d))
        out.append(dict(link=ln, a=a, b=b, gap_est_mm=round(max(dmin - 0.25, 0.0), 2), overlap_pts=ov, where=[round(float(v), 1) for v in Pa[j]]))
        print(out[-1], flush=True)
json.dump(out, open(os.path.join(SIM, "..", "out20", "fit_gaps_v20.json"), "w"), indent=1)
