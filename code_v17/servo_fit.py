"""Servo case fit: distance between each X430 case face and the printed parts of the same link, and the case-screw
hole spots (x430_holes.HOLES) that land on a printed part.

usage: python3 servo_fit.py build18 [out.json]
"""
import json, math, sys, importlib
import numpy as np
import geom4, build4
import x430_holes as X

D_MAX = 2.0          # look this far out of each case face (mm)
D_STEP = 0.025
FACES = {"+x": (0, 1), "-x": (0, -1), "+y(front)": (1, 1), "-y(back)": (1, -1), "+z(horn)": (2, 1), "-z(idler)": (2, -1)}


def printed_parts(link, mass_parts=None):
    """part names of a link that are printed (mass_props kind 'print' when known, else not servo/horn/idler)"""
    kind = {}
    if mass_parts:
        for p in mass_parts:
            kind.setdefault(p["part"], p["kind"])
    out = []
    for pn in link["parts"]:
        if any(k in pn for k in ("servo_X", "_XM430", "_XC430", "_XL430", "horn", "idler", "joint_disc")):
            continue
        if kind and kind.get(pn, "print") != "print":
            continue
        out.append(pn)
    return out


def face_points(axis, sign, step=1.0, inset=0.6):
    lo = np.array([X.CASE["x"][0], X.CASE["y"][0], X.CASE["z"][0]]); hi = np.array([X.CASE["x"][1], X.CASE["y"][1], X.CASE["z"][1]])
    o = [i for i in range(3) if i != axis]
    g = [np.arange(lo[i] + inset, hi[i] - inset + 1e-9, step) for i in o]
    G = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 2)
    P = np.zeros((len(G), 3)); P[:, o[0]] = G[:, 0]; P[:, o[1]] = G[:, 1]; P[:, axis] = hi[axis] if sign > 0 else lo[axis]
    n = np.zeros(3); n[axis] = sign
    return P, n


def scan(links, mass_parts=None, per_solid=False):
    frames = X.servo_frames(links)
    rows = []
    ds = np.arange(-0.3, D_MAX + 1e-9, D_STEP)
    for sv, fr in frames.items():
        L = links[fr["link"]]
        parts = printed_parts(L, mass_parts)
        R = np.c_[fr["X"], fr["Y"], fr["Z"]]
        for fname, (ax, sg) in FACES.items():
            P, n = face_points(ax, sg)
            Pw = P @ R.T + fr["O"]; nw = R @ n
            Q = Pw[:, None, :] + ds[None, :, None] * nw[None, None, :]
            for pn in parts:
                best = None; cover = np.zeros(len(P), bool); first = np.full(len(P), np.inf)
                for si, s in enumerate(L["parts"][pn]):
                    bb = np.array(build4.aabb(build4.sampled(s)))
                    lo, hi = bb[:, 0] - 0.05, bb[:, 1] + 0.05
                    m = np.all((Q >= lo) & (Q <= hi), axis=2)
                    if not m.any():
                        continue
                    ins = np.zeros(m.shape, bool)
                    ins[m] = geom4.inside(s, Q[m])
                    hit = ins.any(1)
                    if not hit.any():
                        continue
                    fd = np.where(hit, ds[np.argmax(ins, axis=1)], np.inf)
                    better = fd < first
                    first[better] = fd[better]
                    cover |= hit
                    g = float(fd.min())
                    if per_solid:
                        rows.append(dict(servo=sv, link=fr["link"], face=fname, part=pn, gap_mm=round(g, 3), solid_index=si,
                                         solid_kind=s[0], solid_axis=s[1][0] if s[0] in ("rprism", "prism", "prismh") else "",
                                         coverage=round(float(hit.mean()), 3), contact_frac=round(float((fd <= 0.05).mean()), 3)))
                    if best is None or g < best[0]:
                        best = (g, si, s[0], s[1][0] if s[0] in ("rprism", "prism", "prismh") else "")
                if best is not None and not per_solid:
                    rows.append(dict(servo=sv, link=fr["link"], face=fname, part=pn, gap_mm=round(best[0], 3), solid_index=best[1],
                                     solid_kind=best[2], solid_axis=best[3], coverage=round(float(cover.mean()), 3),
                                     contact_frac=round(float((first <= 0.05).mean()), 3)))
    return rows, frames


def hole_spots(links, frames, mass_parts=None, depth=(0.05, 1.0)):
    """for each servo and each case hole: the printed solids met just outside the hole (within depth along the normal)"""
    out = []
    for sv, fr in frames.items():
        L = links[fr["link"]]
        parts = printed_parts(L, mass_parts)
        for hi, (p, n) in enumerate(X.HOLES):
            pw, nw = X.to_world(fr, p, n)
            Q = np.array([pw + d * nw for d in np.linspace(depth[0], depth[1], 8)])
            hits = []
            for pn in parts:
                for si, s in enumerate(L["parts"][pn]):
                    if geom4.inside(s, Q).any():
                        hits.append((pn, si, s[0], s[1][0] if s[0] in ("rprism", "prism", "prismh") else ""))
            out.append(dict(servo=sv, link=fr["link"], hole=hi + 1, p_world=[round(float(v), 3) for v in pw],
                            n_world=[int(round(v)) for v in nw], hits=hits))
    return out


if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "build18")
    links, joints, _ = mod.build_model()
    try:
        mp = json.load(open(f"mass_{sys.argv[1]}_light.json"))["parts"]
    except Exception:
        mp = None
    rows, frames = scan(links, mp)
    for r in rows:
        print(r)
    spots = hole_spots(links, frames, mp)
    for s in spots:
        if s["hits"]:
            print(s["servo"], s["hole"], s["hits"])
    if len(sys.argv) > 2:
        json.dump(dict(rows=rows, holes=spots), open(sys.argv[2], "w"), indent=1)
