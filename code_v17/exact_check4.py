"""Exact-ish interference check for v4: same-axis profiles compared in 2D (vertex inclusion + edge crossing),
coaxial cylinders vs profiles with clearance holes, other pairs by bounding box."""
import itertools, math, sys, importlib
from build import IDX, UVI
mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "build4")
links, _, _ = mod.build_model()
TOL = 0.01

def inside(p, poly):
    x, y = p; c = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c

def seg_x(a, b, c, d):
    def o(p, q, r): return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    d1, d2, d3, d4 = o(c, d, a), o(c, d, b), o(a, b, c), o(a, b, d)
    return (d1 * d2 < -1e-9) and (d3 * d4 < -1e-9)

def shrink(poly, eps=0.02):
    cx = sum(p[0] for p in poly)/len(poly); cy = sum(p[1] for p in poly)/len(poly)
    return poly   # vertices on shared boundaries are treated by the strict tests below

def seg_dist(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = max(0, min(1, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / (dx*dx + dy*dy)))
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)

def ov1(a, b): return min(a[1], b[1]) - max(a[0], b[0]) > TOL

def poly_poly(P, Q, holesP, holesQ):
    def strictly_inside(p, poly, holes):
        if not inside(p, poly): return False
        if any(math.hypot(p[0]-u, p[1]-v) < r for u, v, r in holes): return False
        return min(seg_dist(p, poly[i], poly[(i+1) % len(poly)]) for i in range(len(poly))) > TOL
    if any(strictly_inside(p, Q, holesQ) for p in P) or any(strictly_inside(q, P, holesP) for q in Q):
        return True
    for i in range(len(P)):
        for j in range(len(Q)):
            if seg_x(P[i], P[(i+1) % len(P)], Q[j], Q[(j+1) % len(Q)]):
                return True
    return False

def cyl_prism(c, p):
    (cc, a, r, h), (axis, a0, a1, poly, holes) = c[1], p[1]
    k = IDX[axis]
    if abs(abs(a[k]) - 1) > 1e-9: return None
    iu, iv = UVI[axis]
    if not ov1(sorted((cc[k], cc[k] + a[k]*h)), sorted((a0, a1))): return False
    ctr = (cc[iu], cc[iv])
    for hu, hv, hr in holes:
        if math.hypot(ctr[0]-hu, ctr[1]-hv) + r <= hr + 1e-6: return False
    if inside(ctr, poly): return True
    return any(seg_dist(ctr, poly[i], poly[(i+1) % len(poly)]) < r - TOL for i in range(len(poly)))

import geom4
_cache = {}
def sample_test(s1, s2):
    for a, b in ((s1, s2), (s2, s1)):
        key = repr(a)
        if key not in _cache: _cache[key] = geom4.samples(a, 0.6)
        if geom4.inside(b, _cache[key], 0.05).any(): return True
    return False

def collide(s1, s2):
    s1, s2 = mod.sampled(s1), mod.sampled(s2)
    b1, b2 = mod.aabb(s1), mod.aabb(s2)
    if not all(ov1(b1[i], b2[i]) for i in range(3)): return False
    if "prismh" in (s1[0], s2[0]) or "loft" in (s1[0], s2[0]):
        return sample_test(s1, s2)
    if s1[0] == "cyl" and s2[0] == "prism":
        r = cyl_prism(s1, s2); return sample_test(s1, s2) if r is None else r
    if s2[0] == "cyl" and s1[0] == "prism":
        r = cyl_prism(s2, s1); return sample_test(s1, s2) if r is None else r
    if s1[0] == "cyl" and s2[0] == "cyl":
        (c1, a1, r1, h1), (c2, a2, r2, h2) = s1[1], s2[1]
        for k in range(3):
            if abs(abs(a1[k]) - 1) < 1e-9 and abs(abs(a2[k]) - 1) < 1e-9:
                o = [i for i in range(3) if i != k]
                return math.hypot(c1[o[0]]-c2[o[0]], c1[o[1]]-c2[o[1]]) < r1 + r2 - TOL
        return sample_test(s1, s2)
    if s1[0] == s2[0] == "prism" and s1[1][0] == s2[1][0]:
        return poly_poly(s1[1][3], s2[1][3], s1[1][4], s2[1][4])
    return sample_test(s1, s2)

items = [(ln, pn, s) for ln, L in links.items() for pn, sol in L["parts"].items() for s in sol]
bad = [(l1, p1, l2, p2) for (l1, p1, s1), (l2, p2, s2) in itertools.combinations(items, 2)
       if l1 != l2 and collide(s1, s2)]
print("interferences between links:", len(bad))
for b in bad: print(" ", b)
intra = []
for ln, L in links.items():
    sol = [(pn, s) for pn, ss in L["parts"].items() for s in ss]
    for (p1, s1), (p2, s2) in itertools.combinations(sol, 2):
        if p1 == p2: continue
        if any(t in p1 or t in p2 for t in ("_bolt_", "_nut_")) and collide(s1, s2):
            if (("_bolt_" in p1 and "_nut_" in p2) or ("_nut_" in p1 and "_bolt_" in p2)) and s1[0] != s2[0]:
                continue
            intra.append((ln, p1, p2))
print("intra-link fastener issues:", len(intra))
for i in intra[:10]: print(" ", i)
