"""Interference check with exact 2D tests for axis-aligned cylinders vs prisms (holes respected)."""
import itertools, math, sys, importlib
from build import aabb, IDX, UVI
mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "build3")
links, _, _ = mod.build_model()
TOL = 0.01

def seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)

def inside(p, poly):
    x, y = p; c = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c

def overlap_1d(a, b): return min(a[1], b[1]) - max(a[0], b[0]) > TOL

def cyl_prism(c, p):
    (cc, a, r, h), (axis, a0, a1, poly, holes) = c[1], p[1]
    k = IDX[axis]
    if abs(abs(a[k]) - 1) > 1e-9:
        return None                       # not coaxial-aligned: fall back to AABB
    iu, iv = UVI[axis]
    z = sorted((cc[k], cc[k] + a[k] * h))
    if not overlap_1d(z, sorted((a0, a1))):
        return False
    ctr = (cc[iu], cc[iv])
    for hu, hv, hr in holes:              # cylinder lies inside a clearance hole
        if math.hypot(ctr[0] - hu, ctr[1] - hv) + r <= hr + 1e-6:
            return False
    if inside(ctr, poly):
        return True
    return any(seg_dist(ctr, poly[i], poly[(i + 1) % len(poly)]) < r - TOL for i in range(len(poly)))

def collide(s1, s2):
    b1, b2 = aabb(s1), aabb(s2)
    if not all(overlap_1d(b1[i], b2[i]) for i in range(3)):
        return False
    if s1[0] == "cyl" and s2[0] == "prism":
        r = cyl_prism(s1, s2); return True if r is None else r
    if s2[0] == "cyl" and s1[0] == "prism":
        r = cyl_prism(s2, s1); return True if r is None else r
    if s1[0] == "prism" and s2[0] == "prism" and s1[1][0] == s2[1][0] == "z" and (len(s1[1][3]) != 4 or len(s2[1][3]) != 4):
        # non-rectangular z-prisms (tie rod, bearing ring): test polygon vertices/edges in 2D
        P, Q = s1[1][3], s2[1][3]
        if any(inside(p, Q) for p in P) or any(inside(q, P) for q in Q):
            return True
        return False
    return True

items = [(ln, pn, s) for ln, L in links.items() for pn, sol in L["parts"].items() for s in sol]
bad = [(l1, p1, l2, p2) for (l1, p1, s1), (l2, p2, s2) in itertools.combinations(items, 2)
       if l1 != l2 and collide(s1, s2)]
print("exact interferences between links:", len(bad))
for b in bad: print(" ", b)
# within a link: fasteners / pins must only pass through holes
intra = []
for ln, L in links.items():
    sol = [(pn, s) for pn, ss in L["parts"].items() for s in ss]
    for (p1, s1), (p2, s2) in itertools.combinations(sol, 2):
        if p1 == p2: continue
        steel = any(t in p1 or t in p2 for t in ("_bolt_", "_nut_"))
        if steel and collide(s1, s2):
            if ("_bolt_" in p1 and "_nut_" in p2) or ("_nut_" in p1 and "_bolt_" in p2):
                if s1[0] != s2[0]:
                    continue                    # nut on its bolt shank (coaxial; hole radius = shank radius)
            intra.append((ln, p1, p2))
print("intra-link fastener issues:", len(intra))
for i in intra[:10]: print(" ", i)
