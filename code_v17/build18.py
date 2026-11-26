"""v18 = v17 with the inner edge of both feet cut so that one leg can yaw up to +-THETA_MAX (other leg at 0) without
any part of its foot coming within CLEAR mm of the other foot (owner's decision 2026-11-26: trim / round the inner
foot corners).

  * guide rule: with the foot frame at the yaw axis (x forward, y_in = distance toward the other foot), a point
    stays clear for every yaw |theta| <= THETA_MAX when x sin(theta) + y_in cos(theta) <= Y_OTHER - CLEAR
    (Y_OTHER = 45: the other foot's inner edge). For the 70 mm wide sole this is the inner straight edge y_in 35
    kept for |x| <= ~23-27 mm plus two straight cuts, one toward the toe and one toward the heel, tangent to the
    r 42 circle at +-THETA_MAX (the circle itself lies outside y_in 35, so no arc is needed)
  * cut: sole plates z 6..8 / 8..12, wall z 12..28 (and its 1.6 mm inner loop), block z 12..18; new corners
    filleted (R_CUT); pockets of the z 8..12 plate clipped to 1.5 mm inside the new edge (dropped when too small);
    inner rubber pads moved inward along y to stay 1.5 mm inside the sole edge
  * hip yaw range limited to +-THETA_MAX in the joint table (the cut only guarantees clearance up to there)
  * THETA_MAX from the environment variable KITORU_THETA (default below) so variants can be compared
"""
import os, json, math
import build17 as B17
from build17 import *          # noqa
from build6 import PROFI
from shapes import segments

THETA_MAX = float(os.environ.get("KITORU_THETA", "20"))
Y_OTHER, CLEAR = 45.0, 3.0
C_CUT = Y_OTHER - CLEAR          # 42
R_CUT = 6.0                      # fillet at the new corners (>= 5)
POCKET_WALL = 1.5
PAD_MARGIN = 1.5
OUT = "/home/kiro-ai/NAJY/Make_kirotoo/out18"


def cut_lines(theta_deg, c=C_CUT):
    """half-planes a*x + b*y_in <= c (foot frame) for the toe-side and heel-side cuts"""
    t = math.radians(theta_deg)
    return [(math.sin(t), math.cos(t), c), (-math.sin(t), math.cos(t), c)]


def _clip(poly, a, b, c):
    """Sutherland-Hodgman: convex polygon [(x, y)] kept where a x + b y <= c"""
    out = []
    n = len(poly)
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        fp, fq = a * P[0] + b * P[1] - c, a * Q[0] + b * Q[1] - c
        if fp <= 1e-9:
            out.append(P)
        if (fp < -1e-9 < fq) or (fq < -1e-9 < fp):
            t = fp / (fp - fq)
            out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
    # drop near-duplicates
    res = []
    for p in out:
        if not res or math.hypot(p[0] - res[-1][0], p[1] - res[-1][1]) > 1e-6:
            res.append(p)
    if len(res) > 1 and math.hypot(res[0][0] - res[-1][0], res[0][1] - res[-1][1]) < 1e-6:
        res.pop()
    return res


def _area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))


def cut_loop(loop, ys, s, theta, c=C_CUT, r_new=R_CUT):
    """loop [(x, y, r)] in world x-y (a convex rounded polygon of a foot part) -> same loop clipped by the two cut
    lines; surviving corners keep their radius, new corners get r_new. s = -1 right foot, +1 left foot."""
    to_in = lambda y: (ys - y) / s          # world y -> y_in (toward the other foot)
    to_w = lambda yi: ys - s * yi
    P = [(x, to_in(y), r) for x, y, r in loop]
    poly = [(x, yi) for x, yi, _ in P]
    rad = {(round(x, 6), round(yi, 6)): r for x, yi, r in P}
    for a, b, cc in cut_lines(theta, c):
        poly = _clip(poly, a, b, cc)
    out = [(x, to_w(yi), rad.get((round(x, 6), round(yi, 6)), r_new)) for x, yi in poly]
    changed = len(out) != len(loop) or any(abs(o[0] - l[0]) > 1e-6 or abs(o[1] - l[1]) > 1e-6 for o, l in zip(out, loop))
    return out, changed


def _fit_radii(loop):
    """reduce fillet radii that do not fit their edges (segments() asserts)"""
    loop = list(loop)
    for _ in range(40):
        try:
            segments(loop)
            return loop
        except AssertionError:
            loop = [(x, y, r * 0.85 if r > 0.3 else 0.0) for x, y, r in loop]
    return [(x, y, 0.0) for x, y, _ in loop]


WALL_LOW = 16.0                  # wall height kept along the heel cut (bracket swept zone starts ~z 18.5)
X_OPEN = -38.0                   # the upper wall is open on the heel cut from here back
Y_OPEN_BACK = 6.0                # ... and on the back edge down to y_in 6


def c_wall(outer, inner, ys, s, theta):
    """C-shaped outline of the upper wall: the ring (outer, inner loops in world x-y with radii) opened on the heel
    cut between x = X_OPEN and the back edge at y_in = Y_OPEN_BACK."""
    to_in = lambda y: (ys - y) / s
    to_w = lambda yi: ys - s * yi
    O = [(x, to_in(y), r) for x, y, r in outer]; I = [(x, to_in(y), r) for x, y, r in inner]
    def heel_split(L):
        # vertices: find E (x < 0 on the inner straight edge, max y_in) and F (back corner on the cut, x = min x, y_in > 0)
        xmin = min(p[0] for p in L)
        iE = max((i for i, p in enumerate(L) if p[0] < 0), key=lambda i: (L[i][1], L[i][0]))
        iF = max((i for i, p in enumerate(L) if abs(p[0] - xmin) < 1e-6), key=lambda i: L[i][1])
        return iE, iF
    t = math.radians(theta)
    n_in = (math.sin(t), -math.cos(t))          # inward normal of the heel cut line -x sin t + y cos t = c
    def on_cut(L, iE, iF, xq):
        (x0, y0, _), (x1, y1, _) = L[iE], L[iF]
        u = (xq - x0) / (x1 - x0)
        return (xq, y0 + u * (y1 - y0))
    oE, oF = heel_split(O); iE, iF = heel_split(I)
    q1 = on_cut(O, oE, oF, X_OPEN)
    q1i = (q1[0] + 1.6 * n_in[0], q1[1] + 1.6 * n_in[1])
    xb_o, xb_i = O[oF][0], I[iF][0]
    q2, q2i = (xb_o, Y_OPEN_BACK), (xb_i, Y_OPEN_BACK)
    # walk the outer loop from F forward to E the long way round (F -> A -> B -> C -> D -> E), same for the inner loop
    def path(L, i0, i1):
        n = len(L); out = []; i = i0
        while True:
            i = (i + 1) % n
            if i == i1:
                break
            out.append(L[i])
        return out
    # loop order is either F->A->...->E or E->...->F; pick the direction that does not pass along the heel cut
    def long_path(L, iF_, iE_):
        a = path(L, iF_, iE_); b = path(L, iE_, iF_)[::-1]
        return a if len(a) >= len(b) else b
    po = long_path(O, oF, oE); pi = long_path(I, iF, iE)
    pts = [(q2[0], q2[1], 0.0)] + po + [(O[oE][0], O[oE][1], O[oE][2]), (q1[0], q1[1], 0.0), (q1i[0], q1i[1], 0.0),
                                        (I[iE][0], I[iE][1], I[iE][2])] + pi[::-1] + [(q2i[0], q2i[1], 0.0)]
    return _fit_radii([(x, to_w(yi), r) for x, yi, r in pts])


def ymax_in(theta, x, c=C_CUT, y_edge=35.0):
    t = math.radians(theta)
    return min(y_edge, (c - x * math.sin(t)) / math.cos(t), (c + x * math.sin(t)) / math.cos(t))


CUT_LOG = []


def cut_foot(links, side, theta=THETA_MAX):
    s = -1 if side == "R" else 1
    ys = 40.0 * s
    fs = links[f"{side}_foot"]["parts"][f"{side}_foot_sole"]
    new = []
    for x in fs:
        if x[0] == "rprism" and x[1][0] == "z":
            ax, a0, a1, outer, inn, holes = x[1]
            o2, ch = cut_loop(outer, ys, s, theta)
            if not ch:
                new.append(x); continue
            o2 = _fit_radii(o2)
            inn2 = []
            if a0 >= 11.99 and a1 >= 27.99 and inn:          # wall z 12..28: inner loop = outer shifted 1.6 mm inward
                for lp in inn:
                    l2, _ = cut_loop(lp, ys, s, theta, C_CUT - 1.6, R_CUT - 1.6)
                    inn2.append(_fit_radii(l2))
            else:                                              # pockets: clip to POCKET_WALL inside the new edge
                for lp in inn:
                    l2, chp = cut_loop(lp, ys, s, theta, C_CUT - POCKET_WALL, 1.5)
                    if not chp:
                        inn2.append(lp); continue
                    pts = [(p[0], p[1]) for p in l2]
                    if len(pts) < 3 or abs(_area(pts)) < 40.0:
                        CUT_LOG.append((side, "pocket dropped", round(abs(_area([(p[0], p[1]) for p in lp])), 1))); continue
                    inn2.append(_fit_radii(l2))
                    CUT_LOG.append((side, "pocket clipped", round(abs(_area(pts)), 1)))
            if a0 >= 11.99 and a1 >= 27.99 and inn2:
                # the heel cut runs under the ankle bracket (its plates reach down to z 27.2 and to ~z 18.5 when the
                # ankle rolls 20 deg): full ring only up to WALL_LOW, above it a C-shaped wall open over the heel cut
                new.append(PROFI(ax, a0, WALL_LOW, o2, inn2, x[2], holes))
                new.append(PROFI(ax, WALL_LOW, a1, c_wall(o2, inn2[0], ys, s, theta), [], x[2], holes))
                CUT_LOG.append((side, f"wall split: ring z {a0}..{WALL_LOW}, C-wall z {WALL_LOW}..{a1} open x <= {X_OPEN} on the heel cut"))
            else:
                new.append(PROFI(ax, a0, a1, o2, inn2, x[2], holes))
            CUT_LOG.append((side, f"z {a0}..{a1} outline cut", len(outer), "->", len(o2)))
        elif x[0] == "cyl" and abs(x[1][2] - 6) < 1e-9 and abs(x[1][3] - 6) < 1e-9 and tuple(x[1][1]) == (0, 0, 1) \
                and abs(x[1][0][2]) < 1e-9:
            (cx, cy, cz), a, rr, h = x[1]
            yi = (ys - cy) / s
            lim = ymax_in(theta, cx, C_CUT, 35.0)
            # keep the pad clear of the edge and of the new corner fillets
            yi_new = min(yi, lim - rr - PAD_MARGIN - 1.0)
            if yi_new < yi - 1e-9:
                CUT_LOG.append((side, "pad moved", (cx, round(yi, 1)), "->", (cx, round(yi_new, 1))))
            new.append(("cyl", ((cx, ys - s * yi_new, cz), a, rr, h), x[2]))
        else:
            new.append(x)
    links[f"{side}_foot"]["parts"][f"{side}_foot_sole"] = new


def build_model():
    links, joints, fasteners = B17.build_model()
    CUT_LOG.clear()
    for side in ("R", "L"):
        cut_foot(links, side, THETA_MAX)
    if THETA_MAX < 30.0 - 1e-9:
        for j in joints:
            if j["name"].endswith("_hip_yaw"):
                j["lower_deg"], j["upper_deg"] = -THETA_MAX, THETA_MAX
    return links, joints, fasteners


write = B17.write

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    os.makedirs(OUT, exist_ok=True)
    for l in CUT_LOG:
        print(*l)
    w = write(links, f"{OUT}/biped_robot_v18.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners, "theta_max_deg": THETA_MAX}, open(f"{OUT}/model.json", "w"), indent=1)
    print("theta_max", THETA_MAX, "entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))
