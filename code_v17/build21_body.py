"""v21 body = v21 head (build21_head: v20 geometry, nod -20 .. +28) + torso / head print fits opened up (first print
kit for the torso and the head, owner 2026-11-27).  Only torso / head printed parts change; legs and arms stay v20.

Fits (build20 rules: sliding / snap gaps >= 0.25 per side, fits per side >= 0.3 where a part drops over another)
  * snap hooks (build14.hook): every hook arm sat 0.15 from the face its barb hooks behind (head seam hooks, head
    bottom hook, face plate hooks -> catch bars, crest hooks -> top wall slots, lid hooks -> frame wall windows).
    The arm (with its barb) moves SNAP_EXTRA away from that face -> 0.25; the barb catch shrinks 0.85 -> 0.75 and
    the room behind the arm in the crest slots 1.0 -> 0.9 (still more than the 0.75 the barb has to deflect)
  * head seam tongue (back part, under the front part's top wall): 0.15 -> 0.25
  * shorts flange hole round the frame walls (y +-60.2 for the walls at +-60): 0.2 -> 0.3 per side (the 0.07 of
    sim/tests/fit_gaps_v20.py was the flange bottom resting on the pelvis plate top, a seat, not a gap)
  * cover dovetails (frame rail <-> cover channel): flank gap 0.18 / face gap 0.2 -> 0.3 (channel widened)
  * lid plug back edge at x -65 dropped into the frame flush with the back wall face (gap 0): -> -65 + LID_BACK_GAP
Nothing else is moved; parts are written to out20/print_parts/<name>.stp (torso / head only) by __main__.

Assembly durability (owner 2026-11-27 after the first prints: joints broke while assembling; assume a coarse print,
+-0.4 mm, 0.3 mm layers): no snaps, dovetail as a loose guide only, no small press fits, no printed pins < 3 mm
  * every snap hook arm bent across the layers (insertion axis = the part's print-up axis) with a square root:
    lid 4 -> M2.5 screws down into pads on the frame walls; face plate 2 -> M2.5 from the front into bosses on the
    head front; head seam sides 2 -> M2.5 through the front part's side wall into tabs of the back part (heads under
    the ear cups); head seam bottom 1 and crest 4 -> removed (seam held by 2 screws + top tongue, crest glued)
  * fits per side >= 0.4 (tongue, shorts flange, lid back edge, dovetail -> guide; the lid screws hold the covers)
  * inserts glued with 0.4 per side: eyes, mouth, ear discs, antennas (now printed lying on a flat), camera clip
    screwed (M2) instead of the 2 mm printed pins; shelf pins -> steel 3 mm dowels glued in the ledges; yoke pegs ->
    4 x M2.5 down into the lid; yoke uprights 20 -> 24 wide with a round top (wall round the 688 seat 2 -> 4)
  * nod crank redrawn on the XL430 horn (pad to the horn face, centre 9, 4 x M2 x 5 on PCD 16 at 0 / 90 / 180 /
    270 deg, counterbores 4.4); rods' pin holes 3.2 -> 3.4
  * hip yaw servos: + the case holes 3 / 4 (13 / 14) on the outer side wall (pad over the wall window)
"""
import os, sys, json, math
import numpy as np
import build21_head as BH
from build21_head import *          # noqa  (build20 names)
import build14 as B14

SNAP_EXTRA = 0.1            # (ear hooks only; build16 drops them)
TONGUE_GAP = 0.4
SHORTS_GAP = 0.4            # shorts flange hole -> frame wall, per side
DOVE_GAP = 0.4              # cover channel -> frame rail, per face (a loose guide: the lid screws hold the covers)
LID_BACK_GAP = 0.4          # lid plug back edge (x -65) -> frame back wall face / OpenCR channel
GAP = 0.4                   # coarse print: per side for inserts / sliding parts
TAP_R, CLR_R, CB_R = 1.1, 1.45, 2.9   # M2.5 tapping screw: printed pilot 2.2 (drill 2.2 after printing), clearance 2.9, head 5.8
M2_TAP_R, M2_CLR_R = 0.8, 1.2         # M2 (camera)
LOG = {}
BODY_LINKS = ("torso", "head", "ears", "nod_crank", "nod_rod", "ear_rod")
# being printed by the owner (2026-11-27): geometry and STEP stay as v20; fits to them change on the receiving part only
FROZEN = ()                 # 2026-11-27: the parts already printed may change too
PART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out20", "print_parts")


def _hook(ins, flex, root, catch, face, w0, w1, rgb, t=B14.HT_, b=B14.HB):
    """snap hooks -> screws / glue (module docstring). Arm length L = |catch - root| + 3b, thickness t, barb b:
    bending strain at the root eps = 1.5 t b / L^2 (cantilever), logged for the kit table"""
    sf = 1 if flex[0] == "+" else -1
    L = abs(catch - root) + 3 * b
    rec = dict(ins=ins, flex=flex, face=round(face, 2), arm_mm=round(L, 1), t_mm=t, barb_mm=round(b - SNAP_EXTRA, 2),
               strain_pct=round(150 * t * b / L ** 2, 2))
    LOG.setdefault("hooks", []).append(rec)
    if ins[1] == "y":                        # ear hooks: dropped later by build16 (ears turn on the axle)
        rec["now"] = "dropped by build16"
        return _HOOK0(ins, flex, root, catch, face - sf * SNAP_EXTRA, w0, w1, rgb, t=t, b=b)
    if ins == "+x" and flex[1] == "y":       # head seam side: tab of the back part under the front part's side wall
        s = sf
        y_out = s * (B14.HW - B14.T - GAP)
        y_in = s * (B14.HW - B14.T - GAP - 7.0)
        zc = (w0 + w1) / 2
        rec["now"] = "M2.5 x 10 through the front side wall"
        ya, yb = sorted((y_in, s * (B14.HW - B14.T + 0.1)))
        yc, yd = sorted((y_in, y_out))
        return [B14.box((-3, ya, zc - 4.5), (B14.XS, yb, zc + 4.5), rgb),
                B14.poly("z", zc - 4.5, zc + 4.5, [B14.W(x=-3, y=y_in), B14.W(x=-3, y=s * (B14.HW - B14.T + 0.1)),   # 45 deg
                                                   B14.W(x=-3 - abs(y_in - s * (B14.HW - B14.T + 0.1)),             # under
                                                         y=s * (B14.HW - B14.T + 0.1))], rgb),                     # the tab
                PROFI("y", yc, yd, rrect(zc - 4.5, zc + 4.5, B14.XS - 0.5, SEAM_X + 3.5, 0), [], rgb, [(zc, SEAM_X, TAP_R)])]
    rec["now"] = {"-z": "lid: M2.5 into the frame wall pads" if abs(abs(face) - 54.85) < 0.01 else "crest: glued",
                  "+x": "head seam bottom: removed", "-x": "face plate: M2.5 into head front bosses"}.get(ins, "?")
    return []


def _cover_channel(s, c, rgb):
    """build14.cover_channel with DOVE_GAP between the female dovetail and the frame rail on every face"""
    W, poly, box = B14.W, B14.poly, B14.box
    g = DOVE_GAP
    y = lambda d: s * (60.0 + d)
    # rail (build14.frame_rail): face d 8.8, tip d 11.8, flank |x - c| 2.5 -> 4.0 over d 8.8 -> 11.8;
    # channel flank = rail flank moved g along its outward normal (0.894, -0.447)
    a0, a1 = 8.8 + g, 11.8 + g
    fx = lambda d: 2.5 + 0.894 * g + (d - 8.8 + 0.447 * g) * 0.5
    C = [W(x=c - 7, y=y(a0)), W(x=c - fx(a0), y=y(a0)), W(x=c - fx(a1), y=y(a1)), W(x=c + fx(a1), y=y(a1)),
         W(x=c + fx(a0), y=y(a0)), W(x=c + 7, y=y(a0)), W(x=c + 7, y=y(13.8)), W(x=c - 7, y=y(13.8))]
    out = [poly("z", 280, 316, C, rgb),
           box((c - 7, min(y(a0), y(13.8)), 316), (c + 7, max(y(a0), y(13.8)), 318), rgb)]
    for x0, x1 in ((c - 7, c - 4.5), (c + 4.5, c + 7)):
        out.append(poly("x", x0, x1, [W(y=y(a0), z=280), W(y=y(13.8), z=280), W(y=y(13.8), z=275.2 + (a0 - 9.0))], rgb))
    LOG.setdefault("dovetail", []).append(dict(side=s, x=c, face_d=round(a0, 2), tip_d=round(a1, 2),
                                               flank_at_face=round(fx(a0), 3), flank_at_tip=round(fx(a1), 3)))
    return out


_HOOK0 = B14.hook
SEAM_X = 18.7                # head seam side screw: x of its axis (the old window x 17 .. 20.4)
from build6 import PROFI
from shapes import rrect, circle_loop
import build16 as B16, build19 as B19, build20 as B20, x430_holes as X


def tongue(links):
    """head seam tongue (back part, x XS .. XS + 6): top face TOP_IN - 0.15 -> TOP_IN - TONGUE_GAP"""
    hb = links["head"]["parts"]["head_back"]
    top0 = B14.TOP_IN - 0.15
    n = 0
    for i, x in enumerate(hb):
        if x[0] == "rprism" and x[1][0] == "z" and abs(x[1][2] - top0) < 1e-6:
            a = x[1]
            hb[i] = (x[0], (a[0], a[1], B14.TOP_IN - TONGUE_GAP, a[3], a[4], a[5]), x[2]); n += 1
    return dict(solids=n, gap=(0.15, TONGUE_GAP))


def shorts_flange(links):
    """flange hole edges along the frame walls (|y| 60.2) -> 60 + SHORTS_GAP"""
    sh = links["torso"]["parts"]["shorts"]
    n = 0
    for i, x in enumerate(sh):
        if x[0] == "rprism" and x[1][0] == "z" and x[1][4]:
            a = x[1]
            inn = [[(p[0], math.copysign(60.0 + SHORTS_GAP, p[1]) if abs(abs(p[1]) - 60.2) < 1e-6 else p[1]) + tuple(p[2:])
                    for p in l] for l in a[4]]
            n += sum(abs(abs(p[1]) - 60.2) < 1e-6 for l in a[4] for p in l)
            sh[i] = (x[0], (a[0], a[1], a[2], a[3], inn, a[5]), x[2])
    return dict(points=n, gap=(0.2, SHORTS_GAP))


def lid_back(links):
    """lid plug (z 330 .. 336.3, the U outline of build17): outline points at x -65 -> -65 + LID_BACK_GAP"""
    lid = links["torso"]["parts"]["lid"]
    n = 0
    for i, x in enumerate(lid):
        if x[0] == "rprism" and x[1][0] == "z" and x[1][1] >= 329.9 and x[1][2] <= 336.4 and len(x[1][3]) >= 8:
            a = x[1]
            o = [(-65.0 + LID_BACK_GAP if abs(p[0] + 65.0) < 1e-6 else p[0],) + tuple(p[1:]) for p in a[3]]
            n += sum(abs(p[0] + 65.0) < 1e-6 for p in a[3])
            lid[i] = (x[0], (a[0], a[1], a[2], o, a[4], a[5]), x[2])
    return dict(points=n, gap=(0.0, LID_BACK_GAP))


# ---------------------------------------------------------------- assembly durability (2026-11-27)
def _set_holes(s, fn):
    a = s[1]
    return (s[0], (a[0], a[1], a[2], a[3], a[4], fn(list(a[5]))), s[2])


def _in_outline(s, u, v, r=0.0):
    from matplotlib.path import Path
    loop = np.asarray(build4.sampled(s)[1][3])[:, :2]
    t = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    return bool(Path(loop).contains_points(np.c_[u + r * np.cos(t), v + r * np.sin(t)] if r else [[u, v]]).all())


def drill(ss, axis, u, v, r, cb=None):
    """hole r through every rprism of the part along `axis` whose outline holds (u, v); cb = (r_cb, w0, w1): the
    stretch w0..w1 along the axis gets r_cb (solids straddling w0 / w1 are split there)"""
    out, n = [], 0
    for s in ss:
        if s[0] != "rprism" or s[1][0] != axis or not _in_outline(s, u, v):
            out.append(s); continue
        a = s[1]; n += 1
        cuts = sorted({a[1], a[2]} | ({w for w in cb[1:] if a[1] < w < a[2]} if cb else set()))
        for b0, b1 in zip(cuts[:-1], cuts[1:]):
            rr = cb[0] if cb and cb[1] - 1e-6 <= b0 and b1 <= cb[2] + 1e-6 else r
            out.append(("rprism", (a[0], b0, b1, a[3], a[4], list(a[5]) + [(u, v, rr)]), s[2]))
    ss[:] = out
    return n


def lid_screws(links):
    """4 lid hooks -> 4 x M2.5 x 12 down through the lid into pads on the inner face of the frame side walls (pad
    8 x 6.2 x 10 + 45 deg gusset, pilot at |y| 52.3: 2.2 / 1.6 + wall to the pad faces); the lid gets a d 7 boss
    under its cap down to the pad top and a counterbore in the cap"""
    T_ = links["torso"]["parts"]
    fr, lid = T_["torso_frame"], T_["lid"]
    rec = []
    for s in (1, -1):
        for c in B14.LID_HOOK_X:
            yc = s * 52.3
            y0, y1 = sorted((s * 49.0, s * 55.2))
            fr.append(PROFI("z", 320.0, 330.0, rrect(c - 4, c + 4, y0, y1, 0), [], PRINT, [(c, yc, TAP_R)]))
            fr.append(B14.poly("x", c - 4, c + 4, [B14.W(y=s * 55.2, z=320), B14.W(y=s * 49.0, z=320),
                                                   B14.W(y=s * 55.2, z=313.8)], PRINT))
            lid.append(PROFI("z", 330.0, 336.3, circle_loop(c, yc, 3.5), [], B14.WHITE))
            nl = drill(lid, "z", c, yc, CLR_R, cb=(CB_R, 336.3, 338.3))
            rec.append(dict(x=c, y=yc, lid_solids=nl))
    return rec


def face_screws(links):
    """face plate side hooks -> M2.5 x 10 from the front into x bosses (d 7) standing on the catch bars"""
    H = links["head"]["parts"]
    yb, zc = 70.5, B14.FZ
    for s in (1, -1):
        H["head_front"].append(PROFI("x", 40.0, B14.FX - GAP, circle_loop(s * yb, zc, 3.5), [], B14.SKY,
                                     [(s * yb, zc, TAP_R)]))
        H["head_front"].append(B14.poly("z", zc - 3.5, zc + 3.5, [B14.W(x=40.0, y=s * (yb - 3.5)), B14.W(x=40.0, y=s * 72.15),
                                                               B14.W(x=40.0 - (72.15 - yb + 3.5), y=s * 72.15)], B14.SKY))
        drill(H["face_plate"], "x", s * yb, zc, CLR_R)
    return dict(y=yb, z=zc, boss_d=7.0, pilot_depth=round(B14.FX - GAP - 40.0, 1))


def seam_screws(links):
    """head seam side windows (x 17 .. 20.4) in the front part -> clearance hole at the tab's pilot"""
    hf = links["head"]["parts"]["head_front"]
    n = 0
    for i, s in enumerate(hf):
        if s[0] == "rprism" and s[1][0] == "y" and s[1][4]:
            a = s[1]
            keep = [l for l in a[4] if not (abs(min(p[1] for p in l) - 17.0) < 0.05 and abs(max(p[1] for p in l) - 20.4) < 0.05)]
            if len(keep) < len(a[4]):
                zc = (min(p[0] for p in a[4][0]) + max(p[0] for p in a[4][0])) / 2
                hf[i] = (s[0], (a[0], a[1], a[2], a[3], keep, list(a[5]) + [(zc, SEAM_X, CLR_R)]), s[2]); n += 1
    return n


def inserts(links):
    """eyes / mouth / ear discs / antennas: GAP per side + glue; antennas get a flat (printed lying on it);
    camera: M2 x 8 through clip + board into d 5 standoffs (printed 2 mm pins removed)"""
    H, E, T_ = links["head"]["parts"], links["ears"]["parts"], links["torso"]["parts"]
    for nm, y in (("eye_L", B14.EYE_L), ("eye_R", -B14.EYE_L)):
        s0 = H[nm][0]; a = s0[1]
        H[nm][0] = (s0[0], (a[0], a[1], a[2], B14.eye_pill(y, -GAP), a[4], a[5]), s0[2])
    m = B14.SMILE
    s0 = H["mouth"][0]; a = s0[1]
    from shapes import annulus_sector
    H["mouth"][0] = (s0[0], (a[0], a[1], a[2], annulus_sector(0, m[0], m[1] + GAP, m[2] - GAP, m[3] + 1.0, m[4] - 1.0),
                             a[4], a[5]), s0[2])
    for side, s in (("L", 1), ("R", -1)):
        d = E[f"ear_disc_{side}"]
        c, ax, r, h = d[1][1]
        d[1] = ("cyl", (c, ax, r - GAP, h), d[1][2])
        c, ax, r, h = E[f"antenna_{side}"][0][1]
        rgb = E[f"antenna_{side}"][0][2]
        loop = [(max(p[0], c[0] - 3.2),) + tuple(p[1:]) for p in circle_loop(c[0], c[1], r)]
        E[f"antenna_{side}"] = [PROFI("z", c[2], c[2] + h, loop, [], rgb)]
        ear = E[f"ear_{side}"]
        for i, x in enumerate(ear):
            if x[0] == "rprism" and x[1][0] == "z":
                ear[i] = _set_holes(x, lambda hs: [(h[0], h[1], h[2] + GAP) if abs(h[2] - r) < 1e-6 else h for h in hs])
    fp = H["face_plate"]
    cams = [(y, z) for y, z in B14.CAM_HOLES]
    fp[:] = [x for x in fp if not (x[0] == "cyl" and abs(x[1][2] - 1.0) < 1e-6)]
    for i, x in enumerate(fp):
        if x[0] == "cyl" and abs(x[1][2] - 2.2) < 1e-6:
            c, ax, r, h = x[1]
            fp[i] = PROFI("x", c[0] - h, c[0], circle_loop(c[1], c[2], 2.5), [], x[2], [(c[1], c[2], M2_TAP_R)])
    clip = H["camera_clip"]
    clip[:] = [_set_holes(x, lambda hs: [(h[0], h[1], M2_CLR_R) for h in hs]) for x in clip]
    # shelf pins: steel 3 mm dowels (glued in the frame ledges), shelf holes 3.0 + 2 GAP
    T_["shelf_dowels_3mm"] = [(x[0], x[1], B20.STEEL if hasattr(B20, "STEEL") else B16.STEEL) for x in T_.pop("shelf_pins") if x[0] == "cyl" and abs(x[1][2] - 1.5) < 1e-6]
    sh = T_["electronics_shelf"]
    sh[:] = [_set_holes(x, lambda hs: [(h[0], h[1], 1.5 + GAP) if abs(h[2] - 1.5) < 1e-6 else h for h in hs]) for x in sh]
    return dict(camera_screws=len(cams))


def yoke(links):
    """uprights 20 -> 24 wide with a round top (r 12 about the shaft: wall round the 688 seat 2 -> 4, top stays
    inside the r 16.5 head floor circle); pegs -> 4 x M2.5 x 10 down through the yoke base into the lid"""
    T_ = links["torso"]["parts"]
    yk, lid = T_["neck_yoke"], T_["lid"]
    zt = B14.ZT9
    up = B16.hull([(max(u, 338.3), v) for u, v in B16.circ(zt, 0.0, 12.0)] + [(338.3, -12.0), (338.3, 12.0)])
    out = []
    for x in yk:
        if x[0] == "cyl":
            continue
        if x[0] == "rprism" and x[1][0] == "z" and abs(x[1][1] - 338.3) < 1e-6:     # base under the 688 seats:
            a = x[1]                                                                 # top 341.3 -> 340.0 (the
            x = ("rprism", (a[0], a[1], 340.0, a[3], a[4], a[5]), x[2])              # bearings reach down to 340.5)
        if x[0] == "rprism" and x[1][0] == "y" and x[1][5] and abs(x[1][5][0][0] - zt) < 1e-6:
            a = x[1]
            x = ("rprism", (a[0], a[1], a[2], [(u, v, 0) for u, v in up], a[4], a[5]), x[2])
        out.append(x)
    yk[:] = out
    for px, py in B14.YOKE_PEGS:
        drill(yk, "z", px, py, CLR_R)
        for i, x in enumerate(lid):
            if x[0] == "rprism" and x[1][0] == "z":
                lid[i] = _set_holes(x, lambda hs: [(h[0], h[1], TAP_R) if abs(h[0] - px) < 1e-6 and abs(h[1] - py) < 1e-6 else h for h in hs])
    return len(B14.YOKE_PEGS)


def nod_crank_xl(links):
    """crank on the XL430 horn: disc r 12.5 + pad down to the horn face (build20 HORN_FACE), centre hole r HUB_R +
    HUB_CLR, 4 x M2 x 5 on PCD 16 at 0 / 90 / 180 / 270 deg (HORN_HOLE_D), counterbores CB_D with 2.0 under the head
    (build20.xl_horn_fit rules); pin holes 3.0 + 0.2 (pins glued); the pin and ear arms as before"""
    nc = links["nod_crank"]["parts"]
    sx, sz = B14.NOD_S
    y_srv = -3.7                                   # nod servo centre plane (build17.fix_nod)
    yf = y_srv + B20.HORN_FACE                     # horn face 15.3
    ycb = yf + B20.UNDER_HEAD                      # counterbore floor
    y0, y1 = 17.2, 19.7
    P1, P2, _ = B16.ear_geometry()
    px = sx + B14.NOD_R
    ring = [(sz + 8.0 * math.sin(a), sx + 8.0 * math.cos(a)) for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
    hc = B20.HUB_R + B20.HUB_CLR
    hd = B19.HORN_HOLE_D / 2
    arm = B16.hull(B16.circ(sz, sx, 12.5) + B16.circ(sz, px, 4.5) + B16.circ(P1[1], P1[0], 4.5))
    loop = [(u, v, 0) for u, v in arm]
    disc = [(u, v, 0) for u, v in B16.circ(sz, sx, 12.5)]
    sol = [PROFI("y", yf, ycb, disc, [], PRINT, [(sz, sx, hc)] + [(u, v, hd) for u, v in ring]),
           PROFI("y", ycb, y1, loop, [], PRINT, [(sz, sx, hc)] + [(u, v, B20.CB_D / 2) for u, v in ring] +
                 [(sz, px, 1.5 + 0.1), (P1[1], P1[0], 1.5 + 0.1)])]
    nc["nod_crank"] = sol
    for ln, nm in (("nod_rod", "nod_rod"), ("ear_rod", "ear_rod")):
        ss = links[ln]["parts"][nm]
        ss[:] = [_set_holes(x, lambda hs: [(h[0], h[1], 1.7) if abs(h[2] - 1.6) < 1e-6 else h for h in hs]) for x in ss]
    return dict(horn_face_y=round(yf, 2), thick=round(y1 - yf, 2), centre_d=2 * hc, screw_d=2 * hd, cb_d=B20.CB_D,
                thread_mm=round(B20.HORN_SCREW - B20.UNDER_HEAD, 1))


def yaw_side_screws(links):
    """hip yaw servos: case holes at servo y -4 on the outer side wall (holes 3 / 4 right, 13 / 14 left) -> pad
    over the wall window + CASE_HOLE_D through"""
    fr = links["torso"]["parts"]["torso_frame"]
    frames = X.servo_frames(links)
    rec = []
    for sv, f in sorted(frames.items()):
        if "yaw_servo" not in sv:
            continue
        pts = []
        for hn in (3, 4, 13, 14):
            p, n = X.HOLES[hn - 1]
            pw, nw = X.to_world(f, p, n)
            if abs(abs(pw[1]) - 54.25) < 0.6 and pw[1] * nw[1] > 0:     # on the outer face (toward |y| 60)
                pts.append((hn, pw))
        if not pts:
            continue
        s = 1 if pts[0][1][1] > 0 else -1
        xs = [p[0] for _, p in pts]; zs = [p[2] for _, p in pts]
        y0, y1 = sorted((s * 55.0, s * 60.0))
        fr.append(PROFI("y", y0, y1, rrect(min(zs) - 4.0, max(zs) + 5.0, min(xs) - 8.4, max(xs) + 6.4, 0), [], PRINT,
                        [(p[2], p[0], B19.CASE_HOLE_D / 2) for _, p in pts]))
        rec.append(dict(servo=sv, holes=[hn for hn, _ in pts], world=[[round(float(c), 2) for c in p] for _, p in pts]))
    return rec


def build_model():
    LOG.clear()
    saved = (B14.hook, B14.cover_channel)
    B14.hook, B14.cover_channel = _hook, _cover_channel
    try:
        links, joints, fasteners = BH.build_model()
    finally:
        B14.hook, B14.cover_channel = saved
    LOG["tongue"] = tongue(links)
    LOG["shorts_flange"] = shorts_flange(links)
    LOG["lid_back"] = lid_back(links)
    LOG["lid_screws"] = lid_screws(links)
    LOG["face_screws"] = face_screws(links)
    LOG["seam_screws"] = seam_screws(links)
    LOG["inserts"] = inserts(links)
    LOG["yoke_screws"] = yoke(links)
    LOG["nod_crank"] = nod_crank_xl(links)
    LOG["yaw_side_screws"] = yaw_side_screws(links)
    return links, joints, fasteners


GONE = ("shelf_pins",)        # now steel dowels (shelf_dowels_3mm)
UP = {"antenna_L": ("+x", "주황색. 납작한 면을 바닥에 눕혀서. 귀 구멍에 접착"),
      "antenna_R": ("+x", "주황색. 납작한 면을 바닥에 눕혀서. 귀 구멍에 접착"),
      "nod_crank": ("-y", "구조색. 팔 면(카운터보어 쪽)을 바닥에. XL430 혼에 M2 x 5 네 개"),
      "neck_yoke": ("+z", "남색. 688 베어링 2개 손으로 밀어 넣고 고정제, M2.5 x 10 네 개로 뚜껑에")}


def export_body(links, names):
    """write the torso / head printed parts like export_parts.py (same print direction as parts.json), update their
    parts.json rows in place"""
    sys.argv = sys.argv[:1] + ["build21_body", PART_DIR]
    import export_parts as EP, printcheck, build4
    from scipy import ndimage
    table = json.load(open(os.path.join(PART_DIR, "parts.json")))
    table = [r for r in table if r["part"] not in GONE]
    rows = {r["part"]: r for r in table}
    for pn, (up, note) in UP.items():
        rows[pn].update(up=up, note=note)
    done = []
    for ln, pn, sol in printcheck.printed_parts(links):
        if pn not in names:
            continue
        up = rows[pn]["up"]
        rs = [EP.rot_solid(s, EP.ROT[up]) for s in sol]
        bbs = [build4.aabb(build4.sampled(s)) for s in rs]
        lo = [min(b[i][0] for b in bbs) for i in range(3)]
        hi = [max(b[i][1] for b in bbs) for i in range(3)]
        rs = [EP.rot_solid(s, np.eye(3), (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])) for s in rs]
        h = 0.6 if max(b - a for a, b in zip(lo, hi)) < 90 else 1.0
        V, _ = printcheck.voxelize(rs, h)
        need, _ = printcheck.overhang(V[ndimage.find_objects(V.astype(int))[0]], h)
        build4.write({pn: {"origin": (0, 0, 0), "parts": {pn: rs}}}, os.path.join(PART_DIR, f"{pn}.stp"))
        rows[pn].update(link=ln, size_mm="x".join(str(round(b - a)) for a, b in zip(lo, hi)), support_mm2=round(need),
                        volume_cm3=round(sum(build4.volume(s) for s in sol) / 1000, 1),
                        colours=sorted({tuple(round(c, 2) for c in s[2]) for s in sol}))
        done.append(pn)
    with open(os.path.join(PART_DIR, "parts.json"), "w") as f:
        json.dump(table, f, indent=1, ensure_ascii=False)
    return done


if __name__ == "__main__":
    links, joints, fasteners = build_model()
    names = [r["part"] for r in json.load(open(os.path.join(PART_DIR, "parts.json")))
             if r["link"] in BODY_LINKS and not r["part"].startswith("arm_") and r["part"] not in FROZEN + GONE]
    done = export_body(links, names)
    LOG["written"] = done
    LOG["missing"] = sorted(set(names) - set(done))
    json.dump(LOG, open(os.path.join(PART_DIR, "..", "build21_body_log.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in LOG.items() if k != "hooks"}, indent=1, default=str))
    print("hooks", len(LOG.get("hooks", [])))
