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

Second pass (2026-11-27, coarse print: walls >= 2, features >= 3, gaps >= 0.4)
  * shell walls 1.2 -> SHELL_T 2.0, grown inward, outside unchanged (shell_walls): head back / front box shell, back
    plate, crest, ear cups (rim + lid), shirt covers (box shell + shoulder band); the 45 deg stepped rings (head back
    chamfer, face bevel) get BEV_T 2.83 radial = 2.0 normal; the face plate is 0.2 smaller per side so the two rim rings
    keep FACE_GAP 0.4 to it (was 0.2). Things on the old inner faces follow: seam tabs 0.8 inward, seam tongue 0.95 ->
    3.0 thick with 0.4 under the front top wall, 684 seats r 4.5 through the new side-wall layer, amp board 0.8 up,
    lid plug clipped to the new cover inside. Obsolete hook windows filled (ear hook slots, crest slots, seam bottom
    window); nod rod slot y 19.5 -> 19.4 (0.4 / 0.5 to the rod). Shorts stay 1.2 (they set the hip pitch limit).
  * shirt cover dovetail (no screws, owner): rail neck 7 at d 7 -> tip 11.5 at d 11.5 (depth 3 -> 4.5), pedestal 10
    wide with a 1 mm root chamfer, gap 0.5 per face, channel block 20 (+ 1 mm root chamfers) with side walls >= 3.4,
    3 mm stepped entry flare at the channel bottom; rails moved to x 15.5 / -23.5 (block clear of the shoulder slot and
    the cover seam); shorts flange notches c +- 5.3 -> +- 6.4
  * horizontal holes in print orientation -> teardrops pointing up (teardrops, last pass)
"""
import os, sys, json, math
import numpy as np
import build, build4
import build21_head as BH
from build21_head import *          # noqa  (build20 names)
import build14 as B14

SNAP_EXTRA = 0.1            # (ear hooks only; build16 drops them)
TONGUE_GAP = 0.4
TONGUE_T = 3.0              # head seam tongue thickness (was 0.95)
SHORTS_GAP = 0.4            # shorts flange hole -> frame wall, per side
DOVE_GAP = 0.5              # cover channel -> frame rail, per face (a loose guide: the lid screws hold the covers)
SHELL_T = 2.0               # head / crest / ear cup / shirt cover walls (build6.T 1.2), grown inward, outside kept
BEV_T = SHELL_T * math.sqrt(2)   # radial width of the 45 deg stepped rings (head back chamfer, face bevel) -> normal 2.0
FACE_GAP = 0.4              # face plate outline -> face rim, per side (was 0.2): plate 0.2 smaller per side
# shirt cover dovetail (owner 2026-11-27: no extra screws, a bigger slide for a coarse print), d = |y| - 60
RAIL_X = {"front": 15.5, "back": -23.5}   # was 12 / -22: the 22 mm channel block clears the shoulder slot (x <= 4) and
                                           # the cover seam (x -11.75); the pedestal clears the wall windows (x 22, -31)
DV_NECK_D, DV_DEPTH, DV_NECK_HW, DV_SLOPE = 7.0, 4.5, 3.5, 0.5   # neck 7 wide at d 7, tip 11.5 wide at d 11.5
DV_BLOCK_HW = 10.0          # channel block half width (side wall 3.4 at the groove bottom)
DV_FLARE = ((280.0, 1.2), (281.0, 0.8), (282.0, 0.4))   # channel entry flare: 1 mm steps (z0, offset of the groove)
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
        y_out = s * (B14.HW - SHELL_T - GAP)
        y_in = s * (B14.HW - SHELL_T - GAP - 7.0)
        zc = (w0 + w1) / 2
        rec["now"] = "M2.5 x 10 through the front side wall"
        ya, yb = sorted((y_in, s * (B14.HW - SHELL_T + 0.1)))
        yc, yd = sorted((y_in, y_out))
        return [B14.box((-3, ya, zc - 4.5), (B14.XS, yb, zc + 4.5), rgb),
                B14.poly("z", zc - 4.5, zc + 4.5, [B14.W(x=-3, y=y_in), B14.W(x=-3, y=s * (B14.HW - SHELL_T + 0.1)),  # 45 deg
                                                   B14.W(x=-3 - abs(y_in - s * (B14.HW - SHELL_T + 0.1)),            # under
                                                         y=s * (B14.HW - SHELL_T + 0.1))], rgb),                    # the tab
                PROFI("y", yc, yd, rrect(zc - 4.5, zc + 4.5, B14.XS - 0.5, SEAM_X + 3.5, 0), [], rgb, [(zc, SEAM_X, TAP_R)])]
    rec["now"] = {"-z": "lid: M2.5 into the frame wall pads" if abs(abs(face) - 54.85) < 0.01 else "crest: glued",
                  "+x": "head seam bottom: removed", "-x": "face plate: M2.5 into head front bosses"}.get(ins, "?")
    return []


def _frame_rail(s, c):
    """build14.frame_rail made bigger (owner 2026-11-27): pedestal 10 wide d 0 .. DV_NECK_D with a 1 mm 45 deg root
    chamfer on the wall, dovetail neck 2 DV_NECK_HW at d DV_NECK_D -> tip at d + DV_DEPTH (flank slope DV_SLOPE),
    z 270 .. 316 (the rail top is the covers' stop), one 45 deg wedge under it down to the wall (prints +z)"""
    W, poly = B14.W, B14.poly
    y = lambda d: s * (60.0 + d)
    n0, n1 = DV_NECK_D, DV_NECK_D + DV_DEPTH
    w0, w1 = DV_NECK_HW, DV_NECK_HW + DV_SLOPE * DV_DEPTH
    return [poly("z", 270, 316, [W(x=c - 6, y=y(0)), W(x=c + 6, y=y(0)), W(x=c + 5, y=y(1)), W(x=c + 5, y=y(n0)),
                                 W(x=c - 5, y=y(n0)), W(x=c - 5, y=y(1))], PRINT),
            poly("z", 270, 316, [W(x=c - w0, y=y(n0)), W(x=c + w0, y=y(n0)), W(x=c + w1, y=y(n1)), W(x=c - w1, y=y(n1))],
                 PRINT),
            poly("x", c - 6, c + 6, [W(y=y(0), z=270), W(y=y(n1), z=270), W(y=y(0), z=270 - n1)], PRINT)]


def _cover_channel(s, c, rgb):
    """female dovetail block for _frame_rail with DOVE_GAP on every face; the cover wall inner face is at d 15 -
    SHELL_T = 13, the block runs 0.4 into it with a 1 mm 45 deg root chamfer. The bottom 3 mm is a stepped entry flare
    (DV_FLARE: groove flanks out by 1.2 / 0.8 / 0.4, bottom deeper by up to 0.8) so the cover finds the rail top;
    closed top 316 .. 318 sits on the rail top; 45 deg gussets under the side walls stay 0.4 clear of the rail"""
    W, poly, box = B14.W, B14.poly, B14.box
    g, hw = DOVE_GAP, DV_BLOCK_HW
    y = lambda d: s * (60.0 + d)
    a0, a1 = DV_NECK_D + g, DV_NECK_D + DV_DEPTH + g        # mouth face, groove bottom
    dw = 15.0 - SHELL_T                                      # cover wall inner face (d)
    dt = dw + 0.4
    k = 1.0 / math.sqrt(1.0 + DV_SLOPE ** 2)                 # flank normal (k, -DV_SLOPE k)
    fx = lambda d, e=0.0: DV_NECK_HW + (g + e) * k + (d - DV_NECK_D + (g + e) * DV_SLOPE * k) * DV_SLOPE

    def loop(e):
        b = a1 + min(e, dw - 0.2 - a1)
        return [W(x=c - hw, y=y(a0)), W(x=c - fx(a0, e), y=y(a0)), W(x=c - fx(b, e), y=y(b)), W(x=c + fx(b, e), y=y(b)),
                W(x=c + fx(a0, e), y=y(a0)), W(x=c + hw, y=y(a0)), W(x=c + hw, y=y(dw - 1)), W(x=c + hw + 1, y=y(dw)),
                W(x=c + hw + 1, y=y(dt)), W(x=c - hw - 1, y=y(dt)), W(x=c - hw - 1, y=y(dw)), W(x=c - hw, y=y(dw - 1))]
    zs = [z for z, _ in DV_FLARE] + [DV_FLARE[-1][0] + 1.0]
    out = [poly("z", z0, z1, loop(e), rgb) for (z0, e), z1 in zip(DV_FLARE, zs[1:])]
    out.append(poly("z", zs[-1], 316, loop(0.0), rgb))
    out.append(box((c - hw - 1, min(y(a0), y(dt)), 316), (c + hw + 1, max(y(a0), y(dt)), 318), rgb))
    rail_hw = DV_NECK_HW + DV_SLOPE * DV_DEPTH
    for x0, x1 in ((c - hw, c - rail_hw - 0.65), (c + rail_hw + 0.65, c + hw)):
        out.append(poly("x", x0, x1, [W(y=y(a0), z=280), W(y=y(dt), z=280), W(y=y(dt), z=280 - (dt - a0))], rgb))
    LOG.setdefault("dovetail", []).append(dict(
        side=s, x=c, rail=dict(neck_d=DV_NECK_D, tip_d=DV_NECK_D + DV_DEPTH, neck_w=2 * DV_NECK_HW, tip_w=2 * rail_hw,
                               pedestal_w=10.0, z=(270, 316)),
        channel=dict(mouth_d=round(a0, 2), bottom_d=round(a1, 2), w_at_mouth=round(2 * fx(a0), 2),
                     w_at_bottom=round(2 * fx(a1), 2), block_w=2 * hw, side_wall_min=round(hw - fx(a1), 2),
                     behind_bottom=round(dw - a1, 2), z=(280, 318), flare=[list(f) for f in DV_FLARE]), gap=g))
    return out


_HOOK0 = B14.hook
SEAM_X = 18.7                # head seam side screw: x of its axis (the old window x 17 .. 20.4)
from build6 import PROFI
from shapes import rrect, circle_loop
import build16 as B16, build19 as B19, build20 as B20, x430_holes as X


def tongue(links):
    """head seam tongue (back part x 5 .. XS + 6, y +-10, under the top wall): 0.95 -> TONGUE_T thick; the part under
    the front part's top wall (now SHELL_T thick, inner face ZH + HH - SHELL_T) gets its top TONGUE_GAP below it
    (was 0.15), the back half runs 0.3 into its own top wall"""
    hb = links["head"]["parts"]["head_back"]
    top = B14.ZH + B14.HH - SHELL_T
    n = 0
    for i, x in enumerate(hb):
        if x[0] == "rprism" and x[1][0] == "z" and abs(x[1][1] - (B14.TOP_IN - 1.35)) < 1e-6:
            a = x[1]
            hi = top + 0.3 if a[2] > B14.TOP_IN else top - TONGUE_GAP
            hb[i] = (x[0], (a[0], top - TONGUE_GAP - TONGUE_T, hi, a[3], a[4], a[5]), x[2]); n += 1
    assert n == 2, n
    return dict(solids=n, gap=(0.15, TONGUE_GAP), thick=(round(1.35 - TONGUE_GAP, 2), TONGUE_T))


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


# ---------------------------------------------------------------- shell walls 1.2 -> 2.0 (2026-11-27, second pass)
QUAD = ((1, 1, 0, 90), (-1, 1, 90, 180), (-1, -1, 180, 270), (1, -1, 270, 360))      # build6.box_shell corners


def _thicken_shell(ss, axis, cu, cv, A, B, R, t1=SHELL_T, t0=B14.T, layer_holes=None):
    """build6.box_shell walls t0 -> t1 grown inward (outer surface kept): corner arcs r_in R - t0 -> R - t1, flat
    panels get their inner face moved by t1 - t0 (outline, cut-outs, holes kept). layer_holes(panel) may return
    holes for the added layer; then the panel stays and the layer is a separate solid with those holes"""
    from shapes import annulus_sector
    iu, iv = build.UVI[axis]
    nu, nv = "xyz"[iu], "xyz"[iv]
    corners = []
    for fu, fv, s0, s1 in QUAD:
        c = (cu + fu * (A - R), cv + fv * (B - R))
        corners.append((np.asarray(annulus_sector(*c, R - t0, R, s0, s1)), annulus_sector(*c, R - t1, R, s0, s1)))
    spans = [(nu, cu + A - t0, cu + A, cu + A - t1, cu + A), (nu, cu - A, cu - A + t0, cu - A, cu - A + t1),
             (nv, cv + B - t0, cv + B, cv + B - t1, cv + B), (nv, cv - B, cv - B + t0, cv - B, cv - B + t1)]
    out, nc, npn = [], 0, 0
    for s in ss:
        if s[0] == "rprism":
            a = s[1]
            if a[0] == axis and not a[4]:
                hit = [new if np.allclose(old, np.asarray(a[3])) else new[::-1] for old, new in corners
                       if len(old) == len(a[3]) and (np.allclose(old, np.asarray(a[3])) or np.allclose(old[::-1], np.asarray(a[3])))]
                if hit:
                    out.append(("rprism", (a[0], a[1], a[2], hit[0], a[4], a[5]), s[2])); nc += 1
                    continue
            hit = [sp for sp in spans if a[0] == sp[0] and abs(a[1] - sp[1]) < 1e-6 and abs(a[2] - sp[2]) < 1e-6]
            if hit:
                n_, o0, o1, b0, b1 = hit[0]
                npn += 1
                lh = layer_holes(s) if layer_holes else None
                if lh is None:
                    out.append(("rprism", (a[0], b0, b1, a[3], a[4], a[5]), s[2]))
                else:
                    l0, l1 = (b0, o0) if abs(b0 - o0) > 1e-9 else (o1, b1)
                    out += [s, ("rprism", (a[0], l0, l1, a[3], a[4], lh), s[2])]
                continue
        out.append(s)
    ss[:] = out
    return nc, npn


def _thicken_rings(ss, axis, x0, x1, t_of):
    """build6.ring steps (rounded-rect rings along axis between x0 .. x1): inner loop offset T -> t_of(a0, a1, half_u,
    half_v); the outside (the 45 deg stepped chamfer / bevel) is kept"""
    from shapes import orient
    n = 0
    for i, s in enumerate(ss):
        if s[0] != "rprism" or s[1][0] != axis or len(s[1][4]) != 1 or not (x0 - 1e-6 <= s[1][1] and s[1][2] <= x1 + 1e-6):
            continue
        a = s[1]
        us, vs = [p[0] for p in a[3]], [p[1] for p in a[3]]
        if len(a[3]) != 4 or abs(min(p[0] for p in a[4][0]) - min(us) - B14.T) > 1e-6:
            continue
        t = t_of(a[1], a[2], (max(us) - min(us)) / 2, (max(vs) - min(vs)) / 2)
        inner = rrect(min(us) + t, max(us) - t, min(vs) + t, max(vs) - t, max(a[3][0][2] - t, 0.5))
        ss[i] = ("rprism", (a[0], a[1], a[2], a[3], [orient(inner, False)], a[5]), s[2]); n += 1
    return n


def shell_walls(links):
    """head (back / front), crest, ear cups and shirt covers: shell walls T 1.2 -> SHELL_T 2.0, inward. Stepped 45 deg
    rings get BEV_T radial (2.0 normal), the two rings round the face plate keep FACE_GAP to it (plate 0.2 smaller);
    obsolete hook windows in the walls are filled (ear hook slots, crest slots, seam bottom window); what sat on an
    inner face moves with it (684 seats get their r 4.5 through the new layer, amp board 0.8 up)"""
    H, E, T_ = links["head"]["parts"], links["ears"]["parts"], links["torso"]["parts"]
    hb, hf = H["head_back"], H["head_front"]
    HW, HH, HR, ZH, HD = B14.HW, B14.HH, B14.HR, B14.ZH, B14.HD
    rec = {}
    ex, ez = B16.EAR_C

    def seat(s):        # head_back side walls: ear axle hole r 2.5 -> 684 seat r 4.5 through the added layer
        hs = s[1][5]
        return [(h[0], h[1], 4.5) if abs(h[0] - ez) < 1e-6 and abs(h[1] - ex) < 1e-6 else h for h in hs] \
            if any(abs(h[0] - ez) < 1e-6 and abs(h[1] - ex) < 1e-6 for h in hs) else None
    rec["head_back"] = _thicken_shell(hb, "x", 0.0, ZH, HW, HH, HR, layer_holes=seat)
    rec["head_front"] = _thicken_shell(hf, "x", 0.0, ZH, HW, HH, HR)
    assert rec["head_back"] == (4, 4) and rec["head_front"] == (4, 4), rec
    # back plate inside the chamfer
    n = 0
    for i, s in enumerate(hb):
        if s[0] == "rprism" and s[1][0] == "x" and abs(s[1][1] + HD) < 1e-6 and abs(s[1][2] - (-HD + B14.T)) < 1e-6:
            a = s[1]; hb[i] = ("rprism", (a[0], a[1], -HD + SHELL_T, a[3], a[4], a[5]), s[2]); n += 1
    assert n == 1
    # face plate 0.2 smaller per side, rim rings keep FACE_GAP to it
    fp = H["face_plate"]
    a = fp[0][1]
    fa, fb, fr = B14.FA - 0.2, B14.FB - 0.2, B14.FR - 0.2
    fp[0] = PROFI("x", a[1], a[2], rrect(-fa, fa, B14.FZ - fb, B14.FZ + fb, fr), a[4], fp[0][2], a[5])

    def t_front(x0, x1, hu, hv):
        if x1 <= B14.FX + 1e-6:
            return BEV_T
        return max(B14.T, min(BEV_T, hu - fa - FACE_GAP, hv - fb - FACE_GAP))
    rec["rings_back"] = _thicken_rings(hb, "x", -HD, -HD + B14.CH, lambda *a: BEV_T)
    rec["rings_front"] = _thicken_rings(hf, "x", B14.XB, HD, t_front)
    assert rec["rings_back"] == B14.CH and rec["rings_front"] == B14.BEV, rec
    # obsolete windows: ear hook slots (back side walls), crest hook slots (top walls), seam bottom window (front floor);
    # nod rod slot in the back floor y 19.5 -> 19.4 (rod y 19.8 .. 22.8: 0.4 / 0.5 per side)
    filled = 0
    top_z, bot_z = ZH + HH - SHELL_T, ZH - HH + SHELL_T
    for part, ss in (("head_back", hb), ("head_front", hf)):
        for i, s in enumerate(ss):
            if s[0] != "rprism" or not s[1][4]:
                continue
            a = s[1]
            side = a[0] == "y" and min(abs(a[1]), abs(a[2])) >= HW - SHELL_T - 1e-6 and part == "head_back"
            top = a[0] == "z" and abs(a[1] - top_z) < 1e-6
            floor = a[0] == "z" and abs(a[2] - bot_z) < 1e-6 and part == "head_front"
            if side or top or floor:
                filled += len(a[4])
                ss[i] = ("rprism", (a[0], a[1], a[2], a[3], [], a[5]), s[2])
            elif a[0] == "z" and abs(a[2] - bot_z) < 1e-6:
                ss[i] = ("rprism", (a[0], a[1], a[2], a[3], [[(p[0], 19.4 if abs(p[1] - 19.5) < 1e-6 else p[1], p[2])
                                                                for p in l] for l in a[4]], a[5]), s[2])
    rec["windows_filled"] = filled
    H["amp_MAX98357A"] = [B16.translate(x, (0, 0, SHELL_T - B14.T)) for x in H["amp_MAX98357A"]]
    # crest (glued on the head top)
    cr = H["crest"]
    cu, A_ = (B14.XB - 45.0) / 2, (B14.XB + 45.0) / 2
    rec["crest"] = _thicken_shell(cr, "z", cu, 0.0, A_, 27.0, 15.0)
    for i, s in enumerate(cr):
        if s[0] == "rprism" and s[1][0] == "z" and abs(s[1][1] - (B14.CZ1 - B14.T)) < 1e-6:
            a = s[1]; cr[i] = ("rprism", (a[0], B14.CZ1 - SHELL_T, a[2], a[3], a[4], a[5]), s[2])
    # ear cups: rim r 28.8 .. 30 -> 28 .. 30, lid 1.2 -> 2.0 (inward)
    from shapes import annulus_sector
    old = np.asarray(annulus_sector(ez, 0, 30 - B14.T, 30, 8, 352))
    for side, sg in (("L", 1), ("R", -1)):
        ear, n = E[f"ear_{side}"], 0
        for i, s in enumerate(ear):
            a = s[1]
            if s[0] != "rprism" or a[0] != "y":
                continue
            if len(a[3]) == len(old) and np.allclose(old, np.asarray(a[3])):
                ear[i] = ("rprism", (a[0], a[1], a[2], annulus_sector(ez, 0, 30 - SHELL_T, 30, 8, 352), a[4], a[5]), s[2])
                n += 1
            elif abs(a[2] - a[1] - B14.T) < 1e-6 and abs(max(abs(a[1]), abs(a[2])) - (HW + B14.EAR_T + 0.5)) < 1e-6:
                lo, hi = (a[2] - SHELL_T, a[2]) if sg > 0 else (a[1], a[1] + SHELL_T)
                ear[i] = ("rprism", (a[0], lo, hi, a[3], a[4], a[5]), s[2]); n += 1
        assert n == 2, (side, n)
    # shirt covers: box shell + shoulder band, lid plug clipped to the new inside at the top
    for half in ("front", "back"):
        rec[f"torso_cover_{half}"] = _thicken_shell(T_[f"torso_cover_{half}"], "z", B16.CU, 0.0, B16.CA, B16.CY, B16.CR_)
        assert rec[f"torso_cover_{half}"] == (2, 3), rec
    saved, B16.T = B16.T, SHELL_T
    try:
        bands = dict(zip(("front", "back"), B16.shoulder_lofts()))
        do, di = B16.top_inset()
    finally:
        B16.T = saved
    for half, band in bands.items():
        cv = T_[f"torso_cover_{half}"]
        k = [i for i, s in enumerate(cv) if s[0] == "loft"]
        assert len(k) == 1
        cv[k[0]] = band
    clip = B16.ccw(B16.cover_outline(di + 0.5))
    lid, n = T_["lid"], 0
    for i, x in enumerate(lid):
        if x[0] == "rprism" and x[1][0] == "z" and x[1][1] >= 329.9 and x[1][2] <= 336.4 and len(x[1][3]) >= 8:
            a = x[1]
            poly = B16.sh_clip([(p[0], p[1]) for p in a[3]], clip)
            lid[i] = PROFI("z", a[1], a[2], [(u, v, 0) for u, v in poly], a[4], x[2], a[5]); n += 1
    rec["lid_plug_clip"] = dict(solids=n, cover_inner_inset_at_top=round(di, 2))
    rec["t"] = (B14.T, SHELL_T, round(BEV_T, 2))
    return rec


def shorts_notches(links):
    """shorts flange notches over the rails (the shorts drop in from above before the covers): x c +- 5.3 ->
    c +- (rail root 6 + SHORTS_GAP); depth |y| 72.2 stays (rail tip d 11.5 -> 0.7 clear)"""
    sh = links["torso"]["parts"]["shorts"]
    hw, n = 6.0 + SHORTS_GAP, 0

    def fx(p):
        for c in RAIL_X.values():
            if abs(abs(p[0] - c) - 5.3) < 1e-6:
                return c + math.copysign(hw, p[0] - c)
        return p[0]
    for i, x in enumerate(sh):
        if x[0] == "rprism" and x[1][0] == "z" and x[1][4]:
            a = x[1]
            inn = [[(fx(p), p[1]) + tuple(p[2:]) for p in l] for l in a[4]]
            n += sum(fx(p) != p[0] for l in a[4] for p in l)
            sh[i] = (x[0], (a[0], a[1], a[2], a[3], inn, a[5]), x[2])
    assert n == 16, n
    return dict(points=n, notch_half_w=(5.3, hw))


def _seg_dist(P, Q):
    """min distance from the points P (n, 2) to the closed polyline Q (m, 2)"""
    A, B = Q, np.roll(Q, -1, axis=0)
    d = B - A
    t = np.clip(((P[:, None, :] - A[None]) * d[None]).sum(-1) / np.maximum((d * d).sum(-1), 1e-12)[None], 0, 1)
    return float(np.linalg.norm(P[:, None, :] - (A[None] + t[..., None] * d[None]), axis=-1).min())


def _loop_gap(P, Q):
    return min(_seg_dist(P, Q), _seg_dist(Q, P))


def teardrops(links, names):
    """holes whose axis lies across the part's print-up axis -> teardrop loops with the 45 deg point up (coarse print:
    no sagging roof). The point must leave min(SHELL_T, wall of the round hole) to the outline, to the cut-outs and to
    the other holes; if the full point does not, it is cut flat at the highest level that does (flat top >= r + 0.3
    above the centre, a short bridge); otherwise the hole stays round. Runs last (earlier passes find holes by radius)."""
    from matplotlib.path import Path
    from shapes import orient
    table = {r["part"]: r for r in json.load(open(os.path.join(PART_DIR, "parts.json")))}
    rec = {}
    for ln, L in links.items():
        for pn, ss in L["parts"].items():
            if pn not in names:
                continue
            up = UP.get(pn, (table[pn]["up"],))[0]
            k, sg = "xyz".index(up[1]), (1 if up[0] == "+" else -1)
            n = flat = kept = 0
            walls = []
            for i, s in enumerate(ss):
                if s[0] != "rprism" or s[1][0] == up[1] or not s[1][5]:
                    continue
                a = s[1]
                iu, iv = build.UVI[a[0]]
                tip = np.array((sg, 0.0) if iu == k else (0.0, sg))
                sp = build4.sampled(s)
                outer = np.asarray(sp[1][3])[:, :2]
                inn = [np.asarray(l)[:, :2] for l in (sp[1][5] if sp[0] == "prismh" else [])]
                holes, loops = [], list(a[4])
                for j, h in enumerate(a[5]):
                    c, r = np.array(h[:2]), h[2]
                    ring = np.c_[c[0] + r * np.cos(np.linspace(0, 2 * np.pi, 48, endpoint=False)),
                                 c[1] + r * np.sin(np.linspace(0, 2 * np.pi, 48, endpoint=False))]
                    others = [o for jj, o in enumerate(a[5]) if jj != j]

                    def wall(P):
                        w = [_loop_gap(P, outer)] + [_loop_gap(P, q) for q in inn]
                        w += [float(np.hypot(P[:, 0] - o[0], P[:, 1] - o[1]).min() - o[2]) for o in others]
                        return min(w)
                    if not Path(outer).contains_points(ring).all():
                        holes.append(h); kept += 1; continue
                    need = min(SHELL_T, wall(ring))
                    td = np.asarray(B16.teardrop(h[0], h[1], r, tuple(tip)))[:, :2]
                    best = None
                    for hh in np.arange(r * math.sqrt(2), r + 0.3 - 1e-9, -0.1):
                        nrm = np.array((-tip[1], tip[0]))
                        box = B16.ccw([tuple(c + tip * hh + nrm * 3 * r), tuple(c + tip * hh - nrm * 3 * r),
                                       tuple(c - tip * 3 * r - nrm * 3 * r), tuple(c - tip * 3 * r + nrm * 3 * r)])
                        P = np.asarray(B16.sh_clip([tuple(p) for p in td], box))
                        if Path(outer).contains_points(P).all() and not any(Path(q).contains_points(P).any() for q in inn) \
                                and wall(P) >= need - 1e-6:
                            best = (hh, P); break
                    if best is None:
                        holes.append(h); kept += 1; continue
                    loops.append(orient([(u, v, 0) for u, v in best[1]], False))
                    n += 1; flat += best[0] < r * math.sqrt(2) - 1e-6
                    walls.append(round(wall(best[1]), 2))
                ss[i] = ("rprism", (a[0], a[1], a[2], a[3], loops, holes), s[2])
            if n or kept:
                rec[pn] = dict(up=up, teardrops=n, flat_top=int(flat), kept_round=kept, min_wall=min(walls) if walls else None)
    return rec


# no support (owner 2026-11-28: support under a joint breaks it when pulled off), export_parts voxel check:
# * head_front (prints +x): the 45 deg gusset under each face plate screw boss (x 34.85..40) started in the air, 8 mm
#   from the web under the boss row (an island until x 40): web widened so its 45 deg underside runs from the bed
#   (seam plane x 13) to the boss corner, inside the boss height
# * torso_frame right shelf lip (3 mm flat ledge, no gusset like the left ledge has): 45 deg chamfer under it
# * torso_frame OpenCR posts: base (z ..227.5) and slotted post (227.5..) met on one plane, which the 1 mm voxel grid
#   read as an empty layer (2 x 22 mm2): the post reaches 0.05 into the base (shape unchanged, the board slot keeps its floor)
# * torso_frame shoulder bosses: teardrop point line 5 mm out of the wall -> flat 2 above the point on the free 2 mm
# Left as it is: torso cover dovetail stop (a 12.5 mm roof anchored on three sides: a bridge).
def no_support(links):
    out = {}
    hf = links["head"]["parts"]["head_front"]
    x_bed = min(build4.aabb(build4.sampled(x))[0][0] for x in hf)
    rec = []
    for i, x in enumerate(hf):
        a = x[1]
        if x[0] == "rprism" and a[0] == "z" and len(a[3]) == 3 and abs(a[1] - 446.5) < 0.01 and abs(a[2] - 453.5) < 0.01:
            xs = [q[0] for q in a[3]]; ys_ = [q[1] for q in a[3]]
            xt = max(xs)                                         # boss back face
            g = 1 if sum(ys_) > 0 else -1
            yt = min(ys_, key=abs)                               # boss corner toward the head centre
            web = next(w for w in hf if w[0] == "rprism" and w[1][0] == "z" and abs(w[1][1] - 445.0) < 0.01 and len(w[1][3]) == 3
                       and sum(q[1] for q in w[1][3]) * g > 0)     # the web under the boss row (45 deg underside)
            wx = max(q[0] for q in web[1][3])
            yw_c = g * min(abs(q[1]) for q in web[1][3] if abs(q[0] - wx) < 1e-6)    # its corner at the boss back face
            y0 = g * (max(abs(q[1]) for q in web[1][3]) - 0.4)   # 0.4 into the side wall layer (the web goes 0.8 in)
            yb = yt + g * (xt - x_bed)                            # 45 deg line through the boss corner, at the bed
            # the band between that line and the web's underside (0.1 into the web), 0.05 into the boss
            yh = lambda xx: yw_c + g * (wx - xx) + g * 0.1
            x_h0 = wx + 0.1 - (y0 - yw_c) / g
            hf[i] = PROFI("z", a[1], a[2], [(x_bed, y0, 0.0), (x_bed, yb, 0.0), (xt, yt, 0.0), (xt + 0.05, yt, 0.0),
                                            (xt + 0.05, yh(xt + 0.05), 0.0), (x_h0, y0, 0.0)], [], x[2])
            rec.append(dict(side=g, old=[(round(q[0], 2), round(q[1], 2)) for q in a[3]], bed=(round(x_bed, 2), round(yb, 2)), corner=(xt, yt), wall_y=round(y0, 2)))
    out["head_front_web"] = rec
    tf = links["torso"]["parts"]["torso_frame"]; rgb = tf[0][2]
    lips = [x for x in tf if x[0] == "rprism" and x[1][0] == "z" and abs(x[1][1] - 268.0) < 0.01 and abs(x[1][2] - 272.0) < 0.01
            and abs(max(abs(q[1]) for q in x[1][3]) - 55.0) < 0.01 and max(abs(q[1]) for q in x[1][3]) - min(abs(q[1]) for q in x[1][3]) < 3.01]
    out["shelf_lip"] = []
    for x in lips:
        xs = [q[0] for q in x[1][3]]; g = 1 if x[1][3][0][1] > 0 else -1
        yw, yt = g * 55.0, g * 52.0
        tf.append(PROFI("x", min(xs), max(xs), [(yw + g * 0.4, 268.05, 0.0), (yt, 268.05, 0.0), (yw + g * 0.4, 268.05 - 3.45, 0.0)], [], rgb))
        out["shelf_lip"].append(dict(x=(min(xs), max(xs)), y=(yw, yt), chamfer_z=(round(268.05 - 3.45, 2), 268.05)))
    n = 0
    for i, x in enumerate(tf):
        a = x[1]
        if x[0] == "rprism" and a[0] == "z" and abs(a[1] - 227.5) < 1e-6 and len(a[3]) == 8 and min(q[0] for q in a[3]) < -70:
            tf[i] = (x[0], (a[0], a[1] - 0.05, a[2], a[3], a[4], a[5]), x[2]); n += 1
    out["opencr_posts_overlap"] = n
    # shoulder bosses (y 50..55 inside the side walls, teardrop point down at z 296.1): the point line ran 5 mm from the
    # wall; the 2 mm next to the free face is cut flat 2 above the point (a 2 mm ledge on the rest of the boss, the round
    # part and the shaft hole untouched), the wall side reaches 0.05 into the wall (one solid on the voxel grid)
    out["shoulder_boss"] = []
    for i, x in enumerate(list(tf)):
        a = x[1]
        if x[0] == "rprism" and a[0] == "y" and abs(abs(a[1] + a[2]) / 2 - 52.5) < 0.01 and abs(a[2] - a[1] - 5.0) < 0.01 and len(a[3]) == 20:
            zt = min(q[0] for q in a[3])
            g = 1 if a[1] > 0 else -1
            k = tf.index(x)
            wall_side = (g * 52.0, g * 55.05); free = (g * 50.0, g * 52.05)
            tf[k] = PROFI("y", *wall_side, a[3], a[4], x[2], a[5])
            cut = B20._cut(PROFI("y", *free, a[3], a[4], x[2], a[5]), (0.0, 0.0, -1.0), -(zt + 2.0))
            tf.append(cut)
            out["shoulder_boss"].append(dict(y=g, point_z=round(zt, 2), free_face_flat_z=round(zt + 2.0, 2), shaft_hole_kept=len(cut[1][4]) + len(cut[1][5])))
    return out


def build_model():
    LOG.clear()
    saved = (B14.hook, B14.cover_channel, B14.frame_rail, dict(B14.RAIL_X))
    B14.hook, B14.cover_channel, B14.frame_rail = _hook, _cover_channel, _frame_rail
    B14.RAIL_X.update(RAIL_X)
    try:
        links, joints, fasteners = BH.build_model()
    finally:
        B14.hook, B14.cover_channel, B14.frame_rail = saved[:3]
        B14.RAIL_X.update(saved[3])
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
    LOG["shell_walls"] = shell_walls(links)
    LOG["shorts_notches"] = shorts_notches(links)
    LOG["no_support"] = no_support(links)
    LOG["teardrops"] = teardrops(links, body_names())
    return links, joints, fasteners


def body_names():
    """torso / head printed parts this file writes (rows of out20/print_parts/parts.json)"""
    return [r["part"] for r in json.load(open(os.path.join(PART_DIR, "parts.json")))
            if r["link"] in BODY_LINKS and not r["part"].startswith("arm_") and r["part"] not in FROZEN + GONE]


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
    names = body_names()
    done = export_body(links, names)
    LOG["written"] = done
    LOG["missing"] = sorted(set(names) - set(done))
    json.dump(LOG, open(os.path.join(PART_DIR, "..", "build21_body_log.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in LOG.items() if k != "hooks"}, indent=1, default=str))
    print("hooks", len(LOG.get("hooks", [])))
