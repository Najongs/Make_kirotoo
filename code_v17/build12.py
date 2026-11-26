"""v12 = v11 with the head shell split at x = 13 into a back part (bevel-free, carries the neck cheeks, ribs
and nod servo) and a front part (face bevel, catch bars); snap hooks on the back part click into windows hidden
under the ears and under the head, the crest bridges the seam on top.

v11 = v10 with steel shafts in ball bearings (688ZZ, 8x16x5) for the shoulders, the arm crank and a new
neck: the head pivots on a steel shaft in a two-bearing yoke on the lid, and is nodded by a servo inside the head
through a crank and rod anchored on the yoke. Link pins are 3 mm steel dowels.

v10 = v9 with the head pan removed (head only nods, its servo half sunk in the torso on a mount that
stands on the lid) and the arms swinging in opposite phase from one servo: crank shaft under the shelf, two
rods, a lever on each shoulder shaft.

v9 = v8 with (1) every printed part in one colour (no filament change mid-print: separate navy
shorts ring, white shirt covers, no shank band) and (2) a short, stiff neck: the tilt servo sits on a bracket on
the pan horn, exposed, and the closed flat head bottom grips its horn / idler with two cheeks.

v8 = v7 with the head, face, ears, crest, arms and shorts made closer to the KIRO mascot.

v7: printed parts join each other without screws. Screws are used only where a part meets a servo
(case / horn screws from the ROBOTIS kit).

Joints between printed parts
- merged prints: pelvis + side walls + back wall = torso_frame; head frame plates are part of the head
  shell; foot bracket is part of the foot sole.
- dovetail slides: torso cover halves slide down onto rails on the frame walls (trapped from above by the
  lid cap). Arms slide down onto a T-rail on the front cover and lock with a tongue latch.
- snap hooks (45 deg lead-in and return, so they can be pulled apart again): lid -> frame windows,
  face plate -> catch bars in the head, crest -> head top wall, ears -> head side walls.
- beam latch: thigh / shank shells sit on a pad on the frame front and a flexible beam clicks into a groove.
- press fits: shelf pins, eye / mouth inserts, ear discs, antennas, hands, camera pins + clip.

Electronics: OpenCR 1.0 (105 x 75 mm) stands in corner channels on the back of the frame (replaces the two
U2D2); Jetson on the shelf; Raspberry Pi Camera v2 behind the robot's left eye.
"""
import math, os, json
import build4, build5, build6, build
from build4 import RB, SERVO, CYL, DSC, PROF, HA, SL, XC, XM, CAM, RUBBER, ELEC
from build import STEEL
from build5 import SKY, NAVY, WHITE, ORANGE, DARK, PAN_Z, ZT
from build6 import (PROFI, plate, box_shell, ring, T, HW, HD, CH, hexloop)
from shapes import rrect, annulus_sector, circle_loop, orient

PRINT = build.PRINT
IDX, UVI = build.IDX, build.UVI
_V5_TORSO, _V5_NECK = build5.torso, build5.neck
_V6_LEG = build6.right_leg

HB, HT_ = 1.0, 1.6        # snap barb depth, hook arm thickness


# ---------------------------------------------------------------- generic helpers
def poly(axis, a0, a1, pts, rgb, inners=(), holes=()):
    """Prism along `axis` from world points (only the two in-plane coordinates are used)."""
    iu, iv = UVI[axis]
    loop = [(p[iu], p[iv], 0) for p in pts]
    inn = [[(p[iu], p[iv], 0) for p in l] for l in inners]
    return PROFI(axis, a0, a1, loop, inn, rgb, [(p[iu], p[iv], r) for p, r in holes])


def box(lo, hi, rgb, axis="z"):
    return plate(axis, lo, hi, rgb)


def W(**kw):
    """world point from named coordinates (missing ones = 0)"""
    return (kw.get("x", 0.0), kw.get("y", 0.0), kw.get("z", 0.0))


def hook(ins, flex, root, catch, face, w0, w1, rgb, t=HT_, b=HB):
    """Snap hook. ins: axis+sign the hook travels ('-x'); flex: axis+sign the barb points ('+y').
    root / catch: coordinates along the ins axis of the arm root and of the surface the barb hooks behind;
    face: coordinate along the flex axis of the arm's barb-side face; w0..w1: range along the third axis.
    The barb (45 deg in and out) occupies the last 3*b of the arm; its return ramp ends on `catch`."""
    ia, sa = ins[1], (1 if ins[0] == "+" else -1)
    fa, sf = flex[1], (1 if flex[0] == "+" else -1)
    wa = [a for a in "xyz" if a not in (ia, fa)][0]
    tip = catch + sa * 3 * b

    def P(i, f, w=0.0):
        d = {ia: i, fa: f, wa: w}
        return (d["x"], d["y"], d["z"])
    lo, hi = P(min(root, tip), min(face, face - sf * t), w0), P(max(root, tip), max(face, face - sf * t), w1)
    arm = box(lo, hi, rgb, wa)
    barb = poly(wa, w0, w1, [P(tip, face), P(tip - sa * b, face + sf * b), P(tip - sa * 2 * b, face + sf * b),
                             P(tip - sa * 3 * b, face)], rgb)
    return [arm, barb]


def hook_slot(flex, face, w0, w1, wa, t=HT_, b=HB, c=0.15):
    """Slot outline (list of world points, in the wall plane) for a hook through a wall: width t + b + c
    across the flex axis (arm can deflect away from the barb), 0.2 clearance along the width axis."""
    fa, sf = flex[1], (1 if flex[0] == "+" else -1)
    f0, f1 = sorted((face + sf * c, face - sf * (t + b)))
    a0, a1 = w0 - 0.2, w1 + 0.2

    def P(f, w):
        d = {fa: f, wa: w}
        return (d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0))
    return [P(f0, a0), P(f1, a0), P(f1, a1), P(f0, a1)]


def rect_cut(axis_pair, a0, a1, b0, b1):
    """Rectangle (world points) in the plane spanned by the two named axes."""
    A, B = axis_pair

    def P(a, b):
        d = {A: a, B: b}
        return (d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0))
    return [P(a0, b0), P(a1, b0), P(a1, b1), P(a0, b1)]


# ---------------------------------------------------------------- head (tilt link)  -- v8: closer to the mascot
HH, HR = 85.0, 70.0          # taller, rounder head (200 x 170, corner R70)
ZT9 = 348.5                  # neck pivot (steel shaft in the yoke bearings, 10 mm above the lid cap)
HGAP = 16.5                  # pivot -> head bottom (the yoke stays inside a 15.8 mm circle around the pivot)
NOD_S = (-28.75, 383.45)     # nod servo horn axis (x, z) inside the head; servo sits on the head floor
NOD_R, NOD_A = 14.0, (-14.75, 345.0)   # nod crank radius (pointing forward), rod anchor on the yoke
ZH = ZT9 + HGAP + HH         # head centre
BEV = 24                     # 45-degree front bevel (24 x 1 mm steps): the sky rim around the face
XB = HD - BEV                # x where the bevel starts (46)
FX = HD - 2.0                # face plate back face (front face flush with the rim at x 70)
FZ = ZH
TOP_IN = ZH + HH - T
CZ0, CZ1 = ZH + HH, ZH + HH + 18                     # crest: a low ridge front-to-back
FA, FB, FR = HW - BEV - T - 0.2, HH - BEV - T - 0.2, HR - BEV   # face plate 149 x 119, R46
EYE_L, EYE_Z = 37.5, FZ + 3.0                        # eyes 20 x 40, centres +-37.5
CAM_TOP = EYE_Z + 9.5        # Pi Camera v2: lens centre 9.5 mm below the board's top edge
CAM_HOLES = [(EYE_L + dy, CAM_TOP - dz) for dy in (-10.5, 10.5) for dz in (2.0, 14.5)]
SMILE = (FZ + 1.8, 23.1, 28.1, 228.4, 311.6)          # centre z, r_in, r_out, angles (wide shallow smile)
EAR_T = 22.0                 # ear thickness (y)


def eye_pill(y, grow=0.0):
    return rrect(y - 10 - grow, y + 10 + grow, EYE_Z - 20 - grow, EYE_Z + 20 + grow, 10 + grow)


XS = 13.0                    # head split plane: back part x < 13, front part x > 13 (each prints without support)


def head():
    P = {}
    bk, fr = [], []
    for i in range(CH):                                     # back: 10 mm stepped chamfer (as v6)
        o = CH - i
        bk.append(ring("x", -HD + i, -HD + i + 1, 0, ZH, HW - o, HH - o, HR - o, T, SKY))
    for i in range(BEV):                                    # front: 24 mm stepped bevel = rim around the face
        fr.append(ring("x", XB + i, XB + i + 1, 0, ZH, HW - i - 1, HH - i - 1, HR - i - 1, T, SKY))
    bk.append(PROF("x", -HD, -HD + T, rrect(-(HW - CH), HW - CH, ZH - (HH - CH), ZH + (HH - CH), HR - CH), SKY))
    ez = ZH - 5
    BOT_IN, HB = ZH - HH + T, ZH - HH
    crest_x = ((26, 34), (-34, -26))
    crest_slot = lambda s, x0, x1: [(p[0], p[1], CZ0) for p in hook_slot(f"{'+' if s > 0 else '-'}y", s * 12.0, x0, x1, "x")]
    ear_slots = {}
    for s, side in ((1, "+u"), (-1, "-u")):
        ear_slots[side] = [[(p[0], s * HW, p[2]) for p in hook_slot(f"{'+' if sx > 0 else '-'}x", sx * 12.0,
                                                                     ez - 4, ez + 4, "z")] for sx in (1, -1)]
    # seam snap hooks (on the back part, hidden under the ears / under the head) and their windows (front part)
    SZ = (ez + 12.0, ez + 18.0)
    side_win = {side: [[(x, s * HW, z) for x, z in ((17.0, SZ[0] - 0.2), (20.4, SZ[0] - 0.2), (20.4, SZ[1] + 0.2),
                                                         (17.0, SZ[1] + 0.2))]] for s, side in ((1, "+u"), (-1, "-u"))}
    bot_win = [(17.0, -5.2, 0), (20.4, -5.2, 0), (20.4, 1.2, 0), (17.0, 1.2, 0)]
    bk += box_shell("x", -HD + CH, XS, 0, ZH, HW, HH, HR, T, SKY,
                    cuts={"+v": [crest_slot(s, -34, -26) for s in (1, -1)], "+u": ear_slots["+u"], "-u": ear_slots["-u"],
                          "-v": [[(-10, 27.8, 0), (8, 27.8, 0), (8, 29.8, 0), (-10, 29.8, 0)],      # camera FFC slot
                                 [(-30, 19.5, 0), (4, 19.5, 0), (4, 23.3, 0), (-30, 23.3, 0)]]})     # nod rod slot
    fr += box_shell("x", XS, XB, 0, ZH, HW, HH, HR, T, SKY,
                    cuts={"+v": [crest_slot(s, 26, 34) for s in (1, -1)], "+u": side_win["+u"], "-u": side_win["-u"],
                          "-v": [bot_win]})
    for s in (1, -1):
        bk += hook("+x", f"{'+' if s > 0 else '-'}y", 0.0, 17.0, s * (HW - T - 0.15), SZ[0], SZ[1], SKY)
        y0, y1 = sorted((s * (HW - T - 0.15 - HT_), s * (HW - T + 0.3)))
        bk.append(box((-3, y0, SZ[0]), (0.5, y1, SZ[1]), SKY))                              # hook root
    bk += hook("+x", "-z", 3.0, 17.0, BOT_IN + 0.15, -5.0, 1.0, SKY, t=1.4)
    bk.append(box((0, -5, BOT_IN - 0.3), (3.5, 1, BOT_IN + 1.55), SKY))
    # locating tongue along the top wall (back part -> under the front part's top wall)
    bk.append(box((5, -10, TOP_IN - 1.35), (XS, 10, TOP_IN + 0.3), SKY))
    bk.append(box((XS, -10, TOP_IN - 1.35), (XS + 6, 10, TOP_IN - 0.15), SKY))
    # face plate stops: 45-degree ribs on the bevel, top and bottom (the plate back rests on their front ends)
    for sz in (1, -1):
        for y0, y1 in ((22.0, 29.0), (-29.0, -22.0)):
            pts = [W(x=FX, z=ZH + sz * 52), W(x=FX, z=ZH + sz * (HH - BEV + 0.8)), W(x=XB, z=ZH + sz * (HH - T)),
                   W(x=XB, z=ZH + sz * (HH - T + 0.3)), W(x=FX - (HH - T - 52), z=ZH + sz * (HH - T + 0.3))]
            fr.append(poly("y", y0, y1, pts, SKY))
    # catch bars for the side hooks of the face plate (+ 45-degree gussets toward the back = down in print)
    for s in (1, -1):
        y0, y1 = sorted((s * 72.15, s * (HW - T)))
        fr.append(box((40, y0, FZ - 5), (XB, y1, FZ + 5), SKY))
        fr.append(poly("z", FZ - 5, FZ + 5, [W(x=40, y=s * 75), W(x=40, y=s * (HW - T)),
                                             W(x=40 - (HW - T - 75), y=s * (HW - T))], SKY))
    # back part: inner ribs (to the split plane), cheeks on the neck shaft, nod servo floor pad
    for s in (1, -1):
        y0, y1 = sorted((s * 24, s * 27.5))
        bk.append(PROF("y", y0, y1, [(BOT_IN, -56, 0), (BOT_IN, XS, 0), (TOP_IN, XS, 0), (TOP_IN, -56, 0),
                                     (TOP_IN - 11, -HD + T, 0), (BOT_IN + 11, -HD + T, 0)], SKY))
        bk.append(PROF("y", y0, y1, [(HB + 0.6, -36, 0), (ZT9, -9.5, 0), (ZT9 - 9.5, 0, 0), (ZT9, 9.5, 0),
                                     (HB + 0.6, 9.5, 0)], SKY, holes=[(ZT9, 0, 4.0)]))
    bk.append(box((-41, -18, BOT_IN), (2, 18, BOT_IN + 3), SKY))                       # nod servo floor pad
    P["head_back"] = bk
    P["head_front"] = fr
    # nod servo inside the head (screwed to the floor pad), crank on its horn, rod down to the yoke anchor
    nz0 = ZH - HH + T + 3
    P["head_nod_servo_XC430"] = [SERVO(NOD_S[0] - 11.25, NOD_S[0] + 35.25, -17, 17, nz0, nz0 + 28.5, "y", XC)]

    # face plate: flush with the rim; eye / smile openings for press-fit inserts; two long side hooks;
    # camera posts + pins behind the robot's left eye
    mouth = annulus_sector(0, SMILE[0], SMILE[1], SMILE[2], SMILE[3], SMILE[4])
    fp = [PROFI("x", FX, HD, rrect(-FA, FA, FZ - FB, FZ + FB, FR), [eye_pill(EYE_L), eye_pill(-EYE_L), mouth], WHITE)]
    for s in (1, -1):
        fp += hook("-x", f"{'+' if s > 0 else '-'}y", FX, 40, s * 72.0, FZ - 4, FZ + 4, WHITE, t=2.0)
    for y, z in CAM_HOLES:
        fp.append(CYL((FX, y, z), "-x", 2.2, 3.0, WHITE))                   # standoff: board front 3 mm back
        fp.append(CYL((FX - 3.0, y, z), "-x", 1.0, 3.6, WHITE))             # pin through board + clip
    P["face_plate"] = fp

    lens = rrect(EYE_L - 4.5, EYE_L + 4.5, EYE_Z - 4.5, EYE_Z + 4.5, 0.5)  # 9 x 9 lens window
    P["eye_L"] = [PROFI("x", FX, HD, eye_pill(EYE_L), [lens], NAVY),
                  PROFI("x", HD, HD + 1.2, eye_pill(EYE_L, 1.0), [lens], NAVY)]
    P["eye_R"] = [PROF("x", FX, HD, eye_pill(-EYE_L), NAVY),
                  PROF("x", HD, HD + 1.2, eye_pill(-EYE_L, 1.0), NAVY)]
    P["mouth"] = [PROF("x", FX, HD, mouth, NAVY),
                  PROF("x", HD, HD + 1.2, annulus_sector(0, SMILE[0], SMILE[1] - 0.8, SMILE[2] + 0.8,
                                                         SMILE[3] - 2, SMILE[4] + 2), NAVY)]

    # Raspberry Pi Camera Module v2 behind the left eye
    cz0 = CAM_TOP - 23.86
    P["camera_pi_v2"] = [RB(FX - 4, FX - 3, EYE_L - 12.5, EYE_L + 12.5, cz0, CAM_TOP, CAM, "x", 1.0,
                            holes=[((0, y, z), 1.1) for y, z in CAM_HOLES]),
                         RB(FX - 3, FX + 2, EYE_L - 4.25, EYE_L + 4.25, EYE_Z - 4.25, EYE_Z + 4.25, CAM, "x", 0.5)]
    clip = [PROFI("x", FX - 6.6, FX - 5.0, rrect(EYE_L - 12.5, EYE_L + 12.5, CAM_TOP - 16.7, CAM_TOP, 1.0),
                  [rrect(EYE_L - 7.5, EYE_L + 7.5, CAM_TOP - 12.3, CAM_TOP - 4.2, 0.5)], DARK,
                  [(y, z, 1.0) for y, z in CAM_HOLES])]
    clip += [PROF("x", FX - 5.0, FX - 4.0, circle_loop(y, z, 2.2), DARK, [(y, z, 1.0)]) for y, z in CAM_HOLES]
    P["camera_clip"] = clip

    # ears: thicker cup, big sky disc, shorter antenna set further out
    yl = HW + EAR_T - 1.2                     # lid inner face
    ya_ = HW + 12.0                           # antenna axis
    for s, nm in ((1, "L"), (-1, "R")):
        ya, yb = sorted((s * HW, s * yl))
        la, lb = sorted((s * yl, s * (HW + EAR_T)))
        ear = [PROF("y", ya, yb, annulus_sector(ez, 0, 28.8, 30, 8, 352), NAVY),
               PROF("y", la, lb, circle_loop(ez, 0, 30), NAVY, holes=[(ez, 0, 8.0)]),
               plate("z", (-7, min(s * (ya_ - 8), s * yl), ez + 10), (7, max(s * (ya_ - 8), s * yl), ez + 26),
                     NAVY, holes=[((0, s * ya_, 0), 4.0)])]
        for sx in (1, -1):
            ear += hook(f"{'-' if s > 0 else '+'}y", f"{'+' if sx > 0 else '-'}x", s * yl, s * (HW - T),
                        sx * 12.0, ez - 4, ez + 4, NAVY)
        P[f"ear_{nm}"] = ear
        P[f"ear_disc_{nm}"] = [CYL((0, s * (HW + EAR_T), ez), "y" if s > 0 else "-y", 22, 1.5, SKY),
                               CYL((0, s * yl, ez), "y" if s > 0 else "-y", 8.0, 1.2, SKY)]
        P[f"antenna_{nm}"] = [CYL((0, s * ya_, ez + 12), "z", 4, 60, ORANGE)]

    # crest: a low rounded ridge running front to back, four hooks through the head top wall
    cu, A, B = (XB + (-45.0)) / 2, (XB + 45.0) / 2, 27.0
    cr = box_shell("z", CZ0, CZ1 - T, cu, 0.0, A, B, 15, T, SKY)
    cr.append(PROF("z", CZ1 - T, CZ1, rrect(cu - A, cu + A, -B, B, 15), SKY))
    for s in (1, -1):
        for x0, x1 in crest_x:
            cr += hook("-z", f"{'+' if s > 0 else '-'}y", CZ1 - T, TOP_IN, s * 12.0, x0, x1, SKY)
    P["crest"] = cr
    return P



# ---------------------------------------------------------------- torso
CU, CA, CY, CR_ = (build4.COV_XF + build4.COV_XB) / 2, (build4.COV_XF - build4.COV_XB) / 2, build4.COV_Y, build4.COV_R
RAIL_X = {"front": 12.0, "back": -22.0}       # cover rails on the frame walls
LID_HOOK_X = (-32.0, -10.0)
SHORTS_TOP = 237.0
SH_Z = 318.0                                   # shoulder (arm swing) axis height
ARM_SERVO = (12.0, 58.5, -52.0, -18.0, 241.0, 269.5)   # arm swing servo box (horn on the y -19 face)
CRANK = (47.25, 255.25)                        # crank shaft axis (x, z), along y
R_CRANK, L_CRANK = 16.75, 9.8                  # crank radii: right pin forward, left pin backward
R_LEV, L_LEV = 64.0, 37.45                     # lever tip x (shoulder axis at x 0)
YOKE_PEGS = [(x, y) for x in (-8.0, 8.0) for y in (-6.0, 6.0)]
ARM_Y = 93.3                                   # arm centre (1.8 mm air gap to the cover)
ARM_RAIL = (300.0, 314.0)                      # T-rail head on the cover, z
KEY = (284.6, 299.4)                           # keyhole in the arm's inner panel, z (final position)
OCR = dict(x0=-68.0, x1=-69.6, y=37.5, z0=229.0, z1=334.0)     # OpenCR 1.0 board (105 along z, 75 along y)


def frame_rail(s, c):
    """Male dovetail on the outer face of a frame wall (side s), centred at x = c, z 270..316."""
    y = lambda d: s * (60.0 + d)
    out = [box((c - 5, min(y(0), y(8.8)), 270), (c + 5, max(y(0), y(8.8)), 316), PRINT),
           poly("z", 270, 316, [W(x=c - 2.5, y=y(8.8)), W(x=c + 2.5, y=y(8.8)), W(x=c + 4, y=y(11.8)),
                                W(x=c - 4, y=y(11.8))], PRINT),
           poly("x", c - 5, c + 5, [W(y=y(0), z=270), W(y=y(8.8), z=270), W(y=y(0), z=261.2)], PRINT),
           poly("x", c - 2.5, c + 2.5, [W(y=y(0), z=270), W(y=y(11.8), z=270), W(y=y(0), z=258.2)], PRINT)]
    return out


def cover_channel(s, c, rgb):
    """Female dovetail block on the inner face of a cover side panel (open at the bottom, closed on top)."""
    y = lambda d: s * (60.0 + d)
    C = [W(x=c - 7, y=y(9.0)), W(x=c - 2.8, y=y(9.0)), W(x=c - 4.3, y=y(12.0)), W(x=c + 4.3, y=y(12.0)),
         W(x=c + 2.8, y=y(9.0)), W(x=c + 7, y=y(9.0)), W(x=c + 7, y=y(13.8)), W(x=c - 7, y=y(13.8))]
    out = [poly("z", 280, 316, C, rgb),
           box((c - 7, min(y(9.0), y(13.8)), 316), (c + 7, max(y(9.0), y(13.8)), 318), rgb)]
    for x0, x1 in ((c - 7, c - 4.5), (c + 4.5, c + 7)):
        out.append(poly("x", x0, x1, [W(y=y(9.0), z=280), W(y=y(13.8), z=280), W(y=y(13.8), z=275.2)], rgb))
    return out


def torso():
    P, F = _V5_TORSO()
    for k in ("pelvis_plate", "torso_side_R", "torso_side_L", "electronics_shelf", "top_plate", "torso_cover_front",
              "torso_cover_back", "shorts_band", "u2d2_A", "u2d2_B") + \
            tuple(f"arm_{n}_{p}" for n in "LR" for p in ("sleeve", "shoulder", "hand")):
        P.pop(k, None)

    # ---- torso frame: pelvis plate + side walls + back wall, one print (pelvis on the bed)
    fr = [RB(-65, 50, -62, 62, 225, 235, PRINT, "z", 20, [((0, -40, 0), 18.5), ((0, 40, 0), 18.5)])]
    fr.append(box((-65, -55, 235), (-63, 55, 330), PRINT, "x"))                          # back wall
    for s in (1, -1):
        yo, yi = s * 60.0, s * 55.0
        wins = [rect_cut(("x", "z"), c - 3.7, c + 3.7, 308.6, 312.0) for c in LID_HOOK_X]
        wins = [[(p[0], yo, p[2]) for p in w] for w in wins]
        fr.append(plate("y", (-65, min(yo, yi), 235), (50, max(yo, yi), 330), PRINT, cuts=wins, r=12,
                        holes=[((0, 0, SH_Z), 8.0)]))                                         # 688ZZ seat
        y45 = s * 45.0
        x0, x1 = -65, -45                                                                     # lid support tabs (back)
        fr.append(box((x0, min(y45, yi), 324), (x1, max(y45, yi), 330), PRINT))
        fr.append(poly("x", x0, x1, [W(y=yi, z=324), W(y=y45, z=324), W(y=yi, z=314)], PRINT))
        fr.append(box((x0, min(y45, yi), 235), (x1, max(y45, yi), 241), PRINT))               # base stiffeners
        fr.append(plate("z", (-62, min(y45, yi), 268), (-44, max(y45, yi), 272), PRINT,       # shelf ledges + pins
                        holes=[((-53, s * 50, 0), 1.5)], r=3))
        for x0, x1 in ((-62, -57), (-49, -44)):
            fr.append(poly("x", x0, x1, [W(y=yi, z=268), W(y=s * 45.0, z=268), W(y=yi, z=258)], PRINT))
        if s > 0:     # left front ledge (shortened: the left rod passes at x 37)
            fr.append(plate("z", (20, 45, 268), (32, 55, 272), PRINT, holes=[((26, 50, 0), 1.5)], r=3))
            for gx0, gx1 in ((20, 23.5), (28.5, 32)):
                fr.append(poly("x", gx0, gx1, [W(y=55, z=268), W(y=45, z=268), W(y=55, z=258)], PRINT))
        else:         # right front ledge: a 3 mm lip only (the arm servo sits below)
            fr.append(plate("z", (20, -55, 268), (38, -52, 272), PRINT, r=1))
        for c in RAIL_X.values():
            fr += frame_rail(s, c)
        # OpenCR corner channels on the back wall (board slides down from the top; the lid cap stops it)
        Y = OCR["y"]
        for zz in ("bot", "top"):
            za, zb = (225.0, 237.0) if zz == "bot" else (326.0, 334.0)
            C = [W(x=-65, y=s * 34.7), W(x=-65, y=s * 39.5), W(x=-71.4, y=s * 39.5), W(x=-71.4, y=s * 34.7),
                 W(x=-69.8, y=s * 34.7), W(x=-69.8, y=s * (Y + 0.2)), W(x=-67.9, y=s * (Y + 0.2)),
                 W(x=-67.9, y=s * 34.7)]
            if zz == "bot":                       # solid floor under the board edge, channel above
                fr.append(box((-71.4, min(s * 34.7, s * 39.5), 225), (-65, max(s * 34.7, s * 39.5), OCR["z0"]), PRINT))
                fr.append(poly("z", OCR["z0"], zb, C, PRINT))
            else:
                fr.append(poly("z", za, zb, C, PRINT))
                y0, y1 = sorted((s * (Y + 0.2), s * 39.5))
                fr.append(poly("y", y0, y1, [W(x=-65, z=326), W(x=-71.4, z=326), W(x=-65, z=319.6)], PRINT))
    fr.append(box((19, -51, 235), (48, -21, ARM_SERVO[4]), PRINT))                         # arm servo seat
    fr.append(PROF("y", 47.6, 52.6, [(235, 37, 0), (235, 50, 0), (237.3, 50, 0), (244.5, 57.2, 0), (266, 57.2, 0),
                                    (266, 37, 0)], PRINT, holes=[(CRANK[1], CRANK[0], 8.0)]))   # 688ZZ crank bearing
    P["torso_frame"] = fr

    # ---- shelf (slides in from the front over the ledges) + 4 press-fit pins
    pins = [(-53, -50), (-53, 50), (26, 50)]
    P["electronics_shelf"] = [PROFI("z", 272, 276, rrect(-62, 48, -54.5, 54.5, 10),
                                    [rrect(32, 45, 40.5, 45, 1)], PRINT, [(x, y, 1.5) for x, y in pins])]   # left rod slot
    P["shelf_pins"] = [CYL((x, y, 266), "z", 1.5, 10, PRINT) for x, y in pins] + \
                      [CYL((x, y, 276), "z", 3.0, 1.5, PRINT) for x, y in pins]

    P["dcdc_12V"] = [RB(-62, -32, -52, -42, 280, 305, ELEC, "y", 3)]
    P.pop("head_pan_servo_XC430", None)
    P.pop("joint_discs_torso", None)
    # ---- arm swing servo (the former pan servo) in the front-right pocket under the shelf, horn on its inner face
    P["arm_swing_servo_XC430"] = [SERVO(ARM_SERVO[0], ARM_SERVO[1], ARM_SERVO[2], ARM_SERVO[3], ARM_SERVO[4],
                                        ARM_SERVO[5], "y", XC)]

    # ---- battery: ROBOTIS LB-012 (TurtleBot3 Li-Po, 11.1 V 1800 mAh, 88 x 35 x 26 mm, 106 g)
    P.pop("battery_3S_2200", None)
    P["battery_LB012"] = [RB(-55, 33, -17.5, 17.5, 237, 263, build.BATT, "x", 4)]

    # ---- OpenCR 1.0
    P["opencr_1_0"] = [RB(OCR["x1"], OCR["x0"], -OCR["y"], OCR["y"], OCR["z0"], OCR["z1"], ELEC, "x", 3),
                       RB(-80.0, OCR["x1"], -34, 34, 237, 326, ELEC, "x", 2)]            # component envelope

    # ---- lid: plug between the walls + cap over the covers' top edges; four hooks into the wall windows
    plug_in = [rrect(-58, 43, -53, 53, 13)]                     # plug is a 7 mm rim, not a solid slab
    pegs = [(x, y, 1.6) for x, y in YOKE_PEGS]
    U = [(-45, -60), (50, -60), (50, -53), (-45, -53), (-45, -60)]
    plugU = [(-45.0, -60.0), (50.0, -60.0), (50.0, -53.0), (-58.0, -53.0), (-58.0, 53.0), (50.0, 53.0), (50.0, 60.0),
             (-45.0, 60.0), (-65.0, 40.0), (-65.0, -40.0)]
    lid = [poly("z", 330, 336, [W(x=x, y=y) for x, y in plugU], WHITE),
           poly("z", 336, 336.3, [W(x=x, y=y) for x, y in plugU], WHITE),
           PROFI("z", 336.3, 338.3, rrect(CU - CA, CU + CA, -CY, CY, CR_), [rrect(-32, -18, -7, 7, 3)], WHITE, pegs)]   # cable hole
    lid.append(plate("z", (-11, -52.5, 330), (11, 52.5, 336.3), WHITE,              # crossbar under the neck yoke
                     holes=[((x, y, 0), 1.6) for x, y, _ in pegs]))
    for s in (1, -1):
        for c in LID_HOOK_X:
            lid += hook("-z", f"{'+' if s > 0 else '-'}y", 330, 312.0, s * 54.85, c - 3.5, c + 3.5, WHITE)
    P["lid"] = lid

    # ---- covers (shorts navy z 205-225, shirt white above): no holes; dovetail blocks; front half also
    #      carries the arms' T-rails and the tongue latches
    for half, sgn in (("front", 1), ("back", -1)):
        sides = ("+u", "+v", "-v") if sgn > 0 else ("-u", "+v", "-v")
        holes = {}
        if half == "front":
            holes = {"+v": [((0, CY, SH_Z), 4.6)], "-v": [((0, -CY, SH_Z), 4.6)]}
        shell = box_shell("z", SHORTS_TOP + 0.3, 336, CU, 0, CA, CY, CR_, T, WHITE, sides=sides, holes=holes)
        out = []
        for s_ in shell:                                   # trim the side panels to this half
            k_, a_, rgb_ = s_
            if k_ == "rprism" and a_[0] == "y":
                ax, ya0, ya1, outer, inn, hs = a_
                us = [p[1] for p in outer]
                lo_x = CU + 0.25 if sgn > 0 else min(us)
                hi_x = max(us) if sgn > 0 else CU - 0.25
                outer2 = rrect(min(p[0] for p in outer), max(p[0] for p in outer), lo_x, hi_x, 0)
                s_ = ("rprism", (ax, ya0, ya1, outer2, inn, hs), rgb_)
            out.append(s_)
        for s in (1, -1):
            out += cover_channel(s, RAIL_X[half], WHITE)
        P[f"torso_cover_{half}"] = out

    # ---- shorts: one closed navy ring (z 205-237) with a top flange that rests on the pelvis plate edge;
    #      dropped in from above before the shirt covers (notches pass the cover rails), trapped by the covers
    xr = sorted(RAIL_X.values())
    hole = [(-72.0, -60.2)]
    for c in xr:
        hole += [(c - 5.3, -60.2), (c - 5.3, -72.2), (c + 5.3, -72.2), (c + 5.3, -60.2)]
    hole += [(50.3, -60.2), (50.3, 60.2)]
    for c in reversed(xr):
        hole += [(c + 5.3, 60.2), (c + 5.3, 72.2), (c - 5.3, 72.2), (c - 5.3, 60.2)]
    hole += [(-72.0, 60.2)]
    P["shorts"] = box_shell("z", 205, SHORTS_TOP, CU, 0, CA, CY, CR_, T, NAVY) + \
        [poly("z", 235, SHORTS_TOP, [W(x=CU - CA + 20, y=-CY + 1)], NAVY)]
    P["shorts"][-1] = PROFI("z", 235, SHORTS_TOP, rrect(CU - CA + 0.6, CU + CA - 0.6, -CY + 0.6, CY - 0.6, CR_ - 0.6),
                            [[(x, y, 0) for x, y in hole]], NAVY)

    # ---- arms: sleeve (keyhole + slot in the inner panel), shoulder dome; hand press-fits into the sleeve
    for s, nm in ((1, "L"), (-1, "R")):
        yc = s * ARM_Y
        inner = "-v" if s > 0 else "+v"
        yi = s * (ARM_Y - 16)
        arm = box_shell("z", 272, 325, 0, yc, 16, 16, 11, T, SKY, holes={inner: [((0, yi, SH_Z), 4.1)]})
        y0, y1 = sorted((s * (ARM_Y - 16 + 1.2), s * (ARM_Y - 6)))
        arm.append(PROFI("y", y0, y1, circle_loop(SH_Z, 0, 9.0), [circle_loop(SH_Z, 0, 4.0)], SKY))   # D-hole boss
        # shoulder: 8 mm steel shaft (D-cut) through the 688ZZ bearing in the frame wall, lever hub inside
        y0, y1 = sorted((s * 44.5, s * (ARM_Y - 6)))
        sh_ = [PROF("y", y0, y1, circle_loop(SH_Z, 0, 4.0), STEEL)]
        y0, y1 = sorted((s * 55.0, s * 60.0))
        P[f"shoulder_bearing_{nm}_688"] = [PROFI("y", y0, y1, circle_loop(SH_Z, 0, 8.0), [circle_loop(SH_Z, 0, 4.0)], STEEL)]
        P[f"arm_{nm}_shaft"] = sh_
        arm.append(("scap", ((0, yc + s * 2.1, 325), (0, 0, 1), 18.1, 16.9, 0.0), SKY))   # dome, flush on the inner side
        P[f"arm_{nm}_sleeve"] = arm
        yh = s * (ARM_Y + 2.7)                  # big mitten ball, nudged outward (0.5 mm clear of the cover)
        P[f"arm_{nm}_hand"] = [("scap", ((0, yh, 255), (0, 0, 1), 20.0, 18.8, -14.0), WHITE),
                               PROFI("z", 268.5, 277, circle_loop(0, yc, 14.9), [circle_loop(0, yc, 11.0)], WHITE)]
    return P, []


def neck():
    """Pan link: a stiff plate on the pan horn carrying the tilt servo, which stays exposed under the head."""
    # neck yoke (navy, one print): base on the lid cap, two uprights with 688ZZ seats, rod anchor arm on +y
    yoke = [box((-12, -15, 338.3), (12, 15, 341.3), NAVY)]
    yoke += [PROFI("y", y0, y1, rrect(338.3, ZT9 + 10, -10, 10, 0), [], NAVY, [(ZT9, 0, 8.0)])
             for y0, y1 in ((10, 15), (-15, -10))]
    yoke.append(PROF("y", 10, 15, rrect(338.3, NOD_A[1] + 5, NOD_A[0] - 4.5, -9.9, 0), NAVY))     # anchor arm
    yoke.append(PROF("y", 15, 19.7, rrect(338.3, NOD_A[1] + 4.5, NOD_A[0] - 4.5, NOD_A[0] + 4.5, 0), NAVY, holes=[(NOD_A[1], NOD_A[0], 1.5)]))
    yoke += [CYL((x, y, 338.3), "-z", 1.5, 8.0, NAVY) for x, y in YOKE_PEGS]              # pegs into the lid
    return {
        "neck_yoke": yoke,
        "neck_bearings_688": [PROFI("y", y0, y1, circle_loop(ZT9, 0, 8.0), [circle_loop(ZT9, 0, 4.0)], STEEL)
                              for y0, y1 in ((10, 15), (-15, -10))],
        "neck_shaft_8mm": [CYL((0, -27.5, ZT9), "y", 4.0, 55.0, STEEL)],
        "nod_anchor_pin": [CYL((NOD_A[0], 15.2, NOD_A[1]), "y", 1.5, 7.6, STEEL),
                           CYL((NOD_A[0], 22.8, NOD_A[1]), "y", 2.5, 0.8, STEEL)],
    }


# ---------------------------------------------------------------- legs
def _strip(parts):
    return parts[:-2]          # build6 appended a nut block (2 solids) to the frame


def _front_wall_split(parts, z0, z1, ylo, yhi, groove):
    """Replace the front wall (x 14.5..18.5, z0..z1) by an inner layer and an outer layer with a groove."""
    out = []
    for s in parts:
        k, a, rgb = s
        if k == "rprism" and a[0] == "x" and abs(a[1] - 14.5) < 1e-6 and abs(a[2] - 18.5) < 1e-6:
            zs = [p[1] for p in a[3]]
            if abs(min(zs) - z0) < 1e-6 and abs(max(zs) - z1) < 1e-6:
                out.append(plate("x", (14.5, ylo, z0), (16.5, yhi, z1), rgb, r=4))
                out.append(plate("x", (16.5, ylo, z0), (18.5, yhi, z1), rgb, r=4, cuts=[groove]))
                continue
        out.append(s)
    assert len(out) == len(parts) + 1, "front wall not found"
    return out


def shell_latch(ys, zb, rgb):
    """Saddle (sits on the frame pad) + flexible beam + barb on the inner face of a leg shell's front panel.
    zb = shell bottom. Returns (shell solids, frame pad, groove outline)."""
    sol = []
    for sy in (1, -1):
        y0, y1 = sorted((ys + sy * 5.2, ys + sy * 7))
        sol.append(box((19, y0, zb), (22, y1, zb + 4), rgb))                       # saddle legs
    sol.append(box((18.6, ys - 7, zb + 4), (22, ys + 7, zb + 7), rgb))            # saddle bar = beam root
    sol.append(box((18.6, ys - 3.8, zb + 7), (19.8, ys + 3.8, zb + 21), rgb))     # beam (14 long)
    zt = zb + 21
    sol.append(poly("y", ys - 3.8, ys + 3.8, [W(x=18.6, z=zt), W(x=17.6, z=zt - 1), W(x=17.6, z=zt - 2),
                                              W(x=18.6, z=zt - 3)], rgb))
    pad = box((18.5, ys - 5, zb), (21.5, ys + 5, zb + 4), PRINT)
    groove = [(17, ys - 4, zt - 3.2), (17, ys + 4, zt - 3.2), (17, ys + 4, zt + 0.2), (17, ys - 4, zt + 0.2)]
    return sol, pad, groove


def right_leg():
    L, F, J = _V6_LEG()
    ys = -40.0
    # thigh
    tf = _strip(L["thigh"]["thigh_frame"])
    th_sol, th_pad, th_groove = shell_latch(ys, 118.0, WHITE)
    tf = _front_wall_split(tf, 98, 140, ys - 17, ys + 17, th_groove)
    L["thigh"]["thigh_frame"] = tf + [th_pad]
    L["thigh"]["thigh_shell"] = box_shell("z", 118, 150, 0, ys - 0.5, 23.5, 27, 12, 1.5, WHITE,
                                          sides=("+u", "+v", "-v"), u_open_at=-20) + th_sol
    # shank
    sf = _strip(L["shank"]["shank_frame"])
    sk_sol, sk_pad, sk_groove = shell_latch(ys, 44.0, SKY)
    sf = _front_wall_split(sf, 30, 76, ys - 22, ys + 22, sk_groove)
    L["shank"]["shank_frame"] = sf + [sk_pad]
    L["shank"]["shank_shell"] = box_shell("z", 44, 104, 0, ys, 23.5, 30.5, 12, 1.5, SKY,
                                          sides=("+u", "+v", "-v"), u_open_at=-20) + sk_sol
    # foot: the bracket is now part of the sole print (no bolts)
    br = L["foot"].pop("foot_bracket")
    br = [(k, (a[0], a[1], a[2], a[3], a[4], []), NAVY) if k == "rprism" else (k, a, rgb) for k, a, rgb in br]
    sole = [(k, (a[0], a[1], a[2], a[3], a[4], []), rgb) if k == "rprism" else (k, a, rgb)
            for k, a, rgb in L["foot"]["foot_sole"]]
    L["foot"]["foot_sole"] = sole + br
    return L, [], J


# ---------------------------------------------------------------- assembly
def arm_drive():
    """Links of the arm swing mechanism (world coordinates). Crank: right pin forward, left pin backward, so the
    vertical rods move in opposite directions and the two shoulder levers (pointing forward) swing the arms in
    opposite phase."""
    cx, cz = CRANK
    D = lambda y0, y1, arm: PROFI("y", y0, y1, arm, [], PRINT, [(cz, cx, 4.0)])
    crank = [PROFI("y", -18.0, -12.5, circle_loop(cz, cx, 10.0), [], PRINT, [(cz, cx, 4.0)]),        # horn adapter
             PROFI("y", -10.2, -7.2, rrect(cz - 4, cz + 4, cx - 9, cx + R_CRANK + 3, 4), [], PRINT,
                   [(cz, cx, 4.0), (cz, cx + R_CRANK, 1.5)]),
             PROFI("y", 44.5, 47.5, rrect(cz - 4, cz + 4, cx - L_CRANK - 3, cx + 9, 4), [], PRINT,
                   [(cz, cx, 4.0), (cz, cx - L_CRANK, 1.5)])]
    crank_shaft = [PROF("y", -16.0, 52.6, circle_loop(cz, cx, 4.0), STEEL)]
    crank_bearing = [PROFI("y", 47.6, 52.6, circle_loop(cz, cx, 8.0), [circle_loop(cz, cx, 4.0)], STEEL)]
    def pin(x, z, ya, yb, yh=None):     # 3 mm steel dowel, pressed into one part, free in the rod
        return [PROF("y", min(ya, yb), max(ya, yb), circle_loop(z, x, 1.5), STEEL)]
    crank_pins = pin(cx + R_CRANK, cz, -10.2, -4.1) + pin(cx - L_CRANK, cz, 41.4, 47.5)
    rx, lx = cx + R_CRANK, cx - L_CRANK
    rod_r = [PROF("y", -7.1, -4.1, rrect(cz - 3, SH_Z + 3, rx - 3, rx + 3, 3), PRINT,
                  holes=[(cz, rx, 1.6), (SH_Z, R_LEV, 1.6)])]
    rod_l = [PROF("y", 41.4, 44.4, rrect(cz - 3, SH_Z + 3, lx - 3, lx + 3, 3), PRINT,
                  holes=[(cz, lx, 1.6), (SH_Z, L_LEV, 1.6)])]
    hub = lambda y0, y1: PROFI("y", y0, y1, circle_loop(SH_Z, 0, 9.0), [circle_loop(SH_Z, 0, 4.0)], PRINT)
    lev_r = [hub(-50, -45), box((8.5, -50, SH_Z - 3), (58, -47, SH_Z + 3), PRINT),
             poly("z", SH_Z - 3, SH_Z + 3, [W(x=52, y=-47), W(x=58, y=-47), W(x=58, y=-19.3), W(x=67, y=-10.3),
                                            W(x=52, y=-10.3)], PRINT),
             PROF("y", -10.3, -7.3, rrect(SH_Z - 3, SH_Z + 3, 52, 67, 0), PRINT, holes=[(SH_Z, R_LEV, 1.5)])]
    pin_r = pin(R_LEV, SH_Z, -10.3, -4.1)
    lev_l = [hub(44.5, 50.0), PROF("y", 44.5, 47.5, rrect(SH_Z - 3, SH_Z + 3, 8.5, L_LEV + 3, 0), PRINT,
                                     holes=[(SH_Z, L_LEV, 1.5)])]
    pin_l = pin(L_LEV, SH_Z, 41.4, 47.5)
    return crank, crank_pins, rod_r, rod_l, lev_r, lev_l, pin_r, pin_l, crank_shaft, crank_bearing


def build_model():
    saved = (build5.torso, build5.neck, build5.head, build5.right_leg)
    build5.torso, build5.neck, build5.head, build5.right_leg = torso, neck, head, right_leg
    try:
        links, joints, fasteners = build5.build_model()
    finally:
        build5.torso, build5.neck, build5.head, build5.right_leg = saved
    links["head"]["origin"] = (0, 0, ZT9)
    # no pan: the neck parts (nod servo, mount) belong to the torso; the head nods on the torso
    links["torso"]["parts"].update(links.pop("neck")["parts"])
    joints = [j for j in joints if j["name"] != "head_pan"]
    for j in joints:
        if j["name"] == "head_tilt":
            j.update(name="head_nod", parent="torso", point_mm=(0, 0, ZT9), lower_deg=-10, upper_deg=20)
    # nod drive: crank on the head servo's horn, rod down to the anchor pin on the yoke (closed loop)
    sx, sz = NOD_S
    px = sx + NOD_R
    links["nod_crank"] = {"origin": (sx, 0, sz), "parts": {
        "nod_crank": [PROFI("y", 17.2, 19.7, rrect(sz - 5, sz + 5, sx - 6, px + 4, 5), [], PRINT, [(sz, px, 1.5)])],
        "nod_crank_pin": [CYL((px, 17.2, sz), "y", 1.5, 5.6, STEEL)]}}
    links["nod_rod"] = {"origin": (NOD_A[0], 21.3, NOD_A[1]), "parts": {
        "nod_rod": [PROF("y", 19.8, 22.8, rrect(NOD_A[1] - 3.5, sz + 3.5, px - 3.5, px + 3.5, 3.5), PRINT,
                         holes=[(NOD_A[1], NOD_A[0], 1.6), (sz, px, 1.6)])]}}
    for j in joints:
        if j["name"] == "head_nod":
            j["actuator"] = "passive (driven by nod_crank)"
    joints.append({"name": "nod_crank", "parent": "head", "child": "nod_crank", "axis": "y", "point_mm": (sx, 0, sz),
                   "lower_deg": -70, "upper_deg": 40, "actuator": "XC430-W240 (inside the head)"})
    joints.append({"name": "nod_rod", "parent": "torso", "child": "nod_rod", "axis": "y", "point_mm": (NOD_A[0], 0, NOD_A[1]),
                   "lower_deg": -30, "upper_deg": 30, "actuator": "passive (closed loop with head_nod)"})
    # arms become links; crank shaft and rods are links of the closed arm-swing linkage
    crank, crank_pins, rod_r, rod_l, lev_r, lev_l, pin_r, pin_l, crank_shaft, crank_bearing = arm_drive()
    T_ = links["torso"]["parts"]
    for s, nm, lev, pn in ((-1, "R", lev_r, pin_r), (1, "L", lev_l, pin_l)):
        links[f"{nm}_arm"] = {"origin": (0, s * 50.0, SH_Z),
                              "parts": {f"arm_{nm}_sleeve": T_.pop(f"arm_{nm}_sleeve"),
                                        f"arm_{nm}_hand": T_.pop(f"arm_{nm}_hand"),
                                        f"arm_{nm}_shaft": T_.pop(f"arm_{nm}_shaft"),
                                        f"arm_{nm}_lever": lev, f"arm_{nm}_lever_pin": pn}}
        joints.append({"name": f"{nm}_arm_swing", "parent": "torso", "child": f"{nm}_arm", "axis": "y",
                       "point_mm": (0, s * 50.0, SH_Z), "lower_deg": -10, "upper_deg": 10,
                       "actuator": "passive (rod from arm_crank)"})
    links["arm_crank"] = {"origin": (CRANK[0], 0, CRANK[1]), "parts": {"arm_horn_adapter": crank[:1], "arm_crank_R": crank[1:2],
                                                                      "arm_crank_L": crank[2:], "arm_crank_shaft_8mm": crank_shaft,
                                                                      "arm_crank_pins": crank_pins}}
    T_["crank_bearing_688"] = crank_bearing
    links["R_arm_rod"] = {"origin": (CRANK[0] + R_CRANK, -5.6, CRANK[1]), "parts": {"arm_R_rod": rod_r}}
    links["L_arm_rod"] = {"origin": (CRANK[0] - L_CRANK, 42.9, CRANK[1]), "parts": {"arm_L_rod": rod_l}}
    joints.append({"name": "arm_crank", "parent": "torso", "child": "arm_crank", "axis": "y",
                   "point_mm": (CRANK[0], 0, CRANK[1]), "lower_deg": -42, "upper_deg": 42, "actuator": "XC430-W240"})
    for nm, x in (("R", CRANK[0] + R_CRANK), ("L", CRANK[0] - L_CRANK)):
        joints.append({"name": f"{nm}_arm_rod", "parent": "arm_crank", "child": f"{nm}_arm_rod", "axis": "y",
                       "point_mm": (x, 0, CRANK[1]), "lower_deg": -60, "upper_deg": 60,
                       "actuator": f"passive (closed loop with {nm}_arm_swing)"})
    return links, joints, fasteners


write = build4.write
sampled, aabb, volume, to_local = build4.sampled, build4.aabb, build4.volume, build4.to_local

FORCE = {"neck_yoke": "+z", "head_back": "+x", "head_front": "+x", "shorts": "-z", "shank_shell": "+z", "thigh_shell": "+z", "torso_cover": "+z", "antenna": "+z", "face_plate": "-x",
         "torso_frame": "+z", "foot_sole": "+z", "hip_yaw_bracket": "+z", "eye_": "-x", "mouth": "-x",
         "lid": "-z", "crest": "-z", "head_shell_unused": "+x"}

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out12"
    os.makedirs(out, exist_ok=True)
    w = write(links, f"{out}/biped_robot_v12.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))

FORCE.update({"hand": "+z", "shelf_pins": "+z"})
NOTE = {
    "torso_cover": "흰색. 브림 권장", "shorts": "남색. 플랜지(윗면)를 바닥에",
    "shank_shell": "하늘색", "neck_yoke": "남색. 688ZZ 2개 압입, 핀 4개를 덮개에 압입", "lever": "", "rod": "평평하게", "arm_crank_": "D컷 구멍에 축 압입", "horn_adapter": "팔 서보 혼에 체결", "nod_crank": "끄덕임 서보 혼에 체결",
    "head_back": "뒷면을 바닥에. 목 축 판·보강판·서보 패드·이음 걸쇠 포함", "head_front": "자른 면(뒤)을 바닥에, 브림 권장. 얼굴 테두리 경사, 얼굴판 걸림 바",
    "crest": "지붕을 바닥에(걸쇠가 위로)",
    "lid": "캡을 바닥에(걸쇠가 위로)",
    "face_plate": "앞면을 바닥에(걸쇠·카메라 핀이 위로)",
    "eye_": "테두리를 바닥에",
    "mouth": "테두리를 바닥에",
    "antenna": "세워서 출력, 브림 5 mm",
    "sleeve": "축 구멍 있는 안쪽 면을 바닥에",
    "hand": "잘린 면을 바닥에, 위쪽 링을 소매에 끼움",
    "ear_L": "덮개를 바닥에(걸쇠가 위로)", "ear_R": "덮개를 바닥에(걸쇠가 위로)",
    "torso_frame": "골반판을 바닥에. 골반·측벽·뒷벽·레일이 한 부품",
    "shelf_pins": "핀 4개, 세워서",
    "camera_clip": "평평하게",
    "foot_sole": "발판을 바닥에. 발 브래킷이 한 몸",
    "thigh_frame": "앞면 홈 + 받침(셸 걸쇠용)",
    "shank_frame": "앞면 홈 + 받침(셸 걸쇠용)",
}
