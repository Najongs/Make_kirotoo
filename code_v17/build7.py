"""v7: printed parts join each other without screws. Screws are used only where a part meets a servo
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
from build5 import SKY, NAVY, WHITE, ORANGE, DARK, PAN_Z, ZT
from build6 import (PROFI, plate, box_shell, ring, T, ZH, HW, HD, HH, HR, CH, FZ, FX, TOP_IN, CZ0, CZ1)
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


# ---------------------------------------------------------------- head (tilt link)
EYE_L = 34.0                 # robot's left eye (y > 0) carries the camera
EYE_Z = FZ + 13              # eye centre height
CAM_TOP = EYE_Z + 9.5        # Pi Camera v2: lens centre 9.5 mm below the board's top edge
CAM_HOLES = [(EYE_L + dy, CAM_TOP - dz) for dy in (-10.5, 10.5) for dz in (2.0, 14.5)]


def eye_pill(y, grow=0.0):
    return rrect(y - 9 - grow, y + 9 + grow, EYE_Z - 18 - grow, EYE_Z + 18 + grow, 9 + grow)


def head():
    P = {}
    sh = []
    for i in range(CH):                           # 45-degree stepped chamfers (as v6)
        o = CH - i
        sh.append(ring("x", -HD + i, -HD + i + 1, 0, ZH, HW - o, HH - o, HR - o, T, SKY))
        sh.append(ring("x", HD - CH + i, HD - CH + i + 1, 0, ZH, HW - i - 1, HH - i - 1, HR - i - 1, T, SKY))
    sh.append(PROF("x", -HD, -HD + T, rrect(-(HW - CH), HW - CH, ZH - (HH - CH), ZH + (HH - CH), HR - CH), SKY))
    ez = ZH - 5
    crest_slots = [hook_slot(f"{'+' if s > 0 else '-'}y", s * 12.0, 26, 34, "x") for s in (1, -1)]
    crest_slots = [[(p[0], p[1], CZ0) for p in sl] for sl in crest_slots]
    ear_slots = {}
    for s, side in ((1, "+u"), (-1, "-u")):
        ear_slots[side] = [[(p[0], s * HW, p[2]) for p in hook_slot(f"{'+' if sx > 0 else '-'}x", sx * 12.0,
                                                                     ez - 4, ez + 4, "z")] for sx in (1, -1)]
    sh += box_shell("x", -HD + CH, HD - CH, 0, ZH, HW, HH, HR, T, SKY,
                    cuts={"+v": crest_slots, "+u": ear_slots["+u"], "-u": ear_slots["-u"],
                          "-v": [[(-38, -20, 0), (30, -20, 0), (50, 0, 0), (30, 20, 0), (-38, 20, 0)]]})
    # catch bars for the face-plate hooks (+ 45-degree gussets toward the back = downward in print)
    for s in (1, -1):
        y0, y1 = sorted((s * 80.15, s * (HW - T)))
        sh.append(box((46, y0, FZ - 5), (FX, y1, FZ + 5), SKY))
        sh.append(poly("z", FZ - 5, FZ + 5, [W(x=46, y=s * 83), W(x=46, y=s * (HW - T)),
                                             W(x=46 - (HW - T - 83), y=s * (HW - T))], SKY))
    # head frame plates (were a separate print): from the back wall to the tilt horn / idler, one print
    for s in (1, -1):
        y0, y1 = sorted((s * 19, s * 22))
        sh.append(PROF("y", y0, y1, [(ZT - 15, -HD + T, 0), (ZT - 15, 15, 0), (TOP_IN, 45, 0), (TOP_IN, -56, 0),
                                     (TOP_IN - 11, -HD + T, 0)], SKY))
    sh.append(DSC((0, 18, ZT), "y", 8))                                    # child disc of head_tilt
    P["head_shell"] = sh

    # face plate: pill / arc openings for press-fit eye and mouth inserts, two hooks, camera posts + pins
    mouth = annulus_sector(0, FZ - 7, 13, 16.5, 205, 335)
    fp = [PROFI("x", FX, FX + 2, rrect(-88, 88, FZ - 60, FZ + 60, 45), [eye_pill(EYE_L), eye_pill(-EYE_L), mouth],
                WHITE)]
    for s in (1, -1):
        fp += hook("-x", f"{'+' if s > 0 else '-'}y", FX, 46, s * 80.0, FZ - 4, FZ + 4, WHITE)
    for y, z in CAM_HOLES:
        fp.append(CYL((FX, y, z), "-x", 2.2, 3.0, WHITE))                   # standoff: board front at x 57
        fp.append(CYL((FX - 3.0, y, z), "-x", 1.0, 3.6, WHITE))             # pin through board + clip
    P["face_plate"] = fp

    lens = rrect(EYE_L - 4.5, EYE_L + 4.5, EYE_Z - 4.5, EYE_Z + 4.5, 0.5)  # 9 x 9 lens window
    P["eye_L"] = [PROFI("x", FX, FX + 2, eye_pill(EYE_L), [lens], NAVY),
                  PROFI("x", FX + 2, FX + 3.2, eye_pill(EYE_L, 1.5), [lens], NAVY)]
    P["eye_R"] = [PROF("x", FX, FX + 2, eye_pill(-EYE_L), NAVY),
                  PROF("x", FX + 2, FX + 3.2, eye_pill(-EYE_L, 1.5), NAVY)]
    P["mouth"] = [PROF("x", FX, FX + 2, mouth, NAVY),
                  PROF("x", FX + 2, FX + 3.2, annulus_sector(0, FZ - 7, 12.25, 17.25, 202, 338), NAVY)]

    # Raspberry Pi Camera Module v2 (25 x 23.86 x 1 PCB, 8.5 mm lens block), lens in the left eye window
    cz0 = CAM_TOP - 23.86
    P["camera_pi_v2"] = [RB(FX - 4, FX - 3, EYE_L - 12.5, EYE_L + 12.5, cz0, CAM_TOP, CAM, "x", 1.0,
                            holes=[((0, y, z), 1.1) for y, z in CAM_HOLES]),
                         RB(FX - 3, FX + 2, EYE_L - 4.25, EYE_L + 4.25, EYE_Z - 4.25, EYE_Z + 4.25, CAM, "x", 0.5)]
    clip = [PROFI("x", FX - 6.6, FX - 5.0, rrect(EYE_L - 12.5, EYE_L + 12.5, CAM_TOP - 16.7, CAM_TOP, 1.0),
                  [rrect(EYE_L - 7.5, EYE_L + 7.5, CAM_TOP - 12.3, CAM_TOP - 4.2, 0.5)], DARK,
                  [(y, z, 1.0) for y, z in CAM_HOLES])]
    clip += [PROF("x", FX - 5.0, FX - 4.0, circle_loop(y, z, 2.2), DARK, [(y, z, 1.0)]) for y, z in CAM_HOLES]
    P["camera_clip"] = clip

    # ears: cup + lid (centre hole for the disc plug) + antenna socket + two hooks into the head side wall
    for s, nm in ((1, "L"), (-1, "R")):
        ya, yb = sorted((s * HW, s * (HW + 16.8)))
        la, lb = sorted((s * (HW + 16.8), s * (HW + 18)))
        ear = [PROF("y", ya, yb, annulus_sector(ez, 0, 28.8, 30, 8, 352), NAVY),
               PROF("y", la, lb, circle_loop(ez, 0, 30), NAVY, holes=[(ez, 0, 8.0)]),
               plate("z", (-7, min(s * 102, s * 116.8), ez + 10), (7, max(s * 102, s * 116.8), ez + 26),
                     NAVY, holes=[((0, s * 108, 0), 4.0)])]
        for sx in (1, -1):
            ear += hook(f"{'-' if s > 0 else '+'}y", f"{'+' if sx > 0 else '-'}x", s * (HW + 16.8), s * (HW - T),
                        sx * 12.0, ez - 4, ez + 4, NAVY)
        P[f"ear_{nm}"] = ear
        P[f"ear_disc_{nm}"] = [CYL((0, s * (HW + 18), ez), "y" if s > 0 else "-y", 19, 1.5, SKY),
                               CYL((0, s * (HW + 16.8), ez), "y" if s > 0 else "-y", 8.0, 1.2, SKY)]
        P[f"antenna_{nm}"] = [CYL((0, s * 108, ez + 12), "z", 4, 95, ORANGE)]

    # crest: decorative, two hooks from the roof through the head top wall
    cu, A, B = 30.0, 22.0, 34.0
    cr = box_shell("z", CZ0, CZ1 - T, cu, 0.0, A, B, 10, T, SKY)
    cr.append(PROF("z", CZ1 - T, CZ1, rrect(cu - A, cu + A, -B, B, 10), SKY))
    for s in (1, -1):
        cr += hook("-z", f"{'+' if s > 0 else '-'}y", CZ1 - T, TOP_IN, s * 12.0, 26, 34, SKY)
    P["crest"] = cr
    return P


# ---------------------------------------------------------------- torso
CU, CA, CY, CR_ = (build4.COV_XF + build4.COV_XB) / 2, (build4.COV_XF - build4.COV_XB) / 2, build4.COV_Y, build4.COV_R
RAIL_X = {"front": 12.0, "back": -22.0}       # cover rails on the frame walls
LID_HOOK_X = (-32.0, 2.0)
ARM_Y = 92.8                                   # arm centre (1.8 mm air gap to the cover)
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
        fr.append(plate("y", (-65, min(yo, yi), 235), (50, max(yo, yi), 330), PRINT, cuts=wins, r=12))
        y45 = s * 45.0
        for x0, x1 in ((-65, -45), (30, 50)):                                             # lid support tabs
            fr.append(box((x0, min(y45, yi), 324), (x1, max(y45, yi), 330), PRINT))
            fr.append(poly("x", x0, x1, [W(y=yi, z=324), W(y=y45, z=324), W(y=yi, z=314)], PRINT))
            fr.append(box((x0, min(y45, yi), 235), (x1, max(y45, yi), 241), PRINT))           # base stiffeners
        for x0, x1, xp in ((-62, -44, -53), (20, 38, 29)):                                   # shelf ledges + pin holes
            fr.append(plate("z", (x0, min(y45, yi), 268), (x1, max(y45, yi), 272), PRINT,
                            holes=[((xp, s * 50, 0), 1.5)], r=3))
        for x0, x1 in ((-62, -57), (-49, -44), (20, 25), (33, 38)):
            fr.append(poly("x", x0, x1, [W(y=yi, z=268), W(y=s * 45.0, z=268), W(y=yi, z=258)], PRINT))
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
    P["torso_frame"] = fr

    # ---- shelf (slides in from the front over the ledges) + 4 press-fit pins
    pins = [(x, y) for x in (-53, 29) for y in (-50, 50)]
    P["electronics_shelf"] = [RB(-62, 48, -54.5, 54.5, 272, 276, PRINT, "z", 10, [((x, y, 0), 1.5) for x, y in pins])]
    P["shelf_pins"] = [CYL((x, y, 266), "z", 1.5, 10, PRINT) for x, y in pins] + \
                      [CYL((x, y, 276), "z", 3.0, 1.5, PRINT) for x, y in pins]

    # ---- battery: ROBOTIS LB-012 (TurtleBot3 Li-Po, 11.1 V 1800 mAh, 88 x 35 x 26 mm, 106 g)
    P.pop("battery_3S_2200", None)
    P["battery_LB012"] = [RB(-55, 33, -17.5, 17.5, 237, 263, build.BATT, "x", 4)]

    # ---- OpenCR 1.0
    P["opencr_1_0"] = [RB(OCR["x1"], OCR["x0"], -OCR["y"], OCR["y"], OCR["z0"], OCR["z1"], ELEC, "x", 3),
                       RB(-80.0, OCR["x1"], -34, 34, 237, 326, ELEC, "x", 2)]            # component envelope

    # ---- lid: plug between the walls + cap over the covers' top edges; four hooks into the wall windows
    hole = [(-12, 0, 37.0)]
    plug_in = [rrect(-58, 43, -53, 53, 13)]                     # plug is a 7 mm rim, not a solid slab
    lid = [PROFI("z", 330, 336, rrect(-65, 50, -60, 60, 20), plug_in, WHITE),
           PROFI("z", 336, 336.3, rrect(-65, 50, -60, 60, 20), plug_in, WHITE),
           PROF("z", 336.3, 338.3, rrect(CU - CA, CU + CA, -CY, CY, CR_), WHITE, holes=hole)]
    for s in (1, -1):
        for c in LID_HOOK_X:
            lid += hook("-z", f"{'+' if s > 0 else '-'}y", 330, 312.0, s * 54.85, c - 3.5, c + 3.5, WHITE)
    P["lid"] = lid

    # ---- covers (shorts navy z 205-225, shirt white above): no holes; dovetail blocks; front half also
    #      carries the arms' T-rails and the tongue latches
    for half, sgn in (("front", 1), ("back", -1)):
        sides = ("+u", "+v", "-v") if sgn > 0 else ("-u", "+v", "-v")
        cuts = {}
        if half == "front":
            for s, side in ((1, "+v"), (-1, "-v")):
                U = [(-3.5, 266), (-3, 266), (-3, 288), (3, 288), (3, 266), (3.5, 266), (3.5, 288.3), (-3.5, 288.3)]
                cuts[side] = [[(x, s * CY, z) for x, z in U]]
        shell = box_shell("z", 205, 225.0, CU, 0, CA, CY, CR_, T, NAVY, sides=sides) + \
            box_shell("z", 225.0, 336, CU, 0, CA, CY, CR_, T, WHITE, sides=sides, cuts=cuts)
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
            if half == "front":
                y = lambda d: s * (CY + d)
                y0, y1 = sorted((y(0), y(3.2)))
                out.append(box((-1.8, y0, ARM_RAIL[0]), (1.8, y1, ARM_RAIL[1]), WHITE))           # T-rail neck
                y0, y1 = sorted((y(3.2), y(5.2)))
                out.append(box((-3.5, y0, ARM_RAIL[0]), (3.5, y1, ARM_RAIL[1]), WHITE))           # T-rail head
                out.append(poly("x", -1.8, 1.8, [W(y=y(0), z=ARM_RAIL[0]), W(y=y(5.2), z=ARM_RAIL[0]),
                                                 W(y=y(0), z=ARM_RAIL[0] - 5.2)], WHITE))
                # tongue boss (in the 1.8 mm gap) + barb into the arm's keyhole
                y0, y1 = sorted((y(0), y(1.8)))
                out.append(box((-2.5, y0, 284.8), (2.5, y1, 287.8), WHITE))
                out.append(poly("x", -2.5, 2.5, [W(y=y(1.8), z=284.8), W(y=y(2.8), z=285.8), W(y=y(2.8), z=286.8),
                                                 W(y=y(1.8), z=287.8)], WHITE))
        P[f"torso_cover_{half}"] = out

    # ---- arms: sleeve (keyhole + slot in the inner panel), shoulder dome; hand press-fits into the sleeve
    for s, nm in ((1, "L"), (-1, "R")):
        yc = s * ARM_Y
        inner = "-v" if s > 0 else "+v"
        yi = s * (ARM_Y - 16)
        K = [(-3.8, KEY[0]), (3.8, KEY[0]), (3.8, KEY[1]), (2.0, KEY[1]), (2.0, ARM_RAIL[1]), (-2.0, ARM_RAIL[1]),
             (-2.0, KEY[1]), (-3.8, KEY[1])]
        arm = box_shell("z", 262, 325, 0, yc, 16, 16, 12, T, SKY, cuts={inner: [[(x, yi, z) for x, z in K]]})
        arm.append(("scap", ((0, yc, 325), (0, 0, 1), 17.7, 16.5, 0.0), SKY))
        P[f"arm_{nm}_sleeve"] = arm
        P[f"arm_{nm}_hand"] = [("scap", ((0, yc, 250), (0, 0, 1), 17.5, 16.3, -12.4), WHITE),
                               PROFI("z", 260.5, 267, circle_loop(0, yc, 14.9), [circle_loop(0, yc, 11.0)], WHITE)]
    return P, []


def neck():
    return _V5_NECK()


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
    L["shank"]["shank_shell"] = box_shell("z", 44, 98, 0, ys, 23.5, 30.5, 12, 1.5, SKY,
                                          sides=("+u", "+v", "-v"), u_open_at=-20) + sk_sol + \
        box_shell("z", 98, 104, 0, ys, 23.5, 30.5, 12, 1.5, NAVY, sides=("+u", "+v", "-v"), u_open_at=-20)
    # foot: the bracket is now part of the sole print (no bolts)
    br = L["foot"].pop("foot_bracket")
    br = [(k, (a[0], a[1], a[2], a[3], a[4], []), NAVY) if k == "rprism" else (k, a, rgb) for k, a, rgb in br]
    sole = [(k, (a[0], a[1], a[2], a[3], a[4], []), rgb) if k == "rprism" else (k, a, rgb)
            for k, a, rgb in L["foot"]["foot_sole"]]
    L["foot"]["foot_sole"] = sole + br
    return L, [], J


# ---------------------------------------------------------------- assembly
def build_model():
    saved = (build5.torso, build5.neck, build5.head, build5.right_leg)
    build5.torso, build5.neck, build5.head, build5.right_leg = torso, neck, head, right_leg
    try:
        links, joints, fasteners = build5.build_model()
    finally:
        build5.torso, build5.neck, build5.head, build5.right_leg = saved
    return links, joints, fasteners


write = build4.write
sampled, aabb, volume, to_local = build4.sampled, build4.aabb, build4.volume, build4.to_local

FORCE = {"shank_shell": "+z", "thigh_shell": "+z", "torso_cover": "+z", "antenna": "+z", "face_plate": "-x",
         "torso_frame": "+z", "foot_sole": "+z", "hip_yaw_bracket": "+z", "eye_": "-x", "mouth": "-x",
         "lid": "-z", "crest": "-z", "head_shell": "+x"}

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out7"
    os.makedirs(out, exist_ok=True)
    w = write(links, f"{out}/biped_robot_v7.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))

FORCE.update({"arm_L_sleeve": "+y", "arm_R_sleeve": "-y", "hand": "+z", "shelf_pins": "+z"})
NOTE = {
    "torso_cover": "z 20 mm에서 필라멘트 교체: 남색(반바지) → 흰색(셔츠). 브림 권장",
    "shank_shell": "높이 54 mm에서 필라멘트 교체: 하늘색 → 남색(띠)",
    "head_shell": "뒷면을 바닥에, 브림 권장. 머리 프레임 판이 한 몸",
    "crest": "지붕을 바닥에(걸쇠가 위로)",
    "lid": "캡을 바닥에(걸쇠가 위로)",
    "face_plate": "앞면을 바닥에(걸쇠·카메라 핀이 위로)",
    "eye_": "테두리를 바닥에",
    "mouth": "테두리를 바닥에",
    "antenna": "세워서 출력, 브림 5 mm",
    "sleeve": "안쪽 평면(키홀 있는 면)을 바닥에",
    "hand": "잘린 면을 바닥에, 위쪽 링을 소매에 끼움",
    "ear_L": "덮개를 바닥에(걸쇠가 위로)", "ear_R": "덮개를 바닥에(걸쇠가 위로)",
    "torso_frame": "골반판을 바닥에. 골반·측벽·뒷벽·레일이 한 부품",
    "shelf_pins": "핀 4개, 세워서",
    "camera_clip": "평평하게",
    "foot_sole": "발판을 바닥에. 발 브래킷이 한 몸",
    "thigh_frame": "앞면 홈 + 받침(셸 걸쇠용)",
    "shank_frame": "앞면 홈 + 받침(셸 걸쇠용)",
}
