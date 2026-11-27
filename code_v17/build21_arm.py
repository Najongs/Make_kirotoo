"""v21 arm = v21 head (build21_head: v20 geometry, nod -20 .. +28) + one elbow XL430 per arm (proposal, not adopted).

Owner 2026-11-27: "design it and build it, then I'll consider it" -> new files only, the v20 kit is unchanged.
Basis: sim/getup/GETUP_STUDY.md (update section; 110 + 110 mm elbow arm, the shortest cover of the 213 mm prone
path), out_head21/HEAD_CRANK_v21.md (shoulder +-110 deg).  Report: out21_arm/ARM_ELBOW_v21.md.

Arm chain: torso -> <S>_arm_swing (shoulder, unchanged) -> <S>_arm (upper arm) -> <S>_elbow (new) -> <S>_forearm
  * upper arm 110 mm (shoulder axis z 306 -> elbow axis z 196), forearm 110 mm (elbow axis -> hand outer face z 86)
  * both servos ride in the upper arm, centre plane |y| 102: the shoulder servo as in v16-v20 (horn inward on the
    torso-fixed adapter, body turns with the arm), the elbow servo horn end down, horn outward; the two cable ends
    face each other across a 39 mm box between the servos (daisy-chain cable stays inside the upper arm)
  * upper frame (one print): two x walls that take the side case screws of both servos, a box between the servos,
    an outer cover over the shoulder servo and a top cap.  Below z 262 it stays inside |y - 102| <= 20.6 so the
    forearm plates (|y - 102| 21.2 .. 24.7) pass beside it like scissors
  * forearm: two flat plates (horn outside, idler inside, the leg's 21.2 .. 24.7 layout) screwed to a white hand.
    The hand's top bar (the only part across the middle band) starts 50 mm below the elbow axis: that is what
    lets the forearm fold past the upper arm (strip estimate +-137 deg; measured in arm21.py)
  * the v17 cam-driven finger flap cannot cross an elbow: dropped (simple mitten hand with a rubber palm pad);
    the horn adapter loses its cam lobe and gets the 4 horn screw holes (0 / 90 deg) it never had
  * fits by hand from the build20 constants (no global v20 pass is re-run, so the leg kit stays as it is):
    pockets 0.35 per side, one-sided seats 0.3, case holes 2.7, horn holes 2.4, press fit +0.1 (kept from v20)
Angles: geom4.rot convention (+deg about +y turns +x toward -z): shoulder / elbow negative = forward (sim/getup).

Durability pass (2026-11-28, the leg / body rules of CLAUDE.md: coarse print +-0.4 mm, 0.3 mm layers, no snaps, no
small press fits, gaps >= 0.4, walls >= 2, teardrops; legs xl_horn_fit / fdm_joints / assembly_durability /
print_robust, body second pass)
  * base model = build21_body (the torso / head kit being printed), so the regression gate compares against the
    current out20/print_parts/parts.json
  * XL430 horn fit (build20.xl_horn_fit rules, done here by hand: that pass would pad the leg plates twice): elbow horn /
    idler models -> disc + hub r 4 / cap r 5.3; both forearm plates get a pad r 12.5 down to the horn / idler face,
    centre 9 (horn) / 11.3 (idler), M2 x 5 counterbores 4.4 with the floor on the plate face (2.2 under the head,
    2.8 in the tap); the old idler boss + 5.2 counterbores go. Shoulder horn adapter: pad r 11.2 down to the horn face
    (it stood on the hub top with 1.9 mm of air under the screw ring), centre 9, counterbores 4.4 from the torso face
    (floor 2.0 above the horn face, 3.0 in the tap), body r 12.5 out to |y| 75.5 (shaft glued over 5.6 mm instead of a
    3.5 mm +0.1 press fit: bore 9 = 0.5 per side + epoxy)
  * servo pockets / seats / the shoulder idler running gap 0.35 / 0.3 / 0.3 -> 0.4 (x walls at 14.65 like the leg boxes)
  * case screws: the 8 side holes per servo as before + the horn-end holes 7 / 10: shoulder through the top cap
    (counterbore, M2.5 x 6), elbow through a new 4 mm bottom plate that closes the x walls under the elbow servo
    (y - YC from -12 inside to +16.6 outside, 0.4 to the horn disc; M2.5 x 8). The cable-end holes 5 / 6 / 8 / 9 face the box
    between the servos: no driver access once the frame is closed -> not used
  * forearm plate <-> hand ring frames 0.1 -> 0.4 per side; plate / ring screws 3.0 -> 4.7 in from the edges (counterbore
    5.2 x 1.5 on the inner plate, 2.0 under the head), ring windows smaller (>= 2 mm round every pilot)
  * 45 deg gussets: box y walls <-> x walls (legs 3), shoulder cover <-> x walls (legs 2), ring frames <-> lid (legs 3)
  * teardrops (build20._teardrops) on every hole that is horizontal in the print orientation: upper frame (-y R / +y L),
    hand (+z); plates and adapter print flat (holes vertical)
"""
import os, json, math
import numpy as np
import build21_head as BH
import build21_body as BB
from build21_head import *          # noqa  (build20 names)
import build20 as B20
import build19 as B19
import build16 as B16
from build import STEEL, RUBBER, XC, PRINT
from build4 import CYL, SERVO
from build6 import PROFI
from shapes import circle_loop
import x430_holes as X

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out21_arm")
LOG = {}

SKY, WHITE = B16.SKY, B16.WHITE
SHZ = B16.SHZ                       # shoulder axis z 306
UPPER_MM = 110.0                    # shoulder axis -> elbow axis
FORE_MM = 110.0                     # elbow axis -> hand outer face
ELZ = SHZ - UPPER_MM                # elbow axis z 196
YC = 102.0                          # |y| of both servo centre planes (v16 shoulder servo |y| 85 .. 119)
HW, HE, HB, HD = 14.25, 11.25, 35.25, 17.0     # X430 case: half width, horn end, back end, half depth (horn axis)
GAP = B20.FIT_GAP                   # 0.4 per side (coarse print, 2026-11-28)
POCKET, SEAT = GAP, GAP             # were 0.35 / 0.3 (B20.POCKET_MIN, SEAT_GAP)
WALL = B20.PL                       # 3.85 clamp walls (leg value)
XI = HW + POCKET                    # 14.65 x wall inner face (leg thigh box walls 14.65)
XW = XI + WALL                      # 18.5 arm half width in x
PL0, PL1 = 21.2, 24.7               # forearm plates |y - YC| (leg value: horn face 20.9 + 0.3)
SLIM = 20.6                         # upper frame below SLIM_Z0 stays inside |y - YC| <= SLIM (0.6 to the plates)
SLIM_Z0 = 262.0                     # above this the frame may widen (outer cover); the plates reach z ~250 at 140 deg
COVER = (HD + 3.0 + SEAT, HD + 3.0 + SEAT + 2.4)      # outer cover over the shoulder idler: |y - YC| 20.4 .. 22.8
CAP_T = 3.0
CAP_CB = 1.0                        # counterbore depth in the cap for the shoulder end screws (2.0 under the head)
BOT_T = 4.0                         # plate under the elbow servo horn end (M2.5 x 8: 4.0 + 0.4 + 3.6 in the case)
BOT_HW = HD - GAP                   # 16.6: its outer |y - YC| extent, 0.4 to the horn disc (|z_s| >= 17)
BOT_IN = 12.0                       # its inner extent (holes 7 / 10 at |y - YC| 6: 4.65 rim); keeps the inner bottom
                                    # corner away from the thigh at hip roll 33.6 with the arm folded
M25_CB_R = 2.6                      # M2.5 pan head counterbore (head 4.5 - 5.0)
CORNER_IN = 4.7                     # plate <-> ring frame screws: this far in from the plate side / bottom edges
RING_WIN_IN = 7.8                   # ring frame window: this far in from the frame sides (pilot rim >= 2)
SHAFT_R = 4.0                       # torso shoulder shaft (steel 8)
AD_Y_IN = 75.5                      # adapter torso face |y| (torso cover outer face 74.5)
AD_R_BODY, AD_R_PAD = 12.5, 11.2    # adapter body / pad radius (pad turns inside the frame: cap 11.65 above the axis)
FIL_BOX, FIL_COVER = B20.FIL, 2.0   # 45 deg gusset legs
FA_PAD_R = 12.5                     # forearm horn / idler pads (leg PAD_R 10.25): layer section at the pad 3.7 -> ~2 MPa;
                                    # 2.15 to the x walls (14.65), 3.3 round the M2 holes at r 8
HAND_D1 = 50.0                      # hand top bar starts this far below the elbow axis
HOLE_CASE = B19.CASE_HOLE_D / 2     # 1.35
HOLE_HORN = B19.HORN_HOLE_D / 2     # 1.2
SCREW_CLR = 1.35                    # M2.5 clearance in the plates (same as the case holes)
SCREW_PILOT = B20.PILOT_D / 2       # 1.1: M2.5 tapping into printed PLA (print 2.2, drill 2.2-2.3)
TW = 2.0                            # hand shell wall
HAND_R = 10.0                       # hand bottom roll
# part names avoid "horn" / "_idler" (printcheck skips "_idler", mass_props takes "horn" for a 4 g servo horn)
FORCE = {"upper_frame": None, "elbow_bracket_out": None, "elbow_bracket_in": None, "hand_frame": "+z",
         "horn_adapter": None}      # None: outward (R -y, L +y), set per side in export
NOTE = {
    "upper_frame": "하늘색. 몸통 쪽 면을 바닥에(R 은 위가 −y). 서보 둘을 케이스 나사 20개로 잡음(옆 16 · 윗뚜껑 2 · 아래판 2)",
    "elbow_bracket_out": "구조색. 바깥 면을 바닥에(혼 받침이 위). 팔꿈치 혼 M2 × 5 4개 + 손에 M2.5 × 8 4개",
    "elbow_bracket_in": "구조색. 몸통 쪽 면(카운터보어)을 바닥에, 아이들러 받침이 위. 아이들러 M2 × 5 4개 + 손에 M2.5 × 6 4개",
    "hand_frame": "흰색. 손바닥을 바닥에. 고무 패드 접착",
    "horn_adapter": "구조색. 몸통 쪽 면(카운터보어)을 바닥에. 어깨 혼에 M2 × 5 4개, 강철 축에 에폭시(틈 0.5)",
}


def _yr(s, a, b):
    """signed y range of |y| a .. b on side s (-1 right, +1 left)"""
    return min(s * a, s * b), max(s * a, s * b)


def _apex(c, r, p):
    """vertex on the symmetry axis (v = c[1]) whose fillet of radius r is the circle (c, r), for side point p:
    the tangent from p to the far side of the circle, extended to the axis. (u, v) plane, apex at larger u."""
    cu, cv = c
    d = math.hypot(p[0] - cu, p[1] - cv)
    a = math.atan2(p[1] - cv, p[0] - cu)
    phi = math.acos(r / d)
    t = a - phi if p[1] > cv else a + phi          # tangent point on the upper (larger u) side
    tu, tv = cu + r * math.cos(t), cv + r * math.sin(t)
    k = (cv - p[1]) / (tv - p[1])
    return p[0] + k * (tu - p[0])


def upper_parts(s, frames=None):
    """upper arm frame (one print) in world coordinates"""
    Y = lambda v: s * v
    yi, yo = YC - SLIM, YC + SLIM                   # 81.4, 122.6
    yc_o = YC + COVER[1]                            # 124.8 outer cover face
    z_bt = ELZ - HE - GAP                           # 184.35 bottom plate top, 0.4 under the elbow servo horn end
    z0, z1 = z_bt - BOT_T, SHZ + HE + POCKET + CAP_T  # 180.35 .. 320.65
    # x walls: (y, z) outline, slim below SLIM_Z0, out to the cover above
    # (inner bottom corner cut from the bottom plate edge up to ELZ - 3: the thigh at hip roll 33.6 with the arm folded)
    P = [(Y(YC - BOT_IN), z0, 2.0), (Y(yo), z0, 4.0), (Y(yo), SLIM_Z0, 0.0), (Y(yc_o), SLIM_Z0, 0.0), (Y(yc_o), z1, 6.0),
         (Y(yi), z1, 6.0), (Y(yi), ELZ - 3.0, 2.0)]
    # lightening window in the box part of each wall (no screws there): 30 x 30, 2.6 / 4.4 mm rims
    zw0, zw1 = ELZ + HB + 4.75, SHZ - HB - 4.75               # 236 .. 266
    win = [[(Y(YC - 15.0), zw0, 4.0), (Y(YC + 15.0), zw0, 4.0), (Y(YC + 15.0), zw1, 4.0), (Y(YC - 15.0), zw1, 4.0)]]
    walls = [PROFI("x", XI, XW, P, win, SKY), PROFI("x", -XW, -XI, P, win, SKY)]
    # box between the servos: inner / outer y walls, cable notch (torso cable) in the inner wall's top edge
    zb0, zb1 = ELZ + HB + SEAT, SHZ - HB - POCKET            # 231.65 (elbow servo end + 0.4) .. 270.35 (shoulder pocket)
    box_o = [(zb0, -XI - 0.4, 0), (zb0, XI + 0.4, 0), (zb1, XI + 0.4, 0), (zb1, -XI - 0.4, 0)]
    box_i = [(zb0, -XI - 0.4, 0), (zb0, XI + 0.4, 0), (zb1, XI + 0.4, 0), (zb1, 5.0, 0), (zb1 - 7.0, 5.0, 1.0),
             (zb1 - 7.0, -5.0, 1.0), (zb1, -5.0, 0), (zb1, -XI - 0.4, 0)]
    box = [PROFI("y", *_yr(s, yi, yi + 3.0), box_i, [], SKY), PROFI("y", *_yr(s, yo - 3.0, yo), box_o, [], SKY)]
    # 45 deg gussets in the box corners (y walls <-> x walls), full box height
    for sx in (-1, 1):
        for yf, d in ((yi + 3.0, 1), (yo - 3.0, -1)):
            box.append(PROFI("z", zb0, zb1, B20._tri((sx * XI, Y(yf)), -sx * FIL_BOX, s * d * FIL_BOX), [], SKY))
    # outer cover over the shoulder servo (idler side, 0.4 to the idler cap) and the top cap; gussets cover <-> x walls
    # inside the shoulder pocket (tips 1.4 off the case side face, outside the idler disc r 10.25)
    cov = [(SLIM_Z0, -XW, 0), (SLIM_Z0, XW, 0), (z1, XW, 6.0), (z1, -XW, 6.0)]
    cover = [PROFI("y", *_yr(s, YC + COVER[0], yc_o), cov, [], SKY)]
    for sx in (-1, 1):
        cover.append(PROFI("z", zb1, z1 - CAP_T, B20._tri((sx * XI, Y(YC + COVER[0])), -sx * FIL_COVER, -s * FIL_COVER), [], SKY))
    cap_xy = [(-XW, Y(yi), 2.0), (XW, Y(yi), 2.0), (XW, Y(yc_o), 6.0), (-XW, Y(yc_o), 6.0)]
    # cap in two layers: the end screws get a counterbore in the top CAP_CB (holes added by end_holes)
    cap = [PROFI("z", z1 - CAP_T, z1 - CAP_CB, cap_xy, [], SKY), PROFI("z", z1 - CAP_CB, z1, cap_xy, [], SKY)]
    # bottom plate under the elbow servo horn end: closes the x walls (they hung on the servo case only), takes holes 7 / 10
    bot = PROFI("z", z0, z_bt, [(-XW, Y(YC - BOT_IN), 2.0), (XW, Y(YC - BOT_IN), 2.0), (XW, Y(YC + BOT_HW), 2.0),
                                (-XW, Y(YC + BOT_HW), 2.0)], [], SKY)
    return walls + box + cover + cap + [bot]


def end_holes(links, nm, frames):
    """horn-end case holes 7 / 10 (X430 face +y_s): shoulder servo up through the top cap (2.7, counterbore 5.2 in the
    top CAP_CB), elbow servo down through the bottom plate (2.7). Cable-end holes 5 / 6 / 8 / 9 face the closed box."""
    F = links[f"{nm}_arm"]["parts"][f"arm_{nm}_upper_frame"]
    zs = [x[1][2] for x in F if x[0] == "rprism" and x[1][0] == "z"]
    z_top, z_bot = max(zs), min(x[1][1] for x in F if x[0] == "rprism" and x[1][0] == "z")
    out = []
    for sv, up in ((f"arm_{nm}_servo_XL430", True), (f"arm_{nm}_elbow_servo_XL430", False)):
        fr = frames[sv]
        for hn in (7, 10):
            p, n = X.HOLES[hn - 1]
            pw, nw = X.to_world(fr, p, n)
            assert abs(nw[2] - (1.0 if up else -1.0)) < 1e-6, (sv, hn, nw)
            for i, x in enumerate(F):
                a = x[1]
                if x[0] != "rprism" or a[0] != "z":
                    continue
                if up and abs(a[2] - z_top) < 1e-6:
                    r = M25_CB_R
                elif (up and abs(a[2] - (z_top - CAP_CB)) < 1e-6) or (not up and abs(a[1] - z_bot) < 1e-6):
                    r = HOLE_CASE
                else:
                    continue
                F[i] = ("rprism", (a[0], a[1], a[2], a[3], a[4], list(a[5]) + [(pw[0], pw[1], r)]), x[2])
            gap = (z_top - CAP_T - pw[2]) if up else (pw[2] - (z_bot + BOT_T))
            grip = (CAP_T - CAP_CB) if up else BOT_T
            screw = 6.0 if up else 8.0
            out.append(dict(servo=sv, hole=hn, x=round(float(pw[0]), 2), y=round(float(pw[1]), 2), z=round(float(pw[2]), 2),
                            screw=f"M2.5 x {screw:.0f}", in_case_mm=round(screw - grip - gap, 2)))
    return out


def case_holes_x(links, nm, frames):
    """M2.5 case holes (2.7) of the two arm servos into the x walls of the upper frame (X430 side holes 1-4, 11-14)"""
    F = links[f"{nm}_arm"]["parts"][f"arm_{nm}_upper_frame"]
    out = []
    for sv in (f"arm_{nm}_servo_XL430", f"arm_{nm}_elbow_servo_XL430"):
        fr = frames[sv]
        for k, (p, n) in enumerate(X.HOLES):
            pw, nw = X.to_world(fr, p, n)
            if abs(abs(nw[0]) - 1.0) > 1e-6:
                continue                                    # only the side faces (world x)
            i = next(i for i, x in enumerate(F) if x[1][0] == "x" and (x[1][1] > 0) == (nw[0] > 0))
            a = F[i][1]
            F[i] = ("rprism", (a[0], a[1], a[2], a[3], a[4], list(a[5]) + [(pw[1], pw[2], HOLE_CASE)]), F[i][2])
            out.append(dict(servo=sv, hole=k + 1, y=round(float(pw[1]), 2), z=round(float(pw[2]), 2), side="+x" if nw[0] > 0 else "-x"))
    return out


def horn_holes(u, v, r=HOLE_HORN):
    """X430 horn / idler: M2 at 0 / 90 deg on PCD 16 (sim/walk/allxl XL-430_new.stp)"""
    return [(u + 8.0, v, r), (u - 8.0, v, r), (u, v + 8.0, r), (u, v - 8.0, r)]


def forearm_parts(s):
    """forearm plates, hand, horn / idler, pad (world coordinates, zero pose = hanging)"""
    nm = "R" if s < 0 else "L"
    Y = lambda v: s * v
    z_bot = ELZ - FORE_MM                           # 86 hand outer face
    z_top = ELZ - HAND_D1                           # 146 hand top bar
    z_step = z_bot + 24.0                           # 110 plates stand on the hand here
    # ---- plates: circle r 14 round the elbow axis, tangents to x +-XW at z_side, straight down to z_step
    r_e, z_side = 14.0, ELZ - 30.0
    za = _apex((ELZ, 0.0), r_e, (z_side, XW))
    outline = [(z_step, -XW, 2.0), (z_step, XW, 2.0), (z_side, XW, 0.0), (za, 0.0, r_e), (z_side, -XW, 0.0)]
    ci = CORNER_IN
    corners = [(z_step + ci, -(XW - ci)), (z_step + ci, XW - ci), (z_top - 9.0, -(XW - ci)), (z_top - 9.0, XW - ci)]
    scr = [(u, v, SCREW_CLR) for u, v in corners]
    # XL430 horn fit (build20.xl_horn_fit rules): pad r FA_PAD_R from the horn / idler face (|z_s| 19) 0.05 into the plate,
    # M2 x 5 counterbores CB_D from the plate face on (2.2 under the head, 2.8 in the tap); centre 9 over the horn hub
    # (r 4 to 20.9) / 11.3 over the idler cap (r 5.3 to 20.0), 2.5 in the plates (bores at r 8 keep >= 3.3 to it)
    ring_s, ring_cb = horn_holes(ELZ, 0.0), horn_holes(ELZ, 0.0, B20.CB_D / 2)
    disc = circle_loop(ELZ, 0.0, FA_PAD_R)
    horn = [PROFI("y", *_yr(s, YC + B20.HORN_FACE, YC + PL0 + 0.05), disc, [], PRINT, [(ELZ, 0.0, B20.HUB_R + B20.HUB_CLR)] + ring_s),
            PROFI("y", *_yr(s, YC + PL0, YC + PL1), outline, [], PRINT, ring_cb + [(ELZ, 0.0, 2.5)] + scr)]
    # idler plate: pad + plate; the plate <-> hand screws get M2.5 head counterbores on the torso face (2.0 under the head)
    cb_m25 = 1.5
    idler = [PROFI("y", *_yr(s, YC - PL0 - 0.05, YC - B20.HORN_FACE), disc, [], PRINT, [(ELZ, 0.0, B20.CAP_R + B20.CAP_CLR)] + ring_s),
             PROFI("y", *_yr(s, YC - PL1 + cb_m25 - 0.05, YC - PL0), outline, [], PRINT, ring_cb + [(ELZ, 0.0, 2.5)] + scr),
             PROFI("y", *_yr(s, YC - PL1, YC - PL1 + cb_m25), outline, [], PRINT,
                   ring_cb + [(ELZ, 0.0, 2.5)] + [(u, v, M25_CB_R) for u, v in corners])]
    # ---- hand: mitten shell (full forearm width) + lid + rib + two ring frames against the plates
    yA, yB = YC - PL1, YC + PL1                     # 77.3 .. 126.7
    cv, A, B = YC, XW, PL1
    outl = lambda d: [(x, Y(yy)) for x, yy in B16.offset_rr2(0.0, cv, A, B, (12.0, 12.0, 3.0, 3.0), d)]
    zr = z_bot + HAND_R                             # roll start (96)
    hand = [("loft", ("z", ((zr, outl(0.0), outl(TW)), (z_step, outl(0.0), outl(TW))), True), WHITE)]
    lv = []
    for sv in B16.roll_levels(HAND_R, TW, HAND_R):
        do, di = B16.roll(HAND_R, TW, sv)
        lv.append((zr - sv, outl(do), outl(di)))
    lv.sort(key=lambda t: t[0])
    hand.append(("loft", ("z", tuple(lv), True), WHITE))
    do_b = B16.roll(HAND_R, TW, HAND_R)[0]
    hand.append(PROFI("z", z_bot, z_bot + TW, [(u, v, 0) for u, v in B16.ccw(outl(do_b))], [], WHITE))
    hand.append(PROFI("z", z_step - 2.4, z_step, [(u, v, 0) for u, v in B16.ccw(outl(TW - 0.3))], [], WHITE))       # lid
    # rib (x = -1 .. 1) under the lid, ends following the inner surface of the roll (lid bridges <= 16 mm)
    rib = []
    for z_, (do, di) in [(z_bot + TW, B16.roll(HAND_R, TW, zr - z_bot - TW))] + \
            [(zr - sv, B16.roll(HAND_R, TW, sv)) for sv in sorted(B16.roll_levels(HAND_R, TW, zr - z_bot - TW), reverse=True)] + \
            [(z_step - 2.4, (0.0, TW))]:
        rib.append((z_, yA + di - 0.3, yB - di + 0.3))
    rib = sorted(set(rib))
    pts = [(Y(a), z_, 0) for z_, a, b in rib] + [(Y(b), z_, 0) for z_, a, b in rib[::-1]]
    hand.append(PROFI("x", -1.0, 1.0, pts, [], WHITE))
    # ring frames: 6 mm y extrusions on the lid, one inside each plate (0.1 gap), take the plate screws; their top
    # corners (50 mm below the elbow axis) are what limits the fold
    # (0.4 per side to the plates; window >= 2 mm round every pilot)
    fr_out = [(z_step, -XW, 0.0), (z_step, XW, 0.0), (z_top, XW, 8.0), (z_top, -XW, 8.0)]
    wx = XW - RING_WIN_IN
    win = [(z_step + RING_WIN_IN, -wx, 2.0), (z_step + RING_WIN_IN, wx, 2.0), (z_top - 7.0, wx, 2.0), (z_top - 7.0, -wx, 2.0)]
    for y0 in (YC - PL0 + GAP, YC + PL0 - GAP - 6.0):
        hand.append(PROFI("y", *_yr(s, y0, y0 + 6.0), fr_out, [win], WHITE, [(u, v, SCREW_PILOT) for u, v in corners]))
    # 45 deg gussets ring frame inner face <-> lid top (the frames stand on the lid: +z print, root across the layers)
    for yf, d in ((YC - PL0 + GAP + 6.0, 1), (YC + PL0 - GAP - 6.0, -1)):
        hand.append(PROFI("x", -XW + TW - 0.05, XW - TW + 0.05, B20._tri((Y(yf), z_step - 0.05), s * d * FIL_BOX, FIL_BOX + 0.05), [], WHITE))
    # front / back covers between the ring frames (closed mitten look); top 58 mm below the elbow axis, 8 mm under
    # the ring frame tops, so they do not take over the fold limit
    yc0, yc1 = YC - PL0 + GAP + 6.0 - 0.3, YC + PL0 - GAP - 6.0 + 0.3
    cov = [(Y(yc0), z_step - 0.3, 0.0), (Y(yc1), z_step - 0.3, 0.0), (Y(yc1), z_top - 8.0, 0.0), (Y(yc0), z_top - 8.0, 0.0)]
    hand += [PROFI("x", XW - TW, XW, cov, [], WHITE), PROFI("x", -XW, -XW + TW, cov, [], WHITE)]
    pad = PROFI("z", z_bot - 1.0, z_bot, [(u, v, 0) for u, v in B16.ccw(outl(do_b + 1.5))], [], RUBBER)
    # ---- elbow horn / idler (parts of the driven link, as in the legs)
    # ---- elbow horn / idler (parts of the driven link, as in the legs), XL shape (build20.xl_horn_fit models)
    ah, ai = ("y" if s > 0 else "-y"), ("-y" if s > 0 else "y")
    hn = [CYL((0.0, Y(YC + HD), ELZ), ah, 10.25, B20.HORN_FACE - HD, XC), CYL((0.0, Y(YC + HD), ELZ), ah, B20.HUB_R, 20.9 - HD, XC)]
    il = [CYL((0.0, Y(YC - HD), ELZ), ai, 10.25, B20.HORN_FACE - HD, XC), CYL((0.0, Y(YC - HD), ELZ), ai, B20.CAP_R, 20.0 - HD, XC)]
    B20._teardrops(hand, "+z")
    return {f"arm_{nm}_elbow_horn": hn, f"arm_{nm}_elbow_idler": il,
            f"arm_{nm}_elbow_bracket_out": horn, f"arm_{nm}_elbow_bracket_in": idler,
            f"arm_{nm}_hand_frame": hand, f"arm_{nm}_hand_pad": [pad]}


def build_model():
    links, joints, fasteners = BB.build_model()
    T_ = links["torso"]["parts"]
    for s in (-1, 1):
        nm = "R" if s < 0 else "L"
        A = links[f"{nm}_arm"]["parts"]
        dropped = [k for k in list(A) if k != f"arm_{nm}_servo_XL430"]
        for k in dropped:
            A.pop(k)
        links.pop(f"{nm}_hand_flap")
        joints[:] = [j for j in joints if j["name"] != f"{nm}_hand_flap"]
        # horn adapter (torso-fixed, on the steel shaft): v20 disc r 11 at |y| 77.6 .. 81.1 stood on the horn hub top
        # (1.9 mm of air under the screw ring) with a +0.1 press fit on the shaft. Now: pad r AD_R_PAD down to the horn
        # face (|y| YC - 19) with the centre 9 over the hub, body r AD_R_BODY out to AD_Y_IN (0.4 to the frame at
        # |y| 81.4, 1.0 to the torso cover); bore 9 all through (shaft 8: 0.5 per side, epoxy); M2 x 5 at 0 / 90 deg
        # through 4.4 counterbores from the torso face, floor UNDER_HEAD above the horn face (3.0 in the tap)
        ad = T_[f"arm_{nm}_horn_adapter"][0]
        assert ad[1][0] == "y" and abs(ad[1][2] - ad[1][1] - 3.5) < 1e-6, ad[1][:3]
        y_face = YC - B20.HORN_FACE                                 # 83.0
        y_cb = y_face - B20.UNDER_HEAD                              # 81.0
        ctr = [(SHZ, 0.0, B20.HUB_R + B20.HUB_CLR)]
        T_[f"arm_{nm}_horn_adapter"] = [
            PROFI("y", *_yr(s, y_cb - 0.05, y_face), circle_loop(SHZ, 0.0, AD_R_PAD), [], PRINT, ctr + horn_holes(SHZ, 0.0)),
            PROFI("y", *_yr(s, AD_Y_IN, y_cb), circle_loop(SHZ, 0.0, AD_R_BODY), [], PRINT, ctr + horn_holes(SHZ, 0.0, B20.CB_D / 2))]
        LOG.setdefault("horn_adapter", {})[nm] = dict(pad_y=(round(y_cb - 0.05, 2), y_face), body_y=(AD_Y_IN, y_cb), pad_r=AD_R_PAD,
                                                     body_r=AD_R_BODY, bore_d=2 * ctr[0][2], shaft_gap=round(ctr[0][2] - SHAFT_R, 2),
                                                     under_head=B20.UNDER_HEAD, thread_m2x5=B20.HORN_SCREW - B20.UNDER_HEAD,
                                                     glue_len=round(81.1 - AD_Y_IN, 2))
        # elbow servo: horn end down, horn outward
        A[f"arm_{nm}_elbow_servo_XL430"] = [SERVO(-HW, HW, *_yr(s, YC - HD, YC + HD), ELZ - HE, ELZ + HB, "y", XC)]
        A[f"arm_{nm}_upper_frame"] = upper_parts(s)
        links[f"{nm}_forearm"] = {"origin": (0.0, s * YC, ELZ), "parts": forearm_parts(s)}
        k = next(i for i, j in enumerate(joints) if j["name"] == f"{nm}_arm_swing")
        joints[k].update(lower_deg=-110.0, upper_deg=110.0)
        joints.insert(k + 1, {"name": f"{nm}_elbow", "parent": f"{nm}_arm", "child": f"{nm}_forearm", "axis": "y",
                              "point_mm": (0.0, s * YC, ELZ), "lower_deg": -130.0, "upper_deg": 130.0,
                              "actuator": "XL430-W250 (in the upper arm, horn outward)"})
        LOG.setdefault("dropped", {})[nm] = dropped + [f"{nm}_hand_flap (link + joint)", "horn adapter cam lobe"]
    frames = X.servo_frames(links)
    LOG["case_holes"], LOG["end_holes"], LOG["teardrops"] = [], [], {}
    for nm in "RL":
        LOG["case_holes"] += case_holes_x(links, nm, frames)
        LOG["end_holes"] += end_holes(links, nm, frames)
        fr = frames[f"arm_{nm}_elbow_servo_XL430"]
        assert abs(fr["check"]) < 1e-6 and abs(fr["O"][2] - ELZ) < 1e-6, fr
        # last: horizontal holes of the upper frame -> teardrops (prints with the torso face down: up -y R / +y L)
        LOG["teardrops"][f"arm_{nm}_upper_frame"] = B20._teardrops(links[f"{nm}_arm"]["parts"][f"arm_{nm}_upper_frame"], "-y" if nm == "R" else "+y")
    return links, joints, fasteners


write = B19.write
servo_step_parts = B19.servo_step_parts

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    os.makedirs(OUT, exist_ok=True)
    print(json.dumps(LOG, indent=1, default=str)[:3000])
