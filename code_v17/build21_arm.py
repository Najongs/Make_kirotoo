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
"""
import os, json, math
import numpy as np
import build21_head as BH
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
POCKET, SEAT, FIT = B20.POCKET_MIN, B20.SEAT_GAP, B20.FIT_PLUS      # 0.35, 0.3, 0.1
WALL = B20.PL                       # 3.85 clamp walls (leg value)
XI = HW + POCKET                    # 14.6  x wall inner face
XW = XI + WALL                      # 18.45 arm half width in x
PL0, PL1 = 21.2, 24.7               # forearm plates |y - YC| (leg value: horn face 20.9 + 0.3)
SLIM = 20.6                         # upper frame below SLIM_Z0 stays inside |y - YC| <= SLIM (0.6 to the plates)
SLIM_Z0 = 262.0                     # above this the frame may widen (outer cover); the plates reach z ~250 at 140 deg
COVER = (HD + 3.0 + SEAT, HD + 3.0 + SEAT + 2.4)      # outer cover over the shoulder idler: |y - YC| 20.3 .. 22.7
CAP_T = 3.0
HAND_D1 = 50.0                      # hand top bar starts this far below the elbow axis
HOLE_CASE = B19.CASE_HOLE_D / 2     # 1.35
HOLE_HORN = B19.HORN_HOLE_D / 2     # 1.2
SCREW_CLR = 1.35                    # M2.5 clearance in the plates (same as the case holes)
SCREW_PILOT = 1.1                   # M2.5 tapping into printed PLA [estimate]
CB_R, CB_D = 2.6, 2.0               # counterbores on the torso side of the idler plate (heads face the torso cover)
TW = 2.0                            # hand shell wall
HAND_R = 10.0                       # hand bottom roll
# part names avoid "horn" / "_idler" (printcheck skips "_idler", mass_props takes "horn" for a 4 g servo horn)
FORCE = {"upper_frame": None, "elbow_bracket_out": None, "elbow_bracket_in": None, "hand_frame": "+z",
         "horn_adapter": "+y"}      # None: outward (R -y, L +y), set per side in export
NOTE = {
    "upper_frame": "하늘색. 몸통 쪽 면을 바닥에(R 은 위가 −y). 서보 둘을 옆 케이스 나사 16개로 잡음",
    "elbow_bracket_out": "구조색. 평평하게(바깥 면이 위). 팔꿈치 혼 M2 4개 + 손에 M2.5 4개",
    "elbow_bracket_in": "구조색. 몸통 쪽 면(카운터보어)을 바닥에, 아이들러 받침이 위",
    "hand_frame": "흰색. 손바닥을 바닥에. 고무 패드 접착",
    "horn_adapter": "구조색. 평평하게. 어깨 혼에 M2 4개, D컷 축에 압입",
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
    yc_o = YC + COVER[1]                            # 124.7 outer cover face
    z0, z1 = ELZ - 10.0, SHZ + HE + POCKET + CAP_T  # 186 .. 320.6
    # x walls: (y, z) outline, slim below SLIM_Z0, out to the cover above
    P = [(Y(yi), z0, 6.0), (Y(yo), z0, 6.0), (Y(yo), SLIM_Z0, 0.0), (Y(yc_o), SLIM_Z0, 0.0), (Y(yc_o), z1, 6.0), (Y(yi), z1, 6.0)]
    # lightening window in the box part of each wall (no screws there): 30 x 30, 2.6 / 4.4 mm rims
    zw0, zw1 = ELZ + HB + 4.75, SHZ - HB - 4.75               # 236 .. 266
    win = [[(Y(YC - 15.0), zw0, 4.0), (Y(YC + 15.0), zw0, 4.0), (Y(YC + 15.0), zw1, 4.0), (Y(YC - 15.0), zw1, 4.0)]]
    walls = [PROFI("x", XI, XW, P, win, SKY), PROFI("x", -XW, -XI, P, win, SKY)]
    # box between the servos: inner / outer y walls, cable notch (torso cable) in the inner wall's top edge
    zb0, zb1 = ELZ + HB + SEAT, SHZ - HB - POCKET            # 231.55 (one-sided seat) .. 270.4 (shoulder pocket)
    box_o = [(zb0, -XI - 0.4, 0), (zb0, XI + 0.4, 0), (zb1, XI + 0.4, 0), (zb1, -XI - 0.4, 0)]
    box_i = [(zb0, -XI - 0.4, 0), (zb0, XI + 0.4, 0), (zb1, XI + 0.4, 0), (zb1, 5.0, 0), (zb1 - 7.0, 5.0, 1.0),
             (zb1 - 7.0, -5.0, 1.0), (zb1, -5.0, 0), (zb1, -XI - 0.4, 0)]
    box = [PROFI("y", *_yr(s, yi, yi + 3.0), box_i, [], SKY), PROFI("y", *_yr(s, yo - 3.0, yo), box_o, [], SKY)]
    # outer cover over the shoulder servo (idler side, one-sided seat 0.3 to the idler) and the top cap
    cov = [(SLIM_Z0, -XW, 0), (SLIM_Z0, XW, 0), (z1, XW, 6.0), (z1, -XW, 6.0)]
    cover = PROFI("y", *_yr(s, YC + COVER[0], yc_o), cov, [], SKY)
    cap_xy = [(-XW, Y(yi), 2.0), (XW, Y(yi), 2.0), (XW, Y(yc_o), 6.0), (-XW, Y(yc_o), 6.0)]
    cap = PROFI("z", z1 - CAP_T, z1, cap_xy, [], SKY)
    return walls + box + [cover, cap]


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
    corners = [(z_step + 3.0, -(XW - 3.0)), (z_step + 3.0, XW - 3.0), (z_top - 9.0, -(XW - 3.0)), (z_top - 9.0, XW - 3.0)]
    scr = [(u, v, SCREW_CLR) for u, v in corners]
    horn = PROFI("y", *_yr(s, YC + PL0, YC + PL1), outline, [], PRINT, horn_holes(ELZ, 0.0) + [(ELZ, 0.0, 2.5)] + scr)
    # idler plate: counterbores (torso side, |y| YC-PL1 .. +CB_D) + plate + boss up to the idler face
    yp0, yp1 = YC - PL1, YC - PL0                   # 77.3 .. 80.8
    idl_face = YC - HD - 3.0                        # 82.0
    cbh = [(u, v, CB_R) for u, v, _ in horn_holes(ELZ, 0.0)] + [(u, v, CB_R) for u, v in corners] + [(ELZ, 0.0, 4.0)]
    thr = horn_holes(ELZ, 0.0) + [(ELZ, 0.0, 4.0)] + scr
    idler = [PROFI("y", *_yr(s, yp0, yp0 + CB_D), outline, [], PRINT, cbh),
             PROFI("y", *_yr(s, yp0 + CB_D, yp1), outline, [], PRINT, thr),
             PROFI("y", *_yr(s, yp1 - 0.3, idl_face), circle_loop(ELZ, 0.0, 10.25), [], PRINT, horn_holes(ELZ, 0.0) + [(ELZ, 0.0, 4.0)])]
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
    fr_out = [(z_step, -XW, 0.0), (z_step, XW, 0.0), (z_top, XW, 8.0), (z_top, -XW, 8.0)]
    win = [(z_step + 6.0, -XW + 6.0, 2.0), (z_step + 6.0, XW - 6.0, 2.0), (z_top - 6.0, XW - 6.0, 2.0), (z_top - 6.0, -XW + 6.0, 2.0)]
    for y0 in (YC - PL0 + 0.1, YC + PL0 - 0.1 - 6.0):
        hand.append(PROFI("y", *_yr(s, y0, y0 + 6.0), fr_out, [win], WHITE, [(u, v, SCREW_PILOT) for u, v in corners]))
    # front / back covers between the ring frames (closed mitten look); top 58 mm below the elbow axis, 8 mm under
    # the ring frame tops, so they do not take over the fold limit
    yc0, yc1 = YC - PL0 + 0.1 + 6.0 - 0.3, YC + PL0 - 0.1 - 6.0 + 0.3
    cov = [(Y(yc0), z_step - 0.3, 0.0), (Y(yc1), z_step - 0.3, 0.0), (Y(yc1), z_top - 8.0, 0.0), (Y(yc0), z_top - 8.0, 0.0)]
    hand += [PROFI("x", XW - TW, XW, cov, [], WHITE), PROFI("x", -XW, -XW + TW, cov, [], WHITE)]
    pad = PROFI("z", z_bot - 1.0, z_bot, [(u, v, 0) for u, v in B16.ccw(outl(do_b + 1.5))], [], RUBBER)
    # ---- elbow horn / idler (parts of the driven link, as in the legs)
    hn = CYL((0.0, Y(YC + HD), ELZ), "y" if s > 0 else "-y", 10.25, 3.9, XC)
    il = CYL((0.0, Y(YC - HD), ELZ), "-y" if s > 0 else "y", 10.25, 3.0, XC)
    return {f"arm_{nm}_elbow_horn": [hn], f"arm_{nm}_elbow_idler": [il],
            f"arm_{nm}_elbow_bracket_out": [horn], f"arm_{nm}_elbow_bracket_in": idler,
            f"arm_{nm}_hand_frame": hand, f"arm_{nm}_hand_pad": [pad]}


def build_model():
    links, joints, fasteners = BH.build_model()
    T_ = links["torso"]["parts"]
    for s in (-1, 1):
        nm = "R" if s < 0 else "L"
        A = links[f"{nm}_arm"]["parts"]
        dropped = [k for k in list(A) if k != f"arm_{nm}_servo_XL430"]
        for k in dropped:
            A.pop(k)
        links.pop(f"{nm}_hand_flap")
        joints[:] = [j for j in joints if j["name"] != f"{nm}_hand_flap"]
        # horn adapter: v20 disc (bore r 4.1 = shaft + press fit), no cam lobe, + 4 horn screws
        ad = T_[f"arm_{nm}_horn_adapter"][0]
        assert ad[1][0] == "y" and abs(ad[1][2] - ad[1][1] - 3.5) < 1e-6, ad[1][:3]
        T_[f"arm_{nm}_horn_adapter"] = [PROFI("y", ad[1][1], ad[1][2], ad[1][3], [], PRINT, list(ad[1][5]) + horn_holes(SHZ, 0.0))]
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
    LOG["case_holes"] = []
    for nm in "RL":
        LOG["case_holes"] += case_holes_x(links, nm, frames)
        fr = frames[f"arm_{nm}_elbow_servo_XL430"]
        assert abs(fr["check"]) < 1e-6 and abs(fr["O"][2] - ELZ) < 1e-6, fr
    return links, joints, fasteners


write = B19.write
servo_step_parts = B19.servo_step_parts

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    os.makedirs(OUT, exist_ok=True)
    print(json.dumps(LOG, indent=1, default=str)[:3000])
