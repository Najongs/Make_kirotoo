"""v17 = v16 with the real X430 servo geometry (ROBOTIS XC430 STEP) taken into account.

  * every servo gets its horn (r10.25 x 3.9 mm) and idler (r10.25 x 3.0 mm) as parts of the link they drive
  * leg U-brackets moved off the horn / idler faces (they were drawn against the bare case): thigh and shank plates
    at 21.2..24.7 mm from the servo centre plane (horn face 20.9, idler face 20.0), idler-side bosses, horn screw
    pattern (4 x M2 on PCD 16 + centre) and an 8 mm cable bore at the idler
  * standard orientation: horn outboard, idler inboard on every pitch servo
  * cable windows in the clamp walls where the case side slots are (idler half, 8.7..19.5 mm from the horn axis)
  * ankle roll: foot horn plate moved 3.1 mm forward, back plate of the ankle bracket notched around the idler
  * hip yaw hub shortened to the horn face, shelf relieved over the yaw servo idlers
  * nod servo moved 3.7 mm so its horn face meets the crank; arm servo horn / adapter at the real horn thickness
  * longer arms with a sliding finger block driven from the shoulder (see arm section)
"""
import os, json, math
import numpy as np
import build16 as B16
from build16 import *          # noqa
from build4 import XC, XM, PROF, CYL, RB, SERVO
from build import STEEL, DISC
from build6 import PROFI
from shapes import rrect, circle_loop
import build4
import loft as LF

HORN_T, IDLER_T, HR_ = 3.9, 3.0, 10.25
PL0, PL1 = 21.2, 24.7          # |dy| of the moved leg plates
FORCE = dict(B16.FORCE)
FORCE.pop("horn_adapter", None)
NOTE = dict(B16.NOTE)


def horn_holes(u, v, centre=2.5):
    """4 x M2 on PCD 16 (horn mounted with its taps on the diagonals) + centre clearance"""
    d = 8.0 / math.sqrt(2)
    return [(u + d, v + d, 1.1), (u - d, v + d, 1.1), (u + d, v - d, 1.1), (u - d, v - d, 1.1), (u, v, centre)]


def notch_x_plate(s, y_end, y_in, z0, z1, roof="top"):
    """ax-x rect plate (u=y, v=z): cut a notch from the edge at y_end inward to y_in, between z0..z1. The edge that
    is a ceiling in print ('top' = z1 when printed +z, 'bottom' = z0 when printed -z) rises at 45 deg toward the open
    edge so the notch needs no support."""
    ax, a0, a1, outer, inn, holes = s[1]
    ys = [p[0] for p in outer]; zs = [p[1] for p in outer]
    Y0, Y1, Z0, Z1 = min(ys), max(ys), min(zs), max(zs)
    if abs(y_end - Y1) < abs(y_end - Y0):      # notch at the +y edge
        w = Y1 - y_in
        lo, hi = (z0, z1 + w) if roof == "top" else (z0 - w, z1)
        top_in, bot_in = (z1, z0) if roof == "top" else (z1, z0)
        pts = [(Y0, Z0), (Y1, Z0), (Y1, lo), (y_in, z0), (y_in, z1), (Y1, hi), (Y1, Z1), (Y0, Z1)]
    else:
        w = y_in - Y0
        lo, hi = (z0, z1 + w) if roof == "top" else (z0 - w, z1)
        pts = [(Y0, Z0), (Y1, Z0), (Y1, Z1), (Y0, Z1), (Y0, hi), (y_in, z1), (y_in, z0), (Y0, lo)]
    return PROFI(ax, a0, a1, [(u, v, 0) for u, v in pts], inn, s[2], holes)


def set_x_plate_y(s, y0, y1, z0=None, z1=None):
    ax, a0, a1, outer, inn, holes = s[1]
    zs = [p[1] for p in outer]
    Z0 = min(zs) if z0 is None else z0
    Z1 = max(zs) if z1 is None else z1
    r = max(p[2] for p in outer)
    return PROFI(ax, a0, a1, rrect(min(y0, y1), max(y0, y1), Z0, Z1, min(r, 2.0)), inn, s[2], holes)


def fix_leg(links, side):
    s = -1 if side == "R" else 1
    ys = 40.0 * s
    out_y = lambda d: ys + s * d           # outboard
    in_y = lambda d: ys - s * d            # inboard
    def is_disc(x):
        return x[0] == "cyl" and tuple(x[2]) == tuple(DISC)
    for ln in ("hip_yaw_link", "thigh", "shank", "ankle_link", "foot"):
        for pn in list(links[f"{side}_{ln}"]["parts"]):
            links[f"{side}_{ln}"]["parts"][pn] = [x for x in links[f"{side}_{ln}"]["parts"][pn] if not is_disc(x)]
    # ---- hip yaw bracket: hub ends at the yaw horn; side plates get the hip pitch cable notch (inboard end)
    hb = links[f"{side}_hip_yaw_link"]["parts"][f"{side}_hip_yaw_bracket"]
    for i, x in enumerate(hb):
        if x[0] == "cyl":
            c, a, r, h = x[1]
            hb[i] = ("cyl", (c, a, r, 232.1 - 213.0), x[2])
        elif x[0] == "rprism" and x[1][0] == "x":
            hb[i] = notch_x_plate(x, in_y(16.0), in_y(10.4), 164.0, 178.0)
    # ---- thigh: plates on the hip pitch horn / idler moved out, connectors widened, knee cable notch
    tf = links[f"{side}_thigh"]["parts"][f"{side}_thigh_frame"]
    new = []
    for x in tf:
        if x[0] == "rprism" and x[1][0] == "y" and abs(abs(x[1][1] - ys) - 20.0) <= 2.01 and abs(abs(x[1][2] - ys) - 20.0) <= 2.01:
            ax, a0, a1, outer, inn, holes = x[1]
            outboard = (min(a0, a1) - ys) * s > 0
            ya, yb = (out_y(PL0), out_y(PL1)) if outboard else (in_y(PL0), in_y(PL1))
            hl = horn_holes(185.0, 0.0) if outboard else horn_holes(185.0, 0.0, 4.0)
            new.append(PROFI("y", min(ya, yb), max(ya, yb), outer, inn, x[2], hl))
            if not outboard:     # idler boss
                yc, yd = in_y(20.0), in_y(PL0 + 0.3)
                new.append(PROFI("y", min(yc, yd), max(yc, yd), circle_loop(185.0, 0.0, HR_), [], x[2], [(185.0, 0.0, 4.0)]))
        elif x[0] == "rprism" and x[1][0] == "x":
            ax, a0, a1, outer, inn, holes = x[1]
            zs = [p[1] for p in outer]
            if min(zs) >= 139.9:                     # connectors z 140..146
                new.append(set_x_plate_y(x, out_y(PL1), in_y(PL1)))
            else:                                    # knee clamp walls (|dy| <= 17)
                new.append(notch_x_plate(x, in_y(17.0), in_y(10.4), 117.0, 131.0, roof="bottom"))
        else:
            new.append(x)
    links[f"{side}_thigh"]["parts"][f"{side}_thigh_frame"] = new
    # ---- shank: plates on the knee horn / idler moved out; side walls split (upper part widened), ankle cable window
    sf = links[f"{side}_shank"]["parts"][f"{side}_shank_frame"]
    new = []
    for x in sf:
        if x[0] == "rprism" and x[1][0] == "y" and abs(abs(x[1][1] - ys) - 20.0) <= 2.01 and abs(abs(x[1][2] - ys) - 20.0) <= 2.01:
            ax, a0, a1, outer, inn, holes = x[1]
            outboard = (min(a0, a1) - ys) * s > 0
            ya, yb = (out_y(PL0), out_y(PL1)) if outboard else (in_y(PL0), in_y(PL1))
            zs = [p[0] for p in outer]
            hl = (horn_holes(110.0, 0.0) if outboard else horn_holes(110.0, 0.0, 4.0)) if max(zs) > 100 else []
            new.append(PROFI("y", min(ya, yb), max(ya, yb), outer, inn, x[2], hl))
            if not outboard and max(zs) > 100:
                yc, yd = in_y(20.0), in_y(PL0 + 0.3)
                new.append(PROFI("y", min(yc, yd), max(yc, yd), circle_loop(110.0, 0.0, HR_), [], x[2], [(110.0, 0.0, 4.0)]))
        elif x[0] == "rprism" and x[1][0] == "x":
            ax, a0, a1, outer, inn, holes = x[1]
            low = set_x_plate_y(x, out_y(22.0), in_y(22.0), 30.0, 60.0)
            # ankle pitch cable window (inboard, z 43.7..54.5 + margin)
            la = low[1]
            win = rrect(min(in_y(10.4), in_y(17.6)), max(in_y(10.4), in_y(17.6)), 42.5, 55.5, 1.0)
            low = ("rprism", (la[0], la[1], la[2], la[3], [l for l in la[4] if max(p[1] for p in l) <= 60.0] + [win], la[5]), low[2])
            up = set_x_plate_y(x, out_y(PL1), in_y(PL1), 60.0, 76.0)
            ua = up[1]
            up = ("rprism", (ua[0], ua[1], ua[2], ua[3], [l for l in inn if min(p[1] for p in l) >= 60.0], ua[5]), up[2])
            new += [low, up]
        else:
            new.append(x)
    links[f"{side}_shank"]["parts"][f"{side}_shank_frame"] = new
    # ---- ankle link: bosses + holes on the ankle pitch horn / idler; back plate notched around the roll idler
    ab = links[f"{side}_ankle_link"]["parts"][f"{side}_ankle_bracket"]
    new = []
    for x in ab:
        if x[0] == "rprism" and x[1][0] == "y":
            ax, a0, a1, outer, inn, holes = x[1]
            outboard = (min(a0, a1) - ys) * s > 0
            new.append(PROFI("y", a0, a1, outer, inn, x[2], horn_holes(35.0, 0.0) if outboard else horn_holes(35.0, 0.0, 4.0)))
            if outboard:
                yc, yd = out_y(20.9), out_y(23.3)
                new.append(PROFI("y", min(yc, yd), max(yc, yd), circle_loop(35.0, 0.0, HR_), [], x[2], horn_holes(35.0, 0.0)))
            else:
                yc, yd = in_y(20.0), in_y(23.3)
                new.append(PROFI("y", min(yc, yd), max(yc, yd), circle_loop(35.0, 0.0, HR_), [], x[2], [(35.0, 0.0, 4.0)]))
        elif x[0] == "rprism" and x[1][0] == "x":
            ax, a0, a1, outer, inn, holes = x[1]
            Y0, Y1 = min(p[0] for p in outer), max(p[0] for p in outer)
            Z0, Z1 = min(p[1] for p in outer), max(p[1] for p in outer)
            R, zc = HR_ + 1.0, 35.0
            dz = Z0 - zc
            a = math.sqrt(R * R - dz * dz)
            t0 = math.atan2(dz, -a); t1 = math.atan2(dz, a)
            arc = []
            # from (ys - a, Z0) over the top to (ys + a, Z0)
            ang0, ang1 = t0 % (2 * math.pi), t1
            for k in range(25):
                t = ang0 + (ang1 - ang0) * k / 24
                arc.append((ys + R * math.cos(t), zc + R * math.sin(t)))
            pts = [(Y0, Z0)] + arc + [(Y1, Z0), (Y1, Z1), (Y0, Z1)]
            new.append(PROFI(ax, a0, a1, [(u, v, 0) for u, v in pts], [], x[2], holes))
        else:
            new.append(x)
    links[f"{side}_ankle_link"]["parts"][f"{side}_ankle_bracket"] = new
    # ---- foot: horn plate 3.1 mm forward (+x), base lengthened
    fs = links[f"{side}_foot"]["parts"][f"{side}_foot_sole"]
    for i, x in enumerate(fs):
        if x[0] == "rprism" and x[1][0] == "x" and abs(x[1][1] + 33.0) < 0.01:
            ax, a0, a1, outer, inn, holes = x[1]
            fs[i] = PROFI("x", -29.9, -27.4, outer, inn, x[2], horn_holes(ys, 35.0))
        elif x[0] == "rprism" and x[1][0] == "z" and abs(x[1][1] - 12.0) < 0.01 and abs(x[1][2] - 18.0) < 0.01:
            ax, a0, a1, outer, inn, holes = x[1]
            us = [p[0] for p in outer]; vs = [p[1] for p in outer]
            fs[i] = PROFI("z", a0, a1, rrect(min(us), -27.4, min(vs), max(vs), 4.0), inn, x[2], holes)
    # ---- horns / idlers as parts of the driven links (horn outboard, idler inboard)
    def hcyl(c, axis, h):
        return CYL(c, axis, HR_, h, XC)
    sgn_axis = "y" if s > 0 else "-y"
    inw_axis = "-y" if s > 0 else "y"
    for ln, sv, zc in (("thigh", "hip_pitch", 185.0), ("shank", "knee", 110.0), ("ankle_link", "ankle_pitch", 35.0)):
        P = links[f"{side}_{ln}"]["parts"]
        P[f"{side}_{sv}_horn"] = [hcyl((0.0, out_y(17.0), zc), sgn_axis, HORN_T)]
        P[f"{side}_{sv}_idler"] = [hcyl((0.0, in_y(17.0), zc), inw_axis, IDLER_T)]
    F = links[f"{side}_foot"]["parts"]
    F[f"{side}_ankle_roll_horn"] = [hcyl((-34.0, ys, 35.0), "x", HORN_T)]
    F[f"{side}_ankle_roll_idler"] = [hcyl((-68.0, ys, 35.0), "-x", IDLER_T)]
    Y = links[f"{side}_hip_yaw_link"]["parts"]
    Y[f"{side}_hip_yaw_horn"] = [hcyl((0.0, ys, 236.0), "-z", HORN_T)]
    Y[f"{side}_hip_yaw_idler"] = [hcyl((0.0, ys, 270.0), "z", IDLER_T)]


def fix_shelf(T_):
    pins = [(-53, -50), (-53, 50), (26, 50)]
    wins = [(-52, -16, -44, -16), (-52, -16, 16, 44), (13, 38, -44, -16), (13, 38, 16, 44)]
    T_["electronics_shelf"] = [PROFI("z", 272, 276, rrect(-62, 48, -54.5, 54.5, 10),
                                     [rrect(x0, x1, y0, y1, 4.0) for x0, x1, y0, y1 in wins], PRINT,
                                     [(x, y, 1.5) for x, y in pins] + [(0, -40, 11.5), (0, 40, 11.5)])]


def fix_nod(links):
    H = links["head"]["parts"]
    dy = -3.7
    H["head_nod_servo_XL430"] = [translate(x, (0, dy, 0)) for x in H["head_nod_servo_XL430"]]
    hb = H["head_back"]
    for i, x in enumerate(hb):
        if x[0] == "rprism" and x[1][0] == "z" and bbox(x)[1] == (-18, 18) and abs(bbox(x)[0][0] + 41) < 0.01:
            ax, a0, a1, outer, inn, holes = x[1]
            hb[i] = PROFI("z", a0, a1, rrect(-41, 2, -21.5, 18, 0), inn, x[2], holes)
    sx, sz = B14.NOD_S
    links["nod_crank"]["parts"]["nod_servo_horn"] = [CYL((sx, 13.3, sz), "y", HR_, HORN_T, XC)]
    H["head_nod_servo_idler"] = [CYL((sx, -20.7, sz), "-y", HR_, IDLER_T, XC)]


def fix_arm_horns(T_):
    for s, nm in ((-1, "R"), (1, "L")):
        Y = lambda y: s * y
        yr = lambda a, b: (min(Y(a), Y(b)), max(Y(a), Y(b)))
        T_[f"shoulder_shaft_{nm}"] = [PROF("y", *yr(50.0, 81.1), circle_loop(SHZ, 0, 4.0), STEEL)]
        T_[f"arm_{nm}_horn_adapter"] = [PROFI("y", *yr(77.6, 81.1), circle_loop(SHZ, 0, 11.0), [], PRINT, [(SHZ, 0, 4.0)])]
        T_[f"arm_{nm}_horn_XC430"] = [CYL((0, Y(85.0), SHZ), "-y" if s > 0 else "y", HR_, HORN_T, XC)]
        T_[f"arm_{nm}_idler_XC430"] = [CYL((0, Y(119.0), SHZ), "y" if s > 0 else "-y", HR_, IDLER_T, XC)]


# ======================================================================= OpenCR, deeper back, Jetson battery
BACK_X = -93.0                                   # new back of the torso shell (was -85)
CA_NEW = (70.0 - BACK_X) / 2
CU_NEW = 70.0 - CA_NEW
OC_X0, OC_ZC = -69.55, 280.0                     # board top face x, board centre z (power edge up, USB down)
import opencr_boxes


def oc_box(a, b, rgb):
    """STEP board frame -> world: z_s -> -x, x_s -> +y, y_s -> -z"""
    x0, x1 = OC_X0 - b[2], OC_X0 - a[2]
    y0, y1 = a[0] - 0.94, b[0] - 0.94
    z0, z1 = OC_ZC - (b[1] + 1.65), OC_ZC - (a[1] + 1.65)
    return RB(x0, x1, y0, y1, z0, z1, rgb, "x", 0.3)


def opencr_parts():
    board, parts = [], []
    for n, a, b in opencr_boxes.OPENCR_BOXES:
        if n == "Part 1" and (b[0] - a[0]) > 70:
            board.append(oc_box((a[0], a[1], -1.55), (b[0], b[1], 0.0), (0.15, 0.35, 0.6)))
            continue
        if (b[0] - a[0]) < 0.4 or (b[1] - a[1]) < 0.4:
            continue
        parts.append(oc_box(a, b, (0.12, 0.12, 0.12) if ("JST" in n or "Pin" in n or "SMW" in n or "0220" in n) else ELEC))
    # TTL plugs (EHR-3 on the B3B headers) + cable bend room, measured from the XC430 STEP plug height
    plugs = []
    for n, a, b in opencr_boxes.OPENCR_BOXES:
        if n.startswith("JST_B3B"):
            plugs.append(oc_box((a[0], a[1], 6.0), (b[0], b[1], 13.1), (0.9, 0.9, 0.9)))
            plugs.append(oc_box((a[0] + 0.5, a[1] + 1.5, 13.1), (b[0] - 0.5, b[1] - 1.5, 20.0), (0.05, 0.05, 0.05)))
    return board, parts, plugs


def fix_opencr(T_):
    board, parts, plugs = opencr_parts()
    T_["opencr_1_0"] = board + parts
    T_["opencr_ttl_plugs_cables"] = plugs
    fr = T_["torso_frame"]
    for i, x in enumerate(fr):
        b_ = bbox(x)
        # back wall: notch at the top for the DC-jack / power-switch pins, window for the switch pins
        if x[0] == "rprism" and x[1][0] == "x" and abs(b_[0][0] + 65) < 0.01 and abs(b_[0][1] + 63) < 0.01 and b_[2][1] > 329:
            ax, a0, a1, outer, inn, holes = x[1]
            Y0, Y1 = min(p[0] for p in outer), max(p[0] for p in outer)
            Z0, Z1 = min(p[1] for p in outer), max(p[1] for p in outer)
            pts = [(Y0, Z0), (Y1, Z0), (Y1, Z1), (41.0, Z1), (41.0, 307.0), (10.0, 307.0), (10.0, Z1), (Y0, Z1)]
            fr[i] = PROFI(ax, a0, a1, [(u, v, 0) for u, v in pts], list(inn), x[2], holes)
        # bottom OpenCR channels end below the lowest TTL header
        if x[0] == "rprism" and x[1][0] == "z" and abs(x[1][2] - 237.0) < 0.01 and b_[0][0] < -71 and b_[0][1] <= -64.9:
            ax, a0, a1, outer, inn, holes = x[1]
            fr[i] = ("rprism", (ax, a0, 235.5, outer, inn, holes), x[2])
    # top-left channel (and its gusset) sat on the part of the back wall now notched for the power-switch pins
    fr[:] = [x for x in fr if not (bbox(x)[1][0] > 34 and bbox(x)[1][1] < 40 and bbox(x)[2][0] > 318 and bbox(x)[0][0] < -70)]
    # shorts flange: back edge of its hole moved back so the OpenCR USB plug can come in from below
    sh = T_["shorts"]
    for i, x in enumerate(sh):
        if x[0] == "rprism" and x[1][0] == "z" and abs(x[1][1] - 235.0) < 0.01 and x[1][4]:
            ax, a0, a1, outer, inn, holes = x[1]
            inn2 = [[(-86.0 if abs(p[0] + 72.0) < 0.01 else p[0], p[1], p[2]) for p in l] for l in inn]
            sh[i] = ("rprism", (ax, a0, a1, outer, inn2, holes), x[2])
    # lid plug: notch in the back bar over the jack pins (rebuild from the U outline, clip again)
    do, di = top_inset()
    clip = ccw(cover_outline(di + 0.5))
    plugU = [(-45.0, -60.0), (50.0, -60.0), (50.0, -53.0), (-58.0, -53.0), (-58.0, 53.0), (50.0, 53.0), (50.0, 60.0),
             (-45.0, 60.0), (-65.0, 40.0), (-65.0, 31.0), (-63.0, 31.0), (-63.0, 10.0), (-65.0, 10.0), (-65.0, -40.0)]
    lid = T_["lid"]
    for i, x in enumerate(lid):
        if x[0] == "rprism" and x[1][0] == "z" and x[1][1] >= 329.9 and x[1][2] <= 336.4 and len(x[1][3]) >= 8:
            ax, a0, a1, outer, inn, holes = x[1]
            poly = sh_clip(plugU, clip)
            lid[i] = PROFI("z", a0, a1, [(u, v, 0) for u, v in poly], [], x[2], holes)


def jetson_battery(T_):
    T_["battery_jetson_2S_1500"] = [RB(35.0, 50.0, -44.0, 44.0, 235.5, 270.5, BATT, "x", 2)]


# ======================================================================= speaker + microphone in the head
def audio(links):
    H = links["head"]["parts"]
    sx = 0.0
    # 40 mm 3 W speaker behind the mouth, in a ring on the face plate's back
    H["speaker_40mm_3W"] = [CYL((46.0, 0.0, 431.0), "x", 20.0, 18.0, (0.15, 0.15, 0.15)),
                            CYL((64.0, 0.0, 431.0), "x", 20.0, 0.8, (0.5, 0.5, 0.5))]
    fp = H["face_plate"]
    fp.append(PROFI("x", 58.0, 68.5, circle_loop(0.0, 431.0, 22.6), [circle_loop(0.0, 431.0, 20.4)], WHITE))
    H["amp_MAX98357A"] = [RB(20.0, 38.0, -9.0, 9.0, 366.2, 369.4, ELEC, "z", 0.5)]
    H["mic_INMP441"] = [RB(66.0, 67.2, -44.0, -31.0, 446.5, 459.5, ELEC, "x", 0.5)]
    # perforated mouth insert, pin hole in the right eye for the mic
    zc, r_in, r_out = B14.SMILE[0], B14.SMILE[1], B14.SMILE[2]
    rm = (r_in + r_out) / 2
    mh = [(rm * math.cos(math.radians(a)), zc + rm * math.sin(math.radians(a)), 0.8) for a in np.arange(236.0, 305.0, 7.5)]
    H["mouth"] = [("rprism", (x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], list(x[1][5]) + mh), x[2]) for x in H["mouth"]]
    H["eye_R"] = [("rprism", (x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], list(x[1][5]) + [(-37.5, 453.0, 0.7)]), x[2])
                  for x in H["eye_R"]]
    H["eye_R"] = H["eye_R"] + [PROFI("x", 64.0, 68.5, rrect(-45.5, -29.5, 445.0, 461.0, 1.0), [rrect(-44.2, -30.8, 446.3, 459.7, 0.5)], NAVY)]



# ======================================================================= arms: longer hand + cam-driven finger flap
CAM_R0, CAM_LIFT, CAM_A0, CAM_A1 = 11.5, 6.0, 20.0, 70.0
HINGE = (-3.0, 181.0)                            # flap hinge (x, z) in the arm at rest


def cam_f(theta_deg):
    t = (abs(theta_deg) - CAM_A0) / (CAM_A1 - CAM_A0)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def flap_angle(theta_deg):
    """flap opening (deg) for an arm angle: the cam pushes the rod down by CAM_LIFT*f, the rod tip sits 3 mm in front
    of the hinge on the flap's top face"""
    return math.degrees(math.atan(CAM_LIFT * cam_f(theta_deg) / 3.0))


def cam_outline():
    pts = []
    for th in np.arange(-180.0, 180.0, 3.0):
        R = CAM_R0 + CAM_LIFT * cam_f(th)
        t = math.radians(th)
        pts.append((SHZ - R * math.cos(t), -R * math.sin(t), 0))        # (z, x): follower direction (-sin, -cos)
    return pts


def arms_v17(links, joints, T_):
    Tw = 1.6
    for s, nm in ((-1, "R"), (1, "L")):
        Y = lambda y: s * y
        yr = lambda a, b: (min(Y(a), Y(b)), max(Y(a), Y(b)))
        outl = lambda d: [(x, Y(yy)) for x, yy in B16.arm_outline(d)]
        A = links[f"{nm}_arm"]["parts"]
        A.pop(f"arm_{nm}_hand", None); A.pop(f"arm_{nm}_hand_pad", None)
        zb, hb = 262.0, 185.0
        hand = [("loft", ("z", ((hb, outl(0.0), outl(Tw)), (zb, outl(0.0), outl(Tw))), True), WHITE),
                ("loft", ("z", ((zb - 2.0, outl(0.5), outl(Tw + 1.8)), (zb, outl(0.5), outl(Tw + 1.8))), True), WHITE),
                ("loft", ("z", ((zb - 0.6, outl(Tw + 0.2), outl(Tw + 1.8)), (zb + 6.0, outl(Tw + 0.2), outl(Tw + 1.8))), True), WHITE)]
        rear = sh_clip(ccw([(p[0], p[1]) for p in outl(0.0)]), ccw([(-100, -300), (-7.5, -300), (-7.5, 300), (-100, 300)]))
        hand.append(PROFI("z", 177.0, hb + 0.5, [(u, v, 0) for u, v in rear], [], WHITE))
        hx, hz = HINGE
        for ya, yb in ((ARM_Y0, ARM_Y0 + 2.0), (ARM_Y1 - 2.2, ARM_Y1)):
            hand.append(PROFI("y", *yr(ya, yb), rrect(177.0, 187.0, hx - 5.0, hx + 5.0, 3.0), [], WHITE, [(hz, hx, 1.0)]))
        for gx0, gx1 in ((2.3, 5.3), (-5.3, -2.3)):          # rod guides with 45-deg undersides (hand prints upright)
            gp = [(Y(ARM_Y0 + Tw - 0.3), 189.0), (Y(84.6), 194.6), (Y(84.6), 201.0), (Y(ARM_Y0 + Tw - 0.3), 201.0)]
            hand.append(PROFI("x", gx0, gx1, [(u, v, 0) for u, v in gp], [], WHITE))
        A[f"arm_{nm}_hand"] = hand
        A[f"arm_{nm}_flap_pin"] = [PROF("y", *yr(ARM_Y0, ARM_Y1), circle_loop(hz, hx, 1.0), STEEL)]
        # follower rod: rides on the cam at 6 o'clock, pushes the flap 3 mm in front of the hinge
        A[f"arm_{nm}_finger_rod"] = [RB(-2.0, 2.0, *yr(79.2, 84.6), 270.0, SHZ - CAM_R0, PRINT, "z", 0.5),
                                     RB(-2.0, 2.0, *yr(81.0, 84.6), hb + 0.3, 270.5, PRINT, "z", 0.5)]
        A[f"arm_{nm}_shell_in"] = A[f"arm_{nm}_shell_in"] + [RB(2.3, 5.3, *yr(ARM_Y0 + Tw, 84.6), 272.0, 278.0, SKY, "z", 0.5),
                                                               RB(-5.3, -2.3, *yr(ARM_Y0 + Tw, 84.6), 272.0, 278.0, SKY, "z", 0.5)]
        # flap (finger block) on its own link
        arc = [(hz - 4.0 * math.cos(p), hx - 4.0 * math.sin(p), 0) for p in np.linspace(0, math.pi, 13)]
        outline = [(hb, 14.5, 0), (177.0, 14.5, 0)] + arc
        flap = [PROFI("y", *yr(79.5, ARM_Y1 - 2.5), outline, [], WHITE, [(hz, hx, 1.1)]),
                RB(-1.0, 14.0, *yr(81.0, ARM_Y1 - 3.0), 176.0, 177.0, RUBBER, "z", 1.0)]
        links[f"{nm}_hand_flap"] = {"origin": (hx, Y(100.0), hz), "parts": {f"arm_{nm}_flap": flap}}
        joints.append({"name": f"{nm}_hand_flap", "parent": f"{nm}_arm", "child": f"{nm}_hand_flap", "axis": "y",
                       "point_mm": (hx, Y(100.0), hz), "lower_deg": 0, "upper_deg": 70,
                       "actuator": "passive (shoulder cam -> rod; opens with |arm angle|)"})
        # cam lobe on the torso-fixed horn adapter
        T_[f"arm_{nm}_horn_adapter"] = T_[f"arm_{nm}_horn_adapter"] + \
            [PROFI("y", *yr(79.1, 81.1), cam_outline(), [], PRINT, [(SHZ, 0, 4.0)])]


def shorten_thigh_shell(links):
    for side in ("R", "L"):
        P = links[f"{side}_thigh"]["parts"]
        new = []
        for x in P[f"{side}_thigh_shell"]:
            x2 = clamp_rect(x, "z", hi=139.5)
            if x2 is not None:
                new.append(x2)
        P[f"{side}_thigh_shell"] = new


def build_model():
    for m in (B14, B16):
        m.CU, m.CA = CU_NEW, CA_NEW
    B14.OCR.update(z0=OC_ZC - 52.5, z1=OC_ZC + 52.5)
    links, joints, fasteners = B16.build_model()
    for side in ("R", "L"):
        fix_leg(links, side)
    T_ = links["torso"]["parts"]
    fix_shelf(T_)
    fix_nod(links)
    fix_arm_horns(T_)
    fix_opencr(T_)
    shorten_thigh_shell(links)
    arms_v17(links, joints, T_)
    jetson_battery(T_)
    audio(links)
    return links, joints, fasteners


write = B16.write

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out17"
    os.makedirs(out, exist_ok=True)
    w = write(links, f"{out}/biped_robot_v17.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))
