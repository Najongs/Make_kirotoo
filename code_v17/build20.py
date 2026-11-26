"""v20 = v19 + hip roll (layout B of sim/walk/hiproll/HIPROLL_STUDY.md) + servo assignment + generous fits.

Hip roll (layout B, roll axis meets the pitch axis at z 185, leg length and hip spacing +-40 unchanged)
  * chain: hip yaw -> hip roll (axis x, new link <S>_hip_roll_link, origin (0, ys, 185)) -> hip pitch
  * the roll servo (XM430-W350) lies behind the pitch servo, horn forward (+x) on the roll axis; it sits in the hip
    yaw link between two side plates (case screws) under a rear block that joins the yaw top plate
  * the roll output is two plates on the pitch servo case: the back plate takes the roll horn (4 x M2, square
    pattern so it misses the pitch servo case screws), the front plate turns on a steel pin d 4 pressed into a front
    pillar of the yaw link; both plates are the pitch servo clamp walls of v19 (case screws 3, 4 / 13, 14)
  * the yaw top plate is a narrow spine (+-8 mm) where the rolling parts pass, full width only under the hub
  * the yaw hub gets the 4 yaw horn screws (M2 from below, counterbored, 4 mm flange)
  * a navy cover (snap barbs under the side plates) hides the roll servo behind the hip
Servos: XM430 hip roll / hip pitch / knee / ankle roll, XC430 ankle pitch, XL430 hip yaw / arms / head nod
Fits (no more test prints): pockets >= 0.35 per side (v19 0.425 kept), one-sided seats 0.3, 0.5 mm stepped entry
relief on pocket walls, CASE_HOLE_D 2.7 (coupon v2), HORN_HOLE_D 2.4 [estimate], press fits +0.1 per side,
snap / sliding gaps >= 0.25.
"""
import os, json, math, re
os.environ.setdefault("KITORU_CASE_HOLE_D", "2.7")
os.environ.setdefault("KITORU_HORN_HOLE_D", "2.4")
import numpy as np
import build19 as B19
from build19 import *          # noqa
import build4, geom4
from build import IDX, UVI, STEEL
from build4 import CYL
import build18 as B18
from build6 import PROFI
from shapes import rrect
import x430_holes as X
import servo_fit as SF

ROLL_Z = 185.0
PITCH_GAP = 0.425           # pitch servo clamp gap (v19)
ROLL_GAP = 0.425            # roll servo side plate gap
PL = 3.85                   # plate thickness
SEAT_GAP = 0.3              # one-sided seats
POCKET_MIN = 0.35
ENTRY_RELIEF = 0.5
FIT_PLUS = 0.1              # press fits, per side
SNAP_MIN = 0.25
NAVY = (0.05, 0.19, 0.38)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out20")
LOG = {}
SERVO_KIND = {"hip_yaw": "XL430-W250", "hip_roll": "XM430-W350", "hip_pitch": "XM430-W350", "knee": "XM430-W350",
              "ankle_pitch": "XC430-W240", "ankle_roll": "XM430-W350", "arm_swing": "XL430-W250"}


def _ring(cu, cv, r):
    return rrect(cu - r, cu + r, cv - r, cv + r, r)


def _diag(cu, cv, r=8.0):
    d = r / math.sqrt(2)
    return [(cu + d, cv + d), (cu - d, cv + d), (cu + d, cv - d), (cu - d, cv - d)]


def _clip_v(outer, vmax):
    """profile loop [(u, v, r)] clipped to v <= vmax (vertex polygon, radii kept, new corners r 0)"""
    P = [(u, v) for u, v, _ in outer]; rad = {(round(u, 6), round(v, 6)): r for u, v, r in outer}
    P = B18._clip(P, 0.0, 1.0, vmax)
    return [(u, v, rad.get((round(u, 6), round(v, 6)), 0.0)) for u, v in P]


PILLAR_Z0 = 160.0           # front pillar bottom (thigh / knee servo arc about the pitch axis passes 8 mm below)
SKIRT_Z0 = 205.0            # shorts skirt bottom: yaw bracket parts beyond r 25 of the yaw axis stay 1 mm below it


def hip_roll(links, joints, side):
    s = -1 if side == "R" else 1
    ys = 40.0 * s
    Y = links[f"{side}_hip_yaw_link"]["parts"]
    yb = Y[f"{side}_hip_yaw_bracket"]
    roll = {"origin": (0.0, ys, ROLL_Z), "parts": {}}
    R = roll["parts"]
    # ---- pitch servo + its clamp walls move to the roll link; walls cut to z 197 (they must pass under the spine)
    R[f"{side}_hip_pitch_servo_XM430"] = Y.pop(f"{side}_hip_pitch_servo_XM430")
    walls = [x for x in yb if x[0] == "rprism" and x[1][0] == "x"]
    back = next(x for x in walls if x[1][2] < 0); front = next(x for x in walls if x[1][1] > 0)
    hr = B19.HORN_HOLE_D / 2
    for nm, w in (("back", back), ("front", front)):
        ax, a0, a1, outer, inn, holes = w[1]
        o2 = _clip_v(outer, 197.0)
        holes = [h for h in holes if h[1] + h[2] < 197.0]
        if nm == "back":       # roll horn: square pattern at r 8 + centre clearance
            holes = holes + [(ys + 8.0, ROLL_Z, hr), (ys - 8.0, ROLL_Z, hr), (ys, ROLL_Z + 8.0, hr), (ys, ROLL_Z - 8.0, hr), (ys, ROLL_Z, 2.5)]
        else:                  # roll pin bore (d 4 pin, running clearance 0.25 per side)
            holes = holes + [(ys, ROLL_Z, 2.25)]
        R[f"{side}_hip_roll_plate_{nm}"] = [PROFI(ax, a0, a1, o2, inn, w[2], holes)]
    # ---- roll servo (case box like the other X430 boxes), horn / idler on the roll link
    ox = back[1][1] - 0.3 - 20.9            # horn face 0.3 behind the back plate
    sv = [x for x in R[f"{side}_hip_pitch_servo_XM430"]][0]
    rgb_servo = sv[2]
    Y[f"{side}_hip_roll_servo_XM430"] = [PROFI("x", ox - 17.0, ox + 17.0, rrect(ys - 14.25, ys + 14.25, ROLL_Z - 35.25, ROLL_Z + 11.25, 3.25), [], rgb_servo)]
    horn_rgb = links["R_thigh"]["parts"]["R_hip_pitch_horn"][0][2]
    R[f"{side}_hip_roll_horn"] = [CYL((ox + 17.0, ys, ROLL_Z), "x", 10.25, 3.9, horn_rgb)]
    R[f"{side}_hip_roll_idler"] = [CYL((ox - 17.0, ys, ROLL_Z), "-x", 10.25, 3.0, horn_rgb)]
    # ---- yaw link: hub with horn screws, top plate (centre + spine), rear block, side plates, front pillar, pin
    rgb = yb[1][2]
    cb = [(x, y, 2.3) for x, y in _diag(0.0, ys)] + [(0.0, ys, 2.5)]      # counterbores / centre clearance
    sc = [(x, y, hr) for x, y in _diag(0.0, ys)] + [(0.0, ys, 2.5)]
    hub_r = 12.5 - FIT_PLUS
    xr0 = ox - 17.0 - 3.0 - 1.8                 # behind the idler
    pf0 = front[1][2] + 0.5                     # front pillar starts 0.5 in front of the front plate
    side_out = (ys - 14.25 - ROLL_GAP - PL, ys - 14.25 - ROLL_GAP)
    side_in = (ys + 14.25 + ROLL_GAP, ys + 14.25 + ROLL_GAP + PL)
    y_lo, y_hi = side_out[0], side_in[1]
    win = [(178.0, xr0, 0.0), (178.0, ox - 8.0, 0.0), (164.0, ox - 8.0, 0.0), (164.0, xr0, 0.0)]   # cable notch to the back
    def side_plate(y0, y1):
        # closed cable window (bridged roof prints without support; the plug 9.5 x 7.1 passes the 14 x 12 window)
        o = rrect(150.0, 201.0, xr0, ox + 14.0, 1.0)
        return PROFI("y", y0, y1, o, [rrect(164.0, 178.0, xr0 + 1.5, ox - 8.0, 1.0)], rgb)
    yb_new = [PROFI("z", 213.0, 228.0, _ring(0.0, ys, hub_r), [], rgb, cb),
              PROFI("z", 228.0, 232.1, _ring(0.0, ys, hub_r), [], rgb, sc),
              PROFI("z", 201.0, 213.0, rrect(-13.0, 13.0, ys - 12.5, ys + 12.5, 3.0), [], rgb, cb),
              PROFI("z", 201.0, 213.0, rrect(-20.0, pf0 + 3.5, ys - 8.0, ys + 8.0, 2.0), [], rgb, cb),
              PROFI("z", 196.75, SKIRT_Z0 - 1.0, rrect(xr0, -19.0, y_lo, y_hi, 2.0), [], rgb),
              side_plate(*sorted(side_out)), side_plate(*sorted(side_in)),
              PROFI("x", pf0, pf0 + 3.5, rrect(ys - 8.0, ys + 8.0, PILLAR_Z0, 201.0, 1.0), [], rgb, [(ys, ROLL_Z, 2.0)])]
    Y[f"{side}_hip_yaw_bracket"] = yb_new
    px0 = front[1][1] + 0.25                    # pin from just inside the front plate to the pillar's front face
    Y[f"{side}_hip_roll_pin_4mm"] = [CYL((px0, ys, ROLL_Z), "x", 2.0, pf0 + 3.5 - px0, STEEL)]
    # ---- cover (navy) around the roll servo, snap barbs under the side plates
    yo, yi = ys - 14.25 - ROLL_GAP - PL, ys + 14.25 + ROLL_GAP + PL     # side plate outer faces (R: yo < yi)
    lo_y, hi_y = min(yo, yi), max(yo, yi)
    cx0, cx1 = xr0 - 0.5 - 1.5, ox + 14.0
    cov = [PROFI("x", cx0, cx0 + 1.5, rrect(lo_y - 2.0, hi_y + 2.0, 146.0, 201.0, 1.0), [], NAVY),
           PROFI("y", lo_y - 2.0, lo_y - 0.5, rrect(146.0, 201.0, cx0, cx1, 1.0), [], NAVY),
           PROFI("y", hi_y + 0.5, hi_y + 2.0, rrect(146.0, 201.0, cx0, cx1, 1.0), [], NAVY),
           PROFI("z", 146.0, 147.5, rrect(cx0, cx1, lo_y - 2.0, hi_y + 2.0, 1.0), [], NAVY),
           # barbs: 1.2 mm under the side plates (bottom z 150), 0.3 below them
           PROFI("y", lo_y - 0.5, lo_y + 1.0, rrect(148.5, 149.7, ox - 10.0, ox + 2.0, 0.3), [], NAVY),
           PROFI("y", hi_y - 1.0, hi_y + 0.5, rrect(148.5, 149.7, ox - 10.0, ox + 2.0, 0.3), [], NAVY)]
    Y[f"{side}_hip_cover"] = cov
    links[f"{side}_hip_roll_link"] = roll
    # ---- joints
    for j in joints:
        if j["name"] == f"{side}_hip_pitch":
            j["parent"] = f"{side}_hip_roll_link"
            j["upper_deg"] = 5.0          # v20 trims: +16.9 at roll 0, but only 5-10 at roll -15..-20 (ranges v20 pair map)
    # roll range (hiproll_range_v20, design solids): self contact at -23 / +23 (outward 23 - 5 = 18); inward
    # (R +, L -) the feet come within 3 mm of the other leg past +2 deg (feet 11 mm apart at zero)
    lo_r, hi_r = (-18.0, 2.0) if side == "R" else (-2.0, 18.0)
    k = next(i for i, j in enumerate(joints) if j["name"] == f"{side}_hip_yaw")
    joints.insert(k + 1, {"name": f"{side}_hip_roll", "parent": f"{side}_hip_yaw_link", "child": f"{side}_hip_roll_link",
                          "axis": "x", "point_mm": (0.0, ys, ROLL_Z), "lower_deg": lo_r, "upper_deg": hi_r,
                          "actuator": "XM430-W350"})
    return dict(side=side, roll_servo_x=(round(ox - 17, 2), round(ox + 17, 2)), horn_face_x=round(ox + 20.9, 2),
                side_plates_y=[side_out, side_in], front_pillar_x=(round(pf0, 2), round(pf0 + 3.5, 2)), cover_x=(round(cx0, 2), round(cx1, 2)))


# ---------------------------------------------------------------- hip roll, layout A (owner 2026-11-26)
LAYOUT = os.environ.get("KITORU_HIPROLL", "A")
# layout A stack: roll servo top 0.3 under the yaw plate (z 210), roll axis 11.25 below its top; its bottom corners
# sweep r = hypot(35.25, 14.25) = 38.02 about the roll axis; U bracket bottom plate (3.0) 1.0 below that circle;
# pitch servo top 0.3 under the plate, pitch axis 11.25 below the pitch servo top.  (Without the bottom plate the
# stack is 3.3 mm shorter, 36.8, but the U falls apart into a front and a back half held only by the servo cases.)
ROLL_CLEAR = 1.0
U_PLATE = 3.0
WALL_PLUS = 2.0             # hip pitch clamp walls in the U bracket: 3.85 -> 5.85
_ZR_A = 210.0 - 0.3 - 11.25
LEG_EXTRA_MIN = math.ceil((185.0 - (_ZR_A - math.hypot(35.25, 14.25) - ROLL_CLEAR - U_PLATE - SEAT_GAP - 11.25)) * 100) / 100
_le = os.environ.get("LEG_EXTRA_MM", "min")
LEG_EXTRA_MM = LEG_EXTRA_MIN if _le == "min" else float(_le)       # hip pitch axis drop below v19 (A)
assert LEG_EXTRA_MM >= LEG_EXTRA_MIN - 1e-6, f"LEG_EXTRA_MM {LEG_EXTRA_MM} < minimum {LEG_EXTRA_MIN}"
THIGH_EXTRA_MM = float(os.environ.get("THIGH_EXTRA_MM", "20"))     # thigh longer (hip pitch -> knee); kit 2026-11-26: +20
SHANK_EXTRA_MM = float(os.environ.get("SHANK_EXTRA_MM", "20"))     # shank longer (knee -> ankle); kit 2026-11-26: +20
LIFT_MM = (LEG_EXTRA_MM if LAYOUT == "A" else 0.0) + THIGH_EXTRA_MM + SHANK_EXTRA_MM    # feet below z 0 by this
LOWER = ("thigh", "shank", "ankle_link", "foot")


def _shift_link(L, dz):
    L["origin"] = tuple(L["origin"][:2]) + (L["origin"][2] + dz,)
    for pn, ss in L["parts"].items():
        L["parts"][pn] = [build4.to_local(x, (0.0, 0.0, -dz)) for x in ss]


def hip_roll_A(links, joints, side):
    """layout A (serial): roll servo upright under the yaw plate (horn axis x, horn forward), roll output = U bracket
    (front plate on the horn, back plate on the idler, bottom plate) that carries the hip pitch clamp walls; the
    pitch axis and the whole leg below move down by LEG_EXTRA_MM."""
    s = -1 if side == "R" else 1
    ys = 40.0 * s
    dz = -LEG_EXTRA_MM
    Y = links[f"{side}_hip_yaw_link"]["parts"]
    yb = Y[f"{side}_hip_yaw_bracket"]
    rgb = yb[1][2]
    hr = B19.HORN_HOLE_D / 2
    # ---- roll servo: top 0.3 under the yaw plate (z 210-213), horn 11.25 below its top
    top = 210.0 - SEAT_GAP
    zr = top - 11.25
    sv = Y[f"{side}_hip_pitch_servo_XM430"][0]
    Y[f"{side}_hip_roll_servo_XM430"] = [PROFI("x", -17.0, 17.0, rrect(ys - 14.25, ys + 14.25, zr - 35.25, zr + 11.25, 3.25), [], sv[2])]
    # ---- roll link: U bracket + pitch servo + clamp walls (moved down)
    roll = {"origin": (0.0, ys, zr), "parts": {}}
    R = roll["parts"]
    horn_rgb = links["R_thigh"]["parts"]["R_hip_pitch_horn"][0][2]
    R[f"{side}_hip_roll_horn"] = [CYL((17.0, ys, zr), "x", 10.25, 3.9, horn_rgb)]
    R[f"{side}_hip_roll_idler"] = [CYL((-17.0, ys, zr), "-x", 10.25, 3.0, horn_rgb)]
    zp = 185.0 + dz
    pt = zp + 11.25                                    # pitch servo top
    plate_bot, plate_top = pt + SEAT_GAP, pt + SEAT_GAP + U_PLATE
    r_corner = math.hypot(35.25, 14.25)                # roll servo bottom corner about the roll axis
    clear = zr - r_corner - plate_top
    assert clear >= ROLL_CLEAR - 1e-3, f"LEG_EXTRA_MM {LEG_EXTRA_MM}: roll servo corner clears the U plate by {clear:.2f} mm only"
    R[f"{side}_hip_pitch_servo_XM430"] = [build4.to_local(x, (0.0, 0.0, -dz)) for x in Y.pop(f"{side}_hip_pitch_servo_XM430")]
    walls = [x for x in yb if x[0] == "rprism" and x[1][0] == "x"]
    ub = []
    for w in walls:
        w2 = build4.to_local(w, (0.0, 0.0, -dz))
        ax, a0, a1, outer, inn, holes = w2[1]
        # clamp walls 2 mm thicker outward (walking bending at the wall foot 5.8 -> ~3.8 MPa, sim/robust strength)
        a0, a1 = (a0, a1 + WALL_PLUS) if a0 > 0 else (a0 - WALL_PLUS, a1)
        o2 = _clip_v(outer, plate_bot + 0.5)
        ub.append(PROFI(ax, a0, a1, o2, inn, w[2], [h for h in holes if h[1] + h[2] < plate_bot - 1.2]))
    xf0, xb1 = 17.0 + 3.9, -17.0 - 3.0                  # horn face / idler face
    # U plates 32 wide below the roll axis (walking bending 9.7 -> ~4 MPa, sim/robust strength), tapering to the
    # 21 wide rounded top above it: every point stays under z 212 at roll +-35 (shorts block from z 213 at x >= 20)
    up = [(ys - 16.0, plate_bot, 0.0), (ys + 16.0, plate_bot, 0.0), (ys + 16.0, zr + 2.5, 0.0), (ys + 10.5, zr + 6.5, 0.0),
          (ys + 10.5, zr + 10.5, 4.0), (ys - 10.5, zr + 10.5, 4.0), (ys - 10.5, zr + 6.5, 0.0), (ys - 16.0, zr + 2.5, 0.0)]
    hd = [(y, z, hr) for y, z in _diag(ys, zr)] + [(ys, zr, 2.5)]
    ub += [PROFI("x", xf0, xf0 + PL, up, [], rgb, hd),                       # front plate on the horn
           PROFI("x", xb1 - PL, xb1, up, [], rgb, hd),                       # back plate on the idler
           PROFI("z", plate_bot, plate_top, rrect(xb1 - PL, xf0 + PL, ys - 17.5, ys + 17.5, 2.0), [], rgb)]   # bottom plate
    R[f"{side}_hip_roll_bracket"] = ub
    links[f"{side}_hip_roll_link"] = roll
    # ---- yaw link: hub + plate z 210-213 + side plates gripping the roll servo
    cb = [(x, y, 2.3) for x, y in _diag(0.0, ys)] + [(0.0, ys, 2.5)]       # counterbores (driver from below)
    sc = [(x, y, hr) for x, y in _diag(0.0, ys)] + [(0.0, ys, 2.5)]       # M2 yaw horn screws
    hub = [PROFI("z", 213.0, 228.0, _ring(0.0, ys, 12.5 - FIT_PLUS), [], rgb, cb),
           PROFI("z", 228.0, 232.1, _ring(0.0, ys, 12.5 - FIT_PLUS), [], rgb, sc)]
    yin = 14.25 + PITCH_GAP
    Y[f"{side}_hip_yaw_bracket"] = hub + [
        PROFI("z", 210.0, 213.0, rrect(-19.0, 19.0, ys - yin - PL, ys + yin + PL, 2.0), [], rgb, cb),
        PROFI("y", ys - yin - PL, ys - yin, rrect(166.0, 213.0, -17.0, 17.0, 1.0), [], rgb),
        PROFI("y", ys + yin, ys + yin + PL, rrect(166.0, 213.0, -17.0, 17.0, 1.0), [], rgb)]
    # ---- the leg below the pitch axis moves down
    for k in LOWER:
        _shift_link(links[f"{side}_{k}"], dz)
    names = [f"{side}_hip_pitch", f"{side}_knee", f"{side}_ankle_pitch", f"{side}_ankle_roll"]
    for j in joints:
        if j["name"] in names:
            p = j["point_mm"]; j["point_mm"] = (p[0], p[1], p[2] + dz)
        if j["name"] == f"{side}_hip_pitch":
            j["parent"] = f"{side}_hip_roll_link"
            j["lower_deg"], j["upper_deg"] = -80.0, 30.0
    k = next(i for i, j in enumerate(joints) if j["name"] == f"{side}_hip_yaw")
    lo_r, hi_r = (-25.0, 2.0) if side == "R" else (-2.0, 25.0)
    joints.insert(k + 1, {"name": f"{side}_hip_roll", "parent": f"{side}_hip_yaw_link", "child": f"{side}_hip_roll_link",
                          "axis": "x", "point_mm": (0.0, ys, zr), "lower_deg": lo_r, "upper_deg": hi_r, "actuator": "XM430-W350"})
    return dict(side=side, layout="A", leg_extra_mm=LEG_EXTRA_MM, leg_extra_min_mm=LEG_EXTRA_MIN, roll_axis_z=round(zr, 2),
                pitch_axis_z=round(zp, 2), u_plate_z=(round(plate_bot, 2), round(plate_top, 2)), roll_corner_clear_mm=round(clear, 2))


def _stretch(sol, zc, d):
    """lower everything of a solid below z = zc by d (a solid that crosses zc gets longer)"""
    kind, a, rgb = sol
    if kind == "cyl":
        c = a[0]
        return (kind, ((c[0], c[1], c[2] - d),) + tuple(a[1:]), rgb) if c[2] < zc else sol
    if kind != "rprism":
        raise ValueError(kind)
    axn, a0, a1, outer, inn, holes = a
    if axn == "z":
        if a1 <= zc:
            return (kind, (axn, a0 - d, a1 - d, outer, inn, holes), rgb)
        return (kind, (axn, a0 - d, a1, outer, inn, holes), rgb) if a0 < zc else sol
    k = 1 if axn == "x" else 0                      # index of z in the outline (x -> (y, z), y -> (z, x))
    mv = lambda q: tuple((c - d if (j == k and q[k] < zc) else c) for j, c in enumerate(q))
    return (kind, (axn, a0, a1, [mv(q) for q in outer], [[mv(q) for q in l] for l in inn], [mv(h) for h in holes]), rgb)


def shank_plates_wider(links):
    """shank knee horn plates (y-plates, x +-12, v19 z 88-122) -> x +-16: walking bending at the knee end 7.7 -> ~4.5 MPa
    (sim/robust strength); the thigh parts that turn past them stay at |y - c| <= 17 or >= 30 mm from the knee axis"""
    out = []
    for side in ("R", "L"):
        ss = links[f"{side}_shank"]["parts"][f"{side}_shank_frame"]
        for i, x in enumerate(ss):
            a = x[1]
            if x[0] == "rprism" and a[0] == "y" and sorted({round(q[1], 3) for q in a[3]}) == [-12.0, 12.0]:
                o = [(q[0], 16.0 if q[1] > 0 else -16.0) + tuple(q[2:]) for q in a[3]]
                ss[i] = (x[0], (a[0], a[1], a[2], B18._fit_radii(o), a[4], a[5]), x[2]); out.append((side, i))
    return out


def leg_lengths(links, joints, dz):
    """THIGH_EXTRA_MM / SHANK_EXTRA_MM: the thigh is stretched across v19 z 160 (between the hip pitch horn and the
    knee servo), the shank across v19 z 85 (between the knee horn and the ankle pitch servo); dz = layout A drop"""
    out = []
    for side in ("R", "L"):
        # (the shank cut also moves down with the thigh extension: v19 z 85 is then at 85 + dz - THIGH_EXTRA_MM)
        for T, link, zc, below in ((THIGH_EXTRA_MM, "thigh", 160.0 + dz, ("shank", "ankle_link", "foot")),
                                   (SHANK_EXTRA_MM, "shank", 85.0 + dz - THIGH_EXTRA_MM, ("ankle_link", "foot"))):
            if not T:
                continue
            L = links[f"{side}_{link}"]
            for pn, ss in L["parts"].items():
                L["parts"][pn] = [_stretch(x, zc, T) for x in ss]
            for k in below:
                _shift_link(links[f"{side}_{k}"], -T)
            names = {"thigh": ("knee", "ankle_pitch", "ankle_roll"), "shank": ("ankle_pitch", "ankle_roll")}[link]
            for j in joints:
                if j["name"] in [f"{side}_{n}" for n in names]:
                    p = j["point_mm"]; j["point_mm"] = (p[0], p[1], p[2] - T)
            out.append(dict(side=side, link=link, extra_mm=T, cut_z=zc))
    return out


def rename_servos(links, joints):
    out = []
    for ln, L in links.items():
        for pn in list(L["parts"]):
            if re.match(r"yaw_servo_[RL]_XC430$", pn) or re.match(r"arm_[RL]_servo_XC430$", pn):
                new = pn.replace("XC430", "XL430"); L["parts"][new] = L["parts"].pop(pn); out.append((pn, new))
    for j in joints:
        for key, kind in SERVO_KIND.items():
            if j["name"].endswith(key):
                j["actuator"] = kind + j["actuator"][len(j["actuator"].split(" ")[0]):] if " " in j["actuator"] else kind
    return out


def seats(links, frames, mp):
    """one-sided seats -> SEAT_GAP; enclosing faces below POCKET_MIN -> POCKET_MIN (per solid, case-side face moved)"""
    rows, _ = SF.scan(links, mp, per_solid=True)
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["servo"], r["face"][:2]), []).append(r)
    for r in rows:
        if r["gap_mm"] > 0.5:
            continue
        opp = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y", "+z": "-z", "-z": "+z"}[r["face"][:2]]
        enclosed = any(q["gap_mm"] <= 0.5 for q in by.get((r["servo"], opp), []))
        target = POCKET_MIN if enclosed else SEAT_GAP
        if r["gap_mm"] >= target - 1e-6:
            continue
        fr = frames[r["servo"]]; Rm = np.c_[fr["X"], fr["Y"], fr["Z"]]
        ax = "xyz".index(r["face"][1]); sg = 1 if r["face"][0] == "+" else -1
        n_s = np.zeros(3); n_s[ax] = sg; nw = Rm @ n_s; k = int(np.argmax(np.abs(nw)))
        ss = links[r["link"]]["parts"][r["part"]]; s = ss[r["solid_index"]]
        if s[0] != "rprism" or IDX[s[1][0]] != k:
            out.append(dict(r, action="not handled")); continue
        d = target - max(r["gap_mm"], 0.0) + 0.025     # scan step margin
        axn, a0, a1, outer, inn, holes = s[1]
        p_s = np.zeros(3); p_s[ax] = {0: 14.25 * sg, 1: 11.25 if sg > 0 else -35.25, 2: 17.0 * sg}[ax]
        face_c = float((fr["O"] + Rm @ p_s)[k])
        if abs(a0 - face_c) <= abs(a1 - face_c):
            a0 += d * (1 if nw[k] > 0 else -1)
        else:
            a1 += d * (1 if nw[k] > 0 else -1)
        ss[r["solid_index"]] = ("rprism", (axn, min(a0, a1), max(a0, a1), outer, inn, holes), s[2])
        out.append(dict(r, action=f"{'pocket' if enclosed else 'seat'} gap {r['gap_mm']:.3f} -> {target}"))
    return out


def entry_relief(links, frames, mp, entries):
    """pocket walls: the 0.5 mm next to the entry edge is set back another 0.5 mm (stepped 0.5 x 0.5 chamfer).
    entries: {servo: world axis index and sign of the insertion side}"""
    rows, _ = SF.scan(links, mp, per_solid=True)
    out = []
    done = set()
    for r in rows:
        if r["gap_mm"] > 0.6 or r["servo"] not in entries:
            continue
        key = (r["link"], r["part"], r["solid_index"])
        if key in done:
            continue
        ss = links[r["link"]]["parts"][r["part"]]; s = ss[r["solid_index"]]
        if s[0] != "rprism":
            continue
        fr = frames[r["servo"]]; Rm = np.c_[fr["X"], fr["Y"], fr["Z"]]
        ax = "xyz".index(r["face"][1]); sg = 1 if r["face"][0] == "+" else -1
        n_s = np.zeros(3); n_s[ax] = sg; nw = Rm @ n_s; k = int(np.argmax(np.abs(nw)))
        if IDX[s[1][0]] != k:
            continue
        ek, esg = entries[r["servo"]]
        axn, a0, a1, outer, inn, holes = s[1]
        iu, iv = UVI[axn]
        if ek not in (iu, iv):
            continue
        coord = 0 if ek == iu else 1
        vals = [p[coord] for p in outer]
        end = max(vals) if esg > 0 else min(vals)
        # the case end on the entry side (world coordinate along ek)
        corners = [fr["O"] + Rm @ np.array([x, y, z]) for x in (-14.25, 14.25) for y in (-35.25, 11.25) for z in (-17, 17)]
        case_end = max(c[ek] for c in corners) if esg > 0 else min(c[ek] for c in corners)
        cut = end - esg * ENTRY_RELIEF
        def clip(loop, keep_side):
            P = [(p[0], p[1]) for p in loop]; rad = {(round(p[0], 6), round(p[1], 6)): p[2] for p in loop}
            a = [0.0, 0.0]; a[coord] = 1.0 if keep_side > 0 else -1.0
            c = cut if keep_side > 0 else -cut
            Q = B18._clip(P, a[0], a[1], c)
            return [(u, v, rad.get((round(u, 6), round(v, 6)), 0.0)) for u, v in Q]
        # main: the part away from the entry; strip: the 0.5 mm at the entry
        if esg > 0:
            main_loop = clip(outer, +1); strip_loop = clip(outer, -1)
        else:
            main_loop = clip(outer, -1); strip_loop = clip(outer, +1)
        strip_loop = [(u, v, 0.0) for u, v, _ in strip_loop]
        if len(main_loop) < 3 or len(strip_loop) < 3:
            continue
        main_loop = B18._fit_radii(main_loop)
        def area(L):
            return abs(sum(L[i][0] * L[(i + 1) % len(L)][1] - L[(i + 1) % len(L)][0] * L[i][1] for i in range(len(L)))) / 2 if len(L) > 2 else 0
        if area(strip_loop) < 0.1 or area(main_loop) < 1.0:
            continue
        p_s = np.zeros(3); p_s[ax] = {0: 14.25 * sg, 1: 11.25 if sg > 0 else -35.25, 2: 17.0 * sg}[ax]
        face_c = float((fr["O"] + Rm @ p_s)[k])
        b0, b1 = a0, a1
        if abs(a0 - face_c) <= abs(a1 - face_c):
            b0 += ENTRY_RELIEF * (1 if nw[k] > 0 else -1)
        else:
            b1 += ENTRY_RELIEF * (1 if nw[k] > 0 else -1)
        ss[r["solid_index"]] = ("rprism", (axn, a0, a1, main_loop, inn, holes), s[2])
        ss.append(("rprism", (axn, min(b0, b1), max(b0, b1), strip_loop, [], [h for h in holes]), s[2]))
        done.add(key)
        out.append(dict(servo=r["servo"], part=r["part"], solid=r["solid_index"], entry_axis="xyz"[ek] + ("+" if esg > 0 else "-"), edge=round(end, 2)))
    return out


def press_fits(links):
    """coaxial purchased cylinder / printed hole and purchased bore / printed shaft pairs: measure, then +FIT_PLUS"""
    out = []
    buy = ("bearing", "shaft", "pin_4mm", "flap_pin", "axle", "6805", "684", "688")
    for ln, L in links.items():
        cyls = []
        for pn, ss in L["parts"].items():
            if not any(k in pn for k in buy) or "rocker_pin" in pn:
                continue
            for s in ss:
                if s[0] == "cyl":
                    c, a, r, h = s[1]; a = np.array(a, float) / np.linalg.norm(a); c = np.array(c, float)
                    k0 = int(np.argmax(np.abs(a)))
                    cyls.append((pn, "outer", c, np.abs(a), r, sorted((c[k0], c[k0] + a[k0] * h))))
                elif s[0] == "rprism":
                    ax = s[1][0]; k = IDX[ax]; iu, iv = UVI[ax]
                    us = [p[0] for p in s[1][3]]; vs = [p[1] for p in s[1][3]]
                    rr = (max(us) - min(us)) / 2
                    if abs((max(vs) - min(vs)) / 2 - rr) < 0.05:
                        c = np.zeros(3); c[iu] = (max(us) + min(us)) / 2; c[iv] = (max(vs) + min(vs)) / 2; c[k] = s[1][1]
                        a = np.zeros(3); a[k] = 1.0
                        cyls.append((pn, "outer", c, a, rr, (s[1][1], s[1][2])))
                        for hu, hv, hrr in s[1][5]:
                            c2 = np.zeros(3); c2[iu] = hu; c2[iv] = hv; c2[k] = s[1][1]
                            cyls.append((pn, "bore", c2, a, hrr, (s[1][1], s[1][2])))
        for pn_b, kind, c, a, rb, ext in cyls:
            k = int(np.argmax(np.abs(a))); axc = "xyz"[k]; iu, iv = UVI[axc]
            for pn, ss in L["parts"].items():
                if pn == pn_b or any(t in pn for t in buy):
                    continue
                for i, s in enumerate(ss):
                    if s[0] != "rprism" or s[1][0] != axc:
                        continue
                    axn, a0, a1, outer, inn, holes = s[1]
                    if min(a1, ext[1]) - max(a0, ext[0]) <= 0.05:       # no axial overlap
                        continue
                    if kind == "outer":
                        for j, (hu, hv, hrr) in enumerate(holes):
                            if math.hypot(hu - c[iu], hv - c[iv]) < 0.2 and abs(hrr - rb) < 0.4:
                                nh = list(holes); nh[j] = (hu, hv, hrr + FIT_PLUS)
                                ss[i] = ("rprism", (axn, a0, a1, outer, inn, nh), s[2])
                                out.append(dict(link=ln, purchased=pn_b, printed=pn, kind="hole for purchased part",
                                                radial_fit_mm=round(hrr - rb, 3), new_radial_fit_mm=round(hrr + FIT_PLUS - rb, 3)))
                    else:
                        us = [p[0] for p in outer]; vs = [p[1] for p in outer]
                        rr = (max(us) - min(us)) / 2
                        cu, cv = (max(us) + min(us)) / 2, (max(vs) + min(vs)) / 2
                        if abs((max(vs) - min(vs)) / 2 - rr) < 0.05 and math.hypot(cu - c[iu], cv - c[iv]) < 0.2 and abs(rr - rb) < 0.4:
                            out.append(dict(link=ln, purchased=pn_b, printed=pn, kind="printed shaft in purchased bore",
                                            radial_fit_mm=round(rb - rr, 3), new_radial_fit_mm=round(rb - rr, 3),
                                            note="already set (hub radius 12.4)" if "hip_yaw_bracket" in pn else ""))
    return out


BOSS_W = 2.5                # ring of material around a thin-wall case hole
BOSS_T = 2.0


def hole_bosses(links, warnings, frames):
    """v19 case holes with < 1.2 mm wall: a boss ring (hole r + 2.5) on the outer face of the outermost plate"""
    out = []
    seen = set()
    for w in warnings:
        key = (w["servo"], w["hole"])
        if key in seen:
            continue
        seen.add(key)
        fr = frames[w["servo"]]; p, n = X.HOLES[w["hole"] - 1]; pw, nw = X.to_world(fr, p, n)
        k = int(np.argmax(np.abs(nw))); axc = "xyz"[k]; iu, iv = UVI[axc]; sg = 1 if nw[k] > 0 else -1
        part = links[w["link"]]["parts"][w["part"]]
        r = B19.CASE_HOLE_D / 2
        ext = [x[1][2] if sg > 0 else x[1][1] for x in part if x[0] == "rprism" and x[1][0] == axc
               and any(abs(h[0] - pw[iu]) < 0.01 and abs(h[1] - pw[iv]) < 0.01 for h in x[1][5])]
        if not ext:
            continue
        # a boss on the outer face would run into the shell latch beam (frame front, y +-3.8 around the leg
        # centre), so the hole is left out instead: these servos keep 6 of their 8 side screws
        for i, x in enumerate(part):
            if x[0] == "rprism" and x[1][0] == axc:
                a = x[1]
                part[i] = (x[0], (a[0], a[1], a[2], a[3], a[4], [h for h in a[5] if not (abs(h[0] - pw[iu]) < 0.01 and abs(h[1] - pw[iv]) < 0.01)]), x[2])
        out.append(dict(servo=w["servo"], hole=w["hole"], part=w["part"], action="hole removed (wall 0.5 mm; boss would hit the shell latch)"))
    return out


def cable_clips(links):
    """open J-hooks on the back of the leg frames (cable 3.5 x 1.5 slides in past a 1.2 mm lip: printed snap)"""
    out = []
    for side in ("R", "L"):
        ys = -40.0 if side == "R" else 40.0
        # thigh clip at z 112-118 (under the V-cut back wall top, z 119 at the leg centre); shank clip at z 38-44: inside r 24 of the ankle pitch axis (ankle roll servo front face r 34, foot horn plate
        # r 27.4 never sweep it); y kept between the case screw heads at leg centre +-6
        for ln, pn, zc in ((f"{side}_thigh", f"{side}_thigh_frame", 115.0), (f"{side}_shank", f"{side}_shank_frame", 41.0)):
            part = links[ln]["parts"][pn]
            backs = [x for x in part if x[0] == "rprism" and x[1][0] == "x" and x[1][2] < 0
                     and min(q[1] for q in x[1][3]) <= zc - 3 and max(q[1] for q in x[1][3]) >= zc + 3]
            if not backs:
                out.append((ln, "no back wall at z", zc)); continue
            xb = min(x[1][1] for x in backs)          # outer face of the back wall
            rgb = part[0][2]
            z0, z1 = zc - 3.0, zc + 3.0
            part += [PROFI("z", z0, z1, rrect(xb - 3.5, xb, ys - 3.3, ys - 1.8, 0.3), [], rgb),          # post
                     PROFI("z", z0, z1, rrect(xb - 3.5, xb - 2.0, ys - 3.3, ys + 3.2, 0.3), [], rgb),     # bar (2.0 mm channel)
                     PROFI("z", z0, z1, rrect(xb - 2.0, xb - 1.2, ys + 2.4, ys + 3.2, 0.2), [], rgb)]     # lip (1.2 mm opening)
            out.append((ln, round(xb, 2), zc))
    return out


LATCH_X0 = 18.6            # beam / saddle-bar inner face of the shell latch (frame front face at 18.5: 0.1 gap)


def latch_gaps(links):
    """shell beam latch: beam + saddle bar inner face 18.6 -> 18.75 (0.25 to the frame face, barb tip stays at
    17.6 so the catch depth grows 0.15); frame pad y +-5 -> +-4.95 (0.25 to the saddle legs at +-5.2)"""
    out = []
    xn = 18.5 + SNAP_MIN
    for side in ("R", "L"):
        ys = -40.0 if side == "R" else 40.0
        for ln, sh, fr in ((f"{side}_thigh", f"{side}_thigh_shell", f"{side}_thigh_frame"), (f"{side}_shank", f"{side}_shank_shell", f"{side}_shank_frame")):
            P = links[ln]["parts"]
            n = 0
            for i, x in enumerate(P[sh]):
                if x[0] != "rprism" or x[1][0] not in "zy":
                    continue
                a = x[1]; k = 0 if a[0] == "z" else 1          # index of the x coordinate in the outline
                if not any(abs(q[k] - LATCH_X0) < 1e-6 for q in a[3]):
                    continue
                d = xn - LATCH_X0
                o = [tuple((c + d if (j == k and (abs(c - LATCH_X0) < 1e-6 or abs(c - 19.8) < 1e-6)) else c) for j, c in enumerate(q)) for q in a[3]]
                P[sh][i] = (x[0], (a[0], a[1], a[2], o, a[4], a[5]), x[2]); n += 1
            m = 0
            for i, x in enumerate(P[fr]):
                a = x[1]
                if x[0] == "rprism" and a[0] == "z" and sorted({round(q[0], 3) for q in a[3]}) == [18.5, 21.5]:
                    vs = sorted({round(q[1], 3) for q in a[3]})
                    if len(vs) == 2 and abs(vs[1] - vs[0] - 10.0) < 1e-6:
                        o = [(q[0], ys + (4.95 if q[1] > ys else -4.95)) + tuple(q[2:]) for q in a[3]]
                        P[fr][i] = (x[0], (a[0], a[1], a[2], o, a[4], a[5]), x[2]); m += 1
            out.append(dict(link=ln, beam_solids=n, pads=m, beam_gap=(0.1, round(xn - 18.5, 2)), pad_gap=(0.2, 0.25)))
    return out


def narrow_latch(links):
    """v19 knee / ankle pitch case holes 1,2 (11,12) sat 0.65 mm from the latch groove (groove y +-4 around the leg
    centre, holes at +-6): groove -> +-2.6, beam + barb -> +-2.4, so the holes keep ~2 mm of wall (holes stay)"""
    out = []
    for side in ("R", "L"):
        ys = -40.0 if side == "R" else 40.0
        for ln, sh, fr in ((f"{side}_thigh", f"{side}_thigh_shell", f"{side}_thigh_frame"), (f"{side}_shank", f"{side}_shank_shell", f"{side}_shank_frame")):
            P = links[ln]["parts"]
            ng = 0
            for i, x in enumerate(P[fr]):
                a = x[1]
                if x[0] != "rprism" or a[0] != "x" or not a[4]:
                    continue
                inn = []
                for loop in a[4]:
                    us = sorted({round(q[0] - ys, 3) for q in loop})
                    if us == [-4.0, 4.0]:
                        loop = [(ys + (2.6 if q[0] > ys else -2.6),) + tuple(q[1:]) for q in loop]; ng += 1
                    inn.append(loop)
                P[fr][i] = (x[0], (a[0], a[1], a[2], a[3], inn, a[5]), x[2])
            nb = 0
            for i, x in enumerate(P[sh]):
                a = x[1]
                if x[0] != "rprism":
                    continue
                if a[0] == "z":
                    vs = sorted({round(q[1] - ys, 3) for q in a[3]})
                    if vs == [-3.8, 3.8]:
                        o = [(q[0], ys + (2.4 if q[1] > ys else -2.4)) + tuple(q[2:]) for q in a[3]]
                        P[sh][i] = (x[0], (a[0], a[1], a[2], o, a[4], a[5]), x[2]); nb += 1
                elif a[0] == "y" and abs(a[1] - (ys - 3.8)) < 1e-6 and abs(a[2] - (ys + 3.8)) < 1e-6:
                    P[sh][i] = (x[0], (a[0], ys - 2.4, ys + 2.4, a[3], a[4], a[5]), x[2]); nb += 1
            out.append(dict(link=ln, grooves=ng, beam_solids=nb))
    return out


def thin_wall_after(links, frames):
    """wall left around the v19 thin-wall holes after the latch change"""
    out = []
    for w in B19.LOG.get("wall_warnings", []):
        fr = frames[w["servo"]]; p, n = X.HOLES[w["hole"] - 1]; pw, nw = X.to_world(fr, p, n)
        part = links[w["link"]]["parts"][w["part"]]
        k = int(np.argmax(np.abs(nw))); axc = "xyz"[k]; iu, iv = UVI[axc]
        for x in part:
            if x[0] == "rprism" and x[1][0] == axc and any(abs(h[0] - pw[iu]) < 0.01 and abs(h[1] - pw[iv]) < 0.01 for h in x[1][5]) \
                    and abs(x[1][2] - 18.5) < 1e-6:
                a = x[1]
                me = (x[0], (a[0], a[1], a[2], a[3], a[4], [h for h in a[5] if not (abs(h[0] - pw[iu]) < 0.01 and abs(h[1] - pw[iv]) < 0.01)]), x[2])
                ws = [B19._wall_left([y if y is not x else me for y in part], me, (pw[iu], pw[iv]), B19.CASE_HOLE_D / 2)]
                out.append(dict(servo=w["servo"], hole=w["hole"], solid_x=(a[1], a[2]), wall_after=min(ws)))
    return out


def _cut(sol, n, c):
    """keep n . p <= c (world, n 3-vector) of an rprism; exact in the profile plane, conservative across the thickness"""
    if sol[0] != "rprism":
        return sol
    axn, a0, a1, outer, inn, holes = sol[1]
    k, (iu, iv) = IDX[axn], UVI[axn]
    cc = c - max(n[k] * a0, n[k] * a1)
    rad = {(round(q[0], 6), round(q[1], 6)): q[2] for q in outer}
    Q = B18._clip([(q[0], q[1]) for q in outer], n[iu], n[iv], cc)
    if len(Q) < 3:
        return None
    loop = B18._fit_radii([(u, v, rad.get((round(u, 6), round(v, 6)), 0.0)) for u, v in Q])
    inn2 = [l for l in inn if all(n[iu] * q[0] + n[iv] * q[1] <= cc - 1.2 for q in l)]
    holes2 = [h for h in holes if n[iu] * h[0] + n[iv] * h[1] + h[2] * math.hypot(n[iu], n[iv]) <= cc - 1.2]
    return ("rprism", (axn, a0, a1, loop, inn2, holes2), sol[2])


def _cut_part(ss, n, c, sel=lambda x: True):
    out, nc = [], 0
    for x in ss:
        if x[0] == "rprism" and sel(x):
            y = _cut(x, n, c)
            if y is not None:
                out.append(y)
            nc += 1
        else:
            out.append(x)
    ss[:] = out
    return nc


def _split_mid(ss, sel, coord, lo, hi, n, c):
    """solids picked by sel: the band lo..hi of profile coordinate `coord` (0 = u, 1 = v) is cut by n . p <= c,
    the rest of the solid stays whole"""
    out, nc = [], 0
    for x in ss:
        if x[0] != "rprism" or not sel(x):
            out.append(x); continue
        m = [0.0, 0.0, 0.0]; m[UVI[x[1][0]][coord]] = 1.0
        neg = [-v for v in m]
        left = _cut(x, m, lo)                                      # coord <= lo
        right = _cut(x, neg, -hi)                                  # coord >= hi
        mid = _cut(x, neg, -lo)                                    # lo <= coord <= hi, then the cut
        mid = _cut(mid, m, hi) if mid is not None else None
        mid = _cut(mid, n, c) if mid is not None else None
        out += [y for y in (left, right, mid) if y is not None]
        nc += 1
    ss[:] = out
    return nc


def _thigh_back_B(links, side, ys):
    """layout B only: thigh back wall cut under the lying roll servo (+ ties)"""
    P = lambda ln, pn: links[ln]["parts"][pn]
    # hip pitch + (thigh back wall top met the roll servo / bracket): back wall middle (roll servo, bracket side
    # plates and hip cover y-span, +-21 around the leg centre) cut down; outer 3.7 mm strips stay full height
    back = lambda x: x[1][0] == "x" and x[1][2] <= -14.0 and max(q[1] for q in x[1][3]) > 130.0
    # V-shaped top z <= 119 + |y - c| (45 deg both ways: prints upside down without support)
    n1 = _split_mid(P(f"{side}_thigh", f"{side}_thigh_frame"), back, 0, ys, ys + 21.0, (0, -1, 1), 119.0 - ys)
    n1 += _split_mid(P(f"{side}_thigh", f"{side}_thigh_frame"), back, 0, ys - 21.0, ys, (0, 1, 1), 119.0 + ys)
    # inner half: the wall above the knee-pocket notch would be an island, so also z <= 130 there
    lo_i, hi_i = (ys, ys + 21.0) if side == "R" else (ys - 21.0, ys)
    n1 += _split_mid(P(f"{side}_thigh", f"{side}_thigh_frame"), back, 0, lo_i, hi_i, (0, 0, 1), 130.0)
    # ties so nothing is left floating (printcheck pieces):
    tf = P(f"{side}_thigh", f"{side}_thigh_frame"); rgb_t = tf[0][2]
    for g in (-1, 1):       # thigh: outer strips down to z 126 + bridge into the lower back wall (|y - c| <= 17)
        tf += [PROFI("x", -18.5, -14.65, rrect(*sorted((ys + g * 20.9, ys + g * 24.7)), 126.0, 146.0, 0.0), [], rgb_t),
               PROFI("x", -18.5, -14.65, [(ys + g * 16.9, 126.0, 0), (ys + g * 21.0, 126.0, 0), (ys + g * 21.0, 140.0, 0),
                                          (ys + g * 16.9, 135.9, 0)][::g], [], rgb_t)]
    return n1


def _cover_B(links, side, ys):
    """layout B only: hip cover / bracket side plate chamfers"""
    P = lambda ln, pn: links[ln]["parts"][pn]
    # knee + at hip 0 (heel met the hip cover / bracket bottom-back): 45 deg chamfer x + z <= 97 (cover) / 99 (plates)
    cov = P(f"{side}_hip_yaw_link", f"{side}_hip_cover")
    n5 = _cut_part(cov, (-1.0, 0.0, -1.0), -97.0)          # keep x + z >= 97
    # hip pitch + (knee servo top met the cover's front-bottom edge): keep z - x >= 183 (45 deg, from x -37 at z 146)
    n5 += _cut_part(cov, (1.0, 0.0, -1.0), -183.0)
    br = P(f"{side}_hip_yaw_link", f"{side}_hip_yaw_bracket")
    n6 = _cut_part(br, (-1.0, 0.0, -1.0), -99.0, lambda x: x[1][0] == "y")
    return n5, n6


def widen_ranges(links):
    """trim the parts that first stopped a joint (sim/walk/ranges v20pre single-joint contacts)"""
    log = []
    for side in ("R", "L"):
        s = -1 if side == "R" else 1
        ys = 40.0 * s
        P = lambda ln, pn: links[ln]["parts"][pn]
        n1 = n5 = n6 = 0
        if LAYOUT == "B":
            n1 = _thigh_back_B(links, side, ys)
        # ankle pitch + (shank back wall top met the ankle roll servo): back wall above z 60 cut to z 64 in the middle
        sback = lambda x: x[1][0] == "x" and x[1][2] <= -14.0 and min(q[1] for q in x[1][3]) >= 59.9
        # (whole width: the outer strips met the ankle bracket plates; the side plates stay tied through the front walls)
        n2 = _split_mid(P(f"{side}_shank", f"{side}_shank_frame"), sback, 0, ys - 25.0, ys + 25.0, (0, 0, 1), 64.0)
        # ... and its ends trimmed to the lower wall's width (|y - leg centre| <= 22; the bracket plates start at 23)
        sh = P(f"{side}_shank", f"{side}_shank_frame")
        n2 += _cut_part(sh, (0.0, 1.0, 0.0), ys + 22.0, sback) + _cut_part(sh, (0.0, -1.0, 0.0), -(ys - 22.0), sback)
        # hip yaw (the two ankle brackets meet): inner-back vertical corner chamfered 8 x 8 (inner = toward y 0)
        yi = ys - s * 27.0                              # bracket inner face (R: -13); n = (-1, -s, 0)
        # (back plate only: cutting the inner side plate end left a 90 mm2 overhang in the +x print)
        n3 = _cut_part(P(f"{side}_ankle_link", f"{side}_ankle_bracket"), (-1.0, -s * 1.0, 0.0), 72.0 - s * yi - 8.0,
                       lambda x: x[1][0] == "x")
        # ankle roll (foot top rim met the bracket bottom): rim z 16-28 -> 16-22
        sole = P(f"{side}_foot", f"{side}_foot_sole")
        n4 = 0
        for i, x in enumerate(sole):
            a = x[1]
            if x[0] == "rprism" and a[0] == "z" and abs(a[1] - 16.0) < 1e-6 and abs(a[2] - 28.0) < 1e-6:
                sole[i] = (x[0], (a[0], a[1], 22.0, a[3], a[4], a[5]), x[2]); n4 += 1
        # ... and the horn base block (z 12-18, y +-21) whose top edges met the bracket plates: y +-16 (stays inside r 23
        # of the roll axis; the bracket plates start at r 24.3)
        blk = lambda x: x[1][0] == "z" and abs(x[1][1] - 12.0) < 1e-6 and abs(x[1][2] - 18.0) < 1e-6
        n4 += _cut_part(sole, (0.0, 1.0, 0.0), ys + 16.0, blk) + _cut_part(sole, (0.0, -1.0, 0.0), -(ys - 16.0), blk)
        if LAYOUT == "B":
            n5, n6 = _cover_B(links, side, ys)
        rgb_s = sh[0][2]
        for g in (-1, 1):       # shank: posts from the trimmed back wall up into the side plates (clear of the bracket plates)
            sh.append(PROFI("x", -18.5, -14.65, rrect(*sorted((ys + g * 18.0, ys + g * 22.5)), 60.0, 78.0, 0.0), [], rgb_s))
        # ankle bracket: the 0.3 seat moved the back plate off the side plates; strips outside the servo outline rejoin it
        ab = P(f"{side}_ankle_link", f"{side}_ankle_bracket"); rgb_a = ab[0][2]
        for y0, y1 in ((ys + s * 27.0, ys + s * 14.75), (ys - s * 14.75, ys - s * 19.0)):      # outer, inner
            ab.append(PROFI("x", -68.4, -67.9, rrect(*sorted((y0, y1)), 27.0, 50.0, 0.0), [], rgb_a))
        log.append(dict(side=side, thigh_back=n1, shank_back=n2, ankle_bracket=n3, foot_rim=n4, cover=n5, side_plates=n6))
    return log


def build_model():
    links, joints, fasteners = B19.build_model()
    LOG["v19"] = {k: B19.LOG[k] for k in ("case_holes", "wall_warnings", "pockets") if k in B19.LOG}
    LOG["hip_roll"] = [(hip_roll_A if LAYOUT == "A" else hip_roll)(links, joints, s) for s in ("R", "L")]
    LOG["renamed"] = rename_servos(links, joints)
    mp = B19._mass_parts()
    frames = X.servo_frames(links)
    LOG["seats"] = seats(links, frames, mp)
    # roll servo case holes (side plates); other seats already drilled in v19 (CASE / HORN hole sizes via env)
    LOG["case_holes"], LOG["case_holes_skipped"], LOG["wall_warnings"] = B19.case_holes(links, frames, mp)
    # the cover goes on after the screws: no holes in it
    for S in ("RL" if LAYOUT == "B" else ""):
        cv = links[f"{S}_hip_yaw_link"]["parts"][f"{S}_hip_cover"]
        for i, x in enumerate(cv):
            a = x[1]
            cv[i] = (x[0], (a[0], a[1], a[2], a[3], a[4], [h for h in a[5] if abs(h[2] - B19.CASE_HOLE_D / 2) > 1e-6]), x[2])
    LOG["case_holes"] = [h for h in LOG["case_holes"] if "cover" not in h["part"]]
    if LAYOUT == "A":
        # roll servo top-end holes (7, 10) would cut into the hub counterbores: the servo is held by its 8 side screws
        for S in "RL":
            fr = frames[f"{S}_hip_roll_servo_XM430"]
            bad = []
            for hn in (7, 10):
                p, n = X.HOLES[hn - 1]; pw, _ = X.to_world(fr, p, n); bad.append((pw[0], pw[1]))
            ss = links[f"{S}_hip_yaw_link"]["parts"][f"{S}_hip_yaw_bracket"]
            for i, x in enumerate(ss):
                a = x[1]
                if x[0] == "rprism" and a[0] == "z":
                    hs = [h for h in a[5] if not any(abs(h[0] - u) < 0.01 and abs(h[1] - v) < 0.01 for u, v in bad)]
                    ss[i] = (x[0], (a[0], a[1], a[2], a[3], a[4], hs), x[2])
        LOG["case_holes"] = [h for h in LOG["case_holes"] if not ("hip_roll_servo" in h["servo"] and h.get("hole") in (7, 10))]
        LOG["wall_warnings"] = [w for w in LOG["wall_warnings"] if not ("hip_roll_servo" in w["servo"] and w["hole"] in (7, 10))]
    entries = {f"{S}_{n}": v for S in "RL" for n, v in (("knee_servo_XM430", (2, -1)), ("ankle_pitch_servo_XC430", (2, -1)),
                                                        ("hip_pitch_servo_XM430", (2, -1)), ("hip_roll_servo_XM430", (2, -1)),
                                                        ("ankle_roll_servo_XM430", (0, +1)))}
    LOG["entry_relief"] = entry_relief(links, frames, mp, entries)
    LOG["press_fits"] = press_fits(links)
    # the leg trims below use v19 leg coordinates: in layout A the lower leg is lifted back for them
    lift = LEG_EXTRA_MM if LAYOUT == "A" else 0.0
    for S in "RL":
        for k in LOWER:
            _shift_link(links[f"{S}_{k}"], lift) if lift else None
    LOG["latch_narrow"] = narrow_latch(links)
    LOG["thin_wall_after"] = thin_wall_after(links, X.servo_frames(links))
    LOG["widen"] = widen_ranges(links)
    LOG["cable_clips"] = cable_clips(links)
    LOG["latch_gaps"] = latch_gaps(links)
    for S in "RL":
        for k in LOWER:
            _shift_link(links[f"{S}_{k}"], -lift) if lift else None
    LOG["shank_plates_wider"] = shank_plates_wider(links)
    LOG["leg_lengths"] = leg_lengths(links, joints, -LEG_EXTRA_MM if LAYOUT == "A" else 0.0)
    if "theta_max_deg" not in LOG:
        LOG["theta_max_deg"] = B18.THETA_MAX
    return links, joints, fasteners


write = B19.write
servo_step_parts = B19.servo_step_parts

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    os.makedirs(OUT, exist_ok=True)
    w = write(links, os.path.join(OUT, "biped_robot_v20.stp"))
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners, "theta_max_deg": B18.THETA_MAX,
               "case_hole_d_mm": B19.CASE_HOLE_D, "horn_hole_d_mm": B19.HORN_HOLE_D}, open(os.path.join(OUT, "model.json"), "w"), indent=1)
    json.dump(LOG, open(os.path.join(OUT, "build20_log.json"), "w"), indent=1, default=str)
    print("entities", w.n, "roll", LOG["hip_roll"], "seats", len(LOG["seats"]), "new holes", len(LOG["case_holes"]),
          "wall warnings", len(LOG["wall_warnings"]), "entry relief", len(LOG["entry_relief"]), "press fits", len(LOG["press_fits"]))
