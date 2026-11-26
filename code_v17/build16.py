"""v16 = v15 + rounded torso, per-arm shoulder servos, service access, ventilation, ear linkage.

Torso shape
  * shirt covers: vertical up to z 318, then an R20 shoulder roll to 45 deg and a 45 deg chamfer to the top (z 336)
  * lid cap continues the 45 deg chamfer (no overhanging slab), lid plug corners trimmed to the new inside
  * shorts: R8 roll to 45 deg + chamfer at the bottom hem
Arms (no crank any more)
  * one XC430-W240 inside each upper arm; its horn is bolted to a horn adapter on a fixed 8 mm steel shaft in the
    frame wall, so the servo body turns with the arm (shoulder range checked by the sweep)
  * arm shell in two halves (split at |y| 98) screwed to the servo case, white mitten hand with rubber pad pressed
    on the bottom (holds the halves), hand bottom at hip height (z 200)
  * shoulder axis z 306; servo cable leaves the arm at the hub and enters the cover gap through a slot
Service / power
  * battery LB-012 on its side (26 mm wide) in a 92 x 29 slot through the pelvis plate; battery hatch door under
    the pelvis held by four 6 x 3 mm magnets (no screws); battery change from below without opening the torso
  * lid top (under the head, behind the neck): main power rocker switch, Jetson switch, 5.5/2.1 DC jack for the
    12 V SMPS, cable pass-through for the head (nod servo + camera FFC)
  * back cover: OpenCR window (USB, buttons, LEDs) + 8 intake slots
Heat
  * Jetson Nano 2GB dev kit (100 x 80 x 29 with heatsink) + 40 x 40 x 10 mm 5 V fan on the heatsink, 7 exhaust
    slots in the lid above the fan; 5 V 3 A buck converter for the Jetson (USB-C)
Ears
  * both ears on one 4 mm steel axle in two 684ZZ bearings; a second arm on the nod crank drives a rod to a rocker on
    the axle, so the ears turn whenever the head nods (no extra servo)
"""
import os, json, math
import numpy as np
import build14 as B14
import build15 as B15
from build14 import *          # noqa
from build4 import XC, XM, PROF, CYL, RB, SERVO
from build import STEEL
from build6 import PROFI
from shapes import rrect, circle_loop
import build4, build
import loft as LF

JET = build.JET
BATT = build.BATT
SKY, NAVY, WHITE = B14.SKY, B14.NAVY, B14.WHITE

SHZ = 306.0                     # new shoulder axis height
ARM_SPLIT = 98.0                # |y| of the arm shell split plane
EAR_C = (0.0, 445.0)            # ear axle (x, z)
CRANK_S = B14.NOD_S             # nod servo horn axis (x, z)
EAR_R1, EAR_A1 = 16.0, 155.0    # ear arm on the nod crank: radius, angle (deg, in the x-z plane from +x toward +z)
EAR_R2 = 10.0                   # rocker radius on the ear axle (same angle at rest)
ROLL_Z0, ROLL_R, TOP = 318.0, 20.0, 336.0

FORCE = dict(B15.FORCE)
FORCE.update({"pelvis_hatch": "+z", "arm_R_shell_in": "-y", "arm_L_shell_in": "+y", "arm_R_shell_out": "+y",
              "arm_L_shell_out": "-y", "hand": "+z", "horn_adapter": "+y", "ear_rocker": "+y", "ear_rod": "+y",
              "nod_crank": "+y"})
NOTE = dict(B15.NOTE)
NOTE.update({
    "pelvis_hatch": "남색 또는 구조색. 윗면(립·자석 구멍)이 위로. 6x3 자석 4개 압입",
    "shell_in": "하늘색. 안쪽(어깨) 면을 바닥에. 서보 혼 쪽 케이스 나사로 서보에 체결",
    "shell_out": "하늘색. 바깥 면을 바닥에. 서보 뒤쪽 케이스 나사로 서보에 체결",
    "hand": "흰색. 손바닥(아래)을 바닥에. 고무 패드 접착, 팔 껍데기 아래에 끼움",
    "horn_adapter": "구조색. 서보 혼에 체결(서보 나사), D컷 축에 압입",
    "ear_rocker": "구조색. 귀 축 D컷에 압입",
    "ear_rod": "구조색. 평평하게",
    "lid": "흰색. 캡을 바닥에(걸쇠가 위로). 스위치·잭 구멍, 배기 슬롯",
    "torso_cover": "흰색. 세워서(어깨 라운드가 위로). 브림 권장",
    "shorts": "남색. 플랜지(윗면)를 바닥에. 아래 라운드가 위로 감",
})


# ---------------------------------------------------------------- small geometry helpers
def P3(axis, u, v, h):
    return LF.P3(axis, u, v, h)


def rr2(cu, cv, A, B, radii, n=8):
    """rounded rectangle with per-corner radii (corners +u+v, -u+v, -u-v, +u-v), CCW, fixed arc angles"""
    pts = []
    cs = [((cu + A, cv + B), 0), ((cu - A, cv + B), 90), ((cu - A, cv - B), 180), ((cu + A, cv - B), 270)]
    for ((ex, ey), a0), r in zip(cs, radii):
        r = max(r, 0.0)
        sx = -1 if ex > cu else 1
        sy = -1 if ey > cv else 1
        ox, oy = ex + sx * r, ey + sy * r
        for i in range(n + 1):
            t = math.radians(a0 + 90.0 * i / n)
            pts.append((ox + r * math.cos(t), oy + r * math.sin(t)))
    return pts


def offset_rr2(cu, cv, A, B, radii, d, n=8):
    return rr2(cu, cv, A - d, B - d, [r - d for r in radii], n)


def roll(R, T, s):
    """outer / inner inset of a shell whose surface rolls inward with radius R to 45 deg, then a 45 deg chamfer.
    s = distance from the start of the roll along the axis."""
    c45 = math.cos(math.radians(45))
    if s <= 0:
        return 0.0, T
    so = R * c45
    do = R - math.sqrt(R * R - s * s) if s <= so else R * (1 - c45) + (s - so)
    si = (R - T) * c45
    di = R - math.sqrt((R - T) ** 2 - s * s) if s <= si else (R * (1 - c45) + (s - so)) + T * math.sqrt(2)
    return do, di


def roll_levels(R, T, s_top, n=8):
    c45 = math.cos(math.radians(45))
    ss = {0.0, s_top, round(R * c45, 4), round((R - T) * c45, 4)}
    for i in range(1, n):
        ss.add(round(R * math.sin(math.radians(45.0 * i / n)), 4))
    return sorted(s for s in ss if 0 <= s <= s_top)


def clip_v(pts, vmin=None, vmax=None):
    """clip_open by the second coordinate"""
    sw = [(p[1], p[0]) for p in pts]
    out = LF.clip_open(sw, vmin, vmax)
    return [(p[1], p[0]) for p in out]


def sh_clip(poly, clip):
    """Sutherland-Hodgman: subject polygon (u,v) clipped by a convex CCW polygon"""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
    def inter(p, q, a, b):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    out = list(poly)
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(inter(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(inter(p, q, a, b))
    # drop near duplicates
    res = []
    for p in out:
        if not res or math.hypot(p[0] - res[-1][0], p[1] - res[-1][1]) > 1e-6:
            res.append(p)
    if len(res) > 1 and math.hypot(res[0][0] - res[-1][0], res[0][1] - res[-1][1]) < 1e-6:
        res.pop()
    return res


def hull(points):
    pts = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def circ(u, v, r, n=24):
    return [(u + r * math.cos(2 * math.pi * i / n), v + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def link_bar(axis, a0, a1, c1, r1, c2, r2, rgb, holes=()):
    """flat bar around two circles (u, v) in the plane normal to axis"""
    h = hull(circ(*c1, r1) + circ(*c2, r2))
    return PROFI(axis, a0, a1, [(u, v, 0) for u, v in h], [], rgb, list(holes))


def ccw(loop):
    return loop if LF.area(loop) > 0 else loop[::-1]


def translate(s, d):
    return build4.to_local(s, tuple(-x for x in d))


def bbox(s):
    return build4.aabb(build4.sampled(s))


def clamp_rect(s, axis_h, lo=None, hi=None):
    """Truncate a solid along world axis axis_h ('z'): rprism along that axis -> a0/a1, rprism across it -> clamp
    the outline coordinate (rect-like outlines only)."""
    kind, args, rgb = s
    if kind != "rprism":
        return s
    ax, a0, a1, outer, inn, holes = args
    if ax == axis_h:
        b0 = a0 if lo is None else max(a0, lo)
        b1 = a1 if hi is None else min(a1, hi)
        if b1 <= b0:
            return None
        return ("rprism", (ax, b0, b1, outer, inn, holes), rgb)
    iu, iv = LF.UVI[ax]
    k = LF.IDX[axis_h]
    ci = 0 if iu == k else 1
    def cl(p):
        q = list(p)
        if lo is not None:
            q[ci] = max(q[ci], lo)
        if hi is not None:
            q[ci] = min(q[ci], hi)
        return tuple(q)
    new = [cl(p) for p in outer]
    vals = [p[ci] for p in new]
    if max(vals) - min(vals) < 1e-6:
        return None
    return ("rprism", (ax, a0, a1, new, inn, holes), rgb)


# ---------------------------------------------------------------- torso shell
def cover_outline(d):
    return LF.rr_points(CU, 0.0, CA - d, CY - d, CR_ - d, n_arc=12)


def shoulder_lofts():
    """front (x >= -7.2) and back (x <= -7.8) shoulder bands, z 318 .. 336"""
    levels = []
    for s in roll_levels(ROLL_R, T, TOP - ROLL_Z0):
        do, di = roll(ROLL_R, T, s)
        levels.append((ROLL_Z0 + s, cover_outline(do), cover_outline(di)))
    fr = tuple((h, LF.clip_open(o, umin=CU + 0.3), LF.clip_open(i, umin=CU + 0.3)) for h, o, i in levels)
    bk = tuple((h, LF.clip_open(o, umax=CU - 0.3), LF.clip_open(i, umax=CU - 0.3)) for h, o, i in levels)
    return ("loft", ("z", fr, False), WHITE), ("loft", ("z", bk, False), WHITE)


def top_inset():
    return roll(ROLL_R, T, TOP - ROLL_Z0)          # (outer, inner) inset at z 336


def hem_loft():
    """shorts bottom roll, z 213 down to 205"""
    R, z0, zb = 8.0, 213.0, 205.0
    lv = []
    for s in roll_levels(R, T, z0 - zb):
        do, di = roll(R, T, s)
        lv.append((z0 - s, cover_outline(do), cover_outline(di)))
    lv.sort(key=lambda x: x[0])
    return ("loft", ("z", tuple(lv), True), NAVY)


def reshape_covers(T_):
    for pn, cut in (("torso_cover_front", None), ("torso_cover_back", None)):
        new = []
        for s in T_[pn]:
            s2 = clamp_rect(s, "z", hi=ROLL_Z0)
            if s2 is not None:
                new.append(s2)
        T_[pn] = new
    fl, bl = shoulder_lofts()
    T_["torso_cover_front"].append(fl)
    T_["torso_cover_back"].append(bl)
    # front cover side panels: shaft + cable slot replaces the old shaft hole
    for i, s in enumerate(T_["torso_cover_front"]):
        if s[0] == "rprism" and s[1][0] == "y" and s[1][5]:
            ax, a0, a1, outer, inn, holes = s[1]
            T_["torso_cover_front"][i] = ("rprism", (ax, a0, a1, outer, list(inn) + [rrect(SHZ - 10, SHZ + 9, -6.5, 4.0, 5.0)],
                                                     []), s[2])
    # back cover: OpenCR window + intake slots in the back panel (axis x, u = y, v = z)
    for i, s in enumerate(T_["torso_cover_back"]):
        if s[0] == "rprism" and s[1][0] == "x" and s[1][1] <= -83.0:
            ax, a0, a1, outer, inn, holes = s[1]
            wins = [rrect(-20, 20, 266, 310, 6)] + [rrect(y, y + 2.5, 243, 258, 1.2) for y in np.arange(-18.75, 18.8, 5.0)]
            T_["torso_cover_back"][i] = ("rprism", (ax, a0, a1, outer, list(inn) + wins, holes), s[2])
    # shorts hem
    new = []
    for s in T_["shorts"]:
        s2 = clamp_rect(s, "z", lo=213.0)
        if s2 is not None:
            new.append(s2)
    T_["shorts"] = new + [hem_loft()]


def new_lid(T_):
    lid = T_["lid"]
    do, di = top_inset()                       # 9.72, 11.42
    clip = ccw(cover_outline(di + 0.5))        # plug must stay inside the cover's inner surface at the top
    out = []
    for s in lid:
        if s[0] != "rprism":
            out.append(s); continue
        ax, a0, a1, outer, inn, holes = s[1]
        if ax == "z" and a0 >= 336.2:          # old flat cap -> replaced below
            continue
        if ax == "z" and a0 >= 329.9 and a1 <= 336.4 and len(outer) >= 8:     # U plug and its 0.3 slab
            poly = sh_clip([(p[0], p[1]) for p in outer], clip)
            out.append(PROFI("z", a0, a1, [(u, v, 0) for u, v in poly], [], s[2], holes))
            continue
        out.append(s)
    # cap: core + 45 deg chamfer ring continuing the shoulder chamfer
    c0 = do + 0.3                               # 10.02 at z 336.3
    core = cover_outline(12.0)
    MAINSW = (-45.0, -32.0, -40.0, -21.0)       # x0 x1 y0 y1 (KCD1-style rocker, cut 13 x 19)
    JETSW = (-44.0, -35.0, 21.0, 34.0)          # mini rocker, cut 9 x 13
    JACK = (-39.5, 45.0)                        # 5.5/2.1 panel jack, hole 11
    CABLE = (-58.0, -44.0, -8.0, 8.0)           # head cables (servo + camera FFC)
    inners = [rrect(MAINSW[0], MAINSW[1], MAINSW[2], MAINSW[3], 0.6), rrect(JETSW[0], JETSW[1], JETSW[2], JETSW[3], 0.6),
              rrect(CABLE[0], CABLE[1], CABLE[2], CABLE[3], 3.0)] + \
             [rrect(x, x + 2.2, -24.0, 24.0, 1.0) for x in np.arange(15.0, 43.0, 4.5)]
    holes = [(x, y, 1.6) for x, y in B14.YOKE_PEGS] + [(JACK[0], JACK[1], 5.6)]
    out.append(PROFI("z", 336.3, 338.3, [(u, v, 0) for u, v in core], inners, WHITE, holes))
    ring = (("z", ((336.3, cover_outline(c0), cover_outline(12.5)), (338.3, cover_outline(c0 + 2.0), cover_outline(12.5))), True))
    out.append(("loft", ring, WHITE))
    T_["lid"] = out
    # purchased parts on the lid
    T_["switch_main_rocker"] = [RB(MAINSW[0] + 0.3, MAINSW[1] - 0.3, MAINSW[2] + 0.3, MAINSW[3] - 0.3, 318.0, 336.3, ELEC, "z", 0.5),
                                RB(MAINSW[0] + 1, MAINSW[1] - 1, MAINSW[2] + 1, MAINSW[3] - 1, 338.3, 343.0, ELEC, "z", 1)]
    T_["switch_jetson_rocker"] = [RB(JETSW[0] + 0.3, JETSW[1] - 0.3, JETSW[2] + 0.3, JETSW[3] - 0.3, 322.0, 336.3, ELEC, "z", 0.5),
                                  RB(JETSW[0] + 1, JETSW[1] - 1, JETSW[2] + 1, JETSW[3] - 1, 338.3, 342.0, ELEC, "z", 1)]
    T_["dc_jack_5521"] = [CYL((JACK[0], JACK[1], 318.0), "z", 5.5, 18.3, ELEC), CYL((JACK[0], JACK[1], 338.3), "z", 7.0, 2.0, ELEC)]
    return MAINSW, JETSW, JACK, CABLE


# ---------------------------------------------------------------- pelvis / battery hatch
SLOT = (-57.0, 35.0, -14.5, 14.5)
MAGNETS = [(40.0, 11.0), (40.0, -11.0), (-61.0, 7.0), (-61.0, -7.0)]


def pelvis(fr):
    outer = fr[0][1][3]
    bear = [(0, -40, 18.5), (0, 40, 18.5)]
    slot = rrect(SLOT[0], SLOT[1], SLOT[2], SLOT[3], 3.0)
    keep_r = [(-37, 13, 23, 57), (-37, 13, -57, -23), (-66, -43, 43, 63), (-66, -43, -63, -43),
              (SLOT[0] - 3, SLOT[1] + 3, SLOT[2] - 3, SLOT[3] + 3)]
    keep_c = [(0, 40, 25.0), (0, -40, 25.0)] + [(x, y, 6.5) for x, y in MAGNETS]
    xs = B15.cells(-60.5, 46.0, 6)
    ys = B15.cells(-52.0, 52.0, 6)
    pockets = B15.grid_pockets(xs, ys, keep_r, keep_c, B15.PELVIS_BOUND)
    L1 = ("rprism", ("z", 225.0, 228.2, outer, [slot], bear + [(x, y, 3.1) for x, y in MAGNETS]), PRINT)
    L2 = ("rprism", ("z", 228.2, 235.0, outer, [slot] + pockets, bear), PRINT)
    return [L1, L2], len(pockets)


def hatch():
    out_pts = [(-65.5, -18.0), (44.5, -18.0), (44.5, -6.0), (40.5, -6.0), (40.5, 6.0), (44.5, 6.0), (44.5, 18.0), (-65.5, 18.0)]
    lp = [(u, v, 0) for u, v in out_pts]
    d = [PROFI("z", 221.0, 221.8, lp, [], PRINT),
         PROFI("z", 221.8, 225.0, lp, [], PRINT, [(x, y, 3.1) for x, y in MAGNETS])]
    # locating lips inside the slot (battery ends and sides)
    d.append(RB(SLOT[0] + 0.6, SLOT[0] + 1.6, -10, 10, 225.0, 227.5, PRINT, "z", 0.4))
    d.append(RB(SLOT[1] - 1.6, SLOT[1] - 0.6, -10, 10, 225.0, 227.5, PRINT, "z", 0.4))
    for s in (1, -1):
        y0, y1 = sorted((s * 13.4, s * 14.2))
        d.append(RB(-20, 0, y0, y1, 225.0, 227.5, PRINT, "z", 0.3))
    mags = [CYL((x, y, 225.1), "z", 3.0, 3.0, STEEL) for x, y in MAGNETS] + \
           [CYL((x, y, 221.9), "z", 3.0, 3.0, STEEL) for x, y in MAGNETS]
    return d, mags


# ---------------------------------------------------------------- arms
ARM_A, ARM_Y0, ARM_Y1 = 20.0, 77.3, 125.0
ARM_RAD = (14.0, 3.0, 3.0, 14.0)            # corners (+x,+Y) (-x,+Y) (-x,-Y) (+x,-Y) in (x, Y)... set below


def arm_outline(d, n=8):
    """(x, Y) outline of the arm section, Y = |y|; small radius on the torso side, large on the outside"""
    cv = (ARM_Y0 + ARM_Y1) / 2
    B = (ARM_Y1 - ARM_Y0) / 2
    return offset_rr2(0.0, cv, ARM_A, B, (14.0, 14.0, 3.0, 3.0), d, n)


def arm_parts(s):
    """s = -1 right, +1 left. Returns (torso-fixed parts, arm parts)."""
    nm = "R" if s < 0 else "L"
    Y = lambda y: s * y
    def yr(a, b):
        return min(Y(a), Y(b)), max(Y(a), Y(b))
    def outl(d):
        return [(x, Y(yy)) for x, yy in arm_outline(d)]
    tor, arm = {}, {}
    # torso-fixed: steel shaft (pressed in the wall boss), horn adapter, servo horn
    tor[f"shoulder_shaft_{nm}"] = [PROF("y", *yr(50.0, 82.0), circle_loop(SHZ, 0, 4.0), STEEL)]
    tor[f"arm_{nm}_horn_adapter"] = [PROFI("y", *yr(78.5, 82.0), circle_loop(SHZ, 0, 11.0), [], PRINT, [(SHZ, 0, 4.0)])]
    tor[f"arm_{nm}_horn_XC430"] = [CYL((0, Y(82.0), SHZ), "y" if s > 0 else "-y", 10.0, 3.0, XC)]
    # servo body turns with the arm: horn axis 11.25 below its top end
    arm[f"arm_{nm}_servo_XC430"] = [SERVO(-14.25, 14.25, *yr(85.0, 119.0), SHZ - 35.25, SHZ + 11.25, "y", XC)]
    # shell: vertical walls z 262..322, roll 322..330 + top plate; split at |y| 98
    zb, z0, R, Tw = 262.0, 322.0, 12.0, 1.6
    o0, i0 = outl(0.0), outl(Tw)
    walls_o, walls_i = [], []
    # torso-side face as a plate with the hub hole (x -14.2..14.2), the rest as an open loft
    face = PROFI("y", *yr(ARM_Y0, ARM_Y0 + Tw), [(z, x, 0) for z, x in ((zb, -17.0), (z0, -17.0), (z0, 17.0), (zb, 17.0))],
                 [], SKY, [(SHZ, 0, 12.0)])
    def cut_face(l):            # remove the straight torso-side face (|Y| < ARM_Y0 + small) from a closed outline
        return l
    # open outline: start at (+14.2, Y0) go CCW around the outside back to (-14.2, Y0)
    def open_poly(d, ysign):
        pts = arm_outline(d)
        # rotate so that the torso-side straight edge (min Y) is the gap
        ymin = min(p[1] for p in pts)
        n = len(pts)
        idx = [i for i in range(n) if abs(pts[i][1] - ymin) < 1e-6]
        # the straight face runs from corner (-x,-Y) end to corner (+x,-Y) start: pick the gap between them
        i_end = max(idx, key=lambda i: pts[i][0])       # +x end of the face
        seq = [pts[(i_end + k) % n] for k in range(n)]
        # keep until the -x end of the face
        j = max(k for k in range(n) if abs(seq[k][1] - ymin) < 1e-6 and seq[k][0] < 0)
        seq = seq[:j + 1]
        return [(x, ysign * yy) for x, yy in seq]
    def split(seq, part):
        """part 'in': |Y| <= split, 'out': |Y| >= split. seq in signed y. returns list of polylines"""
        res, cur = [], []
        def ok(p):
            return abs(p[1]) <= ARM_SPLIT + 1e-9 if part == "in" else abs(p[1]) >= ARM_SPLIT - 1e-9
        for k in range(len(seq)):
            p = seq[k]
            if ok(p):
                if not cur and k > 0 and not ok(seq[k - 1]):
                    q = seq[k - 1]; t = (s * ARM_SPLIT - q[1]) / (p[1] - q[1]); cur.append((q[0] + t * (p[0] - q[0]), s * ARM_SPLIT))
                cur.append(p)
            else:
                if cur:
                    q = seq[k - 1]; t = (s * ARM_SPLIT - q[1]) / (p[1] - q[1]); cur.append((q[0] + t * (p[0] - q[0]), s * ARM_SPLIT))
                    res.append(cur); cur = []
        if cur:
            res.append(cur)
        return res
    shells = {"in": [face], "out": []}
    for part in ("in", "out"):
        po, pi_ = split(open_poly(0.0, s), part), split(open_poly(Tw, s), part)
        for a, b in zip(po, pi_):
            if len(a) >= 2 and len(a) == len(b):
                shells[part].append(("loft", ("z", ((zb, a, b), (z0, a, b)), False), SKY))
            else:      # vertex counts differ after the split: rebuild inner by intersecting per-level
                pass
    # roll ring (closed) split into the two halves
    levels = []
    for sv in roll_levels(R, Tw, 8.0):
        do, di = roll(R, Tw, sv)
        levels.append((z0 + sv, outl(do), outl(di)))
    def half(l, part):
        if part == "in":
            return clip_v(l, vmin=None, vmax=s * ARM_SPLIT) if s > 0 else clip_v(l, vmin=s * ARM_SPLIT)
        return clip_v(l, vmin=s * ARM_SPLIT) if s > 0 else clip_v(l, vmax=s * ARM_SPLIT)
    for part in ("in", "out"):
        lv = tuple((h, half(o, part), half(i, part)) for h, o, i in levels)
        if all(len(a) == len(lv[0][1]) and len(b) == len(a) for _, a, b in lv):
            shells[part].append(("loft", ("z", lv, False), SKY))
        top = ccw([(p[0], p[1]) for p in outl(roll(R, Tw, 8.0)[0])])
        # top plate clipped to the half
        hp = (lambda p: abs(p[1]) <= ARM_SPLIT) if part == "in" else (lambda p: abs(p[1]) >= ARM_SPLIT)
        clipbox = ccw([(-50, s * ARM_Y0 - s * 10), (50, s * ARM_Y0 - s * 10), (50, s * ARM_SPLIT), (-50, s * ARM_SPLIT)]) \
            if part == "in" else ccw([(-50, s * ARM_SPLIT), (50, s * ARM_SPLIT), (50, s * ARM_Y1 + s * 10), (-50, s * ARM_Y1 + s * 10)])
        tp = sh_clip(top, clipbox)
        shells[part].append(PROFI("z", z0 + 8.0 - Tw, z0 + 8.0, [(u, v, 0) for u, v in tp], [], SKY))
    arm[f"arm_{nm}_shell_in"] = shells["in"]
    arm[f"arm_{nm}_shell_out"] = shells["out"]
    # hand: closed ring z 213..262, bottom roll to z 200, plug ring 262..268 inside the shell, rubber pad
    hz0, hR, hb = 213.0, 10.0, 200.0
    hand = [("loft", ("z", ((hz0, outl(0.0), outl(Tw)), (zb, outl(0.0), outl(Tw))), True), WHITE),
            ("loft", ("z", ((zb, outl(Tw + 0.2), outl(Tw + 1.8)), (zb + 6.0, outl(Tw + 0.2), outl(Tw + 1.8))), True), WHITE)]
    lv = []
    for sv in roll_levels(hR, Tw, hz0 - hb):
        do, di = roll(hR, Tw, sv)
        lv.append((hz0 - sv, outl(do), outl(di)))
    lv.sort(key=lambda x: x[0])
    hand.append(("loft", ("z", tuple(lv), True), WHITE))
    bot = ccw(outl(roll(hR, Tw, hz0 - hb)[0]))
    hand.append(PROFI("z", hb, hb + Tw, [(u, v, 0) for u, v in bot], [], WHITE))
    arm[f"arm_{nm}_hand"] = hand
    pad = ccw(outl(roll(hR, Tw, hz0 - hb)[0] + 1.5))
    arm[f"arm_{nm}_hand_pad"] = [PROFI("z", hb - 1.0, hb, [(u, v, 0) for u, v in pad], [], RUBBER)]
    return tor, arm


# ---------------------------------------------------------------- ears
def ear_geometry():
    sx, sz = CRANK_S
    a = math.radians(EAR_A1)
    P1 = (sx + EAR_R1 * math.cos(a), sz + EAR_R1 * math.sin(a))      # pin on the nod crank's ear arm (x, z)
    ex, ez = EAR_C
    P2 = (ex + EAR_R2 * math.cos(a), ez + EAR_R2 * math.sin(a))      # rocker pin
    L = math.hypot(P2[0] - P1[0], P2[1] - P1[1])
    return P1, P2, L


def teardrop(u0, v0, r, tip=(0.0, -1.0)):
    """(u, v) loop: circle r about (u0, v0) with a 45-deg point in direction tip (du, dv) = print-down"""
    t0 = math.atan2(tip[1], tip[0])
    pts = []
    for k in range(0, 19):
        a = t0 + math.radians(45 + 270 * k / 18)
        pts.append((u0 + r * math.cos(a), v0 + r * math.sin(a), 0))
    pts.append((u0 + r * math.sqrt(2) * tip[0], v0 + r * math.sqrt(2) * tip[1], 0))
    return pts


def ears(links, joints):
    H = links["head"]["parts"]
    ex, ez = EAR_C
    ear = {}
    for side, s in (("L", 1), ("R", -1)):
        sol = [x for x in H.pop(f"ear_{side}") if bbox(x)[1][0 if s > 0 else 1] * s > 99.9 * 1]   # drop hooks
        sol = [x for x in sol if min(abs(bbox(x)[1][0]), abs(bbox(x)[1][1])) >= 99.9]
        sol = [translate(x, (0, s * 0.5, 0)) for x in sol]
        cz = lambda r: [(ez + r * math.cos(2 * math.pi * k / 32), ex + r * math.sin(2 * math.pi * k / 32)) for k in range(32)]
        cone = tuple(sorted(((s * 111.0, cz(9.5), cz(2.0)), (s * 117.0, cz(9.5), cz(8.0))), key=lambda l: l[0]))
        hub = [PROF("y", *sorted((s * 100.5, s * 111.0)), circle_loop(ez, ex, 9.5), sol[0][2], [(ez, ex, 2.0)]),
               ("loft", ("y", cone, True), sol[0][2]),
               PROF("y", *sorted((s * 117.0, s * 121.3)), circle_loop(ez, ex, 9.5), sol[0][2], [(ez, ex, 8.0)])]
        ear[f"ear_{side}"] = sol + hub
        ear[f"ear_disc_{side}"] = [translate(x, (0, s * 0.5, 0)) for x in H.pop(f"ear_disc_{side}")]
        ear[f"antenna_{side}"] = [translate(x, (0, s * 0.5, 0)) for x in H.pop(f"antenna_{side}")]
    ear["ear_axle_4mm"] = [PROF("y", -121.3, 121.3, circle_loop(ez, ex, 2.0), STEEL)]
    P1, P2, L = ear_geometry()
    ear["ear_rocker"] = [link_bar("y", 17.2, 19.7, (ez, ex), 5.0, (P2[1], P2[0]), 4.0, PRINT,
                                  [(ez, ex, 2.0), (P2[1], P2[0], 1.5)])]
    ear["ear_rocker_pin"] = [PROF("y", 17.2, 23.0, circle_loop(P2[1], P2[0], 1.5), STEEL)]
    links["ears"] = {"origin": (ex, 0.0, ez), "parts": ear}
    # head: axle holes in the side walls, teardrop bosses with 684ZZ seats, bearings
    hb = H["head_back"]
    n = 0
    for i, s_ in enumerate(hb):
        if s_[0] == "rprism" and s_[1][0] == "y":
            ax, a0, a1, outer, inn, holes = s_[1]
            if abs(abs(a0) - 98.8) < 0.05 or abs(abs(a1) - 98.8) < 0.05:
                if abs(abs(a0) - 100) < 0.05 or abs(abs(a1) - 100) < 0.05:
                    hb[i] = ("rprism", (ax, a0, a1, outer, inn, list(holes) + [(ez, ex, 2.5)]), s_[2])
                    n += 1
    assert n == 2, n
    for s in (1, -1):
        hb.append(PROFI("y", *sorted((s * 94.0, s * 94.8)), teardrop(ez, ex, 7.0), [], SKY, [(ez, ex, 2.5)]))
        hb.append(PROFI("y", *sorted((s * 94.8, s * 98.8)), teardrop(ez, ex, 7.0), [], SKY, [(ez, ex, 4.5)]))
    H["ear_bearings_684"] = [PROFI("y", *sorted((s * 94.8, s * 98.8)), circle_loop(ez, ex, 4.5), [circle_loop(ez, ex, 2.0)], STEEL)
                             for s in (1, -1)]
    # head floor: cable hole behind the nod servo (servo cable + camera FFC)
    m = 0
    for i, s_ in enumerate(hb):
        if s_[0] == "rprism" and s_[1][0] == "z" and abs(s_[1][1] - 365.0) < 0.05 and abs(s_[1][2] - 366.2) < 0.05:
            ax, a0, a1, outer, inn, holes = s_[1]
            hb[i] = ("rprism", (ax, a0, a1, outer, list(inn) + [rrect(-58.0, -46.0, -8.0, 8.0, 3.0)], holes), s_[2])
            m += 1
    assert m == 1, m
    # ear arm on the nod crank + its pin; ear rod link
    sx, sz = CRANK_S
    nc = links["nod_crank"]["parts"]
    nc["nod_crank"].append(link_bar("y", 17.2, 19.7, (sz, sx), 5.0, (P1[1], P1[0]), 4.0, PRINT, [(P1[1], P1[0], 1.5)]))
    nc["nod_ear_pin"] = [PROF("y", 17.2, 23.0, circle_loop(P1[1], P1[0], 1.5), STEEL)]
    links["ear_rod"] = {"origin": (P1[0], 21.4, P1[1]), "parts": {
        "ear_rod": [link_bar("y", 19.9, 22.9, (P1[1], P1[0]), 3.5, (P2[1], P2[0]), 3.5, PRINT,
                             [(P1[1], P1[0], 1.6), (P2[1], P2[0], 1.6)])]}}
    joints.append({"name": "ear_axle", "parent": "head", "child": "ears", "axis": "y", "point_mm": (ex, 0, ez),
                   "lower_deg": -60, "upper_deg": 60, "actuator": "passive (driven by nod_crank through ear_rod)"})
    joints.append({"name": "ear_rod", "parent": "nod_crank", "child": "ear_rod", "axis": "y", "point_mm": (round(P1[0], 2), 0, round(P1[1], 2)),
                   "lower_deg": -60, "upper_deg": 60, "actuator": "passive (closed loop with ear_axle)"})


# ---------------------------------------------------------------- model
def build_model():
    B15.WALL_WINDOWS = [(-41, -31, 245, 296), (-12, 2, 245, 293), (22, 44, 277, 306), (24, 44, 242, 254),
                        (-59, -49, 278, 306), (-59, -49, 244, 254)]
    links, joints, fasteners = B14.build_model()
    B15.lighten(links)
    B15.reassign(links, joints)
    T_ = links["torso"]["parts"]
    fr = T_["torso_frame"]
    # --- remove the arm crank drive
    for k in ("arm_swing_servo_XL430", "crank_bearing_688", "shoulder_bearing_L_688", "shoulder_bearing_R_688"):
        T_.pop(k, None)
    for k in ("arm_crank", "R_arm_rod", "L_arm_rod"):
        links.pop(k, None)
    joints[:] = [j for j in joints if j["name"] not in ("arm_crank", "R_arm_rod", "L_arm_rod")]
    def is_drive(s):
        b = bbox(s)
        if s[0] == "rprism" and s[1][0] == "z" and abs(b[0][0] - 19) < 0.1 and abs(b[0][1] - 48) < 0.1 and abs(b[1][0] + 51) < 0.1:
            return True                                    # arm servo seat
        if s[0] == "rprism" and s[1][0] == "y" and abs(b[1][0] - 38.3) < 0.1 and abs(b[1][1] - 43.3) < 0.1:
            return True                                    # crank bearing block
        return False
    fr[:] = [s for s in fr if not is_drive(s)]
    # --- pelvis plate with the battery slot (replaces the v15 two layers)
    assert fr[0][1][1:3] == (225.0, 227.0) and fr[1][1][1:3] == (227.0, 235.0)
    pl, npk = pelvis(fr)
    fr[0:2] = pl
    # --- walls: fixed shoulder shaft (r4) instead of the 688ZZ seat; boss inside
    n = 0
    for i, s in enumerate(fr):
        if s[0] == "rprism" and s[1][0] == "y" and any(abs(h[2] - 8.0) < 1e-6 and abs(h[0] - 316.0) < 1e-6 for h in s[1][5]):
            ax, a0, a1, outer, inn, holes = s[1]
            fr[i] = ("rprism", (ax, a0, a1, outer, inn, [(SHZ, 0, 4.0)]), s[2])
            n += 1
    assert n == 2, n
    for s in (1, -1):
        fr.append(PROFI("y", *sorted((s * 50.0, s * 55.0)), teardrop(SHZ, 0, 7.0, (-1.0, 0.0)), [], PRINT, [(SHZ, 0, 4.0)]))
    # --- shelf without the rod slot
    sh = T_["electronics_shelf"][0]
    pins = [(-53, -50), (-53, 50), (26, 50)]
    T_["electronics_shelf"] = [PROFI("z", 272, 276, rrect(-62, 48, -54.5, 54.5, 10),
                                     [rrect(x0, x1, y0, y1, 4.0) for x0, x1, y0, y1 in B15.SHELF_WINDOWS], PRINT,
                                     [(x, y, 1.5) for x, y in pins])]
    # --- battery on its side in the slot, hatch, magnets
    T_["battery_LB012"] = [RB(-55, 33, -13, 13, 225.6, 260.6, BATT, "x", 3)]
    d, mags = hatch()
    T_["pelvis_hatch"] = d
    T_["hatch_magnets_6x3"] = mags
    # --- Jetson Nano 2GB dev kit, fan, 5 V buck
    T_.pop("jetson_orin_nano_devkit", None)
    T_["jetson_nano_2gb_devkit"] = [RB(-51, 49, -40, 40, 282, 311, JET, "z", 3)]
    T_["jetson_fan_40mm_5V"] = [RB(5, 45, -20, 20, 311, 321, ELEC, "z", 3)]
    T_["buck_5V_3A_jetson"] = T_.pop("dcdc_12V")
    # --- shell, lid, service openings
    reshape_covers(T_)
    lid_info = new_lid(T_)
    # --- arms: new links (servo inside the arm)
    for s in (-1, 1):
        nm = "R" if s < 0 else "L"
        tor, arm = arm_parts(s)
        T_.update(tor)
        links[f"{nm}_arm"] = {"origin": (0, s * 50.0, SHZ), "parts": arm}
        for j in joints:
            if j["name"] == f"{nm}_arm_swing":
                j.update(point_mm=(0, s * 50.0, SHZ), lower_deg=-120, upper_deg=120, actuator="XC430-W240 (inside the arm)")
    # --- ears + head cable hole
    ears(links, joints)
    return links, joints, fasteners


write = B14.write

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out16"
    os.makedirs(out, exist_ok=True)
    w = write(links, f"{out}/biped_robot_v16.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))
    P1, P2, L = ear_geometry()
    print("ear arm pin", np.round(P1, 2), "rocker pin", np.round(P2, 2), "rod", round(L, 2))
