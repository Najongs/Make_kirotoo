"""Biped concept model v4 — 12 servos, rounded styling.
Legs 5-DOF each (hip yaw, hip pitch, knee, ankle pitch, ankle roll), head pan + tilt, no arms.
Servos: XM430-W350 x6 (knee, ankle pitch, ankle roll), XC430-W240 x6 (hip yaw, hip pitch, head pan, head tilt).
World: X forward, Y left, Z up, mm, origin on the ground between the feet, zero pose."""
import json, math, os, itertools
from build import (fastener, write_step, BOLT, XM, XC, PRINT, DISC, BATT, JET, CAM, ELEC, STEEL, RUBBER, IDX, UVI)
import shapes
from shapes import rrect, annulus_sector, circle_loop, circle_cap, sample, area as loop_area

HA, SL, SD = 11.25, 46.5, 34.0
SHELL = (0.93, 0.93, 0.95)          # cover / head shell colour
VISOR = (0.08, 0.10, 0.14)


# ---------- solid constructors ----------
def RB(x0, x1, y0, y1, z0, z1, rgb, axis="z", r=3.0, holes=()):
    lo, hi = (x0, y0, z0), (x1, y1, z1)
    k, (iu, iv) = IDX[axis], UVI[axis]
    prof = rrect(lo[iu], hi[iu], lo[iv], hi[iv], r)
    hs = [(p[iu], p[iv], rr) for p, rr in holes]
    return ("rprism", (axis, lo[k], hi[k], prof, [], hs), rgb)


def SERVO(x0, x1, y0, y1, z0, z1, axis, rgb):
    """Servo envelope rounded R3.25 about its horn axis (ROBOTIS drawing: 4-R3.25)."""
    return RB(x0, x1, y0, y1, z0, z1, rgb, axis, 3.25)


def CYL(c, a, r, h, rgb):
    return ("cyl", (c, {"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0),
                        "z": (0, 0, 1), "-z": (0, 0, -1)}[a], r, h), rgb)


def DSC(c, a, r, h=1):
    return CYL(c, a, r, h, DISC)


def PROF(axis, a0, a1, prof, rgb, holes=()):
    return ("rprism", (axis, a0, a1, prof, [], list(holes)), rgb)


# ---------- geometry helpers for checks / rendering ----------
def sampled(s):
    """rprism -> ('prism', ...) with a sampled outline (inner loops dropped: none used in this model)."""
    if s[0] != "rprism":
        return s
    axis, a0, a1, outer, inners, holes = s[1]
    if inners:
        return ("prismh", (axis, a0, a1, sample(outer), holes, [sample(l) for l in inners]), s[2])
    return ("prism", (axis, a0, a1, sample(outer), holes), s[2])


def aabb(s):
    kind, args, _ = s
    if kind == "loft":
        import loft; return loft.aabb(args)
    if kind == "cyl":
        c, a, r, h = args
        n = math.sqrt(sum(x * x for x in a)); a = [x / n for x in a]
        out = []
        for i in range(3):
            e = c[i] + a[i] * h; w = r * math.sqrt(max(0.0, 1 - a[i] ** 2))
            out.append((min(c[i], e) - w, max(c[i], e) + w))
        return tuple(out)
    if kind == "scap":
        c, a, ro, ri, d = args
        n = math.sqrt(sum(x * x for x in a)); a = [x / n for x in a]
        rc = math.sqrt(ro * ro - d * d)
        out = []
        for i in range(3):
            w = rc * math.sqrt(max(0.0, 1 - a[i] ** 2)); pc = c[i] + d * a[i]
            hi = c[i] + ro if a[i] * ro >= d else pc + w
            lo = c[i] - ro if -a[i] * ro >= d else pc - w
            out.append((lo, hi))
        return tuple(out)
    if kind == "rprism":
        axis, a0, a1, outer, _, _ = args
        poly = sample(outer)
    else:
        axis, a0, a1, poly = args[:4]
    k, (iu, iv) = IDX[axis], UVI[axis]
    b = [None] * 3
    b[k] = (min(a0, a1), max(a0, a1))
    b[iu] = (min(p[0] for p in poly), max(p[0] for p in poly))
    b[iv] = (min(p[1] for p in poly), max(p[1] for p in poly))
    return tuple(b)


def volume(s):
    kind, args, _ = s
    if kind == "loft":
        import loft; return loft.volume(args)
    if kind == "cyl":
        return math.pi * args[2] ** 2 * args[3]
    if kind == "scap":
        c, a, ro, ri, d = args
        cap = lambda R: math.pi / 3 * (R - d) ** 2 * (2 * R + d)
        return cap(ro) - cap(ri)
    if kind == "rprism":
        axis, a0, a1, outer, inners, holes = args
        a = abs(loop_area(sample(outer))) - sum(abs(loop_area(sample(l))) for l in inners)
    else:
        axis, a0, a1, poly, holes = args
        a = abs(loop_area(poly))
    a -= sum(math.pi * r * r for _, _, r in holes)
    return a * abs(a1 - a0)


def to_local(s, o):
    kind, args, rgb = s
    if kind == "loft":
        import loft; return ("loft", loft.to_local(args, o), rgb)
    if kind == "cyl":
        c, a, r, h = args
        return ("cyl", (tuple(ci - oi for ci, oi in zip(c, o)), a, r, h), rgb)
    if kind == "scap":
        c, a, ro, ri, d = args
        return ("scap", (tuple(ci - oi for ci, oi in zip(c, o)), a, ro, ri, d), rgb)
    if kind == "rprism":
        axis, a0, a1, outer, inners, holes = args
        k, (iu, iv) = IDX[axis], UVI[axis]
        sh = lambda l: [(u - o[iu], v - o[iv], r) for u, v, r in l]
        return ("rprism", (axis, a0 - o[k], a1 - o[k], sh(outer), [sh(l) for l in inners],
                           [(u - o[iu], v - o[iv], r) for u, v, r in holes]), rgb)
    if kind == "prism":
        axis, a0, a1, poly, holes = args
        k, (iu, iv) = IDX[axis], UVI[axis]
        return ("prism", (axis, a0 - o[k], a1 - o[k], [(u - o[iu], v - o[iv]) for u, v in poly],
                          [(u - o[iu], v - o[iv], r) for u, v, r in holes]), rgb)
    raise ValueError(kind)


def mirror(s):
    kind, args, rgb = s
    if kind == "cyl":
        c, a, r, h = args
        return ("cyl", ((c[0], -c[1], c[2]), (a[0], -a[1], a[2]), r, h), rgb)
    axis, a0, a1, outer, inners, holes = args
    if axis == "y":
        return ("rprism", (axis, -a1, -a0, outer, inners, holes), rgb)
    if axis == "z":   # (u,v) = (x,y)
        f = lambda l: [(u, -v, r) for u, v, r in l][::-1]
        return ("rprism", (axis, a0, a1, f(outer), [f(l) for l in inners], [(u, -v, r) for u, v, r in holes]), rgb)
    f = lambda l: [(-u, v, r) for u, v, r in l][::-1]           # axis x: (u,v) = (y,z)
    return ("rprism", (axis, a0, a1, f(outer), [f(l) for l in inners], [(-u, v, r) for u, v, r in holes]), rgb)


# ---------- the robot ----------
def right_leg():
    ys = -40
    M3 = BOLT["M3"]["hole"]
    L = {}
    L["hip_yaw_link"] = {
        "hip_pitch_servo_XC430": [SERVO(-14.25, 14.25, ys - 17, ys + 17, 185 + HA - SL, 185 + HA, "y", XC)],
        "hip_yaw_bracket": [
            CYL((0, ys, 213), "z", 12.5, 23, PRINT),                                  # boss through the bearing
            RB(-20, 20, ys - 16, ys + 16, 201, 213, PRINT, "z", 8),                     # web
            RB(14.5, 18.5, ys - 16, ys + 16, 160, 201, PRINT, "x", 4),                  # cheeks
            RB(-18.5, -14.5, ys - 16, ys + 16, 160, 201, PRINT, "x", 4),
            DSC((0, ys - 17, 185), "-y", 10),                                          # parent disc of hip_pitch
        ],
    }
    L["thigh"] = {
        "knee_servo_XM430": [SERVO(-14.25, 14.25, ys - 17, ys + 17, 110 - HA, 110 - HA + SL, "y", XM)],
        "thigh_frame": [
            RB(-18.5, 18.5, ys - 22, ys - 18, 146, 196, PRINT, "y", 10),
            RB(-18.5, 18.5, ys + 18, ys + 22, 146, 196, PRINT, "y", 10),
            RB(14.5, 18.5, ys - 22, ys + 22, 98, 146, PRINT, "x", 4),
            RB(-18.5, -14.5, ys - 22, ys + 22, 98, 146, PRINT, "x", 4),
            DSC((0, ys - 22, 185), "-y", 8), DSC((0, ys - 17, 110), "-y", 10),
        ],
    }
    L["shank"] = {
        "ankle_pitch_servo_XM430": [SERVO(-14.25, 14.25, ys - 17, ys + 17, 35 - HA, 35 - HA + SL, "y", XM)],
        "shank_frame": [
            RB(-12, 12, ys - 22, ys - 18, 88, 122, PRINT, "y", 10),
            RB(-12, 12, ys + 18, ys + 22, 88, 122, PRINT, "y", 10),
            RB(-18.5, 18.5, ys - 22, ys - 18, 76, 88, PRINT, "y", 3),
            RB(-18.5, 18.5, ys + 18, ys + 22, 76, 88, PRINT, "y", 3),
            RB(14.5, 18.5, ys - 22, ys + 22, 23, 76, PRINT, "x", 4),
            RB(-18.5, -14.5, ys - 22, ys + 22, 23, 76, PRINT, "x", 4),
            DSC((0, ys - 22, 110), "-y", 8), DSC((0, ys - 17, 35), "-y", 10),
        ],
    }
    L["ankle_link"] = {
        "ankle_roll_servo_XM430": [SERVO(-60, -26, ys - 14.25, ys + 14.25, 35 - HA, 35 - HA + SL, "x", XM)],
        "ankle_bracket": [
            RB(-60, 12, ys - 27, ys - 23, 16, 50, PRINT, "y", 10),
            RB(-60, 12, ys + 23, ys + 27, 16, 50, PRINT, "y", 10),
            RB(-64, -60, ys - 27, ys + 27, 16, 50, PRINT, "x", 6),
            DSC((0, ys - 23, 35), "y", 8), DSC((-26, ys, 35), "x", 10),
        ],
    }
    ft = [((-40, ys - 7, 0), M3), ((-40, ys + 7, 0), M3)]
    L["foot"] = {
        "foot_sole": [RB(-60, 60, ys - 35, ys + 35, 6, 12, PRINT, "z", 18, ft)] +
                     [CYL((x, y, 0), "z", 6, 6, RUBBER) for x in (-48, 48) for y in (ys - 23, ys + 23)],
        "foot_bracket": [RB(-25, -22.5, ys - 14, ys + 14, 12, 47, PRINT, "x", 4),
                         RB(-50, -25, ys - 14, ys + 14, 12, 18, PRINT, "z", 4, ft),
                         DSC((-22.5, ys, 35), "x", 8)],
    }
    F = [("foot", "M3", (-40, y, 18), "-z", 16, 12) for y in (ys - 7, ys + 7)]
    J = [
        ("hip_yaw", "torso", "hip_yaw_link", "z", (0, ys, 213), -30, 30, "XC430-W240"),
        ("hip_pitch", "hip_yaw_link", "thigh", "y", (0, ys, 185), -90, 30, "XC430-W240"),
        ("knee", "thigh", "shank", "y", (0, ys, 110), 0, 130, "XM430-W350"),
        ("ankle_pitch", "shank", "ankle_link", "y", (0, ys, 35), -60, 45, "XM430-W350"),
        ("ankle_roll", "ankle_link", "foot", "x", (0, ys, 35), -30, 30, "XM430-W350"),
    ]
    return L, F, J


COV_Y, COV_XF, COV_XB, COV_R = 75, 70, -85, 50


def cover_half(front):
    t, r, Y = 1.2, COV_R, COV_Y
    if front:
        x0, x1 = -7.5, COV_XF
        pts = [(x0, -Y, 0), (x1, -Y, r), (x1, Y, r), (x0, Y, 0),
               (x0, Y - t, 0), (x1 - t, Y - t, r - t), (x1 - t, -Y + t, r - t), (x0, -Y + t, 0)]
    else:
        x0, x1 = -8.5, COV_XB
        pts = [(x0, Y, 0), (x1, Y, r), (x1, -Y, r), (x0, -Y, 0),
               (x0, -Y + t, 0), (x1 + t, -Y + t, r - t), (x1 + t, Y - t, r - t), (x0, Y - t, 0)]
    return PROF("z", 225, 336, pts, SHELL)


def torso():
    M4, M3 = BOLT["M4"]["hole"], BOLT["M3"]["hole"]
    bolt_pts = [(x, y) for x in (-55, 40) for y in (-50, 50)]
    hz4 = [((x, y, 0), M4) for x, y in bolt_pts]
    shelf_pts = [(x, y) for x in (-53, 29) for y in (-50, 50)]
    hz3 = [((x, y, 0), M3) for x, y in shelf_pts]

    def side(ys, tab_y):
        def holes_in(x0, x1, pts):
            return [h for h in pts if x0 < h[0][0] < x1 and tab_y[0] < h[0][1] < tab_y[1]]
        tabs = [RB(x0, x0 + 20, tab_y[0], tab_y[1], z0, z1, PRINT, "z", 3, holes_in(x0, x0 + 20, hz4))
                for x0 in (-65, 30) for z0, z1 in ((235, 241), (324, 330))]
        ledges = [RB(x0, x0 + 18, tab_y[0], tab_y[1], 268, 272, PRINT, "z", 3, holes_in(x0, x0 + 18, hz3))
                  for x0 in (-62, 20)]
        return [RB(-65, 50, ys[0], ys[1], 235, 330, PRINT, "y", 12)] + tabs + ledges

    P = {
        "yaw_servo_R_XC430": [SERVO(HA - SL, HA, -54.25, -25.75, 236, 236 + SD, "z", XC)],
        "yaw_servo_L_XC430": [SERVO(HA - SL, HA, 25.75, 54.25, 236, 236 + SD, "z", XC)],
        "pelvis_plate": [RB(-65, 50, -62, 62, 225, 235, PRINT, "z", 20,
                            hz4 + [((0, -40, 0), 18.5), ((0, 40, 0), 18.5)])],
        "yaw_bearing_R_6805": [PROF("z", 226, 233, circle_loop(0, -40, 18.5), STEEL, [(0, -40, 12.5)])],
        "yaw_bearing_L_6805": [PROF("z", 226, 233, circle_loop(0, 40, 18.5), STEEL, [(0, 40, 12.5)])],
        "torso_side_R": side((-60, -55), (-55, -45)),
        "torso_side_L": side((55, 60), (45, 55)),
        "electronics_shelf": [RB(-62, 48, -54.5, 54.5, 272, 276, PRINT, "z", 10, hz3)],
        "top_plate": [RB(-65, 50, -60, 60, 330, 335, SHELL, "z", 20, hz4)],
        "battery_3S_2200": [RB(-60, 45, -17, 17, 237, 262, BATT, "x", 5)],
        "jetson_orin_nano_devkit": [RB(-50, 48, -40, 40, 282, 312, JET, "z", 6)],
        "imu_bno085": [RB(-10, 10, -10, 10, 276, 280, ELEC, "z", 2)],
        "u2d2_A": [RB(-40, 8, 42, 52, 276, 296, ELEC, "y", 3)],
        "u2d2_B": [RB(-40, 8, -52, -42, 276, 296, ELEC, "y", 3)],
        "dcdc_12V": [RB(12, 42, 42, 52, 280, 305, ELEC, "y", 3)],
        "head_pan_servo_XC430": [SERVO(HA - SL, HA, -14.25, 14.25, 335, 335 + SD, "z", XC)],
        "torso_cover_front": [cover_half(True)],
        "torso_cover_back": [cover_half(False)],
        "joint_discs_torso": [DSC((0, 0, 369), "z", 10)],              # parent disc of head_pan
    }
    F = [("torso", "M4", (x, y, 241), "-z", 20, 16) for x, y in bolt_pts] + \
        [("torso", "M4", (x, y, 335), "-z", 16, 11) for x, y in bolt_pts] + \
        [("torso", "M3", (x, y, 276), "-z", 12, 8) for x, y in shelf_pts]
    return P, F


ZT = 392.0      # head tilt axis height


def neck():
    """Pan link: bracket on the pan horn carrying the tilt servo (horn +Y, body along X)."""
    return {
        "neck_bracket": [RB(-30, 8, -16, 16, 371, ZT - 14.25, PRINT, "z", 6),
                         DSC((0, 0, 370), "z", 8)],                            # child disc of head_pan
        "head_tilt_servo_XC430": [SERVO(HA - SL, HA, -17, 17, ZT - 14.25, ZT + 14.25, "y", XC)],
        "joint_disc_neck": [DSC((0, 17, ZT), "y", 10)],                       # parent disc of head_tilt
    }


HEAD_RO, HEAD_RI, HEAD_CUT = 45.0, 42.5, -6.0     # helmet sphere about the tilt axis, open below z = ZT - 6
VIS_TILT = 20.0                                    # visor / camera look-down angle


def head():
    """Tilt link: spherical helmet centred on the tilt axis (so tilt and pan never sweep it outward),
    dark visor cap facing forward-down, camera board + lens looking through the visor."""
    th = math.radians(VIS_TILT)
    dv = (math.cos(th), 0.0, -math.sin(th))
    cx, cz, t, h = 29, ZT, 10, 25
    cam = []
    for dx, dz in ((-t / 2, -h / 2), (t / 2, -h / 2), (t / 2, h / 2), (-t / 2, h / 2)):
        x = cx + dx * math.cos(th) + dz * math.sin(th)
        z = cz - dx * math.sin(th) + dz * math.cos(th)
        cam.append((z, x, 0))
    # lens: from the camera front face out to just past the helmet surface, along the view direction
    c0 = (cx + t / 2 * dv[0], 0.0, cz + t / 2 * dv[2])
    rel = (c0[0], 0.0, c0[2] - ZT)
    b = sum(p * q for p, q in zip(rel, dv)); cc = sum(p * p for p in rel)
    s_out = -b + math.sqrt(b * b - cc + (HEAD_RO + 1.0) ** 2)
    return {
        "head_shell": [("scap", ((0, 0, ZT), (0, 0, 1), HEAD_RO, HEAD_RI, HEAD_CUT), SHELL),
                       DSC((0, 18, ZT), "y", 8)],                              # child disc of head_tilt
        "head_visor": [("scap", ((0, 0, ZT), (1, 0, 0), HEAD_RO + 1.0, HEAD_RO, 35.0), VISOR)],   # round face plate, horizontal,
        "camera_mount": [RB(16, 19, -19, 19, 384, 406, PRINT, "x", 3)],
        "head_camera_small": [PROF("y", -13, 13, cam, CAM)],
        "camera_lens": [("cyl", (c0, dv, 6, s_out), VISOR)],
    }


def build_model():
    links = {"torso": {"origin": (0, 0, 0), "parts": {}}}
    tp, tf = torso()
    links["torso"]["parts"].update(tp)
    fasteners = [dict(link="torso", size=s, p=p, d=d, length=ln, grip=g) for _, s, p, d, ln, g in tf]
    RL, RF, RJ = right_leg()
    origin_of = {child: p for _, _, child, _, p, _, _, _ in RJ}
    joints = []
    for side in ("R", "L"):
        for link, parts in RL.items():
            o = origin_of[link]
            if side == "L":
                parts = {k: [mirror(s) for s in v] for k, v in parts.items()}
                o = (o[0], -o[1], o[2])
            links[f"{side}_{link}"] = {"origin": o, "parts": {f"{side}_{k}": v for k, v in parts.items()}}
        for l, s, p, d, ln, g in RF:
            if side == "L":
                p = (p[0], -p[1], p[2])
            fasteners.append(dict(link=f"{side}_{l}", size=s, p=p, d=d, length=ln, grip=g))
        for name, parent, child, ax, p, lo, hi, act in RJ:
            pp = p if side == "R" else (p[0], -p[1], p[2])
            joints.append({"name": f"{side}_{name}", "parent": "torso" if parent == "torso" else f"{side}_{parent}",
                           "child": f"{side}_{child}", "axis": ax, "point_mm": pp, "lower_deg": lo, "upper_deg": hi,
                           "actuator": act})
    links["neck"] = {"origin": (0, 0, 369), "parts": neck()}
    links["head"] = {"origin": (0, 0, ZT), "parts": head()}
    joints += [
        {"name": "head_pan", "parent": "torso", "child": "neck", "axis": "z", "point_mm": (0, 0, 369),
         "lower_deg": -90, "upper_deg": 90, "actuator": "XC430-W240"},
        {"name": "head_tilt", "parent": "neck", "child": "head", "axis": "y", "point_mm": (0, 0, ZT),
         "lower_deg": -20, "upper_deg": 20, "actuator": "XC430-W240"},
    ]
    count = {}
    for f in fasteners:
        bolt, nut = fastener(f["size"], f["p"], f["d"], f["length"], f["grip"])
        key = (f["link"], f["size"], f["length"])
        count[key] = count.get(key, 0) + 1
        n = count[key]
        links[f["link"]]["parts"][f"{f['link']}_bolt_{f['size']}x{f['length']}_{n}"] = bolt
        links[f["link"]]["parts"][f"{f['link']}_nut_{f['size']}_{n}"] = nut
    return links, joints, fasteners


def _crossings(A, B, C, D, same):
    """proper crossings between the segments A[i]->B[i] and C[j]->D[j] (same polyline: neighbours left out)"""
    out = []
    for i in range(len(A)):
        p, r = A[i], B[i] - A[i]
        s = D - C
        den = r[0] * s[:, 1] - r[1] * s[:, 0]
        ok = abs(den) > 1e-12
        den = [d if o else 1.0 for d, o in zip(den, ok)]
        qp = C - p
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / den
        u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / den
        for j in range(len(C)):
            if same and (j - i) % len(A) in (0, 1, len(A) - 1):
                continue
            if ok[j] and 1e-9 < t[j] < 1 - 1e-9 and 1e-9 < u[j] < 1 - 1e-9:
                out.append((round(float(p[0] + t[j] * r[0]), 2), round(float(p[1] + t[j] * r[1]), 2)))
    return out


def check_profile(s):
    """an rprism as OCC will read it from the STEP: holes fully outside the outline (or inside an inner loop) are
    dropped (they cut nothing; written, OCC makes each a loose cylinder solid that can fill a hole of the plate next
    to it: 60 leg case holes, 2026-11-28); returns (solid, problems) where problems are the faults that need a design
    fix: a hole circle crossing a loop, a loop crossing itself (overlapping fillets too), an inner loop crossing or
    outside the outline (2026-11-28: antenna fillets, lid plug, shorts flange hole)"""
    import numpy as np
    from matplotlib.path import Path
    if s[0] != "rprism":
        return s, []
    axis, a0, a1, outer, inners, holes = s[1]
    seg = lambda L: (np.asarray(L, float)[:, :2], np.roll(np.asarray(L, float)[:, :2], -1, axis=0))
    def dist(p, A, B):
        d = B - A
        t = np.clip(((p - A) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
        return float(np.min(np.hypot(*(A + t[:, None] * d - p).T)))
    probs = []
    O = sample(outer); AO, BO = seg(O); PO = Path(AO)
    I = [sample(l) for l in inners]; SI = [seg(l) for l in I]; PI = [Path(x[0]) for x in SI]
    for nm, (A, B) in [("outline", (AO, BO))] + [(f"inner loop {k}", x) for k, x in enumerate(SI)]:
        x = _crossings(A, B, A, B, True)
        if x:
            probs.append(f"{nm} crosses itself at {x[:2]}")
    for k, (A, B) in enumerate(SI):
        if _crossings(A, B, AO, BO, False):
            probs.append(f"inner loop {k} crosses the outline")
        elif not PO.contains_point(A[0]):
            probs.append(f"inner loop {k} lies outside the outline")
    keep = []
    for h in holes:
        p = np.array(h[:2], float); r = h[2]
        if dist(p, AO, BO) < r - 1e-6:
            probs.append(f"hole {tuple(round(v, 2) for v in h)} crosses the outline"); keep.append(h); continue
        if not PO.contains_point(p):
            continue                                   # fully outside: cuts nothing
        cut = [dist(p, A, B) < r - 1e-6 for A, B in SI]
        if any(cut):
            probs.append(f"hole {tuple(round(v, 2) for v in h)} crosses an inner loop"); keep.append(h); continue
        if any(P.contains_point(p) for P in PI):
            continue                                   # fully inside an opening: cuts nothing
        keep.append(h)
    return ("rprism", (axis, a0, a1, outer, inners, keep), s[2]), probs


def write(links, path, check=True):
    """write_step through to_local; check: every rprism through check_profile (holes that cut nothing dropped, any
    remaining fault raises ValueError naming the part, so a broken STEP is not written)"""
    import build
    if check:
        bad = []
        links = {ln: dict(L, parts={pn: list(ss) for pn, ss in L["parts"].items()}) for ln, L in links.items()}
        for ln, L in links.items():
            for pn, ss in L["parts"].items():
                for i, x in enumerate(ss):
                    ss[i], probs = check_profile(x)
                    bad += [f"{pn} solid {i}: {m}" for m in probs]
        if bad:
            raise ValueError(f"{path}: profile faults (fix the design, OCC misreads these):\n  " + "\n  ".join(bad))
    saved = build.to_local
    build.to_local = to_local          # write_step localises solids through build.to_local
    try:
        return write_step(links, path)
    finally:
        build.to_local = saved


if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out4"
    os.makedirs(out, exist_ok=True)
    w = write(links, f"{out}/biped_robot_v4.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))
