"""Biped concept model v2 (bolted printed joints).
World frame: X forward, Y left, Z up, mm, origin on the ground between the feet, zero pose."""
import json, itertools, math, os
from stepwriter import StepWriter

XM = (0.22, 0.22, 0.25); XC = (0.55, 0.56, 0.60); PRINT = (0.95, 0.55, 0.15)
DISC = (0.85, 0.12, 0.12); BATT = (0.20, 0.40, 0.85); JET = (0.15, 0.55, 0.30)
CAM = (0.12, 0.12, 0.12); ELEC = (0.55, 0.25, 0.60); STEEL = (0.78, 0.80, 0.83)
RUBBER = (0.10, 0.10, 0.10)

AX = {"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}
IDX = {"x": 0, "y": 1, "z": 2}
UVI = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}     # world indices of (u, v) for each extrusion axis

BOLT = {  # shank r, head dia, head h, nut AF, nut t, clearance hole r
    "M3": dict(r=1.5, hd=5.5, hh=3.0, af=5.5, nt=2.4, hole=1.7),
    "M4": dict(r=2.0, hd=7.0, hh=4.0, af=7.0, nt=3.2, hole=2.25),
}


def BH(x0, x1, y0, y1, z0, z1, rgb, axis="z", holes=()):
    """Box extruded along `axis`, optional through-holes given as (world point, radius)."""
    lo, hi = (x0, y0, z0), (x1, y1, z1)
    k, (iu, iv) = IDX[axis], UVI[axis]
    poly = [(lo[iu], lo[iv]), (hi[iu], lo[iv]), (hi[iu], hi[iv]), (lo[iu], hi[iv])]
    hs = [(p[iu], p[iv], r) for p, r in holes]
    for u, v, r in hs:   # every hole must sit fully inside the face, with at least 1 mm of wall
        assert lo[iu] + r + 1 <= u <= hi[iu] - r - 1 and lo[iv] + r + 1 <= v <= hi[iv] - r - 1, (x0, y0, z0, u, v)
    return ("prism", (axis, lo[k], hi[k], poly, hs), rgb)


def D(c, a, r, h, rgb=DISC):
    return ("cyl", (c, AX[a], r, h), rgb)


def fastener(size, p, d, length, grip):
    """Bolt head bears on point p (outside face), shank runs along direction d ('+y', '-z', ...)."""
    s = BOLT[size]
    axis, sgn = d[-1], (-1 if d[0] == "-" else 1)
    k = IDX[axis]
    a = AX[axis] if sgn > 0 else AX["-" + axis]
    back = "-" + axis if sgn > 0 else axis
    bolt = [D(p, back, s["hd"] / 2, s["hh"], STEEL), D(p, d.lstrip("+"), s["r"], length, STEEL)]
    iu, iv = UVI[axis]
    R = s["af"] / math.sqrt(3)
    hexa = [(p[iu] + R * math.cos(i * math.pi / 3), p[iv] + R * math.sin(i * math.pi / 3)) for i in range(6)]
    n0 = p[k] + sgn * grip
    nut = [("prism", (axis, n0, n0 + sgn * s["nt"], hexa, [(p[iu], p[iv], s["r"])]), STEEL)]
    return bolt, nut


def right_leg():
    L = {}
    M3 = BOLT["M3"]["hole"]
    L["hip_yaw_link"] = {
        "hip_roll_servo_XM430": [BH(-53, -19, -54.25, -25.75, 173, 219.5, XM)],
        "hip_yaw_bracket": [BH(-53, 14, -56, -24, 219.5, 239, PRINT),
                            D((0, -40, 239), "z", 8, 1), D((-19, -40, 185), "x", 10, 1)],
    }
    L["hip_roll_link"] = {
        "hip_pitch_servo_XC430": [BH(-14.25, 14.25, -57, -23, 150.5, 197, XC)],
        "hip_roll_bracket": [BH(-17, -14.5, -56, -24, 150.5, 215, PRINT),
                             D((-18, -40, 185), "x", 8, 1), D((0, -57, 185), "-y", 10, 1)],
    }
    th = [((21, 0, 144), M3), ((21, 0, 152), M3)]
    L["thigh"] = {
        "knee_servo_XM430": [BH(-14.25, 14.25, -57, -23, 98, 144.5, XM)],
        "thigh_plate_outer": [BH(-12, 12, -62, -59, 98, 197, PRINT), BH(12, 30, -62, -59, 132, 164, PRINT, "y", th),
                              D((0, -58, 185), "-y", 8, 1), D((0, -57, 110), "-y", 10, 1)],
        "thigh_plate_inner": [BH(-12, 12, -22, -19, 98, 164, PRINT), BH(12, 30, -22, -19, 132, 164, PRINT, "y", th)],
        "thigh_spacer": [BH(15, 27, -59, -22, 136, 160, PRINT, "y", th)],
    }
    sh = [((21, 0, 76), M3), ((21, 0, 88), M3)]
    L["shank"] = {
        "ankle_pitch_servo_XM430": [BH(-14.25, 14.25, -57, -23, 23, 69.5, XM)],
        "shank_plate_outer": [BH(-12, 12, -67, -64, 23, 122, PRINT), BH(12, 30, -67, -64, 64, 100, PRINT, "y", sh),
                              D((0, -64, 110), "y", 8, 1), D((0, -57, 35), "-y", 10, 1)],
        "shank_plate_inner": [BH(-12, 12, -18, -15, 23, 122, PRINT), BH(12, 30, -18, -15, 64, 100, PRINT, "y", sh)],
        "shank_spacer": [BH(15, 27, -64, -18, 68, 96, PRINT, "y", sh)],
    }
    L["ankle_link"] = {
        "ankle_roll_servo_XC430": [BH(-53, -19, -54.25, -25.75, 23, 69.5, XC)],
        "ankle_bracket": [BH(-53, 14, -72, -69, 12, 50, PRINT), BH(-57, -53, -72, -20, 12, 50, PRINT),
                          D((0, -69, 35), "y", 8, 1), D((-19, -40, 35), "x", 10, 1)],
    }
    ft = [((-31, -47, 0), M3), ((-31, -33, 0), M3)]
    L["foot"] = {
        "foot_sole": [BH(-60, 60, -75, -5, 4, 9, PRINT, "z", ft)] +
                     [BH(x0, x0 + 12, y0, y0 + 12, 0, 4, RUBBER) for x0 in (-58, 46) for y0 in (-73, -19)],
        "foot_bracket": [BH(-17, -14.5, -54, -26, 9, 47, PRINT), BH(-40, -17, -54, -26, 9, 13, PRINT, "z", ft),
                         D((-18, -40, 35), "x", 8, 1)],
    }
    # fasteners: (link, size, head point, direction, length, grip)
    F = [("thigh", "M3", (21, -62, z), "+y", 50, 43) for z in (144, 152)] + \
        [("shank", "M3", (21, -67, z), "+y", 60, 52) for z in (76, 88)] + \
        [("foot", "M3", (-31, y, 13), "-z", 12, 9) for y in (-47, -33)]
    J = [
        ("hip_yaw", "torso", "hip_yaw_link", "z", (0, -40, 240), -30, 30),
        ("hip_roll", "hip_yaw_link", "hip_roll_link", "x", (0, -40, 185), -35, 20),
        ("hip_pitch", "hip_roll_link", "thigh", "y", (0, -40, 185), -90, 30),
        ("knee", "thigh", "shank", "y", (0, -40, 110), 0, 130),
        ("ankle_pitch", "shank", "ankle_link", "y", (0, -40, 35), -60, 45),
        ("ankle_roll", "ankle_link", "foot", "x", (0, -40, 35), -30, 30),
    ]
    return L, F, J


def torso():
    M4 = BOLT["M4"]["hole"]
    pts = [(x, y) for x in (-45, 35) for y in (-50, 50)]
    hz = [((x, y, 0), M4) for x, y in pts]

    def side(ys, tab_y):
        """Side plate with four inward tabs (bottom and top) that take the M4 bolts."""
        def inside(x0):
            return [h for h in hz if x0 < h[0][0] < x0 + 20 and tab_y[0] < h[0][1] < tab_y[1]]
        tabs = [BH(x0, x0 + 20, tab_y[0], tab_y[1], z0, z1, PRINT, "z", inside(x0))
                for x0 in (-55, 25) for z0, z1 in ((281, 285), (361, 365))]
        return [BH(-55, 45, ys[0], ys[1], 281, 365, PRINT)] + tabs

    P = {
        "hip_yaw_servo_R_XC430": [BH(-34.5, 12, -54.25, -25.75, 241, 275, XC)],
        "hip_yaw_servo_L_XC430": [BH(-34.5, 12, 25.75, 54.25, 241, 275, XC)],
        "pelvis_plate": [BH(-60, 45, -60, 60, 275, 281, PRINT, "z", hz),
                         D((0, -40, 241), "-z", 10, 1), D((0, 40, 241), "-z", 10, 1)],
        "torso_side_R": side((-60, -55), (-55, -45)),
        "torso_side_L": side((55, 60), (45, 55)),
        "top_plate": [BH(-55, 45, -60, 60, 365, 370, PRINT, "z", hz)],
        "battery_3S_2200": [BH(-55, 50, -17, 17, 287, 312, BATT)],
        "jetson_orin_nano_devkit": [BH(-55, 45, -40, 40, 318, 348, JET)],
        "camera": [BH(46, 71, -45, 45, 335, 360, CAM)],
        "imu_bno085": [BH(-10, 10, -10, 10, 281, 285, ELEC)],
        "u2d2_A": [BH(10, 48, -45, -25, 287, 297, ELEC)],
        "u2d2_B": [BH(10, 48, 25, 45, 287, 297, ELEC)],
        "dcdc_12V": [BH(-30, 0, 22, 46, 287, 302, ELEC)],
    }
    F = [("torso", "M4", (x, y, 285), "-z", 16, 10) for x, y in pts] + \
        [("torso", "M4", (x, y, 370), "-z", 16, 9) for x, y in pts]
    return P, F


def mirror_solid(s):
    kind, args, rgb = s
    if kind == "cyl":
        c, a, r, h = args
        return ("cyl", ((c[0], -c[1], c[2]), (a[0], -a[1], a[2]), r, h), rgb)
    axis, a0, a1, poly, holes = args
    if axis == "y":
        return ("prism", (axis, -a1, -a0, poly, holes), rgb)
    if axis == "z":   # (u,v) = (x,y): flip v
        return ("prism", (axis, a0, a1, [(u, -v) for u, v in poly][::-1], [(u, -v, r) for u, v, r in holes]), rgb)
    return ("prism", (axis, a0, a1, [(-u, v) for u, v in poly][::-1], [(-u, v, r) for u, v, r in holes]), rgb)


def build_model():
    links = {"torso": {"origin": (0, 0, 0), "parts": {}}}
    tp, tf = torso()
    links["torso"]["parts"].update(tp)
    RL, RF, RJ = right_leg()
    origin_of = {child: p for _, _, child, _, p, _, _ in RJ}
    fasteners = [dict(link=l, size=s, p=p, d=d, length=ln, grip=g) for l, s, p, d, ln, g in tf]
    joints = []
    for side in ("R", "L"):
        for link, parts in RL.items():
            o = origin_of[link]
            if side == "L":
                parts = {k: [mirror_solid(s) for s in v] for k, v in parts.items()}
                o = (o[0], -o[1], o[2])
            links[f"{side}_{link}"] = {"origin": o, "parts": {f"{side}_{k}": v for k, v in parts.items()}}
        for l, s, p, d, ln, g in RF:
            if side == "L":
                p = (p[0], -p[1], p[2])
                if d[-1] == "y":
                    d = ("-y" if d[0] == "+" else "+y")
            fasteners.append(dict(link=f"{side}_{l}", size=s, p=p, d=d, length=ln, grip=g))
        for name, parent, child, ax, p, lo, hi in RJ:
            pp = p if side == "R" else (p[0], -p[1], p[2])
            if side == "L" and name == "hip_roll":   # +X rotation swings the foot toward +Y
                lo, hi = -hi, -lo
            joints.append({"name": f"{side}_{name}", "parent": "torso" if parent == "torso" else f"{side}_{parent}",
                           "child": f"{side}_{child}", "axis": ax, "point_mm": pp, "lower_deg": lo, "upper_deg": hi})
    count = {}
    for f in fasteners:
        bolt, nut = fastener(f["size"], f["p"], f["d"], f["length"], f["grip"])
        key = (f["link"], f["size"], f["length"])
        count[key] = count.get(key, 0) + 1
        n = count[key]
        links[f["link"]]["parts"][f"{f['link']}_bolt_{f['size']}x{f['length']}_{n}"] = bolt
        links[f["link"]]["parts"][f"{f['link']}_nut_{f['size']}_{n}"] = nut
    return links, joints, fasteners


def to_local(s, o):
    kind, args, rgb = s
    if kind == "cyl":
        c, a, r, h = args
        return ("cyl", (tuple(ci - oi for ci, oi in zip(c, o)), a, r, h), rgb)
    axis, a0, a1, poly, holes = args
    k, (iu, iv) = IDX[axis], UVI[axis]
    return ("prism", (axis, a0 - o[k], a1 - o[k], [(u - o[iu], v - o[iv]) for u, v in poly],
                      [(u - o[iu], v - o[iv], r) for u, v, r in holes]), rgb)


def aabb(s):
    kind, args, _ = s
    if kind == "cyl":
        c, a, r, h = args
        lo, hi = [], []
        for i in range(3):
            e = c[i] + a[i] * h
            if abs(a[i]) > 0.5:
                lo.append(min(c[i], e)); hi.append(max(c[i], e))
            else:
                lo.append(c[i] - r); hi.append(c[i] + r)
        return tuple(zip(lo, hi))
    axis, a0, a1, poly, holes = args
    k, (iu, iv) = IDX[axis], UVI[axis]
    b = [None] * 3
    b[k] = (min(a0, a1), max(a0, a1))
    b[iu] = (min(p[0] for p in poly), max(p[0] for p in poly))
    b[iv] = (min(p[1] for p in poly), max(p[1] for p in poly))
    return tuple(b)


def volume(s):
    kind, args, _ = s
    if kind == "cyl":
        return math.pi * args[2] ** 2 * args[3]
    axis, a0, a1, poly, holes = args
    n = len(poly)
    area = abs(sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))) / 2
    area -= sum(math.pi * r * r for _, _, r in holes)
    return area * abs(a1 - a0)


def write_step(links, path):
    w = StepWriter(os.path.basename(path))
    subs = []
    for lname, L in links.items():
        o = L["origin"]
        parts = [(w.part(pn, [to_local(s, o) for s in solids]), (0, 0, 0)) for pn, solids in L["parts"].items()]
        subs.append((w.assembly(lname, parts), o))
    w.assembly("biped_robot", subs)
    open(path, "w").write(w.text())
    return w


def check_interference(links, tol=0.01):
    items = [(ln, pn, s) for ln, L in links.items() for pn, sol in L["parts"].items() for s in sol]
    bad = []
    for (l1, p1, s1), (l2, p2, s2) in itertools.combinations(items, 2):
        if l1 == l2:
            continue
        b1, b2 = aabb(s1), aabb(s2)
        ov = [min(b1[i][1], b2[i][1]) - max(b1[i][0], b2[i][0]) for i in range(3)]
        if all(v > tol for v in ov):
            bad.append((l1, p1, l2, p2, [round(v, 2) for v in ov]))
    return bad


if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out"
    os.makedirs(out, exist_ok=True)
    w = write_step(links, f"{out}/biped_robot.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))
    bad = check_interference(links)
    print("interferences between links:", len(bad))
    for b in bad:
        print(" ", b)
