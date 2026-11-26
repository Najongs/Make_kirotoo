"""Biped concept model v3 — 12 servos.
Legs 4-DOF (hip pitch, knee, ankle pitch, ankle roll) x2, one coupled hip-yaw servo (right yaw axis driven,
left yaw axis linked by a diagonal tie rod -> opposite rotation), shoulder pitch x2, head pan x1.
World: X forward, Y left, Z up, mm, origin on the ground between the feet, zero pose.
XC430/XM430 envelope 28.5 x 46.5 x 34 mm, horn axis 11.25 mm from the end face (ROBOTIS drawing)."""
import json, math, os
from build import (BH, D, fastener, mirror_solid, to_local, aabb, volume, write_step, check_interference,
                   BOLT, XM, XC, PRINT, DISC, BATT, JET, CAM, ELEC, STEEL, RUBBER, IDX, UVI)

HA = 11.25          # horn axis offset from servo end face
SL, SW, SD = 46.5, 28.5, 34.0


# hip-yaw coupling: levers of length LR perpendicular to the tie rod in the zero pose -> 1:1 opposite rotation
LR = 28.0
TIP_Y = (LR * LR - 1600) / 40                     # right lever tip (relative to world), yaw axes at y = -/+40
TIP_X = math.sqrt(-TIP_Y * TIP_Y - 40 * TIP_Y)
TIP_R, TIP_L = (TIP_X, TIP_Y), (-TIP_X, -TIP_Y)


def bar(p0, p1, w, ext0=0.0, ext1=0.0):
    """Rectangle (u,v) from p0 to p1 with width w, extended past the ends."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    px, py = -uy * w / 2, ux * w / 2
    a = (p0[0] - ux * ext0, p0[1] - uy * ext0)
    b = (p1[0] + ux * ext1, p1[1] + uy * ext1)
    return [(a[0] + px, a[1] + py), (a[0] - px, a[1] - py), (b[0] - px, b[1] - py), (b[0] + px, b[1] + py)]


def ngon(cx, cy, r, n=24):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def right_leg():
    ys = -40
    L = {}
    M3 = BOLT["M3"]["hole"]
    L["hip_yaw_link"] = {
        "hip_pitch_servo_XC430": [BH(-14.25, 14.25, ys - 17, ys + 17, 185 + HA - SL, 185 + HA, XC)],
        "hip_yaw_bracket": [
            ("cyl", ((0, ys, 213), (0, 0, 1), 12.5, 23), PRINT),                 # boss: through the bearing, up to the horn
            BH(-20, 20, ys - 16, ys + 16, 201, 213, PRINT),                     # web
            BH(14.5, 18.5, ys - 16, ys + 16, 160, 201, PRINT),                  # cheeks holding the pitch servo
            BH(-18.5, -14.5, ys - 16, ys + 16, 160, 201, PRINT),
            ("prism", ("z", 209, 213, bar((0, ys), TIP_R, 8, 0, 5), []), PRINT),  # yaw lever, perpendicular to the rod
            ("cyl", ((TIP_R[0], TIP_R[1], 213), (0, 0, 1), 2, 8), STEEL),         # lever pin, through the rod
            D((0, ys - 17, 185), "-y", 10, 1),                                   # parent disc of hip_pitch
        ],
    }
    L["thigh"] = {
        "knee_servo_XM430": [BH(-14.25, 14.25, ys - 17, ys + 17, 110 - HA, 110 - HA + SL, XM)],
        "thigh_frame": [
            BH(-18.5, 18.5, ys - 22, ys - 18, 146, 196, PRINT), BH(-18.5, 18.5, ys + 18, ys + 22, 146, 196, PRINT),
            BH(14.5, 18.5, ys - 22, ys + 22, 98, 146, PRINT), BH(-18.5, -14.5, ys - 22, ys + 22, 98, 146, PRINT),
            D((0, ys - 22, 185), "-y", 8, 1),                                    # child disc of hip_pitch
            D((0, ys - 17, 110), "-y", 10, 1),                                   # parent disc of knee
        ],
    }
    L["shank"] = {
        "ankle_pitch_servo_XM430": [BH(-14.25, 14.25, ys - 17, ys + 17, 35 - HA, 35 - HA + SL, XM)],
        "shank_frame": [
            BH(-12, 12, ys - 22, ys - 18, 88, 122, PRINT), BH(-12, 12, ys + 18, ys + 22, 88, 122, PRINT),
            BH(-18.5, 18.5, ys - 22, ys - 18, 76, 88, PRINT), BH(-18.5, 18.5, ys + 18, ys + 22, 76, 88, PRINT),
            BH(14.5, 18.5, ys - 22, ys + 22, 23, 76, PRINT), BH(-18.5, -14.5, ys - 22, ys + 22, 23, 76, PRINT),
            D((0, ys - 22, 110), "-y", 8, 1),                                    # child disc of knee
            D((0, ys - 17, 35), "-y", 10, 1),                                    # parent disc of ankle_pitch
        ],
    }
    L["ankle_link"] = {
        "ankle_roll_servo_XC430": [BH(-60, -26, ys - 14.25, ys + 14.25, 35 - HA, 35 - HA + SL, XC)],
        "ankle_bracket": [
            BH(-60, 12, ys - 27, ys - 23, 16, 50, PRINT), BH(-60, 12, ys + 23, ys + 27, 16, 50, PRINT),
            BH(-64, -60, ys - 27, ys + 27, 16, 50, PRINT),
            D((0, ys - 23, 35), "y", 8, 1),                                      # child disc of ankle_pitch
            D((-26, ys, 35), "x", 10, 1),                                        # parent disc of ankle_roll
        ],
    }
    ft = [((-40, ys - 7, 0), M3), ((-40, ys + 7, 0), M3)]
    L["foot"] = {
        "foot_sole": [BH(-60, 60, ys - 35, ys + 35, 6, 12, PRINT, "z", ft)] +
                     [BH(x0, x0 + 12, y0, y0 + 12, 0, 6, RUBBER) for x0 in (-58, 46) for y0 in (ys - 33, ys + 21)],
        "foot_bracket": [BH(-25, -22.5, ys - 14, ys + 14, 12, 47, PRINT),
                         BH(-50, -25, ys - 14, ys + 14, 12, 18, PRINT, "z", ft),
                         D((-22.5, ys, 35), "x", 8, 1)],                         # child disc of ankle_roll
    }
    F = [("foot", "M3", (-40, y, 18), "-z", 16, 12) for y in (ys - 7, ys + 7)]
    J = [  # name, parent, child, axis, point, lower, upper (deg), actuation
        ("hip_yaw", "torso", "hip_yaw_link", "z", (0, ys, 213), -15, 15, None),
        ("hip_pitch", "hip_yaw_link", "thigh", "y", (0, ys, 185), -90, 30, "XC430-W240"),
        ("knee", "thigh", "shank", "y", (0, ys, 110), 0, 130, "XM430-W350"),
        ("ankle_pitch", "shank", "ankle_link", "y", (0, ys, 35), -60, 45, "XM430-W350"),
        ("ankle_roll", "ankle_link", "foot", "x", (0, ys, 35), -30, 30, "XC430-W240"),
    ]
    return L, F, J


def torso():
    M4, M3 = BOLT["M4"]["hole"], BOLT["M3"]["hole"]
    bolt_pts = [(x, y) for x in (-55, 40) for y in (-50, 50)]
    hz4 = [((x, y, 0), M4) for x, y in bolt_pts]
    shelf_pts = [(x, y) for x in (-53, 29) for y in (-50, 50)]
    hz3 = [((x, y, 0), M3) for x, y in shelf_pts]

    def side(ys, tab_y):
        def holes_in(x0, x1, pts):
            return [h for h in pts if x0 < h[0][0] < x1 and tab_y[0] < h[0][1] < tab_y[1]]
        tabs = [BH(x0, x0 + 20, tab_y[0], tab_y[1], z0, z1, PRINT, "z", holes_in(x0, x0 + 20, hz4))
                for x0 in (-65, 30) for z0, z1 in ((235, 241), (324, 330))]
        ledges = [BH(x0, x0 + 18, tab_y[0], tab_y[1], 268, 272, PRINT, "z", holes_in(x0, x0 + 18, hz3))
                  for x0 in (-62, 20)]
        return [BH(-65, 50, ys[0], ys[1], 235, 330, PRINT)] + tabs + ledges

    P = {
        "yaw_servo_XC430": [BH(HA - SL, HA, -54.25, -25.75, 236, 236 + SD, XC)],          # drives the right hip yaw
        "pelvis_plate": [BH(-65, 50, -62, 62, 225, 235, PRINT, "z", hz4 + [((0, -40, 0), 18.5), ((0, 40, 0), 18.5)])],
        "yaw_bearing_R_6805": [("prism", ("z", 226, 233, ngon(0, -40, 18.5), [(0, -40, 12.5)]), STEEL)],
        "yaw_bearing_L_6805": [("prism", ("z", 226, 233, ngon(0, 40, 18.5), [(0, 40, 12.5)]), STEEL)],
        "torso_side_R": side((-60, -55), (-55, -45)),
        "torso_side_L": side((55, 60), (45, 55)),
        "electronics_shelf": [BH(-62, 48, -54.5, 54.5, 272, 276, PRINT, "z", hz3)],
        "top_plate": [BH(-65, 50, -60, 60, 330, 335, PRINT, "z", hz4)],
        "battery_3S_2200": [BH(-60, 45, -17, 17, 237, 262, BATT)],
        "jetson_orin_nano_devkit": [BH(-50, 48, -40, 40, 282, 312, JET)],
        "imu_bno085": [BH(-10, 10, -10, 10, 276, 280, ELEC)],
        "u2d2_A": [BH(-40, 8, 42, 52, 276, 296, ELEC)],
        "u2d2_B": [BH(-40, 8, -52, -42, 276, 296, ELEC)],
        "dcdc_12V": [BH(12, 42, 42, 52, 280, 305, ELEC)],
        "head_pan_servo_XC430": [BH(HA - SL, HA, -14.25, 14.25, 335, 335 + SD, XC)],
        "shoulder_servo_L_XC430": [BH(-14.25, 14.25, 60, 60 + SD, 318 - SL + HA, 318 + HA, XC)],
        "shoulder_servo_R_XC430": [BH(-14.25, 14.25, -60 - SD, -60, 318 - SL + HA, 318 + HA, XC)],
        "joint_discs_torso": [D((0, 0, 369), "z", 10, 1),            # parent disc of head_yaw
                              D((0, 94, 318), "y", 10, 1),           # parent disc of L shoulder
                              D((0, -94, 318), "-y", 10, 1)],        # parent disc of R shoulder
    }
    F = [("torso", "M4", (x, y, 241), "-z", 20, 16) for x, y in bolt_pts] + \
        [("torso", "M4", (x, y, 335), "-z", 16, 11) for x, y in bolt_pts] + \
        [("torso", "M3", (x, y, 276), "-z", 12, 8) for x, y in shelf_pts]
    return P, F


def head():
    # small camera tilted 20 deg down: a prism along Y whose (u, v) = (z, x) outline is a rotated rectangle
    th = math.radians(20)
    cx, cz, t, h = 40, 392, 10, 25
    pts = []
    for dx, dz in ((-t / 2, -h / 2), (t / 2, -h / 2), (t / 2, h / 2), (-t / 2, h / 2)):
        x = cx + dx * math.cos(th) + dz * math.sin(th)
        z = cz - dx * math.sin(th) + dz * math.cos(th)
        pts.append((z, x))
    return {
        "head_frame": [BH(-20, 30, -20, 20, 370, 374, PRINT), BH(26, 30, -16, 16, 374, 404, PRINT),
                       D((0, 0, 374), "z", 8, 1)],                    # child disc of head_yaw
        "head_camera_small": [("prism", ("y", -13, 13, pts, []), CAM)],
    }


def arm(side):
    s = 1 if side == "L" else -1
    y0, y1 = sorted((s * 96, s * 100))
    return {f"{side}_arm_link": [BH(-10, 10, y0, y1, 210, 326, PRINT), BH(-14, 14, y0, y1, 196, 210, PRINT),
                                 D((0, s * 100, 318), "y" if s > 0 else "-y", 8, 1)]}   # child disc of shoulder


def tie_rod():
    poly = bar(TIP_R, TIP_L, 8, 5, 5)
    return {"yaw_tie_rod": [("prism", ("z", 217, 221, poly, [(TIP_R[0], TIP_R[1], 2), (TIP_L[0], TIP_L[1], 2)]), PRINT)]}


MASS_NOTE = "see README"


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
                parts = {k: [mirror_solid(s) for s in v] for k, v in parts.items()}
                o = (o[0], -o[1], o[2])
            links[f"{side}_{link}"] = {"origin": o, "parts": {f"{side}_{k}": v for k, v in parts.items()}}
        for l, s, p, d, ln, g in RF:
            if side == "L":
                p = (p[0], -p[1], p[2])
            fasteners.append(dict(link=f"{side}_{l}", size=s, p=p, d=d, length=ln, grip=g))
        for name, parent, child, ax, p, lo, hi, act in RJ:
            pp = p if side == "R" else (p[0], -p[1], p[2])
            if name == "hip_yaw":
                act = "XC430-W240 (driven, shared)" if side == "R" else "coupled: q_L = -q_R via tie rod"
            joints.append({"name": f"{side}_{name}", "parent": "torso" if parent == "torso" else f"{side}_{parent}",
                           "child": f"{side}_{child}", "axis": ax, "point_mm": pp, "lower_deg": lo, "upper_deg": hi,
                           "actuator": act})
    # the left leg's yaw lever must point -X (mirror_solid flips only Y): rebuild it
    lp = links["L_hip_yaw_link"]["parts"]["L_hip_yaw_bracket"]
    lp[4] = ("prism", ("z", 209, 213, bar((0, 40), TIP_L, 8, 0, 5), []), PRINT)
    lp[5] = ("cyl", ((TIP_L[0], TIP_L[1], 213), (0, 0, 1), 2, 8), STEEL)
    links["tie_rod"] = {"origin": (TIP_R[0], TIP_R[1], 217), "parts": tie_rod()}
    links["head"] = {"origin": (0, 0, 369), "parts": head()}
    links["L_arm"] = {"origin": (0, 94, 318), "parts": arm("L")}
    links["R_arm"] = {"origin": (0, -94, 318), "parts": arm("R")}
    joints += [
        {"name": "head_yaw", "parent": "torso", "child": "head", "axis": "z", "point_mm": (0, 0, 369),
         "lower_deg": -90, "upper_deg": 90, "actuator": "XC430-W240"},
        {"name": "L_shoulder_pitch", "parent": "torso", "child": "L_arm", "axis": "y", "point_mm": (0, 94, 318),
         "lower_deg": -150, "upper_deg": 150, "actuator": "XC430-W240"},
        {"name": "R_shoulder_pitch", "parent": "torso", "child": "R_arm", "axis": "y", "point_mm": (0, -94, 318),
         "lower_deg": -150, "upper_deg": 150, "actuator": "XC430-W240"},
        {"name": "tie_rod_R_pin", "parent": "R_hip_yaw_link", "child": "tie_rod", "axis": "z",
         "point_mm": (round(TIP_R[0], 2), round(TIP_R[1], 2), 217), "lower_deg": None, "upper_deg": None, "actuator": "passive"},
        {"name": "tie_rod_L_pin", "parent": "tie_rod", "child": "L_hip_yaw_link", "axis": "z",
         "point_mm": (round(TIP_L[0], 2), round(TIP_L[1], 2), 217), "lower_deg": None, "upper_deg": None, "actuator": "passive (closes the loop)"},
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


if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out3"
    os.makedirs(out, exist_ok=True)
    w = write_step(links, f"{out}/biped_robot_v3.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))
    bad = check_interference(links)
    print("interferences between links:", len(bad))
    for b in bad:
        print(" ", b)
