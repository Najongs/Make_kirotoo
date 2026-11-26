"""v15 = v14 + lighter structure + optimised servo assignment.

Lightening (all openings are open to the top in the print orientation or are vertical windows <= 30 mm wide, so no
support is added):
  * pelvis plate: 2 mm floor + 8 mm ribbed layer with pockets (top side), solid around the yaw bearings,
    under the yaw servos, the arm servo seat, the crank bearing block and along the walls
  * side walls: six windows each (also give the leg cables more room)
  * electronics shelf: four windows, cross band in the middle (IMU) and a 10 mm rim
  * foot soles: 2 mm floor + 4 mm ribbed layer with pockets in front of the ankle bracket

Servo assignment (servo_opt.py, see README): XM430-W350 at hip pitch, knee, ankle roll; XC430-W240 at hip yaw and
ankle pitch; XL430-W250 (TurtleBot3 spares) on the head nod and the arm crank. Same X430 housing everywhere, so the
geometry does not change.
"""
import os, json, math
import build14 as B
from build14 import *          # noqa: F401,F403  (constants, helpers)
from build4 import XC, XM
from shapes import rrect

FORCE = dict(B.FORCE)
NOTE = dict(B.NOTE)
NOTE.update({
    "torso_frame": "골반판을 바닥에. 골반·측벽·뒷벽·레일이 한 부품. 골반판 윗면 포켓, 측벽 창",
    "electronics_shelf": "평평하게. 창 4개",
    "foot_sole": "발판을 바닥에. 발 브래킷이 한 몸. 발판 윗면 포켓",
})

SERVO_MAP = {   # part name -> new part name, colour
    "R_hip_pitch_servo_XC430": ("R_hip_pitch_servo_XM430", XM), "L_hip_pitch_servo_XC430": ("L_hip_pitch_servo_XM430", XM),
    "R_ankle_pitch_servo_XM430": ("R_ankle_pitch_servo_XC430", XC), "L_ankle_pitch_servo_XM430": ("L_ankle_pitch_servo_XC430", XC),
    "head_nod_servo_XC430": ("head_nod_servo_XL430", XC), "arm_swing_servo_XC430": ("arm_swing_servo_XL430", XC),
}
ACT = {"hip_pitch": "XM430-W350", "ankle_pitch": "XC430-W240", "knee": "XM430-W350", "ankle_roll": "XM430-W350",
       "hip_yaw": "XC430-W240"}

# ---------------------------------------------------------------- pocket helpers
RIB = 3.0


def _ok_rect(x0, x1, y0, y1, keep_rects, keep_circles, bound):
    bx0, bx1, by0, by1 = bound
    if x0 < bx0 or x1 > bx1 or y0 < by0 or y1 > by1:
        return False
    for a0, a1, b0, b1 in keep_rects:
        if x0 < a1 and x1 > a0 and y0 < b1 and y1 > b0:
            return False
    for cx, cy, r in keep_circles:
        nx, ny = min(max(cx, x0), x1), min(max(cy, y0), y1)
        if math.hypot(nx - cx, ny - cy) < r:
            return False
    return True


def grid_pockets(xs, ys, keep_rects, keep_circles, bound, r=2.0):
    """xs, ys: lists of (lo, hi) cell intervals; returns rrect loops (u=x, v=y) of the accepted cells"""
    out = []
    for x0, x1 in xs:
        for y0, y1 in ys:
            if _ok_rect(x0, x1, y0, y1, keep_rects, keep_circles, bound):
                out.append(rrect(x0, x1, y0, y1, r))
    return out


def cells(lo, hi, n, rib=RIB):
    w = (hi - lo - (n - 1) * rib) / n
    return [(lo + i * (w + rib), lo + i * (w + rib) + w) for i in range(n)]


# ---------------------------------------------------------------- pelvis plate
PELVIS_KEEP_R = [(-37, 13, 23, 57), (-37, 13, -57, -23),        # yaw servos (sit on the plate)
                 (17, 50, -53, -19),                             # arm servo seat
                 (35, 59, 35, 46),                               # crank bearing block
                 (-66, -43, 43, 63), (-66, -43, -63, -43)]       # back stiffeners / ledges
PELVIS_KEEP_C = [(0, 40, 25.0), (0, -40, 25.0)]                 # 6805 seats
PELVIS_BOUND = (-60.5, 46.0, -52.0, 52.0)                       # walls at |y| 55, back wall x -63


def pelvis_pockets():
    xs = cells(-60.5, 46.0, 6)
    ys = cells(-52.0, 52.0, 6)
    return grid_pockets(xs, ys, PELVIS_KEEP_R, PELVIS_KEEP_C, PELVIS_BOUND)


# ---------------------------------------------------------------- side wall windows (x, z)
WALL_WINDOWS = [(-41, -31, 245, 296), (-12, 2, 245, 298), (22, 44, 277, 306), (24, 44, 242, 254),
                (-59, -49, 278, 306), (-59, -49, 244, 254)]

# ---------------------------------------------------------------- shelf windows (x, y)
SHELF_WINDOWS = [(-52, -16, -44, -16), (-52, -16, 16, 44), (-4, 38, -44, -16), (-4, 38, 16, 44)]


def foot_pockets(ys):
    xs = cells(-26.0, 56.0, 4)
    yy = cells(ys - 31.0, ys + 31.0, 3)
    return grid_pockets(xs, yy, [], [], (-27, 57, ys - 32, ys + 32))


def lighten(links):
    T = links["torso"]["parts"]
    fr = T["torso_frame"]
    # pelvis plate -> floor + pocketed layer
    k, (ax, a0, a1, outer, inn, holes) = 0, fr[0][1]
    assert fr[0][0] == "rprism" and ax == "z" and (a0, a1) == (225, 235)
    fr[0] = ("rprism", ("z", 225.0, 227.0, outer, [], holes), fr[0][2])
    fr.insert(1, ("rprism", ("z", 227.0, 235.0, outer, pelvis_pockets(), holes), fr[0][2]))
    # side walls: rprism along y spanning x -65..50, z 235..330
    n = 0
    for i, s in enumerate(fr):
        if s[0] == "rprism" and s[1][0] == "y" and abs(s[1][2] - s[1][1]) == 5.0 and len(s[1][3]) >= 4:
            us = [p[0] for p in s[1][3]]
            if min(us) == 235 and max(us) == 330:
                ax, a0, a1, outer, inn, holes = s[1]
                wins = [rrect(z0, z1, x0, x1, 3.0) for x0, x1, z0, z1 in WALL_WINDOWS]
                fr[i] = ("rprism", (ax, a0, a1, outer, list(inn) + wins, holes), s[2])
                n += 1
    assert n == 2, n
    # shelf
    sh = T["electronics_shelf"][0]
    ax, a0, a1, outer, inn, holes = sh[1]
    T["electronics_shelf"] = [("rprism", (ax, a0, a1, outer, list(inn) + [rrect(x0, x1, y0, y1, 4.0)
                                                                         for x0, x1, y0, y1 in SHELF_WINDOWS], holes), sh[2])]
    # feet
    for side, ys in (("R", -40.0), ("L", 40.0)):
        ss = links[f"{side}_foot"]["parts"][f"{side}_foot_sole"]
        for i, s in enumerate(ss):
            if s[0] == "rprism" and s[1][0] == "z" and (s[1][1], s[1][2]) == (6, 12):
                ax, a0, a1, outer, inn, holes = s[1]
                ss[i] = ("rprism", ("z", 6.0, 8.0, outer, list(inn), holes), s[2])
                ss.insert(i + 1, ("rprism", ("z", 8.0, 12.0, outer, list(inn) + foot_pockets(ys), holes), s[2]))
                break
        else:
            raise RuntimeError("sole plate not found")


def reassign(links, joints):
    for ln, L in links.items():
        for old, (new, rgb) in SERVO_MAP.items():
            if old in L["parts"]:
                L["parts"][new] = [(k, a, rgb) for k, a, _ in L["parts"].pop(old)]
    for j in joints:
        for key, act in ACT.items():
            if j["name"] in (f"R_{key}", f"L_{key}"):
                j["actuator"] = act
        if j["name"] == "nod_crank":
            j["actuator"] = "XL430-W250 (inside the head)"
        if j["name"] == "arm_crank":
            j["actuator"] = "XL430-W250"


def build_model():
    links, joints, fasteners = B.build_model()
    lighten(links)
    reassign(links, joints)
    return links, joints, fasteners


write = B.write

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out15"
    os.makedirs(out, exist_ok=True)
    w = write(links, f"{out}/biped_robot_v15.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))
    print("pelvis pockets", len(pelvis_pockets()), "foot pockets", len(foot_pockets(-40)))
