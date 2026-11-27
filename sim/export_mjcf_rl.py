"""Reinforcement-learning variant of an exported Kitoru MJCF: same bodies, joints, inertials, visual meshes, actuators,
closed loop; collision replaced by a few primitives per link (fast GPU contact, MuJoCo Warp).

Run:  cd sim && .venv/bin/python export_mjcf_rl.py            (KITORU_TAG default kitoru_v21, KITORU_DT default 0.002)
In:   model/<tag>.xml + model/meshes_<tag>/col_*.obj (body frame, mm; the hull points the primitives are fitted to)
Out:  model/<tag>_rl.xml (uses the same meshdir, only the visual meshes)

Rules (all from the collision hull points of each body, body frame, zero pose):
  * torso: one box over the frame / covers / lid / shorts hulls (shoulder shafts, horn adapters, arm idlers left out:
    they sit inside the upper arm, which is the torso's child anyway)
  * head: one box over every head hull.  ears: one box per ear (ear + disc; antennas left out)
  * hip roll link, ankle link: one box (AABB of the link's hulls).  hip yaw link: none (inside the shorts)
  * thigh, shank, upper arm, forearm: one capsule along z over the link's hull AABB, radius = mean of the x / y half sizes
  * foot: one box (AABB of the sole hull, bottom 1 mm above the sole so standing contacts go through the pads) +
    5 spheres r 5 mm, bottoms flush with the sole bottom: 4 near the corners of the sole's bottom face (bottom-face
    vertex nearest to each corner of the bottom-face AABB, moved r toward its centre) + 1 at that centre. Named <S>_foot_pad0..4 like
    the v18 pads (sim/walk kw_common reads <S>_foot_pad*). These are not real pad positions (v20 has no pads).
  * nod crank: none (inside the head)
Contact bits (floor contype 2 / conaffinity 1 as in the mesh model):
  R leg contype 1|4 conaff 2|8, L leg contype 1|8 conaff 2|4   -> right leg <-> left leg, never inside one leg
  torso contype 1|16 conaff 2, arms contype 1 conaff 2|16       -> forearm <-> torso (upper arm <-> torso is
                                                                   parent-child, filtered by MuJoCo)
  head, ears, foot pads contype 1 conaff 2                      -> floor only (foot <-> foot goes through the boxes)

v22 rules (KITORU_TAG kitoru_v22 and later; the v21 output above is unchanged). Every collision geom is a box (or a
foot pad sphere); sizes below are the R side in body frames (mm), the L side mirrors y. The tables (SHORTS_V22, HEAD_V22,
LEG_V22) come from the shape voxels of build21_arm and are fitted so the first-contact angles match the shape sweeps
(sim/tests/contact_v22.py rl / mesh vs sim/tests/shape_contact_v22.py, sim/walk/ranges/results/v22).
  * torso: torso_box = AABB of the frame / covers / lid hulls above the pelvis underside z PELVIS_Z (221) + the shorts
    as a hollow ring of four wall boxes (z 205 .. 221; the front / back wall bottoms 206.5 / 207.5 stand in for the R8
    hem roll). The ring's inside (hip yaw / roll links) stays empty, so no torso <-> hip roll link <exclude>.
  * head: three nested boxes (HEAD_V22) following the rounded head (+-100 wide only at z 39 .. 164); ears as v21
  * hip roll link: upper (U bracket) + lower (bracket below z -49, pitch servo) boxes
  * thigh: frame box + shell band box. shank: main (plates + shell z -86 .. -6), low (frame bars z -100 .. -86),
    servo (ankle pitch servo bottom). ankle link: low (bracket up to z 20) + back (roll servo, bracket behind it)
  * upper arm, forearm: one box each (AABB of the link's hulls; the forearm cut at the elbow axis)
  * foot: <S>_foot_box over the sole plate (bottom 1 mm above the pads, top SOLE_TOP_V22 -23.5) + <S>_foot_rim (inner
    rim beside the shank, top -20) + <S>_foot_column (horn column, meets the shank back at ankle pitch +30)
  Contact (every non-floor pair that can touch; parent-child pairs are filtered by MuJoCo anyway):
    groups: T torso, H head + ears, RA / LA arm (upper arm + forearm), RU / LU hip roll link + thigh, RS / LS shank,
            RF / LF ankle link + foot box, RSd / LSd shank low + servo, RFd / LFd foot rim + column;
            contype = 1 | own bit, conaffinity = 2 | partner bits (BIT22 / PART22)
    T  <-> H, RA, LA, all leg groups          H <-> T, RA, LA, all leg groups
    RA <-> T, H, all leg groups (not LA: the arms turn about y only and stay at |y| >= 77)
    RU <-> T, H, RA, LA, LU, LS, LF, RF       RS <-> T, H, RA, LA, LU, LS, LF, RF, RFd
    RF <-> T, H, RA, LA, LU, LS, LF, RS, RU, RSd
    RSd <-> RF, RFd     RFd <-> RS, RSd       (detail boxes: own shank <-> foot only, plus the floor)
    (L the same with R / L swapped.) Within one leg: shank <-> foot (ankle pitch at ankle roll +-20), hip roll link /
    thigh <-> ankle link / foot (deep knee with the hip forward: hip roll link <-> ankle roll servo, knee 109 .. 114);
    thigh <-> shank and shank <-> ankle link are parent-child. The mesh model's shank-foot <exclude> is dropped in the
    rl file. Foot pads contype 1 conaffinity 2 (floor only).
"""
import os, sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("KITORU_TAG", "kitoru_v21")
DT = float(os.environ.get("KITORU_DT", "0.002"))
SRC = os.path.join(HERE, "model", f"{TAG}.xml")
DST = os.path.join(HERE, "model", f"{TAG}_rl.xml")
PAD_R = 5.0          # mm
BOX_LIFT = 1.0       # mm, foot box bottom above the pad bottoms
TORSO_SKIP = ("shoulder_shaft", "horn_adapter", "horn_XC430", "idler_XC430", "nod_rod", "neck_", "nod_anchor")
import re
V22 = int(re.match(r"kitoru_v(\d+)", TAG).group(1)) >= 22
PELVIS_Z = 221.0     # v22: torso box bottom = pelvis hatch underside (mm, torso frame = world at zero pose)
# v22 shorts ring walls (torso frame mm): bottom z, inner face; outer faces = the shorts hull AABB (x -93 .. 70, |y| 75)
SHORTS_V22 = dict(front_z0=206.5, front_x_in=62.0, back_z0=207.5, back_x_in=-84.0, side_z0=205.0, side_y_in=68.0)
SOLE_TOP_V22 = -23.5   # foot frame z of the sole plate top (shape voxels); the -19.5 rim is LEG_V22 foot_rim
# v22 leg boxes (R leg, body frame mm; the L leg is the mirror y -> -y), read off the shape voxels (1 mm) of
# build21_arm (sim/tests/contact_v22.py re-checks them against the shape sweeps):
#   thigh: frame x -17.5 .. 18.5 |y| 24 up to z 10, shell band x -19.5 .. 22.5 |y| 27 at z -89 .. -66, knee end
#          x -22.5 .. 17.5 at z -107 .. -89
#   shank: two side plates |y| 15 .. 24 x -17.5 .. 18.5, shell x -19.5 .. 22.5 z -86 .. -6, frame bars between the
#          plates (back bar x -22.5 only at z -102 .. -86), ankle pitch servo down to z -106; above z -6 the frame
#          narrows round the knee (inside the thigh box)
#   hip roll link: U bracket x -23.4 .. 24.6 down to z -49, x -20.4 .. 19.6 below; pitch servo x +-14 down to z -88.3
#   ankle link: bracket x -73 .. 12 up to z 14, then only behind (x <= 5 .. -44 going up to z 32); roll servo
#          x -67.5 .. -34.5 |y| 14 up to z 35.2
#   foot sole: plate top z -23.5 with a -19.5 rim (4 mm) round the edge, a raised block x -62 .. -14 |y| 16 at -15.5,
#          the horn column x -33.5 .. -27.5 |y| 13.5 up to z 11.5
# Contacts they carry (sim/walk/ranges/results/v22): hip pitch -117.8 thigh shell / +114.8 thigh frame <-> shorts,
# ankle pitch -77.8 shank shell front corner <-> rim, +31.1 shank frame back <-> column top, ankle roll +-20 with
# pitch -50 frame plate corner <-> plate top, knee 105 .. 110 with the hip forward: hip pitch servo <-> ankle roll servo.
# v22 head: three nested boxes (head frame = nod axis, mm) instead of the AABB (y +-100 over the full height): the
# head is +-100 wide only at z 39 .. 164 (shape voxels: |y| <= 60 z -8 .. 204, |y| 60 .. 80 z 24 .. 179, |y| 80 .. 100
# z 39 .. 164, x narrowing toward the sides). The AABB box met the raised arms 10 - 20 mm before the real head did.
HEAD_V22 = {"head_core": ((-70.0, -60.0, 5.0), (71.0, 60.0, 204.5)),
            "head_mid": ((-70.0, -80.0, 24.0), (70.0, 80.0, 179.5)),
            "head_wide": ((-69.0, -100.0, 39.0), (65.0, 100.0, 164.5))}
FOREARM_TOP_V22 = 0.0   # forearm box top = elbow axis (forearm frame); shape: x +-18 up to z -27, +-10 at z +9
# detail boxes that only take part in the same-leg shank <-> foot contact (own bits, see PART22): fewer pairs
DETAIL_V22 = ("shank_low", "shank_servo", "foot_rim", "foot_column")
LEG_V22 = {
    "thigh": {"thigh_frame": ((-18.0, -24.7, -107.0), (19.0, 24.3, 11.0)),
              "thigh_shell": ((-20.0, -27.5, -89.0), (23.0, 27.5, -65.5))},
    "shank": {"shank_main": ((-18.0, -30.5, -86.0), (23.0, 27.5, -6.0)),
              "shank_low": ((-23.0, -21.7, -100.0), (18.0, 22.3, -86.0)),
              "shank_servo": ((-14.3, -17.0, -106.7), (14.7, 17.0, -100.0))},
    "hip_roll_link": {"roll_upper": ((-23.9, -17.5, -49.0), (25.1, 17.5, 10.5)),
                      "roll_lower": ((-20.9, -17.0, -88.8), (20.1, 17.0, -49.0))},
    "ankle_link": {"ankle_low": ((-73.5, -27.0, -11.2), (12.0, 27.0, 20.0)),
                   "ankle_back": ((-68.0, -27.0, 20.0), (-34.5, 27.0, 35.2))},      # roll servo + bracket behind it
    "foot": {"foot_rim": ((-20.0, 20.5, -34.0), (40.0, 25.0, -20.0)),      # inner rim beside the shank (top fitted: -19.5 .. -21.5 tried)
             "foot_column": ((-34.0, -14.0, -16.0), (-27.0, 14.0, 12.0))},
}

tree = ET.parse(SRC, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
root = tree.getroot()
meshdir = os.path.join(os.path.dirname(SRC), root.find("compiler").get("meshdir"))


def obj_v(name):
    with open(os.path.join(meshdir, name + ".obj")) as f:
        return np.array([[float(x) for x in l.split()[1:4]] for l in f if l.startswith("v ")])


def f3(v):
    return " ".join("%.6g" % (x * 1e-3) for x in v)


def box(nm, lo, hi, cls):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return dict(name=nm, type="box", pos=f3((lo + hi) / 2), size=f3((hi - lo) / 2), **{"class": cls})


def capsule(nm, lo, hi, cls):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = (lo + hi) / 2; r = 0.25 * ((hi[0] - lo[0]) + (hi[1] - lo[1]))
    z0, z1 = lo[2] + r, hi[2] - r
    assert z1 > z0, (nm, lo, hi)
    return dict(name=nm, type="capsule", fromto=f3((c[0], c[1], z0)) + " " + f3((c[0], c[1], z1)),
                size="%.6g" % (r * 1e-3), **{"class": cls})


GROUP22 = lambda b: ("T" if b == "torso" else "H" if b in ("head", "ears", "nod_crank") else
                     (b[0] + "A") if b.endswith(("_arm", "_forearm")) else (b[0] + "U") if b.endswith(("_hip_roll_link", "_thigh", "_hip_yaw_link")) else
                     (b[0] + "S") if b.endswith("_shank") else (b[0] + "F"))
BIT22 = {g: 1 << (2 + i) for i, g in enumerate(["T", "H", "RA", "LA", "RU", "RS", "RF", "LU", "LS", "LF", "RSd", "RFd", "LSd", "LFd"])}
LEGS22 = ["RU", "RS", "RF", "LU", "LS", "LF"]
PART22 = {"T": ["H", "RA", "LA"] + LEGS22, "H": ["T", "RA", "LA"] + LEGS22,
          "RA": ["T", "H"] + LEGS22, "LA": ["T", "H"] + LEGS22}
for S_, O_ in (("R", "L"), ("L", "R")):
    other = [O_ + "U", O_ + "S", O_ + "F"]
    PART22[S_ + "U"] = ["T", "H", "RA", "LA"] + other + [S_ + "F"]
    PART22[S_ + "S"] = ["T", "H", "RA", "LA"] + other + [S_ + "F", S_ + "Fd"]
    PART22[S_ + "F"] = ["T", "H", "RA", "LA"] + other + [S_ + "S", S_ + "U", S_ + "Sd"]
    PART22[S_ + "Sd"] = [S_ + "F", S_ + "Fd"]          # shank detail boxes: own foot only (and the floor)
    PART22[S_ + "Fd"] = [S_ + "S", S_ + "Sd"]          # foot detail boxes: own shank only (and the floor)
for g, ps in PART22.items():          # partner lists must be symmetric
    for q in ps:
        assert g in PART22[q], (g, q)


def cls_of(b):
    if V22:
        return "rl22_" + GROUP22(b)
    if b[:2] in ("R_", "L_") and ("arm" in b or "forearm" in b):
        return "rl_arm"
    if b[:2] == "R_":
        return "rl_leg_R"
    if b[:2] == "L_":
        return "rl_leg_L"
    return {"torso": "rl_torso"}.get(b, "rl_head")


LOG = []
mesh_used = set()
for body in root.iter("body"):
    b = body.get("name")
    cols = [g for g in body.findall("geom") if g.get("class") in ("col", "col_leg", "pad")]
    pts = {g.get("mesh"): obj_v(g.get("mesh")) for g in cols if g.get("mesh")}
    for g in cols:
        body.remove(g)
    if not pts:
        continue
    allp = np.concatenate(list(pts.values()))
    lo, hi = allp.min(0), allp.max(0)
    c = cls_of(b); new = []
    if b == "torso" and V22:
        P = np.concatenate([v for k, v in pts.items() if not any(s in k for s in TORSO_SKIP) and "shorts" not in k])
        P = P[P[:, 2] >= PELVIS_Z - 1e-6]
        new.append(box("torso_box", (P[:, 0].min(), P[:, 1].min(), PELVIS_Z), P.max(0), c))
        Q = pts["col_shorts"]; qlo, qhi = Q.min(0), Q.max(0); S_ = SHORTS_V22
        new.append(box("shorts_front", (S_["front_x_in"], qlo[1], S_["front_z0"]), (qhi[0], qhi[1], PELVIS_Z), c))
        new.append(box("shorts_back", (qlo[0], qlo[1], S_["back_z0"]), (S_["back_x_in"], qhi[1], PELVIS_Z), c))
        for nm_, y0, y1 in (("shorts_side_R", qlo[1], -S_["side_y_in"]), ("shorts_side_L", S_["side_y_in"], qhi[1])):
            new.append(box(nm_, (S_["back_x_in"], y0, S_["side_z0"]), (S_["front_x_in"], y1, PELVIS_Z), c))
        LOG.append(f"torso v22: box z >= {PELVIS_Z}, shorts hull {np.round(qlo, 1)} .. {np.round(qhi, 1)}, walls {S_}")
    elif b == "torso":
        P = np.concatenate([v for k, v in pts.items() if not any(s in k for s in TORSO_SKIP)])
        new.append(box("torso_box", P.min(0), P.max(0), c))
    elif b == "head" and V22:
        for nm_, (l_, h_) in HEAD_V22.items():
            assert np.all(np.array(l_) >= lo - 1.0) and np.all(np.array(h_) <= hi + 1.0), (nm_, l_, h_, lo, hi)
            new.append(box(nm_, l_, h_, c))
    elif b == "head":
        new.append(box("head_box", lo, hi, c))
    elif b == "ears":
        for S in "LR":
            P = np.concatenate([pts[f"col_ear_{S}"], pts[f"col_ear_disc_{S}"]])
            new.append(box(f"ear_{S}_box", P.min(0), P.max(0), c))
    elif V22 and b[2:] in LEG_V22 and b[2:] != "foot":
        for nm_, (l_, h_) in LEG_V22[b[2:]].items():
            l_, h_ = np.array(l_), np.array(h_)
            if b[0] == "L":
                l_[1], h_[1] = -h_[1], -l_[1]
            assert np.all(l_ >= lo - 1.0) and np.all(h_ <= hi + 1.0), (b, nm_, l_, h_, lo, hi)
            new.append(box(f"{b[0]}_{nm_}", l_, h_, c + "d" if nm_ in DETAIL_V22 else c))
    elif b.endswith(("_hip_roll_link", "_ankle_link")):
        new.append(box(f"{b}_box", lo, hi, c))
    elif V22 and b.endswith("_arm"):
        new.append(box(f"{b}_box", lo, hi, c))
    elif V22 and b.endswith("_forearm"):          # up to the elbow axis: above it the forearm rounds off inside the upper arm box
        new.append(box(f"{b}_box", lo, (hi[0], hi[1], min(hi[2], FOREARM_TOP_V22)), c))
    elif b.endswith(("_thigh", "_shank", "_arm", "_forearm")):
        new.append(capsule(f"{b}_cap", lo, hi, c))
    elif b.endswith("_foot"):
        S = b[0]
        P = pts[f"col_{S}_foot_sole"]; slo, shi = P.min(0), P.max(0)
        if V22:
            new.append(box(f"{b}_box", (slo[0], slo[1], slo[2] + BOX_LIFT), (shi[0], shi[1], SOLE_TOP_V22), c))
            for nm_, (l_, h_) in LEG_V22["foot"].items():
                l_, h_ = np.array(l_), np.array(h_)
                if S == "L":
                    l_[1], h_[1] = -h_[1], -l_[1]
                new.append(box(f"{S}_{nm_}", l_, h_, c + "d" if nm_ in DETAIL_V22 else c))
        else:
            new.append(box(f"{b}_box", (slo[0], slo[1], slo[2] + BOX_LIFT), shi, c))
        bot = P[P[:, 2] < slo[2] + 0.5][:, :2]
        blo, bhi = bot.min(0), bot.max(0); ctr = 0.5 * (blo + bhi)
        pads = []
        for cx, cy in ((blo[0], blo[1]), (blo[0], bhi[1]), (bhi[0], blo[1]), (bhi[0], bhi[1])):
            q = bot[np.argmin(np.hypot(bot[:, 0] - cx, bot[:, 1] - cy))]
            d = ctr - q; q = q + PAD_R * d / np.linalg.norm(d)
            pads.append(q)
        pads.append(ctr)
        for i, q in enumerate(pads):
            new.append(dict(name=f"{b}_pad{i}", type="sphere", size="%.6g" % (PAD_R * 1e-3),
                            pos=f3((q[0], q[1], slo[2] + PAD_R)), **{"class": "rl_pad"}))
        LOG.append(f"{b} pads (mm, body frame): " + ", ".join("(%.1f, %.1f)" % tuple(q) for q in pads))
    # hip yaw link, nod crank: no collision geom
    k = max(i for i, e in enumerate(list(body)) if e.tag == "geom") + 1 if body.findall("geom") else 0
    for i, a in enumerate(new):
        body.insert(k + i, ET.Element("geom", a))
    LOG.append(f"{b}: {len(pts)} hulls -> " + ", ".join(a["type"] for a in new) if new else f"{b}: {len(pts)} hulls -> none")

# drop the collision mesh assets
asset = root.find("asset")
for m in [m for m in asset.findall("mesh") if m.get("name").startswith("col_")]:
    asset.remove(m)

# defaults: rl classes under "robot"
robot = next(d for d in root.find("default").iter("default") if d.get("class") == "robot")
RL_CLASSES = (("rl_leg_R", 1 | 4, 2 | 8, "0.9 0.2 0.6 0.4"), ("rl_leg_L", 1 | 8, 2 | 4, "0.6 0.2 0.9 0.4"),
              ("rl_torso", 1 | 16, 2, "0.9 0.4 0.2 0.4"), ("rl_arm", 1, 2 | 16, "0.2 0.6 0.9 0.4"),
              ("rl_head", 1, 2, "0.9 0.7 0.2 0.4"), ("rl_pad", 1, 2, "0.1 0.1 0.1 1"))
if V22:
    RGBA22 = {"T": "0.9 0.4 0.2 0.4", "H": "0.9 0.7 0.2 0.4", "A": "0.2 0.6 0.9 0.4", "R": "0.9 0.2 0.6 0.4", "L": "0.6 0.2 0.9 0.4"}
    RL_CLASSES = tuple((f"rl22_{g}", 1 | BIT22[g], 2 | sum(BIT22[q] for q in PART22[g]),
                        RGBA22["A" if g[1:2] == "A" else g[0]]) for g in BIT22) + (("rl_pad", 1, 2, "0.1 0.1 0.1 1"),)
    # within-leg shank <-> foot is on in v22: drop the mesh model's shank-foot excludes
    con = root.find("contact")
    for e in list(con):
        if e.tag == "exclude" and e.get("body1", "").endswith("_shank") and e.get("body2", "").endswith("_foot"):
            con.remove(e)
        elif e.tag is ET.Comment and "shank vs foot" in (e.text or ""):
            e.text = (" v22 rl: same-leg shank <-> foot collide (foot box = sole plate only); the mesh model keeps the "
                      "shank-foot exclude (its hulls overlap at the zero pose) ")
for nm, ct, ca, rgba in RL_CLASSES:
    d = ET.SubElement(robot, "default", {"class": nm})
    ET.SubElement(d, "geom", dict(group="3", contype=str(ct), conaffinity=str(ca), density="0", rgba=rgba))

# timestep and the closed-loop solref (MuJoCo clamps solref time constants below 2 x timestep)
opt = root.find("option"); dt0 = float(opt.get("timestep")); opt.set("timestep", "%g" % DT)
for e in root.find("equality"):
    if e.tag is ET.Comment and "solref" in (e.text or ""):
        e.text = e.text.replace("solref time constant 0.002 = 2 x timestep", "solref time constant = max(0.002, 2 x timestep)")
    if e.tag == "joint":
        e.set("solref", "%g 1" % max(float(e.get("solref").split()[0]), 2 * DT))
root.set("model", root.get("model") + "_rl")
root.insert(0, ET.Comment(f" generated by sim/export_mjcf_rl.py from {os.path.basename(SRC)}: collision = primitives "
                          f"(see the script docstring{', v22 rules' if V22 else ''}), visual meshes unchanged, timestep {dt0:g} -> {DT:g}, "
                          f"equality solref >= 2 x timestep. Do not edit by hand. "
                          + ("Contact (v22): every pair that can touch except parent-child; inside one leg only shank <-> foot and hip roll "
                             "link / thigh <-> ankle link / foot; arm <-> other arm off. Groups T torso + shorts ring, H head + ears, "
                             "RA / LA arms, RU / LU hip roll link + thigh, RS / LS shank, RF / LF ankle link + foot box, RSd / LSd "
                             "shank low + servo boxes and RFd / LFd foot rim + column boxes (these two only with their own shank / foot); "
                             "contype 1 | own bit, conaffinity 2 | partner bits "
                             f"({', '.join(f'{g} {BIT22[g]}' for g in BIT22)}); pads floor only. " if V22 else "")))
ET.indent(tree, space="  ")
tree.write(DST, encoding="unicode")
for s in LOG:
    print(s)

import mujoco
m = mujoco.MjModel.from_xml_path(DST)
print(f"compiled {os.path.basename(DST)}: nbody {m.nbody} njnt {m.njnt} nu {m.nu} ngeom {m.ngeom} "
      f"(collision {int(((m.geom_contype | m.geom_conaffinity) > 0).sum())}) nmesh {m.nmesh} timestep {m.opt.timestep}")
