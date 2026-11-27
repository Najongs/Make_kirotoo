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


def cls_of(b):
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
    if b == "torso":
        P = np.concatenate([v for k, v in pts.items() if not any(s in k for s in TORSO_SKIP)])
        new.append(box("torso_box", P.min(0), P.max(0), c))
    elif b == "head":
        new.append(box("head_box", lo, hi, c))
    elif b == "ears":
        for S in "LR":
            P = np.concatenate([pts[f"col_ear_{S}"], pts[f"col_ear_disc_{S}"]])
            new.append(box(f"ear_{S}_box", P.min(0), P.max(0), c))
    elif b.endswith(("_hip_roll_link", "_ankle_link")):
        new.append(box(f"{b}_box", lo, hi, c))
    elif b.endswith(("_thigh", "_shank", "_arm", "_forearm")):
        new.append(capsule(f"{b}_cap", lo, hi, c))
    elif b.endswith("_foot"):
        S = b[0]
        P = pts[f"col_{S}_foot_sole"]; slo, shi = P.min(0), P.max(0)
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
for nm, ct, ca, rgba in (("rl_leg_R", 1 | 4, 2 | 8, "0.9 0.2 0.6 0.4"), ("rl_leg_L", 1 | 8, 2 | 4, "0.6 0.2 0.9 0.4"),
                         ("rl_torso", 1 | 16, 2, "0.9 0.4 0.2 0.4"), ("rl_arm", 1, 2 | 16, "0.2 0.6 0.9 0.4"),
                         ("rl_head", 1, 2, "0.9 0.7 0.2 0.4"), ("rl_pad", 1, 2, "0.1 0.1 0.1 1")):
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
                          f"(see the script docstring), visual meshes unchanged, timestep {dt0:g} -> {DT:g}, "
                          f"equality solref >= 2 x timestep. Do not edit by hand. "))
ET.indent(tree, space="  ")
tree.write(DST, encoding="unicode")
for s in LOG:
    print(s)

import mujoco
m = mujoco.MjModel.from_xml_path(DST)
print(f"compiled {os.path.basename(DST)}: nbody {m.nbody} njnt {m.njnt} nu {m.nu} ngeom {m.ngeom} "
      f"(collision {int(((m.geom_contype | m.geom_conaffinity) > 0).sum())}) nmesh {m.nmesh} timestep {m.opt.timestep}")
