"""Mass model for the v22 sim export: every link of the current build (code_v17/build21_arm.build_model =
build20 legs incl. print_robust + build21_body torso / head second pass + elbow arm durability pass), profile 'light'.

Differences from out21_arm/mass_build21_arm_light.json (which re-voxelised only the arms and the torso and took every
other link from the v20 mass file):
  * every link is voxelised from the current geometry (legs, head, ears, nod crank / rods too)
  * all 17 servos are XL430-W250-T 65.2 g (owner 2026-11-27) inside mass_props itself (same case envelope as the
    XM430 / XC430 the build still names on the legs), so sim/export_mjcf.py has no swap to do
  * printed at 100 % infill as the print kit says (out20/PRINT_v2/README.md): R/L_foot_sole, arm_R/L_horn_adapter
    (the 'light' result without this override is logged as well)

Run:  cd code_v17 && ../sim/.venv/bin/python ../sim/tools/mass_v22.py [out.json]
      (default out: sim/model/mass_kitoru_v22_light.json)
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "code_v17")
sys.path.insert(0, CODE)
os.chdir(CODE)
import mass_props as MP

MP.SERVO_G = {"XM430": 65.2, "XC430": 65.2, "XL430": 65.2}
FULL = ("_foot_sole", "_horn_adapter")          # printed solid (kit notes)
_pp = MP.part_props
LIGHT = {}


def part_props(pn, sol, prof):
    r = _pp(pn, sol, prof)
    if r is None or not any(pn.endswith(k) for k in FULL) or r[0]["kind"] != "print":
        return r
    info, W, lo = r
    LIGHT[pn] = float(W.sum())
    V = W > 0
    W = V * MP.RHO_PLA * MP.H ** 3
    info = dict(info, mass=W.sum(), infill="100 %")
    return info, W, lo


MP.part_props = part_props
import build21_arm
t0 = time.time()
out, parts, joints = MP.link_props(build21_arm, "light")
tot = sum(v["mass_g"] for v in out.values())
com = sum(np.array(v["com_world"]) * v["mass_g"] for v in out.values()) / tot
dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "model", "mass_kitoru_v22_light.json")
note = ("sim/tools/mass_v22.py: every link voxelised from build21_arm (build20 print_robust legs, build21_body second "
        "pass torso / head, arm durability), profile light, all servos XL430 65.2 g, foot soles + shoulder horn adapters 100 % infill")
json.dump(dict(links=out, parts=parts, total_g=tot, com=list(com), note=note,
               light_only_g={k: round(v, 2) for k, v in LIGHT.items()}), open(dst, "w"), indent=1)
print(f"total {tot:.1f} g  CoM {np.round(com, 1)}  ({time.time() - t0:.0f} s) -> {dst}")
print("100 % infill parts (light -> full g):", {k: (round(v, 1), round(next(p['mass'] for p in parts if p['part'] == k), 1)) for k, v in LIGHT.items()})
for ln, v in out.items():
    print(f"  {ln:16s} {v['mass_g']:7.1f} g  com {np.round(v['com_world'], 1)}")
