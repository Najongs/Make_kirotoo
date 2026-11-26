"""v21 head = v20 geometry with the wider nod range (code_v17/head21.py, out_head21/HEAD_CRANK_v21.md).

The crank (R 14, anchor (-14.75, 345), rod 38.45) and the ear link stay as in v20: head21.py shows the loop already
reaches head -20 .. +28 with transmission angles >= 52 deg, so no part changes and no STEP is written here. Only the
joint table changes: head_nod -10 .. +20 -> -20 .. +28; nod_crank / ear_axle limits follow the loop.
"""
import build20 as B20
from build20 import *          # noqa
import head21 as H21


def build_model():
    links, joints, fasteners = B20.build_model()
    lo, hi = H21.NEW_RANGE
    a_lo, a_hi = H21.nod(lo)["alpha"], H21.nod(hi)["alpha"]
    e_lo, e_hi = H21.ear(a_lo)["psi"], H21.ear(a_hi)["psi"]
    for j in joints:
        if j["name"] == "head_nod":
            j.update(lower_deg=lo, upper_deg=hi)
        elif j["name"] == "nod_crank":
            j.update(lower_deg=round(a_lo, 1), upper_deg=round(a_hi, 1))
        elif j["name"] == "ear_axle":
            j.update(lower_deg=round(e_lo, 1), upper_deg=round(e_hi, 1))
    return links, joints, fasteners
