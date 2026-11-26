"""X430 servo case: mounting holes and servo frames.

* HOLES: the M2.5 tapped holes of the case, read directly from the official ROBOTIS STEP
  (references/XC-430_idle.stp, product DC11_B01_CASE_M_DUMMY, CYLINDRICAL_SURFACE r 1.05 = tap drill 2.1 for M2.5).
  Servo frame (handover 3.2): origin on the horn axis at mid case thickness, z = horn side (+z), y = long axis toward
  the horn end, case x +-14.25, y -35.25..11.25, z +-17. The assembly transforms of the STEP are identity for the case
  parts, so the STEP coordinates are already servo coordinates.
* servo_frames(links): the frame of every servo in a build model, derived from its case box (34 mm = horn axis,
  46.5 mm = long axis) and the horn cylinder next to it.

usage: python3 x430_holes.py  -> prints the table (also out19/x430_case_holes.md when run by build19)
"""
import math, os, re
import numpy as np

STEP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references", "XC-430_idle.stp")
CASE = dict(x=(-14.25, 14.25), y=(-35.25, 11.25), z=(-17.0, 17.0))


def _parse_holes(path=STEP):
    t = open(path).read()
    data = t.split("DATA;")[1].split("ENDSEC;")[0]
    D = {int(m.group(1)): " ".join(m.group(2).split()) for m in re.finditer(r"#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|$)", data, re.S)}
    refs = lambda s: [int(x) for x in re.findall(r"#(\d+)", re.sub(r"'[^']*'", "", s))]
    nums = lambda s: [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)]
    tup = lambda i: np.array(nums(D[i].split("(", 1)[1].split("(", 1)[1].split(")")[0]))
    # product name of each MANIFOLD_SOLID_BREP: brep -> shape rep -> SDR -> PDS -> PD -> PDF -> PRODUCT
    pname = {i: re.match(r"PRODUCT\(\s*'([^']*)'", v).group(1) for i, v in D.items() if v.startswith("PRODUCT(")}
    form = {i: refs(v)[0] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_FORMATION")}
    pdef = {i: pname[form[refs(v)[0]]] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION(")}
    pds = {i: pdef[refs(v)[0]] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_SHAPE") and refs(v)[0] in pdef}
    rep2name = {}
    for i, v in D.items():
        if v.startswith("SHAPE_DEFINITION_REPRESENTATION"):
            a, b = refs(v)[:2]
            if a in pds:
                rep2name[b] = pds[a]
    for i, v in D.items():
        if v.startswith("SHAPE_REPRESENTATION_RELATIONSHIP"):
            a, b = refs(v)[:2]
            if a in rep2name and b not in rep2name:
                rep2name[b] = rep2name[a]
            if b in rep2name and a not in rep2name:
                rep2name[a] = rep2name[b]
    reps = {i for i, v in D.items() if "SHAPE_REPRESENTATION" in v and not v.startswith(("SHAPE_DEFINITION", "CONTEXT")) and "RELATIONSHIP" not in v}
    out = []

    def walk(i, seen, acc):
        if i in seen:
            return
        seen.add(i); v = D[i]
        if v.startswith("CYLINDRICAL_SURFACE"):
            a = refs(v)[0]; r = nums(v.split(",")[-1])[0]; ra = refs(D[a])
            acc.append((r, tup(ra[0]), tup(ra[1]))); return
        if v.startswith(("CARTESIAN_POINT", "DIRECTION", "AXIS2_PLACEMENT", "VERTEX_POINT")):
            return
        for j in refs(v):
            if j not in reps:
                walk(j, seen, acc)
    for r in reps:
        if rep2name.get(r) != "DC11_B01_CASE_M_DUMMY":
            continue
        for it in refs(D[r])[:-1]:
            if D[it].startswith(("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS")):
                walk(it, set(), out)
    holes = {}
    for r, o, z in out:
        if abs(r - 1.05) > 1e-6:
            continue
        o = np.round(o, 3)
        # outward normal: the case face the hole opens on
        k = int(np.argmax(np.abs(z)))
        n = np.zeros(3); n[k] = math.copysign(1.0, o[k])
        holes[tuple(o)] = tuple(n)
    return [(np.array(p), np.array(n)) for p, n in sorted(holes.items())]


HOLES = _parse_holes()
TAP_D = 2.1


def table():
    face = {(1, 0, 0): "side +x", (-1, 0, 0): "side -x", (0, -1, 0): "back -y (cable end)", (0, 1, 0): "front +y (horn end)"}
    rows = ["| # | x | y | z | face (outward normal) |", "|---|---|---|---|---|"]
    for i, (p, n) in enumerate(HOLES):
        rows.append(f"| {i + 1} | {p[0]:+.2f} | {p[1]:+.2f} | {p[2]:+.2f} | {face[tuple(int(v) for v in n)]} |")
    return "\n".join(rows)


def servo_frames(links):
    """{part name: dict(link, O, X, Y, Z, model)} for every servo case box of a build model (world mm)."""
    import build4
    horns = []
    for ln, L in links.items():
        for pn, ss in L["parts"].items():
            if "horn" in pn and "adapter" not in pn:
                for s in ss:
                    if s[0] == "cyl":
                        c, a, r, h = s[1]; a = np.array(a, float) / np.linalg.norm(a)
                        horns.append((np.array(c, float) + a * h / 2, a))
    out = {}
    for ln, L in links.items():
        for pn, ss in L["parts"].items():
            m = re.search(r"(XM430|XC430|XL430)$", pn)
            if not m or "horn" in pn or "idler" in pn:
                continue
            bb = np.array(build4.aabb(build4.sampled(ss[0])), float)
            size = bb[:, 1] - bb[:, 0]; ctr = bb.mean(1)
            kz = int(np.argmin(np.abs(size - 34.0))); ky = int(np.argmin(np.abs(size - 46.5)))
            # nearest horn: on the case's horn axis side
            hc, ha = min(horns, key=lambda h: np.linalg.norm(h[0] - ctr))
            Z = np.zeros(3); Z[kz] = math.copysign(1.0, hc[kz] - ctr[kz])
            Y = np.zeros(3)
            # horn axis sits 11.25 from the front face along the long axis
            front_hi = abs(bb[ky, 1] - hc[ky]) < abs(bb[ky, 0] - hc[ky])
            Y[ky] = 1.0 if front_hi else -1.0
            X = np.cross(Y, Z)
            O = ctr.copy(); O[ky] = hc[ky]
            out[pn] = dict(link=ln, O=O, X=X, Y=Y, Z=Z, model=m.group(1),
                           check=float(abs((bb[ky, 1] if front_hi else bb[ky, 0]) - hc[ky]) - 11.25))
    return out


def to_world(fr, p, n=None):
    R = np.c_[fr["X"], fr["Y"], fr["Z"]]
    pw = fr["O"] + R @ np.asarray(p, float)
    return (pw, R @ np.asarray(n, float)) if n is not None else pw


if __name__ == "__main__":
    print(f"{len(HOLES)} M2.5 tapped holes (r 1.05) in {os.path.basename(STEP)} DC11_B01_CASE_M_DUMMY, servo frame")
    print(table())
