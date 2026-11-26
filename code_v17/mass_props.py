"""Mass model: per-link mass, CoM and inertia (link-local frame = link origin, world axes at zero pose).

Printed parts: voxelised, wall band (distance to surface <= wall) solid PLA, interior at infill fraction.
Purchased parts: catalogue mass spread uniformly over their modelled envelope.

usage: python3 mass_props.py build14 [profile]   profile = v14 | light
"""
import sys, importlib, json, re, numpy as np
from scipy import ndimage
import printcheck, build4

RHO_PLA = 1.24e-3          # g/mm3
RHO_STEEL = 7.85e-3
RHO_RUBBER = 1.1e-3
H = 0.7                    # voxel mm

# catalogue masses (g); +8 g per servo for horn, idler, screws and its share of cable
SERVO_G = {"XM430": 82.0 + 8, "XC430": 65.0 + 8, "XL430": 57.2 + 8}
FIXED_G = {"jetson_nano_2gb": 140.0, "jetson_fan": 12.0, "buck_5V": 12.0, "switch_main": 6.0, "switch_jetson": 3.0,
           "dc_jack": 4.0, "magnets": 6.0, "jetson": 176.0, "battery": 106.0, "opencr": 60.0, "imu": 3.0, "dcdc": 15.0, "camera_pi": 5.0,
           "6805": 18.0, "688": 3.6}

STRUCT = ("frame", "bracket", "shelf", "yoke", "lever", "crank", "rod", "adapter", "foot_sole", "pins", "clip",
          "hatch", "rocker")

PROFILES = {
    # name: (structural wall mm, structural infill, cosmetic wall mm, cosmetic infill)
    "v14": (1.2, 0.30, 1.2, 0.30),
    "light": (1.2, 0.20, 0.8, 0.10),
}


def classify(pn):
    if "hatch" in pn and "magnet" not in pn:
        return None, None
    if "horn" in pn and "adapter" not in pn:
        return "fixed", 4.0
    if "idler" in pn:
        return "fixed", 3.0
    for k, g in (("battery_jetson", 85.0), ("opencr_ttl", 8.0), ("speaker", 30.0), ("amp_", 3.0), ("mic_", 2.0)):
        if k in pn:
            return "fixed", g
    for k, g in SERVO_G.items():
        if k in pn:
            return "servo", g
    for k, g in FIXED_G.items():
        if k in pn:
            return "fixed", g
    return None, None


def vox(sol, h=H):
    V, lo = printcheck.voxelize(sol, h)
    return V, lo


def props_from_vox(W, lo, h):
    """W: weights per voxel (g). returns mass, com, inertia about com (g mm2)"""
    idx = np.nonzero(W)
    w = W[idx]
    P = np.stack(idx, 1) * h + lo + h / 2
    m = w.sum()
    c = (P * w[:, None]).sum(0) / m
    d = P - c
    I = np.zeros((3, 3))
    r2 = (d ** 2).sum(1)
    for i in range(3):
        for j in range(3):
            I[i, j] = (w * ((r2 if i == j else 0) - d[:, i] * d[:, j])).sum()
    I += np.eye(3) * m * h * h / 6      # voxel self inertia
    return m, c, I


def part_props(pn, sol, prof):
    from build import STEEL, RUBBER
    kind, g = classify(pn)
    V, lo = vox(sol)
    if not V.any():
        return None
    if kind:
        W = V * (g / V.sum())
        return dict(mass=g, kind=kind), W, lo
    steel = all(tuple(s[2]) == tuple(STEEL) for s in sol)
    rubber = all(tuple(s[2]) == tuple(RUBBER) for s in sol)
    if steel or rubber or "bearing" in pn:
        rho = RHO_STEEL if (steel or "bearing" in pn) else RHO_RUBBER
        W = V * rho * H ** 3
        return dict(mass=W.sum(), kind="steel" if rho == RHO_STEEL else "rubber"), W, lo
    sw, si, cw, ci = prof
    struct = any(k in pn for k in STRUCT)
    wall, inf = (sw, si) if struct else (cw, ci)
    D = ndimage.distance_transform_edt(np.pad(V, 1)) [1:-1, 1:-1, 1:-1] * H
    shell = V & (D <= wall + 0.4 + H / 2)    # +0.4: gap fill / infill overlap; thin walls print solid
    W = (shell * 1.0 + (V & ~shell) * inf) * RHO_PLA * H ** 3
    return dict(mass=W.sum(), kind="print", solid_cm3=V.sum() * H ** 3 / 1000, struct=struct), W, lo


def link_props(mod, profile="v14", only_links=None):
    links, joints, _ = mod.build_model()
    prof = PROFILES[profile]
    out, parts = {}, []
    for ln, L in links.items():
        if only_links and ln not in only_links:
            continue
        acc = []
        for pn, sol in L["parts"].items():
            r = part_props(pn, sol, prof)
            if r is None:
                continue
            info, W, lo = r
            m, c, I = props_from_vox(W, lo, H)
            info.update(link=ln, part=pn, mass=round(float(m), 1), com=[round(float(x), 1) for x in c])
            parts.append(info)
            acc.append((m, c, I))
        M = sum(a[0] for a in acc)
        C = sum(a[0] * a[1] for a in acc) / M
        I = np.zeros((3, 3))
        for m, c, Ii in acc:
            d = c - C
            I += Ii + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        o = np.array(L["origin"], float)
        out[ln] = dict(origin=list(o), mass_g=float(M), com_world=[float(x) for x in C],
                       com_local=[float(x) for x in C - o], inertia_gmm2=I.tolist())
    return out, parts, joints


if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1])
    profile = sys.argv[2] if len(sys.argv) > 2 else "v14"
    out, parts, joints = link_props(mod, profile)
    tot = sum(v["mass_g"] for v in out.values())
    com = sum(np.array(v["com_world"]) * v["mass_g"] for v in out.values()) / tot
    print(f"total {tot:.0f} g  CoM {np.round(com, 1)}")
    for ln, v in out.items():
        print(f"  {ln:16s} {v['mass_g']:7.1f} g  com {np.round(v['com_world'], 1)}")
    by = {}
    for p in parts:
        by[p["kind"]] = by.get(p["kind"], 0) + p["mass"]
    print({k: round(v) for k, v in by.items()})
    for p in sorted(parts, key=lambda p: -p["mass"])[:25]:
        print(f"    {p['part']:28s} {p['kind']:6s} {p['mass']:6.1f}")
    json.dump(dict(links=out, parts=parts, total_g=tot, com=list(com)),
              open(f"mass_{sys.argv[1]}_{profile}.json", "w"), indent=1)
