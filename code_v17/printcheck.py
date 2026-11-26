"""Printability check: split every printed part into its physical pieces, then for each piece try the six
'up' directions and report bed fit (256 mm class, 250 usable) and the area that would need support.

Overhang model (voxel h): a voxel is self-supporting if any voxel in the 3x3 neighbourhood of the layer
below is filled (= 45 deg rule). Unsupported voxels are still fine when they form a bridge: each
unsupported patch whose widest inscribed half-width is <= BRIDGE/2 counts as bridgeable.
"""
import sys, importlib, numpy as np
from scipy import ndimage
import build4, geom4

BED = 250.0
BRIDGE = 30.0          # mm, PLA bridges this far without support when anchored on two sides
CANTILEVER = 3.0       # mm, a ledge anchored on one side only
SKIP_NAMES = ("bolt", "nut", "servo", "battery", "jetson", "imu", "u2d2", "dcdc", "bearing", "camera_small",
              "camera_lens", "joint_disc", "opencr", "speaker", "amp_MAX", "mic_INMP", "switch_", "dc_jack", "magnets",
              "jetson_fan", "buck_", "horn_XC430", "idler_XC430", "_idler", "servo_horn")
from build import DISC, STEEL, RUBBER, XC, XM, BATT, JET, ELEC, CAM
SKIP_RGB = {tuple(c) for c in (DISC, STEEL, RUBBER, XC, XM, BATT, JET, ELEC, CAM)}
DIRS = {"+z": (2, 1), "-z": (2, -1), "+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}


def printed_parts(links):
    for ln, L in links.items():
        for pn, ss in L["parts"].items():
            if any(k in pn for k in SKIP_NAMES):
                continue
            sol = [s for s in ss if tuple(s[2]) not in SKIP_RGB]
            if sol:
                yield ln, pn, sol


def voxelize(sol, h):
    bbs = [build4.aabb(build4.sampled(s)) for s in sol]
    lo = np.array([min(b[i][0] for b in bbs) for i in range(3)]) - h
    hi = np.array([max(b[i][1] for b in bbs) for i in range(3)]) + h
    ax = [np.arange(lo[i] + h / 2, hi[i], h) for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    m = np.zeros(len(G), bool)
    for s, b in zip(sol, bbs):
        sel = np.all((G >= [x[0] - h for x in b]) & (G <= [x[1] + h for x in b]), axis=1)
        if sel.any():
            m[sel] |= geom4.inside(s, G[sel], 0.0)
    return m.reshape(len(ax[0]), len(ax[1]), len(ax[2])), lo


def orient(V, d):
    """Return array indexed [x, y, z] with z = build direction for 'up' direction d."""
    axis, sgn = DIRS[d]
    order = [i for i in range(3) if i != axis] + [axis]
    A = np.transpose(V, order)
    if sgn < 0:
        A = A[:, :, ::-1]
    return A


def overhang(A, h):
    nz = A.shape[2]
    st = np.ones((3, 3), bool)
    need = 0
    worst = []
    for k in range(1, nz):
        cur, below = A[:, :, k], A[:, :, k - 1]
        if not cur.any():
            continue
        sup = ndimage.binary_dilation(below, st)
        un = cur & ~sup
        if not un.any():
            continue
        lab, n = ndimage.label(un, st)
        supported_cells = cur & sup
        dist = ndimage.distance_transform_edt(un) * h            # half-width of each unsupported patch
        for i in range(1, n + 1):
            comp = lab == i
            ring = ndimage.binary_dilation(comp, st) & supported_cells
            _, n_anchor = ndimage.label(ring, st)
            # distance from every cell of the patch to the nearest anchoring (supported) cell
            reach = ndimage.distance_transform_edt(~ring) * h
            cantilever_ok = reach[comp].max() <= CANTILEVER
            bridge_ok = n_anchor >= 2 and dist[comp].max() <= BRIDGE / 2
            if not (cantilever_ok or bridge_ok):
                a = comp.sum() * h * h
                need += a
                worst.append((round(k * h, 1), round(a)))
    return need, worst


def analyse(mod, h_small=0.6, h_big=1.0, only=None):
    links, _, _ = mod.build_model()
    rows = []
    for ln, pn, sol in printed_parts(links):
        if only and not any(o in pn for o in only):
            continue
        bbs = [build4.aabb(build4.sampled(s)) for s in sol]
        ext = max(max(b[i][1] for b in bbs) - min(b[i][0] for b in bbs) for i in range(3))
        h = h_small if ext < 90 else h_big
        V, lo = voxelize(sol, h)
        lab, n = ndimage.label(V, np.ones((3, 3, 3), bool))
        sizes = ndimage.sum(V, lab, range(1, n + 1))
        for ci in range(1, n + 1):
            if sizes[ci - 1] * h ** 3 < 20:          # specks from sampling
                continue
            P = lab == ci
            sl = ndimage.find_objects(P.astype(int))[0]
            P = P[sl]
            best = None
            for d in DIRS:
                A = orient(P, d)
                dims = sorted(np.array(A.shape[:2]) * h) + [A.shape[2] * h]
                fits = dims[0] <= BED and dims[1] <= BED and dims[2] <= BED
                need, worst = overhang(A, h)
                bed = A[:, :, 0].sum() * h * h
                key = (not fits, round(need, -1), -bed, dims[2])
                if best is None or key < best[0]:
                    best = (key, d, dims, need, bed, worst[:3], fits)
            vol = P.sum() * h ** 3 / 1000
            rows.append((ln, pn, ci, best[1], best[2], best[3], best[4], best[5], best[6], vol))
    return rows


if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "build5")
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    rows = analyse(mod, only=only)
    print(f"{'part':34s} {'pc':>2s} {'up':>3s} {'size (mm)':>18s} {'support mm2':>11s} {'bed mm2':>8s} {'cm3':>6s}  worst")
    for ln, pn, ci, d, dims, need, bed, worst, fits, vol in rows:
        flag = "" if fits else "  TOO BIG"
        print(f"{pn:34s} {ci:2d} {d:>3s} {'x'.join(str(round(v)) for v in dims):>18s} {need:11.0f} {bed:8.0f} {vol:6.1f}  {worst}{flag}")
