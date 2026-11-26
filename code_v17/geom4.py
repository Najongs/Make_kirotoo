"""Point-in-solid tests and volume sampling for every solid kind used in v4 (box-free: cyl, prism, rprism, scap).
Used for interference checks that the 2D exact tests cannot cover (spheres, tilted cylinders, rotated poses)."""
import math
import numpy as np
from matplotlib.path import Path
from build import IDX, UVI
import build4


def _unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def inside(s, P, margin=0.0):
    """Boolean mask: points P (N,3) strictly inside solid s by at least `margin` mm (approximate for prisms)."""
    s = build4.sampled(s)
    kind, args, _ = s
    P = np.asarray(P, float)
    if kind == "loft":
        import loft; return loft.inside(args, P, margin)
    if kind == "cyl":
        c, a, r, h = args
        a = _unit(a); d = P - np.asarray(c, float); t = d @ a
        rad = np.linalg.norm(d - np.outer(t, a), axis=1)
        return (t > margin) & (t < h - margin) & (rad < r - margin)
    if kind == "scap":
        c, a, ro, ri, dd = args
        a = _unit(a); d = P - np.asarray(c, float); n = np.linalg.norm(d, axis=1)
        return (n > ri + margin) & (n < ro - margin) & (d @ a > dd + margin)
    axis, a0, a1, poly, holes = args[:5]
    ipolys = args[5] if kind == "prismh" else []
    k, (iu, iv) = IDX[axis], UVI[axis]
    lo, hi = min(a0, a1), max(a0, a1)
    m = (P[:, k] > lo + margin) & (P[:, k] < hi - margin)
    uv = P[:, [iu, iv]]
    path = Path(np.asarray(poly))
    # a negative radius shrinks the path for a ccw polygon; use both signs and keep the stricter result
    r = -margin if margin else 0.0
    m &= path.contains_points(uv, radius=r) & path.contains_points(uv, radius=-r)
    for u, v, hr in holes:
        m &= np.hypot(uv[:, 0] - u, uv[:, 1] - v) > hr + margin
    for ip in ipolys:
        hp = Path(np.asarray(ip))
        m &= ~(hp.contains_points(uv, radius=margin) | hp.contains_points(uv, radius=-margin))
    return m


def samples(s, step=0.7):
    """Grid points inside solid s (spacing `step`)."""
    b = build4.aabb(build4.sampled(s))
    ax = [np.arange(lo + step / 2, hi, step) for lo, hi in b]
    if any(len(x) == 0 for x in ax):
        ax = [np.array([(lo + hi) / 2]) if len(x) == 0 else x for x, (lo, hi) in zip(ax, b)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return G[inside(s, G)]


def rot(axis, deg, about):
    """4x4 rotation about a world axis line through `about`."""
    t = math.radians(deg); c, s_ = math.cos(t), math.sin(t)
    R = {"x": [[1, 0, 0], [0, c, -s_], [0, s_, c]],
         "y": [[c, 0, s_], [0, 1, 0], [-s_, 0, c]],
         "z": [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]}[axis]
    M = np.eye(4); M[:3, :3] = R
    T = np.eye(4); T[:3, 3] = about
    Ti = np.eye(4); Ti[:3, 3] = -np.asarray(about, float)
    return T @ M @ Ti


def apply(M, P):
    return P @ M[:3, :3].T + M[:3, 3]
