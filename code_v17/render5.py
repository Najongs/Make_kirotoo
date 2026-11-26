"""v5 preview: small z-buffer rasteriser (orthographic) with flat 'toon' shading and outlines drawn where
the part or the depth changes, so the mascot styling reads like the reference art."""
import math, sys, numpy as np
import shapes; shapes.ARC_STEPS = 10
import build5
from build4 import sampled
from build import IDX, UVI
from geom4 import rot
from PIL import Image

links, joints, _ = build5.build_model()
J = {j["name"]: j for j in joints}
OUTLINE = np.array([0.04, 0.12, 0.26])


# ---------------- tessellation
def ear_clip(poly):
    pts = [tuple(p) for p in poly]
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    a = shapes.area(pts)
    idx = list(range(len(pts))) if a > 0 else list(range(len(pts)))[::-1]
    tris = []
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    guard = 0
    while len(idx) > 3 and guard < 20000:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            o, p, q = pts[i0], pts[i1], pts[i2]
            if cross(o, p, q) <= 1e-12:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                r = pts[j]
                if cross(o, p, r) >= 0 and cross(p, q, r) >= 0 and cross(q, o, r) >= 0:
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return pts, tris


def P3(axis, u, v, a):
    k, (iu, iv) = IDX[axis], UVI[axis]
    p = [0.0, 0.0, 0.0]; p[iu] = u; p[iv] = v; p[k] = a
    return p


def tris_of(s):
    s = sampled(s)
    kind, args, rgb = s
    T = []
    if kind == "prism":
        axis, a0, a1, poly = args[:4]
        pts, tr = ear_clip(poly)
        n = len(pts)
        for i, j, k in tr:
            T.append([P3(axis, *pts[i], a0), P3(axis, *pts[k], a0), P3(axis, *pts[j], a0)])
            T.append([P3(axis, *pts[i], a1), P3(axis, *pts[j], a1), P3(axis, *pts[k], a1)])
        for i in range(n):
            p, q = pts[i], pts[(i + 1) % n]
            A, B, C, D = P3(axis, *p, a0), P3(axis, *q, a0), P3(axis, *q, a1), P3(axis, *p, a1)
            T += [[A, B, C], [A, C, D]]
    elif kind == "cyl":
        c, a, r, h = args
        a = np.array(a, float); a /= np.linalg.norm(a); c = np.array(c, float)
        ref = np.array([0, 0, 1.]) if abs(a[2]) < .9 else np.array([1., 0, 0]); ref -= ref.dot(a) * a; ref /= np.linalg.norm(ref)
        bb = np.cross(a, ref); n = 32
        ring = [c + r * (math.cos(t) * ref + math.sin(t) * bb) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        top = [p + a * h for p in ring]
        for i in range(n):
            j = (i + 1) % n
            T += [[c, ring[j], ring[i]], [c + a * h, top[i], top[j]], [ring[i], ring[j], top[j]], [ring[i], top[j], top[i]]]
    elif kind == "scap":
        c, a, ro, ri, d = args
        a = np.array(a, float); a /= np.linalg.norm(a); c = np.array(c, float)
        ref = np.array([0, 0, 1.]) if abs(a[2]) < .9 else np.array([1., 0, 0]); ref -= ref.dot(a) * a; ref /= np.linalg.norm(ref)
        bb = np.cross(a, ref); nu, nv = 40, 16
        phis = np.linspace(0, math.acos(d / ro), nv + 1); th = np.linspace(0, 2 * math.pi, nu + 1)
        Pp = lambda ph, t: c + ro * (math.cos(ph) * a + math.sin(ph) * (math.cos(t) * ref + math.sin(t) * bb))
        for i in range(nv):
            for j in range(nu):
                A, B, C, D = Pp(phis[i], th[j]), Pp(phis[i + 1], th[j]), Pp(phis[i + 1], th[j + 1]), Pp(phis[i], th[j + 1])
                T += [[A, B, C], [A, C, D]]
    return np.array(T, float), rgb


# ---------------- pose
CHAINS = {"torso": []}
for S in "RL":
    names = [f"{S}_hip_yaw", f"{S}_hip_pitch", f"{S}_knee", f"{S}_ankle_pitch", f"{S}_ankle_roll"]
    for i, ln in enumerate([f"{S}_hip_yaw_link", f"{S}_thigh", f"{S}_shank", f"{S}_ankle_link", f"{S}_foot"]):
        CHAINS[ln] = names[:i + 1]
CHAINS["neck"] = ["head_pan"]; CHAINS["head"] = ["head_pan", "head_tilt"]


def link_pose(ln, ang):
    M = np.eye(4)
    for n in CHAINS[ln]:
        M = M @ rot(J[n]["axis"], ang.get(n, 0.0), J[n]["point_mm"])
    return M


from build import DISC
GEOM = []
pid = 0
for ln, L in links.items():
    for pn, sol in L["parts"].items():
        pid += 1                                     # one id per part: no outlines between its own solids
        for s in sol:
            if s[0] == "cyl" and tuple(s[2]) == tuple(DISC):
                continue                             # joint discs are modelling aids, not parts to show
            T, rgb = tris_of(s)
            GEOM.append((ln, pn, pid, T, np.array(rgb)))


# ---------------- raster
def render(elev, azim, ang=None, skip=(), scale=2.2, pad=20):
    ang = ang or {}
    e, z = math.radians(elev), math.radians(azim)
    view = np.array([math.cos(e) * math.cos(z), math.cos(e) * math.sin(z), math.sin(e)])   # toward the camera
    right = np.array([-math.sin(z), math.cos(z), 0.0])
    up = np.cross(view, right)
    light = 0.55 * view + 0.45 * np.array([0.3, -0.4, 0.85]); light /= np.linalg.norm(light)
    allT = []
    for ln, pn, pid_, T, rgb in GEOM:
        if any(k in pn for k in skip):
            continue
        M = link_pose(ln, ang)
        W = T @ M[:3, :3].T + M[:3, 3]
        allT.append((W, rgb, pid_))
    pts = np.vstack([W.reshape(-1, 3) for W, _, _ in allT])
    U, V = pts @ right, pts @ up
    u0, u1, v0, v1 = U.min(), U.max(), V.min(), V.max()
    w, h = int((u1 - u0) * scale) + 2 * pad, int((v1 - v0) * scale) + 2 * pad
    zbuf = np.full((h, w), -1e9); col = np.ones((h, w, 3)); ids = np.zeros((h, w), int)
    for W, rgb, pid_ in allT:
        n = np.cross(W[:, 1] - W[:, 0], W[:, 2] - W[:, 0])
        nn = np.linalg.norm(n, axis=1); good = nn > 1e-9
        n[good] /= nn[good, None]
        lam = np.abs(n @ light)
        shade = 0.62 + 0.38 * lam
        su = (W @ right - u0) * scale + pad; sv = (v1 - W @ up) * scale + pad; sd = W @ view
        for t in np.nonzero(good)[0]:
            x, y, d = su[t], sv[t], sd[t]
            xa, xb = max(int(math.floor(x.min())), 0), min(int(math.ceil(x.max())), w - 1)
            ya, yb = max(int(math.floor(y.min())), 0), min(int(math.ceil(y.max())), h - 1)
            if xa > xb or ya > yb:
                continue
            gx, gy = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
            den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(den) < 1e-12:
                continue
            l0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
            l1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
            l2 = 1 - l0 - l1
            m = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not m.any():
                continue
            dep = l0 * d[0] + l1 * d[1] + l2 * d[2]
            sub = zbuf[ya:yb + 1, xa:xb + 1]
            upd = m & (dep > sub)
            if upd.any():
                sub[upd] = dep[upd]
                col[ya:yb + 1, xa:xb + 1][upd] = np.clip(rgb * shade[t], 0, 1)
                ids[ya:yb + 1, xa:xb + 1][upd] = pid_
    # outlines: part boundaries and depth jumps
    edge = np.zeros((h, w), bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = ids[:h - dy, :w - dx], ids[dy:, dx:]
        za, zb = zbuf[:h - dy, :w - dx], zbuf[dy:, dx:]
        e_ = (a != b) | ((np.abs(za - zb) > 3.0) & (a > 0))
        edge[:h - dy, :w - dx] |= e_
        edge[dy:, dx:] |= e_
    col[edge] = col[edge] * 0.15 + OUTLINE * 0.85
    return (col * 255).astype(np.uint8)


def panel(views, path, gap=30):
    imgs = [Image.fromarray(v) for v in views]
    H = max(i.height for i in imgs); Wt = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (Wt, H), "white")
    x = 0
    for im in imgs:
        out.paste(im, (x, H - im.height)); x += im.width + gap
    out.save(path)


if __name__ == "__main__":
    walk = {"head_pan": 25, "head_tilt": 12, "R_hip_pitch": -28, "R_knee": 40, "R_ankle_pitch": -12,
            "L_hip_pitch": 12, "L_knee": 8, "L_ankle_pitch": -20}
    views = [render(15, -35), render(0, 0), render(0, -90), render(12, -55, walk)]
    panel(views, "/home/claude/biped/out5/biped_v5_preview.png")
    print("saved")
