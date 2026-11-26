import numpy as np, matplotlib, math
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shapes; shapes.ARC_STEPS = 10
from build4 import build_model, sampled
from build import IDX, UVI
links, joints, _ = build_model()
LIGHT = np.array([0.45, -0.35, 0.82]); LIGHT /= np.linalg.norm(LIGHT)

def prism_faces(axis, a0, a1, poly):
    k, (iu, iv) = IDX[axis], UVI[axis]
    def P(u, v, a):
        p = [0, 0, 0]; p[iu] = u; p[iv] = v; p[k] = a; return p
    b = np.array([P(u, v, a0) for u, v in poly]); t = np.array([P(u, v, a1) for u, v in poly]); n = len(poly)
    return [b[::-1], t] + [np.array([b[i], b[(i + 1) % n], t[(i + 1) % n], t[i]]) for i in range(n)]

def cyl_faces(c, a, r, h, n=28):
    a = np.array(a, float); c = np.array(c, float)
    ref = np.array([0, 0, 1.]) if abs(a[2]) < .9 else np.array([1., 0, 0]); ref -= ref.dot(a) * a; ref /= np.linalg.norm(ref)
    bb = np.cross(a, ref); th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.array([c + r * (np.cos(t) * ref + np.sin(t) * bb) for t in th]); top = ring + a * h
    return [ring[::-1], top] + [np.array([ring[i], ring[(i + 1) % n], top[(i + 1) % n], top[i]]) for i in range(n)]

def scap_faces(c, a, ro, ri, d, nu=36, nv=14):
    a = np.array(a, float); a /= np.linalg.norm(a); c = np.array(c, float)
    ref = np.array([0, 0, 1.]) if abs(a[2]) < .9 else np.array([1., 0, 0]); ref -= ref.dot(a) * a; ref /= np.linalg.norm(ref)
    bb = np.cross(a, ref); th = np.linspace(0, 2 * np.pi, nu + 1)
    fs = []
    R = ro
    phis = np.linspace(0, np.arccos(d / R), nv + 1)          # polar angle from the pole down to the cut
    P = lambda ph, t: c + R * (np.cos(ph) * a + np.sin(ph) * (np.cos(t) * ref + np.sin(t) * bb))
    for i in range(nv):
        for j in range(nu):
            fs.append(np.array([P(phis[i], th[j]), P(phis[i + 1], th[j]), P(phis[i + 1], th[j + 1]), P(phis[i], th[j + 1])]))
    return fs

def shade(f, rgb):
    n = np.zeros(3)
    for i in range(len(f)):
        a, b = f[i], f[(i + 1) % len(f)]
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    nn = np.linalg.norm(n)
    lam = abs(n.dot(LIGHT)) / nn if nn > 0 else 0.5
    return tuple(min(1, c * (0.42 + 0.62 * lam)) for c in rgb)

def collect(only=None, skip=()):
    faces, cols = [], []
    for ln, L in links.items():
        if only and ln not in only: continue
        for pn, sol in L["parts"].items():
            if any(s in pn for s in skip): continue
            for s in sol:
                s = sampled(s)
                fs = prism_faces(*s[1][:4]) if s[0] == "prism" else scap_faces(*s[1]) if s[0] == "scap" else cyl_faces(*s[1])
                for f in fs:
                    faces.append(f); cols.append(shade(f, s[2]))
    return faces, cols

def draw(ax, elev, azim, title, lim=None, **kw):
    faces, cols = collect(**kw)
    ax.add_collection3d(Poly3DCollection(faces, facecolors=cols, edgecolors=(0, 0, 0, 0.12), linewidths=0.15))
    (x0, x1), (y0, y1), (z0, z1) = lim or ((-120, 120), (-120, 120), (0, 440))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0)); ax.view_init(elev, azim)
    ax.set_proj_type("ortho"); ax.set_axis_off(); ax.set_title(title, fontsize=12, pad=0)

fig = plt.figure(figsize=(18, 8), facecolor="white")
HID = ("torso_side", "jetson", "battery", "u2d2", "dcdc", "imu", "shelf", "yaw_servo", "bearing", "pelvis")
draw(fig.add_subplot(141, projection="3d"), 18, -38, "Isometric", skip=HID)
draw(fig.add_subplot(142, projection="3d"), 0, 0, "Front", skip=HID)
draw(fig.add_subplot(143, projection="3d"), 0, -90, "Side", skip=HID)
draw(fig.add_subplot(144, projection="3d"), 22, -45, "Covers hidden (frame inside)", skip=("cover",))
plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0)
plt.savefig("out4/biped_v4_preview.png", dpi=120, facecolor="white")
