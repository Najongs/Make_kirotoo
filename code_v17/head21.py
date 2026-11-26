"""v21 head nod: can the nod crank reach about -20 .. +28 deg (owner 2026-11-26), and what has to change?

Result (see out_head21/HEAD_CRANK_v21.md): the v20 linkage already does it. Crank R 14, anchor (-14.75, 345), rod
38.45 stay; only the joint limits widen (-10 .. +20 -> -20 .. +28). The first contact is the head shell on the lid
(-25.5 / +32.5), not the crank loop.

This file
  * closed-form loop: crank angle, transmission angles (crank vs rod, rod vs output link P->A), ear angle
  * servo torque by virtual work (head + ears + crank, link_mass_v20.csv), plus a 1 Hz nod inertia term
  * analytic clearances: rod in the head floor slot, yoke anchor arm under the head floor, crank pin over the pad
  * search over crank radius / anchor position (alternatives, kept for the table; none adopted)
  * --sweep: sweep17-style point-sampling collision sweep (head, crank, rod, ears, ear rod vs torso / head, and
    both arms vs head + ears) on the v20 model -> out_head21/sweep_head21.json (about 15 min)
  * default run: tables -> out_head21/head21.json and the figure out_head21/head_crank_v21.png

Angles: rot() convention of geom4 (+deg about +y turns +x toward -z), beta = head_nod, alpha = nod_crank, psi = ear_axle,
beta = 0 <-> alpha = 0 <-> psi = 0.
"""
import os, sys, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_head21")
sys.path.insert(0, HERE)
import build14 as B14
import build16 as B16

# ---------------------------------------------------------------- geometry (v20 values, unchanged in v21)
P = np.array([0.0, B14.ZT9])               # head nod axis (x, z)
S = np.array(B14.NOD_S)                    # nod servo horn axis in the head
NOD_R, NOD_A = B14.NOD_R, tuple(B14.NOD_A) # crank radius (pointing +x at rest), rod anchor on the yoke
E0 = np.array(B16.EAR_C)                   # ear axle
EAR_R1, EAR_A1, EAR_R2 = B16.EAR_R1, B16.EAR_A1, B16.EAR_R2

# ---------------------------------------------------------------- v21 proposal
OLD_RANGE = (-10.0, 20.0)                  # v14 .. v20 joint table
NEW_RANGE = (-20.0, 28.0)                  # v21 joint table (owner target about -20 .. +28)

# head shell / floor / pad (build14.head, build17.fix_nod)
HB = B14.ZH - B14.HH                       # head floor underside z 365
BOT_IN = HB + 1.2                          # floor top 366.2
PAD_TOP, PAD_X, PAD_YMAX = BOT_IN + 3.0, (-41.0, 2.0), 18.0
SLOT_X = (-30.0, 4.0)                      # nod rod slot in the head floor (y 19.5 .. 23.3)

# masses (out20/link_mass_v20.csv: kg, CoM from link origin in mm, Iyy about CoM kg m^2)
M_HEAD, C_HEAD, IYY_HEAD = 0.3906, (2.92, 86.66), 2.080e-3      # origin = nod axis
M_EARS, C_EARS, IYY_EARS = 0.0644, (-0.07, 4.18), 2.473e-5      # origin = ear axle
M_CRANK, C_CRANK = 0.0058, (-0.05, 0.67)                        # origin = horn axis
STALL = 1.4                # XL430-W250 stall torque, N m (ROBOTIS e-manual, 11.1 V; sim/export_mjcf.py uses 1.4)
TICK = 4096 / 360.0        # XL430 position resolution (e-manual: 0 .. 4095 = 360 deg)
NOD_HZ = 1.0               # nod frequency used in the walking test (STATUS 2026-11-26)
G = 9.81


def rot2(v, deg, c):
    a = math.radians(deg); d = np.asarray(v, float) - c
    return c + np.array([d[0] * math.cos(a) + d[1] * math.sin(a), -d[0] * math.sin(a) + d[1] * math.cos(a)])


def _ang(u, w):
    return math.degrees(math.acos(np.clip(np.dot(u, w) / np.linalg.norm(u) / np.linalg.norm(w), -1, 1)))


def nod(beta, R=None, A=None):
    """crank angle for head angle beta on the assembled branch; None past a toggle.
    Returns dict(alpha, pin (head frame), gamma = crank vs rod at the pin, mu = rod vs output link at the anchor)."""
    R = NOD_R if R is None else R
    A = np.array(NOD_A if A is None else A, float)
    pin0 = S + np.array([R, 0.0]); L = np.linalg.norm(pin0 - A)
    Ah = rot2(A, -beta, P)                     # anchor seen from the head
    d = Ah - S; dd = np.linalg.norm(d)
    c = (R * R + dd * dd - L * L) / (2 * R * dd)
    if abs(c) > 1:
        return None
    psi0 = math.atan2((A - S)[1], (A - S)[0]); dl0 = math.acos((R * R + np.linalg.norm(A - S) ** 2 - L * L) / (2 * R * np.linalg.norm(A - S)))
    sg = 1 if abs(math.remainder(psi0 + dl0, 2 * math.pi)) < 1e-6 else -1      # branch through alpha = 0
    tip = math.atan2(d[1], d[0]) + sg * math.acos(c)
    alpha = math.degrees(math.remainder(-tip, 2 * math.pi))
    pin = S + R * np.array([math.cos(tip), math.sin(tip)])
    return dict(alpha=alpha, pin=pin, Ah=Ah, L=L, gamma=_ang(S - pin, Ah - pin), mu=_ang(pin - Ah, P - Ah))


def ear_geom(R1=None, R2=None):
    R1 = EAR_R1 if R1 is None else R1; R2 = EAR_R2 if R2 is None else R2
    a = math.radians(EAR_A1)
    P1 = S + R1 * np.array([math.cos(a), math.sin(a)]); P2 = E0 + R2 * np.array([math.cos(a), math.sin(a)])
    return P1, P2, np.linalg.norm(P2 - P1)


def ear(alpha, R1=None, R2=None):
    """ear angle psi for crank angle alpha (branch through psi = 0) and the rod-rocker angle; None past a toggle."""
    R2_ = EAR_R2 if R2 is None else R2
    P1, P2, LE = ear_geom(R1, R2)
    q1 = rot2(P1, alpha, S); d = q1 - E0; dd = np.linalg.norm(d)
    c = (R2_ * R2_ + dd * dd - LE * LE) / (2 * R2_ * dd)
    if abs(c) > 1:
        return None
    t0 = math.radians(EAR_A1)
    cands = [math.atan2(d[1], d[0]) + s * math.acos(c) for s in (1, -1)]
    # the assembled branch: the one that gives the rocker pin P2 at alpha 0
    d0 = P1 - E0; c0 = (R2_ * R2_ + d0 @ d0 - LE * LE) / (2 * R2_ * np.linalg.norm(d0))
    s0 = min((1, -1), key=lambda s: abs(math.remainder(math.atan2(d0[1], d0[0]) + s * math.acos(c0) - t0, 2 * math.pi)))
    t = cands[0 if s0 == 1 else 1]
    q2 = E0 + R2_ * np.array([math.cos(t), math.sin(t)])
    return dict(psi=math.degrees(math.remainder(t0 - t, 2 * math.pi)), rocker=_ang(q1 - q2, E0 - q2))


def potential(alpha, beta, psi):
    """gravity potential (J) of head + ears + crank; head frame turned by beta about P, ears by psi about E0, crank by alpha about S"""
    ch = rot2(P + np.array(C_HEAD), beta, P)
    ce = rot2(rot2(E0 + np.array(C_EARS), psi, E0), beta, P)
    cc = rot2(rot2(S + np.array(C_CRANK), alpha, S), beta, P)
    return G * 1e-3 * (M_HEAD * ch[1] + M_EARS * ce[1] + M_CRANK * cc[1])


def table(betas, R=None, A=None, ears=True):
    rows = []
    for b in betas:
        n = nod(b, R, A)
        if n is None:
            rows.append(dict(beta=float(b), ok=False)); continue
        r = dict(beta=float(b), ok=True, alpha=n["alpha"], gamma=n["gamma"], mu=n["mu"])
        if ears:
            e = ear(n["alpha"])
            r.update(psi=e["psi"] if e else None, rocker=e["rocker"] if e else None)
        rows.append(r)
    return rows


def torque(beta, h=0.25):
    """servo torque (N m) holding the head at beta against gravity: -dU/dalpha (virtual work), and dbeta/dalpha"""
    def U(b):
        a = nod(b)["alpha"]; return a, potential(a, b, ear(a)["psi"])
    a0, u0 = U(beta - h); a1, u1 = U(beta + h)
    da = math.radians(a1 - a0)
    return -(u1 - u0) / da, (2 * h) / (a1 - a0)


def inertia_about_axis():
    dh = np.hypot(*C_HEAD) * 1e-3
    de = np.hypot(*(E0 + np.array(C_EARS) - P)) * 1e-3
    return IYY_HEAD + M_HEAD * dh ** 2 + IYY_EARS + M_EARS * de ** 2


# ---------------------------------------------------------------- analytic clearances (mm, head frame)
def clearances(betas, R=None, A=None):
    R = NOD_R if R is None else R
    A = np.array(NOD_A if A is None else A, float)
    ax, az = A
    arm = [(ax - 4.5, az + 5), (ax + 4.5, az + 4.5), (-9.9, az + 5)] + \
          [(ax + 3.5 * math.cos(t), az + 3.5 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 25)]
    out = []
    for b in betas:
        n = nod(b, R, A)
        # yoke anchor arm + rod end around the anchor pin under the head floor
        top = max(rot2(np.array(c), -b, P)[1] for c in arm)
        # rod (3 mm thick plate, half width 3.5 about its centre line) through the floor slot
        u = n["pin"] - n["Ah"]; u = u / np.linalg.norm(u)
        xs = []
        for z in (HB, BOT_IN):
            t = (z - n["Ah"][1]) / u[1]; xc = n["Ah"][0] + t * u[0]
            w = 3.5 / abs(u[1])                     # half width along x of a tilted bar
            xs += [xc - w, xc + w]
        # crank pin boss (r 4) over the floor pad, rod end (r 3.5) over the floor
        pin = n["pin"]
        over_pad = PAD_X[0] - 4 <= pin[0] <= PAD_X[1] + 4
        out.append(dict(beta=float(b), anchor_to_floor=HB - top, slot_back=min(xs) - SLOT_X[0], slot_front=SLOT_X[1] - max(xs),
                        pin_to_pad=(pin[1] - 4 - PAD_TOP) if over_pad else None, rodend_to_floor=pin[1] - 3.5 - BOT_IN))
    return out


# ---------------------------------------------------------------- alternatives (crank radius x anchor position)
def search(betas=np.arange(NEW_RANGE[0], NEW_RANGE[1] + 0.01, 1.0)):
    res = []
    for R in np.arange(10, 18.01, 1.0):
        for ax in np.arange(-40, 6, 1.0):
            for az in np.arange(343.5, 352.1, 0.5):
                if math.hypot(ax, az - P[1]) < 9:       # shaft r 4 + rod end r 3.5 + gap
                    continue
                rows = table(betas, R, (ax, az), ears=False)
                if not all(r["ok"] for r in rows):
                    continue
                al = np.array([r["alpha"] for r in rows])
                if not (np.all(np.diff(al) > 0) or np.all(np.diff(al) < 0)):
                    continue
                cl = clearances(list(betas[::4]) + [betas[-1]], R, (ax, az))
                floor = min(c["anchor_to_floor"] for c in cl)
                if floor < 1.0:
                    continue
                g = min(min(r["gamma"], 180 - r["gamma"]) for r in rows); m = min(min(r["mu"], 180 - r["mu"]) for r in rows)
                ratio = float(np.abs(np.gradient(betas, al)).max())
                res.append(dict(R=float(R), A=(float(ax), float(az)), L=float(nod(0, R, (ax, az))["L"]), trans_min=float(min(g, m)),
                                gamma_min=float(g), mu_min=float(m), crank=(float(al.min()), float(al.max())),
                                ratio_max=ratio, anchor_to_floor=float(floor)))
    res.sort(key=lambda r: -r["trans_min"])
    return res


# ---------------------------------------------------------------- collision sweep (sweep17 method) on the v20 model
def sweep(betas, arm_betas):
    import geom4, build4
    from geom4 import rot, apply
    import build21_head as BM
    import build17 as B17
    links, joints, _ = BM.build_model()
    J = {j["name"]: j for j in joints}
    STEP = 0.6

    def solids(link):
        return [(pn, s) for pn, ss in links[link]["parts"].items() if "joint_disc" not in pn for s in ss
                if not (s[0] == "cyl" and s[2] == build4.DISC)]
    PTS = {}

    def pts(link):
        if link not in PTS:
            PTS[link] = [(pn, geom4.samples(s, STEP)) for pn, s in solids(link)]
        return PTS[link]
    FIX = {ln: solids(ln) for ln in links}
    BB = {id(s): build4.aabb(build4.sampled(s)) for ss in FIX.values() for _, s in ss}

    def hits(moving_link, M, fixed_link, Mf=None, margin=0.05):
        out = []
        Mi = np.linalg.inv(Mf) @ M if Mf is not None else M
        for pn, Pp in pts(moving_link):
            Q = apply(Mi, Pp)
            qlo, qhi = Q.min(0), Q.max(0)
            for fn, f in FIX[fixed_link]:
                b = BB[id(f)]
                if any(qhi[i] < b[i][0] or qlo[i] > b[i][1] for i in range(3)):
                    continue
                m = np.all((Q >= [x[0] for x in b]) & (Q <= [x[1] for x in b]), axis=1)
                if m.any():
                    n = int(geom4.inside(f, Q[m], margin).sum())
                    if n:
                        out.append((pn, fn, n))
        return out

    def Rj(name, deg):
        j = J[name]; return rot(j["axis"], deg, j["point_mm"])
    A_ = np.array(NOD_A); pin0 = S + np.array([NOD_R, 0.0]); P1, P2, LE = ear_geom()

    def angd(u):
        return math.degrees(math.atan2(u[1], u[0]))

    def poses(beta):
        al = nod(beta)["alpha"]; ps = ear(al)["psi"]
        pw = rot2(rot2(pin0, al, S), beta, P)
        d = -math.degrees(math.atan2(*(pw - A_)[::-1]) - math.atan2(*(pin0 - A_)[::-1]))
        Mh = Rj("head_nod", beta); Mc = Mh @ rot("y", al, (S[0], 0, S[1])); Mr = rot("y", d, (A_[0], 0, A_[1]))
        q1 = rot2(P1, al, S); q2 = rot2(P2, ps, E0); ph = -(angd(q2 - q1) - angd(P2 - P1))
        Me = Mh @ rot("y", ps, (E0[0], 0, E0[1]))
        Mer = rot("y", ph, (P1[0], 0, P1[1])); Mer[0, 3] += q1[0] - P1[0]; Mer[2, 3] += q1[1] - P1[1]; Mer = Mh @ Mer
        return al, ps, Mh, Mc, Mr, Me, Mer

    head = []
    for beta in betas:
        al, ps, Mh, Mc, Mr, Me, Mer = poses(beta)
        bad = []
        note = lambda tag, h: bad.extend([tag, x[0], x[1], x[2]] for x in h)
        note("head-torso", hits("head", Mh, "torso"))
        note("crank-torso", hits("nod_crank", Mc, "torso"))
        note("crank-head", hits("nod_crank", Mc, "head", Mh))                      # not in sweep17: crank over the floor pad
        note("rod-head", hits("nod_rod", Mr, "head", Mh))
        note("rod-crank", [x for x in hits("nod_rod", Mr, "nod_crank", Mc) if "pin" not in x[1]])
        note("rod-torso", [x for x in hits("nod_rod", Mr, "torso") if "anchor_pin" not in x[1]])
        note("ears-head", [x for x in hits("ears", Me, "head", Mh) if not ("axle" in x[0] and "bearing" in x[1])])
        note("ears-torso", hits("ears", Me, "torso"))
        note("ears-crank", hits("ears", Me, "nod_crank", Mc))
        note("erod-head", hits("ear_rod", Mer, "head", Mh))
        note("erod-crank", [x for x in hits("ear_rod", Mer, "nod_crank", Mc) if "pin" not in x[1]])
        note("erod-ears", [x for x in hits("ear_rod", Mer, "ears", Me) if "pin" not in x[1]])
        note("erod-nodrod", hits("ear_rod", Mer, "nod_rod", Mr))
        note("erod-torso", hits("ear_rod", Mer, "torso"))
        head.append(dict(beta=float(beta), crank=al, ear=ps, hits=bad))
        print(f"head {beta:6.1f} crank {al:6.1f} ear {ps:6.1f} hits {len(bad)}", sorted(set(tuple(x[:3]) for x in bad))[:4]); sys.stdout.flush()
    arms = []
    for beta in arm_betas:
        al, ps, Mh, Mc, Mr, Me, Mer = poses(beta)
        for nm in ("R", "L"):
            bad = {}
            for a in range(-180, 181, 5):
                M = Rj(f"{nm}_arm_swing", a); Mf = M @ Rj(f"{nm}_hand_flap", B17.flap_angle(a))
                h = []
                for mv, MM in ((f"{nm}_arm", M), (f"{nm}_hand_flap", Mf)):
                    h += [["head", x[0], x[1]] for x in hits(mv, MM, "head", Mh)]
                    h += [["ears", x[0], x[1]] for x in hits(mv, MM, "ears", Me)]
                if h:
                    bad[a] = sorted(set(tuple(x) for x in h))
            lo = 0
            while lo - 5 >= -180 and (lo - 5) not in bad: lo -= 5
            hi = 0
            while hi + 5 <= 180 and (hi + 5) not in bad: hi += 5
            arms.append(dict(beta=float(beta), ear=ps, arm=nm, free=(lo, hi),
                             first={str(k): [list(x) for x in bad[k]] for k in (lo - 5, hi + 5) if k in bad}))
            print(f"arm {nm} head {beta:6.1f} ear {ps:6.1f} free {lo}..{hi}", arms[-1]["first"]); sys.stdout.flush()
    return dict(model="build21_head (= build20 geometry)", step_mm=STEP, margin_mm=0.05, head=head, arms=arms)


# ---------------------------------------------------------------- figure
def figure(rows, free, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = "/home/kiro-ai/.local/share/fonts/NotoSansMonoCJKkr.otf"
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    ink, ink2, grid = "#191919", "#666666", "#e6e6e6"
    c1, c2, c3 = "#257de0", "#f47230", "#00b894"
    ok = [r for r in rows if r["ok"]]
    b = np.array([r["beta"] for r in ok])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.patch.set_facecolor("white")
    for ax in (a1, a2):
        ax.axvspan(*free, color="#f2f2f2", zorder=0, lw=0)
        ax.axvspan(*NEW_RANGE, color="#cce3fd", zorder=0, lw=0)
        ax.axvspan(*OLD_RANGE, color="#9dc8f7", zorder=0, lw=0)
        ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ink2)
        ax.tick_params(colors=ink2)
        ax.set_xlabel("머리 끄덕임 각 (도, + = 앞으로 숙임)", color=ink)
        ax.set_xlim(b.min(), b.max())
    a1.plot(b, [r["alpha"] for r in ok], color=c1, lw=2, label="크랭크(서보) 각")
    a1.plot(b, [r["psi"] if r.get("psi") is not None else np.nan for r in ok], color=c3, lw=2, label="귀 각")
    a1.set_ylabel("각 (도)", color=ink)
    a1.legend(frameon=False, loc="upper left", labelcolor=ink)
    a1.set_title("서보 각 · 귀 각", color=ink, loc="left", fontsize=11)
    for r in ok:
        if r["beta"] in NEW_RANGE:
            a1.annotate(f"{r['alpha']:.1f}", (r["beta"], r["alpha"]), textcoords="offset points", xytext=(6, -16) if r["beta"] > 0 else (-34, 6), color=ink, fontsize=9)
    g = np.array([min(r["gamma"], 180 - r["gamma"]) for r in ok]); m = np.array([min(r["mu"], 180 - r["mu"]) for r in ok])
    gr = np.array([min(r["rocker"], 180 - r["rocker"]) if r.get("rocker") is not None else np.nan for r in ok])
    a2.plot(b, g, color=c1, lw=2, label="크랭크와 막대 사이")
    a2.plot(b, m, color=c2, lw=2, label="막대와 출력 링크(축→고정점) 사이")
    a2.plot(b, gr, color=c3, lw=2, label="귀 막대와 귀 로커 사이")
    a2.axhline(30, color=ink2, lw=1, ls="--")
    a2.text(b.min() + 0.5, 31.5, "목표 하한 30°", color=ink2, fontsize=9)
    a2.set_ylim(0, 95)
    a2.set_ylabel("일직선까지 남은 각 (도, 90 = 직각)", color=ink)
    a2.legend(frameon=False, loc="lower left", labelcolor=ink, fontsize=9)
    a2.set_title("전달각 (작을수록 일직선 · 특이 자세에 가까움)", color=ink, loc="left", fontsize=11)
    sel = (b >= NEW_RANGE[0]) & (b <= NEW_RANGE[1])
    a2.annotate(f"최소 {min(g[sel].min(), m[sel].min()):.1f}°", (b[sel][np.argmin(np.minimum(g, m)[sel])], min(g[sel].min(), m[sel].min())),
                textcoords="offset points", xytext=(-60, -18), color=ink, fontsize=9, arrowprops=dict(arrowstyle="-", color=ink2))
    fig.suptitle("v20 크랭크를 그대로 두고 한계만 넓혀 머리 −20~+28° 를 냅니다 — 전달각 최소 "
                 f"{min(g[sel].min(), m[sel].min()):.0f}°, 충돌 없는 범위 {free[0]:+.0f}~{free[1]:+.0f}°",
                 color=ink, fontsize=12.5, x=0.01, ha="left")
    fig.text(0.01, 0.01, "띠: 진한 파랑 = 지금 표 −10~+20°, 연한 파랑 = 제안 −20~+28°, 회색 = 충돌 없음(sweep17 방식 점검, 2° 간격 · 경계 근처 0.5°). "
             "선은 계산(닫힌 고리 풀이), 출처 code_v17/head21.py", color=ink2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=150)


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--sweep" in sys.argv:
        betas = sorted(set(list(np.arange(-30, 37, 2.0)) + [-25.5, -25.0, -24.5, 32.5, 33.0, 33.5] + list(NEW_RANGE)))
        res = sweep(betas, list(np.arange(NEW_RANGE[0], NEW_RANGE[1] + 0.1, 4.0)))
        json.dump(res, open(os.path.join(OUT, "sweep_head21.json"), "w"), indent=1)
        return
    betas = np.arange(-30, 40.01, 0.5)
    rows = table(betas)
    sp = os.path.join(OUT, "sweep_head21.json")
    free = (-25.0, 32.0)
    if os.path.exists(sp):
        sw = json.load(open(sp))
        okb = sorted(h["beta"] for h in sw["head"] if not h["hits"])
        hit = sorted(h["beta"] for h in sw["head"] if h["hits"])
        lo = max([x for x in hit if x < 0], default=None); hi = min([x for x in hit if x > 0], default=None)
        free = (min(x for x in okb if lo is None or x > lo), max(x for x in okb if hi is None or x < hi))
    sel = [r for r in rows if r["ok"] and NEW_RANGE[0] <= r["beta"] <= NEW_RANGE[1]]
    tq = {b: torque(b) for b in (NEW_RANGE[0], -10.0, 0.0, 10.0, 20.0, NEW_RANGE[1])}
    I = inertia_about_axis()
    amp = math.radians((NEW_RANGE[1] - NEW_RANGE[0]) / 2); acc = amp * (2 * math.pi * NOD_HZ) ** 2
    dyn = {b: abs(t[0]) + I * acc * abs(t[1]) for b, t in tq.items() if b in NEW_RANGE}
    cl = clearances(np.arange(NEW_RANGE[0], NEW_RANGE[1] + 0.01, 1.0))
    alts = search()
    cur = dict(R=NOD_R, A=NOD_A, L=nod(0)["L"], ear=(EAR_R1, EAR_A1, EAR_R2, float(ear_geom()[2])))
    summ = dict(
        geometry=cur, old_range=OLD_RANGE, new_range=NEW_RANGE, collision_free=free,
        crank=(sel[0]["alpha"], sel[-1]["alpha"]), ear=(sel[0]["psi"], sel[-1]["psi"]),
        crank_old=(nod(OLD_RANGE[0])["alpha"], nod(OLD_RANGE[1])["alpha"]), ear_old=(ear(nod(OLD_RANGE[0])["alpha"])["psi"], ear(nod(OLD_RANGE[1])["alpha"])["psi"]),
        gamma_min=min(min(r["gamma"], 180 - r["gamma"]) for r in sel), mu_min=min(min(r["mu"], 180 - r["mu"]) for r in sel),
        rocker_min=min(min(r["rocker"], 180 - r["rocker"]) for r in sel),
        gamma_min_old=min(min(r["gamma"], 180 - r["gamma"]) for r in rows if r["ok"] and OLD_RANGE[0] <= r["beta"] <= OLD_RANGE[1]),
        mu_min_old=min(min(r["mu"], 180 - r["mu"]) for r in rows if r["ok"] and OLD_RANGE[0] <= r["beta"] <= OLD_RANGE[1]),
        toggle=[r["beta"] for r in rows if not r["ok"]][:1],
        torque_static={str(b): dict(servo_Nm=abs(t[0]), dbeta_dalpha=t[1], of_stall=abs(t[0]) / STALL) for b, t in tq.items()},
        inertia_kgm2=I, nod_test=dict(hz=NOD_HZ, amp_deg=math.degrees(amp), peak_acc=acc),
        torque_with_nod={str(b): dict(servo_Nm=v, of_stall=v / STALL) for b, v in dyn.items()},
        clearance_min={k: min(c[k] for c in cl if c[k] is not None) for k in ("anchor_to_floor", "slot_back", "slot_front", "rodend_to_floor")},
        pin_to_pad_min=min([c["pin_to_pad"] for c in cl if c["pin_to_pad"] is not None], default=None),
        ticks_crank=(2048 + sel[0]["alpha"] * TICK, 2048 + sel[-1]["alpha"] * TICK),
        ticks_if_head_angle=(2048 + NEW_RANGE[0] * TICK, 2048 + NEW_RANGE[1] * TICK),
        alternatives=alts[:8], alternatives_n=len(alts),
    )
    json.dump(dict(summary=summ, rows=rows, clearances=cl), open(os.path.join(OUT, "head21.json"), "w"), indent=1, default=float)
    figure(rows, free, os.path.join(OUT, "head_crank_v21.png"))
    s = summ
    print(f"geometry R {NOD_R} A {NOD_A} rod {cur['L']:.2f}; collision-free {free}")
    print(f"head {NEW_RANGE}: crank {s['crank'][0]:.1f} .. {s['crank'][1]:.1f}, ear {s['ear'][0]:.1f} .. {s['ear'][1]:.1f}")
    print(f"transmission min: crank-rod {s['gamma_min']:.1f}, rod-output {s['mu_min']:.1f}, ear rod-rocker {s['rocker_min']:.1f} (old range {s['gamma_min_old']:.1f} / {s['mu_min_old']:.1f})")
    print("loop opens (toggle) at head", s["toggle"])
    for b, t in s["torque_static"].items():
        print(f"  static beta {b}: {t['servo_Nm']:.3f} N m ({100 * t['of_stall']:.1f} % stall), dbeta/dalpha {t['dbeta_dalpha']:.3f}")
    for b, t in s["torque_with_nod"].items():
        print(f"  + 1 Hz nod at beta {b}: {t['servo_Nm']:.3f} N m ({100 * t['of_stall']:.1f} %)")
    print("clearance min (mm)", {k: round(v, 2) for k, v in s["clearance_min"].items()}, "pin to pad", s["pin_to_pad_min"])
    print("ticks crank", [round(x) for x in s["ticks_crank"]], "ticks if head angle used", [round(x) for x in s["ticks_if_head_angle"]])
    print("alternatives", len(alts))
    for a in alts[:8]:
        print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()})


if __name__ == "__main__":
    main()
