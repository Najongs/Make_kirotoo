"""Camera FFC route, Pi Camera v2 (face plate, left eye) -> head floor slot -> lid cap slot -> Jetson CSI, over the head
nod range (build21_body model). Centre line = waypoints joined by straight runs and arcs of radius R_BEND (FFC bent
about its width axis; width 16 kept along y except the one sideways run, made with a 45 deg fold).

Reports per nod angle: run length (camera -> CSI), the neck run between the two slots (straight-ish S curve, the
slack slides through the slots into the head), clearance of the ribbon (16 wide) to the head / torso solids and to the
moving nod crank, nod servo horn, ear rod and nod rod, and the cable length to buy (longest pose + MARGIN).

Run:  cd code_v17 && ../sim/.venv/bin/python camera_cable21.py [model.pkl]     (model build ~2 min without the pickle)
Out:  stdout + ../out20/camera_cable21.json
The Jetson CSI connector place is an estimate (the model has the dev kit as a box): two cases, near (back edge, by
the slot) and far (front edge); the length to buy covers the far case.
"""
import os, sys, json, math, pickle
import numpy as np
import geom4, build4
from geom4 import rot, apply

R_BEND = 8.0                 # arc radius of the centre line at each waypoint (FFC least bend ~5 [estimate])
W = 16.0                     # FFC width (15 pin, 1 mm pitch)
MARGIN = 30.0                # length to buy = longest pose + MARGIN
NOD = [-20.0, -14.0, -7.0, 0.0, 7.0, 14.0, 21.0, 28.0]     # head_nod range -20 .. +28
PIVOT = np.array([0.0, 0.0, 348.5])                        # head_nod axis (y) point

# head frame (= world at nod 0): camera connector exit (board bottom edge, back side) -> down -> back over the speaker
# side -> sideways to the middle (45 deg fold) in front of the nod servo walls -> back over the servo -> down behind it
HEAD = [(62.0, 37.5, 438.6), (60.0, 37.5, 425.0), (42.0, 37.5, 418.0), (28.0, 0.0, 418.0), (-52.0, 0.0, 418.0),
        (-52.0, 0.0, 367.0)]
FLOOR_SLOT = (-52.0, 0.0, 366.0)       # head floor slot centre (x -58 .. -46, z 365-367)
LID_SLOT = (-51.0, 0.0, 337.3)         # lid cap slot centre (x -58 .. -44, z 336.3-338.3)
# torso: lid slot -> down past the lid plug back bar (inner face x -58) -> CSI connector [estimate]
CSI = {"near (back edge)": [(-51.0, 0.0, 330.0), (-52.0, 0.0, 300.0), (-46.0, 0.0, 292.0)],
       "far (front edge)": [(-51.0, 0.0, 330.0), (-40.0, 0.0, 326.0), (52.0, 0.0, 326.0), (53.0, 0.0, 300.0),
                            (46.0, 0.0, 292.0)]}


def polyline_len(P, r=R_BEND):
    """length of the polyline with each inner corner replaced by an arc of radius r (tangent lengths clipped)"""
    P = np.asarray(P, float)
    L = sum(np.linalg.norm(P[i + 1] - P[i]) for i in range(len(P) - 1))
    for i in range(1, len(P) - 1):
        a, b = P[i - 1] - P[i], P[i + 1] - P[i]
        th = math.pi - math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))   # turn angle
        t = min(r * math.tan(th / 2), np.linalg.norm(a) / 2, np.linalg.norm(b) / 2)
        rr = t / math.tan(th / 2) if th > 1e-6 else r
        L += rr * th - 2 * t
    return L


def min_radius(P, r=R_BEND):
    P = np.asarray(P, float); out = r
    for i in range(1, len(P) - 1):
        a, b = P[i - 1] - P[i], P[i + 1] - P[i]
        th = math.pi - math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))
        if th < 1e-6:
            continue
        t = min(r * math.tan(th / 2), np.linalg.norm(a) / 2, np.linalg.norm(b) / 2)
        out = min(out, t / math.tan(th / 2))
    return out


def ribbon(P, step=1.0):
    """points of the ribbon: centre line every `step`, +- W/2 along y (width) at 5 across"""
    P = np.asarray(P, float); pts = []
    for i in range(len(P) - 1):
        n = max(2, int(np.linalg.norm(P[i + 1] - P[i]) / step) + 1)
        for t in np.linspace(0, 1, n):
            c = P[i] + t * (P[i + 1] - P[i])
            pts += [c + [0, dy, 0] for dy in np.linspace(-W / 2, W / 2, 5)]
    return np.array(pts)


def neck_curve(p0, t0, p1, t1, n=60):
    """cubic Bezier from p0 (tangent t0) to p1 (tangent t1), handles chosen for the largest least radius"""
    best = None
    d = np.linalg.norm(p1 - p0)
    for k in np.linspace(0.15, 0.6, 19):
        h = k * d
        C = [p0, p0 + h * t0, p1 - h * t1, p1]
        s = np.linspace(0, 1, n)[:, None]
        B = (1 - s) ** 3 * C[0] + 3 * (1 - s) ** 2 * s * C[1] + 3 * (1 - s) * s ** 2 * C[2] + s ** 3 * C[3]
        d1 = 3 * ((1 - s) ** 2 * (C[1] - C[0]) + 2 * (1 - s) * s * (C[2] - C[1]) + s ** 2 * (C[3] - C[2]))
        d2 = 6 * ((1 - s) * (C[2] - 2 * C[1] + C[0]) + s * (C[3] - 2 * C[2] + C[1]))
        cr = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12)
        rmin = 1 / max(cr.max(), 1e-9)
        L = np.sum(np.linalg.norm(np.diff(B, axis=0), axis=1))
        if best is None or rmin > best[0]:
            best = (rmin, L, B)
    return best


def clearance(P, sols, cap=12.0):
    """least distance (to cap) from the points to the solids: grow the solids in 0.5 steps (geom4.inside margin < 0)"""
    for c in np.arange(0.0, cap + 1e-9, 0.5):
        for nm, s in sols:
            b = build4.aabb(build4.sampled(s))
            sel = np.all((P >= [x[0] - c for x in b]) & (P <= [x[1] + c for x in b]), axis=1)
            if sel.any() and geom4.inside(s, P[sel], -c).any():
                return c, nm
    return cap, None


def main():
    if len(sys.argv) > 1:
        links, joints = pickle.load(open(sys.argv[1], "rb"))
    else:
        sys.argv = sys.argv[:1]
        import build21_body as B
        links, joints, _ = B.build_model()
    J = {j["name"]: j for j in joints}
    assert J["head_nod"]["point_mm"] == (0, 0, 348.5) and J["head_nod"]["axis"] == "y"
    head_static = [(pn, s) for pn, ss in links["head"]["parts"].items() for s in ss]
    torso_static = [(pn, s) for pn, ss in links["torso"]["parts"].items() for s in ss]
    movers = {ln: [(pn, s) for pn, ss in links[ln]["parts"].items() for s in ss]
              for ln in ("nod_crank", "ear_rod", "nod_rod", "ears") if ln in links}
    res = dict(r_bend=R_BEND, width=W, margin=MARGIN, poses=[])
    # head run (fixed in the head frame): length, clearance to the head's own parts
    LH = polyline_len(HEAD) + abs(HEAD[-1][2] - FLOOR_SLOT[2])
    RH = ribbon(HEAD)
    ch, chn = clearance(RH, [x for x in head_static if x[0] not in ("camera_pi_v2",)])
    # moving parts in the head: the crank / horn / ear rod / ear rocker turn with the nod; their y bands
    yb = {ln: (min(build4.aabb(build4.sampled(s))[1][0] for _, s in ss), max(build4.aabb(build4.sampled(s))[1][1] for _, s in ss))
          for ln, ss in movers.items()}
    # crank (+ servo horn, ear pin) over its whole range and the ears over theirs, in the head frame
    mv = []
    for ln, jn in (("nod_crank", "nod_crank"), ("ears", "ear_axle")):
        j = J[jn]
        for a in np.linspace(j["lower_deg"], j["upper_deg"], 12):
            Mi = np.linalg.inv(rot(j["axis"], a, j["point_mm"]))
            mv.append((ln, a, *clearance(apply(Mi, RH), movers[ln])))
    worst = min(mv, key=lambda x: x[2])
    # ear rod / nod rod: closed loops, kept to their y bands; the ribbon is within |y| <= W/2 wherever x < 28
    far = RH[RH[:, 0] < 28.0]
    res["head_run"] = dict(waypoints=HEAD, length_mm=round(LH, 1), least_radius_mm=round(min_radius(HEAD), 1),
                           ribbon_clear_to_head_parts_mm=ch, nearest=chn, mover_y_bands=yb,
                           ribbon_y_band=(round(RH[:, 1].min(), 1), round(RH[:, 1].max(), 1)),
                           ribbon_y_band_behind_x28=(round(far[:, 1].min(), 1), round(far[:, 1].max(), 1)),
                           clear_to_crank_ears_over_range=dict(link=worst[0], angle=round(float(worst[1]), 1), mm=worst[2], solid=worst[3]))
    for nm, T in CSI.items():
        res.setdefault("torso_run", {})[nm] = dict(waypoints=T, length_mm=round(polyline_len([LID_SLOT] + T), 1),
                                                   least_radius_mm=round(min_radius([LID_SLOT] + T), 1))
    RT = ribbon([LID_SLOT] + CSI["far (front edge)"][:3])
    ct, ctn = clearance(RT, [x for x in torso_static if x[0] not in ("lid",)])
    res["torso_run_clear_mm"] = (ct, ctn)
    for th in NOD:
        M = rot("y", th, PIVOT)
        p1 = apply(M, np.array([FLOOR_SLOT]))[0]
        t1 = M[:3, :3] @ np.array([0, 0, 1.0])
        p0 = np.array(LID_SLOT, float)
        rmin, Ln, Bz = neck_curve(p0, np.array([0, 0, 1.0]), p1, t1)
        # neck ribbon vs torso / head static parts (head parts posed)
        RN = ribbon(Bz[5:-5], 1.0)
        Mi = np.linalg.inv(M)
        c_t, n_t = clearance(RN, [x for x in torso_static if x[0] != "lid"], 8.0)
        c_h, n_h = clearance(apply(Mi, RN), [x for x in head_static if x[0] != "head_back"], 8.0)
        res["poses"].append(dict(nod=th, floor_slot=[round(v, 1) for v in p1], slot_gap_mm=round(float(np.linalg.norm(p1 - p0)), 1),
                                 neck_len_mm=round(Ln, 1), neck_least_radius_mm=round(rmin, 1),
                                 neck_clear_torso=(c_t, n_t), neck_clear_head=(c_h, n_h)))
    neck_max = max(p["neck_len_mm"] for p in res["poses"])
    for nm, t in res["torso_run"].items():
        tot = LH + neck_max + t["length_mm"]
        t["total_longest_pose_mm"] = round(tot, 1)
        t["buy_mm"] = round(tot + MARGIN, 1)
    res["slack_in_head_mm"] = round(neck_max - min(p["neck_len_mm"] for p in res["poses"]), 1)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out20", "camera_cable21.json"), "w"),
              indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
