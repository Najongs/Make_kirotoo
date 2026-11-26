"""Leg-to-leg clearance on the design solids (no MuJoCo hulls): the right leg yaws by theta, the left leg keeps
yaw 0; both legs share the hip pitch / knee pattern (feet flat: ankle pitch = -(hip + knee)).
Distance = nearest-neighbour distance between volume point samples (geom4.samples, spacing STEP) of the moving
right foot / ankle link / shank and the left foot / ankle link / shank. Sampling makes the measured distance
over-estimate the true gap by at most STEP*sqrt(3) (both clouds), reported as `sampling_error_mm`.

usage (module):  from foot_clearance import Clearance; C = Clearance("build18"); C.min_gap(theta_list, grid)
"""
import importlib, math, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "code_v17"))
import geom4, build4
from geom4 import rot

LINKS = ["foot", "ankle_link", "shank"]
CHAIN = ["hip_yaw", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
CHAIN_LINK = {"hip_yaw_link": 1, "thigh": 2, "shank": 3, "ankle_link": 4, "foot": 5}
GRID_TEST5 = [(p, k) for p in (-30, -15, 0, 15, 30) for k in (0, 30, 60)]      # scissor: R pitch p, L pitch -p
GRID_CROUCH = "crouch"


class Clearance:
    def __init__(self, module, step=0.6, links_joints=None):
        if links_joints is None:
            mod = importlib.import_module(module)
            links, joints, _ = mod.build_model()
        else:
            links, joints = links_joints
        self.links = links
        self.J = {j["name"]: j for j in joints}
        self.step = step
        self.P, self.lab = {}, {}
        for S in "RL":
            ys = -40.0 if S == "R" else 40.0
            for l in LINKS:
                ln = f"{S}_{l}"
                pts, lab = [], []
                for pn, ss in links[ln]["parts"].items():
                    if "joint_disc" in pn:
                        continue
                    for s in ss:
                        if s[0] == "cyl" and tuple(s[2]) == tuple(build4.DISC):
                            continue
                        q = geom4.samples(s, step)
                        keep = (q[:, 1] - ys) * (-1 if S == "R" else 1) < 12.0   # inner half (+ margin)
                        pts.append(q[keep]); lab += [pn] * int(keep.sum())
                self.P[ln] = np.concatenate(pts); self.lab[ln] = np.array(lab)

    def pose(self, S, ang):
        """world 4x4 of every leg link of side S for joint angles ang {short name: deg}"""
        M = np.eye(4); out = {}
        for i, j in enumerate(CHAIN):
            jj = self.J[f"{S}_{j}"]
            M = M @ rot(jj["axis"], ang.get(j, 0.0), jj["point_mm"])
            out[i + 1] = M.copy()
        return {l: out[CHAIN_LINK[l]] for l in LINKS}

    BOUND = 12.0          # gaps above this are reported as >= BOUND

    def _gap(self, aR, aL):
        key = tuple(sorted(aL.items()))
        if not hasattr(self, "_trees"):
            self._trees = {}
        if key not in self._trees:
            ML = self.pose("L", aL)
            self._trees[key] = {lb: cKDTree(self.P[f"L_{lb}"] @ ML[lb][:3, :3].T + ML[lb][:3, 3]) for lb in LINKS}
        T = self._trees[key]
        MR = self.pose("R", aR)
        best = (self.BOUND, None)
        for la in LINKS:
            P = self.P[f"R_{la}"] @ MR[la][:3, :3].T + MR[la][:3, 3]
            for lb in LINKS:
                dd, ii = T[lb].query(P, k=1, distance_upper_bound=best[0])
                j = int(np.argmin(dd))
                if dd[j] < best[0]:
                    best = (float(dd[j]), (f"R_{la}", str(self.lab[f"R_{la}"][j]), f"L_{lb}", str(self.lab[f"L_{lb}"][ii[j]]),
                                            [float(x) for x in np.round(P[j], 1)]))
        return best

    def gap(self, theta, p, k, L_theta=0.0):
        """scissor grid pose: R pitch p, L pitch -p, both knees k, feet flat"""
        return self._gap({"hip_yaw": theta, "hip_pitch": p, "knee": k, "ankle_pitch": -(p + k)},
                         {"hip_yaw": L_theta, "hip_pitch": -p, "knee": k, "ankle_pitch": -(-p + k)})

    def crouch_grid(self):
        """both feet flat on the floor: symmetric crouch 0 / 15 / 30 / 45 mm (servo_study leg_ik)"""
        import servo_study as S
        g = []
        for c in (0.0, 15.0, 30.0, 45.0):
            q1, q2 = S.leg_ik(0.0, -(150.0 - c))
            g.append((math.degrees(q1), math.degrees(q2), c))
        return g

    def scan(self, thetas, grid="both"):
        """{theta: (min gap, where)} over the grids (crouch with both feet down + test5 scissor grid)"""
        cfg = []
        if grid in ("both", "crouch"):
            cfg += [(p, k, f"crouch {c:.0f}") for p, k, c in self.crouch_grid()]
        if grid in ("both", "test5"):
            cfg += [(p, k, f"scissor p{p} k{k}") for p, k in GRID_TEST5]
        res = {}
        for th in thetas:
            best = (1e9, None, None)
            for p, k, tag in cfg:
                if tag.startswith("crouch"):
                    aR_p, aR_k = p, k
                    g = self.gap_crouch(th, p, k)
                else:
                    g = self.gap(th, p, k)
                if g[0] < best[0]:
                    best = (g[0], g[1], tag)
            res[th] = best
        return res

    def gap_crouch(self, theta, q1, q2):
        """both legs the same crouch (ankle under the hip, feet flat, both on the floor)"""
        return self._gap({"hip_yaw": theta, "hip_pitch": q1, "knee": q2, "ankle_pitch": -(q1 + q2)},
                         {"hip_yaw": 0.0, "hip_pitch": q1, "knee": q2, "ankle_pitch": -(q1 + q2)})


if __name__ == "__main__":
    # usage: foot_clearance.py <module> <theta_max> <out json>   (KITORU_THETA picks the build18 cut)
    import json
    mod = sys.argv[1] if len(sys.argv) > 1 else "build17"
    tmax = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    C = Clearance(mod)
    print({k: len(v) for k, v in C.P.items()}, "sampling error <=", round(C.step * math.sqrt(3), 2), "mm", flush=True)
    out = {"module": mod, "theta_env": os.environ.get("KITORU_THETA"), "step_mm": C.step,
           "sampling_error_mm": C.step * math.sqrt(3), "bound_mm": C.BOUND, "crouch": {}, "scissor": {}}
    for th in range(-tmax, tmax + 1):
        best = min(((C.gap_crouch(th, p, k), f"crouch {c:.0f}") for p, k, c in C.crouch_grid()), key=lambda x: x[0][0])
        out["crouch"][th] = {"gap": best[0][0], "where": best[0][1], "pose": best[1]}
        print(f"crouch theta {th:+3d}: {best[0][0]:6.2f} mm {best[1]} {best[0][1]}", flush=True)
    for th in range(-tmax, tmax + 1, 5):
        best = min(((C.gap(th, p, k), f"scissor p{p} k{k}") for p, k in GRID_TEST5), key=lambda x: x[0][0])
        out["scissor"][th] = {"gap": best[0][0], "where": best[0][1], "pose": best[1],
                              "per_pose": {f"p{p} k{k}": C.gap(th, p, k)[0] for p, k in GRID_TEST5}}
        print(f"scissor theta {th:+3d}: {best[0][0]:6.2f} mm {best[1]} {best[0][1]}", flush=True)
    json.dump(out, open(sys.argv[3], "w"), indent=1)
