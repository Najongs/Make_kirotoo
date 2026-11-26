"""Servo assignment search: every symmetric choice of 3 XM430 among the 5 leg joints (rest XC430), arm/nod on
XC430 or XL430. Scores each by the worst combined torque-speed utilisation u = |tau|/tau_stall + |w|/w0 over all
cases, and by the rms torque / stall (heating) in walking.

usage: python3 servo_opt.py mass_xxx.json out.json
"""
import sys, json, itertools, numpy as np
import servo_study as S


def hip_strategy(M, A_deg=8.0, Tp=0.5, n=120):
    """single support on the right leg; the upper body (pelvis and everything above the stance hip pitch)
    oscillates about the stance hip-pitch axis: theta = A sin(2 pi t / Tp)."""
    ys = -40.0
    H = 150.0 - 15.0
    q1, q2 = S.leg_ik(0.0, -H)
    psi = np.radians(12.5)
    R0 = S.rot("x", psi)
    P0 = np.array([0.0, ys, 35.0]) - R0 @ np.array([0.0, ys, 185.0 - H])
    q_st = (0.0, q1, q2, -(q1 + q2), -psi)
    bod0, jts0 = S.poses(M, ys, q_st, q_st, P0, R0, (0, 0))
    hp, ax = jts0["R_hip_pitch"]
    stance_fixed = {f"R_{l}" for l in S.LEGL[1:]}
    dt = Tp / n
    rec = []
    for k in range(-2, n + 3):
        t = k * dt
        th = np.radians(A_deg) * np.sin(2 * np.pi * t / Tp)
        # rotation about axis 'ax' through hp
        a = ax / np.linalg.norm(ax)
        K_ = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        Rr = np.eye(3) + np.sin(th) * K_ + (1 - np.cos(th)) * K_ @ K_
        bod = {}
        for b, (Rb, cb) in bod0.items():
            bod[b] = (Rb, cb) if b in stance_fixed else (Rr @ Rb, hp + Rr @ (cb - hp))
        jts = dict(jts0)
        qs = dict(st=(0.0, q1 - th, q2, -(q1 + q2), -psi), sw=q_st)
        rec.append((t, bod, jts, qs, psi))
    tq, qs, zmp = S.dynamics(M, rec, dt)
    return {"stance_hip_pitch": (tq["stance_hip_pitch"], qs["stance_hip_pitch"])}


def collect(M):
    """{joint: [(tau array, w array), ...]} over every case and both legs' roles"""
    res = {j: [] for j in S.LEGJ}
    rms = {j: 0.0 for j in S.LEGJ}
    per_case = {}
    for nm, c in S.CASES.items():
        tq, qs, zmp, _ = S.run_case(M, c)
        per_case[nm] = {k: (float(np.max(np.abs(v))), float(np.max(np.abs(qs[k])))) for k, v in tq.items()}
        for j in S.LEGJ:
            for role in ("stance", "swing"):
                res[j].append((tq[f"{role}_{j}"], qs[f"{role}_{j}"], nm))
            if nm == "walk":       # a joint is stance half the time and swing half the time
                rms[j] = float(np.sqrt((np.mean(tq[f"stance_{j}"] ** 2) + np.mean(tq[f"swing_{j}"] ** 2)) / 2))
    for nm, (tqs, zmp, psi) in S.static_cases(M).items():
        per_case[nm] = {k: (float(v), 0.0) for k, v in tqs.items()}
        for k, v in tqs.items():
            res[k.replace("stance_", "")].append((np.array([v]), np.array([0.0]), nm))
    h = hip_strategy(M)
    t, w = h["stance_hip_pitch"]
    per_case["hip_strategy"] = {"stance_hip_pitch": (float(np.max(np.abs(t))), float(np.max(np.abs(w))))}
    res["hip_pitch"].append((t, w, "hip_strategy"))
    return res, rms, per_case


def score(res, rms, assign, cases=None):
    """cases: None = everything except 'jog' (the design envelope); or a list of case names"""
    u, heat = {}, {}
    for j in S.LEGJ:
        ts, w0 = S.SPEC[assign[j]]
        u[j] = max(float(np.max(np.abs(t) / ts + np.abs(w) / w0)) for t, w, nm in res[j]
                   if (nm != "jog" if cases is None else nm in cases))
        heat[j] = rms[j] / ts
    return u, heat


if __name__ == "__main__":
    mass = json.load(open(sys.argv[1]))
    M0 = S.Model(mass)
    for nm, c in S.CASES.items():
        deg, err = S.tune_psi(M0, c)
        S.CASES[nm]["psi_amp"] = np.radians(deg)
    out = {"psi_deg": {k: float(np.degrees(v["psi_amp"])) for k, v in S.CASES.items()}, "options": []}
    for xm in itertools.combinations(S.LEGJ, 3):
        assign = {j: ("XM430-W350" if j in xm else "XC430-W240") for j in S.LEGJ}
        M = S.Model(mass, assign, arm_servo="XL430-W250", nod_servo="XL430-W250")
        res, rms, per_case = collect(M)
        u, heat = score(res, rms, assign)
        uj, _ = score(res, rms, assign, ["jog"])
        out["options"].append(dict(xm=list(xm), mass=M.total(), u=u, u_jog=uj, heat=heat, worst_u=max(u.values()),
                                   worst_u_jog=max(uj.values()), worst_heat=max(heat.values()), per_case=per_case))
        print(f"XM {str(xm):44s} env u {max(u.values()):.2f} jog u {max(uj.values()):.2f} heat {max(heat.values()):.2f}  "
              + " ".join(f"{j[:6]} {u[j]:.2f}/{uj[j]:.2f}/{heat[j]:.2f}" for j in S.LEGJ))
    # arms on / off: stance hip yaw torque
    M = S.Model(mass)
    for arms in (True, False):
        rec, dt = S.gait(M, arms=arms, **S.CASES["fast"])
        tq, qs, zmp = S.dynamics(M, rec, dt)
        out[f"hip_yaw_fast_arms_{arms}"] = float(np.max(np.abs(tq["stance_hip_yaw"])))
        print("arms", arms, "stance hip yaw peak", round(out[f"hip_yaw_fast_arms_{arms}"], 3))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
