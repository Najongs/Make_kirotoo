"""Load check of the v22 MJCFs (mesh-collision kitoru_v22.xml and primitive-collision kitoru_v22_rl.xml) — the same
items as tests/check_v21.py (copied; v22 mass json already has every servo at XL430 65.2 g, so no swap correction).

Run:  cd sim && .venv/bin/python tests/check_v22.py
Out:  stdout + sim/model/check_v22.json
Checks: total mass vs the design mass json (and json - XM/XC -> XL430 servo swap), joint / actuator counts, collision
candidate pairs (MuJoCo broad-phase filter: same weld body, parent-child, contype / conaffinity, <exclude>),
standing drop at the zero keyframe for 2 s with every actuator holding 0, at timestep 1 ms and 2 ms
(equality solref raised to 2 x timestep at 2 ms, as export_mjcf_rl.py does), with the model kp (XL430 6.4 N*m/rad
[estimate]) and with kp x 2 (the position-gain doubling every sim/walk study uses).
"""
import json, math, os, time
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "..", "model")
MASSF = os.path.join(HERE, "..", "model", "mass_kitoru_v22_light.json")
SWAP_G = 0.0                                          # v22 mass json: servos are XL430 65.2 g already


def pairs(m):
    """geom pairs MuJoCo's broad phase would test (filterparent on): returns (count, by type pair)"""
    excl = {(m.exclude_signature[i] >> 16, m.exclude_signature[i] & 0xFFFF) for i in range(m.nexclude)} \
        if hasattr(m, "exclude_signature") else set()
    weld = m.body_weldid; par = m.body_parentid
    col = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    n, kinds = 0, {}
    tn = {0: "plane", 2: "sphere", 3: "capsule", 6: "box", 7: "mesh"}
    for a_i, a in enumerate(col):
        for b in col[a_i + 1:]:
            ba, bb = m.geom_bodyid[a], m.geom_bodyid[b]
            wa, wb = weld[ba], weld[bb]
            if wa == wb:
                continue
            if wa != 0 and wb != 0 and (wa == weld[par[wb]] or wb == weld[par[wa]]):
                continue
            if not ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])):
                continue
            if (min(ba, bb), max(ba, bb)) in excl or (max(ba, bb), min(ba, bb)) in excl:
                continue
            n += 1
            k = "-".join(sorted((tn.get(int(m.geom_type[a]), str(m.geom_type[a])), tn.get(int(m.geom_type[b]), str(m.geom_type[b])))))
            kinds[k] = kinds.get(k, 0) + 1
    return n, kinds, len(col)


def stand(path, dt, T=2.0, kp_scale=1.0):
    m = mujoco.MjModel.from_xml_path(path)
    m.opt.timestep = dt
    m.actuator_gainprm[:, 0] *= kp_scale; m.actuator_biasprm[:, 1] *= kp_scale     # as sim/walk kw_common.scale_kp
    for i in range(m.neq):
        m.eq_solref[i, 0] = max(m.eq_solref[i, 0], 2 * dt)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("zero").id)
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2])
    jh, jc, je = (m.joint(n).qposadr[0] for n in ("head_nod", "nod_crank", "ear_axle"))
    eqs = [m.equality(n) for n in ("loop_head", "loop_ear")]
    cf = [np.array(m.eq_data[e.id][:5]) for e in eqs]
    zmin = zmax = z0; tilt = 0.0; loop = 0.0; ncon = []; vmax = 0.0
    n = int(round(T / dt)); t0 = time.perf_counter()
    for k in range(n):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            return dict(ok=False, reason=f"nan at step {k}")
        z = float(d.qpos[2]); zmin, zmax = min(zmin, z), max(zmax, z)
        q = d.qpos[3:7]; up = 1 - 2 * (q[1] ** 2 + q[2] ** 2)          # torso z axis . world z
        tilt = max(tilt, math.degrees(math.acos(max(-1.0, min(1.0, up)))))
        c = d.qpos[jc]
        loop = max(loop, abs(d.qpos[jh] - np.polynomial.polynomial.polyval(c, cf[0])),
                   abs(d.qpos[je] - np.polynomial.polynomial.polyval(c, cf[1])))
        ncon.append(d.ncon)
        if k > n // 2:
            vmax = max(vmax, float(np.abs(d.qvel).max()))
    wall = time.perf_counter() - t0
    warn = {mujoco.mjtWarning(i).name: int(d.warning[i].number) for i in range(len(d.warning)) if d.warning[i].number}
    ok = tilt < 5.0 and (z0 - zmin) < 0.01 and not warn and vmax < 0.1
    return dict(ok=bool(ok), dt=dt, kp_scale=kp_scale, torso_z0_mm=round(z0 * 1e3, 2), torso_z_end_mm=round(float(d.qpos[2]) * 1e3, 2),
                torso_z_min_mm=round(zmin * 1e3, 2), torso_z_max_mm=round(zmax * 1e3, 2), tilt_max_deg=round(tilt, 3),
                xy_drift_mm=round(float(np.hypot(*d.qpos[:2])) * 1e3, 2), qvel_max_2nd_half=round(vmax, 4),
                loop_resid_max_deg=round(math.degrees(loop), 4), ncon_mean=round(float(np.mean(ncon)), 1),
                ncon_end=int(d.ncon), warnings=warn, wall_s=round(wall, 2), sim_x_realtime=round(T / wall, 1))


def info(path):
    m = mujoco.MjModel.from_xml_path(path)
    hinge = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    act = [m.actuator(i).name for i in range(m.nu)]
    passive = sorted(set(hinge) - set(m.joint(int(m.actuator_trnid[i, 0])).name for i in range(m.nu)))
    js = json.load(open(MASSF))
    tot = float(m.body_mass.sum()) * 1e3
    n, kinds, ncol = pairs(m)
    return dict(file=os.path.basename(path), mass_g=round(tot, 2), mass_json_g=round(js["total_g"], 2),
                mass_json_xl_g=round(js["total_g"] + SWAP_G, 2), mass_minus_json_xl_g=round(tot - js["total_g"] - SWAP_G, 3),
                nbody=m.nbody, hinge=len(hinge), actuators=len(act), passive=passive, neq=m.neq,
                eq=[m.equality(i).name for i in range(m.neq)],
                stall=sorted({round(float(x), 3) for x in m.actuator_forcerange[:, 1]}),
                kp=sorted({round(float(x), 3) for x in m.actuator_gainprm[:, 0]}),
                damping=sorted({round(float(m.dof_damping[m.jnt_dofadr[m.actuator_trnid[i, 0]]]), 4) for i in range(m.nu)}),
                armature=sorted({round(float(m.dof_armature[m.jnt_dofadr[m.actuator_trnid[i, 0]]]), 4) for i in range(m.nu)}),
                ranges_deg={m.joint(i).name: [round(math.degrees(x), 2) for x in m.jnt_range[i]] for i in range(m.njnt)
                            if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE},
                collision_geoms=ncol, collision_pairs=n, pair_types=kinds, timestep=m.opt.timestep, actuator_names=act)


out = {}
for fn in ("kitoru_v22.xml", "kitoru_v22_rl.xml"):
    p = os.path.join(MD, fn)
    r = info(p)
    r["stand"] = {f"{dt * 1e3:g}ms_kp_x{ks:g}": stand(p, dt, kp_scale=ks) for ks in (1.0, 2.0) for dt in (0.001, 0.002)}
    out[fn] = r
    print(json.dumps({k: v for k, v in r.items() if k not in ("ranges_deg", "actuator_names")}, indent=1, ensure_ascii=False))
print(json.dumps(out["kitoru_v22.xml"]["ranges_deg"]))
json.dump(out, open(os.path.join(MD, "check_v22.json"), "w"), indent=1)
