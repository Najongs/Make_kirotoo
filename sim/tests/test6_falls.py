"""Test 6: falls. Free model standing at the zero pose (position hold); a horizontal push on the torso centre of
mass for 0.2 s toward front / back / left / right. Smallest push (of a ladder) that topples it, and for that push
and for a hard push the order in which collision shapes first touch the floor. Impact forces are reference only
(they depend on the contact softness settings).
Run: cd sim && uv run python tests/test6_falls.py -> results/test6_falls.json, .png
"""
import math, numpy as np
from common import *

DIRS = {"front": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1)}
LADDER = [4, 6, 8, 10, 12, 15, 20, 30]


def push(dname, F, shot=False):
    m, d = load(); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    tb = bid(m, "torso"); dx, dy = DIRS[dname]
    order = []; seen = set(); t_fall = None; fmax_c = 0.0; amax = np.zeros(m.nu); img = None
    ren = mujoco.Renderer(m, 360, 360) if shot else None
    for k in range(3500):
        t = k * m.opt.timestep
        d.xfrc_applied[tb, :] = 0
        if 0.5 <= t < 0.7:
            # force at the subtree centre of mass: xfrc acts at the body com, so add the moment for the offset
            F3 = np.array([dx * F, dy * F, 0.0])
            r = d.subtree_com[tb] - d.xipos[tb]
            d.xfrc_applied[tb, :3] = F3; d.xfrc_applied[tb, 3:] = np.cross(r, F3)
        mujoco.mj_step(m, d)
        for i, c in enumerate(d.contact[:d.ncon]):
            g1, g2 = gname(m, c.geom1), gname(m, c.geom2)
            if "floor" in (g1, g2):
                g = g2 if g1 == "floor" else g1
                if "_pad" in g:
                    continue
                if g not in seen:
                    seen.add(g); order.append((round(t - 0.5, 3), g))
                    if shot and img is None:
                        ren.update_scene(d, "track") if False else None
                f6 = np.zeros(6); mujoco.mj_contactForce(m, d, i, f6); fmax_c = max(fmax_c, abs(f6[0]))
        amax = np.maximum(amax, np.abs(d.actuator_force) / m.actuator_forcerange[:, 1])
        R3 = d.xmat[tb].reshape(3, 3)
        if t_fall is None and math.degrees(math.acos(np.clip(R3[2, 2], -1, 1))) > 45:
            t_fall = t - 0.5
        body_hits = [o for o in order if "foot_sole" not in o[1]]
        if shot and img is None and body_hits and t - 0.5 >= body_hits[0][0] + 0.03:
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [d.subtree_com[tb][0], d.subtree_com[tb][1], 0.1]; cam.distance = 1.0
            cam.azimuth = {"front": 90, "back": 90, "left": 0, "right": 0}[dname] + 180 * 0; cam.elevation = -15
            ren.update_scene(d, cam); img = ren.render().copy()
    return {"push_N": F, "push_s": 0.2, "fallen": t_fall is not None, "fall_time_s": t_fall, "first_floor_contacts": order[:8],
            "peak_contact_normal_N_reference": fmax_c,
            "peak_actuator_force_over_stall": {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): float(amax[i]) for i in range(m.nu)}}, img


out = {}; imgs = {}
for dn in DIRS:
    smallest = None
    for F in LADDER:
        r, _ = push(dn, F)
        if r["fallen"]:
            smallest = F; break
    r_min, img = push(dn, smallest, shot=True) if smallest else (None, None)
    r_hard, _ = push(dn, 30)
    out[dn] = {"smallest_toppling_push_N": smallest, "at_smallest": r_min, "hard_push_30N": r_hard}
    imgs[dn] = img
    top = sorted(r_hard["peak_actuator_force_over_stall"].items(), key=lambda x: -x[1])[:3]
    print(dn, "smallest push", smallest, "N; order", r_min and r_min["first_floor_contacts"][:5], "| hard", r_hard["first_floor_contacts"][:5],
          "peak contact", round(r_hard["peak_contact_normal_N_reference"]), "top act", [(a, round(b, 2)) for a, b in top])
jdump(out, "test6_falls.json")
plt = korean_font()
fig, ax = plt.subplots(1, 4, figsize=(16, 5))
KR = {"front": "앞", "back": "뒤", "left": "왼쪽", "right": "오른쪽"}
for a, dn in zip(ax, DIRS):
    if imgs[dn] is not None:
        a.imshow(imgs[dn])
    a.axis("off"); o = out[dn]["at_smallest"]
    first = ", ".join([g.replace("col_", "") for _, g in o["first_floor_contacts"] if "foot_sole" not in g][:3]) if o else "-"
    a.set_title(f"{KR[dn]}으로 {out[dn]['smallest_toppling_push_N']} N × 0.2 s\n발 다음에 처음 닿는 곳: {first}", fontsize=8)
fig.suptitle("서 있는 로봇은 8–12 N 을 0.2 s 밀면 넘어진다 — 가장 작은 밀기로는 앞·뒤·왼쪽 모두 반바지와 몸통 커버, 오른쪽은 머리 뒤와 귀가 먼저 닿고,\n30 N 으로 옆을 밀면 손과 손 날개가 먼저 닿는다 (발 아닌 부품이 처음 닿고 0.03 s 뒤 장면, 시뮬 실측)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(RES, "test6_falls.png"), dpi=100)
print("saved")
