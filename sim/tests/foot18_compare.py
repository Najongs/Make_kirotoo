"""v18 foot cut: compare theta_max 20 / 25 / 30 with v17 (support area, test1/2/3 results, design clearance).
Inputs: results_v18/{v17,t20,t25,t30}/test{1,2,3}*.json, results_v18/clearance_*.json, model/kitoru_v17*.xml,
model/kitoru_v18_t*.xml. Output: results_v18/theta_compare.json, theta_compare.md, foot_outline_compare.png
Run: cd sim && uv run python tests/foot18_compare.py
"""
import json, math, os, sys
import numpy as np
from common import *

R18 = os.path.join(SIM, "results_v18")
VARS = [("v17", "kitoru_v17", "v17 (자르기 전)"), ("t20", "kitoru_v18_t20", "θmax 20°"), ("t25", "kitoru_v18_t25", "θmax 25°"),
        ("t30", "kitoru_v18_t30", "θmax 30°")]
rows = {}; outlines = {}
for key, stem, lab in VARS:
    m = mujoco.MjModel.from_xml_path(os.path.join(SIM, "model", f"{stem}.xml"))
    pads, sole = foot_support(m, "R"); outlines[key] = (pads, sole)
    t1 = json.load(open(os.path.join(R18, key, "test1_static_hold.json")))["cases"]
    t2 = json.load(open(os.path.join(R18, key, "test2_gait_torque.json")))
    t3 = json.load(open(os.path.join(R18, key, "test3_openloop_walk.json")))["kp_x4"]
    cl = json.load(open(os.path.join(R18, f"clearance_{'v17' if key == 'v17' else key}.json")))
    tmax = 30 if key == "v17" else int(key[1:])
    cr = {int(k): v for k, v in cl["crouch"].items() if abs(int(k)) <= tmax}
    neg = [g for t, g in ((t, v["gap"]) for t, v in cr.items()) if t < 0]
    pos = [g for t, g in ((t, v["gap"]) for t, v in cr.items()) if t > 0]
    worst = min(cr.items(), key=lambda kv: kv[1]["gap"])
    # largest |theta| (each direction) with gap >= 3 mm for all smaller angles
    def lim(sign):
        t = 0
        while abs(t + sign) <= tmax and cr[t + sign]["gap"] >= 3.0:
            t += sign
        return t
    sc = {int(k): v for k, v in cl["scissor"].items()}
    rows[key] = {
        "label": lab, "mass_total_g": float(m.body_mass[1:].sum() * 1e3),
        "sole_area_mm2": hull_area(sole), "pad_hull_area_mm2": hull_area(pads),
        "test1_kp4_fallen": {c: t1[c]["hold_kp_x4"]["fallen"] for c in t1},
        "test1_com_margin_pads_mm": {c: t1[c]["com_margin_pads_mm"] for c in t1},
        "test1_com_margin_sole_mm": {c: t1[c]["com_margin_sole_mm"] for c in t1},
        "test2_cop_outside_sole_pct": {c: 100 * t2["cases"][c]["cop_outside_sole_frac"] for c in t2["cases"]},
        "test2_cop_outside_pads_pct": {c: 100 * t2["cases"][c]["cop_outside_pads_frac"] for c in t2["cases"]},
        "test3_kp4": {"fallen": t3["fallen"], "full_steps": t3["full_steps_survived"], "progress_mm": t3["com_x_progress_mm"],
                      "clearance_mm": t3["max_swing_foot_clearance_mm"], "single_support": t3["full_steps_support"]["single_support_frac"],
                      "yaw_deg": t3["torso_yaw_deg_full_steps"], "slip_mm": t3["max_slip_mm"]},
        "clearance_crouch_min_gap_mm": worst[1]["gap"], "clearance_worst": [worst[0], worst[1]["where"]],
        "clearance_ok_range_deg": [lim(-1), lim(1)],
        "clearance_scissor_min_gap_mm": {t: v["gap"] for t, v in sc.items() if abs(t) <= tmax},
        "sampling_error_mm": cl["sampling_error_mm"]}
base = rows["v17"]
for key in ("t20", "t25", "t30"):
    r = rows[key]
    r["sole_area_change_pct"] = 100 * (r["sole_area_mm2"] / base["sole_area_mm2"] - 1)
    r["pad_hull_area_change_pct"] = 100 * (r["pad_hull_area_mm2"] / base["pad_hull_area_mm2"] - 1)
    r["test2_cop_outside_sole_delta_pp"] = {c: r["test2_cop_outside_sole_pct"][c] - base["test2_cop_outside_sole_pct"][c] for c in base["test2_cop_outside_sole_pct"]}
    r["test2_cop_outside_pads_delta_pp"] = {c: r["test2_cop_outside_pads_pct"][c] - base["test2_cop_outside_pads_pct"][c] for c in base["test2_cop_outside_pads_pct"]}
    worse_fall = any(r["test1_kp4_fallen"][c] and not base["test1_kp4_fallen"][c] for c in base["test1_kp4_fallen"]) or r["test3_kp4"]["fallen"]
    r["support_ok"] = (not worse_fall) and max(r["test2_cop_outside_sole_delta_pp"].values()) <= 3.0
    r["clearance_ok"] = r["clearance_crouch_min_gap_mm"] >= 3.0
jdump(rows, "../results_v18/theta_compare.json")

KR = {"crouch": "앉기", "toe": "발끝", "heel": "뒤꿈치", "walk": "walk", "fast": "fast", "jog": "jog"}
L = ["| 항목 | v17 | θmax 20° | θmax 25° | θmax 30° |", "|---|---|---|---|---|"]
def row(name, f):
    L.append(f"| {name} | " + " | ".join(f(rows[k]) for k, _, _ in VARS) + " |")
row("발바닥 넓이 (mm², 변화)", lambda r: f"{r['sole_area_mm2']:.0f}" + (f" ({r['sole_area_change_pct']:+.1f}%)" if "sole_area_change_pct" in r else ""))
row("패드 볼록 껍질 넓이 (mm², 변화)", lambda r: f"{r['pad_hull_area_mm2']:.0f}" + (f" ({r['pad_hull_area_change_pct']:+.1f}%)" if "pad_hull_area_change_pct" in r else ""))
row("검사 1 이득 4배: 넘어짐 (앉기/발끝/뒤꿈치)", lambda r: "/".join("넘어짐" if r["test1_kp4_fallen"][c] else "버팀" for c in ("crouch", "toe", "heel")))
row("검사 1 무게중심 여유 패드 (mm)", lambda r: "/".join(f"{r['test1_com_margin_pads_mm'][c]:.1f}" for c in ("crouch", "toe", "heel")))
row("검사 2 압력중심 발바닥 밖 % (walk/fast/jog)", lambda r: "/".join(f"{r['test2_cop_outside_sole_pct'][c]:.1f}" for c in ("walk", "fast", "jog")))
row("검사 2 압력중심 패드 껍질 밖 %", lambda r: "/".join(f"{r['test2_cop_outside_pads_pct'][c]:.1f}" for c in ("walk", "fast", "jog")))
row("검사 3 이득 4배 10걸음", lambda r: ("넘어짐" if r["test3_kp4"]["fallen"] else "버팀") + f", {r['test3_kp4']['progress_mm']:.0f} mm 전진, 요 {r['test3_kp4']['yaw_deg'][0]:.0f}~{r['test3_kp4']['yaw_deg'][1]:.0f}°")
row("설계 형상 최소 간격 (앉기 격자, ±θmax)", lambda r: f"{r['clearance_crouch_min_gap_mm']:.2f} mm ({r['clearance_worst'][0]:+d}°, {r['clearance_worst'][1][0].replace('R_', '')}–{r['clearance_worst'][1][2].replace('L_', '')})")
row("3 mm 지키는 요 범위 (앉기 격자)", lambda r: f"{r['clearance_ok_range_deg'][0]}° ~ +{r['clearance_ok_range_deg'][1]}°")
open(os.path.join(R18, "theta_compare.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
for k in ("t20", "t25", "t30"):
    print(k, "support_ok", rows[k]["support_ok"], "clearance_ok", rows[k]["clearance_ok"])

# figure: right-foot outlines (sole bottom hull and pad hull), v17 vs variants, plus the guide circle
plt = korean_font()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for a, key in zip(ax, ("t20", "t25", "t30")):
    for k2, col, ls, lab in (("v17", "#999999", "--", "v17"), (key, "#2a6fdb", "-", rows[key]["label"])):
        pads, sole = outlines[k2]
        a.fill(np.r_[sole[:, 0], sole[0, 0]], np.r_[sole[:, 1], sole[0, 1]], color=col, alpha=0.12 if k2 == key else 0.0)
        a.plot(np.r_[sole[:, 0], sole[0, 0]], np.r_[sole[:, 1], sole[0, 1]], color=col, ls=ls, lw=1.8, label=f"발바닥 ({lab})")
        a.plot(np.r_[pads[:, 0], pads[0, 0]], np.r_[pads[:, 1], pads[0, 1]], color="#e07b39" if k2 == key else "#cfa27f", ls=ls, lw=1.4,
               label=f"패드 볼록 껍질 ({lab})")
    t = int(key[1:])
    th = np.linspace(-math.radians(t), math.radians(t), 50)
    a.plot(42 * np.sin(th), -40 + 42 * np.cos(th), color="#c0392b", lw=1, ls=":", label="r 42 안내 원 (±θmax)")
    a.axhline(5, color="#555", lw=0.8); a.text(-70, 6.5, "왼발 안쪽 끝 (y +5)", fontsize=8, color="#555")
    a.plot([0], [-40], "k+", ms=10); a.set_aspect("equal"); a.grid(alpha=0.3); a.set_xlabel("x (mm, 앞)")
    r = rows[key]
    a.set_title(f"{r['label']}: 발바닥 넓이 {r['sole_area_change_pct']:+.0f}%, 패드 볼록 껍질 {r['pad_hull_area_change_pct']:+.0f}%", fontsize=10)
ax[0].set_ylabel("y (mm, 왼쪽)"); ax[0].legend(fontsize=7, loc="lower left")
fig.suptitle("오른발 안쪽 앞뒤 모서리를 자른 모양 — 자를수록 패드 볼록 껍질이 크게 줄고, 25·30° 는 발목 브래킷끼리 먼저 닿아 요 범위를 다 못 쓴다\n"
             "(바닥에서 본 오른발, 점선 회색 = v17, 설계 형상 + 시뮬 모델)", fontsize=11)
fig.subplots_adjust(left=0.05, right=0.99, bottom=0.12, top=0.80, wspace=0.05); fig.savefig(os.path.join(R18, "foot_outline_compare.png"), dpi=110)
print("saved")
