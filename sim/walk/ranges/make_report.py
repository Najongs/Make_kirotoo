"""joint_ranges.py 결과 → 표 · 그림 · csv · MJCF range.
usage: cd sim/walk && uv run python ranges/make_report.py --tag v19 --joints-csv ../../out19/joints.csv
결과: ranges/results/<tag>/joint_ranges_<tag>.csv, mjcf_ranges_<tag>.txt, tables_<tag>.md, fig_single_<tag>.png, fig_pairs_<tag>.png, fig_lying_<tag>.png"""
import os, sys, json, math, argparse, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.dirname(HERE))
from kw_common import korean_font

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v19")
ap.add_argument("--joints-csv", default=os.path.join(ROOT, "out19", "joints.csv"))
ap.add_argument("--margin-deg", type=float, default=5.0)
ap.add_argument("--margin-mm", type=float, default=3.0)
ap.add_argument("--work-extra-deg", type=float, default=5.0)
A = ap.parse_args()
OUT = os.path.join(HERE, "results", A.tag)
single = json.load(open(os.path.join(OUT, "single.json")))
pos = json.load(open(os.path.join(OUT, "posture.json")))
PZ = np.load(os.path.join(OUT, "pairs.npz"))
rows_csv = list(csv.DictReader(open(A.joints_csv)))
CSV = {r["joint"]: r for r in rows_csv}
SERVO_HALF = 180.0          # 위치 모드 0–4095 = 한 바퀴. 영점을 2048 에 두면 ±180°


def dxl(theta, d=1, off=0):
    """각(도) → Dynamixel 위치값: raw = 2048 + off + d · θ · 4096/360 (d = ±1 회전 방향, off = 조립 영점 오프셋)"""
    return int(round(2048 + off + d * theta * 4096.0 / 360.0))


S = {}
for r in single:
    S.setdefault(r["joint"], {})[r["dir"]] = r
table = []
for jn, d in S.items():
    lo, hi = d[-1], d[1]
    def marg(r):
        if r["radius_mm"] and r["first_hit"] is not None:
            return max(A.margin_deg, math.degrees(A.margin_mm / max(r["radius_mm"], 1.0)))
        return A.margin_deg
    m_lo, m_hi = marg(lo), marg(hi)
    shape_lo, shape_hi = lo["limit"], hi["limit"]
    soft_lo = min(0.0, shape_lo + m_lo) if shape_lo > -SERVO_HALF else -SERVO_HALF + A.margin_deg
    soft_hi = max(0.0, shape_hi - m_hi) if shape_hi < SERVO_HALF else SERVO_HALF - A.margin_deg
    work_lo = min(0.0, soft_lo + A.work_extra_deg); work_hi = max(0.0, soft_hi - A.work_extra_deg)
    c = CSV.get(jn, {})
    # 작업 범위 손질(초안): 무릎은 뒤로 꺾는 쪽(음수)을 쓰지 않음(편 무릎 = 특이점), 머리 끄덕임은 크랭크 고리가 정한 범위(joints.csv) 안
    if jn.endswith("_knee"):
        work_lo = 0.0
    if jn == "head_nod" and c:
        work_lo = max(work_lo, float(c["lower_deg"])); work_hi = min(work_hi, float(c["upper_deg"]))
    table.append(dict(joint=jn, shape_lo=shape_lo, shape_hi=shape_hi, hit_lo=lo["pair"], hit_hi=hi["pair"],
                      r_lo=lo["radius_mm"], r_hi=hi["radius_mm"], m_lo=m_lo, m_hi=m_hi, soft_lo=soft_lo, soft_hi=soft_hi,
                      work_lo=work_lo, work_hi=work_hi, csv_lo=float(c["lower_deg"]) if c else None, csv_hi=float(c["upper_deg"]) if c else None,
                      dxl_min=dxl(soft_lo), dxl_max=dxl(soft_hi)))

# ---- 짝 지도: 관절 A 의 자유 범위가 B 에 따라 어떻게 바뀌나 + 보수적 범위
pairs = {}
for k in PZ.files:
    name, f = k.split("|")
    pairs.setdefault(name, {})[f] = PZ[k]
W = {t["joint"]: (t["work_lo"], t["work_hi"]) for t in table}
coupled = []
for name, p in pairs.items():
    ja, jb = [str(x) for x in p["joints"]]
    Aa, Bb, F = p["A"], p["B"], p["free"]
    def run_through_zero(row, grid):
        """0 을 포함하는 연속 자유 구간 [lo, hi] (0 이 막혀 있으면 None)"""
        i0 = int(np.argmin(np.abs(grid)))
        if not row[i0]:
            return None
        i, k = i0, i0
        while i - 1 >= 0 and row[i - 1]: i -= 1
        while k + 1 < len(row) and row[k + 1]: k += 1
        return float(grid[i]), float(grid[k])
    # B 를 고정했을 때 A 범위
    var_a = []
    for kb, b in enumerate(Bb):
        rr = run_through_zero(F[:, kb], Aa)
        var_a.append((float(b), rr))
    # A 를 B 의 작업 범위 안 어디에 두어도 충돌 없는 범위(보수)
    def conservative(F_, grid_main, grid_other, wr):
        if wr is None:
            return None
        sel = (grid_other >= wr[0] - 1e-9) & (grid_other <= wr[1] + 1e-9)
        rowall = F_[:, sel].all(axis=1)
        return run_through_zero(rowall, grid_main)
    ca = conservative(F, Aa, Bb, W.get(jb)) if not name.startswith("두 다리") else None
    cb = conservative(F.T, Bb, Aa, W.get(ja)) if not name.startswith("두 다리") else None
    coupled.append(dict(name=name, ja=ja, jb=jb, conservative_a=ca, conservative_b=cb, free_frac=float(F.mean()),
                        a_range_vs_b=[(b, r) for b, r in var_a]))

# ---- csv · MJCF
with open(os.path.join(OUT, f"joint_ranges_{A.tag}.csv"), "w", newline="") as f:
    cols = list(rows_csv[0].keys()) + ["shape_lower_deg", "shape_upper_deg", "soft_lower_deg", "soft_upper_deg",
                                       "work_lower_deg", "work_upper_deg", "dxl_min_raw", "dxl_max_raw", "note"]
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    T = {t["joint"]: t for t in table}
    for r in rows_csv:
        rr = dict(r)
        t = T.get(r["joint"])
        if t:
            rr.update(lower_deg=f"{t['soft_lo']:.1f}", upper_deg=f"{t['soft_hi']:.1f}", shape_lower_deg=f"{t['shape_lo']:.1f}",
                      shape_upper_deg=f"{t['shape_hi']:.1f}", soft_lower_deg=f"{t['soft_lo']:.1f}", soft_upper_deg=f"{t['soft_hi']:.1f}",
                      work_lower_deg=f"{t['work_lo']:.1f}", work_upper_deg=f"{t['work_hi']:.1f}", dxl_min_raw=t["dxl_min"], dxl_max_raw=t["dxl_max"],
                      note="초안: lower/upper = 소프트웨어 한계(형상 한계 − max(5°, 3 mm)), 관절 하나만 움직인 값")
        else:
            rr.update(note="이번 검사에서 안 봄(수동 · 고리 관절은 joints.csv 값 그대로)")
        w.writerow(rr)
with open(os.path.join(OUT, f"mjcf_ranges_{A.tag}.txt"), "w") as f:
    f.write("<!-- 초안: 소프트웨어 한계(rad). MJCF <joint name=... range=...> 에 넣는 값 -->\n")
    for t in table:
        f.write(f'<joint name="{t["joint"]}" range="{math.radians(t["soft_lo"]):.6f} {math.radians(t["soft_hi"]):.6f}"/>\n')

# ---- 표(md)
md = ["| 관절 | joints.csv 지금 (°) | 형상 한계 − 쪽 (°) / 처음 닿는 곳 | 형상 한계 + 쪽 (°) / 처음 닿는 곳 | 여유 −/+ (°) | 소프트웨어 한계 초안 (°) | 작업 범위 초안 (°) | Dynamixel Min / Max (영점 2048, 정방향) |",
      "|---|---|---|---|---|---|---|---|"]
for t in table:
    md.append(f"| {t['joint']} | {t['csv_lo']:.0f} ~ {t['csv_hi']:.0f} | {t['shape_lo']:.1f} / {t['hit_lo']} | {t['shape_hi']:.1f} / {t['hit_hi']} | "
              f"{t['m_lo']:.1f} / {t['m_hi']:.1f} | {t['soft_lo']:.1f} ~ {t['soft_hi']:.1f} | {t['work_lo']:.1f} ~ {t['work_hi']:.1f} | {t['dxl_min']} / {t['dxl_max']} |")
md.append("")
md.append("| 짝 | 자유 영역 비율 | 관절 A 보수 범위(B 가 작업 범위 어디에 있어도) | 관절 B 보수 범위(A 가 작업 범위 어디에 있어도) |")
md.append("|---|---|---|---|")
for c in coupled:
    f_ = lambda r: "–" if r is None else f"{r[0]:.0f} ~ {r[1]:.0f}"
    md.append(f"| {c['name']} ({c['ja']} × {c['jb']}) | {100 * c['free_frac']:.0f} % | {f_(c['conservative_a'])} | {f_(c['conservative_b'])} |")
cr = pos["crouch"]
md.append("")
fb = cr["first_collision"]
md.append(f"- 두 발 평평히 웅크리기(고관절 피치 = 발목 피치 = −무릎/2): 형상으로는 무릎 {fb['knee'] - 2 if fb else 160:.0f}° 까지 충돌 없음"
          + (f" (무릎 {fb['knee']:.0f}° 에서 {fb['hit']})" if fb else "") +
          f"; joints.csv 범위로는 무릎 {cr['knee_max_by_joints_csv']:.0f}° = 웅크림 {cr['crouch_max_by_joints_csv_mm']:.0f} mm 까지")
open(os.path.join(OUT, f"tables_{A.tag}.md"), "w").write("\n".join(md) + "\n")
json.dump(dict(table=table, coupled=coupled), open(os.path.join(OUT, f"summary_{A.tag}.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("\n".join(md))

# ---- 그림
plt = korean_font()
# 1) 관절 하나씩
fig, ax = plt.subplots(figsize=(12, 0.45 * len(table) + 2))
for i, t in enumerate(table):
    y = len(table) - 1 - i
    ax.barh(y, t["shape_hi"] - t["shape_lo"], left=t["shape_lo"], color="#d6eaf8", edgecolor="#2e86c1", height=0.8, label="형상 한계" if i == 0 else None)
    ax.barh(y, t["soft_hi"] - t["soft_lo"], left=t["soft_lo"], color="#5dade2", height=0.5, label="소프트웨어 한계 초안" if i == 0 else None)
    ax.barh(y, t["work_hi"] - t["work_lo"], left=t["work_lo"], color="#1b4f72", height=0.25, label="작업 범위 초안" if i == 0 else None)
    if t["csv_lo"] is not None:
        ax.plot([t["csv_lo"], t["csv_hi"]], [y + 0.45, y + 0.45], color="#e67e22", lw=2.5, label="joints.csv 지금" if i == 0 else None)
ax.set_yticks(range(len(table))); ax.set_yticklabels([t["joint"] for t in table][::-1]); ax.axvline(0, color="k", lw=0.6)
ax.set_xlim(-185, 185); ax.set_xlabel("관절 각 (°)"); ax.grid(alpha=0.3, axis="x"); ax.legend(fontsize=8, loc="lower right")
nlt = sum(1 for t in table if t["csv_lo"] is not None and (t["csv_lo"] < t["soft_lo"] - 0.1 or t["csv_hi"] > t["soft_hi"] + 0.1))
ax.set_title(f"관절 하나만 움직였을 때 형상 한계와 안전 범위 초안 ({A.tag}, 설계 형상 정밀 검사) — joints.csv 범위가 소프트웨어 한계를 넘는 관절 {nlt}개", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_single_{A.tag}.png"), dpi=90); plt.close(fig)
# 2) 짝 지도
names = list(pairs)
nc = 4; nr = int(math.ceil(len(names) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.0 * nr))
for ax, name in zip(axs.ravel(), names):
    p = pairs[name]; ja, jb = [str(x) for x in p["joints"]]
    Aa, Bb, F = p["A"], p["B"], p["free"]
    ax.imshow(np.where(F.T, 1.0, 0.0), origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
              extent=[Aa[0] - (Aa[1] - Aa[0]) / 2, Aa[-1] + (Aa[1] - Aa[0]) / 2, Bb[0] - (Bb[1] - Bb[0]) / 2, Bb[-1] + (Bb[1] - Bb[0]) / 2])
    for jn, style, col in ((ja, "csv", "#e67e22"), (jb, "csv", "#e67e22")):
        pass
    ca, cb = CSV.get(ja), CSV.get(jb)
    if ca and cb:
        ax.add_patch(plt.Rectangle((float(ca["lower_deg"]), float(cb["lower_deg"])), float(ca["upper_deg"]) - float(ca["lower_deg"]),
                                   float(cb["upper_deg"]) - float(cb["lower_deg"]), fill=False, ec="#1b4f72", lw=1.5, ls="--"))
    ax.set_xlabel(f"{ja} (°)"); ax.set_ylabel(f"{jb} (°)"); ax.set_title(name, fontsize=9)
for ax in axs.ravel()[len(names):]:
    ax.axis("off")
fig.suptitle(f"짝 관절 충돌 지도 ({A.tag}): 초록 = 충돌 없음, 빨강 = 자기 충돌, 점선 = joints.csv 범위 — 다른 관절 각도에 따라 한계가 달라진다 (설계 형상 정밀 검사)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(OUT, f"fig_pairs_{A.tag}.png"), dpi=80); plt.close(fig)
# 3) 누운 자세
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
for ax, (nm, L) in zip(axs, pos["lying"].items()):
    for jn, col in (("R_arm_swing", "#2e86c1"), ("L_arm_swing", "#85c1e9"), ("R_hip_pitch", "#c0392b"), ("R_knee", "#e67e22"), ("R_ankle_pitch", "#27ae60")):
        if jn in L:
            r = np.array(L[jn], dtype=object)
            a = r[:, 0].astype(float); z = r[:, 1].astype(float); okk = r[:, 2].astype(bool)
            ax.plot(a, z, color=col, label=jn)
            ax.scatter(a[~okk], z[~okk], color=col, marker="x", s=25)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-80, 120); ax.grid(alpha=0.3)
    ax.set_xlabel("관절 각 (°) — × = 자기 충돌"); ax.set_ylabel("움직이는 부분의 가장 낮은 점 − 바닥 (mm)")
    ax.set_title(f"{nm}: 받치는 곳 {', '.join(x[0] for x in L['resting_links'][:2])}", fontsize=10); ax.legend(fontsize=8)
fig.suptitle(f"누운 자세에서 관절을 돌리면 0 아래로 내려가는 각도부터 바닥을 민다 (일어나기에 쓰는 구간) — {A.tag}, 몸을 통째로 눕힌 근사, 설계 형상", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(OUT, f"fig_lying_{A.tag}.png"), dpi=90); plt.close(fig)
print("saved")
