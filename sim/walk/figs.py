"""보고서 그림 · 표: results/*.json 을 읽고, 시계열이 필요한 몇 개는 다시 돌린다.
python figs.py → results/fig_*.png, results/tables.md, results/walk_side_0.076mps.mp4"""
import json, os, math
from multiprocessing import Pool
import numpy as np
from kw_common import RES, korean_font, jdump
from run_tests import BASE, VARIANTS

J = lambda n: json.load(open(os.path.join(RES, f"{n}.json")))
GROUPS = [("hip_yaw", "고관절 요 (XC430)"), ("hip_pitch", "고관절 피치 (XM430)"), ("knee", "무릎 (XM430)"),
          ("ankle_pitch", "발목 피치 (XC430)"), ("ankle_roll", "발목 롤 (XM430)"), ("nod_crank", "끄덕임 크랭크 (XL430)"),
          ("hip_roll", "고관절 롤 (XM430, 가상)")]


def series_job(args):
    from kw_sim import run
    name, cfg = args
    c = dict(BASE); c.update(cfg)
    o, R, m = run(**c)
    keep = {k: R[k] for k in ("t", "com", "c_est", "c_ref", "p_ref", "p_d", "cop", "margin", "rpy", "u", "crank_tau", "head_deg", "ik_roll")}
    keep["names"] = [m.actuator(i).name for i in range(m.nu)]
    return name, o, keep


def jgroup(d, key, stat=max):
    return stat(v for k, v in d.items() if k.endswith(key)) if any(k.endswith(key) for k in d) else None


def util_rows(o):
    out = {}
    for g, _ in GROUPS:
        if jgroup(o["u_max"], g) is None:
            continue
        out[g] = (jgroup(o["force_over_stall_max"], g), jgroup(o["u_max"], g), jgroup(o["u_p99"], g), jgroup(o["sat_frac"], g),
                  jgroup(o["u_over1_frac"], g))
    return out


def main():
    plt = korean_font()
    md = []
    # ------------------------------------------------ 시계열 다시 돌리기
    jobs = [("walk46", dict(stride=0.03)), ("walk46_nod1", dict(stride=0.03, nod=(1.0, 3.0))),
            ("walk46_nod05", dict(stride=0.03, nod=(0.5, 3.0))), ("walk76", dict(stride=0.05)),
            ("walk83", dict(stride=0.055)), ("walk90_fall", dict(stride=0.06)),
            ("walk46_noff", dict(stride=0.03, ff=False, dyn_ff=False)), ("walk46_nodynff", dict(stride=0.03, dyn_ff=False)),
            ("walk46_k0", dict(stride=0.03, k_dcm=0.0))]
    with Pool(len(jobs)) as p:
        S = {n: (o, R) for n, o, R in p.map(series_job, jobs)}
    from kw_sim import run
    c = dict(BASE); c.update(stride=0.05, video=os.path.join(RES, "walk_side_0.076mps.mp4"), eye_max_s=10.0)
    c["eye_t0"] = 2.0
    run(**c)

    # ------------------------------------------------ 그림 1: 걷기 한 판(0.046 m/s) 시계열
    o, R = S["walk46"]
    t5 = np.arange(len(R["com"])) * 5e-3; tc = R["t"]
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    ax[0].plot(t5, np.degrees(R["rpy"][:, 0]), label="몸통 롤(실제)", color="#2a6fdb")
    ax[0].plot(tc, np.degrees(R["ik_roll"]), label="몸통 롤(역기구학 목표)", color="#2a6fdb", ls="--", lw=1)
    ax[0].plot(t5, np.degrees(R["rpy"][:, 1]), label="몸통 피치(실제)", color="#e07b39")
    ax[0].set_ylabel("각도 (°)"); ax[0].legend(fontsize=9, ncol=3); ax[0].grid(alpha=.3)
    ax[1].plot(t5, R["com"][:, 1] * 1e3, label="무게중심 y(실제)", color="#2a6fdb")
    ax[1].plot(tc, R["c_est"][:, 1] * 1e3, label="무게중심 y(센서로 추정)", color="#2a6fdb", ls=":", lw=1.2)
    ax[1].plot(tc, R["c_ref"][:, 1] * 1e3, label="무게중심 y(계획)", color="black", lw=0.8)
    ax[1].plot(tc, R["p_ref"][:, 1] * 1e3, label="압력중심 y(계획)", color="#999", ls="--", lw=0.8)
    ax[1].plot(t5, R["cop"][:, 1] * 1e3, label="압력중심 y(실제)", color="#c0392b", lw=0.6)
    ax[1].set_ylabel("옆 위치 (mm)"); ax[1].legend(fontsize=8, ncol=5); ax[1].grid(alpha=.3)
    ax[2].plot(t5, R["margin"] * 1e3, color="#16a085", lw=0.8)
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_ylim(-5, 45)
    ax[2].set_ylabel("압력중심 여유 (mm)\n닿은 패드 껍질 가장자리까지"); ax[2].grid(alpha=.3)
    for i, nm in enumerate(R["names"]):
        if nm.startswith("R_") and not nm.endswith("arm_swing"):
            ax[3].plot(t5, R["u"][:, i], label=nm.replace("R_", "오른 "), lw=0.8)
    ax[3].axhline(1, color="k", lw=0.6); ax[3].set_ylim(0, 1.8); ax[3].set_ylabel("사용률 u"); ax[3].legend(fontsize=8, ncol=5); ax[3].grid(alpha=.3)
    ax[3].set_xlabel("시간 (s) — 0–1.5 s 웅크림, 1.5–2.3 s 첫 무게 옮김, 그 뒤 18걸음(0.65 s/걸음)")
    fig.suptitle(f"보통 걸음(0.046 m/s)에서 몸통이 좌우로 ±{max(abs(np.degrees(R['rpy'][:,0]).min()), np.degrees(R['rpy'][:,0]).max()):.0f}° 기울며 "
                 f"무게를 옮기고, 압력중심 여유 하위 5 % 는 {o['cop_margin_mm']['p5']:.0f} mm 이다\n"
                 f"(기본 제어기: 이득 2배 · 200 Hz · 지연 10 ms, 시뮬 실측)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(RES, "fig_walk_timeseries.png"), dpi=90); plt.close(fig)

    # ------------------------------------------------ 그림 2: 속도 · 이득 · 제어 주기
    sp = J("speed")["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    cols = {1.0: "#7f8c8d", 2.0: "#2a6fdb", 4.0: "#e07b39"}
    for kp in (1.0, 2.0, 4.0):
        rs = sorted([r for r in sp if r["meta"]["kp"] == kp and r["meta"]["hz"] == 200 and r["meta"]["delay"] == 10.0], key=lambda r: r["meta"]["stride"])
        v = [r["meta"]["stride"] / (BASE["Ts"] + BASE["Td"]) for r in rs]
        ok = [not r["res"]["fallen"] for r in rs]
        stp = [r["res"]["steps_done"] for r in rs]
        ax[0].plot(v, stp, "-", color=cols[kp], lw=1)
        ax[0].scatter([a for a, b in zip(v, ok) if b], [s for s, b in zip(stp, ok) if b], color=cols[kp], marker="o", label=f"이득 {kp:.0f}배 — 버팀")
        ax[0].scatter([a for a, b in zip(v, ok) if not b], [s for s, b in zip(stp, ok) if not b], color=cols[kp], marker="x", s=60, label=f"이득 {kp:.0f}배 — 넘어짐")
    ax[0].set_xlabel("명령 속도 (m/s) = 보폭 / 0.65 s"); ax[0].set_ylabel("넘어지기 전 걸음 수 (18 = 끝까지)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    labels = []; vals = []
    for hz, dl in ((200, 0.0), (200, 10.0), (100, 0.0), (100, 10.0)):
        rs = [r for r in sp if r["meta"]["kp"] == 2.0 and r["meta"]["hz"] == hz and r["meta"]["delay"] == dl and not r["res"]["fallen"]]
        vmax = max(r["meta"]["stride"] for r in rs) / (BASE["Ts"] + BASE["Td"])
        labels.append(f"{hz} Hz\n지연 {dl:.0f} ms"); vals.append(vmax)
    b = ax[1].bar(labels, vals, color=["#2a6fdb", "#5b8def", "#e07b39", "#f0a36b"])
    for bb, v in zip(b, vals):
        ax[1].text(bb.get_x() + bb.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)
    ax[1].set_ylabel("넘어지지 않은 가장 빠른 속도 (m/s)"); ax[1].set_ylim(0, 0.1); ax[1].grid(alpha=.3, axis="y")
    ax[1].set_title("이득 2배에서 제어 주기 · 명령 지연 비교", fontsize=10)
    fig.suptitle("앞으로 걷기: 이득 2 · 4배는 명령 0.085 m/s(보폭 55 mm, 실측 평균 0.083)까지 18걸음을 끝까지 가고, 이득 1배는 0.046 m/s 까지만 간다\n"
                 "명령 지연 10 ms 는 결과를 바꾸지 않고, 제어 주기 100 Hz 는 최고 속도를 명령 0.077 m/s 로 낮춘다 (걸음 주기 0.65 s 고정, 이득 2배와 4배 점은 겹침, 시뮬 실측)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(os.path.join(RES, "fig_speed.png"), dpi=90); plt.close(fig)

    # ------------------------------------------------ 그림 3: 밀기 · 마찰
    pu = J("push")["rows"]
    cases = [("+y", "ss", "옆(흔드는 발 쪽)\n한 발 지지 중"), ("-y", "ss", "옆(디딤발 바깥쪽)\n한 발 지지 중"),
             ("+y", "ds", "옆(+y)\n착지 · 두 발 지지"), ("-y", "ds", "옆(−y)\n착지 · 두 발 지지"),
             ("+x", "ss", "앞으로\n한 발 지지 중"), ("-x", "ss", "뒤로\n한 발 지지 중"),
             ("+y", "stand", "옆(+y)\n서 있기"), ("+x", "stand", "앞으로\n서 있기"), ("-x", "stand", "뒤로\n서 있기")]
    def thr(fr, dr, wh):
        rs = sorted([r for r in pu if r["meta"].get("dir") == dr and r["meta"].get("where") == wh and r["meta"]["friction"] == fr], key=lambda r: r["meta"]["F"])
        if not rs:
            return None, None
        firstfail = next((r["meta"]["F"] for r in rs if r["res"]["fallen"]), None)
        below = [r["meta"]["F"] for r in rs if not r["res"]["fallen"] and (firstfail is None or r["meta"]["F"] < firstfail)]
        return (max(below) if below else 0), firstfail
    fig, ax = plt.subplots(figsize=(14, 5.5))
    W_ = 0.26
    for k, (fr, col) in enumerate(((0.5, "#7f8c8d"), (0.8, "#2a6fdb"), (1.0, "#e07b39"))):
        xs, ys = [], []
        for i, (dr, wh, lab) in enumerate(cases):
            s, f = thr(fr, dr, wh)
            if s is None:
                continue
            xs.append(i + (k - 1) * W_); ys.append(s)
        ax.bar(xs, ys, W_, color=col, label=f"마찰 {fr}")
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.2, f"{y}", ha="center", fontsize=8)
    ax.set_xticks(range(len(cases))); ax.set_xticklabels([c[2] for c in cases], fontsize=9)
    ax.set_ylabel("버틴 가장 큰 밀기 (N, 0.2 s)\n(그보다 약한 밀기는 모두 버팀)"); ax.grid(alpha=.3, axis="y"); ax.legend()
    ax.text(0.99, 0.95, "충격량 = 힘 × 0.2 s, 4 N → 0.8 N·s → 몸 전체 속도 0.27 m/s", transform=ax.transAxes, ha="right", fontsize=9)
    a_, _ = thr(0.8, "-y", "ss"); b_, _ = thr(0.8, "+y", "ss"); c_, _ = thr(0.8, "+x", "ss")
    fig.suptitle(f"제자리 걸음 중 옆에서 밀면, 디딤발 바깥쪽으로 미는 경우 {a_} N(0.2 s)까지만 버틴다 — 흔드는 발 쪽은 {b_} N, 앞으로는 {c_} N\n"
                 "마찰 0.5–1.0 은 결과를 바꾸지 않는다(미끄러짐보다 발 가장자리로 넘어가는 것이 먼저, 시뮬 실측)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(RES, "fig_push.png"), dpi=90); plt.close(fig)

    # ------------------------------------------------ 그림 4: 머리 끄덕임
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for key, lab, col in (("walk46", "끄덕임 없음", "#7f8c8d"), ("walk46_nod05", "0.5 Hz", "#2a6fdb"), ("walk46_nod1", "1 Hz", "#e07b39")):
        o, R = S[key]; t5 = np.arange(len(R["com"])) * 5e-3
        ax[0].plot(t5, R["head_deg"], color=col, lw=1, label=lab)
        ax[1].plot(t5, R["crank_tau"], color=col, lw=0.8, label=lab)
        ax[2].plot(t5, np.degrees(R["rpy"][:, 1]), color=col, lw=0.9, label=lab)
    ax[0].set_ylabel("머리 각 (°, + 아래)"); ax[1].set_ylabel("XL430 크랭크 힘 (N·m)\n정지토크 1.4"); ax[2].set_ylabel("몸통 피치 (°)")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=9, ncol=3)
    ax[2].set_xlabel("시간 (s) — 3 s 부터 끄덕임(1 초에 걸쳐 진폭을 키움)")
    nd = {(r["meta"]["stride"], r["meta"]["nod"]): r["res"] for r in J("nod")["rows"]}
    fig.suptitle(f"걸으며(0.046 m/s) 머리를 −10~+20° 로 끄덕여도 넘어지지 않고 몸통 피치는 ±{max(abs(nd[(0.03, 1.0)]['torso_pitch_deg'][0]), nd[(0.03, 1.0)]['torso_pitch_deg'][1]):.1f}° 안이다\n"
                 f"크랭크 부하 RMS: 0.5 Hz {nd[(0.03, 0.5)]['crank_tau_rms']:.2f} · 1 Hz {nd[(0.03, 1.0)]['crank_tau_rms']:.2f} N·m (정지토크 1.4, 시뮬 실측)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(RES, "fig_nod.png"), dpi=90); plt.close(fig)

    # ------------------------------------------------ 그림 5: 제어기 부품 빼기(같은 걸음)
    names = [("walk46", "전부 켬"), ("walk46_k0", "DCM 되먹임 끔"), ("walk46_nodynff", "속도 · 가속도 앞먹임 끔"), ("walk46_noff", "휨 보상 · 앞먹임 모두 끔")]
    abl = []
    for k, lab in names:
        o, R = S[k]
        abl.append((lab, o["fallen"], o["steps_done"], o["cop_margin_mm"]["p5"], o["fall_t"]))

    # ------------------------------------------------ 그림 6: 설계 변형 비교
    va = J("variants")["rows"]
    vn = list(VARIANTS)
    vmax, pin, pout, roll, mg = [], [], [], [], []
    for v in vn:
        rs = [r for r in va if r["meta"]["variant"] == v]
        sp_ok = [r["meta"]["stride"] for r in rs if r["meta"]["test"] == "speed" and not r["res"]["fallen"]]
        vmax.append(max(sp_ok) / (BASE["Ts"] + BASE["Td"]) if sp_ok else 0.03 / 0.65)
        for dr, L in (("+y", pin), ("-y", pout)):
            pr = sorted([r for r in rs if r["meta"]["test"] == "push" and r["meta"]["dir"] == dr], key=lambda r: r["meta"]["F"])
            ff = next((r["meta"]["F"] for r in pr if r["res"]["fallen"]), None)
            L.append(max([r["meta"]["F"] for r in pr if not r["res"]["fallen"] and (ff is None or r["meta"]["F"] < ff)] or [0]))
        nr = next(r for r in rs if r["meta"]["test"] == "nod")["res"]
        roll.append(max(abs(nr["torso_roll_deg"][0]), abs(nr["torso_roll_deg"][1]))); mg.append(nr["cop_margin_mm"]["p5"])
    fig, ax = plt.subplots(1, 4, figsize=(17, 5.5), sharey=True)
    yy = np.arange(len(vn))
    for a, vals, lab, fmt in ((ax[0], vmax, "넘어지지 않은 최고 속도 (m/s)", "{:.3f}"), (ax[1], pout, "디딤발 바깥쪽 밀기 버팀 (N, 0.2 s)", "{:.0f}"),
                              (ax[2], roll, "끄덕이며 걸을 때 몸통 롤 최대 (°)", "{:.1f}"), (ax[3], mg, "끄덕이며 걸을 때 압력중심 여유 하위 5 % (mm)", "{:.1f}")):
        cs = ["#2a6fdb" if i == 0 else "#95a5a6" for i in range(len(vn))]
        a.barh(yy, vals, color=cs)
        for y_, v in zip(yy, vals):
            a.text(v, y_, " " + fmt.format(v), va="center", fontsize=9)
        a.set_title(lab, fontsize=10); a.grid(alpha=.3, axis="x")
    ax[0].set_yticks(yy); ax[0].set_yticklabels(vn); ax[0].invert_yaxis()
    i6 = vn.index("바깥 패드 6 mm 바깥으로"); i10 = vn.index("바깥 패드 10 mm 바깥으로")
    fig.suptitle(f"시뮬에서 바꿔 본 설계: 고관절 롤을 더하면 몸통 롤이 12° → 3° 로 줄고, 바깥 패드를 6 · 10 mm 옮기면 바깥쪽 밀기 {pout[0]} → {pout[i6]} · {pout[i10]} N, "
                 "회전자 관성이 절반이면 최고 명령 속도 0.085 → 0.092 m/s\n머리 질량 −20 % 는 이 시험들에서 차이가 거의 없다 (같은 제어기, 고관절 롤은 옆 발 디딤을 안 쓴 채 비교, 시뮬 실측)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(os.path.join(RES, "fig_variants.png"), dpi=90); plt.close(fig)

    # ------------------------------------------------ 표들
    md.append("## 앞으로 걷기 (걸음 주기 0.65 s = 한 발 0.45 + 두 발 0.2, 18걸음)\n")
    md.append("| 보폭 (mm) | 명령 속도 (m/s) | 이득 1배 | 이득 2배 · 200 Hz · 지연 0 | 이득 2배 · 200 Hz · 지연 10 ms | 이득 2배 · 100 Hz · 지연 0 | 이득 2배 · 100 Hz · 지연 10 ms | 이득 4배 | 실측 평균 속도 (이득 2배, 200 Hz, 10 ms) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted({r["meta"]["stride"] for r in sp}):
        def cell(kp, hz, dl):
            r = next(r for r in sp if r["meta"]["stride"] == s and r["meta"]["kp"] == kp and r["meta"]["hz"] == hz and r["meta"]["delay"] == dl)
            o = r["res"]
            return "18걸음 ✓" if not o["fallen"] else f"{o['steps_done']}걸음 뒤 넘어짐"
        rb = next(r for r in sp if r["meta"]["stride"] == s and r["meta"]["kp"] == 2.0 and r["meta"]["hz"] == 200 and r["meta"]["delay"] == 10.0)["res"]
        md.append(f"| {s*1e3:.0f} | {s/0.65:.3f} | {cell(1.0,200,10.0)} | {cell(2.0,200,0.0)} | {cell(2.0,200,10.0)} | {cell(2.0,100,0.0)} | {cell(2.0,100,10.0)} | {cell(4.0,200,10.0)} | "
                  f"{(rb['speed_mps'] or 0):.3f} |")
    md.append("\n## 제어기 부품을 하나씩 끈 비교 (0.046 m/s, 이득 2배)\n")
    md.append("| 설정 | 결과 | 압력중심 여유 하위 5 % (mm) |"); md.append("|---|---|---|")
    for lab, f, stp, p5, ft in abl:
        md.append(f"| {lab} | {'18걸음 ✓' if not f else f'{stp}걸음 뒤 넘어짐 ({ft:.1f} s)'} | {p5:.1f} |")
    md.append("\n## 머리 끄덕임 (−10~+20°)\n")
    md.append("| 걸음 | 끄덕임 | 넘어짐 | 몸통 롤 (°) | 몸통 피치 (°) | 압력중심 여유 최소 / 하위 5 % (mm) | 크랭크 힘 최대 / RMS (N·m) | 크랭크 u 최대 | 머리 각 범위 (°) | 머리 각속도 RMS (°/s) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for (s, f), o in sorted(nd.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        md.append(f"| {'제자리' if s == 0 else f'{s/0.65:.3f} m/s'} | {('없음' if not f else f'{f} Hz')} | {'예' if o['fallen'] else '아니오'} | "
                  f"{o['torso_roll_deg'][0]:.1f} ~ {o['torso_roll_deg'][1]:.1f} | {o['torso_pitch_deg'][0]:.1f} ~ {o['torso_pitch_deg'][1]:.1f} | "
                  f"{o['cop_margin_mm']['min']:.1f} / {o['cop_margin_mm']['p5']:.1f} | {o['crank_tau_max']:.2f} / {o['crank_tau_rms']:.2f} | {o['u_max']['nod_crank']:.2f} | "
                  f"{o['head_deg_range'][0]:.1f} ~ {o['head_deg_range'][1]:.1f} | {o['head_w_rms_dps']:.0f} |")
    md.append("\n## 밀기 (0.2 s, 몸 전체 무게중심 높이에 수평으로; 제자리 걸음 7번째 걸음, 디딤발 = 오른발)\n")
    md.append("| 방향 · 시점 | 마찰 0.5 | 마찰 0.8 | 마찰 1.0 |"); md.append("|---|---|---|---|")
    for dr, wh, lab in cases:
        cells = []
        for fr in (0.5, 0.8, 1.0):
            s_, f_ = thr(fr, dr, wh)
            cells.append("–" if s_ is None else f"{s_} N 버팀 / {f_ if f_ else '>15'} N 넘어짐")
        md.append(f"| {lab.replace(chr(10), ' ')} | " + " | ".join(cells) + " |")
    md.append("\n## 마찰만 바꿔 걷기\n")
    md.append("| 마찰 | 속도 (m/s) | 결과 | 압력중심 여유 하위 5 % (mm) | 몸통 요 끝 (°) |"); md.append("|---|---|---|---|---|")
    for r in [r for r in pu if r["meta"].get("walk")]:
        o = r["res"]
        md.append(f"| {r['meta']['friction']} | {o['speed_mps']:.3f} | {'18걸음 ✓' if not o['fallen'] else '넘어짐'} | {o['cop_margin_mm']['p5']:.1f} | {o['torso_yaw_deg'][2]:.1f} |")
    md.append("\n## 제자리 돌기 (걸음마다 요 각을 더함, 고관절 요 ±20° 안)\n")
    md.append("| 걸음마다 (°) | 결과 | 돈 각도 (°) | 압력중심 여유 하위 5 % (mm) | 고관절 요 힘/정지토크 최대 | u 최대 관절 |"); md.append("|---|---|---|---|---|---|")
    for r in J("turn")["rows"]:
        o = r["res"]; top = max(o["u_max"].items(), key=lambda kv: kv[1])
        verdict = "끝까지 ✓" if not o["fallen"] else f"{o['steps_done']}걸음 뒤 넘어짐"
        md.append(f"| {r['meta']['yaw_step_deg']} | {verdict} | {o['torso_yaw_deg'][2]:.0f} | "
                  f"{o['cop_margin_mm']['p5']:.1f} | {jgroup(o['force_over_stall_max'], 'hip_yaw'):.2f} | {top[0]} {top[1]:.2f} |")
    # 사용률 표
    md.append("\n## 관절별 부하 (각 칸: 최대 |힘|/정지토크 · 그 99백분위 · u 최대 · u 99백분위 · 포화 시간 %)\n")
    reps = [("제자리 걸음", next(r["res"] for r in sp if r["meta"]["stride"] == 0.0 and r["meta"]["kp"] == 2.0 and r["meta"]["hz"] == 200 and r["meta"]["delay"] == 10.0)),
            ("0.046 m/s", S["walk46"][0]), ("0.076 m/s", S["walk76"][0]), ("0.083 m/s(최고)", S["walk83"][0]),
            ("0.046 m/s + 끄덕 1 Hz", S["walk46_nod1"][0]), ("0.046 m/s + 끄덕 0.5 Hz", S["walk46_nod05"][0]),
            ("제자리 돌기 20°/걸음", next(r["res"] for r in J("turn")["rows"] if r["meta"]["yaw_step_deg"] == 20)),
            ("0.092 m/s(넘어짐, 넘어지기 전까지)", S["walk90_fall"][0])]
    def push_rep(dr, F, fr=0.8, wh="ss"):
        return next(r["res"] for r in pu if r["meta"].get("dir") == dr and r["meta"].get("where") == wh and r["meta"].get("F") == F and r["meta"]["friction"] == fr)
    s_out, _ = thr(0.8, "-y", "ss"); s_in, _ = thr(0.8, "+y", "ss")
    reps += [(f"밀기 디딤발 바깥쪽 {s_out} N(버팀)", push_rep("-y", s_out)), (f"밀기 흔드는 발 쪽 {s_in} N(버팀)", push_rep("+y", s_in)),
             ("마찰 0.5 · 0.076 m/s", next(r["res"] for r in pu if r["meta"].get("walk") and r["meta"]["friction"] == 0.5 and r["meta"]["stride"] == 0.05))]
    md.append("| 관절 | " + " | ".join(n for n, _ in reps) + " |"); md.append("|---" * (len(reps) + 1) + "|")
    for g, lab in GROUPS[:6]:
        row = [lab]
        for n, o in reps:
            u = util_rows(o).get(g)
            fp = jgroup(o["force_over_stall_p99"], g)
            row.append("–" if u is None else f"{u[0]:.2f} · {fp:.2f} · {u[1]:.2f} · {u[2]:.2f} · {100*u[3]:.1f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("\n## 설계 변형 (같은 제어기)\n")
    md.append("| 변형 | 최고 속도 (m/s) | 흔드는 발 쪽 밀기 버팀 (N) | 디딤발 바깥쪽 밀기 버팀 (N) | 끄덕이며 걸을 때 몸통 롤 최대 (°) | 압력중심 여유 하위 5 % (mm) | 카메라(머리) 각속도 RMS (°/s) | 크랭크 RMS (N·m) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, v in enumerate(vn):
        nr = next(r for r in va if r["meta"]["variant"] == v and r["meta"]["test"] == "nod")["res"]
        md.append(f"| {v} | {vmax[i]:.3f} | {pin[i]} | {pout[i]} | {roll[i]:.1f} | {mg[i]:.1f} | {nr['head_w_rms_dps']:.0f} | {nr['crank_tau_rms']:.2f} |")
    open(os.path.join(RES, "tables.md"), "w").write("\n".join(md) + "\n")
    jdump({k: v[0] for k, v in S.items()}, os.path.join(RES, "series_runs.json"))
    print("\n".join(md))


if __name__ == "__main__":
    main()
