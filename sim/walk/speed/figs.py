"""그림 2장: results/fig_speed_run.png (질문 1 · 2), results/fig_leg_length.png (질문 3). usage: ../../.venv/bin/python figs.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from kw_common import korean_font
RES = os.path.join(HERE, "results")
C = ["#257de0", "#f47230", "#00b894", "#f2a600"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK2); ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig_speed(plt):
    S = json.load(open(os.path.join(RES, "summary.json")))["servo_rep"]
    J = json.load(open(os.path.join(RES, "jump.json")))
    names = list(S.keys())
    short = {"지금 (늘림 40.5 mm)": "지금 서보", "무릎만 W210": "무릎만 W210",
             "고관절 피치 · 무릎 W210": "고관절 피치 · 무릎 W210", "다리 XM 모두 W210": "다리 XM430 넷 모두 W210",
             "발목 피치 XC430-W150": "발목 피치 XC430-W150", "피치 · 무릎 W210 + 발목 피치 W150": "피치 · 무릎 W210 + 발목 피치 W150",
             "발목 피치 XM430-W350": "발목 피치 XM430-W350", "피치 · 무릎 W210 + 발목 피치 XM430-W210": "피치 · 무릎 W210 + 발목 피치 W210"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    y = np.arange(len(names))[::-1]
    vg = [S[n]["vgood"]["v"] if S[n]["vgood"] else 0 for n in names]
    sg = [S[n]["sgood"]["stride"] * 1000 if S[n]["sgood"] else 0 for n in names]
    ax[0].barh(y, vg, height=0.55, color=C[0])
    for yi, v, s in zip(y, vg, sg):
        ax[0].text(v + 0.002, yi, f"{v:.3f} m/s", va="center", fontsize=8, color=INK)
    ax[0].set_yticks(y); ax[0].set_yticklabels([short[n] for n in names], fontsize=8, color=INK)
    ax[0].set_xlim(0, 0.15); ax[0].set_xlabel("쓸 만한 최고 걷기 속도 (m/s)\n세 번 다 넘어지지 않음 + 압력중심 여유 ≥ 5 mm + 무릎 포화 < 10 %", fontsize=8, color=INK2)
    ax[0].set_title("걷기(두 발이 늘 하나는 닿음): 서보 변형별", fontsize=10, color=INK, loc="left"); style(ax[0])
    # 도약 체공
    cfgs = ["지금 (늘림 40.5 mm)", "고관절 피치 · 무릎 W210", "다리 XM 모두 W210", "피치 · 무릎 W210 + 발목 피치 W150", "발목 피치 XM430-W350"]
    def best(n, half):
        rs = [r for r in J if r["name"] == n and r["half"] == half and "error" not in r and r["p"] == 0.0 and r["upright_end"]]
        return max([r["flight_s"] * 1e3 for r in rs] or [0])
    y2 = np.arange(len(cfgs))[::-1]
    two = [best(n, False) for n in cfgs]; one = [best(n, True) for n in cfgs]
    ax[1].axvspan(30, 50, color="#f2a600", alpha=0.18, lw=0)
    ax[1].text(40, -0.55, "달리기에 필요한\n한 발 체공 30–50 ms", ha="center", va="bottom", fontsize=8, color=INK2)
    ax[1].barh(y2 + 0.15, two, height=0.28, color=C[0], label="두 발로 제자리 도약 (넘어지지 않은 것 중 최대)")
    ax[1].barh(y2 - 0.15, one, height=0.28, color=C[1], label="한 발 뛰기 흉내 (다리 서보 토크 절반)")
    for yi, a, b in zip(y2, two, one):
        ax[1].text(a + 0.6, yi + 0.15, f"{a:.0f}", va="center", fontsize=8, color=INK); ax[1].text(b + 0.6, yi - 0.15, f"{b:.0f}", va="center", fontsize=8, color=INK)
    ax[1].set_yticks(y2); ax[1].set_yticklabels([short[n] for n in cfgs], fontsize=8, color=INK)
    ax[1].set_xlim(0, 55); ax[1].set_xlabel("두 발이 모두 뜬 시간 (ms)", fontsize=8, color=INK2)
    ax[1].set_title("도약 시험: 서보가 몸을 띄울 수 있나", fontsize=10, color=INK, loc="left"); style(ax[1])
    ax[1].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=1)
    fig.suptitle("지금 서보로는 몸을 띄우지 못해 달리기가 안 된다 — 고관절 피치 · 무릎을 XM430-W210 으로 바꾸면 걷기가 0.084 → 0.11 m/s 로 빨라지지만 한 발 뛰기 체공은 여전히 4 ms 이하\n"
                 "(MuJoCo 시뮬 실측, 몸무게 3.05 kg, 고관절 롤 안 A, 걷기 한 칸 = 설정 하나에 마찰 세 가지, 도약 = 웅크림 깊이 세 가지 중 최대)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(RES, "fig_speed_run.png"), dpi=110)


def fig_leg(plt):
    S = json.load(open(os.path.join(RES, "summary.json")))["seg"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    exts = sorted({r["ext"] for r in S})
    for ci, cr in enumerate((0.015, 0.025, 0.035)):
        for L, ls in ((0.0, "-"), (0.02, "--")):
            R = sorted([r for r in S if r["crouch"] == cr and r["L"] == L], key=lambda r: r["ext"])
            x = [r["ext"] * 1e3 for r in R]
            lab = f"웅크림 {cr*1e3:.0f} mm, 롤-피치 늘림 {L*1e3:.0f} mm"
            ax[0].plot(x, [r["smax"] * 1e3 for r in R], ls, color=C[ci], lw=2, marker="o", ms=6, label=lab)
            ax[1].plot(x, [r["p5"] if r["p5"] is not None else np.nan for r in R], ls, color=C[ci], lw=2, marker="o", ms=6, label=lab)
    R0 = sorted([r for r in S if r["crouch"] == 0.025 and r["L"] == 0.0], key=lambda r: r["ext"])
    st = json.load(open(os.path.join(RES, "static.json")))
    for ci, cr in enumerate(("0.015", "0.025", "0.035")):
        xs = [x["ext"] * 1e3 for x in st if x["L"] == 0.0]; ys = [x["one_leg"][cr]["knee_Nm"] / 3.8 for x in st if x["L"] == 0.0]
        ax[2].plot(xs, ys, "-", color=C[ci], lw=2, marker="o", ms=6, label=f"한 발 정지 무릎 토크, 웅크림 {float(cr)*1e3:.0f} mm")
    ax[2].plot([r["ext"] * 1e3 for r in R0], [r["knee"]["torque_rms"] if r["knee"] else np.nan for r in R0], ":", color=INK2, lw=2, marker="s", ms=6,
               label="걷기 무릎 토크 RMS (웅크림 25 mm, 발열 지표)")
    xs = [x["ext"] * 1e3 for x in st if x["L"] == 0.0]
    ax[2].plot(xs, [x["squat"]["100"]["knee_Nm"] / 3.8 for x in st if x["L"] == 0.0], "-.", color=C[3], lw=2, marker="^", ms=6, label="쪼그려 앉음(일어나기) 무릎 토크")
    ax[0].set_ylabel("최대 보폭 (mm, 한 걸음 0.65 s)", fontsize=8, color=INK2)
    ax[1].set_ylabel("압력중심 여유 하위 5 % (mm)", fontsize=8, color=INK2)
    ax[2].set_ylabel("무릎 토크 ÷ XM430-W350 정지토크 3.8 N·m", fontsize=8, color=INK2)
    for a in ax:
        a.set_xlabel("허벅지 · 정강이 각각 늘린 길이 (mm, 질량 반영)", fontsize=8, color=INK2); a.set_xticks(exts and [e * 1e3 for e in exts])
        style(a); a.grid(axis="y", color=GRID, lw=0.8)
    ax[0].legend(fontsize=7, frameon=False); ax[2].legend(fontsize=7, frameon=False); ax[2].set_ylim(0, 0.6)
    ax[0].set_title("넓은 걸음: 마찰 세 가지 모두 18걸음 넘어지지 않은 최대 보폭\n(한 걸음에 앞발이 나아가는 거리)", fontsize=9, color=INK, loc="left")
    ax[1].set_title("균형: 기준 걸음(0.65 s, 보폭 50 mm), 세 번 중 최소\n(값이 들쭉날쭉 — 웅크림 25 mm 만 보면 8.7 → 5.2 mm)", fontsize=9, color=INK, loc="left")
    ax[2].set_title("무릎 부담(발열 · 일어나기), 롤-피치 늘림 0 mm\n", fontsize=9, color=INK, loc="left")
    return fig


if __name__ == "__main__":
    plt = korean_font()
    fig_speed(plt)
    if "seg" in json.load(open(os.path.join(RES, "summary.json"))):
        fig = fig_leg(plt)
        t = sys.argv[1] if len(sys.argv) > 1 else "허벅지 · 정강이를 늘리면 보폭이 커지고 무릎 부담은 조금 늘어난다"
        fig.suptitle(t + "\n(MuJoCo 시뮬 실측 + 정적 계산, 지금 서보 배치, 고관절 롤 안 A, 실선 = 롤-피치 늘림 0 mm, 점선 = 20 mm)", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(os.path.join(RES, "fig_leg_length.png"), dpi=110)
