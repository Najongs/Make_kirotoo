"""강건성 그림: python robust/figs.py → robust/results/robust_oat.png, robust_combo.png
제목 문장은 결과 json 을 보고 고친다(TITLE1 · TITLE2)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rb_common import RES
from kw_common import korean_font
plt = korean_font()
C_NO, C_BP = "#257de0", "#f47230"          # 배낭 없음 · 배낭 있음
INK, INK2, GRID = "#191919", "#666666", "#e4e4e4"
BASEVAL = {"바닥 마찰": 0.8, "기어 백래시 (°)": 0.0, "내부 마찰 (정지토크 대비)": 0.0, "전압 (V)": 11.1, "제어 지연 (ms)": 10.0,
           "질량 배수": 1.0, "무게중심 앞뒤 (m)": 0.0}
ASKED = {"바닥 마찰": (0.3, 1.2), "기어 백래시 (°)": (0.2, 0.5), "내부 마찰 (정지토크 대비)": (0.0, 0.05), "전압 (V)": (10.5, 11.1),
         "제어 지연 (ms)": (10, 20), "질량 배수": (0.85, 1.15), "무게중심 앞뒤 (m)": (-0.01, 0.01)}
XLAB = {"바닥 마찰": "바닥 미끄럼 마찰 계수", "기어 백래시 (°)": "기어 백래시, 관절마다 (도)", "내부 마찰 (정지토크 대비)": "서보 내부 마찰 (정지토크 대비 %)",
        "전압 (V)": "배터리 전압 (V)", "제어 지연 (ms)": "센서에서 명령까지 지연 (ms)", "질량 배수": "실제 질량 / 설계 질량 (%)",
        "무게중심 앞뒤 (m)": "무게중심 앞뒤 어긋남 (mm, + 앞)"}
XS = {"내부 마찰 (정지토크 대비)": 100, "질량 배수": 100, "무게중심 앞뒤 (m)": 1000}
TITLE1 = os.environ.get("TITLE1", "강건성: 하나씩 바꿨을 때 0.077 m/s 걷기 압력중심 여유")
TITLE2 = os.environ.get("TITLE2", "나쁜 쪽 조합")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig_oat():
    rows = json.load(open(os.path.join(RES, "oat.json")))["rows"]
    pe = os.path.join(RES, "extra.json")
    if os.path.exists(pe):        # 무게중심 앞 경계를 촘촘히 더 본 점
        rows += [r for r in json.load(open(pe))["rows"] if r["meta"]["axis"] in BASEVAL]
    walk = [r for r in rows if r["meta"]["test"].startswith("걷기")]
    stand = {(r["meta"]["backpack"], r["meta"]["axis"], r["meta"]["val"]): r["res"] for r in rows if r["meta"]["test"] == "정지 자세"}
    axes = list(BASEVAL)
    fig, AX = plt.subplots(2, 4, figsize=(15, 7.4))
    AX = AX.ravel()
    for k, an in enumerate(axes):
        ax = AX[k]; style(ax); sc = XS.get(an, 1)
        lo, hi = ASKED[an]
        ax.axvspan(lo * sc, hi * sc, color="#f2f2f2", zorder=0)
        for bp, col, lab in ((False, C_NO, "배낭 없음 (지금 배터리)"), (True, C_BP, "배낭 있음 (250 g 등 뒤)")):
            pts = []
            base = next(r["res"] for r in walk if r["meta"]["backpack"] == bp and r["meta"]["axis"] == "기준")
            pts.append((BASEVAL[an], base))
            pts += [(r["meta"]["val"], r["res"]) for r in walk if r["meta"]["backpack"] == bp and r["meta"]["axis"] == an]
            pts.sort(key=lambda p: p[0])
            x = np.array([p[0] for p in pts]) * sc
            ok = np.array([("error" not in p[1]) and (not p[1]["fallen"]) for p in pts])
            y = np.array([p[1]["cop_margin_mm"]["p5"] if ok[i] else np.nan for i, p in enumerate(pts)])
            ax.plot(x[ok], y[ok], "-", color=col, lw=2, zorder=3)
            ax.plot(x[ok], y[ok], "o", color=col, ms=6, mec="white", mew=1.5, zorder=4, label=lab + ", 18걸음 성공")
            if (~ok).any():
                ax.plot(x[~ok], np.full((~ok).sum(), -1.5 if not bp else -3.0), "x", color=col, ms=8, mew=2, zorder=4,
                        label=lab + ", 넘어짐")
            # 정지 자세 실패 표시
            for xv, p in pts:
                key = (bp, "기준" if xv == BASEVAL[an] else an, None if xv == BASEVAL[an] else xv)
                s = stand.get(key)
                if s is not None and ("error" in s or s["fallen"]):
                    ax.plot([xv * sc], [24 if not bp else 22.5], "v", color=col, ms=7, zorder=4)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(XLAB[an], fontsize=10, color=INK, loc="left")
        ax.set_ylim(-4.5, 26)
        if k % 4 == 0:
            ax.set_ylabel("압력중심 여유, 하위 5 % (mm)", fontsize=9, color=INK2)
    ax = AX[7]; ax.axis("off")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Line2D([], [], color=C_NO, marker="o", lw=2, mec="white", label="배낭 없음 (지금 배터리), 18걸음 성공"),
          Line2D([], [], color=C_BP, marker="o", lw=2, mec="white", label="배낭 있음 (250 g 등 뒤), 18걸음 성공"),
          Line2D([], [], color=INK2, marker="x", lw=0, mew=2, label="0 아래 ×: 걷다가 넘어짐"),
          Line2D([], [], color=INK2, marker="v", lw=0, label="위쪽 ▼: 정지 자세에서도 넘어짐"),
          Patch(color="#f2f2f2", label="요청하신 검사 범위")]
    ax.legend(handles=hd, loc="center left", fontsize=9, frameon=False)
    ax.text(0, 0.02, "MuJoCo 시뮬 실측. 안 A(직렬) 고관절 롤 + v20 서보 배치를\nv18 형상에 얹은 근사 모델. 한 칸 = 한 번 실행.\n"
            "기준: 마찰 0.8, 백래시 0, 내부 마찰 0, 11.1 V, 지연 10 ms.", fontsize=8, color=INK2, transform=ax.transAxes)
    fig.suptitle(TITLE1, fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(RES, "robust_oat.png"); fig.savefig(p, dpi=130); print(p)


def fig_combo():
    rows = json.load(open(os.path.join(RES, "combo.json")))["rows"]
    walk = [r for r in rows if r["meta"]["test"].startswith("걷기")]
    NAME = {"전부": "나쁜 쪽 전부", "friction 뺌": "마찰 0.5 → 0.8 로 되돌림", "backlash_deg 뺌": "백래시 0.5° → 0",
            "frictionloss_frac 뺌": "내부 마찰 5 % → 0", "volt 뺌": "10.5 V → 11.1 V", "delay_ms 뺌": "지연 20 → 10 ms",
            "mass_f 뺌": "질량 +15 % → 0"}
    fig, AX = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, dx in zip(AX, (0.01, 0.0, -0.01)):
        style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.8)
        labels = list(NAME)
        for j, bp in enumerate((False, True)):
            col = C_BP if bp else C_NO
            for i, c in enumerate(labels):
                r = next((r for r in walk if r["meta"]["combo"] == c and r["meta"]["backpack"] == bp and abs(r["meta"]["com_dx"] - dx) < 1e-9), None)
                if r is None:
                    continue
                o = r["res"]; yv = i + (0.2 if bp else -0.2)
                if o.get("fallen") or "error" in o:
                    ax.barh(yv, -2, height=0.36, color="white", edgecolor=col, hatch="///", lw=1.2)
                    ax.text(-2.3, yv, f"{o.get('steps_done', 0)}걸음에서 넘어짐", va="center", ha="right", fontsize=8, color=INK)
                else:
                    v = o["cop_margin_mm"]["p5"]
                    ax.barh(yv, v, height=0.36, color=col)
                    ax.text(v + 0.3, yv, f"{v:.1f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels([NAME[c] for c in labels], fontsize=9)
        ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-14, 16)
        tt = "무게중심 어긋남 0 (실물을 재서 맞춘 경우)" if dx == 0 else f"무게중심 {dx * 1000:+.0f} mm {'앞' if dx > 0 else '뒤'} (제어기는 모름)"
        ax.set_title(tt, fontsize=10, loc="left", color=INK)
        ax.set_xlabel("압력중심 여유, 하위 5 % (mm) · 빗금 = 넘어짐", fontsize=9, color=INK2)
    from matplotlib.patches import Patch
    AX[2].legend(handles=[Patch(color=C_NO, label="배낭 없음"), Patch(color=C_BP, label="배낭 있음")], fontsize=9, frameon=False, loc="lower right")
    fig.suptitle(TITLE2, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(RES, "robust_combo.png"); fig.savefig(p, dpi=130); print(p)


if __name__ == "__main__":
    fig_oat()
    if os.path.exists(os.path.join(RES, "combo.json")):
        fig_combo()
