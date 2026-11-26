"""그림 1장: 구성별 무릎 · 고관절 롤 추정 정상 온도(열저항 15–25 K/W 범위 막대) — 0.05 m/s 걷기, 마찰 0.8 판.
usage: cd sim/walk/allxl && TITLE="..." ../../.venv/bin/python ax_fig.py → results/allxl_heat.png"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from kw_common import korean_font
plt = korean_font()
S = json.load(open(os.path.join(HERE, "results", "summary.json")))
PICK = ["지금 v20A (XM · XC)", "모두 XL 11.1 V", "모두 XL 12 V", "(a) 허벅지 · 정강이 0", "(b) 무릎 XL 두 개", "(c) 무릎 2:1",
        "(d) 웅크림 25", "(d) 몸무게 −20 %", "(e) 무릎 XL 두 개 + 발목 피치 XL 두 개", "C1 무릎 XL 둘 + 2:1, 롤 2:1",
        "C2 무릎 XL 둘 + 2:1, 롤 XL 둘", "D2 무릎 · 롤 XL 둘, 웅크림 25, −300 g", "D6 무릎 · 롤 XL 둘, 웅크림 35, −300 g",
        "D6 회전자 관성 × 2", "D7 무릎 · 롤 XL 둘, 웅크림 25, 몸 전체 −20 %", "D8 D2 를 12 V 로"]
ORDER = [k for k in PICK if k in S]
XMAX = 200
BLUE, ORANGE, INK, INK2, GRID = "#257de0", "#f47230", "#191919", "#666666", "#e6e6e6"

fig, ax = plt.subplots(figsize=(10, 0.42 * len(ORDER) + 1.9), dpi=150)
for i, name in enumerate(ORDER):
    o = S[name]; y = len(ORDER) - 1 - i
    ref = o["w05"]["ref"] or {}
    J = ref.get("joints") or {}
    fell = ref.get("fallen", True)
    for k, col, dy in (("knee", BLUE, 0.17), ("hip_roll", ORANGE, -0.17)):
        h = (J.get(k) or {}).get("heat")
        if not h:
            continue
        lo, hi = h["T_lo"], h["T_hi"]
        ax.plot([min(lo, XMAX), min(hi, XMAX)], [y + dy] * 2, color=col, lw=5, solid_capstyle="round")
        if hi > XMAX:
            ax.annotate("", xy=(XMAX + 4, y + dy), xytext=(XMAX - 2, y + dy), arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    walk_ok = o["w05"]["all_ok"]
    sat = (J.get("knee") or {}).get("sat_pct")
    tag = ("세 판 모두 걸음" if walk_ok else f"넘어짐 ({o['w05']['n_ok']}/3 판만 걸음)") + (f" · 무릎 포화 {sat:.1f} %" if sat is not None else "")
    ax.text(XMAX + 10, y, tag, va="center", fontsize=8, color=INK2)
ax.axvline(65, color=INK, lw=1, ls="--"); ax.text(64, len(ORDER) - 0.35, "목표 65 °C", fontsize=8, color=INK, ha="right")
ax.axvline(72, color="#ec4647", lw=1); ax.text(73, len(ORDER) - 0.35, "XL 과열 정지 72 °C", fontsize=8, color=INK, ha="left")
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(list(reversed(ORDER)), fontsize=8.5, color=INK)
ax.set_xlim(20, XMAX + 95); ax.set_ylim(-0.7, len(ORDER) + 0.1)
ax.set_xticks(range(25, XMAX + 1, 25))
ax.set_xlabel("서보 한 개 추정 정상 온도 (°C, 막대 = 열저항 15–25 K/W 범위, 실내 25 °C, 열 모델은 추정) — 200 °C 넘으면 화살표", fontsize=8.5, color=INK2)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color=BLUE, lw=5), Line2D([0], [0], color=ORANGE, lw=5)], labels=["무릎", "고관절 롤"],
          loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=9)
ax.set_title(os.environ.get("TITLE", "XL430 만으로는 무릎 · 고관절 롤이 걷기 열 목표를 넘는다"), fontsize=11, color=INK, loc="left", pad=22)
fig.text(0.01, 0.005, "MuJoCo 4.0.1 시뮬 실측 부하(0.05 m/s, 한 걸음 0.65 s, 18걸음, 마찰 0.8 한 판) → heat.py 식(구리 손실, e-manual Kt · 정지전류) · sim/walk/allxl",
         fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(os.path.join(HERE, "results", "allxl_heat.png"))
print("saved")
