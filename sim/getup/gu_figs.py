"""그림: fig_reach.png(팔 닿는 길이 vs 손에서 발로 무게가 넘어가는 순간 필요한 길이, 정적 계산)
usage: ../.venv/bin/python gu_figs.py"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gu_model import ARMS, RES, reach, korean_font

# gu_static.py 결과(results/static.json, 고관절 −100 · −120° 는 문서 2절 민감도 계산) [정적 계산]
REQ = [("엎드림 길(몸통 앞으로), 설계 범위", 292, "#257de0", "-"), ("누움 길(몸통 뒤로), 설계 범위", 279, "#f47230", "-"),
       ("엎드림 길, 고관절 굽힘 −100° 로 넓힘", 245, "#257de0", "--")]


def fig_reach(path):
    plt = korean_font()
    names = [n for n in ARMS if "XM" not in n]
    v = [reach(ARMS[n]) * 1e3 for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(np.arange(len(names)), v, height=0.6, color="#9aa3ad")
    for i, x in enumerate(v):
        ax.text(x + 3, i, f"{x:.0f}", va="center", fontsize=8, color="#333")
    for lab, x, c, ls in REQ:
        ax.axvline(x, color=c, ls=ls, lw=1.6, label=f"{lab}: {x} mm")
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 330); ax.set_xlabel("어깨 축에서 손 바깥면까지 닿는 길이 (mm)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=1, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("무게중심이 발 위에 온 채 손이 바닥에 닿으려면 팔이 280–290 mm 필요하다\n(다리 +20 mm, 무릎 100° · 고관절 −80° 설계 범위, 정적 계산; 고관절 −100° 면 245 mm)", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_best(path, tags):
    """팔 후보마다(모든 묶음 중) 끝 1 s 무게중심 높이가 가장 높았던 시도, 엎드림 · 누움 따로 [시뮬 실측]"""
    rows = []
    for t in tags:
        rows += json.load(open(os.path.join(RES, f"summary_{t}.json")))
    grp = lambda s: "누움" if s.startswith("supine") else "엎드림"
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    best = {}
    for r in rows:
        k = (r["arm"], grp(r["start"]))
        if k not in best or r["z_end_pct"] > best[k]["z_end_pct"]:
            best[k] = r
    plt = korean_font()
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(arms) + 1.6))
    col = {"엎드림": "#257de0", "누움": "#f47230"}
    for k, g in enumerate(("엎드림", "누움")):
        y = np.arange(len(arms)) + (k - 0.5) * 0.4
        v = [best.get((a, g), {}).get("z_end_pct", np.nan) for a in arms]
        ax.barh(y, v, height=0.36, color=col[g], label=f"{g}에서 시작")
        for yy, a in zip(y, arms):
            if best.get((a, g), {}).get("ok"):
                ax.text(best[(a, g)]["z_end_pct"] + 1, yy, "섬", va="center", fontsize=8, color="#222")
    ax.axvline(90, color="#555", lw=1, ls="--")
    ax.text(89, len(arms) - 0.2, "성공 기준 90 %", fontsize=7.5, color="#333", ha="right")
    ax.set_yticks(np.arange(len(arms))); ax.set_yticklabels([f"{a} ({reach(ARMS[a]) * 1e3:.0f} mm)" for a in arms], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 110); ax.set_xlabel("동작이 끝난 뒤 1 s 동안 무게중심 높이 (서 있을 때 대비 %)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("XL430 팔은 길이 · 모양과 상관없이 한 번도 서지 못했고, 팔꿈치 140+140 에 2.7 N·m 이상 서보일 때만 누움에서 섰다\n"
                 "(후보 · 시작 자세마다 키프레임 648–6568 번 최적화, 다리 +20 mm, 시뮬 실측; 「· XM」「· W210」은 어깨 · 팔꿈치 서보를 바꾼 민감도)", fontsize=8.5)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_reach(os.path.join(RES, "fig_reach.png"))
    if len(sys.argv) > 1:
        fig_best(os.path.join(RES, "fig_getup_best.png"), sys.argv[1:])

