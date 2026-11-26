"""최적화 결과 정리: results/<tag>.json 들의 가장 좋은 키프레임을 다시 돌려 표(results/summary_<tag>.md · .json)와 그림 1(fig_getup_best.png)을 만든다.
usage: ../.venv/bin/python gu_summary.py <tag> [<tag> ...]   (같은 고관절 · 무릎 한계로 돈 묶음끼리)"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gu_model import ARMS, Sim, RES, jdump, reach, arm_mass_g, korean_font

START_KO = {"prone": "엎드림", "pike": "엎드림(ㅅ자 시작)", "supine": "누움"}


def redo(r):
    from gu_search import replay
    S, res, H, _ = replay(r)
    inst = np.maximum(H[:, 2] - 5, 0) / 30 + 3 * np.maximum(0.95 - H[:, 1] / S.z_stand, 0); i = int(np.argmin(inst))
    up = np.argmin(H[:, 2] + 1000 * (H[:, 1] < 0.6 * S.z_stand))        # 무게중심이 60 % 넘은 동안 가장 곧게 선 순간
    return dict(arm=r["arm"], start=r["start"], ok=res["ok"], n_ok=r.get("n_ok_total", 0), n_eval=r.get("n_eval", 0),
                best_z_pct=float(H[i, 1] / S.z_stand * 100), best_tilt=float(H[i, 2]), max_z_pct=float(H[:, 1].max() / S.z_stand * 100),
                z_end_pct=res["z_end"] / S.z_stand * 100, tilt_end=res["tilt_end"], score=res["score"],
                tq=res["torque_p99"], sat=res["sat_frac"], hip_min_deg=r.get("hip_min_deg", -80), knee_max_deg=r.get("knee_max_deg", 100))


def main(tags):
    rows = []
    for t in tags:
        rows += [dict(redo(r), tag=t) for r in json.load(open(os.path.join(RES, f"{t}.json")))]
    jdump(rows, os.path.join(RES, f"summary_{'_'.join(tags)}.json"))
    md = ["| 팔 | 닿는 길이 mm | 팔 하나 g | 시작 | 성공 | 가장 서 있음에 가까운 순간: 무게중심 높이 % · 몸통 기울기 ° | 끝 높이 % | "
          "어깨 토크 몫(99번째 백분위) · 포화 시간 % | 팔꿈치 토크 몫 · 포화 % | 무릎 토크 몫 | 고관절 토크 몫 |", "|" + "---|" * 11]
    for r in rows:
        a = ARMS[r["arm"]]
        el = f"{r['tq'].get('elbow', 0):.2f} · {r['sat'].get('elbow', 0) * 100:.0f}" if "elbow" in r["tq"] else "—"
        md.append(f"| {r['arm']} | {reach(a) * 1e3:.0f} | {arm_mass_g(a):.0f} | {START_KO[r['start']]} | {'성공' if r['ok'] else '실패'} ({r['n_ok']}/{r['n_eval']}) | "
                  f"{r['best_z_pct']:.0f} · {r['best_tilt']:.0f} | {r['z_end_pct']:.0f} | {r['tq']['arm_swing']:.2f} · {r['sat']['arm_swing'] * 100:.0f} | {el} | "
                  f"{r['tq']['knee']:.2f} | {r['tq']['hip_pitch']:.2f} |")
    open(os.path.join(RES, f"summary_{'_'.join(tags)}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return rows


def fig(rows, path, title):
    plt = korean_font()
    arms = list(dict.fromkeys(r["arm"] for r in rows)); starts = list(dict.fromkeys(r["start"] for r in rows))
    col = {"prone": "#257de0", "pike": "#00b894", "supine": "#f47230"}
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(arms) + 1.4))
    h = 0.8 / len(starts)
    for k, s in enumerate(starts):
        y = np.arange(len(arms)) + (k - (len(starts) - 1) / 2) * h
        v = [next((r["max_z_pct"] for r in rows if r["arm"] == a and r["start"] == s), np.nan) for a in arms]
        ax.barh(y, v, height=h * 0.9, color=col[s], label=START_KO[s])
    ax.axvline(90, color="#555", lw=1, ls="--"); ax.text(90.5, -0.7, "성공 기준 90 %", fontsize=8, color="#333")
    ax.set_yticks(np.arange(len(arms))); ax.set_yticklabels([f"{a} ({reach(ARMS[a]) * 1e3:.0f} mm)" for a in arms], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 105)
    ax.set_xlabel("일어나는 동안 가장 높았던 무게중심 높이 (서 있을 때 대비 %, 시뮬 실측)")
    ax.legend(fontsize=8, loc="lower right", frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.25)
    ax.set_title(title, fontsize=9.5)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    tags = sys.argv[1:]
    rows = main(tags)
