"""머리 받침 + 모터 없는 팔꿈치 조사(2026-11-27) 정리: results/<tag>.json 의 가장 좋은 시도를 다시 돌려 표(results/head_<tag>.md · .json)와
그림(results/fig_head_<tag>.png)을 만든다. 목 측정은 gu_model.Sim.neck(머리 · 귀 ↔ 바닥 힘, 고리 · 멈춤턱이 head_nod 에 준 토크, 크랭크 서보 부하).
usage: ../.venv/bin/python gu_head.py <tag> [<tag> ...]"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gu_model import ARMS, RES, jdump, reach, arm_mass_g, korean_font

START_KO = {"headpike": "머리 받침 ㅅ자", "tuck": "다리 당기기", "pike": "ㅅ자", "supine6": "누움"}
KIND_KO = {"bent": "꺾어 고정", "spring": "용수철 팔꿈치", "elbow": "팔꿈치 서보(XL)"}
XL = 1.4


def redo(r):
    from gu_search import replay
    S, res, H, _ = replay(r)
    assert res["ok"] == r["res"]["ok"] and abs(res["z_end"] - r["res"]["z_end"]) < 1e-9, "다시 돌린 결과가 저장값과 다름"
    return dict(arm=r["arm"], start=r["start"], kind=ARMS[r["arm"]]["kind"], ok=res["ok"], n_ok=r.get("n_ok_total", 0), n_eval=r.get("n_eval", 0),
                z_end_pct=res["z_end"] / S.z_stand * 100, max_z_pct=float(H[:, 1].max() / S.z_stand * 100), tilt_end=res["tilt_end"],
                tq=res["torque_p99"], tqmax=res["torque_max"], sat=res["sat_frac"], neck=res["neck"], elbow_deg=res.get("elbow_deg"),
                elbow_spring_max=res.get("elbow_spring_max"), reach_mm=reach(ARMS[r["arm"]]) * 1e3, arm_g=arm_mass_g(ARMS[r["arm"]]))


def main(tags):
    from multiprocessing import Pool
    rs = [r for t in tags for r in json.load(open(os.path.join(RES, f"{t}.json")))]
    with Pool(min(len(rs), 28)) as P:
        rows = P.map(redo, rs)
    name = "_".join(tags)
    jdump(rows, os.path.join(RES, f"head_{name}.json"))
    md = ["| 방식 | 팔 (닿는 길이 mm · 팔 하나 g) | 시작 | 성공 (성공한 후보 / 돌린 수) | 끝 높이 % | 지나는 동안 최고 % | 어깨 토크 몫 · 포화 % | 팔꿈치 | "
          "머리 ↔ 바닥 닿은 시간 % · 힘 99번째 백분위 / 최대 N | head_nod 가 받은 토크 99번째 백분위 / 최대 N·m | 크랭크 서보 토크: 환산 99번째 백분위 · 시뮬 부하 99번째 백분위 N·m (XL 대비) · 포화 % | 멈춤턱에 닿은 시간 % | 머리 각 범위 ° |",
          "|" + "---|" * 13]
    for r in rows:
        n = r["neck"]
        if r["kind"] == "elbow":
            el = f"서보 몫 {r['tq']['elbow']:.2f} · 포화 {r['sat']['elbow'] * 100:.0f} %"
        elif r["kind"] == "spring":
            el = f"각 {r['elbow_deg'][0]:.0f} ~ {r['elbow_deg'][1]:.0f}° · 용수철 최대 {r['elbow_spring_max']:.2f} N·m"
        else:
            el = "—"
        md.append(f"| {KIND_KO[r['kind']]} | {r['arm']} ({r['reach_mm']:.0f} · {r['arm_g']:.0f}) | {START_KO[r['start']]} | {'**성공**' if r['ok'] else '실패'} ({r['n_ok']}/{r['n_eval']}) | "
                  f"{r['z_end_pct']:.0f} | {r['max_z_pct']:.0f} | {r['tq']['arm_swing']:.2f} · {r['sat']['arm_swing'] * 100:.0f} | {el} | "
                  f"{n['contact_frac'] * 100:.0f} · {n['contact_p99_N']:.0f} / {n['contact_max_N']:.0f} | {n['link_tq_p99']:.2f} / {n['link_tq_max']:.2f} | "
                  f"{n['crank_calc_p99']:.2f} · {n['crank_load_p99']:.2f} ({n['crank_load_p99'] / XL:.1f} 배) · {n['crank_sat_frac'] * 100:.0f} | {n['stop_frac'] * 100:.0f} | "
                  f"{n['head_deg'][0]:.0f} ~ {n['head_deg'][1]:.0f} |")
    open(os.path.join(RES, f"head_{name}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    fig(rows, os.path.join(RES, f"fig_head_{name}.png"))
    sp = []
    for r in rows:
        if r["kind"] == "spring":
            T = ARMS[r["arm"]]["T90"]; sp.append(dict(arm=r["arm"], start=r["start"], full=spring_calc(T), used=spring_calc(T, max(-r["elbow_deg"][0], 1.0))))
    jdump(sp, os.path.join(RES, f"head_{name}_spring.json"))
    for T in (0.5, 1.0, 2.0):
        print("용수철", T, "N·m @ 90°, 130° 까지 접힘:", {k: round(v, 1) for k, v in spring_calc(T).items()})
    return rows


def spring_calc(T90, fold_deg=130.0, D=0.016, sig=1.2e9, E=206e9, rho=7850.0):
    """비틀림 용수철(피아노선)을 만들 수 있는지 [계산]: 가장 많이 접힌 각에서 토크 M = T90 × (미리 감김 + 접힘) / (미리 감김 + 90°),
    선 지름 d 는 굽힘 응력 1.1 × 32 M / (π d³) ≤ sig(피아노선 정적 허용 ≈ 1.2 GPa [추정]), 감김 수 N = E d⁴ θ / (64 D M),
    선 길이 = π D N, 질량 = ρ π d² / 4 × 선 길이. θ 는 0 에서 가장 많이 접힐 때까지 전체 비틀림(미리 감김 포함)."""
    from gu_model import SPRING_PRE
    pre = np.degrees(SPRING_PRE); th = np.radians(pre + fold_deg); M = T90 * (pre + fold_deg) / (pre + 90.0)
    d = (1.1 * 32 * M / (np.pi * sig)) ** (1 / 3); d = np.ceil(d * 1e4) / 1e4          # 0.1 mm 올림
    N = E * d ** 4 * th / (64 * D * M); Lw = np.pi * D * N
    return dict(T90=T90, fold_deg=fold_deg, M_max=float(M), d_mm=float(d * 1e3), D_mm=D * 1e3, turns=float(N), body_len_mm=float(N * d * 1e3),
                mass_g=float(rho * np.pi * d * d / 4 * Lw * 1e3))


def fig(rows, path):
    """왼쪽: 끝 1 s 무게중심 높이(서 있을 때 대비 %), 오른쪽: 목 크랭크 서보 부하 99번째 백분위(N·m) — 팔 × 시작 자세 [시뮬 실측]"""
    plt = korean_font()
    arms = list(dict.fromkeys(r["arm"] for r in rows)); starts = [s for s in START_KO if any(r["start"] == s for r in rows)]
    col = dict(zip(START_KO, ["#257de0", "#f47230", "#00b894", "#f2a600"]))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 0.55 * len(arms) + 2.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    h = 0.8 / len(starts)
    for k, s in enumerate(starts):
        y = np.arange(len(arms)) + (k - (len(starts) - 1) / 2) * h
        g = {r["arm"]: r for r in rows if r["start"] == s}
        axs[0].barh(y, [g[a]["z_end_pct"] if a in g else np.nan for a in arms], height=h * 0.85, color=col[s], label=START_KO[s] + "에서 시작")
        axs[1].barh(y, [g[a]["neck"]["crank_load_p99"] if a in g else np.nan for a in arms], height=h * 0.85, color=col[s])
        for yy, a in zip(y, arms):
            if a in g and g[a]["ok"]:
                axs[0].text(g[a]["z_end_pct"] + 1, yy, "섬", va="center", fontsize=7.5, color="#222")
    axs[0].axvline(90, color="#555", lw=1, ls="--"); axs[0].text(89, -0.75, "성공 기준 90 %", fontsize=7.5, color="#333", ha="right")
    axs[1].axvline(XL, color="#555", lw=1, ls="--"); axs[1].text(XL + 0.03, -0.75, "XL430 정지토크 1.4 N·m", fontsize=7.5, color="#333")
    axs[0].set_yticks(np.arange(len(arms))); axs[0].set_yticklabels([f"{a}\n({KIND_KO[ARMS[a]['kind']]}, {reach(ARMS[a]) * 1e3:.0f} mm)" for a in arms], fontsize=7.5)
    axs[0].invert_yaxis(); axs[0].set_xlim(0, 110)
    axs[0].set_xlabel("동작이 끝난 뒤 무게중심 높이 (서 있을 때 대비 %)")
    axs[1].set_xlabel("목 크랭크 서보가 받은 부하 토크, 99번째 백분위 (N·m)")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.25)
    axs[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(1.0, -0.09), ncol=len(starts), frameon=False)
    ok_free = [r for r in rows if r["ok"] and r["kind"] != "elbow"]; ok_servo = [r for r in rows if r["ok"] and r["kind"] == "elbow"]
    head = ("모터 없는 팔꿈치(꺾어 고정 · 용수철)는 머리를 받침으로 써도 한 번도 서지 못했다" if not ok_free else
            f"모터 없는 팔꿈치가 {len(ok_free)} 조합에서 섰다")
    head += (f"; XL 팔꿈치 서보만 {', '.join(START_KO[r['start']] for r in ok_servo)} 시작에서 섰다" if ok_servo else "")
    head += ", 목 크랭크 서보는 모든 조합에서 XL 정지토크를 넘었다" if all(r["neck"]["crank_load_p99"] > XL for r in rows) else ""
    fig.suptitle(head + f"\n(팔 {len(arms)} 가지 × 시작 자세 {len(starts)} 가지, 조합마다 키프레임 {rows[0]['n_eval']} 번 최적화, "
                 "고관절 −115° · 무릎 115° · 어깨 ±110°, 머리 v21 XL 크랭크, 시뮬 실측)", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
