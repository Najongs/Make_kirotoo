"""results/*.json → 표(results/summary.md)와 요약 수치(results/summary.json). usage: ../../.venv/bin/python summarize.py"""
import json, os, sys
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
GOOD_MARGIN, GOOD_SAT = 5.0, 0.10     # 「쓸 만한」 걸음: 압력중심 여유 하위 5 % ≥ 5 mm, 무릎 포화 < 10 %


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p))["rows"] if os.path.exists(p) else []


def rep_table(rows):
    """(cfg, Ts, Td, stride) → 통과 비율, 평균 실측 속도, 여유 p5 최소, 무릎 포화 최대 · u 몫"""
    g = defaultdict(list)
    for r in rows:
        m = r["meta"]; g[(m["cfg"], m["Ts"], m["Td"], m["stride"])].append(r["res"])
    T = {}
    for k, rs in g.items():
        ok = [o for o in rs if not o.get("fallen") and "error" not in o]
        T[k] = dict(n=len(rs), n_ok=len(ok), v=float(np.mean([o["speed_mps"] for o in ok])) if ok else None,
                    p5=float(min(o["cop_margin_mm"]["p5"] for o in ok)) if ok else None,
                    knee_sat=float(max(o["split"]["knee"]["sat_frac"] for o in ok)) if ok else None,
                    split={j: {q: float(np.mean([o["split"][j][q] for o in ok if o["split"][j][q] is not None] or [np.nan]))
                               for q in ("torque_p99", "speed_p99", "sat_frac", "speed_at_sat", "u_p99", "torque_rms") if all(q in o["split"][j] for o in ok)}
                           for j in ("hip_pitch", "knee", "ankle_pitch", "hip_roll", "ankle_roll")} if ok else None)
    return T


def best(T, cfg):
    """모두 통과한 점 중 최고 속도 · 최대 보폭, 그리고 「쓸 만한」(여유 · 포화 조건) 최고 속도 · 최대 보폭"""
    pts = [(k, v) for k, v in T.items() if k[0] == cfg]
    allok = [(k, v) for k, v in pts if v["n_ok"] == v["n"]]
    good = [(k, v) for k, v in allok if v["p5"] >= GOOD_MARGIN and v["knee_sat"] < GOOD_SAT]
    f = lambda L, key: max(L, key=key) if L else None
    return dict(vmax=f(allok, lambda kv: kv[1]["v"]), smax=f(allok, lambda kv: (kv[0][3], kv[1]["v"])),
                vgood=f(good, lambda kv: kv[1]["v"]), sgood=f(good, lambda kv: (kv[0][3], kv[1]["v"])))


def push_thr(rows):
    g = defaultdict(dict)
    for r in rows:
        m = r["meta"]; g[(m["cfg"], m["where"], m["dir"])][m["F"]] = r["res"]
    out = defaultdict(dict)
    for (c, wh, dr), d in g.items():
        if dr == "none":
            continue
        th = 0
        for F in sorted(d):
            if d[F].get("fallen"):
                break
            th = F
        out[c][f"{wh}{dr}"] = th
    return out


def fmt_pt(x):
    if not x:
        return "–"
    (c, a, b, s), v = x
    return f"{v['v']:.3f} m/s (보폭 {s*1000:.0f} mm, 한 걸음 {a+b:.2f} s, 여유 {v['p5']:.1f} mm, 무릎 포화 {v['knee_sat']*100:.0f} %)"


def main():
    out = {}; md = []
    for grp in ("legs_rep", "servo_rep"):
        rows = load(grp)
        if not rows:
            continue
        T = rep_table(rows); cfgs = list(dict.fromkeys(r["meta"]["cfg"] for r in rows))
        md.append(f"## {grp}\n\n| 설정 | 세 번 다 통과한 최고 속도 | 쓸 만한 최고 속도 (여유 ≥ {GOOD_MARGIN:.0f} mm, 무릎 포화 < {GOOD_SAT*100:.0f} %) | 세 번 다 통과한 최대 보폭 | 쓸 만한 최대 보폭 |\n|---|---|---|---|---|")
        out[grp] = {}
        for c in cfgs:
            b = best(T, c); out[grp][c] = {k: (dict(Ts=v[0][1], Td=v[0][2], stride=v[0][3], **{q: v[1][q] for q in ("v", "p5", "knee_sat")}) if v else None) for k, v in b.items()}
            md.append(f"| {c} | {fmt_pt(b['vmax'])} | {fmt_pt(b['vgood'])} | {fmt_pt(b['smax'])} | {fmt_pt(b['sgood'])} |")
        md.append("\n통과 비율(세 번 중) — 행: 설정 · 한 발 지지 시간, 열: 보폭 mm\n")
        strides = sorted({k[3] for k in T}); cads = sorted({(k[1], k[2]) for k in T}, reverse=True)
        md.append("| 설정 | 한 걸음 (s) | " + " | ".join(f"{s*1000:.1f}" for s in strides) + " |\n|---|---|" + "---|" * len(strides))
        for c in cfgs:
            for a, b_ in cads:
                if (c, a, b_, strides[0]) not in T:
                    continue
                md.append(f"| {c} | {a+b_:.2f} | " + " | ".join(f"{T[(c, a, b_, s)]['n_ok']}/{T[(c, a, b_, s)]['n']}" for s in strides) + " |")
        # 기준 걸음(0.65 s, 보폭 50 mm)에서 관절 사용률 나누기
        md.append("\n기준 걸음(한 걸음 0.65 s, 보폭 50 mm) 관절 사용률: 토크 몫 99백분위 / 속도 몫 99백분위 / 포화 % / 포화 순간 평균 속도 몫\n")
        md.append("| 설정 | 고관절 피치 | 무릎 | 발목 피치 | 고관절 롤 | 압력중심 여유 하위 5 % (mm) |\n|---|---|---|---|---|---|")
        out[grp + "_ref"] = {}
        for c in cfgs:
            v = T.get((c, 0.45, 0.2, 0.05))
            if not v or not v["split"]:
                md.append(f"| {c} | 넘어짐 | | | | |"); continue
            sp = v["split"]; out[grp + "_ref"][c] = dict(split=sp, p5=v["p5"], n_ok=v["n_ok"])
            cell = lambda j: f"{sp[j]['torque_p99']:.2f} / {sp[j]['speed_p99']:.2f} / {sp[j]['sat_frac']*100:.1f} / {sp[j]['speed_at_sat']:.2f}"
            md.append(f"| {c} | {cell('hip_pitch')} | {cell('knee')} | {cell('ankle_pitch')} | {cell('hip_roll')} | {v['p5']:.1f} |")
        md.append("")
    P = push_thr(load("push"))
    if P:
        out["push"] = P
        md.append("## push — 버틴 가장 큰 힘 (N, 0.2 s, 몸 무게중심 높이에 수평)\n\n| 설정 | 제자리 걸음 한 발 지지: 흔드는 발 쪽 (+y) | 디딤발 바깥쪽 (−y) | 앞 (+x) | 뒤 (−x) | 서 있기: 옆 +y | 옆 −y | 앞 | 뒤 |\n|---|---|---|---|---|---|---|---|---|")
        for c, d in P.items():
            md.append(f"| {c} | " + " | ".join(str(d.get(k, "–")) for k in ("ss+y", "ss-y", "ss+x", "ss-x", "stand+y", "stand-y", "stand+x", "stand-x")) + " |")
    so, smd = seg_summary()
    if so:
        out["seg"] = so; md += smd
    open(os.path.join(RES, "summary.md"), "w").write("\n".join(md) + "\n")
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print("\n".join(md))


def seg_summary():
    """허벅지 · 정강이 늘림(질량 반영) × 롤-피치 늘림 × 웅크림"""
    rows = load("seg")
    if not rows:
        return None, []
    for r in rows:
        m = r["meta"]; m["cfg"] = (m["L"], m["ext"], m["crouch"])
    T = rep_table(rows)
    P = defaultdict(dict)
    for r in load("seg_push"):
        m = r["meta"]; P[(m["L"], m["ext"], m["crouch"], m["where"], m["dir"])][m["F"]] = r["res"]
    def thr(d):
        th = 0
        for F in sorted(d):
            if d[F].get("fallen"):
                break
            th = F
        return th
    S = json.load(open(os.path.join(RES, "static.json"))) if os.path.exists(os.path.join(RES, "static.json")) else []
    st = {(round(x["L"], 4), round(x["ext"], 3)): x for x in S}
    out = []; md = ["## seg — 허벅지 · 정강이 늘림(질량 반영) × 롤-피치 늘림 × 웅크림\n",
                    "| 롤-피치 늘림 | 허벅지 · 정강이 각 | 웅크림 | 몸무게 (kg) | 세 번 다 통과한 최대 보폭 (0.65 s) | 쓸 만한 최고 속도 | 기준 걸음 여유 하위 5 % (mm) | 기준 걸음 무릎 토크 몫 99백분위 / RMS / 포화 % | 밀기 버팀 한 발 지지 옆 · 앞 / 서 있기 옆 · 앞 (N) | 한 발 정지 무릎 토크 (N·m, 정지토크 3.8 대비) | 쪼그림 무릎 토크 (N·m) |",
                    "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in sorted({k[0] for k in T}):
        L, e, cr = cfg
        pts = {k: v for k, v in T.items() if k[0] == cfg}
        b = best(T, cfg)
        smax = max([k[3] for k, v in pts.items() if k[1] == 0.45 and v["n_ok"] == v["n"]] or [0])
        ref = T.get((cfg, 0.45, 0.2, 0.05))
        sx = st.get((round(L, 4), round(e, 3)))
        ol = sx["one_leg"][str(cr)]["knee_Nm"] if sx else None
        sq = sx["squat"]["100"]["knee_Nm"] if sx and sx["squat"].get("100") else None
        pu = [thr(P[(L, e, cr, w, d)]) for w in ("ss", "stand") for d in ("+y", "+x")]
        mass = next((r["res"].get("mass_kg") for r in rows if r["meta"]["cfg"] == cfg and "mass_kg" in r["res"]), None)
        row = dict(L=L, ext=e, crouch=cr, mass=mass, smax=smax, vgood=b["vgood"][1]["v"] if b["vgood"] else None,
                   vgood_pt=b["vgood"][0][1:] if b["vgood"] else None, p5=ref["p5"] if ref and ref["n_ok"] else None,
                   knee=ref["split"]["knee"] if ref and ref["split"] else None, push=pu, one_leg_knee=ol, squat_knee=sq)
        out.append(row)
        kn = row["knee"]
        md.append(f"| {L*1e3:.0f} | {e*1e3:.0f} | {cr*1e3:.0f} | {mass:.3f} | {smax*1e3:.0f} | " +
                  (f"{row['vgood']:.3f} (한 걸음 {sum(row['vgood_pt'][:2]):.2f} s, 보폭 {row['vgood_pt'][2]*1e3:.0f})" if row["vgood"] else "–") + " | " +
                  (f"{row['p5']:.1f}" if row["p5"] is not None else "넘어짐") + " | " +
                  (f"{kn['torque_p99']:.2f} / {kn['torque_rms']:.2f} / {kn['sat_frac']*100:.1f}" if kn else "–") + " | " +
                  f"{pu[0]} · {pu[1]} / {pu[2]} · {pu[3]} | " + (f"{ol:.2f} ({ol/3.8:.2f})" if ol else "–") + " | " + (f"{sq:.2f}" if sq else "–") + " |")
    return out, md


if __name__ == "__main__":
    main()
