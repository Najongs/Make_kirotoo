"""rest.json 요약 → results/rest_summary.md. 걷기:쉬기 비율 한계 f = (P_한계 − P_쉬기) / (P_걷기 − P_쉬기), 관절마다(무릎 · 고관절 롤) 작은 쪽.
P_한계: 25 K/W 로 65 °C → 1.6 W, 15 K/W 로 65 °C → 2.67 W. 「대기 포함」은 서보마다 대기 전력 52 mA × 11.1 V = 0.58 W [e-manual] 를 걷기 · 쉬기 둘 다에 더함
(heat.py 는 대기 전력을 뺐다). 열 시정수(몇 분, 값 모름)보다 짧게 번갈아 한다고 보고 평균 전력으로 본다 [추정].
usage: cd sim/walk/allxl && ../../.venv/bin/python ax_rest_summary.py"""
import os, sys, json, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ax_common as A
R = json.load(open(os.path.join(A.RES, "rest.json")))["rows"]
STBY = 0.052 * 11.1
LIM = {"hi": 40 / 25, "lo": 40 / 15}
REST_CR = 0.001


def P(r, k):
    J = (r["res"].get("joints") or {}).get(k)
    return J["heat"]["P_W"] if J and J.get("heat") else None


def frac(pw, pr, lim):
    if pw <= lim:
        return 1.0
    return max(0.0, (lim - pr) / (pw - pr))


rest = {(r["meta"]["cfg"], r["meta"]["crouch"]): r for r in R if r["meta"]["test"] == "rest"}
trans = {(r["meta"]["cfg"], r["meta"]["crouch"]): r for r in R if r["meta"]["test"] == "trans"}
walk = collections.defaultdict(list)
for r in R:
    if r["meta"]["test"] == "walk":
        walk[(r["meta"]["cfg"], r["meta"]["crouch"], r["meta"]["stride"])].append(r)
cfgs = list(dict.fromkeys(r["meta"]["cfg"] for r in R))
L = ["**쉬기(두 발 서기 20 s): 무릎 서보 한 개 — 부하 RMS N·m / 구리 손실 W / 정상 온도 °C (15–25 K/W)**", "",
     "| 구성 | 웅크림 1 mm (무릎 약 12°) | 3 mm (약 20°) | 35 mm (약 63°, 지금 v20A 걷기 자세) |", "|---|---|---|---|"]
for c in cfgs:
    cells = []
    for cr in (0.001, 0.003, 0.035):
        r = rest[(c, cr)]; j = r["res"]["joints"]["knee"]; h = j["heat"]
        cells.append(f"{j['load_rms']:.2f} / {h['P_W']:.2f} / {h['T_lo']:.0f}–{h['T_hi']:.0f}" + (" (넘어짐)" if r["res"]["fallen"] else ""))
    L.append(f"| {c} | " + " | ".join(cells) + " |")
L += ["", "**걷기(한 걸음 0.65 s) + 편 무릎 쉬기(웅크림 1 mm): 걷기 시간 비율 한계**", "",
      "칸: 「세 판 통과 수, 압력중심 여유 mm(세 판 최소) / 무릎 포화 % · 롤 포화 % / 무릎 · 롤 걷기 손실 W / 비율 한계 25 K/W · 15 K/W · 25 K/W 대기 포함」", "",
      "| 구성 | 웅크림 | 0.04 m/s | 0.05 m/s | 0.077 m/s |", "|---|---|---|---|---|"]
best = {}
for c in cfgs:
    pr = {k: P(rest[(c, REST_CR)], k) or 0.0 for k in ("knee", "hip_roll")}
    for cr in (0.015, 0.02, 0.025):
        cells = []
        for s in (0.026, 0.0325, 0.05):
            rs = walk[(c, cr, s)]; n_ok = sum(not x["res"]["fallen"] for x in rs)
            p5 = min([x["res"]["cop_margin_mm"]["p5"] for x in rs if not x["res"]["fallen"]] or [float("nan")])
            r8 = [x for x in rs if x["meta"]["fr"] == 0.8][0]; J = r8["res"]["joints"]
            pw = {k: P(r8, k) for k in ("knee", "hip_roll")}
            f = {m: min(frac(pw[k], pr[k], LIM[m]) for k in pw) for m in ("hi", "lo")}
            f["stby"] = min(frac(pw[k] + STBY, pr[k] + STBY, LIM["hi"]) for k in pw)
            cells.append(f"{n_ok}/3, {p5:.1f} / {J['knee']['sat_pct']:.1f} · {J['hip_roll']['sat_pct']:.1f} / {pw['knee']:.2f} · {pw['hip_roll']:.2f} / "
                         f"**{100 * f['hi']:.0f}** · {100 * f['lo']:.0f} · {100 * f['stby']:.0f} %")
            if n_ok == 3:
                key = (c, s)
                cand = dict(crouch=cr, stride=s, p5=p5, f=f, knee_sat=J["knee"]["sat_pct"], roll_sat=J["hip_roll"]["sat_pct"], pw=pw, pr=pr)
                if p5 >= 5 and (key not in best or f["hi"] > best[key]["f"]["hi"]):
                    best[key] = cand
        L.append(f"| {c} | {cr * 1000:.0f} mm | " + " | ".join(cells) + " |")
L += ["", "**전환 동작(편 무릎 2 mm 4 s → 1 s 웅크림 → 0.05 m/s 18걸음 → 1 s 펴기 → 4 s 쉬기, 마찰 0.8)**", "",
      "| 구성 | 걷기 웅크림 20 mm | 35 mm |", "|---|---|---|"]
for c in cfgs:
    cells = []
    for cr in (0.02, 0.035):
        r = trans[(c, cr)]["res"]
        cells.append(("넘어짐 " + f"{r['fall_t']:.1f} s ({r['steps_done']}걸음)") if r["fallen"] else f"✓ 18걸음, 여유 {r['cop_margin_mm']['p5']:.1f} mm")
    L.append(f"| {c} | " + " | ".join(cells) + " |")
L += ["", "**구성 · 속도마다 가장 좋은 칸(세 판 통과 · 여유 ≥ 5 mm 중 비율 한계 25 K/W 가 가장 큰 웅크림)**", "",
      "| 구성 | 속도 | 웅크림 | 여유 mm | 무릎 · 롤 포화 % | 비율 한계 25 K/W · 15 K/W · 25 K/W 대기 포함 |", "|---|---|---|---|---|---|"]
for (c, s), b in best.items():
    L.append(f"| {c} | {s / 0.65:.3f} m/s | {b['crouch'] * 1000:.0f} mm | {b['p5']:.1f} | {b['knee_sat']:.1f} · {b['roll_sat']:.1f} | "
             f"{100 * b['f']['hi']:.0f} · {100 * b['f']['lo']:.0f} · {100 * b['f']['stby']:.0f} % |")
open(os.path.join(A.RES, "rest_summary.md"), "w").write("\n".join(L) + "\n")
json.dump({f"{c} | {s}": b for (c, s), b in best.items()}, open(os.path.join(A.RES, "rest_best.json"), "w"), indent=1, ensure_ascii=False)
print("\n".join(L))
