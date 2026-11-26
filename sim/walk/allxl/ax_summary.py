"""묶음 결과 요약표. usage: cd sim/walk/allxl && ../../.venv/bin/python ax_summary.py base fix combo → results/summary.md (+ summary.json)
- 서기: 12 s 넘어지지 않음.  0.05 · 0.077 m/s: 한 걸음 0.65 s, 보폭 32.5 · 50 mm, 마찰 0.7 · 0.8 · 0.9 세 판 모두 18걸음.
- 관절 지표(오른다리)는 마찰 0.8 판: 포화 % · u 99 % · 걷기 구간 부하 RMS(N·m) · 발열 정상 온도(15–25 K/W).
- 최고 속도: 세 판 다 통과한 가장 빠른 칸. 「쓸 만한」: 거기에 압력중심 여유 하위 5 % ≥ 5 mm(세 판 최소), 무릎 포화 < 10 %(세 판 최대)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ax_common as A
from collections import defaultdict
JN = ("knee", "ankle_pitch", "hip_pitch", "hip_roll", "ankle_roll", "hip_yaw")
KO = dict(knee="무릎", ankle_pitch="발목 피치", hip_pitch="고관절 피치", hip_roll="고관절 롤", ankle_roll="발목 롤", hip_yaw="고관절 요")


def ok(r):
    return "error" not in r and not r.get("fallen")


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["meta"]["cfg"]].append(r)
    out = {}
    for name, rs in by.items():
        o = dict(name=name)
        st = [r for r in rs if r["meta"]["test"] == "stand"][0]
        o["mass_kg"] = next((r["res"]["mass_kg"] for r in rs if "mass_kg" in r["res"]), None)
        o["early_fall_all"] = all("zero-size array" in r["res"].get("error", "") for r in rs)   # 시작 1 초 안에 넘어져 요약 구간이 없음
        o["stand_ok"] = ok(st["res"]); o["stand"] = st["res"].get("joints")
        cells = defaultdict(list)
        for r in rs:
            m = r["meta"]
            if m["test"] == "walk":
                cells[(m["Ts"], m["Td"], m["stride"])].append(r)
        def cell(key):
            c = cells.get(key, [])
            return dict(all_ok=len(c) == 3 and all(ok(r["res"]) for r in c), n_ok=sum(ok(r["res"]) for r in c),
                        p5=min([r["res"]["cop_margin_mm"]["p5"] for r in c if ok(r["res"])] or [None]) if c else None,
                        knee_sat=max([r["res"]["joints"]["knee"]["sat_pct"] for r in c if "joints" in r["res"] and r["res"]["joints"]] or [None]) if c else None,
                        ref=next((r["res"] for r in c if r["meta"]["fr"] == 0.8), None),
                        speed=[r["res"].get("speed_mps") for r in c if r["meta"]["fr"] == 0.8])
        o["w05"] = cell((0.45, 0.2, 0.0325)); o["w077"] = cell((0.45, 0.2, 0.05))
        best = None; usable = None
        for key, c in cells.items():
            v = key[2] / (key[0] + key[1]); ce = cell(key)
            if ce["all_ok"]:
                if best is None or v > best[0]:
                    best = (v, key, ce)
                if (ce["p5"] or 0) >= 5 and (ce["knee_sat"] or 0) < 10 and (usable is None or v > usable[0]):
                    usable = (v, key, ce)
        o["best"] = best and dict(v=best[0], Ts=best[1][0], Td=best[1][1], stride=best[1][2], p5=best[2]["p5"], knee_sat=best[2]["knee_sat"])
        o["usable"] = usable and dict(v=usable[0], Ts=usable[1][0], Td=usable[1][1], stride=usable[1][2], p5=usable[2]["p5"], knee_sat=usable[2]["knee_sat"])
        out[name] = o
    return out


def jrow(J, k):
    if not J or k not in J:
        return "–"
    j = J[k]; h = j.get("heat") or {}
    t = f"{h['T_lo']:.0f}–{h['T_hi']:.0f}" if h else "?"
    return f"{j['sat_pct']:.1f} / {j['u_p99']:.2f} / {j['load_rms']:.2f} / {t}"


def fmt(o):
    w05, w77 = o["w05"], o["w077"]
    def walk(c):
        return ("✓" if c["all_ok"] else f"✗ ({c['n_ok']}/3)") + (f", 여유 {c['p5']:.1f} mm" if c["all_ok"] and c["p5"] is not None else "")
    b = o["best"]; u = o["usable"]
    if o.get("early_fall_all"):
        return f"| {o['name']} | – | ✗ (시작 1 초 안에 넘어짐) | ✗ | ✗ | – | 없음 |"
    L = [f"| {o['name']} | {(o['mass_kg'] or 0) * 1000:.0f} | {'✓' if o['stand_ok'] else '✗'} | {walk(w05)} | {walk(w77)} | "
         + (f"{b['v']:.3f} ({b['stride'] * 1000:.0f} mm, {b['Ts'] + b['Td']:.2f} s)" if b else "–") + " | "
         + (f"{u['v']:.3f}" if u else "없음") + " |"]
    return L[0]


if __name__ == "__main__":
    allo = {}
    for g in sys.argv[1:]:
        p = os.path.join(A.RES, f"{g}.json")
        if os.path.exists(p):
            allo.update(summarize(json.load(open(p))["rows"]))
    lines = ["| 구성 | 몸무게 (g) | 서기 12 s | 0.05 m/s | 0.077 m/s | 넘어지지 않은 최고 속도 m/s (보폭, 한 걸음) | 쓸 만한 최고 속도 m/s |", "|---|---|---|---|---|---|---|"]
    lines += [fmt(o) for o in allo.values()]
    for title, key in (("0.05 m/s 걷기 (마찰 0.8)", "w05"), ("0.077 m/s 걷기 (마찰 0.8)", "w077"), ("서기 12 s", "stand")):
        lines += ["", f"**{title}** — 칸마다 「포화 % / 사용률 99 % / 부하 RMS N·m / 정상 온도 °C」", "",
                  "| 구성 | " + " | ".join(KO[k] for k in JN) + " |", "|---" * (len(JN) + 1) + "|"]
        for o in allo.values():
            J = o[key] if key == "stand" else (o[key]["ref"] or {}).get("joints")
            fell = "" if key == "stand" or (o[key]["ref"] and not o[key]["ref"].get("fallen")) else " (넘어짐)"
            lines.append(f"| {o['name']}{fell} | " + " | ".join(jrow(J, k) for k in JN) + " |")
    open(os.path.join(A.RES, "summary.md"), "w").write("\n".join(lines) + "\n")
    json.dump(allo, open(os.path.join(A.RES, "summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print("\n".join(lines))
