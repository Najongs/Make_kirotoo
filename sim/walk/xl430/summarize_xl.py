"""walk_xl.json 요약 → xl430/results/walk_xl.md"""
import os, json
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
rows = json.load(open(os.path.join(RES, "walk_xl.json")))
G = defaultdict(dict)
for r in rows:
    m = r["meta"]; G[(m["model"], m["kp"], m["crouch"])][m["test"]] = r["res"]
def jg(d, key):
    v = [x for k, x in d.items() if k.endswith(key)]; return max(v) if v else None
md = ["| 모델 | 이득 | 웅크림 mm | 서기 | 제자리 | 최고 속도 (m/s, 시험한 최대 보폭) | 제자리: 무릎 u99 / 포화 % | 제자리: 발목 피치 u99 / 포화 % | 제자리: 고관절 피치 u99 / 포화 % | 0.046 m/s 여유 하위 5 % (mm) | 넘어진 판의 원인(포화가 가장 큰 관절) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for (mn, kp, cr), T in G.items():
    ok = [int(k.split()[1]) for k, o in T.items() if k.startswith("보폭") and not o["fallen"]]
    strides = sorted(int(k.split()[1]) for k in T if k.startswith("보폭"))
    allok = all(not T[f"보폭 {s}"]["fallen"] for s in strides if s <= max(ok)) if ok else False
    vmax = (max(ok) / 1000 / 0.65) if ok else 0.0
    st = T["서기"]; ip = T["제자리"]
    def cell(o):
        return "✓" if not o["fallen"] else f"넘어짐 {o['fall_t']:.1f} s"
    def us(o, k):
        if "u_p99" not in o: return "–"
        return f"{jg(o['u_p99'], k):.2f} / {100 * jg(o['sat_frac'], k):.1f}"
    w46 = T.get("보폭 30", {})
    mg = f"{w46['cop_margin_mm']['p5']:.1f}" if w46 and not w46["fallen"] else "넘어짐"
    w77 = T.get("보폭 50", {})
    k77 = us(w77, "knee") if w77 and not w77["fallen"] else "넘어짐"
    fails = [o for k, o in T.items() if o["fallen"]]
    cause = ""
    if fails:
        o = min(fails, key=lambda o: o["fall_t"])
        top = max(o["sat_frac"].items(), key=lambda kv: kv[1]) if "sat_frac" in o else ("", 0)
        cause = f"{top[0]} 포화 {100 * top[1]:.1f} %"
    md.append(f"| {mn} | {kp:.0f} | {cr * 1000:.0f} | {cell(st)} | {cell(ip)} | {vmax:.3f} ({max(strides)} mm){'' if allok else ' (중간 보폭 넘어짐)'} | {us(ip, 'knee')} | {us(ip, 'ankle_pitch')} | {us(ip, 'hip_pitch')} | {mg} | {k77} | {cause} |")
open(os.path.join(RES, "walk_xl.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
