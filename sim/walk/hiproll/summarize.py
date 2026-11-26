"""compare.json 요약표 → hiproll/results/compare_tables.md (그리고 화면)"""
import os, json
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
rows = json.load(open(os.path.join(RES, "compare.json")))
G = defaultdict(list)
for r in rows:
    m = r["meta"]; G[(m["model"], m["k_lat"], m["pack"])].append(r)
md = ["| 모델 | 옆 발 디딤 | 배터리 | 최고 명령 속도 (m/s) | 0.076 m/s 여유 하위 5 % (mm) | 0.076 m/s 무릎 u 99 / 포화 % | 0.076 m/s 고관절 롤 u 99 / 포화 % | 0.076 m/s 몸통 롤 최대 (°) | 밀기 버팀 N: 디딤발 바깥 · 흔드는 발 쪽 · 앞 · 뒤 |",
      "|---|---|---|---|---|---|---|---|---|"]
def thr(rs, dr):
    pr = sorted([r for r in rs if r["meta"]["test"] == "push" and r["meta"]["dir"] == dr], key=lambda r: r["meta"]["F"])
    ff = next((r["meta"]["F"] for r in pr if r["res"]["fallen"]), None)
    return max([r["meta"]["F"] for r in pr if not r["res"]["fallen"] and (ff is None or r["meta"]["F"] < ff)] or [0])
for (mn, kl, pk), rs in G.items():
    sp = [r for r in rs if r["meta"]["test"] == "speed"]
    ok = [r["meta"]["stride"] for r in sp if not r["res"]["fallen"]]
    vmax = max(ok) / 0.65 if ok else 0.0
    r76 = next((r["res"] for r in sp if abs(r["meta"]["stride"] - 0.05) < 1e-9), None)
    def jg(d, key):
        v = [x for k, x in d.items() if k.endswith(key)]; return max(v) if v else None
    if r76 and not r76["fallen"]:
        kn = f"{jg(r76['u_p99'], 'knee'):.2f} / {100 * jg(r76['sat_frac'], 'knee'):.1f}"
        hrl = f"{jg(r76['u_p99'], 'hip_roll'):.2f} / {100 * jg(r76['sat_frac'], 'hip_roll'):.1f}" if jg(r76['u_p99'], 'hip_roll') is not None else "–"
        mg = f"{r76['cop_margin_mm']['p5']:.1f}"; rl = f"{max(abs(r76['torso_roll_deg'][0]), abs(r76['torso_roll_deg'][1])):.1f}"
    else:
        kn = hrl = mg = rl = "넘어짐"
    md.append(f"| {mn} | {'켬' if kl else '끔'} | {pk} | {vmax:.3f} | {mg} | {kn} | {hrl} | {rl} | {thr(rs, '-y')} · {thr(rs, '+y')} · {thr(rs, '+x')} · {thr(rs, '-x')} |")
open(os.path.join(RES, "compare_tables.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
