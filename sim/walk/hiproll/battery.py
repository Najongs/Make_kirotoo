"""배터리 사용 시간 · 전압 처짐 계산 + 버스 전류 그림. current_budget.json 을 읽는다.
python hiproll/battery.py → hiproll/results/battery.json, battery.md, current_bus.png"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
rows = {r["name"]: r for r in json.load(open(os.path.join(RES, "current_budget.json")))}
OPENCR_A = 0.10          # OpenCR 보드 자체(STM32F7 · IMU · LED) [추정]
USABLE = 0.8             # LiPo 용량 중 쓸 수 있는 몫 [추정: 3.5 V/셀 까지]
PACKS = {"LB-012 3S 1800 mAh (5C, 지금)": dict(Ah=1.8, R=0.09, cont=9.0, g=106),
         "3S 2200 mAh (25C급) [추정]": dict(Ah=2.2, R=0.04, cont=55.0, g=180),
         "3S 3000 mAh (25C급) [추정]": dict(Ah=3.0, R=0.03, cont=75.0, g=240)}
KEYS = ["서 있기", "걷기 0.076 m/s", "걷기 0.083 m/s", "끄덕이며 걷기 0.076 m/s · 1 Hz", "고관절 롤 B(XM) 걷기 0.076 m/s",
        "고관절 롤 B(XM) 끄덕이며 걷기 0.076 m/s · 1 Hz"]
out = {"assume": dict(opencr_A=OPENCR_A, usable=USABLE, V=11.1), "runtime_min": {}, "sag": {}}
md = ["| 상황 | 평균 전류 상한(스펙 방식) A | 평균 전류(전력 방식) A | " + " | ".join(f"{p}: 사용 시간 분(상한 ~ 전력 방식)" for p in PACKS) + " |",
      "|---|---|---|" + "---|" * len(PACKS)]
for k in KEYS:
    r = rows[k]; hi = r["total"]["mean_A"] + OPENCR_A; lo = r["total"]["power_based_mean_A"] + OPENCR_A
    cells = []
    for pn, p in PACKS.items():
        a, b = 60 * p["Ah"] * USABLE / hi, 60 * p["Ah"] * USABLE / lo
        cells.append(f"{a:.0f} ~ {b:.0f}"); out["runtime_min"].setdefault(k, {})[pn] = [a, b]
    md.append(f"| {k} | {hi:.2f} | {lo:.2f} | " + " | ".join(cells) + " |")
md.append("")
md.append("| 상황 | 20 ms 최대 A | 1 s 최대 A | " + " | ".join(f"{p}: 20 ms 최대 때 처짐 V (만충 12.6 → / 끝무렵 10.5 →)" for p in PACKS) + " |")
md.append("|---|---|---|" + "---|" * len(PACKS))
for k in KEYS + ["넘어짐(걷다 앞 15 N 밀기)", "넘어짐(서 있다 옆 25 N 밀기)", "고관절 롤 B(XM) 넘어짐(서 있다 옆 25 N)"]:
    r = rows[k]; pk = r["total"]["max_20ms_A"] + OPENCR_A; p1 = r["total"]["max_1s_A"] + OPENCR_A
    cells = []
    for pn, p in PACKS.items():
        dv = pk * p["R"]; cells.append(f"{dv:.2f} ({12.6 - dv:.1f} / {10.5 - dv:.1f})")
        out["sag"].setdefault(k, {})[pn] = dv
    md.append(f"| {k} | {pk:.1f} | {p1:.1f} | " + " | ".join(cells) + " |")
# Jetson
J = {"Jetson 5 W 모드 + 주변 1 W": 6.0, "Jetson 10 W 모드 + 주변 1.5 W": 11.5}
wh = 7.4 * 1.5 * USABLE * 0.88       # 2S 1500 mAh, 강압 효율 88 % [추정]
md.append("")
md.append("| Jetson 배터리 2S 1500 mAh (11.1 Wh, 쓸 몫 80 %, 강압 88 %) | 소비 W | 사용 시간 분 | 5 V 3 A 강압 여유 |")
md.append("|---|---|---|---|")
for k, w in J.items():
    md.append(f"| {k} | {w:.1f} | {60 * wh / w:.0f} | {w / 15 * 100:.0f} % of 15 W |")
    out.setdefault("jetson_min", {})[k] = 60 * wh / w
open(os.path.join(RES, "battery.md"), "w").write("\n".join(md) + "\n")
json.dump(out, open(os.path.join(RES, "battery.json"), "w"), indent=1, ensure_ascii=False)
print("\n".join(md))

# 그림
import sys
sys.path.insert(0, os.path.dirname(HERE))
from kw_common import korean_font
plt = korean_font()
fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
for a, k in zip(ax, ("걷기 0.076 m/s", "넘어짐(걷다 앞 15 N 밀기)")):
    s = rows[k]["series"]; t = np.array(s["t"])
    tot = np.zeros_like(t)
    for bn, v in s["bus"].items():
        v = np.array(v); a.plot(t, v, lw=0.8, label=bn); tot += v
    a.plot(t, tot, color="k", lw=1.0, label="전체")
    w = 50; mv = np.convolve(tot, np.ones(w) / w, mode="same")
    a.plot(t, mv, color="#c0392b", lw=1.6, label="전체 1 s 이동 평균")
    a.axhline(10, color="#8e44ad", ls="--", lw=1, label="OpenCR 입력 퓨즈 10 A")
    a.axhline(9, color="#e67e22", ls=":", lw=1, label="LB-012 연속 9 A (5C)")
    a.set_ylabel("전류 상한 (A)"); a.grid(alpha=0.3); a.legend(fontsize=8, ncol=4); a.set_title(k, fontsize=10)
ax[1].set_xlabel("시간 (s)")
r = rows["걷기 0.076 m/s"]["total"]
fig.suptitle(f"걷기(0.076 m/s) 서보 전류는 평균 {r['mean_A']:.1f} A · 1 s 최대 {r['max_1s_A']:.1f} A 로 퓨즈 10 A · 배터리 연속 9 A 아래지만, "
             f"5 ms 순간값은 발 뗄 때 · 디딜 때 {r['max_A']:.0f} A 까지 튄다\n(스펙 방식 상한 = 대기 전류 + 부하 토크 / 토크상수, e-manual 11.1 V 값, 시뮬 실측 토크)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(RES, "current_bus.png"), dpi=90)
print("saved")
