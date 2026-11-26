"""GETUP_STUDY.md 의 표 셋(GETUP_TABLE · ROBUST_TABLE · JOINT_TABLE)을 results 에서 만들어 끼움(한 번만, 자리표시가 남아 있을 때)."""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gu_model import ARMS, RES, reach, arm_mass_g
from kw_common import SERVO_SIM

SUMS = ["search", "pike_hip100", "r3_r3hip100", "r4xl_r4rob_r5arm110_r6prone"]
grp = lambda s: "누움" if s.startswith("supine") else "엎드림"


def getup_table():
    rows = [r for t in SUMS for r in json.load(open(os.path.join(RES, f"summary_{t}.json")))]
    best, tot = {}, {}
    for r in rows:
        k = (r["arm"], grp(r["start"]))
        tot.setdefault(k, [0, 0]); tot[k][0] += r["n_ok"]; tot[k][1] += r["n_eval"]
        if k not in best or (r["ok"], r["z_end_pct"]) > (best[k]["ok"], best[k]["z_end_pct"]):
            best[k] = r
    md = ["| 팔 | 닿는 길이 mm | 팔 하나 g | 시작 | 성공 (성공한 후보 수 / 돌린 수, 모든 묶음) | 가장 좋은 시도: 끝 높이 % | 지나는 동안 최고 높이 % | 어깨 토크 몫 · 포화 % | 팔꿈치 토크 몫 · 포화 % | 묶음 · 고관절 한계 |",
          "|" + "---|" * 10]
    for (a, g), r in best.items():
        el = f"{r['tq']['elbow']:.2f} · {r['sat']['elbow'] * 100:.0f}" if "elbow" in r["tq"] else "—"
        md.append(f"| {a} | {reach(ARMS[a]) * 1e3:.0f} | {arm_mass_g(ARMS[a]):.0f} | {g} | {'**성공**' if tot[(a, g)][0] else '실패'} ({tot[(a, g)][0]}/{tot[(a, g)][1]}) | "
                  f"{r['z_end_pct']:.0f} | {r['max_z_pct']:.0f} | {r['tq']['arm_swing']:.2f} · {r['sat']['arm_swing'] * 100:.0f} | {el} | {r['tag']} · {r['hip_min_deg']:.0f}° |")
    return "\n".join(md) + "\n\n- 토크 몫은 그 팔의 서보 정지토크 대비입니다(XL 1.4, W210 2.7, XM 3.8 N·m). 1 을 넘는 것은 감쇠 몫을 뺀 부하가 정지토크보다 컸던 순간입니다(서보 힘 자체는 한계에서 잘림).\n- `r4rob` 의 성공 수는 조건 5 가지(기준 · 마찰 0.6 · 1.0 · 시간 ×0.95 · 1.05)에서 모두 선 후보만 셉니다."


def robust_table():
    md = ["| 동작(묶음, 팔) | 성공 / 15 조건 | 실패한 조건 |", "|---|---|---|"]
    for t, a in (("r3", "팔꿈치 140+140 · XM"), ("r4rob", "팔꿈치 140+140 · XM"), ("r4xl", "팔꿈치 140+140 · W210"),
                 ("r5arm110", "팔꿈치 140+140 · XM"), ("r5arm110", "팔꿈치 140+140 · W210"), ("r3hip100", "팔꿈치 140+140 · XM")):
        p = os.path.join(RES, f"robust_{t}.json")
        if not os.path.exists(p) or f"{a}/supine6" not in json.load(open(p)):
            continue
        R = json.load(open(p))[f"{a}/supine6"]
        md.append(f"| {t}, {a}, 누움 | {sum(r['ok'] for r in R)} | {', '.join(r['name'] for r in R if not r['ok']) or '없음'} |")
    return "\n".join(md) + ("\n\n- 15 조건: 기준, 바닥 마찰 0.5 · 0.65 · 1.0 · 1.2(기준 0.8), 동작 시간 ×0.9 · 0.95 · 1.05 · 1.1, 위치 이득 ×0.75 · 1.25, 팔 질량 ×0.8 · 1.2, "
                            "처음 가라앉는 시간 +0.05 · +0.2 s.\n- `r4rob` 은 조건 5 가지의 평균 비용으로 다시 최적화한 것이라 버티는 조건이 가장 많습니다.")


def joint_table():
    md = ["| 동작 | 어깨 범위 ° | 팔꿈치 ° | 고관절 피치 ° | 무릎 ° | 발목 피치 ° | 어깨 최대 N·m (99번째 백분위) | 팔꿈치 최대 N·m | 고관절 최대 N·m | 무릎 최대 N·m |",
          "|" + "---|" * 10]
    for t, a in (("r4rob", "팔꿈치 140+140 · XM"), ("r4xl", "팔꿈치 140+140 · W210"), ("r5arm110", "팔꿈치 140+140 · XM")):
        r = next(x for x in json.load(open(os.path.join(RES, f"{t}.json"))) if x["arm"] == a and x["start"] == "supine6")
        K = r["keys"]
        rng = lambda j: f"{min(k[1][j] for k in K) * 57.3:.0f} ~ {max(k[1][j] for k in K) * 57.3:.0f}"
        ts = {j: SERVO_SIM[ARMS[a].get("shoulder" if j == "arm_swing" else "elbow_servo", "XL430-W250")][0] if j in ("arm_swing", "elbow") else 3.8 for j in ("arm_swing", "elbow", "hip_pitch", "knee")}
        tq = lambda j: f"{r['res']['torque_max'][j] * ts[j]:.1f} ({r['res']['torque_p99'][j] * ts[j]:.1f})"
        md.append(f"| {t}, {a} | {rng('arm')} | {rng('elbow')} | {rng('hip')} | {rng('knee')} | {rng('ankle')} | {tq('arm_swing')} | {tq('elbow')} | {tq('hip_pitch')} | {tq('knee')} |")
    return "\n".join(md) + ("\n\n- 범위는 키프레임 목표의 최소 ~ 최대입니다(실제 관절은 목표를 따라가며 조금 덜 갑니다). 무릎 100° · 고관절 −80° · 발목 −50° 설계 끝까지 다 씁니다.\n"
                            "- 어깨는 +120° 끝까지 썼지만 ±110° 로 줄여 다시 찾아도 섰습니다(`r5arm110`).\n- 토크 [시뮬 실측]: 부하 토크(서보 힘 − 감쇠 × 속도). XL430 정지토크 1.4 N·m 를 어깨가 2배 넘게 요구합니다.")


if __name__ == "__main__":
    p = os.path.join(HERE, "GETUP_STUDY.md"); s = open(p).read()
    s = s.replace("GETUP_TABLE", getup_table()).replace("ROBUST_TABLE", robust_table()).replace("JOINT_TABLE", joint_table())
    open(p, "w").write(s)
    print(getup_table()); print(robust_table()); print(joint_table())
