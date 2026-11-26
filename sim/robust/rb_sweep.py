"""강건성 묶음 실행: python robust/rb_sweep.py [oat combo]  → robust/results/<묶음>.json
oat  : 한 번에 하나씩 바꾸기(배낭 없음 · 배낭 있음 두 기준 × 정지 자세 12 s · 0.077 m/s 18걸음)
combo: 나쁜 쪽 조합 하나 + 하나씩 빼 본 조합(어느 것이 결정적인지)
extra: oat 결과를 보고 더한 것(무게중심 경계 · 제어기가 무게중심을 알 때 · 흩어짐)"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rb_common import RES, STRIDE, ROB0, run_jobs, brief, jdump

AXES = {
    "바닥 마찰": ("friction", (0.1, 0.15, 0.2, 0.3, 0.5, 1.2)),            # 기준 0.8
    "기어 백래시 (°)": ("backlash_deg", (0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)),   # 기준 0
    "내부 마찰 (정지토크 대비)": ("frictionloss_frac", (0.01, 0.03, 0.05, 0.10, 0.15, 0.20)),
    "전압 (V)": ("volt", (12.0, 10.5, 10.0, 9.5, 9.0, 8.5)),                  # 기준 11.1
    "제어 지연 (ms)": ("delay_ms", (15.0, 20.0, 30.0, 40.0, 50.0)),           # 기준 10
    "질량 배수": ("mass_f", (0.7, 0.8, 0.85, 0.9, 1.1, 1.15, 1.2, 1.3)),
    "무게중심 앞뒤 (m)": ("com_dx_m", (-0.03, -0.02, -0.015, -0.01, 0.01, 0.015, 0.02, 0.03)),
}
TESTS = {"정지 자세": dict(t_walk=99, T_end=12.0), "걷기 0.077 m/s": dict(stride=STRIDE)}
# 나쁜 쪽 조합: 요청 범위의 나쁜 끝(마찰 0.5 는 범위 안 「흔한 미끄러운 바닥」, 0.3 은 따로)
WORST = dict(friction=0.5, backlash_deg=0.5, frictionloss_frac=0.05, volt=10.5, delay_ms=20.0, mass_f=1.15, com_dx_m=0.01)


def g_oat():
    J = []
    for bp in (False, True):
        for tn, tc in TESTS.items():
            J.append((dict(backpack=bp, test=tn, axis="기준", val=None), dict(tc, backpack=bp)))
            for an, (key, vals) in AXES.items():
                for v in vals:
                    J.append((dict(backpack=bp, test=tn, axis=an, val=v), dict(tc, backpack=bp, **{key: v})))
    return J


def g_combo():
    """나쁜 쪽 조합: 무게중심 +10 · −10 mm(제어기 모름)와 0(실물 무게중심을 재서 맞춘 경우) 세 갈래, 각각 전부 + 하나씩 뺌"""
    J = []
    for bp in (False, True):
        for dx in (0.01, -0.01, 0.0):
            w = dict(WORST, com_dx_m=dx)
            for tn, tc in TESTS.items():
                J.append((dict(backpack=bp, test=tn, combo="전부", com_dx=dx), dict(tc, backpack=bp, **w)))
                for k in WORST:
                    if k == "com_dx_m":
                        continue
                    ww = {kk: vv for kk, vv in w.items() if kk != k}
                    J.append((dict(backpack=bp, test=tn, combo=f"{k} 뺌", com_dx=dx), dict(tc, backpack=bp, **ww)))
    return J


def g_extra():
    """앞 결과를 보고 더한 것: 무게중심 앞 경계 촘촘히, 무게중심을 제어기가 알 때, 같은 설정 근처의 흩어짐"""
    J = []
    for bp in (False, True):
        for dx in (0.002, 0.004, 0.006, 0.008, -0.005):
            for tn, tc in TESTS.items():
                J.append((dict(backpack=bp, test=tn, axis="무게중심 앞뒤 (m)", val=dx), dict(tc, backpack=bp, com_dx_m=dx)))
        for dx in (0.01, 0.02, -0.02, -0.03):
            for tn, tc in TESTS.items():
                J.append((dict(backpack=bp, test=tn, axis="무게중심 (제어기가 앎)", val=dx), dict(tc, backpack=bp, com_dx_m=dx, com_known=True)))
        for k, v in (("friction", 0.78), ("friction", 0.79), ("friction", 0.81), ("friction", 0.82), ("mass_f", 0.99), ("mass_f", 1.01)):
            J.append((dict(backpack=bp, test="걷기 0.077 m/s", axis="흩어짐", val=f"{k}={v}"), dict(TESTS["걷기 0.077 m/s"], backpack=bp, **{k: v})))
    return J


GROUPS = dict(oat=g_oat, combo=g_combo, extra=g_extra)

if __name__ == "__main__":
    for g in sys.argv[1:] or list(GROUPS):
        t = time.time()
        jobs = GROUPS[g]()
        jobs.sort(key=lambda j: 0 if j[1].get("t_walk", 1.5) < 50 else 1)      # 긴 걷기부터
        out = run_jobs(jobs, procs=int(os.environ.get("RB_PROCS", "20")))
        rows = [dict(meta=m_, cfg={k: v for k, v in c.items() if k not in ("hip_roll", "servo_map")}, res=r) for m_, c, r in out]
        jdump(dict(base={k: v for k, v in ROB0.items()}, rows=rows), os.path.join(RES, f"{g}.json"))
        for r in rows:
            print(json.dumps(r["meta"], ensure_ascii=False), brief(r["res"]), flush=True)
        ne = sum(1 for r in rows if "error" in r["res"])
        print(f"{g}: {len(rows)} runs, {ne} errors, {time.time() - t:.0f}s", flush=True)
