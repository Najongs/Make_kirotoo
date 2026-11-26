"""질문 1 역산(해석): 무릎 서보의 토크-속도 직선 τ = τs (1 − ω/ω0) 로 낼 수 있는 무게중심 최고 위로 속도와 체공 시간의 상한.
가정: 허벅지 = 정강이 = l, 몸통 곧게(고관절 −a, 무릎 2a, 발목 −a), 무릎 모멘트 팔 = l sin a, 엉덩이 높이 변화 속도 v = l sin a · ω무릎.
가속 없이(등속) 버틸 수 있는 최고 속도 = 상한: v = l sin a · ω0 (1 − F l sin a / τs), F = m g / (다리 수).
회전자 관성 · 고관절 · 발목 · 발끝 밀기는 넣지 않음(이 셋은 상한을 더 낮추거나(관성) 조금 올림(발끝)).
달리기에 필요한 값: 체공 t 동안 뜨려면 이륙 위로 속도 v = g t / 2 → 30 ms 0.147 m/s, 50 ms 0.245 m/s, 한 다리로.
usage: ../../.venv/bin/python run_calc.py → results/run_calc.json"""
import math, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.81; M = 3.05
SERVO = {"XM430-W350": (3.8, 43), "XM430-W210": (2.7, 70), "XH430-W210": (2.2, 46), "XC430-W150": (1.4, 99)}   # e-manual 11.1 V


def vmax(ts, rpm, l, legs, m=M):
    w0 = rpm * 2 * math.pi / 60; F = m * G / legs; best = (0, 0)
    for a in np.radians(np.arange(5, 71, 1)):
        r = l * math.sin(a); v = r * w0 * (1 - F * r / ts)
        if v > best[0]:
            best = (v, math.degrees(2 * a))
    return best


if __name__ == "__main__":
    out = {}
    for sv, (ts, rpm) in SERVO.items():
        for l in (0.075, 0.085, 0.095, 0.105):
            for legs in (1, 2):
                v, k = vmax(ts, rpm, l, legs)
                out[f"{sv}|{l*1e3:.0f}|{legs}"] = dict(servo=sv, l_mm=l * 1e3, legs=legs, v_max=v, knee_deg_at_best=k, flight_ms=2 * v / G * 1e3)
                print(f"{sv:11s} l={l*1e3:.0f} mm legs={legs}: v≤{v:.3f} m/s (무릎 {k:.0f}°) → 체공 ≤ {2*v/G*1e3:.0f} ms")
    json.dump(out, open(os.path.join(HERE, "results", "run_calc.json"), "w"), indent=1, ensure_ascii=False)
