"""서보 발열 어림(MuJoCo 밖, 전부 [추정] 모델): 걷기 중 관절 부하 토크 → 권선 구리 손실 → 정상 상태 온도 상승.

- 부하 토크 τ = 액추에이터 힘 − 감쇠(역기전력) b·ω [시뮬 실측, rb_common.extra_metrics load_rms · load_mean_abs].
- 전류 I = τ / Kt, Kt = 정지토크/정지전류 (XM430-W350 1.810, XC430-W240 1.308, XL430-W250 1.077 N·m/A, 11.1 V) [e-manual].
- 권선 저항 R = 11.1 V / 정지전류 (정지 때 전압이 모두 권선에 걸린다고 봄) [추정] → 구리 손실 P = I_rms² R.
  대기 전력(약 0.5 W)과 철손 · 기어 마찰 열은 뺐다 → 실제보다 작게 나오는 쪽.
- 열저항 15–25 K/W (케이스 겉넓이 약 7000 mm², 자연 대류 10 W/m²K ≈ 15 K/W + 권선 → 케이스) [추정].
- 정지 온도 한계: XM · XC 80 °C, XL 72 °C [e-manual 동작 온도 상한 · Temperature Limit 기본값]. 실내 25 °C 에서 출발.
- e-manual 에는 연속 토크 값이 없다(「정지토크는 연속 출력과 다르다」는 주의만). 그래서 「연속 토크」 대신 위 온도 상승으로 판정.
python robust/heat.py → robust/results/heat.json"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
EM = {3.8: ("XM430-W350", 1.810, 2.1, 80.0), 1.75: ("XC430-W240", 1.308, 1.3, 80.0), 1.4: ("XL430-W250", 1.077, 1.3, 72.0)}
RTH = (15.0, 25.0)
T_AMB = 25.0


def servo_heat(res):
    out = {}
    for nm, st in res["stall"].items():
        k = min(EM, key=lambda s_: abs(s_ - st))          # 전압을 바꾼 시험은 정지토크가 몇 % 작다 → 가장 가까운 종류
        if abs(k - st) > 0.15 * k or nm[:2] not in ("R_", "L_") or nm.endswith("arm_swing"):
            continue
        typ, Kt, Ist, Tlim = EM[k]
        R = 11.1 / Ist
        I = res["load_rms"][nm] / Kt
        P = I * I * R
        out[nm] = dict(servo=typ, tau_rms=res["load_rms"][nm], tau_mean=res["load_mean_abs"][nm], rms_over_stall=res["load_rms"][nm] / st,
                       I_rms_A=I, P_W=P, T_lo=T_AMB + P * RTH[0], T_hi=T_AMB + P * RTH[1], T_limit=Tlim)
    return out


if __name__ == "__main__":
    rows = {}
    oat = json.load(open(os.path.join(HERE, "results", "oat.json")))["rows"]
    combo = json.load(open(os.path.join(HERE, "results", "combo.json")))["rows"] if os.path.exists(os.path.join(HERE, "results", "combo.json")) else []
    for r in oat:
        mt = r["meta"]
        if mt["axis"] == "기준" and "load_rms" in r["res"]:
            rows[f"{mt['test']} · 배낭 {'있음' if mt['backpack'] else '없음'}"] = servo_heat(r["res"])
        if mt["axis"] == "질량 배수" and mt["val"] == 1.15 and mt["test"].startswith("걷기") and "load_rms" in r["res"] and not r["res"]["fallen"]:
            rows[f"{mt['test']} · 질량 +15 % · 배낭 {'있음' if mt['backpack'] else '없음'}"] = servo_heat(r["res"])
    for r in combo:
        mt = r["meta"]
        if mt["combo"] == "전부" and mt["test"].startswith("걷기") and "load_rms" in r["res"] and not r["res"]["fallen"]:
            rows[f"나쁜 조합 걷기 (무게중심 {mt['com_dx'] * 1000:+.0f} mm) · 배낭 {'있음' if mt['backpack'] else '없음'}"] = servo_heat(r["res"])
    json.dump(dict(note=__doc__, rows=rows), open(os.path.join(HERE, "results", "heat.json"), "w"), indent=1, ensure_ascii=False)
    for k, v in rows.items():
        top = sorted(v.items(), key=lambda kv: -kv[1]["T_hi"])[:4]
        print(k, [(n, round(x["tau_rms"], 2), round(x["P_W"], 2), round(x["T_lo"]), round(x["T_hi"])) for n, x in top])
