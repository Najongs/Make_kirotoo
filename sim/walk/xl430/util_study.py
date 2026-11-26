"""모든 다리 관절 XL430-W250 일 때 사용률 u (설계 파일 방식: code_v17/servo_study.py · servo_opt.py 를 읽기만 하고 그대로 씀).

질량: code_v17/mass_build19_light.json (v19) 에서 서보 질량만 바꿈(XM 90 → XL 65.2 g, XC 73 → XL 65.2 g,
       질량 파일의 서보 질량 = 혼 · 케이블 포함 [설계 파일]). 링크 질량 · 무게중심 · 관성을 점 질량으로 고침.
경우: servo_study CASES walk · fast · jog(몸 기울임 진폭은 tune_psi 로 다시 맞춤), static_cases crouch · toe · heel,
       servo_opt.hip_strategy(상체 흔들기). u = |τ|/τs + |ω|/ω0, 포장선(envelope) = jog 를 뺀 전부(servo_opt 와 같음).
서보 값: e-manual 11.1 V — XL430 1.4 N·m · 57 rpm(5.97 rad/s), XC430 1.7 · 65 rpm, XM430 3.8 · 43 rpm.
회전자 관성(armature) 몫: τ += a · dω/dt, a = 0.045 × (기어비/353.5)² [추정: XM 값을 기어비로 환산; XL 은 코어 모터라 더 클 수 있음].
발열: 걷기(walk) 토크 RMS / 정지토크.

usage: cd sim/walk && uv run python xl430/util_study.py → xl430/results/util.json, util.md
"""
import sys, os, json, copy
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "..", "..", "code_v17")
sys.path.insert(0, CODE)
import servo_study as S
import servo_opt as O
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)

EM = {"XM430-W350": (3.8, 43 * 2 * np.pi / 60, 353.5), "XC430-W240": (1.7, 65 * 2 * np.pi / 60, 245.22),
      "XL430-W250": (1.4, 57 * 2 * np.pi / 60, 258.5), "2xXL430": (2.8, 57 * 2 * np.pi / 60, 258.5)}
ARM = {k: 0.045 * (v[2] / 353.5) ** 2 * (2 if k.startswith("2x") else 1) for k, v in EM.items()}
SG = {"XM430-W350": 90.0, "XC430-W240": 73.0, "XL430-W250": 65.2, "2xXL430": 130.4}
V19 = {"hip_yaw": "XC430-W240", "hip_pitch": "XM430-W350", "knee": "XM430-W350", "ankle_pitch": "XC430-W240", "ankle_roll": "XM430-W350"}
PART = {"hip_yaw": "yaw_servo_{s}_XC430", "hip_pitch": "{s}_hip_pitch_servo_XM430", "knee": "{s}_knee_servo_XM430",
        "ankle_pitch": "{s}_ankle_pitch_servo_XC430", "ankle_roll": "{s}_ankle_roll_servo_XM430"}


def point_mass(L, dm, p):
    m0 = L["mass_g"]; c0 = np.array(L["com_world"]); I0 = np.array(L["inertia_gmm2"]); p = np.array(p, float)
    m1 = m0 + dm; c1 = (m0 * c0 + dm * p) / m1
    def pa(m, r):
        return m * (r @ r * np.eye(3) - np.outer(r, r))
    I1 = I0 + pa(m0, c0 - c1) + pa(dm, p - c1)
    L.update(mass_g=m1, com_world=c1.tolist(), inertia_gmm2=I1.tolist())


def mass_for(assign, arm="XC430-W240", scale_body=1.0):
    mass = json.load(open(os.path.join(CODE, "mass_build19_light.json")))
    mass = copy.deepcopy(mass)
    parts = {p["part"]: p for p in mass["parts"]}
    for j, t in assign.items():
        for s in "RL":
            p = parts[PART[j].format(s=s)]
            point_mass(mass["links"][p["link"]], SG[t] - SG[V19[j]], p["com"])
    for s in "RL":
        p = parts[f"arm_{s}_servo_XC430"]; point_mass(mass["links"][p["link"]], SG[arm] - SG["XC430-W240"], p["com"])
    if scale_body != 1.0:
        # 서보를 뺀 질량만 배수(몸 · 머리 · 팔 · 다리 출력물 · 전자부품): 링크마다 서보 질량은 그대로 둠
        sv = {}
        for p in mass["parts"]:
            if p["kind"] == "servo":
                sv[p["link"]] = sv.get(p["link"], 0.0) + p["mass"]
        for ln, L in mass["links"].items():
            ns = L["mass_g"] - sv.get(ln, 0.0)
            f = (ns * scale_body + sv.get(ln, 0.0)) / L["mass_g"]
            L["mass_g"] *= f; L["inertia_gmm2"] = (np.array(L["inertia_gmm2"]) * f).tolist()
    return mass


def evaluate(assign, arm="XC430-W240", scale_body=1.0, armature=True, cases=None):
    mass = mass_for(assign, arm, scale_body)
    M = S.Model(mass)
    C = copy.deepcopy(S.CASES) if cases is None else cases
    psi = {}
    for nm, c in C.items():
        deg, err = S.tune_psi(M, c); C[nm]["psi_amp"] = np.radians(deg); psi[nm] = (deg, err)
    saved = S.CASES; S.CASES = C
    try:
        res, rms, per = O.collect(M)
    finally:
        S.CASES = saved
    dts = {nm: c["T"] / 200 for nm, c in C.items()}; dts["hip_strategy"] = 0.5 / 120
    out = {"total_g": M.total(), "psi_deg": psi, "u": {}, "u_jog": {}, "u_arm": {}, "u_arm_jog": {}, "heat": {}, "heat_arm": {},
           "tau_max": {}, "w_max": {}, "worst_case": {}}
    for j in S.LEGJ:
        ts, w0, _ = EM[assign[j]]; a = ARM[assign[j]]
        best = {}
        for t, w, nm in res[j]:
            t = np.asarray(t, float); w = np.asarray(w, float)
            alpha = np.gradient(w, dts[nm]) if len(w) > 3 else np.zeros_like(w)
            ta = t + a * alpha * np.sign(t + 1e-12) * np.sign(t + a * alpha + 1e-12) if False else t + a * alpha
            u0 = float(np.max(np.abs(t) / ts + np.abs(w) / w0)); u1 = float(np.max(np.abs(ta) / ts + np.abs(w) / w0))
            key = "jog" if nm == "jog" else "env"
            if u0 > best.get((key, 0), (-1, ""))[0]:
                best[(key, 0)] = (u0, nm)
            best[(key, 1)] = max(best.get((key, 1), (-1, "")), (u1, nm))
            out["tau_max"][j] = max(out["tau_max"].get(j, 0), float(np.max(np.abs(t))))
            out["w_max"][j] = max(out["w_max"].get(j, 0), float(np.max(np.abs(w))))
        out["u"][j], out["worst_case"][j] = best[("env", 0)]
        out["u_jog"][j] = best[("jog", 0)][0]
        out["u_arm"][j] = best[("env", 1)][0]; out["u_arm_jog"][j] = best[("jog", 1)][0]
        out["heat"][j] = rms[j] / ts
        # 회전자 관성 몫을 넣은 걷기 RMS
        tw = [(np.asarray(t), np.asarray(w)) for t, w, nm in res[j] if nm == "walk"]
        if tw:
            sq = [np.mean((t + a * np.gradient(w, dts["walk"])) ** 2) for t, w in tw]
            out["heat_arm"][j] = float(np.sqrt(np.mean(sq))) / ts
    out["per_case"] = per
    return out


CONFIGS = {
    "v19 지금(XM 3 + XC 2)": (dict(V19), "XC430-W240", 1.0),
    "모두 XL430": ({j: "XL430-W250" for j in S.LEGJ}, "XL430-W250", 1.0),
    "모두 XL430, 서보 뺀 몸 −10 %": ({j: "XL430-W250" for j in S.LEGJ}, "XL430-W250", 0.9),
    "모두 XL430, 서보 뺀 몸 −20 %": ({j: "XL430-W250" for j in S.LEGJ}, "XL430-W250", 0.8),
    "모두 XL430, 서보 뺀 몸 −30 %": ({j: "XL430-W250" for j in S.LEGJ}, "XL430-W250", 0.7),
    "XL + 무릎 · 발목 피치 XC": (dict({j: "XL430-W250" for j in S.LEGJ}, knee="XC430-W240", ankle_pitch="XC430-W240"), "XL430-W250", 1.0),
    "XL + 무릎 · 발목 피치 · 발목 롤 XC": (dict({j: "XL430-W250" for j in S.LEGJ}, knee="XC430-W240", ankle_pitch="XC430-W240", ankle_roll="XC430-W240"), "XL430-W250", 1.0),
    "XL + 무릎 XM": (dict({j: "XL430-W250" for j in S.LEGJ}, knee="XM430-W350"), "XL430-W250", 1.0),
    "XL + 무릎 · 발목 롤 XM": (dict({j: "XL430-W250" for j in S.LEGJ}, knee="XM430-W350", ankle_roll="XM430-W350"), "XL430-W250", 1.0),
    "XL, 무릎 · 발목 피치 · 발목 롤 XL 두 개": (dict({j: "XL430-W250" for j in S.LEGJ}, knee="2xXL430", ankle_pitch="2xXL430", ankle_roll="2xXL430"), "XL430-W250", 1.0),
}

if __name__ == "__main__":
    out = {}
    for nm, (asg, arm, sc) in CONFIGS.items():
        r = evaluate(asg, arm, sc); r["assign"] = asg; out[nm] = r
        print(nm, round(r["total_g"]), "u", {j[:6]: round(v, 2) for j, v in r["u"].items()}, "u+arm", {j[:6]: round(v, 2) for j, v in r["u_arm"].items()},
              "jog", {j[:6]: round(v, 2) for j, v in r["u_jog"].items()}, "heat", {j[:6]: round(v, 2) for j, v in r["heat"].items()}, flush=True)
    json.dump(out, open(os.path.join(RES, "util.json"), "w"), indent=1, ensure_ascii=False, default=float)
