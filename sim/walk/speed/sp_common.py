"""속도 · 달리기 · 다리 길이 조사 공용(sim/walk 제어기 그대로 재사용, kw_common 은 고치지 않고 이 모듈에서 덧붙임).

- 서보 추가: XM430-W210, XC430-W150 [e-manual 11.1 V: 2.7 N·m · 70 rpm / 1.4 N·m · 99 rpm].
  kp 는 같은 P 이득 레지스터라고 보고 정지토크에 비례(XM350 17.5 × τs/3.8, XL 6.4 와 같은 환산) [추정],
  armature 는 0.045 × (기어비/353.5)² [추정, 기존 study 와 같은 근사].
- 사용률 무부하속도: kw_common.Idx 는 forcerange 로 서보를 알아내 XC150(1.4) 과 XL(1.4) 을 못 가린다 →
  관절 감쇠 b = τs/ω0 에서 ω0 = τs/b 로 다시 계산(kw_sim.Idx 만 바꿔치기).
- 다리 늘림: 안 A(직렬) — 롤 축은 요 판 아래 그대로(요 축 아래 14.2 mm), 피치 축을 L 만큼 내림.
  d_axis = 13.8 mm + L (L = 40.5 mm 에서 hr_cfg.hr("A") 와 같음). 롤 서보 질량 · 자리는 hr_cfg 그대로.
- 허벅지 · 정강이 늘림(비교용): 무릎 · 발목 body 를 내림. 질량 · 관성은 그대로 [추정: 실제로는 몇 g 늘어남].
  kw_sim.run 의 계획용 모델에도 같은 변형이 들어가도록 kw_sim.build_model 을 감싼다(전역 SEG)."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hiproll"))
import numpy as np
import kw_common as C
import kw_sim

C.SERVO_SIM.update({"XM430-W210": (2.7, 70 * 2 * math.pi / 60, 17.5 * 2.7 / 3.8, 0.045 * (212.6 / 353.5) ** 2),
                    "XC430-W150": (1.4, 99 * 2 * math.pi / 60, 17.5 * 1.4 / 3.8, 0.045 * (159.59 / 353.5) ** 2)})
C.SERVO_G.update({"XM430-W210": 90.0, "XC430-W150": 73.0})
# e-manual 11.1 V 값(비교표용, 시뮬 기존 서보는 kw_common 값 그대로: XC240 1.75 · 6.81, XL 1.4 · 5.45)
EMAN = {"XM430-W350": (3.8, 43, 82, 353.5), "XM430-W210": (2.7, 70, 82, 212.6), "XC430-W240": (1.7, 65, 65, 245.22),
        "XC430-W150": (1.4, 99, 65, 159.59), "XL430-W250": (1.4, 57, 57.2, 258.5), "2XL430-W250": (1.4, 57, 98.2, 257.4),
        "2XC430-W250": (1.6, 59, 102, 257.4), "XH430-W210": (2.2, 46, 82, 212.6)}


class Idx2(C.Idx):
    def __init__(self, m):
        super().__init__(m)
        b = np.array([m.dof_damping[self.vadr[j]] for j in self.act_j])
        self.w0 = np.where(b > 1e-6, self.stall / np.maximum(b, 1e-9), 5.0)


kw_sim.Idx = Idx2

# 지금 서보 배치: 롤 · 피치 · 무릎 · 발목 롤 XM430-W350, 발목 피치 XC430-W240, 요 · 팔 XL430(머리는 모델이 이미 XL)
CUR = {"hip_yaw": "XL430-W250", "arm_swing": "XL430-W250"}
ROLL_M = 0.082        # XM430 질량 [e-manual]; hr_cfg 와 같음


def hrA(L=0.0405, servo="XM430-W350"):
    m_ = {"XM430-W350": 0.082, "XM430-W210": 0.082, "XC430-W240": 0.065, "XL430-W250": 0.0572}[servo]
    return dict(servo=servo, d_axis=0.0138 + L, lengthen=L, mass=m_, mass_pos=(0.0, 0.0, -0.026))


SEG = {"thigh": 0.0, "shank": 0.0, "mass": False}
# 링크 늘림 질량 [추정]: 출력 뼈대 선밀도 = (링크 질량 − 서보 질량) / 75 mm — 허벅지 (129 − 90) / 75 ≈ 0.52 g/mm,
# 정강이 (120 − 73) / 75 ≈ 0.63 g/mm. 무릎 서보(허벅지 안, 무릎 축 위 12 mm) · 발목 피치 서보(정강이 안)는 축과 함께 내려간다.
SEG_GMM = {"thigh": 0.52, "shank": 0.63}
_bm = C.build_model


def _seg(spec):
    dt, ds = SEG["thigh"], SEG["shank"]
    for s in "RL":
        if SEG["mass"]:
            sg = 1 if s == "R" else -1
            for bn, L, key in ((f"{s}_thigh", dt, "knee"), (f"{s}_shank", ds, "ankle_pitch")):
                if not L:
                    continue
                b = spec.body(bn); _, p, _ = C.SERVO_AT[key]; p = np.array(p) * [1, sg, 1]
                g = C.SERVO_G[(SEG.get("servo") or {}).get(key, "XM430-W350" if key == "knee" else "XC430-W240")] / 1000
                C.add_point_mass(b, -g, p); C.add_point_mass(b, g, p - [0, 0, L])          # 서보를 축과 함께 내림
                C.add_point_mass(b, SEG_GMM["thigh" if key == "knee" else "shank"] * L, [0, 0, -(0.075 + L) / 2])   # 늘린 뼈대
        if dt:
            b = spec.body(f"{s}_shank"); b.pos = np.array(b.pos) - [0, 0, dt]
        if ds:
            b = spec.body(f"{s}_ankle_link"); b.pos = np.array(b.pos) - [0, 0, ds]
    if dt or ds:
        spec.body("torso").pos = np.array(spec.body("torso").pos) + [0, 0, dt + ds]
        for k in list(spec.keys):
            spec.delete(k)


def build_model2(*a, **kw):
    if SEG["thigh"] or SEG["shank"]:
        prev = kw.get("extra_bodies")
        kw["extra_bodies"] = (lambda sp: (_seg(sp), prev and prev(sp)))
    return _bm(*a, **kw)


kw_sim.build_model = build_model2


def split_u(m, R, ix, t0=1.0):
    """u 를 토크 몫(|부하 토크|/τs)과 속도 몫(|ω|/ω0)으로 나눔 + 포화 순간의 평균 속도 몫. R: kw_sim 기록(5 ms 간격)."""
    tau = np.asarray(R["tau"]); w = np.asarray(R["w_act"]); sat = np.asarray(R["sat"])
    if len(tau) < 10:
        return {}
    st = np.arange(len(tau)) * 5e-3 >= t0
    b = np.array([m.dof_damping[ix.vadr[j]] for j in ix.act_j])
    tq = np.abs(tau - b * w) / ix.stall; sp = np.abs(w) / ix.w0
    out = {}
    for i, nm in enumerate(ix.act):
        if not nm.startswith("R_") or nm.endswith("arm_swing"):
            continue
        s_ = sat[st, i]
        out[nm[2:]] = dict(torque_p99=float(np.percentile(tq[st, i], 99)), speed_p99=float(np.percentile(sp[st, i], 99)),
                           speed_max=float(sp[st, i].max()), sat_frac=float(s_.mean()),
                           speed_at_sat=float(sp[st, i][s_].mean()) if s_.any() else None,
                           u_p99=float(np.percentile(tq[st, i] + sp[st, i], 99)),
                           torque_rms=float(np.sqrt(np.mean(tq[st, i] ** 2))))
    return out


def one(job):
    """run_tests.one 과 같되 SEG 설정과 u 나누기 결과를 붙인다."""
    meta, cfg = job
    from run_tests import BASE
    c = dict(BASE); c.update(cfg)
    seg = c.pop("seg", (0.0, 0.0)); SEG["thigh"], SEG["shank"] = seg[:2]; SEG["mass"] = len(seg) > 2 and seg[2]
    SEG["servo"] = c.get("servo_map")
    try:
        o, R, m = kw_sim.run(**c)
        o["split"] = split_u(m, R, Idx2(m))
        o["mass_kg"] = float(m.body_subtreemass[1])
        import mujoco
        d = mujoco.MjData(m); d.qpos[:] = m.qpos0; mujoco.mj_forward(m, d)
        o["com_z0"] = float(d.subtree_com[1][2])
    except Exception as e:
        import traceback
        return meta, c, {"error": repr(e) + traceback.format_exc()[-300:]}
    finally:
        SEG["thigh"], SEG["shank"], SEG["mass"] = 0.0, 0.0, False
    c["seg"] = seg
    return meta, c, o


def run_jobs(jobs, procs=76):
    from multiprocessing import Pool
    with Pool(min(procs, len(jobs))) as p:
        return p.map(one, jobs, chunksize=1)
