"""v20 안 A 키트를 모든 관절 XL430-W250-T 로: 공용 설정(sp_common · check_v20A 의 모델 근사를 그대로 쓰고 서보 표만 덧붙임).

모델: check_v20A.C0 와 같음 — v18 MJCF + 고관절 롤 안 A(롤-피치 53.58 mm, 다리 늘림 40.13 mm) + 허벅지 · 정강이 각 +20 mm(질량 넣음)
      + 웅크림 35 mm. 이 근사의 질량은 3092 g 이라 v20 질량 파일(3168 g, code_v17/mass_build20_light.json)과 맞추려고
      몸통 무게중심 자리에 +76 g 을 더한다 [추정: 차이가 어디서 오는지 나누지 않음].
      sim/model/kitoru_v20.xml 은 발 패드가 없어 지금 걷기 제어기(패드 힘 센서)를 그대로 못 쓴다 → 이 근사를 쓴다.

서보 값(11.1 V, e-manual): XL430-W250 정지토크 1.4 N·m · 정지전류 1.3 A · 무부하 57 rpm(5.97 rad/s) · 기어 258.5:1 · 57.2 g,
      12 V: 1.5 N·m · 1.4 A · 61 rpm(6.39 rad/s) [e-manual]. kw_common 의 XL 값(무부하 5.45 rad/s, 인계 표)은 쓰지 않는다.
      kp: XL 6.4 N·m/rad(공장 P 이득 800 환산, kitoru_v18 · v20 주석) × 이득 2배(BASE kp_scale), 12 V 는 V/11.1 배 [추정].
      armature: 0.045 × (258.5/353.5)² = 0.0241 kg·m² [추정: 코어 모터라 실제는 더 클 수 있음].
      질량 차이: 질량 파일 기준(혼 · 케이블 포함) XM 90 → XL 65.2 g, XC 73 → XL 65.2 g [설계 파일]. 고관절 롤은 hr 질량이
      e-manual 기준(XM 82 g)이라 XL 57.2 g.
대책 서보:
  2xXL430e   무릎 한 관절에 XL 두 개(같은 축 양쪽): 토크 · kp · 감쇠 · 관성 두 배, 질량 +65.2 g(무릎 서보 자리) [추정: 자리 근사].
  XL430e_g2  XL 하나 + 2:1 감속(벨트 · 기어, 효율 85 % [추정]): 출력 토크 1.4 × 2 × 0.85 = 2.38 N·m, 무부하 속도 절반,
             kp = 6.4 × 2² × 0.85(서보는 모터 쪽 각으로 닫힘), 관성 × 2², 감속 부품 +15 g [추정].
  2xXL430e_g2 둘 다.  2xXL430e_g15 XL 두 개 + 1.5:1.  XL430e_g3 XL 하나 + 3:1 (같은 식, 조합 비교용).
발열(heat.py 와 같은 식, [추정]): 서보 한 개 전류 I = τ_서보/Kt (Kt = 1.4/1.3 = 1.077 N·m/A), 권선 R = 11.1/1.3 = 8.54 Ω,
      P = I_rms² R, 열저항 15–25 K/W, 실내 25 °C. τ_서보 = 관절 부하 토크 / (서보 개수 × 감속비 × 효율).
      XL 은 케이스가 플라스틱이라 열이 더 안 빠질 수 있다(같은 열저항을 씀) [추정]."""
import sys, os, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
SPEED = os.path.join(os.path.dirname(HERE), "speed")
for p in (SPEED, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import sp_common as S                   # kw_sim.Idx · build_model 바꿔치기(SEG) 가 여기서 걸린다
from sp_common import SEG
import kw_common as C
import kw_sim
from run_tests import BASE

RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)

XL_W0 = 57 * 2 * math.pi / 60           # 5.969 rad/s [e-manual 11.1 V]
XL_W12 = 61 * 2 * math.pi / 60          # 6.388 rad/s [e-manual 12 V]
ARM_XL = 0.045 * (258.5 / 353.5) ** 2   # 0.0241 [추정]
ETA = 0.85                              # 감속 효율 [추정]
G2_EXTRA = 15.0                         # 감속 부품 g [추정]
C.SERVO_SIM.update({
    "XL430e": (1.4, XL_W0, 6.4, ARM_XL),
    "XL430e12": (1.5, XL_W12, 6.4 * 12 / 11.1, ARM_XL),
    "2xXL430e": (2.8, XL_W0, 12.8, 2 * ARM_XL),
    "XL430e_g2": (1.4 * 2 * ETA, XL_W0 / 2, 6.4 * 4 * ETA, 4 * ARM_XL),
    "2xXL430e_g2": (2.8 * 2 * ETA, XL_W0 / 2, 12.8 * 4 * ETA, 8 * ARM_XL),
    "2xXL430e12": (3.0, XL_W12, 2 * 6.4 * 12 / 11.1, 2 * ARM_XL),
    "2xXL430e_g15": (2.8 * 1.5 * ETA, XL_W0 / 1.5, 12.8 * 2.25 * ETA, 2 * 2.25 * ARM_XL),
    "XL430e_g3": (1.4 * 3 * ETA, XL_W0 / 3, 6.4 * 9 * ETA, 9 * ARM_XL),
})
C.SERVO_G.update({"XL430e": 65.2, "XL430e12": 65.2, "2xXL430e": 130.4, "XL430e_g2": 65.2 + G2_EXTRA,
                  "2xXL430e_g2": 130.4 + G2_EXTRA, "2xXL430e12": 130.4,
                  "2xXL430e_g15": 130.4 + G2_EXTRA, "XL430e_g3": 65.2 + G2_EXTRA})
# 서보 종류 → (서보 개수, 감속비 × 효율, 한 개 Kt N·m/A, 한 개 정지전류 A, 전압 V, 온도 한계 °C)
HEAT = {"XL430e": (1, 1.0, 1.4 / 1.3, 1.3, 11.1, 72.0), "XL430e12": (1, 1.0, 1.5 / 1.4, 1.4, 12.0, 72.0),
        "2xXL430e": (2, 1.0, 1.4 / 1.3, 1.3, 11.1, 72.0), "XL430e_g2": (1, 2 * ETA, 1.4 / 1.3, 1.3, 11.1, 72.0),
        "2xXL430e_g2": (2, 2 * ETA, 1.4 / 1.3, 1.3, 11.1, 72.0), "2xXL430e12": (2, 1.0, 1.5 / 1.4, 1.4, 12.0, 72.0),
        "2xXL430e_g15": (2, 1.5 * ETA, 1.4 / 1.3, 1.3, 11.1, 72.0), "XL430e_g3": (1, 3 * ETA, 1.4 / 1.3, 1.3, 11.1, 72.0),
        "XM430-W350": (1, 1.0, 3.8 / 2.1, 2.1, 11.1, 80.0), "XC430-W240": (1, 1.0, 1.7 / 1.3, 1.3, 11.1, 80.0),
        "XL430-W250": (1, 1.0, 1.4 / 1.3, 1.3, 11.1, 72.0)}
RTH = (15.0, 25.0); T_AMB = 25.0; T_OK = 65.0

LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
# 지금 v20A 서보(check_v20A): 롤 · 피치 · 무릎 · 발목 롤 XM430-W350, 발목 피치 XC430-W240, 요 · 팔 · 머리 XL430
V20A = dict(hip_yaw="XL430-W250", hip_roll="XM430-W350", hip_pitch="XM430-W350", knee="XM430-W350",
            ankle_pitch="XC430-W240", ankle_roll="XM430-W350", arm_swing="XL430-W250")
ALLXL = {k: "XL430e" for k in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll", "arm_swing", "nod_crank")}
HR_MASS = {"XM430-W350": 0.082}         # hr 질량(e-manual 기준 82 g): XL 계열은 SERVO_G 차이만큼


def hr_of(servo):
    m_ = 0.082 + (C.SERVO_G.get(servo, 90.0) - 90.0) / 1000
    return dict(servo=servo, d_axis=0.05358, lengthen=0.04013, mass=m_, mass_pos=(0.0, 0.0, -0.02655))


def _torso_com():
    import mujoco
    sp = mujoco.MjSpec.from_file(C.XML)
    return tuple(float(x) for x in sp.body("torso").ipos)


TORSO_COM = _torso_com()
MASS_FIX = ("torso", 0.076, TORSO_COM)          # 3092 → 3168 g (v20 질량 파일) [추정 자리]
BACKPACK = None


def backpack():
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "battery6000")); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hiproll"))
    from sim_runs import pack, BACK
    return tuple(pack(BACK, 0.250))


def cfg(servos=None, seg=0.02, crouch=0.035, mass_f=1.0, pack_=False, **kw):
    """한 구성의 kw_sim.run 설정. servos: 관절 끝 이름 → 서보(빠진 관절은 ALLXL). seg: 허벅지 · 정강이 각 늘림 m."""
    sv = dict(ALLXL); sv.update(servos or {})
    hr_s = sv.pop("hip_roll")
    em = (MASS_FIX,) + (backpack() if pack_ else ())
    c = dict(hip_roll=hr_of(hr_s), servo_map=sv, seg=(seg, seg, True), crouch=crouch, extra_mass=em)
    if mass_f != 1.0:
        c["mass_scale"] = {"": mass_f}      # 모든 body 질량 · 관성 배수(서보 포함, 무게중심 그대로) — 제어기도 앎
    c.update(kw)
    return c


def heat_of(servo, tau_rms):
    if servo not in HEAT:
        return None
    n, gr, Kt, Ist, V, Tlim = HEAT[servo]
    I = tau_rms / (n * gr) / Kt; P = I * I * (V / Ist)
    return dict(I_A=I, P_W=P, T_lo=T_AMB + P * RTH[0], T_hi=T_AMB + P * RTH[1], T_limit=Tlim)


def one(job):
    """kw_sim.run + 관절별 지표(오른쪽 다리): 포화 %, u99, 걷기 구간 부하 RMS(서보 몫 빼기 전 관절 토크), 발열."""
    meta, c0 = job
    c = dict(kw_sim.DEFAULT); c.update(BASE); c.update(c0)
    seg = c.pop("seg", (0.0, 0.0)); SEG["thigh"], SEG["shank"] = seg[:2]; SEG["mass"] = len(seg) > 2 and seg[2]
    SEG["servo"] = c.get("servo_map")
    try:
        o, R, m = kw_sim.run(**c)
        ix = S.Idx2(m)
        o["split"] = S.split_u(m, R, ix)
        o["mass_kg"] = float(m.body_subtreemass[1])
        tau = np.asarray(R["tau"]); w = np.asarray(R["w_act"]); sat = np.asarray(R["sat"])
        tt = np.arange(len(tau)) * 5e-3
        t0 = (c["t_walk"] + c["Tinit"]) if c["t_walk"] < 50 else 1.5
        sel = tt >= t0
        b = np.array([m.dof_damping[ix.vadr[j]] for j in ix.act_j])
        load = tau - b * w
        sv = dict(c["servo_map"]); sv["hip_roll"] = c["hip_roll"]["servo"]
        J = {}
        for i, nm in enumerate(ix.act):
            if not nm.startswith("R_") or nm.endswith("arm_swing") or sel.sum() < 10:
                continue
            k = nm[2:]; rms = float(np.sqrt(np.mean(load[sel, i] ** 2)))
            J[k] = dict(servo=sv.get(k), stall=float(ix.stall[i]), w0=float(ix.w0[i]),
                        sat_pct=float(100 * sat[sel, i].mean()), load_rms=rms, load_mean_abs=float(np.abs(load[sel, i]).mean()),
                        u_p99=float(np.percentile(np.abs(load[sel, i]) / ix.stall[i] + np.abs(w[sel, i]) / ix.w0[i], 99)),
                        damping=float(b[i]), armature=float(m.dof_armature[ix.vadr[ix.act_j[i]]]),
                        kp=float(m.actuator_gainprm[i, 0]), heat=heat_of(sv.get(k), rms))
        o["joints"] = J
    except Exception as e:
        import traceback
        return meta, c0, {"error": repr(e) + traceback.format_exc()[-400:]}
    finally:
        SEG["thigh"], SEG["shank"], SEG["mass"] = 0.0, 0.0, False
    for k in ("tau", "u_max", "u_p99", "force_over_stall_max", "force_over_stall_p99"):
        o.pop(k, None)
    return meta, c0, o


def run_jobs(jobs, procs=76):
    from multiprocessing import Pool
    with Pool(min(procs, len(jobs))) as p:
        return p.map(one, jobs, chunksize=1)


# 시험 묶음: 서기 12 s, 0.05 · 0.077 m/s(한 걸음 0.65 s, 보폭 32.5 · 50 mm), 최고 속도 격자 — 마찰 0.7 · 0.8 · 0.9 세 번씩(SPEED_STUDY 규칙)
CAD = [(0.45, 0.20), (0.38, 0.15), (0.32, 0.12), (0.28, 0.10)]
STR = [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09]
FRIC = (0.7, 0.8, 0.9)


def jobs_for(name, c, speed_grid=True):
    J = [(dict(cfg=name, test="stand", fr=0.8), dict(c, t_walk=99, T_end=12.0))]
    J += [(dict(cfg=name, test="walk", Ts=0.45, Td=0.2, stride=0.0325, fr=f), dict(c, Ts=0.45, Td=0.2, stride=0.0325, friction=f)) for f in FRIC]
    cads = CAD if speed_grid else [(0.45, 0.20)]
    strs = STR if speed_grid else [0.05]
    J += [(dict(cfg=name, test="walk", Ts=a, Td=b, stride=s, fr=f), dict(c, Ts=a, Td=b, stride=s, friction=f))
          for a, b in cads for s in strs for f in FRIC]
    return J
