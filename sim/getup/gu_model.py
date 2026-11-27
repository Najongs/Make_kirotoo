"""일어나기 조사 공용: 팔 모양 후보를 MjSpec 로 바꿔 끼운 모델 + 키프레임 위치 제어 시뮬.

모델 = sim/walk 걷기 모델(v18 형상) + 고관절 롤 안 A(hr_cfg 와 같은 hrA(40.5 mm)) + 허벅지 · 정강이 각 +20 mm(질량 포함, sp_common.SEG)
       + 서보 지금 배치(sp_common.CUR: 요 · 팔 XL430-W250) + 팔 바꿔 끼우기(arm_mod).
sp_common · getup 은 import 만 한다(고치지 않음).

팔 모델(모두 [추정], 원래 팔 메시는 지우고 캡슐 · 공으로 다시 만듦):
- 닿는 길이 R = 어깨 축에서 손 바깥면까지. 지금 팔 131 mm.
- 질량: 어깨 서보 XL430 65.2 g(kw_common.SERVO_G, 혼 · 케이블 포함) + 막대 ARM_GMM g/mm × 길이 + 손(플랩 포함) HAND_G g.
  지금 팔 132.5 g(모델, XC430 73 g 포함)에서 서보를 빼면 59.5 g → 손 15 g, 막대 44.5 g / 131 mm = 0.34 g/mm.
- 팔꿈치: XL430 하나 65.2 g 을 팔꿈치 축에(위팔 끝), 감쇠 · armature · kp 는 kw_common.SERVO_SIM 의 XL430 값. 범위 ±130° [가정].
- 꺾은 팔: 위 L1 은 곧게, 아래 L2 는 앞으로 beta 만큼 꺾어 고정.
- 팔 geom 은 contype 1 / conaffinity 2(바닥과만 닿음, 몸과는 안 닿음 — 원래 모델과 같은 규칙).
- 용수철 팔꿈치(kind spring, 2026-11-27): 서보 없는 경첩 + 비틀림 용수철. 범위 [−130°, 0°](− = 아래팔을 앞으로 접음, 0° 곧은 자리에 멈춤턱),
  용수철 토크 = k (springref − q), springref = SPRING_PRE(미리 감김) → 펴는 쪽(+)으로 밂. k 는 90° 굽혔을 때 토크 T90 이 되게. 경첩 · 용수철 ELBOW_SPRING_G g [추정].
- 머리 v21(HEAD21=1, 2026-11-27): 크랭크 고리 다항식을 head21.json(v21 링크 표)으로 다시 맞추고, head_nod 범위 = 껍데기가 덮개에 닿는 각
  −25.5 / +32.5°(멈춤턱), 크랭크 XL 위치 명령은 머리 −20 ~ +28° 안. 키의 head 값은 머리 각(rad, + 숙임) → 크랭크 각으로 바꿔 명령."""
import sys, os, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SIM, "walk", "speed"))
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")
import numpy as np
import mujoco
from sp_common import CUR, hrA, SEG, build_model2, Idx2, EMAN
from kw_common import SERVO_SIM, SERVO_G, scale_kp, jdump, korean_font
from getup import widen

RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
LEG_EXT = 0.020                  # 허벅지 · 정강이 각 +20 mm
HAND_G = 15.0                    # 손 + 플랩 g [추정]
ARM_GMM = (132.49 - 73.0 - HAND_G) / 131.0     # 막대 선밀도 g/mm [추정] ≈ 0.34
SERVO_XL_G = SERVO_G["XL430-W250"]             # 65.2 g
ROD_R, HAND_R, Y_ARM = 0.016, 0.018, 0.051     # 막대 반지름(원래 팔 x 폭 ±20 mm), 손 공 반지름, 팔 가운데 y(팔 body 좌표)
ELBOW_RANGE = 2.27                             # ±130° [가정]
ELBOW_SPRING_G = 10.0                          # 용수철 팔꿈치 경첩 · 용수철 g [추정]
SPRING_PRE = math.radians(90)                  # 용수철 미리 감김(곧은 자리에서 이미 90° 감김 → 곧을 때 토크 = T90 / 2) [가정]
SPRING_DAMP = 0.01                             # 경첩 마찰 흉내 N·m·s/rad [추정]
H21 = json.load(open(os.path.join(os.path.dirname(SIM), "code_v17", "out_head21", "head21.json")))
H21_ROWS = [r for r in H21["rows"] if r["ok"] and r.get("psi") is not None and -27.0 <= r["beta"] <= 34.0]
H21_B = np.radians([r["beta"] for r in H21_ROWS]); H21_A = np.radians([r["alpha"] for r in H21_ROWS]); H21_E = np.radians([r["psi"] for r in H21_ROWS])
HEAD_STOP = (math.radians(-25.5), math.radians(32.5))    # 머리 껍데기 ↔ 몸통 덮개 닿는 각(HEAD_CRANK_v21.md, 점검)
HEAD_CMD = (math.radians(-20.0), math.radians(28.0))     # v21 머리 끄덕임 관절 범위(서보 명령 한계)


def head_to_crank(beta):
    """머리 각 beta(rad, + 숙임) → 크랭크(서보) 각 rad, v21 링크 표 보간"""
    return float(np.interp(beta, H21_B, H21_A))

# 팔 후보: kind straight(R) · elbow(L1 위팔, L2 아래팔 = 팔꿈치 축에서 손 바깥면) · bent(L1, L2, beta 앞으로 꺾은 각)
ARMS = {
    "곧은 131 (지금)": dict(kind="straight", R=0.131),
    "곧은 160": dict(kind="straight", R=0.160),
    "곧은 190": dict(kind="straight", R=0.190),
    "곧은 220": dict(kind="straight", R=0.220),
    "곧은 260": dict(kind="straight", R=0.260),
    "곧은 290": dict(kind="straight", R=0.290),
    "팔꿈치 80+80": dict(kind="elbow", L1=0.080, L2=0.080),
    "팔꿈치 100+100": dict(kind="elbow", L1=0.100, L2=0.100),
    "팔꿈치 110+110": dict(kind="elbow", L1=0.110, L2=0.110),
    "팔꿈치 90+130": dict(kind="elbow", L1=0.090, L2=0.130),
    "팔꿈치 140+140": dict(kind="elbow", L1=0.140, L2=0.140),
    "팔꿈치 120+120": dict(kind="elbow", L1=0.120, L2=0.120),
    "꺾은 120+80 (40°)": dict(kind="bent", L1=0.120, L2=0.080, beta=math.radians(40)),
    "꺾은 130+100 (60°)": dict(kind="bent", L1=0.130, L2=0.100, beta=math.radians(60)),
}


# 머리 받침 + 모터 없는 팔꿈치(2026-11-27): 닿는 길이 ≥ 213 mm(엎드림 길) 가 되게 꺾은 각마다 길이를 늘림
ARMS["꺾은 110+110 (30°)"] = dict(kind="bent", L1=0.110, L2=0.110, beta=math.radians(30))
ARMS["꺾은 125+125 (60°)"] = dict(kind="bent", L1=0.125, L2=0.125, beta=math.radians(60))
ARMS["꺾은 150+150 (90°)"] = dict(kind="bent", L1=0.150, L2=0.150, beta=math.radians(90))
for _t in (0.5, 1.0, 2.0):                                          # T90: 90° 굽혔을 때 용수철 토크 N·m
    ARMS[f"용수철 110+110 ({_t} N·m)"] = dict(kind="spring", L1=0.110, L2=0.110, T90=_t)


for _n in ("곧은 220", "팔꿈치 110+110", "팔꿈치 140+140"):          # 민감도: 어깨 · 팔꿈치를 XM430-W350(3.8 N·m, 90 g)으로 — 과제 한계(XL430) 밖
    ARMS[_n + " · XM"] = dict(ARMS[_n], shoulder="XM430-W350", elbow_servo="XM430-W350")
ARMS["팔꿈치 140+140 · W210"] = dict(ARMS["팔꿈치 140+140"], shoulder="XM430-W210", elbow_servo="XM430-W210")   # 2.7 N·m, 90 g (sp_common 이 더한 값)


def reach(a):
    """어깨 축에서 손 바깥면까지 가장 먼 거리(팔꿈치는 편 상태)"""
    if a["kind"] == "straight":
        return a["R"]
    if a["kind"] in ("elbow", "spring"):
        return a["L1"] + a["L2"]
    b = a["beta"]; c = a["L2"] - HAND_R
    tip = np.array([c * math.sin(b), -(a["L1"] + c * math.cos(b))])
    return float(np.linalg.norm(tip) + HAND_R)


def arm_mass_g(a):
    sh = SERVO_G[a.get("shoulder", "XL430-W250")]
    if a["kind"] == "straight":
        return sh + ARM_GMM * 1e3 * a["R"] + HAND_G
    extra = SERVO_G[a.get("elbow_servo", "XL430-W250")] if a["kind"] == "elbow" else (ELBOW_SPRING_G if a["kind"] == "spring" else 0.0)
    return sh + ARM_GMM * 1e3 * (a["L1"] + a["L2"]) + HAND_G + extra


def _set_inertia(body, pts):
    """pts: [(kg, (x, y, z))] → body 질량 · 무게중심 · 관성(점마다 반지름 12 mm 공으로 봐서 관성이 0 이 되지 않게)"""
    M = sum(p[0] for p in pts); c = sum(p[0] * np.asarray(p[1]) for p in pts) / M
    I = np.zeros(6)
    for mm, p in pts:
        x, y, z = np.asarray(p) - c; s = 0.4 * mm * 0.012 ** 2
        I += mm * np.array([y * y + z * z, x * x + z * z, x * x + y * y, -x * y, -x * z, -y * z]) + [s, s, s, 0, 0, 0]
    body.explicitinertial = True; body.mass = M; body.ipos = c; body.fullinertia = I; body.iquat = [1, 0, 0, 0]


def _rod(p0, p1, g, n=6):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return [(g / 1000 / n, p0 + (p1 - p0) * (i + 0.5) / n) for i in range(n)]


def _caps(body, p0, p1, r, rgba):
    g = body.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_CAPSULE; g.fromto = list(p0) + list(p1); g.size = [r, 0, 0]
    g.contype = 1; g.conaffinity = 2; g.density = 0; g.group = 1; g.rgba = rgba; g.friction = [0.8, 0.005, 0.0001]
    return g


def _ball(body, p, r, rgba):
    g = body.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_SPHERE; g.pos = list(p); g.size = [r, 0, 0]
    g.contype = 1; g.conaffinity = 2; g.density = 0; g.group = 1; g.rgba = rgba; g.friction = [0.8, 0.005, 0.0001]


def arm_mod(a):
    """a 에 shoulder · elbow_servo 키가 있으면 그 서보(민감도용, 기본 XL430-W250)"""
    ts, w0, kp0, arm_ = SERVO_SIM[a.get("elbow_servo", "XL430-W250")]
    g_sh = SERVO_G[a.get("shoulder", "XL430-W250")] / 1000; g_el = SERVO_G[a.get("elbow_servo", "XL430-W250")] / 1000
    spring = a["kind"] == "spring"
    if spring:
        g_el = ELBOW_SPRING_G / 1000
    col, hand_c = [0.95, 0.55, 0.15, 1], [0.3, 0.3, 0.35, 1]

    def f(spec):
        for s in "RL":
            y = -Y_ARM if s == "R" else Y_ARM
            b = spec.body(f"{s}_arm")
            for g in list(b.geoms):
                spec.delete(g)
            servo = [(g_sh, (0.0, y, -0.0121))]
            if a["kind"] == "straight":
                tip = np.array([0, y, -(a["R"] - HAND_R)])
                _caps(b, (0, y, 0.012), tip, ROD_R, col); _ball(b, tip, HAND_R, hand_c)
                _set_inertia(b, servo + _rod((0, y, 0), tip, ARM_GMM * 1e3 * a["R"]) + [(HAND_G / 1000, tip)])
            elif a["kind"] == "bent":
                el = np.array([0, y, -a["L1"]]); c = a["L2"] - HAND_R
                tip = el + c * np.array([math.sin(a["beta"]), 0, -math.cos(a["beta"])])
                _caps(b, (0, y, 0.012), el, ROD_R, col); _caps(b, el, tip, ROD_R, col); _ball(b, tip, HAND_R, hand_c)
                _set_inertia(b, servo + _rod((0, y, 0), el, ARM_GMM * 1e3 * a["L1"]) + _rod(el, tip, ARM_GMM * 1e3 * a["L2"]) + [(HAND_G / 1000, tip)])
            else:
                el = np.array([0, y, -a["L1"]])
                _caps(b, (0, y, 0.012), el, ROD_R, col)
                _set_inertia(b, servo + _rod((0, y, 0), el, ARM_GMM * 1e3 * a["L1"]) + [(g_el, el)])
                fb = b.add_body(); fb.name = f"{s}_forearm"; fb.pos = list(el)
                j = fb.add_joint(); j.name = f"{s}_elbow"; j.type = mujoco.mjtJoint.mjJNT_HINGE; j.axis = [0, 1, 0]
                j.range = [-ELBOW_RANGE, ELBOW_RANGE]; j.damping = [ts / w0, 0, 0]; j.armature = arm_
                tip = np.array([0, 0, -(a["L2"] - HAND_R)])
                _caps(fb, (0, 0, 0), tip, ROD_R * 0.9, col); _ball(fb, tip, HAND_R, hand_c)
                _set_inertia(fb, _rod((0, 0, 0), tip, ARM_GMM * 1e3 * a["L2"]) + [(HAND_G / 1000, tip)])
                if spring:                       # 서보 없음: 용수철이 펴는 쪽으로 밀고 0° 에서 멈춤턱
                    j.range = [-ELBOW_RANGE, 0.0]; j.damping = [SPRING_DAMP, 0, 0]; j.armature = 0.0
                    j.stiffness = [a["T90"] / (math.pi / 2 + SPRING_PRE), 0, 0]; j.springref = SPRING_PRE
                    j.solref_limit = [0.004, 1.0]        # 멈춤턱을 단단하게(기본 0.02 면 미리 감김에 0° 를 2.7° 넘어감)
                    continue
                ac = spec.add_actuator(); ac.name = f"{s}_elbow"; ac.target = f"{s}_elbow"; ac.trntype = mujoco.mjtTrn.mjTRN_JOINT
                ac.set_to_position(kp=kp0); ac.forcerange = [-ts, ts]; ac.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                ac.ctrlrange = [-ELBOW_RANGE, ELBOW_RANGE]; ac.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    return f


def build(arm, knee_max=1.745, hip_min=-1.396, ext=LEG_EXT, kp=2.0):
    """arm: ARMS 의 값(dict) 또는 None(원래 팔 메시 그대로). 관절 범위는 설계값(무릎 0–100°, 고관절 피치 −80 – +30°)이 기본."""
    wd = widen(knee_max, hip_min)
    am = arm_mod(arm) if arm else None
    SEG.update(thigh=ext, shank=ext, mass=True, servo=CUR)
    try:
        sm = dict(CUR, arm_swing=arm.get("shoulder", "XL430-W250")) if arm else CUR
        m, spec = build_model2(hip_roll=hrA(0.0405), servo_map=sm, extra_bodies=lambda sp: (wd(sp), am and am(sp)))
    finally:
        SEG.update(thigh=0.0, shank=0.0, mass=False, servo=None)
    if os.environ.get("SELFCOL"):
        selfcol(m)
    if os.environ.get("HEAD21"):
        head21(m)
    if kp:
        scale_kp(m, kp)
    return m, spec


def selfcol(m):
    """다리(허벅지 · 정강이 · 발목 · 발) ↔ 몸통 · 머리 충돌을 켬(원래 모델은 계산 안 함). 고관절을 −115° 까지 굽히면 몸통 앞 아래와 닿을 수 있어서.
    몸통 · 머리 geom 의 conaffinity 에 비트 4 를 더하고, 허벅지 geom 의 contype 에 비트 4 를 더한다(정강이 · 발목 · 발은 이미 비트 4)."""
    for i in range(m.ngeom):
        b = m.body(m.geom_bodyid[i]).name
        if b in ("torso", "head") and m.geom_contype[i] and "arm_" not in (m.geom(i).name or ""):
            m.geom_conaffinity[i] |= 4
        if b.endswith("_thigh") and m.geom_contype[i]:
            m.geom_contype[i] |= 4


def head21(m):
    """머리 끄덕임을 v21 로: 고리 다항식(머리 · 귀 = 4차식(크랭크 각)) 다시 맞춤, head_nod 범위 = 껍데기 멈춤턱 −25.5 / +32.5°,
    크랭크 · 귀 관절 범위는 그보다 넓게(먼저 걸리지 않게), 크랭크 서보 명령은 머리 −20 ~ +28° 에 해당하는 크랭크 각."""
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    jh, jc, je = J("head_nod"), J("nod_crank"), J("ear_axle")
    ph = np.polynomial.polynomial.polyfit(H21_A, H21_B, 4); pe = np.polynomial.polynomial.polyfit(H21_A, H21_E, 4)
    for e in range(m.neq):
        if m.eq_obj2id[e] == jc and m.eq_obj1id[e] == jh:
            m.eq_data[e][:5] = ph
        elif m.eq_obj2id[e] == jc and m.eq_obj1id[e] == je:
            m.eq_data[e][:5] = pe
    a_stop = [head_to_crank(b) for b in HEAD_STOP]
    m.jnt_range[jh] = HEAD_STOP
    m.jnt_range[jc] = [a_stop[0] - 0.03, a_stop[1] + 0.03]
    m.jnt_range[je] = [float(np.interp(a_stop[0], H21_A, H21_E)) - 0.05, float(np.interp(a_stop[1], H21_A, H21_E)) + 0.05]
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "nod_crank")
    m.actuator_ctrlrange[a] = [head_to_crank(b) for b in HEAD_CMD]
    return ph, pe


def place(m, d, R):
    """몸통 회전 R 로 눕히고, 가장 낮은 충돌 geom 이 바닥 위 3 mm 가 되게 올림(메시 · 캡슐 · 공 모두, mj_geomDistance)."""
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(9)); d.qpos[3:7] = q; d.qpos[0:3] = [0, 0, 1.0]
    mujoco.mj_forward(m, d)
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"); ft = np.zeros(6)
    dmin = min(mujoco.mj_geomDistance(m, d, i, fl, 5.0, ft) for i in range(m.ngeom) if i != fl and m.geom_contype[i])
    d.qpos[2] += 0.003 - dmin
    mujoco.mj_forward(m, d)


R_PRONE = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)     # 몸통 앞(x)이 바닥 쪽
R_SUPINE = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)    # 등이 바닥 쪽
JK = {"hip": "hip_pitch", "knee": "knee", "ankle": "ankle_pitch", "arm": "arm_swing", "elbow": "elbow", "head": "nod_crank"}


class Sim:
    """한 팔 후보 모델을 한 번 만들어 두고 키프레임을 여러 번 돌린다(최적화용)."""

    def __init__(self, arm, knee_max=1.745, hip_min=-1.396):
        self.m, _ = build(arm, knee_max, hip_min)
        m = self.m; self.ix = Idx2(m)
        # 팔꿈치 액추에이터의 정지토크 · 무부하속도(Idx 는 forcerange 1.4 → XL 로 맞게 앎)
        d0 = mujoco.MjData(m); d0.qpos[:] = m.qpos0; mujoco.mj_forward(m, d0); self.z_stand = float(d0.subtree_com[1][2])
        self.amap = []
        for a, nm in enumerate(self.ix.act):
            for k, jn in JK.items():
                if nm.endswith(jn):
                    self.amap.append((a, k))
        self.tb = self.ix.torso
        self.fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_bodies = {self.ix.foot["R"], self.ix.foot["L"], self.ix.body("R_ankle_link"), self.ix.body("L_ankle_link")}
        # 목 측정(머리 받침 조사): 머리 · 귀 ↔ 바닥 접촉력, 고리(등식 제약)와 멈춤턱(관절 한계)이 head_nod 에 주는 토크, 크랭크 서보 힘
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.jh, self.jc = J("head_nod"), J("nod_crank")
        self.eq_head = next(e for e in range(m.neq) if m.eq_obj1id[e] == self.jh and m.eq_obj2id[e] == self.jc)
        self.head_bodies = {self.ix.body("head"), self.ix.body("ears"), self.ix.body("nod_crank")}
        self.a_crank = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "nod_crank")
        self.j_el = [J(f"{s}_elbow") for s in "RL" if J(f"{s}_elbow") >= 0]

    def run(self, keys, start="prone", T_hold=1.5, render=None, settle=0.8):
        """keys: [(dur s, {hip, knee, ankle, arm, elbow: rad})] 좌우 같음, 앞 키에서 선형 보간.
        render: (Renderer, cam_fn, fps) 이면 프레임 목록도 돌려줌."""
        m = self.m; ix = self.ix
        d = mujoco.MjData(m); d.qpos[:] = m.qpos0
        place(m, d, R_PRONE if start == "prone" else R_SUPINE)
        cur = {k: 0.0 for k in JK}; sched = [(settle, dict(cur))]
        for dur, tg in keys:
            c = dict(sched[-1][1]); c.update(tg); sched.append((max(dur, 0.05), c))
        T = sum(s[0] for s in sched) + T_hold
        n = int(T / m.opt.timestep); dt = m.opt.timestep
        ends = np.cumsum([s[0] for s in sched])
        b = np.array([m.dof_damping[ix.vadr[j]] for j in ix.act_j])
        nrec = 0; tq_all = []; hist = []; frames = []; sat = np.zeros(m.nu); neck = []; el = []
        next_frame = 0.0
        for k in range(n):
            t = k * dt
            i = int(np.searchsorted(ends, t, side="right"))
            if i >= len(sched):
                tgt = sched[-1][1]
            else:
                p = sched[i - 1][1] if i else sched[0][1]; c = sched[i][1]
                s = (t - (ends[i - 1] if i else 0.0)) / sched[i][0]
                tgt = {kk: p[kk] + (c[kk] - p[kk]) * s for kk in c}
            for a, kk in self.amap:
                d.ctrl[a] = head_to_crank(tgt[kk]) if kk == "head" else tgt[kk]
            np.clip(d.ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1], out=d.ctrl)
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                return dict(ok=False, error="nan"), hist, frames
            if t > settle and k % 5 == 0:
                w = np.array([d.qvel[ix.vadr[j]] for j in ix.act_j])
                tq_all.append(np.abs(d.actuator_force - b * w) / ix.stall)
                sat += np.abs(d.actuator_force) >= 0.999 * m.actuator_forcerange[:, 1]; nrec += 1
                neck.append(self.neck(d)); el.append([d.qpos[m.jnt_qposadr[j]] for j in self.j_el] + [d.qfrc_spring[m.jnt_dofadr[j]] for j in self.j_el])
            if k % 20 == 0:
                fm = 0.5 * (d.xpos[self.ix.foot["R"]][:2] + d.xpos[self.ix.foot["L"]][:2])
                hist.append((t, d.subtree_com[1][2], math.degrees(math.acos(np.clip(d.xmat[self.tb][8], -1, 1))), d.xmat[self.tb][2],
                             float(np.linalg.norm(d.subtree_com[1][:2] - fm))))
            if render is not None and t >= next_frame:
                ren, camf, fps = render; ren.update_scene(d, camf(d)); frames.append(ren.render().copy()); next_frame += 1.0 / fps
        H = np.array(hist); last = H[:, 0] > T - 1.0
        # 끝 1 s 동안 발(발목 링크 · 발) 밖의 몸이 바닥에 닿았는지
        touch = set()
        for c_i in range(d.ncon):
            c = d.contact[c_i]
            if self.fl in (c.geom1, c.geom2):
                g = c.geom2 if c.geom1 == self.fl else c.geom1
                bd = m.geom_bodyid[g]
                if bd not in self.foot_bodies:
                    touch.add(m.body(bd).name)
        ok = bool((H[last, 2] < 15).all() and (H[last, 1] > 0.9 * self.z_stand).all())
        TQ = np.array(tq_all)
        res = dict(ok=ok, z_end=float(H[-1, 1]), z_stand=self.z_stand, tilt_end=float(H[-1, 2]), T=T,
                   z_last_min=float(H[last, 1].min()), tilt_last_max=float(H[last, 2].max()), touch_end=sorted(touch),
                   score=self.score(H, T))
        N = np.array(neck)
        if len(N):
            # 열: 0 머리 · 귀 ↔ 바닥 힘 N, 1 고리가 head_nod 에 준 토크, 2 멈춤턱 토크, 3 고리 토크 × dβ/dα(크랭크 쪽, 가상일), 4 크랭크 서보 힘(시뮬),
            #     5 가장 큰 접촉 geom, 6 머리 각, 7 크랭크 서보 부하(힘 − 감쇠 × 속도)
            g = int(N[np.argmax(N[:, 0]), 5]) if N[:, 0].max() > 0 else -1
            res["neck"] = dict(contact_max_N=float(N[:, 0].max()), contact_frac=float((N[:, 0] > 0.05).mean()), contact_geom=(m.geom(g).name if g >= 0 else ""),
                               link_tq_max=float(np.abs(N[:, 1]).max()), link_tq_p99=float(np.percentile(np.abs(N[:, 1]), 99)),
                               stop_tq_max=float(np.abs(N[:, 2]).max()), stop_frac=float((np.abs(N[:, 2]) > 1e-3).mean()),
                               crank_calc_max=float(np.abs(N[:, 3]).max()), crank_calc_p99=float(np.percentile(np.abs(N[:, 3]), 99)),
                               crank_act_max=float(np.abs(N[:, 4]).max()), crank_act_p99=float(np.percentile(np.abs(N[:, 4]), 99)),
                               crank_sat_frac=float((np.abs(N[:, 4]) >= 0.999 * m.actuator_forcerange[self.a_crank, 1]).mean()),
                               head_deg=[float(np.degrees(N[:, 6].min())), float(np.degrees(N[:, 6].max()))],
                               crank_load_max=float(np.abs(N[:, 7]).max()), crank_load_p99=float(np.percentile(np.abs(N[:, 7]), 99)),
                               contact_p99_N=float(np.percentile(N[:, 0], 99)), link_tq_p99_signed=[float(np.percentile(N[:, 1], 1)), float(np.percentile(N[:, 1], 99))])
        if self.j_el and el:
            E = np.array(el); k = len(self.j_el)
            res["elbow_deg"] = [float(np.degrees(E[:, :k].min())), float(np.degrees(E[:, :k].max()))]
            res["elbow_spring_max"] = float(np.abs(E[:, k:]).max())
        for a, nm in enumerate(ix.act):
            if nm.startswith("R_") and any(nm.endswith(j) for j in JK.values()):
                res.setdefault("torque_p99", {})[nm[2:]] = float(np.percentile(TQ[:, a], 99))
                res.setdefault("torque_max", {})[nm[2:]] = float(TQ[:, a].max())
                res.setdefault("sat_frac", {})[nm[2:]] = float(sat[a] / max(nrec, 1))
        return res, H, frames

    def neck(self, d):
        m = self.m; fc = 0.0; gmax = -1; fmax = 0.0; f6 = np.zeros(6)
        for c_i in range(d.ncon):
            c = d.contact[c_i]
            if self.fl in (c.geom1, c.geom2):
                g = c.geom2 if c.geom1 == self.fl else c.geom1
                if m.geom_bodyid[g] in self.head_bodies:
                    mujoco.mj_contactForce(m, d, c_i, f6); fc += f6[0]
                    if f6[0] > fmax:
                        fmax = f6[0]; gmax = g
        link = stop = 0.0
        for r in range(d.nefc):
            if d.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[r] == self.eq_head:
                link += d.efc_force[r]                       # 이 행의 J 는 head_nod 열이 1 → 힘 = head_nod 토크
            elif d.efc_type[r] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT and d.efc_id[r] == self.jh:
                stop += d.efc_force[r]
        ac = d.qpos[m.jnt_qposadr[self.jc]]
        dbda = np.polynomial.polynomial.polyval(ac, np.polynomial.polynomial.polyder(m.eq_data[self.eq_head][:5]))
        vc = m.jnt_dofadr[self.jc]       # 크랭크 서보 부하 = 서보 힘 − 감쇠 × 속도(다른 관절의 토크 몫과 같은 정의)
        return [fc, link, stop, link * dbda, d.actuator_force[self.a_crank], gmax, d.qpos[m.jnt_qposadr[self.jh]],
                d.actuator_force[self.a_crank] - m.dof_damping[vc] * d.qvel[vc]]

    def score(self, H, T):
        """최적화 비용(작을수록 좋음): 순간 비용 = 몸통 기울기(5° 넘는 몫/30°) + 3 × 무게중심 높이 부족(서 있을 때 95 % 대비)
        + 1.5 × 두 발 가운데와 무게중심 사이 수평 거리(100 mm 단위, 3 에서 자름). 끝 1 s 평균(반) + 지나는 동안 가장 작은 순간(반).
        수평 거리 항은 「팔로 버틴 판자 자세」와 「발 위 쪼그림」을 가리기 위함(높이 · 기울기만으로는 둘이 비슷함)."""
        last = H[:, 0] > T - 1.0
        inst = np.maximum(H[:, 2] - 5, 0) / 30 + 3 * np.maximum(0.95 - H[:, 1] / self.z_stand, 0) + 1.5 * np.minimum(H[:, 4] / 0.1, 3)
        return float(0.5 * inst[last].mean() + 0.5 * inst.min())

def side_cam(dist=0.9, elev=-8, az=90):
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance = dist; cam.azimuth = az; cam.elevation = elev

    def f(d):
        cam.lookat[:] = [d.subtree_com[1][0], d.subtree_com[1][1], 0.16]
        return cam
    return f


def strip(sim, keys, start, path, fps=2, title=None):
    """키프레임을 돌리며 fps 장/초로 옆 모습을 찍어 한 줄 그림으로 저장"""
    ren = mujoco.Renderer(sim.m, 240, 320)
    r, H, fr = sim.run(keys, start=start, render=(ren, side_cam(), fps))
    plt = korean_font()
    n = len(fr); fig, ax = plt.subplots(1, n, figsize=(1.6 * n, 1.6))
    for i, (a, img) in enumerate(zip(np.atleast_1d(ax), fr)):
        a.imshow(img); a.axis("off"); a.set_title(f"{i / fps:.1f} s", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)
    return r
