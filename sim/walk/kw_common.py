"""sim/walk 공용: 키토루 v18 모델 읽기(MjSpec 변형), 센서(제어기가 보는 것만), 평가(정답 상태, 제어기는 안 봄).

tests/common.py 를 import 하지 않는다(다른 에이전트가 v19 로 고치는 중, 기본값이 v17). 필요한 것만 v18 로 옮겼다.
"""
import json, math, os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "4")
import numpy as np
import mujoco

WALK = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(WALK)
XML = os.path.join(SIM, "model", "kitoru_v18.xml")
LOOP = json.load(open(os.path.join(SIM, "model", "closed_loop_fit_kitoru_v18.json")))
RES = os.path.join(WALK, "results"); os.makedirs(RES, exist_ok=True)
PV = np.polynomial.polynomial.polyval

LEGJ = ["hip_yaw", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
LEG = [f"{s}_{j}" for s in "RL" for j in LEGJ]
SERVO_OF = {"hip_yaw": "XC430-W240", "hip_pitch": "XM430-W350", "knee": "XM430-W350", "ankle_pitch": "XC430-W240",
            "ankle_roll": "XM430-W350", "arm_swing": "XC430-W240", "nod_crank": "XL430-W250", "hip_roll": "XM430-W350"}
# (정지토크 N*m, 무부하속도 rad/s) — 모델 · 인계 표 값(kitoru_v18.xml 주석과 같음) [설계 파일]
SPEC = {"XM430-W350": (3.8, 4.5), "XC430-W240": (1.75, 6.81), "XL430-W250": (1.4, 5.45)}
# 렌즈: 설계 build18 camera_pi_v2 렌즈 앞면 (70, 37.5, 453) mm, head body 원점 z 348.5 mm (tests/camera_view.py 와 같음)
LENS_HEAD = np.array([70.0, 37.5, 453.0]) / 1000 - np.array([0, 0, 0.3485])
CAM_FOVY = 48.8


def servo_of(jname):
    for k, v in SERVO_OF.items():
        if jname.endswith(k):
            return v
    raise KeyError(jname)


def build_model(friction=None, head_mass=1.0, pad_dy_out=0.0, pad_dx=0.0, armature=1.0, camera=False,
                hip_roll=False, extra_bodies=None):
    """MjSpec 로 v18 을 읽고 시뮬에서만 바꿀 수 있는 것을 바꾼다(설계 파일은 안 건드림).
    friction: 모든 충돌 geom(바닥 포함)의 미끄럼 마찰. MuJoCo 는 두 geom 중 큰 값을 쓰므로 양쪽을 다 바꾼다.
    head_mass: 머리 body 질량 · 관성 배수(무게중심 위치는 그대로).
    pad_dy_out: 발 바깥쪽 패드 두 개를 바깥으로 옮기는 거리 m(발 폭 넓히기 흉내).  pad_dx: 앞 패드 앞으로 / 뒤 패드 뒤로 m.
    armature: 모든 서보 관절 armature 배수.  camera: 왼쪽 눈 렌즈에 Pi Camera v2 카메라 추가.
    hip_roll: 허벅지 위(고관절 피치 축 자리)에 롤 관절 + XM430 서보를 더한 가상 설계(형상 · 질량 그대로)."""
    spec = mujoco.MjSpec.from_file(XML)
    if friction is not None:
        for g in spec.geoms:
            if g.contype or g.conaffinity:
                g.friction = [friction, g.friction[1], g.friction[2]]
    if head_mass != 1.0:
        b = spec.body("head"); b.mass = b.mass * head_mass; b.fullinertia = np.asarray(b.fullinertia) * head_mass
    for s, sg in (("R", -1.0), ("L", 1.0)):
        for i in range(4):
            g = spec.geom(f"{s}_foot_pad{i}")
            p = np.array(g.pos)
            # pad0/2 는 R 에서 바깥(y -0.023), L 은 pad 이름 순서가 같다고 보고 |y| 큰 쪽을 바깥으로 판정
            outer = (p[1] * sg) > 0 if s == "L" else p[1] < -0.01
            if pad_dy_out and outer:
                p[1] += sg * pad_dy_out
            if pad_dx:
                p[0] += pad_dx if p[0] > 0 else -pad_dx
            g.pos = p
    if armature != 1.0:
        for j in spec.joints:
            if j.name and not j.name.startswith(("head_nod", "ear_axle")) and j.type == mujoco.mjtJoint.mjJNT_HINGE:
                j.armature = j.armature * armature
    if camera:
        c = spec.body("head").add_camera(); c.name = "eye"; c.pos = LENS_HEAD + [0.0015, 0, 0]
        c.fovy = CAM_FOVY; c.quat = [0.5, 0.5, -0.5, -0.5]      # 카메라 -z 가 +x(정면), 영상 위쪽 +z
        spec.visual.map.znear = 0.0005
    if hip_roll:
        for s in "RL":
            th = spec.body(f"{s}_thigh")
            j = th.add_joint(); j.name = f"{s}_hip_roll"; j.type = mujoco.mjtJoint.mjJNT_HINGE; j.axis = [1, 0, 0]
            j.range = [-0.35, 0.35]; j.damping = [3.8 / 4.5, 0, 0]; j.armature = 0.045
            a = spec.add_actuator(); a.name = f"{s}_hip_roll"; a.target = f"{s}_hip_roll"
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.set_to_position(kp=17.5); a.forcerange = [-3.8, 3.8]; a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.ctrlrange = [-0.35, 0.35]; a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    if extra_bodies:
        extra_bodies(spec)
    m = spec.compile()
    return m, spec


def scale_kp(m, kp_scale):
    """Dynamixel 위치 P 이득 레지스터를 올린 것과 같은 효과: position 액추에이터 kp 배수(test3 와 같은 방법)"""
    if isinstance(kp_scale, dict):
        for i in range(m.nu):
            nm = m.actuator(i).name
            s = next((v for k, v in kp_scale.items() if nm.endswith(k)), 1.0)
            m.actuator_gainprm[i, 0] *= s; m.actuator_biasprm[i, 1] *= s
    else:
        m.actuator_gainprm[:, 0] *= kp_scale; m.actuator_biasprm[:, 1] *= kp_scale


# ---------------------------------------------------------------- 머리 크랭크 (닫힌 고리)
_B = np.radians(LOOP["beta_deg"]); _A = np.radians(LOOP["crank_deg"])


def crank_for_head(beta):
    """머리 각 beta(rad, + 아래) → 크랭크 각(rad), 설계 고리 표 보간"""
    return float(np.interp(beta, _B, _A))


def loop_qpos(alpha):
    return PV(alpha, LOOP["polycoef_head"]), PV(alpha, LOOP["polycoef_ear"])


# ---------------------------------------------------------------- 이름 → 주소
class Idx:
    def __init__(self, m):
        self.m = m
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.jnames = [m.joint(i).name for i in range(m.njnt)]
        self.act = [m.actuator(i).name for i in range(m.nu)]
        self.act_j = [m.joint(m.actuator_trnid[i][0]).name for i in range(m.nu)]
        self.qadr = {n: m.jnt_qposadr[J(n)] for n in self.jnames if n != "root"}
        self.vadr = {n: m.jnt_dofadr[J(n)] for n in self.jnames if n != "root"}
        self.aid = {n: i for i, n in enumerate(self.act)}
        self.body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.torso = self.body("torso"); self.head = self.body("head")
        self.foot = {s: self.body(f"{s}_foot") for s in "RL"}
        self.pads = {s: [i for i in range(m.ngeom) if (m.geom(i).name or "").startswith(f"{s}_foot_pad")] for s in "RL"}
        self.pad_owner = {g: s for s in "RL" for g in self.pads[s]}
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.robot_geoms = set(range(m.ngeom)) - {self.floor}
        self.legj = [n for n in self.jnames if n.startswith(("R_", "L_")) and not n.endswith("arm_swing")]
        self.mass = float(m.body_subtreemass[self.torso])
        self.stall = np.array([SPEC[servo_of(j)][0] for j in self.act_j])
        self.w0 = np.array([SPEC[servo_of(j)][1] for j in self.act_j])


# ---------------------------------------------------------------- 센서: 제어기가 받는 것은 이것뿐
def sense(m, d, ix):
    """실기에서 얻을 수 있는 것만:
    - IMU: 몸통 자세 쿼터니언(자세 필터 결과라고 봄), 몸통 좌표 각속도(자이로)
    - 발 압력: 패드 8개 각각의 수직 힘(패드 밑 힘 센서, FSR 같은 것) → 발마다 합 · 압력중심(발 좌표)
    - 관절: 액추에이터 관절 각 · 속도(Dynamixel present position / velocity)"""
    quat = d.xquat[ix.torso].copy()
    gyro = d.qvel[3:6].copy()                        # free joint 각속도 = 몸통 좌표
    padf = {}
    f6 = np.zeros(6)
    for c_i in range(d.ncon):
        c = d.contact[c_i]
        g = c.geom1 if c.geom1 in ix.pad_owner else (c.geom2 if c.geom2 in ix.pad_owner else None)
        if g is None or ix.floor not in (c.geom1, c.geom2):
            continue
        mujoco.mj_contactForce(m, d, c_i, f6)
        padf[g] = padf.get(g, 0.0) + abs(f6[0])
    feet = {}
    for s in "RL":
        fz = [padf.get(g, 0.0) for g in ix.pads[s]]
        tot = sum(fz)
        loc = np.array([m.geom_pos[g][:2] for g in ix.pads[s]])     # 발 body 좌표의 패드 위치(센서 설치 위치 = 아는 값)
        cop = (np.array(fz) @ loc / tot) if tot > 1e-6 else np.zeros(2)
        feet[s] = {"fz": tot, "cop_local": cop, "pad_fz": np.array(fz)}
    q = np.array([d.qpos[ix.qadr[j]] for j in ix.act_j])
    qd = np.array([d.qvel[ix.vadr[j]] for j in ix.act_j])
    return {"quat": quat, "gyro": gyro, "feet": feet, "q": q, "qd": qd}


# ---------------------------------------------------------------- 평가(정답 상태) — 제어기는 쓰지 않는다
def true_cop_margin(m, d, ix):
    """바닥이 로봇에 주는 힘의 압력중심과, 지금 닿아 있는 패드들의 볼록 껍질 가장자리까지 거리(m, + 안쪽)."""
    from scipy.spatial import ConvexHull
    f6 = np.zeros(6); F = 0.0; M = np.zeros(2); pts = []
    for c_i in range(d.ncon):
        c = d.contact[c_i]
        if ix.floor not in (c.geom1, c.geom2):
            continue
        mujoco.mj_contactForce(m, d, c_i, f6)
        fn = abs(f6[0]); p = c.pos[:2]
        F += fn; M += fn * p
        g = c.geom2 if c.geom1 == ix.floor else c.geom1
        if g in ix.pad_owner and fn > 0.05:
            ctr = d.geom_xpos[g][:2]; r = m.geom_size[g][0]
            pts += [ctr + r * np.array([math.cos(t), math.sin(t)]) for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    if F < 1e-3:
        return None, None, 0.0
    cop = M / F
    if len(pts) < 3:
        return cop, -1.0, F
    P = np.array(pts)
    try:
        h = ConvexHull(P)
    except Exception:
        return cop, 0.0, F
    eq = h.equations          # a x + b y + c <= 0 inside, (a,b) unit
    dist = -(eq[:, :2] @ cop + eq[:, 2])
    return cop, float(dist.min()), F


def utilisation(m, d, ix):
    """관절별 |힘|/정지토크, u = |τ|/τs + |ω|/ω0, 포화 여부.
    τ 는 관절에 실제로 걸리는 부하 토크 = 액추에이터 힘 − 감쇠(역기전력) 토크 b·ω. 모델의 감쇠 b = τs/ω0 가 이미 속도 몫이라
    액추에이터 힘을 그대로 쓰면 |ω|/ω0 를 두 번 센다(검토 문서 2.1 의 역동역학 u 와 같은 정의)."""
    w_s = np.array([d.qvel[ix.vadr[j]] for j in ix.act_j])
    b = np.array([m.dof_damping[ix.vadr[j]] for j in ix.act_j])
    load = np.abs(d.actuator_force - b * w_s)
    w = np.abs(w_s)
    fr = m.actuator_forcerange[:, 1]
    return np.abs(d.actuator_force) / ix.stall, load / ix.stall + w / ix.w0, np.abs(d.actuator_force) >= 0.999 * fr


def jdump(obj, path):
    def conv(x):
        if isinstance(x, dict): return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [conv(v) for v in x]
        if isinstance(x, (np.floating, np.integer, np.bool_)): return x.item()
        if isinstance(x, np.ndarray): return conv(x.tolist())
        return x
    json.dump(conv(obj), open(path, "w"), indent=1, ensure_ascii=False)


def korean_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = os.path.expanduser("~/.local/share/fonts/NotoSansKR.ttf")
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return plt
