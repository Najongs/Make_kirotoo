"""키토루 피드백 걷기 제어기.

구성(제어 주기마다):
  1) 걸음 계획: 발자국 목록 → 압력중심(ZMP) 기준 궤적(한 발 지지 때는 디딤발 가운데, 두 발 지지 때는 직선으로 옮김)
     → 선형 역진자(LIPM) ZMP 미리보기 제어(Kajita 2003)로 무게중심 기준 궤적.
  2) 상태 추정: IMU 자세 + 관절 각 → 계획용 모델 정기구학으로 「디딤발에서 본 무게중심」. 디딤발은 발 압력 센서로 고르고,
     디딤발이 바뀔 때 새 발 위치를 운동학으로 이어 붙인다(다리 주행 거리계). 속도는 차분 + 저역 통과.
  3) 되먹임: 발산 성분(DCM, 캡처 포인트) ξ = c + ċ/ω 의 오차로 압력중심 목표를 옮긴다 p_d = p_ref + Kξ (ξ − ξ_ref).
     밀기처럼 오차가 크면 다음 발 디딤 위치도 옮긴다(앞뒤만 — 옆은 고관절 롤이 없어 거의 못 옮긴다).
  4) 전신 역기구학(수치, 감쇠 최소제곱): 두 발 자세(6차원), 무게중심 xy, 몸통 피치 · 요, 몸통 높이 → 관절 각. 몸통 롤은 자유:
     고관절 롤이 없어 옆으로 무게를 옮기려면 몸 전체가 디딤발 발목 롤로 기운다(그 각을 역기구학이 정한다).
  5) 서보 휨 보상: 계획 자세 · 압력중심 목표에서의 준정적 관절 토크 τ = g(q) − Jᵀ F 를 kp 로 나눠 위치 명령에 더한다.
     (Dynamixel 위치 모드에서 목표각을 τ/kp 만큼 앞에 두는 것과 같다.) DCM 되먹임은 F 의 작용점(p_d)을 통해 들어간다.
  6) 몸통 자세 되먹임: IMU 몸통 피치 · 롤과 역기구학이 예상한 값의 차이를 디딤발 고관절 피치 · 발목 롤로 줄인다.
제어기는 kw_common.sense() 가 주는 값(IMU, 발 패드 힘, 관절 각 · 속도)만 쓴다.
"""
import math
import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are
from kw_common import Idx, loop_qpos, crank_for_head

G = 9.81


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_err(Rd, Rc):
    """world 회전 벡터 log(Rd Rcᵀ)"""
    q = np.zeros(4); mujoco.mju_mat2Quat(q, (Rd @ Rc.T).reshape(9))
    v = np.zeros(3); mujoco.mju_quat2Vel(v, q, 1.0); return v


def rpy_of(R):
    """ZYX: yaw, pitch, roll"""
    yaw = math.atan2(R[1, 0], R[0, 0]); pitch = math.asin(-max(-1, min(1, R[2, 0]))); roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def mjerk(s):
    s = min(max(s, 0.0), 1.0); return s ** 3 * (10 - 15 * s + 6 * s * s)


# ---------------------------------------------------------------- ZMP 미리보기 제어
class Preview:
    def __init__(self, dt, zc, horizon=1.6, Qe=1.0, R=1e-6):
        self.dt, self.N = dt, int(round(horizon / dt))
        A = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1.0]]); B = np.array([[dt ** 3 / 6], [dt * dt / 2], [dt]])
        C = np.array([[1, 0, -zc / G]])
        At = np.block([[np.eye(1), C @ A], [np.zeros((3, 1)), A]]); Bt = np.vstack([C @ B, B])
        It = np.array([[1.0], [0], [0], [0]]); Ft = np.vstack([C @ A, A])
        Q = np.diag([Qe, 0, 0, 0])
        P = solve_discrete_are(At, Bt, Q, R * np.eye(1))
        S = (R + Bt.T @ P @ Bt).item()
        self.Gi = (Bt.T @ P @ It).item() / S
        self.Gx = (Bt.T @ P @ Ft / S).ravel()
        Ac = At - Bt @ (Bt.T @ P @ At) / S
        X = -Ac.T @ P @ It
        Gp = [-self.Gi]
        for _ in range(1, self.N):
            Gp.append((Bt.T @ X).item() / S); X = Ac.T @ X
        self.Gp = np.array(Gp)
        self.A, self.B, self.C = A, B.ravel(), C.ravel()
        self.x = np.zeros((3, 2)); self.esum = np.zeros(2)

    def reset(self, c0):
        self.x[:] = 0; self.x[0] = c0; self.esum[:] = 0

    def step(self, zmp_future):
        """zmp_future: (N, 2) 지금부터 N 칸 ZMP 기준. 상태를 한 칸 진행하고 (c, ċ, c̈, p) 를 돌려준다."""
        p_now = self.C @ self.x
        self.esum += p_now - zmp_future[0]
        u = -self.Gi * self.esum - self.Gx @ self.x - self.Gp[:self.N - 1] @ zmp_future[1:self.N]
        # Kajita 식: u = −Gi Σe − Gx x − Σ_{j=1..N} Gp(j) p_ref(k+j),  Gp(1) = −Gi
        self.x = self.A @ self.x + np.outer(self.B, u)
        return self.x[0].copy(), self.x[1].copy(), self.x[2].copy(), self.C @ self.x


# ---------------------------------------------------------------- 발자국 계획
class Footsteps:
    """steps[i] = dict(side, pos(2), yaw). steps[0], steps[1] = 처음 두 발(서 있는 자리). 걸음 i(≥2) = steps[i] 로 흔드는 발을
    옮기는 것, 그동안 steps[i-1] 이 디딤발. 시간표: t0 에서 첫 두 발 지지 Tinit(무게를 첫 디딤발로) → [한 발 Ts, 두 발 Td] 반복."""

    def __init__(self, steps, t0, Ts, Td, Tinit, Tend, zmp_off, n_active=None):
        self.steps, self.t0, self.Ts, self.Td, self.Tinit, self.Tend = steps, t0, Ts, Td, Tinit, Tend
        self.zmp_off = zmp_off       # 발 좌표 압력중심 목표(x, y_out): 발 바깥쪽 + 쪽
        self.n = len(steps)

    def center(self, i):
        st = self.steps[i]; sg = -1.0 if st["side"] == "R" else 1.0
        off = np.array([self.zmp_off[0], sg * self.zmp_off[1]])
        return st["pos"] + rz(st["yaw"])[:2, :2] @ off

    def phase(self, t):
        """('pre', None) / ('ds', k, s) k=다음 디딤발 번호 / ('ss', k, s) k=흔드는 발 걸음 번호 / ('end', s)"""
        T = self.Ts + self.Td
        if t < self.t0:
            return ("pre", 0, 0.0)
        if t < self.t0 + self.Tinit:
            return ("ds", 1, (t - self.t0) / self.Tinit)
        tw = t - self.t0 - self.Tinit
        k = int(tw // T); r = tw - k * T; i = k + 2
        if i >= self.n:
            te = tw - (self.n - 2) * T
            return ("end", self.n, min(te / self.Tend, 1.0))
        if r < self.Ts:
            return ("ss", i, r / self.Ts)
        return ("ds", i, (r - self.Ts) / self.Td)

    def zmp(self, t):
        ph = self.phase(t)
        if ph[0] == "pre":
            return 0.5 * (self.center(0) + self.center(1))
        if ph[0] == "ds":
            k, s = ph[1], ph[2]
            a = 0.5 * (self.center(0) + self.center(1)) if k == 1 else self.center(k - 1)
            b = self.center(k)
            return a + (b - a) * s
        if ph[0] == "ss":
            return self.center(ph[1] - 1)
        n = self.n
        a = self.center(n - 1); b = 0.5 * (self.center(n - 1) + self.center(n - 2))
        return a + (b - a) * mjerk(ph[2])


def make_steps(n_steps, stride, first="R", yaw_step=0.0, width=0.08, ramp=2, stride_fn=None):
    """평지 직진 · 제자리(stride=0) · 제자리 돌기(yaw_step) 발자국. 첫 디딤발 = first, 처음 흔드는 발 = 반대.
    stride = 걸음마다 앞 발이 나아가는 거리(앞 발 기준). 속도 = stride / (Ts + Td)."""
    R0 = dict(side="R", pos=np.array([0.0, -width / 2]), yaw=0.0); L0 = dict(side="L", pos=np.array([0.0, width / 2]), yaw=0.0)
    steps = [L0, R0] if first == "R" else [R0, L0]       # steps[1] = 첫 디딤발, steps[0] = 처음 흔드는 발
    yaw = 0.0; x = 0.0
    side = steps[0]["side"]
    for i in range(n_steps):
        sc = min(1.0, (i + 1) / (ramp + 1), (n_steps - 1 - i) / (ramp + 1)) if ramp else 1.0   # 출발 · 멈춤 때 보폭을 줄였다 늘림
        if i >= n_steps - 1:
            sc = 0.0                                  # 마지막 걸음은 나란히 모으기
        s = stride_fn(i) if stride_fn else stride * sc
        if yaw_step:
            # 왼쪽으로 돌기: 왼발이 먼저 +yaw_step 만큼 벌리고, 오른발이 따라와 나란히
            if (side == "L") == (yaw_step > 0):
                yaw += yaw_step
        x += s
        sg = -1.0 if side == "R" else 1.0
        c = np.array([x, 0.0])
        steps.append(dict(side=side, pos=c + rz(yaw)[:2, :2] @ np.array([0.0, sg * width / 2]), yaw=yaw))
        side = "R" if side == "L" else "L"
    if yaw_step:      # 마지막에 나란히
        sg = -1.0 if side == "R" else 1.0
        steps.append(dict(side=side, pos=np.array([x, 0.0]) + rz(yaw)[:2, :2] @ np.array([0.0, sg * width / 2]), yaw=yaw))
    return steps


# ---------------------------------------------------------------- 전신 역기구학
class WholeBodyIK:
    def __init__(self, m):
        self.m = m; self.d = mujoco.MjData(m); self.ix = Idx(m)
        ix = self.ix
        self.cols = list(range(6)) + [ix.vadr[j] for j in ix.legj]
        self.legq = np.array([ix.qadr[j] for j in ix.legj])
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ix.legj]
        self.lo = m.jnt_range[jid, 0]; self.hi = m.jnt_range[jid, 1]
        self.jp = np.zeros((3, m.nv)); self.jr = np.zeros((3, m.nv)); self.jc = np.zeros((3, m.nv))
        self.has_roll = any(j.endswith("hip_roll") for j in ix.legj)

    def set_aux(self, q, aux):
        """팔 · 크랭크(→머리 · 귀) 관절 값을 qpos 에 넣는다"""
        ix = self.ix
        for k, v in aux.items():
            q[ix.qadr[k]] = v
        if "nod_crank" in aux:
            h, e = loop_qpos(aux["nod_crank"]); q[ix.qadr["head_nod"]] = h; q[ix.qadr["ear_axle"]] = e

    def solve(self, q, tgt, iters=3, lam=1e-4):
        """q: 시작 qpos(제자리에서 바뀜). tgt: feet{s:(pos3, yaw, wpos3, wrot)}, com(2), wcom, pitch, yaw, wori, z, wz"""
        m, d, ix = self.m, self.d, self.ix
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            rows = []; errs = []; ws = []
            for s, (p, yaw, wp, wr) in tgt["feet"].items():
                b = ix.foot[s]
                mujoco.mj_jacBody(m, d, self.jp, self.jr, b)
                e = p - d.xpos[b]
                rows += [self.jp]; errs += [e]; ws += [wp]
                er = rot_err(rz(yaw), d.xmat[b].reshape(3, 3))
                rows += [self.jr]; errs += [er]; ws += [np.full(3, wr)]
                rows = [r.copy() for r in rows]
            mujoco.mj_jacSubtreeCom(m, d, self.jc, ix.torso)
            rows.append(self.jc[:2].copy()); errs.append(tgt["com"] - d.subtree_com[ix.torso][:2]); ws.append(np.full(2, tgt["wcom"]))
            mujoco.mj_jacBody(m, d, self.jp, self.jr, ix.torso)
            Rt = d.xmat[ix.torso].reshape(3, 3)
            yaw_c, pit_c, rol_c = rpy_of(Rt)
            roll_t = 0.0 if self.has_roll else rol_c
            Rd = rz(tgt["yaw"]) @ np.array([[math.cos(tgt["pitch"]), 0, math.sin(tgt["pitch"])], [0, 1, 0],
                                             [-math.sin(tgt["pitch"]), 0, math.cos(tgt["pitch"])]]) @ \
                np.array([[1, 0, 0], [0, math.cos(roll_t), -math.sin(roll_t)], [0, math.sin(roll_t), math.cos(roll_t)]])
            er = rot_err(Rd, Rt)
            if self.has_roll:
                rows.append(self.jr.copy()); errs.append(er); ws.append(np.full(3, tgt["wori"]))
            else:
                ay = rz(yaw_c)[:, 1]
                rows.append(np.vstack([self.jr[2], ay @ self.jr])); errs.append(np.array([er[2], ay @ er])); ws.append(np.full(2, tgt["wori"]))
            rows.append(self.jp[2:3].copy()); errs.append(np.array([tgt["z"] - d.xpos[ix.torso][2]])); ws.append(np.array([tgt["wz"]]))
            J = np.vstack(rows)[:, self.cols]; e = np.concatenate(errs); w = np.concatenate(ws)
            JW = J.T * w
            dq = np.linalg.solve(JW @ J + lam * np.eye(len(self.cols)), JW @ e)
            nrm = np.abs(dq).max()
            if nrm > 0.1:
                dq *= 0.1 / nrm
            v = np.zeros(m.nv); v[self.cols] = dq
            mujoco.mj_integratePos(m, q, v, 1.0)
            q[self.legq] = np.clip(q[self.legq], self.lo, self.hi)
        return q

    def static_torque(self, q, loads):
        """준정적 관절 토크(액추에이터 관절 qfrc 성분): g(q) − Σ Jᵀ F. loads: [(body, point3, F3)]"""
        m, d = self.m, self.d
        d.qpos[:] = q; d.qvel[:] = 0
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        res = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, res)
        for b, pt, F in loads:
            mujoco.mj_jac(m, d, self.jp, None, pt, b)
            res -= self.jp.T @ F
        return res


# ---------------------------------------------------------------- 제어기
class Controller:
    def __init__(self, m_plan, kp, dt, plan, crouch=0.015, lift=0.015, k_dcm=2.0, k_step=0.0, ff=True,
                 posture=(0.0, 0.0, 0.0, 0.0), head_fn=None, vel_fc=12.0, zc=None, pitch=0.0, com_bias=(0.0, 0.0),
                 dcm_to_com=0.0, land_press=0.0, arm=0.0, dyn_ff=True, lead_s=0.0, k_lat=0.0, lat_db=0.0, lat_in=True):
        self.m = m_plan; self.ik = WholeBodyIK(m_plan); self.ix = self.ik.ix
        self.kp = kp; self.dt = dt; self.plan = plan
        self.crouch, self.lift, self.k_dcm, self.k_step, self.ff = crouch, lift, k_dcm, k_step, ff
        self.posture = posture            # (K 피치→고관절, D 피치, K 롤→발목 롤, D 롤)
        self.head_fn = head_fn or (lambda t: 0.0)
        self.pitch = pitch; self.com_bias = np.array(com_bias); self.dcm_to_com = dcm_to_com
        self.land_press = land_press; self.arm = arm; self.vel_fc = vel_fc
        self.dyn_ff, self.lead_s = dyn_ff, lead_s
        self.k_lat = k_lat; self.lat_off = 0.0; self.lat_i = None
        self.lat_db = lat_db; self.lat_in = lat_in
        ix = self.ix
        q0 = m_plan.qpos0.copy(); q0[3:7] = [1, 0, 0, 0]
        self.zb = float(q0[2])           # 시작 몸통 높이(다리를 늘린 변형은 0 이 아님)
        d = self.ik.d; d.qpos[:] = q0; mujoco.mj_kinematics(m_plan, d); mujoco.mj_comPos(m_plan, d)
        self.foot0 = {s: d.xpos[ix.foot[s]].copy() for s in "RL"}
        self.com0 = d.subtree_com[ix.torso].copy()
        # 무릎을 편 자세는 역기구학 특이점: 무릎을 조금 굽힌 자세에서 시작해 두 발 평평 · 무게중심 그대로로 맞춘다
        for s in "RL":
            q0[ix.qadr[f"{s}_hip_pitch"]] = -0.1; q0[ix.qadr[f"{s}_knee"]] = 0.2; q0[ix.qadr[f"{s}_ankle_pitch"]] = -0.1
        self.q = q0
        tgt0 = dict(feet={s: (self.foot0[s], 0.0, np.full(3, 100.0), 100.0) for s in "RL"}, com=self.com0[:2], wcom=10.0,
                    pitch=0.0, yaw=0.0, wori=10.0, z=self.zb - 0.002, wz=3.0)
        self.q = self.ik.solve(self.q, tgt0, iters=60)
        d.qpos[:] = self.q; mujoco.mj_kinematics(m_plan, d); self.z0 = float(d.xpos[ix.torso][2])
        zc = zc or (self.com0[2] - crouch)
        self.omega = math.sqrt(G / zc)
        self.pre = Preview(dt, zc)
        self.pre.reset(self.com0[:2] + self.com_bias)
        self.t = 0.0
        # 추정
        self.odo = {s: self.foot0[s][:2].copy() for s in "RL"}
        self.odo_yaw = {"R": 0.0, "L": 0.0}
        self.support = "R"; self.c_est = None; self.cd_est = np.zeros(2); self.c_prev = None
        self.swing_start = None; self.swing_i = None
        self.dq_ext = np.zeros(2)          # 밀기 대응 발 디딤 이동(누적)
        self.log = {}
        self.act_j = ix.act_j; self.q_start = None
        self.qj_prev = None; self.qd_ik = np.zeros(len(ix.act_j)); self.qdd_ik = np.zeros(len(ix.act_j))

    # ---- 추정
    def estimate(self, sen):
        m, d, ix = self.m, self.ik.d, self.ix
        q = self.m.qpos0.copy()
        q[3:7] = sen["quat"]
        for j, v in zip(ix.act_j, sen["q"]):
            q[ix.qadr[j]] = v
        h, e = loop_qpos(q[ix.qadr["nod_crank"]]); q[ix.qadr["head_nod"]] = h; q[ix.qadr["ear_axle"]] = e
        d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        pk = {s: d.xpos[ix.foot[s]][:2].copy() for s in "RL"}
        yk = {s: rpy_of(d.xmat[ix.foot[s]].reshape(3, 3))[0] for s in "RL"}
        fz = {s: sen["feet"][s]["fz"] for s in "RL"}
        tot = fz["R"] + fz["L"]
        # 디딤발 판정: 한 발이 전체 힘의 65 % 를 넘을 때만 바꾼다(흔드는 발이 스칠 때 깜빡임 방지)
        new = self.support
        if tot > 2.0:
            for s_ in "RL":
                if fz[s_] > 0.65 * tot:
                    new = s_
        if new != self.support:
            old = self.support
            self.odo[new] = self.odo[old] + (pk[new] - pk[old])
            self.odo_yaw[new] = self.odo_yaw[old] + (yk[new] - yk[old])
            self.support = new
        st = self.support
        # 디딤발 좌표로 옮길 때 IMU 요 대신 디딤발 요(주행 거리계) 기준
        c = self.odo[st] + rz(self.odo_yaw[st] - yk[st])[:2, :2] @ (d.subtree_com[ix.torso][:2] - pk[st])
        other = "L" if st == "R" else "R"
        self.odo_other = self.odo[st] + rz(self.odo_yaw[st] - yk[st])[:2, :2] @ (pk[other] - pk[st])
        self.yaw_other = self.odo_yaw[st] + (yk[other] - yk[st])
        a = 1.0 - math.exp(-2 * math.pi * self.vel_fc * self.dt) if hasattr(self, "vel_fc") else 0.4
        if self.c_prev is not None:
            v = (c - self.c_prev) / self.dt
            self.cd_est = self.cd_est + a * (v - self.cd_est)
        self.c_prev = c.copy(); self.c_est = c
        self.Rt_meas = d.xmat[ix.torso].reshape(3, 3).copy()
        self.fz = fz
        return c

    def step(self, t, sen):
        self.t = t; ix = self.ix; plan = self.plan
        c = self.estimate(sen)
        N = self.pre.N
        fut = np.array([plan.zmp(t + k * self.dt) for k in range(N)])
        c_ref, cd_ref, cdd_ref, p_ref = self.pre.step(fut)
        c_ref = c_ref - self.com_bias * 0       # bias 는 reset 에 들어 있음
        ph = plan.phase(t)
        # ---- DCM 되먹임
        xi = c + self.cd_est / self.omega; xi_r = c_ref + cd_ref / self.omega
        exi = xi - xi_r
        p_d = p_ref + self.k_dcm * exi
        # ---- 발 목표
        feet = {}
        crouch = (self.zb - self.z0) + (self.crouch - (self.zb - self.z0)) * min(1.0, max(0.0, t / 1.0))   # 시작 높이 기준 웅크림
        z_foot = self.foot0["R"][2]
        W_ST = (100, 100, 100); wr = 100
        if ph[0] in ("pre", "ds", "end"):
            if ph[0] == "pre" or (ph[0] == "ds" and ph[1] == 1):
                si = {"R": 0 if plan.steps[0]["side"] == "R" else 1, "L": 1 if plan.steps[0]["side"] == "R" else 0}
            elif ph[0] == "ds":
                k = ph[1]; si = {plan.steps[k]["side"]: k, plan.steps[k - 1]["side"]: k - 1}
            else:
                n = plan.n; si = {plan.steps[n - 1]["side"]: n - 1, plan.steps[n - 2]["side"]: n - 2}
            for s in "RL":
                st = plan.steps[si[s]]
                feet[s] = (np.array([*st["pos"], z_foot - self.land_press]), st["yaw"], np.array([100, 10, 100.0]), wr)
            self.swing_start = None
            sup_poly = [plan.center(si["R"]), plan.center(si["L"])]
        else:
            i, s = ph[1], ph[2]
            sw = plan.steps[i]["side"]; stn = plan.steps[i - 1]
            if self.swing_i != i:
                self.swing_i = i
                prev = plan.steps[i - 2]
                self.swing_start = (prev["pos"].copy(), prev["yaw"])
                # 착지 위치를 실제(운동학)로 고쳐 둔다: 디딤발의 계획 위치 = 추정 위치
                if self.support == stn["side"]:
                    stn["pos"] = self.odo[stn["side"]].copy()
            # 밀기 대응: 흔드는 동안 앞뒤 DCM 오차로 착지점 이동(남은 걸음 모두 같이)
            if self.k_step and s < 0.6:
                dx = self.k_step * exi
                dx[1] = 0.0
                dx = np.clip(dx, -0.05, 0.05)
                inc = dx - self.dq_ext
                if np.linalg.norm(inc) > 1e-4:
                    for j in range(i, plan.n):
                        plan.steps[j]["pos"] = plan.steps[j]["pos"] + inc
                    self.dq_ext = dx
            p0, y0 = self.swing_start; tg = plan.steps[i]
            a = mjerk(s)
            pos = p0 + (tg["pos"] - p0) * a
            z = z_foot + self.lift * (1 - math.cos(2 * math.pi * s)) / 2
            yaw = y0 + (tg["yaw"] - y0) * a
            feet[stn["side"]] = (np.array([*stn["pos"], z_foot]), stn["yaw"], np.array(W_ST, float), wr)
            # 옆 발 디딤: 흔드는 발 착지점을 DCM 옆 오차만큼 옮긴다(바깥 +30 mm, 안쪽 −5 mm 까지; 계획은 착지 때만 고침)
            if self.k_lat:
                if self.lat_i != i:
                    self.lat_i = i; self.lat_off = 0.0; self.lat_done = False
                sgn = 1.0 if sw == "L" else -1.0
                if s < 0.7:
                    e_ = sgn * exi[1]                               # 흔드는 발 바깥쪽으로 + 인 DCM 오차
                    e_ = max(0.0, abs(e_) - self.lat_db) * (1 if e_ > 0 else -1)
                    lo_ = -0.005 if self.lat_in else 0.0
                    outw = float(np.clip(self.k_lat * e_, lo_, 0.03))
                    self.lat_off = sgn * outw
                pos = pos + np.array([0.0, self.lat_off * a])
                if s > 0.95 and not self.lat_done:
                    plan.steps[i]["pos"] = plan.steps[i]["pos"] + np.array([0.0, self.lat_off]); self.lat_done = True
            wy = 50.0 if self.ik.has_roll else 5.0
            feet[sw] = (np.array([*pos, z]), yaw, np.array([50, wy, 50.0]), 50)
            sup_poly = [plan.center(i - 1)]
        if ph[0] == "end" and ph[2] >= 1.0:
            self.dq_ext[:] = 0
        # ---- p_d 를 지지 영역 안으로(지지 발 가운데 사이 선분 근처, 발 반폭 안)
        p_d = self.clip_support(p_d, sup_poly)
        # ---- 역기구학
        body_yaw = 0.5 * (feet["R"][1] + feet["L"][1])
        com_t = c_ref + self.dcm_to_com * exi
        tgt = dict(feet=feet, com=com_t, wcom=10.0, pitch=self.pitch, yaw=body_yaw, wori=10.0,
                   z=self.zb - crouch, wz=3.0)
        aux = {"nod_crank": crank_for_head(self.head_fn(t)), "R_arm_swing": 0.0, "L_arm_swing": 0.0}
        self.ik.set_aux(self.q, aux)
        self.q = self.ik.solve(self.q, tgt)
        qcmd = {j: self.q[ix.qadr[j]] for j in ix.act_j}
        # ---- 서보 휨 보상(준정적 토크 / kp)
        if self.ff:
            mass = ix.mass
            F = np.array([*(mass * self.omega ** 2 * (com_t - p_d)), mass * G])
            loads = []
            d = self.ik.d
            if len(sup_poly) == 2:
                cR = sup_poly[0]; cL = sup_poly[1]; L = cL - cR
                aL = float(np.clip((p_d - cR) @ L / max(L @ L, 1e-9), 0, 1))
                base = cR + aL * L; off = p_d - base
                for s, wgt, cc in (("R", 1 - aL, cR), ("L", aL, cL)):
                    loads.append((ix.foot[s], np.array([*(cc + off), 0.0]), wgt * F))
            else:
                loads.append((ix.foot[plan.steps[ph[1] - 1]["side"]], np.array([*p_d, 0.0]), F))
            tau = self.ik.static_torque(self.q.copy(), loads)
            # 크랭크: 머리 · 귀 중력 토크를 고리 기울기로 옮겨 더함
            from kw_common import PV, LOOP
            al = qcmd["nod_crank"]
            dH = PV(al, np.polynomial.polynomial.polyder(LOOP["polycoef_head"]))
            dE = PV(al, np.polynomial.polynomial.polyder(LOOP["polycoef_ear"]))
            tc = tau[ix.vadr["nod_crank"]] + dH * tau[ix.vadr["head_nod"]] + dE * tau[ix.vadr["ear_axle"]]
            for k, j in enumerate(ix.act_j):
                tj = tc if j == "nod_crank" else tau[ix.vadr[j]]
                qcmd[j] = qcmd[j] + tj / self.kp[k]
        # ---- 동역학 앞먹임: 감쇠(역기전력) b·q̇ + 회전자 관성 a·q̈ (Dynamixel 속도 · 가속도 앞먹임 레지스터에 해당)
        qj = np.array([self.q[ix.qadr[j]] for j in ix.act_j])
        if self.qj_prev is not None:
            v = (qj - self.qj_prev) / self.dt
            al = 1.0 - math.exp(-2 * math.pi * 25.0 * self.dt)
            acc = (v - self.qd_ik) / self.dt
            self.qd_ik = self.qd_ik + al * (v - self.qd_ik)
            self.qdd_ik = self.qdd_ik + al * (acc - self.qdd_ik)
        self.qj_prev = qj
        if self.dyn_ff:
            for k, j in enumerate(ix.act_j):
                dv = ix.vadr[j]
                tl = self.m.dof_damping[dv] * self.qd_ik[k] + self.m.dof_armature[dv] * self.qdd_ik[k]
                qcmd[j] = qcmd[j] + tl / self.kp[k] + self.lead_s * self.qd_ik[k]
        # ---- 몸통 자세 되먹임(IMU)
        Kp_, Dp_, Kr_, Dr_ = self.posture
        if any(self.posture):
            d = self.ik.d
            Rik = None
            yi, pi, ri = rpy_of(self._ik_torso_R())
            ym, pm, rm = rpy_of(self.Rt_meas)
            ep, er = pm - pi, rm - ri
            gy = sen["gyro"]
            st = self.support
            dp = Kp_ * ep + Dp_ * gy[1]
            dr = Kr_ * er + Dr_ * gy[0]
            sides = "RL" if ph[0] in ("pre", "ds", "end") else st
            for s in sides:
                qcmd[f"{s}_hip_pitch"] += dp
                qcmd[f"{s}_ankle_roll"] -= dr
        d_ = self.ik.d; d_.qpos[:] = self.q; mujoco.mj_kinematics(self.m, d_); mujoco.mj_comPos(self.m, d_)
        ik_rpy = rpy_of(d_.xmat[ix.torso].reshape(3, 3)); ik_com = d_.subtree_com[ix.torso][:2].copy()
        self.log = dict(ik_roll=ik_rpy[2], ik_pitch=ik_rpy[1], ik_com=ik_com, meas_roll=rpy_of(self.Rt_meas)[2], c=c, cd=self.cd_est.copy(), c_ref=c_ref, p_ref=p_ref, p_d=p_d, xi=xi, xi_r=xi_r,
                        phase=ph[0], support=self.support)
        out = np.array([qcmd[j] for j in ix.act_j])
        # 시작: 실제 자세(영점)에서 첫 명령까지 0.5 초에 걸쳐 섞는다(무릎 굽히는 계단 방지)
        if self.q_start is None:
            self.q_start = sen["q"].copy()
        if t < 0.5:
            a = mjerk(t / 0.5); out = self.q_start + (out - self.q_start) * a
        lo = self.m.actuator_ctrlrange[:, 0]; hi = self.m.actuator_ctrlrange[:, 1]
        return np.clip(out, lo, hi)

    def _ik_torso_R(self):
        d = self.ik.d; d.qpos[:] = self.q; mujoco.mj_kinematics(self.m, d)
        return d.xmat[self.ix.torso].reshape(3, 3).copy()

    def clip_support(self, p, centers, half=(0.045, 0.018)):
        """지지 발 가운데(들)를 중심으로 한 사각형(발 좌표 앞뒤 half[0], 옆 half[1]) 또는 두 발이면 그 사이 볼록 영역으로 자른다(근사)."""
        if len(centers) == 1:
            c = centers[0]
            return np.array([np.clip(p[0], c[0] - half[0], c[0] + half[0]), np.clip(p[1], c[1] - half[1], c[1] + half[1])])
        a, b = centers
        L = b - a; s = float(np.clip((p - a) @ L / max(L @ L, 1e-9), 0, 1)); base = a + s * L
        off = p - base
        return base + np.array([np.clip(off[0], -half[0], half[0]), np.clip(off[1], -half[1], half[1])])
