"""figure for ASSEMBLY_CHECK_v19: right leg, real servo points, screw driver axes (green clear / orange later part /
red blocked), open cable slot sides (arrows). Run after assembly_check_v20.py."""
import os, sys, json, math, numpy as np
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(SIM, "..", "code_v17")); sys.path.insert(0, os.path.join(SIM, "tests"))
import build20 as B, build4, geom4, x430_geom as XG, x430_holes as X
from common import korean_font
links, joints, _ = B.build_model()
SKIP, frames = B.servo_step_parts(links)
R = json.load(open(os.path.join(SIM, "..", "out20", "assembly_check", "assembly_check_v20.json")))
P_s, G_s, GN = XG.points()
RIGHT = ["R_ankle_roll_servo_XM430", "R_ankle_pitch_servo_XC430", "R_knee_servo_XM430", "R_hip_pitch_servo_XM430", "R_hip_roll_servo_XM430", "yaw_servo_R_XL430"]
LEG = ["R_hip_yaw_link", "R_hip_roll_link", "R_thigh", "R_shank", "R_ankle_link", "R_foot"]
pts = []
for ln in LEG:
    for pn, ss in links[ln]["parts"].items():
        if pn in SKIP: continue
        for s in ss:
            q = geom4.samples(s, 2.0)
            if len(q): pts.append(q)
pts = np.vstack(pts)
plt = korean_font()
fig, ax = plt.subplots(1, 2, figsize=(13, 9))
for a, (i, j, lab) in zip(ax, ((0, 2, "옆에서 (x 앞, z 위)"), (1, 2, "앞에서 (y 왼쪽, z 위)"))):
    a.scatter(pts[:, i], pts[:, j], s=0.3, c="#bbbbbb")
    for sv in RIGHT:
        fr = frames[sv]; Rm = np.c_[fr["X"], fr["Y"], fr["Z"]]
        keep = np.isin(G_s, [GN.index("case"), GN.index("horn"), GN.index("idler")])
        Q = P_s[keep][::25] @ Rm.T + fr["O"]
        a.scatter(Q[:, i], Q[:, j], s=0.5, c="#2a6fdb")
    for e in R["E_driver"]:
        fr = frames[e["servo"]]; Rm = np.c_[fr["X"], fr["Y"], fr["Z"]]
        if e["kind"].startswith("case"):
            p, n = X.HOLES[e["n"] - 1]; pw, nw = X.to_world(fr, p, n)
        else:
            hx, hy = ([(8.0, 0.0), (-8.0, 0.0), (0.0, 8.0), (0.0, -8.0)] if "hip_roll" in e["servo"] else [(5.657, 5.657), (-5.657, 5.657), (5.657, -5.657), (-5.657, -5.657)])[e["n"] - 1]
            pw = fr["O"] + Rm @ np.array([hx, hy, 20.9]); nw = fr["Z"]
        col = "#c0392b" if not e["clear"] else ("#e07b39" if e["later_parts"] else "#27ae60")
        q = np.array([pw, pw + 25 * np.asarray(nw)])
        a.plot(q[:, i], q[:, j], color=col, lw=1.2)
    a.set_aspect("equal"); a.set_title(lab, fontsize=10); a.grid(alpha=0.3)
    if j == 2: a.set_ylabel("z (mm)")
ax[0].set_xlabel("x (mm)"); ax[1].set_xlabel("y (mm)")
nb = sum(1 for e in R["E_driver"] if not e["clear"]); nl = sum(1 for e in R["E_driver"] if e["clear"] and e["later_parts"])
fig.suptitle(f"오른 다리 서보 나사 {len(R['E_driver'])}개 중 드라이버 길(지름 6 mm × 60 mm)이 막힌 것 {nb}개(빨강), 나중 부품만 가리는 것 {nl}개(주황), 열림(초록)\n"
             "(회색 = 출력물, 파랑 = 공식 STEP 의 실제 서보 형상, 선 = 나사 축 25 mm, 영점 자세, 설계 형상 계산)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(SIM, "..", "out20", "assembly_check", "right_leg_screw_access.png"), dpi=110)
print("saved")
