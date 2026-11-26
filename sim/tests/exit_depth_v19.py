"""free space outside each servo cable slot (right leg + right yaw): march from the case side face outward through the
slot window and report the first solid met (wall, shell ...) and the free distance, at 5 points of the slot."""
import os, sys, json, numpy as np
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(SIM, "..", "code_v17"))
import build19 as B, build4, geom4
links, joints, _ = B.build_model()
SKIP, frames = B.servo_step_parts(links)
SOL = [(ln, pn, s, np.array(build4.aabb(build4.sampled(s)), float)) for ln, L in links.items() for pn, ss in L["parts"].items() if pn not in SKIP for s in ss]
out = {}
for sv in ["R_ankle_roll_servo_XM430", "R_ankle_pitch_servo_XC430", "R_knee_servo_XM430", "R_hip_pitch_servo_XM430", "yaw_servo_R_XC430"]:
    fr = frames[sv]; R = np.c_[fr["X"], fr["Y"], fr["Z"]]; out[sv] = {}
    for sg in (1, -1):
        rows = []
        for ys_, zs_ in ((-14.1, -14.0), (-9.5, -12.0), (-18.5, -12.0), (-14.1, -16.0), (-14.1, -11.5)):
            hit = None; d = 0.0
            while d < 40:
                q = fr["O"] + R @ np.array([sg * (14.25 + d), ys_, zs_])
                for ln, pn, s, bb in SOL:
                    if np.all(q >= bb[:, 0]) and np.all(q <= bb[:, 1]) and geom4.inside(s, q[None])[0]:
                        hit = (round(d, 1), pn); break
                if hit: break
                d += 0.25
            rows.append(hit or (40.0, "free"))
        out[sv]["+x" if sg > 0 else "-x"] = rows
        print(sv, "+x" if sg > 0 else "-x", rows, flush=True)
json.dump(out, open(os.path.join(SIM, "..", "out19", "assembly_check", "exit_depth.json"), "w"), indent=1)
