"""same-viewpoint before / after views for the v20 range changes.
usage: BUILD=build20 python render_range20.py <tag>   -> ../out20/range_figs/legs_<tag>.png"""
import os, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "after"
os.environ.setdefault("BUILD", "build20")
import render20 as R20
r = R20.r
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out20", "range_figs"); os.makedirs(out, exist_ok=True)
poses = [({}, "zero"),
         ({"R_hip_pitch": 20, "L_hip_pitch": -60, "L_knee": 60}, "hip back / forward"),
         ({"R_knee": 100, "L_hip_yaw": 20, "R_hip_yaw": -20}, "knee 100, yaw"),
         ({"R_hip_roll": -15, "L_hip_roll": -15, "R_ankle_roll": 15, "L_ankle_roll": 15}, "roll sway")]
imgs = [r.render(0, -90, p, scale=2.6) for p, _ in poses[:3]] + [r.render(5, 160, poses[3][0], scale=2.6)]
r.panel(imgs, os.path.join(out, f"legs_{tag}.png"))
print("saved", tag)
