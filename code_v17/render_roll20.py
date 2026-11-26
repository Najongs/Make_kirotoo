"""front views of the feet / shanks for the inward hip roll change (same viewpoint before / after).
usage: [FOOT_OUT_MM=0 SHANK_SHELL_IN=40] python render_roll20.py <tag> -> ../out20/range_figs/roll_<tag>.png"""
import os, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "after"
os.environ.setdefault("BUILD", "build20")
import render20 as R20
r = R20.r
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out20", "range_figs"); os.makedirs(out, exist_ok=True)
poses = [{}, {"R_hip_roll": 5.0, "R_ankle_roll": -5.0}, {"R_hip_roll": 2.5, "L_hip_roll": -2.5, "R_ankle_roll": -2.5, "L_ankle_roll": 2.5}]
r.panel([r.render(0, 0, p, scale=2.6) for p in poses], os.path.join(out, f"roll_{tag}.png"))
print("saved", tag)
