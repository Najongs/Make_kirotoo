import os; os.environ["BUILD"] = "build7"
import render6 as r
walk = {"head_pan": 25, "head_tilt": 12, "R_hip_pitch": -28, "R_knee": 40, "R_ankle_pitch": -12,
        "L_hip_pitch": 12, "L_knee": 8, "L_ankle_pitch": -20}
r.panel([r.render(15, -35), r.render(0, 0), r.render(0, -90), r.render(12, -55, walk)],
        "/home/claude/biped/out7/biped_v7_preview.png")
sk = ("torso_cover", "lid", "_shell", "face_plate", "eye_", "mouth", "arm_", "ear_", "antenna", "crest")
sk2 = ("torso_cover", "lid", "face_plate", "eye_", "mouth", "arm_", "antenna", "crest")
r.panel([r.render(20, -40, skip=("torso_cover", "lid", "arm_", "thigh_shell", "shank_shell")),
         r.render(20, -150, skip=("torso_cover", "lid", "arm_", "thigh_shell", "shank_shell", "face_plate", "eye_", "mouth")),
         r.render(10, -60, skip=("lid",))],
        "/home/claude/biped/out7/biped_v7_inside.png")
print("ok")
