# v19 servo seat log

SERVO_CLR 0.15 mm, CASE_HOLE_D 2.9 mm, HORN_HOLE_D 2.2 mm

## X430 case holes (official STEP, servo frame)

| # | x | y | z | face (outward normal) |
|---|---|---|---|---|
| 1 | -14.25 | -28.00 | -6.00 | side -x |
| 2 | -14.25 | -28.00 | +6.00 | side -x |
| 3 | -14.25 | -4.00 | -6.00 | side -x |
| 4 | -14.25 | -4.00 | +6.00 | side -x |
| 5 | -8.00 | -35.25 | -6.00 | back -y (cable end) |
| 6 | -8.00 | -35.25 | +6.00 | back -y (cable end) |
| 7 | -8.00 | +11.25 | +6.00 | front +y (horn end) |
| 8 | +8.00 | -35.25 | -6.00 | back -y (cable end) |
| 9 | +8.00 | -35.25 | +6.00 | back -y (cable end) |
| 10 | +8.00 | +11.25 | +6.00 | front +y (horn end) |
| 11 | +14.25 | -28.00 | -6.00 | side +x |
| 12 | +14.25 | -28.00 | +6.00 | side +x |
| 13 | +14.25 | -4.00 | -6.00 | side +x |
| 14 | +14.25 | -4.00 | +6.00 | side +x |

## gaps between servo case faces and printed parts (same link), before -> after

| servo | face | part | gap before (mm) | gap after (mm) | face covered | in contact before |
|---|---|---|---|---|---|---|
| yaw_servo_R_XC430 | -x | torso_frame | 0.775 | 0.775 | 71% | 0% |
| yaw_servo_R_XC430 | +z(horn) | torso_frame | 1.025 | 1.025 | 41% | 0% |
| yaw_servo_L_XC430 | +x | torso_frame | 0.775 | 0.775 | 71% | 0% |
| yaw_servo_L_XC430 | +z(horn) | torso_frame | 1.025 | 1.025 | 41% | 0% |
| R_hip_pitch_servo_XM430 | +x | R_hip_yaw_bracket | 0.275 | 0.425 | 70% | 0% |
| R_hip_pitch_servo_XM430 | -x | R_hip_yaw_bracket | 0.275 | 0.425 | 70% | 0% |
| R_knee_servo_XM430 | +x | R_thigh_frame | 0.275 | 0.425 | 93% | 0% |
| R_knee_servo_XM430 | -x | R_thigh_frame | 0.275 | 0.425 | 93% | 0% |
| R_ankle_pitch_servo_XC430 | +x | R_shank_frame | 0.275 | 0.425 | 81% | 0% |
| R_ankle_pitch_servo_XC430 | -x | R_shank_frame | 0.275 | 0.425 | 81% | 0% |
| R_ankle_roll_servo_XM430 | -z(idler) | R_ankle_bracket | 0.025 | 0.025 | 22% | 22% |
| L_hip_pitch_servo_XM430 | +x | L_hip_yaw_bracket | 0.275 | 0.425 | 70% | 0% |
| L_hip_pitch_servo_XM430 | -x | L_hip_yaw_bracket | 0.275 | 0.425 | 70% | 0% |
| L_knee_servo_XM430 | +x | L_thigh_frame | 0.275 | 0.425 | 93% | 0% |
| L_knee_servo_XM430 | -x | L_thigh_frame | 0.275 | 0.425 | 93% | 0% |
| L_ankle_pitch_servo_XC430 | +x | L_shank_frame | 0.275 | 0.425 | 81% | 0% |
| L_ankle_pitch_servo_XC430 | -x | L_shank_frame | 0.275 | 0.425 | 81% | 0% |
| L_ankle_roll_servo_XM430 | -z(idler) | L_ankle_bracket | 0.025 | 0.025 | 22% | 22% |
| head_nod_servo_XL430 | +x | head_back | 0.025 | 0.025 | 93% | 91% |
| arm_R_servo_XC430 | +x | arm_R_shell_out | 1.650 | 1.650 | 3% | 0% |
| arm_R_servo_XC430 | -x | arm_R_shell_out | 1.650 | 1.650 | 3% | 0% |
| arm_R_servo_XC430 | +z(horn) | arm_R_shell_in | 0.425 | 0.425 | 3% | 0% |
| arm_R_servo_XC430 | +z(horn) | arm_R_finger_rod | 0.425 | 0.425 | 8% | 0% |
| arm_R_servo_XC430 | -z(idler) | arm_R_shell_out | 1.700 | 1.700 | 7% | 0% |
| arm_L_servo_XC430 | +x | arm_L_shell_out | 1.650 | 1.650 | 3% | 0% |
| arm_L_servo_XC430 | -x | arm_L_shell_out | 1.650 | 1.650 | 3% | 0% |
| arm_L_servo_XC430 | +z(horn) | arm_L_shell_in | 0.400 | 0.400 | 3% | 0% |
| arm_L_servo_XC430 | +z(horn) | arm_L_finger_rod | 0.400 | 0.400 | 8% | 0% |
| arm_L_servo_XC430 | -z(idler) | arm_L_shell_out | 1.700 | 1.700 | 7% | 0% |

## pockets widened

- R_hip_pitch_servo_XM430 +x R_hip_yaw_bracket[2]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- R_hip_pitch_servo_XM430 -x R_hip_yaw_bracket[3]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- R_knee_servo_XM430 +x R_thigh_frame[5]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- R_knee_servo_XM430 +x R_thigh_frame[7]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- R_knee_servo_XM430 -x R_thigh_frame[3]: case-side face moved 0.15 mm (plate 2.00 -> 1.85 mm)
- R_knee_servo_XM430 -x R_thigh_frame[6]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- R_ankle_pitch_servo_XC430 +x R_shank_frame[9]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- R_ankle_pitch_servo_XC430 +x R_shank_frame[10]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- R_ankle_pitch_servo_XC430 -x R_shank_frame[5]: case-side face moved 0.15 mm (plate 2.00 -> 1.85 mm)
- R_ankle_pitch_servo_XC430 -x R_shank_frame[6]: case-side face moved 0.15 mm (plate 2.00 -> 1.85 mm)
- L_hip_pitch_servo_XM430 +x L_hip_yaw_bracket[3]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- L_hip_pitch_servo_XM430 -x L_hip_yaw_bracket[2]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- L_knee_servo_XM430 +x L_thigh_frame[3]: case-side face moved 0.15 mm (plate 2.00 -> 1.85 mm)
- L_knee_servo_XM430 +x L_thigh_frame[6]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- L_knee_servo_XM430 -x L_thigh_frame[5]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- L_knee_servo_XM430 -x L_thigh_frame[7]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- L_ankle_pitch_servo_XC430 +x L_shank_frame[5]: case-side face moved 0.15 mm (plate 2.00 -> 1.85 mm)
- L_ankle_pitch_servo_XC430 +x L_shank_frame[6]: case-side face moved 0.15 mm (plate 2.00 -> 1.85 mm)
- L_ankle_pitch_servo_XC430 -x L_shank_frame[9]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)
- L_ankle_pitch_servo_XC430 -x L_shank_frame[10]: case-side face moved 0.15 mm (plate 4.00 -> 3.85 mm)

## case holes cut (84)

| servo | hole # | part | axis | u, v | plate (mm) | wall left (mm) |
|---|---|---|---|---|---|---|
| yaw_servo_R_XC430 | 1 | torso_frame | y | 259.0, -28.0 | 5.0 | 1.5 |
| yaw_servo_R_XC430 | 2 | torso_frame | y | 247.0, -28.0 | 5.0 | 1.6 |
| yaw_servo_L_XC430 | 11 | torso_frame | y | 259.0, -28.0 | 5.0 | 1.5 |
| yaw_servo_L_XC430 | 12 | torso_frame | y | 247.0, -28.0 | 5.0 | 1.6 |
| R_hip_pitch_servo_XM430 | 3 | R_hip_yaw_bracket | x | -34.0, 181.0 | 3.85 | 3.9 |
| R_hip_pitch_servo_XM430 | 4 | R_hip_yaw_bracket | x | -46.0, 181.0 | 3.85 | 6.0 |
| R_hip_pitch_servo_XM430 | 13 | R_hip_yaw_bracket | x | -34.0, 181.0 | 3.85 | 3.9 |
| R_hip_pitch_servo_XM430 | 14 | R_hip_yaw_bracket | x | -46.0, 181.0 | 3.85 | 6.0 |
| R_knee_servo_XM430 | 1 | R_thigh_frame | x | -34.0, 138.0 | 1.85 | 6.0 |
| R_knee_servo_XM430 | 1 | R_thigh_frame | x | -34.0, 138.0 | 2.0 | 0.5 |
| R_knee_servo_XM430 | 2 | R_thigh_frame | x | -46.0, 138.0 | 1.85 | 6.0 |
| R_knee_servo_XM430 | 2 | R_thigh_frame | x | -46.0, 138.0 | 2.0 | 0.5 |
| R_knee_servo_XM430 | 3 | R_thigh_frame | x | -34.0, 114.0 | 1.85 | 3.9 |
| R_knee_servo_XM430 | 3 | R_thigh_frame | x | -34.0, 114.0 | 2.0 | 3.9 |
| R_knee_servo_XM430 | 4 | R_thigh_frame | x | -46.0, 114.0 | 1.85 | 6.0 |
| R_knee_servo_XM430 | 4 | R_thigh_frame | x | -46.0, 114.0 | 2.0 | 6.0 |
| R_knee_servo_XM430 | 11 | R_thigh_frame | x | -34.0, 138.0 | 3.85 | 6.0 |
| R_knee_servo_XM430 | 12 | R_thigh_frame | x | -46.0, 138.0 | 3.85 | 6.0 |
| R_knee_servo_XM430 | 13 | R_thigh_frame | x | -34.0, 114.0 | 3.85 | 3.9 |
| R_knee_servo_XM430 | 14 | R_thigh_frame | x | -46.0, 114.0 | 3.85 | 6.0 |
| R_ankle_pitch_servo_XC430 | 1 | R_shank_frame | x | -34.0, 63.0 | 1.85 | 6.0 |
| R_ankle_pitch_servo_XC430 | 1 | R_shank_frame | x | -34.0, 63.0 | 2.0 | 0.5 |
| R_ankle_pitch_servo_XC430 | 2 | R_shank_frame | x | -46.0, 63.0 | 1.85 | 6.0 |
| R_ankle_pitch_servo_XC430 | 2 | R_shank_frame | x | -46.0, 63.0 | 2.0 | 0.5 |
| R_ankle_pitch_servo_XC430 | 3 | R_shank_frame | x | -34.0, 39.0 | 1.85 | 4.5 |
| R_ankle_pitch_servo_XC430 | 3 | R_shank_frame | x | -34.0, 39.0 | 2.0 | 4.5 |
| R_ankle_pitch_servo_XC430 | 4 | R_shank_frame | x | -46.0, 39.0 | 1.85 | 6.0 |
| R_ankle_pitch_servo_XC430 | 4 | R_shank_frame | x | -46.0, 39.0 | 2.0 | 6.0 |
| R_ankle_pitch_servo_XC430 | 11 | R_shank_frame | x | -34.0, 63.0 | 3.85 | 6.0 |
| R_ankle_pitch_servo_XC430 | 12 | R_shank_frame | x | -46.0, 63.0 | 3.85 | 6.0 |
| R_ankle_pitch_servo_XC430 | 13 | R_shank_frame | x | -34.0, 39.0 | 3.85 | 4.5 |
| R_ankle_pitch_servo_XC430 | 14 | R_shank_frame | x | -46.0, 39.0 | 3.85 | 6.0 |
| R_ankle_roll_servo_XM430 | 3 | R_ankle_bracket | y | 39.0, -57.0 | 8.6 | 2.5 |
| R_ankle_roll_servo_XM430 | 3 | R_ankle_bracket | y | 39.0, -57.0 | 4.0 | 6.0 |
| R_ankle_roll_servo_XM430 | 4 | R_ankle_bracket | y | 39.0, -45.0 | 8.6 | 2.5 |
| R_ankle_roll_servo_XM430 | 4 | R_ankle_bracket | y | 39.0, -45.0 | 4.0 | 6.0 |
| R_ankle_roll_servo_XM430 | 13 | R_ankle_bracket | y | 39.0, -57.0 | 8.6 | 2.5 |
| R_ankle_roll_servo_XM430 | 13 | R_ankle_bracket | y | 39.0, -57.0 | 4.0 | 6.0 |
| R_ankle_roll_servo_XM430 | 14 | R_ankle_bracket | y | 39.0, -45.0 | 8.6 | 2.5 |
| R_ankle_roll_servo_XM430 | 14 | R_ankle_bracket | y | 39.0, -45.0 | 4.0 | 6.0 |
| L_hip_pitch_servo_XM430 | 3 | L_hip_yaw_bracket | x | 34.0, 181.0 | 3.85 | 3.9 |
| L_hip_pitch_servo_XM430 | 4 | L_hip_yaw_bracket | x | 46.0, 181.0 | 3.85 | 6.0 |
| L_hip_pitch_servo_XM430 | 13 | L_hip_yaw_bracket | x | 34.0, 181.0 | 3.85 | 3.9 |
| L_hip_pitch_servo_XM430 | 14 | L_hip_yaw_bracket | x | 46.0, 181.0 | 3.85 | 6.0 |
| L_knee_servo_XM430 | 1 | L_thigh_frame | x | 34.0, 138.0 | 3.85 | 6.0 |
| L_knee_servo_XM430 | 2 | L_thigh_frame | x | 46.0, 138.0 | 3.85 | 6.0 |
| L_knee_servo_XM430 | 3 | L_thigh_frame | x | 34.0, 114.0 | 3.85 | 3.9 |
| L_knee_servo_XM430 | 4 | L_thigh_frame | x | 46.0, 114.0 | 3.85 | 6.0 |
| L_knee_servo_XM430 | 11 | L_thigh_frame | x | 34.0, 138.0 | 1.85 | 6.0 |
| L_knee_servo_XM430 | 11 | L_thigh_frame | x | 34.0, 138.0 | 2.0 | 0.5 |
| L_knee_servo_XM430 | 12 | L_thigh_frame | x | 46.0, 138.0 | 1.85 | 6.0 |
| L_knee_servo_XM430 | 12 | L_thigh_frame | x | 46.0, 138.0 | 2.0 | 0.5 |
| L_knee_servo_XM430 | 13 | L_thigh_frame | x | 34.0, 114.0 | 1.85 | 3.9 |
| L_knee_servo_XM430 | 13 | L_thigh_frame | x | 34.0, 114.0 | 2.0 | 3.9 |
| L_knee_servo_XM430 | 14 | L_thigh_frame | x | 46.0, 114.0 | 1.85 | 6.0 |
| L_knee_servo_XM430 | 14 | L_thigh_frame | x | 46.0, 114.0 | 2.0 | 6.0 |
| L_ankle_pitch_servo_XC430 | 1 | L_shank_frame | x | 34.0, 63.0 | 3.85 | 6.0 |
| L_ankle_pitch_servo_XC430 | 2 | L_shank_frame | x | 46.0, 63.0 | 3.85 | 6.0 |
| L_ankle_pitch_servo_XC430 | 3 | L_shank_frame | x | 34.0, 39.0 | 3.85 | 4.5 |
| L_ankle_pitch_servo_XC430 | 4 | L_shank_frame | x | 46.0, 39.0 | 3.85 | 6.0 |
| L_ankle_pitch_servo_XC430 | 11 | L_shank_frame | x | 34.0, 63.0 | 1.85 | 6.0 |
| L_ankle_pitch_servo_XC430 | 11 | L_shank_frame | x | 34.0, 63.0 | 2.0 | 0.5 |
| L_ankle_pitch_servo_XC430 | 12 | L_shank_frame | x | 46.0, 63.0 | 1.85 | 6.0 |
| L_ankle_pitch_servo_XC430 | 12 | L_shank_frame | x | 46.0, 63.0 | 2.0 | 0.5 |
| L_ankle_pitch_servo_XC430 | 13 | L_shank_frame | x | 34.0, 39.0 | 1.85 | 4.5 |
| L_ankle_pitch_servo_XC430 | 13 | L_shank_frame | x | 34.0, 39.0 | 2.0 | 4.5 |
| L_ankle_pitch_servo_XC430 | 14 | L_shank_frame | x | 46.0, 39.0 | 1.85 | 6.0 |
| L_ankle_pitch_servo_XC430 | 14 | L_shank_frame | x | 46.0, 39.0 | 2.0 | 6.0 |
| L_ankle_roll_servo_XM430 | 3 | L_ankle_bracket | y | 39.0, -57.0 | 8.6 | 2.5 |
| L_ankle_roll_servo_XM430 | 3 | L_ankle_bracket | y | 39.0, -57.0 | 4.0 | 6.0 |
| L_ankle_roll_servo_XM430 | 4 | L_ankle_bracket | y | 39.0, -45.0 | 8.6 | 2.5 |
| L_ankle_roll_servo_XM430 | 4 | L_ankle_bracket | y | 39.0, -45.0 | 4.0 | 6.0 |
| L_ankle_roll_servo_XM430 | 13 | L_ankle_bracket | y | 39.0, -57.0 | 8.6 | 2.5 |
| L_ankle_roll_servo_XM430 | 13 | L_ankle_bracket | y | 39.0, -57.0 | 4.0 | 6.0 |
| L_ankle_roll_servo_XM430 | 14 | L_ankle_bracket | y | 39.0, -45.0 | 8.6 | 2.5 |
| L_ankle_roll_servo_XM430 | 14 | L_ankle_bracket | y | 39.0, -45.0 | 4.0 | 6.0 |
| head_nod_servo_XL430 | 11 | head_back | z | -0.75, -9.7 | 3.0 | 1.3 |
| head_nod_servo_XL430 | 11 | head_back | z | -0.75, -9.7 | 1.2 | 6.0 |
| head_nod_servo_XL430 | 12 | head_back | z | -0.75, 2.3 | 3.0 | 1.3 |
| head_nod_servo_XL430 | 12 | head_back | z | -0.75, 2.3 | 1.2 | 6.0 |
| head_nod_servo_XL430 | 13 | head_back | z | -24.75, -9.7 | 3.0 | 6.0 |
| head_nod_servo_XL430 | 13 | head_back | z | -24.75, -9.7 | 1.2 | 6.0 |
| head_nod_servo_XL430 | 14 | head_back | z | -24.75, 2.3 | 3.0 | 6.0 |
| head_nod_servo_XL430 | 14 | head_back | z | -24.75, 2.3 | 1.2 | 6.0 |

## holes not cut (0)


## wall left < 1.2 mm (8)

- R_knee_servo_XM430 #1 R_thigh_frame: 0.5 mm
- R_knee_servo_XM430 #2 R_thigh_frame: 0.5 mm
- R_ankle_pitch_servo_XC430 #1 R_shank_frame: 0.5 mm
- R_ankle_pitch_servo_XC430 #2 R_shank_frame: 0.5 mm
- L_knee_servo_XM430 #11 L_thigh_frame: 0.5 mm
- L_knee_servo_XM430 #12 L_thigh_frame: 0.5 mm
- L_ankle_pitch_servo_XC430 #11 L_shank_frame: 0.5 mm
- L_ankle_pitch_servo_XC430 #12 L_shank_frame: 0.5 mm

horn screw holes resized to 2.2: 64

mid pads: [('R', (0.0, -13.0, 0.0)), ('L', (0.0, 13.0, 0.0))]
