| 관절 | joints.csv 지금 (°) | 형상 한계 − 쪽 (°) / 처음 닿는 곳 | 형상 한계 + 쪽 (°) / 처음 닿는 곳 | 여유 −/+ (°) | 소프트웨어 한계 초안 (°) | 작업 범위 초안 (°) | Dynamixel Min / Max (영점 2048, 정방향) |
|---|---|---|---|---|---|---|---|
| R_hip_yaw | -20 ~ 20 | -24.0 / R_ankle_link:R_ankle_bracket ↔ L_ankle_link:L_ankle_bracket | 26.1 / R_foot:R_foot_sole ↔ L_foot:L_foot_sole | 5.0 / 5.0 | -19.0 ~ 21.1 | -14.0 ~ 16.1 | 1832 / 2288 |
| R_hip_pitch | -80 ~ 30 | -86.6 / R_thigh:R_thigh_shell ↔ torso:shorts | 90.5 / torso:shorts ↔ R_shank:R_shank_shell | 5.0 / 5.0 | -81.6 ~ 85.5 | -76.6 ~ 80.5 | 1119 / 3021 |
| R_knee | 0 ~ 100 | -36.2 / R_shank:R_shank_shell ↔ R_thigh:R_thigh_shell | 103.2 / torso:shorts ↔ R_foot:R_foot_sole | 7.2 / 5.0 | -29.0 ~ 98.2 | 0.0 ~ 93.2 | 1718 / 3166 |
| R_ankle_pitch | -50 ~ 22 | -72.5 / R_shank:R_shank_shell ↔ R_foot:R_foot_sole | 25.8 / R_ankle_link:R_ankle_roll_servo_XM430 ↔ R_shank:R_shank_frame | 5.0 / 5.0 | -67.5 ~ 20.8 | -62.5 ~ 15.8 | 1280 / 2284 |
| R_ankle_roll | -20 ~ 20 | -24.0 / R_foot:R_foot_sole ↔ R_ankle_link:R_ankle_bracket | 24.0 / R_foot:R_foot_sole ↔ R_ankle_link:R_ankle_bracket | 6.4 / 6.4 | -17.6 ~ 17.6 | -12.6 ~ 12.6 | 1848 / 2248 |
| L_hip_yaw | -20 ~ 20 | -26.2 / R_foot:R_foot_sole ↔ L_foot:L_foot_sole | 23.2 / R_ankle_link:R_ankle_bracket ↔ L_ankle_link:L_ankle_bracket | 5.0 / 5.0 | -21.2 ~ 18.2 | -16.2 ~ 13.2 | 1806 / 2256 |
| L_hip_pitch | -80 ~ 30 | -86.6 / L_thigh:L_thigh_shell ↔ torso:shorts | 90.5 / torso:shorts ↔ L_shank:L_shank_shell | 5.0 / 5.0 | -81.6 ~ 85.5 | -76.6 ~ 80.5 | 1119 / 3021 |
| L_knee | 0 ~ 100 | -36.2 / L_shank:L_shank_shell ↔ L_thigh:L_thigh_shell | 103.2 / torso:shorts ↔ L_foot:L_foot_sole | 7.2 / 5.0 | -29.0 ~ 98.2 | 0.0 ~ 93.2 | 1718 / 3166 |
| L_ankle_pitch | -50 ~ 22 | -72.5 / L_shank:L_shank_shell ↔ L_foot:L_foot_sole | 25.8 / L_ankle_link:L_ankle_roll_servo_XM430 ↔ L_shank:L_shank_frame | 5.0 / 5.0 | -67.5 ~ 20.8 | -62.5 ~ 15.8 | 1280 / 2284 |
| L_ankle_roll | -20 ~ 20 | -24.0 / L_foot:L_foot_sole ↔ L_ankle_link:L_ankle_bracket | 24.0 / L_foot:L_foot_sole ↔ L_ankle_link:L_ankle_bracket | 6.4 / 6.4 | -17.6 ~ 17.6 | -12.6 ~ 12.6 | 1848 / 2248 |
| R_arm_swing | -120 ~ 120 | -137.6 / head:head_front ↔ R_arm:arm_R_hand | 131.8 / R_arm:arm_R_hand ↔ head:head_back | 5.0 / 5.0 | -132.6 ~ 126.8 | -127.6 ~ 121.8 | 539 / 3490 |
| L_arm_swing | -120 ~ 120 | -137.6 / head:head_front ↔ L_arm:arm_L_hand | 131.8 / head:head_back ↔ L_arm:arm_L_hand | 5.0 / 5.0 | -132.6 ~ 126.8 | -127.6 ~ 121.8 | 539 / 3490 |
| head_nod | -10 ~ 20 | -25.8 / head:head_back ↔ torso:lid | 33.0 / head:head_front ↔ torso:lid | 5.0 / 5.0 | -20.8 ~ 28.0 | -10.0 ~ 20.0 | 1812 / 2367 |

| 짝 | 자유 영역 비율 | 관절 A 보수 범위(B 가 작업 범위 어디에 있어도) | 관절 B 보수 범위(A 가 작업 범위 어디에 있어도) |
|---|---|---|---|
| R 고관절 피치 × 무릎 (R_hip_pitch × R_knee) | 52 % | -85 ~ 10 | -30 ~ 40 |
| R 무릎 × 발목 피치 (R_knee × R_ankle_pitch) | 40 % | -30 ~ 95 | -70 ~ 25 |
| R 발목 피치 × 발목 롤 (R_ankle_pitch × R_ankle_roll) | 23 % | -65 ~ 25 | -15 ~ 15 |
| R 팔 × 머리 끄덕임 (R_arm_swing × head_nod) | 73 % | -120 ~ 120 | -15 ~ 25 |
| R 팔 × 고관절 피치 (R_arm_swing × R_hip_pitch) | 61 % | -130 ~ 130 | -80 ~ 60 |
| L 고관절 피치 × 무릎 (L_hip_pitch × L_knee) | 52 % | -85 ~ 10 | -30 ~ 40 |
| L 무릎 × 발목 피치 (L_knee × L_ankle_pitch) | 40 % | -30 ~ 95 | -70 ~ 25 |
| L 발목 피치 × 발목 롤 (L_ankle_pitch × L_ankle_roll) | 23 % | -65 ~ 25 | -15 ~ 15 |
| L 팔 × 머리 끄덕임 (L_arm_swing × head_nod) | 73 % | -120 ~ 120 | -15 ~ 25 |
| L 팔 × 고관절 피치 (L_arm_swing × L_hip_pitch) | 61 % | -130 ~ 130 | -80 ~ 60 |
| 두 다리 고관절 요 (다리 편 자세) (R_hip_yaw × L_hip_yaw) | 30 % | – | – |
| 두 다리 고관절 요 (웅크림(피치 −30, 무릎 60)) (R_hip_yaw × L_hip_yaw) | 30 % | – | – |
| 두 다리 고관절 요 (무릎만 60) (R_hip_yaw × L_hip_yaw) | 52 % | – | – |

- 두 발 평평히 웅크리기(고관절 피치 = 발목 피치 = −무릎/2): 형상으로는 무릎 118° 까지 충돌 없음 (무릎 120° 에서 R_thigh:R_thigh_frame ↔ R_shank:R_shank_frame); joints.csv 범위로는 무릎 100° = 웅크림 54 mm 까지
