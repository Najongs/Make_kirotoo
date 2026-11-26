| 관절 | joints.csv 지금 (°) | 형상 한계 − 쪽 (°) / 처음 닿는 곳 | 형상 한계 + 쪽 (°) / 처음 닿는 곳 | 여유 −/+ (°) | 소프트웨어 한계 초안 (°) | 작업 범위 초안 (°) | Dynamixel Min / Max (영점 2048, 정방향) |
|---|---|---|---|---|---|---|---|
| R_hip_yaw | -20 ~ 20 | -25.4 / R_ankle_link:R_ankle_bracket ↔ L_ankle_link:L_ankle_bracket | 26.1 / R_foot:R_foot_sole ↔ L_foot:L_foot_sole | 5.0 / 5.0 | -20.4 ~ 21.1 | -15.4 ~ 16.1 | 1816 / 2288 |
| R_hip_roll | -18 ~ 2 | -23.1 / R_hip_roll_link:R_hip_pitch_servo_XM430 ↔ R_hip_yaw_link:R_hip_yaw_bracket | 3.2 / R_foot:R_foot_sole ↔ L_foot:L_foot_sole | 8.9 / 5.0 | -14.2 ~ 0.0 | -9.2 ~ 0.0 | 1886 / 2048 |
| R_hip_pitch | -80 ~ 11 | -86.6 / R_thigh:R_thigh_shell ↔ torso:shorts | 16.9 / R_thigh:R_knee_servo_XM430 ↔ R_hip_yaw_link:R_hip_roll_servo_XM430 | 5.0 / 5.0 | -81.6 ~ 11.9 | -76.6 ~ 6.9 | 1119 / 2183 |
| R_knee | 0 ~ 100 | -36.2 / R_shank:R_shank_shell ↔ R_thigh:R_thigh_shell | 71.0 / R_hip_yaw_link:R_hip_roll_servo_XM430 ↔ R_ankle_link:R_ankle_roll_servo_XM430 | 7.2 / 5.0 | -29.0 ~ 66.0 | 0.0 ~ 61.0 | 1718 / 2799 |
| R_ankle_pitch | -50 ~ 22 | -79.2 / R_shank:R_shank_shell ↔ R_foot:R_foot_sole | 30.2 / R_foot:R_foot_sole ↔ R_shank:R_shank_frame | 5.0 / 5.8 | -74.2 ~ 24.5 | -69.2 ~ 19.5 | 1203 / 2326 |
| R_ankle_roll | -20 ~ 20 | -31.6 / R_ankle_link:R_ankle_bracket ↔ R_foot:R_foot_sole | 29.6 / L_foot:L_foot_sole ↔ R_foot:R_foot_sole | 7.1 / 5.0 | -24.6 ~ 24.6 | -19.6 ~ 19.6 | 1768 / 2328 |
| L_hip_yaw | -20 ~ 20 | -26.2 / R_foot:R_foot_sole ↔ L_foot:L_foot_sole | 25.4 / R_ankle_link:R_ankle_bracket ↔ L_ankle_link:L_ankle_bracket | 5.0 / 5.0 | -21.2 ~ 20.4 | -16.2 ~ 15.4 | 1806 / 2280 |
| L_hip_roll | -2 ~ 18 | -3.2 / R_foot:R_foot_sole ↔ L_foot:L_foot_sole | 22.1 / L_hip_roll_link:L_hip_pitch_servo_XM430 ↔ L_hip_yaw_link:L_hip_yaw_bracket | 5.0 / 8.7 | 0.0 ~ 13.4 | 0.0 ~ 8.4 | 2048 / 2201 |
| L_hip_pitch | -80 ~ 11 | -86.6 / L_thigh:L_thigh_shell ↔ torso:shorts | 16.9 / L_thigh:L_knee_servo_XM430 ↔ L_hip_yaw_link:L_hip_roll_servo_XM430 | 5.0 / 5.0 | -81.6 ~ 11.9 | -76.6 ~ 6.9 | 1119 / 2183 |
| L_knee | 0 ~ 100 | -36.2 / L_shank:L_shank_shell ↔ L_thigh:L_thigh_shell | 71.0 / L_hip_yaw_link:L_hip_roll_servo_XM430 ↔ L_ankle_link:L_ankle_roll_servo_XM430 | 7.2 / 5.0 | -29.0 ~ 66.0 | 0.0 ~ 61.0 | 1718 / 2799 |
| L_ankle_pitch | -50 ~ 22 | -79.2 / L_shank:L_shank_shell ↔ L_foot:L_foot_sole | 30.2 / L_foot:L_foot_sole ↔ L_shank:L_shank_frame | 5.0 / 5.8 | -74.2 ~ 24.5 | -69.2 ~ 19.5 | 1203 / 2326 |
| L_ankle_roll | -20 ~ 20 | -27.8 / R_foot:R_foot_sole ↔ L_foot:L_foot_sole | 32.9 / L_ankle_link:L_ankle_bracket ↔ L_foot:L_foot_sole | 5.0 / 6.9 | -22.8 ~ 26.0 | -17.8 ~ 21.0 | 1789 / 2343 |
| R_arm_swing | -120 ~ 120 | -137.6 / head:head_front ↔ R_arm:arm_R_hand | 131.8 / R_arm:arm_R_hand ↔ head:head_back | 5.0 / 5.0 | -132.6 ~ 126.8 | -127.6 ~ 121.8 | 539 / 3490 |
| L_arm_swing | -120 ~ 120 | -137.6 / head:head_front ↔ L_arm:arm_L_hand | 131.8 / head:head_back ↔ L_arm:arm_L_hand | 5.0 / 5.0 | -132.6 ~ 126.8 | -127.6 ~ 121.8 | 539 / 3490 |
| head_nod | -10 ~ 20 | -25.8 / head:head_back ↔ torso:lid | 33.0 / head:head_front ↔ torso:lid | 5.0 / 5.0 | -20.8 ~ 28.0 | -10.0 ~ 20.0 | 1812 / 2367 |

| 짝 | 자유 영역 비율 | 관절 A 보수 범위(B 가 작업 범위 어디에 있어도) | 관절 B 보수 범위(A 가 작업 범위 어디에 있어도) |
|---|---|---|---|
| R 고관절 피치 × 무릎 (R_hip_pitch × R_knee) | 35 % | -85 ~ 15 | -30 ~ 65 |
| R 무릎 × 발목 피치 (R_knee × R_ankle_pitch) | 38 % | -30 ~ 60 | -75 ~ 30 |
| R 발목 피치 × 발목 롤 (R_ankle_pitch × R_ankle_roll) | 30 % | -65 ~ 25 | -15 ~ 15 |
| R 고관절 롤 × 고관절 피치 (R_hip_roll × R_hip_pitch) | 20 % | -20 ~ 0 | -80 ~ 10 |
| R 팔 × 머리 끄덕임 (R_arm_swing × head_nod) | 73 % | -120 ~ 120 | -15 ~ 25 |
| R 팔 × 고관절 피치 (R_arm_swing × R_hip_pitch) | 41 % | -130 ~ 130 | -80 ~ 10 |
| L 고관절 피치 × 무릎 (L_hip_pitch × L_knee) | 35 % | -85 ~ 15 | -30 ~ 65 |
| L 무릎 × 발목 피치 (L_knee × L_ankle_pitch) | 38 % | -30 ~ 60 | -75 ~ 30 |
| L 발목 피치 × 발목 롤 (L_ankle_pitch × L_ankle_roll) | 30 % | -55 ~ 25 | -10 ~ 15 |
| L 고관절 롤 × 고관절 피치 (L_hip_roll × L_hip_pitch) | 20 % | 0 ~ 20 | -80 ~ 15 |
| L 팔 × 머리 끄덕임 (L_arm_swing × head_nod) | 73 % | -120 ~ 120 | -15 ~ 25 |
| L 팔 × 고관절 피치 (L_arm_swing × L_hip_pitch) | 41 % | -130 ~ 130 | -80 ~ 10 |
| 두 다리 고관절 요 (다리 편 자세) (R_hip_yaw × L_hip_yaw) | 32 % | – | – |
| 두 다리 고관절 요 (웅크림(피치 −30, 무릎 60)) (R_hip_yaw × L_hip_yaw) | 32 % | – | – |
| 두 다리 고관절 요 (무릎만 60) (R_hip_yaw × L_hip_yaw) | 31 % | – | – |

- 두 발 평평히 웅크리기(고관절 피치 = 발목 피치 = −무릎/2): 형상으로는 무릎 102° 까지 충돌 없음 (무릎 104° 에서 R_thigh:R_thigh_frame ↔ R_shank:R_shank_frame); joints.csv 범위로는 무릎 100° = 웅크림 54 mm 까지
