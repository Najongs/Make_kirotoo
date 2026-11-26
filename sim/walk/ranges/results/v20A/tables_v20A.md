| 관절 | joints.csv 지금 (°) | 형상 한계 − 쪽 (°) / 처음 닿는 곳 | 형상 한계 + 쪽 (°) / 처음 닿는 곳 | 여유 −/+ (°) | 소프트웨어 한계 초안 (°) | 작업 범위 초안 (°) | Dynamixel Min / Max (영점 2048, 정방향) |
|---|---|---|---|---|---|---|---|
| R_hip_yaw | -20 ~ 20 | -25.4 / R_ankle_link:R_ankle_bracket ↔ L_ankle_link:L_ankle_bracket | 56.0 / R_foot:R_foot_sole ↔ L_foot:L_foot_sole | 5.0 / 5.0 | -20.4 ~ 51.0 | -15.4 ~ 46.0 | 1816 / 2628 |
| R_hip_roll | -25 ~ 2 | -33.6 / R_thigh:R_thigh_frame ↔ R_hand_flap:arm_R_flap | 6.0 / R_ankle_link:R_ankle_bracket ↔ L_ankle_link:L_ankle_bracket | 5.0 / 5.0 | -28.6 ~ 1.0 | -23.6 ~ 0.0 | 1722 / 2059 |
| R_hip_pitch | -80 ~ 30 | -117.8 / torso:shorts ↔ R_thigh:R_thigh_shell | 114.8 / torso:shorts ↔ R_thigh:R_thigh_frame | 5.0 / 5.0 | -112.8 ~ 109.8 | -107.8 ~ 104.8 | 765 / 3297 |
| R_knee | 0 ~ 100 | -36.8 / R_thigh:R_thigh_shell ↔ R_shank:R_shank_shell | 117.8 / R_thigh:R_thigh_frame ↔ R_shank:R_shank_frame | 7.1 / 5.0 | -29.7 ~ 112.8 | 0.0 ~ 107.8 | 1710 / 3331 |
| R_ankle_pitch | -50 ~ 22 | -77.8 / R_shank:R_shank_shell ↔ R_foot:R_foot_sole | 30.2 / R_foot:R_foot_sole ↔ R_shank:R_shank_frame | 7.3 / 5.8 | -70.5 ~ 24.5 | -65.5 ~ 19.5 | 1246 / 2326 |
| R_ankle_roll | -20 ~ 20 | -31.6 / R_ankle_link:R_ankle_bracket ↔ R_foot:R_foot_sole | 26.9 / R_foot:R_foot_sole ↔ R_ankle_link:R_ankle_bracket | 7.1 / 6.2 | -24.6 ~ 20.7 | -19.6 ~ 15.7 | 1768 / 2284 |
| L_hip_yaw | -20 ~ 20 | -54.6 / L_foot:L_foot_sole ↔ R_foot:R_foot_sole | 25.4 / R_ankle_link:R_ankle_bracket ↔ L_ankle_link:L_ankle_bracket | 5.0 / 5.0 | -49.6 ~ 20.4 | -44.6 ~ 15.4 | 1483 / 2280 |
| L_hip_roll | -2 ~ 25 | -6.0 / R_ankle_link:R_ankle_bracket ↔ L_ankle_link:L_ankle_bracket | 33.9 / L_thigh:L_thigh_frame ↔ L_hand_flap:arm_L_flap | 5.0 / 5.0 | -1.0 ~ 28.9 | 0.0 ~ 23.9 | 2037 / 2377 |
| L_hip_pitch | -80 ~ 30 | -117.8 / torso:shorts ↔ L_thigh:L_thigh_shell | 114.8 / torso:shorts ↔ L_thigh:L_thigh_frame | 5.0 / 5.0 | -112.8 ~ 109.8 | -107.8 ~ 104.8 | 765 / 3297 |
| L_knee | 0 ~ 100 | -36.8 / L_thigh:L_thigh_shell ↔ L_shank:L_shank_shell | 117.8 / L_thigh:L_thigh_frame ↔ L_shank:L_shank_frame | 7.1 / 5.0 | -29.7 ~ 112.8 | 0.0 ~ 107.8 | 1710 / 3331 |
| L_ankle_pitch | -50 ~ 22 | -76.9 / L_foot:L_foot_sole ↔ L_shank:L_shank_shell | 30.2 / L_foot:L_foot_sole ↔ L_shank:L_shank_frame | 7.3 / 5.8 | -69.6 ~ 24.5 | -64.6 ~ 19.5 | 1256 / 2326 |
| L_ankle_roll | -20 ~ 20 | -27.4 / L_foot:L_foot_sole ↔ L_ankle_link:L_ankle_bracket | 32.9 / L_ankle_link:L_ankle_bracket ↔ L_foot:L_foot_sole | 6.2 / 6.9 | -21.1 ~ 26.0 | -16.1 ~ 21.0 | 1807 / 2343 |
| R_arm_swing | -120 ~ 120 | -137.6 / head:head_front ↔ R_arm:arm_R_hand | 131.8 / R_arm:arm_R_hand ↔ head:head_back | 5.0 / 5.0 | -132.6 ~ 126.8 | -127.6 ~ 121.8 | 539 / 3490 |
| L_arm_swing | -120 ~ 120 | -137.6 / head:head_front ↔ L_arm:arm_L_hand | 131.8 / head:head_back ↔ L_arm:arm_L_hand | 5.0 / 5.0 | -132.6 ~ 126.8 | -127.6 ~ 121.8 | 539 / 3490 |
| head_nod | -10 ~ 20 | -25.8 / head:head_back ↔ torso:lid | 33.0 / head:head_front ↔ torso:lid | 5.0 / 5.0 | -20.8 ~ 28.0 | -10.0 ~ 20.0 | 1812 / 2367 |

| 짝 | 자유 영역 비율 | 관절 A 보수 범위(B 가 작업 범위 어디에 있어도) | 관절 B 보수 범위(A 가 작업 범위 어디에 있어도) |
|---|---|---|---|
| R 고관절 피치 × 무릎 (R_hip_pitch × R_knee) | 73 % | -110 ~ 30 | -30 ~ 75 |
| R 무릎 × 발목 피치 (R_knee × R_ankle_pitch) | 48 % | -30 ~ 110 | -75 ~ 30 |
| R 발목 피치 × 발목 롤 (R_ankle_pitch × R_ankle_roll) | 29 % | -65 ~ 25 | -15 ~ 15 |
| R 고관절 롤 × 고관절 피치 (R_hip_roll × R_hip_pitch) | 48 % | -25 ~ 5 | -105 ~ 60 |
| R 팔 × 머리 끄덕임 (R_arm_swing × head_nod) | 73 % | -120 ~ 120 | -15 ~ 25 |
| R 팔 × 고관절 피치 (R_arm_swing × R_hip_pitch) | 73 % | -130 ~ 130 | -110 ~ 60 |
| L 고관절 피치 × 무릎 (L_hip_pitch × L_knee) | 73 % | -110 ~ 30 | -30 ~ 75 |
| L 무릎 × 발목 피치 (L_knee × L_ankle_pitch) | 48 % | -30 ~ 110 | -75 ~ 30 |
| L 발목 피치 × 발목 롤 (L_ankle_pitch × L_ankle_roll) | 29 % | -50 ~ 25 | -15 ~ 15 |
| L 고관절 롤 × 고관절 피치 (L_hip_roll × L_hip_pitch) | 48 % | -5 ~ 25 | -105 ~ 60 |
| L 팔 × 머리 끄덕임 (L_arm_swing × head_nod) | 73 % | -120 ~ 120 | -15 ~ 25 |
| L 팔 × 고관절 피치 (L_arm_swing × L_hip_pitch) | 73 % | -130 ~ 130 | -110 ~ 60 |
| 두 다리 고관절 요 (다리 편 자세) (R_hip_yaw × L_hip_yaw) | 65 % | – | – |
| 두 다리 고관절 요 (웅크림(피치 −30, 무릎 60)) (R_hip_yaw × L_hip_yaw) | 49 % | – | – |
| 두 다리 고관절 요 (무릎만 60) (R_hip_yaw × L_hip_yaw) | 58 % | – | – |

- 두 발 평평히 웅크리기(고관절 피치 = 발목 피치 = −무릎/2): 형상으로는 무릎 116° 까지 충돌 없음 (무릎 118° 에서 R_thigh:R_thigh_frame ↔ R_shank:R_shank_frame); joints.csv 범위로는 무릎 100° = 웅크림 54 mm 까지
