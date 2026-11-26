"""조합 묶음(2번 추천을 고르려고): 1차 결과에서 무릎뿐 아니라 고관절 롤도 열 한계를 넘어서(걷기 부하 RMS 0.72–0.82 N·m) 롤 대책을 함께 본다.
ax_sweep.py 가 import 해서 GROUPS["combo"] 로 쓴다."""
import ax_common as A
K22, G2, K15, G3, K2 = "2xXL430e_g2", "XL430e_g2", "2xXL430e_g15", "XL430e_g3", "2xXL430e"
COMBO = {
    "C1 무릎 XL 둘 + 2:1, 롤 2:1": A.cfg(dict(knee=K22, hip_roll=G2)),
    "C2 무릎 XL 둘 + 2:1, 롤 XL 둘": A.cfg(dict(knee=K22, hip_roll=K2)),
    "C3 C1 + 웅크림 25": A.cfg(dict(knee=K22, hip_roll=G2), crouch=0.025),
    "C4 C1 + 발목 피치 2:1": A.cfg(dict(knee=K22, hip_roll=G2, ankle_pitch=G2)),
    "C5 무릎 XL 둘 + 1.5:1, 롤 2:1": A.cfg(dict(knee=K15, hip_roll=G2)),
    "C6 무릎 XL 하나 + 3:1, 롤 2:1": A.cfg(dict(knee=G3, hip_roll=G2)),
    "C7 무릎 XL 둘, 롤 2:1, 웅크림 25, 몸무게 −20 %": A.cfg(dict(knee=K2, hip_roll=G2), crouch=0.025, mass_f=0.8),
    "C8 무릎 2:1, 롤 2:1, 웅크림 25, 몸무게 −20 %": A.cfg(dict(knee=G2, hip_roll=G2), crouch=0.025, mass_f=0.8),
    "C9 C1 + 배낭 250 g": A.cfg(dict(knee=K22, hip_roll=G2), pack_=True),
    "C10 C1 + 허벅지 · 정강이 +10": A.cfg(dict(knee=K22, hip_roll=G2), seg=0.01),
}

# 2차 조합: 1차에서 감속(2:1 · 1.5:1 · 3:1)은 되비친 회전자 관성(감속비²) 때문에 걷기 부하 RMS 가 오히려 커졌고(무릎 2.0–2.9 N·m),
# 롤은 XL 두 개가 2:1 보다 시원했다 → 감감속 없이 「XL 두 개」 쪽으로 몸무게 · 웅크림 · 다리 길이를 조인다.
# 몸무게 줄이기(현실 쪽): 몸통 · 머리 body(1.497 kg, 출력물 · 전자부품 · 몸통 서보 포함)를 0.8 배 = −300 g [추정: 채움 · 두께 줄이기 몫].
LIGHT = {"torso": 0.8, "head": 0.8}
XL12 = {k: "XL430e12" for k in A.ALLXL}
COMBO2 = {
    "D1 무릎 · 롤 XL 둘, 웅크림 25": A.cfg(dict(knee=K2, hip_roll=K2), crouch=0.025),
    "D2 무릎 · 롤 XL 둘, 웅크림 25, −300 g": A.cfg(dict(knee=K2, hip_roll=K2), crouch=0.025, mass_scale=LIGHT),
    "D3 무릎 · 롤 XL 둘, 웅크림 15, −300 g": A.cfg(dict(knee=K2, hip_roll=K2), crouch=0.015, mass_scale=LIGHT),
    "D4 D2 + 허벅지 · 정강이 0": A.cfg(dict(knee=K2, hip_roll=K2), crouch=0.025, mass_scale=LIGHT, seg=0.0),
    "D5 D2 + 허벅지 · 정강이 +10": A.cfg(dict(knee=K2, hip_roll=K2), crouch=0.025, mass_scale=LIGHT, seg=0.01),
    "D6 무릎 · 롤 XL 둘, 웅크림 35, −300 g": A.cfg(dict(knee=K2, hip_roll=K2), crouch=0.035, mass_scale=LIGHT),
    "D7 무릎 · 롤 XL 둘, 웅크림 25, 몸 전체 −20 %": A.cfg(dict(knee=K2, hip_roll=K2), crouch=0.025, mass_f=0.8),
    "D8 D2 를 12 V 로": A.cfg(dict(XL12, knee="2xXL430e12", hip_roll="2xXL430e12"), crouch=0.025, mass_scale=LIGHT),
}
