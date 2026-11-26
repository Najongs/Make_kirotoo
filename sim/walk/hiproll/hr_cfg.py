"""고관절 롤 배치안의 시뮬 설정(layout_check.py 결과에서 옮김, 설계 형상 + 상자 근사).
A 직렬: 롤 축 z 198.8, 피치 축 z 144.5 (다리 +40.5 mm), 롤 서보 무게중심 z 186.8 (요 링크 원점 z 213 아래 26 mm).
B 축 교차: 롤 축 = 피치 축 z 185, 롤 서보 무게중심 x −38.6, z 173 (요 링크 원점 아래 40 mm), 다리 길이 그대로.
서보 질량: XM430 82 g, XC430 65 g [e-manual]. 브래킷 질량은 넣지 않음(빼낸 요 브래킷과 비슷하다고 봄 [추정])."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M = {"XM430-W350": 0.082, "XC430-W240": 0.065}


def hr(layout, servo="XM430-W350"):
    if layout == "A":
        return dict(servo=servo, d_axis=0.0543, lengthen=0.0405, mass=M[servo], mass_pos=(0.0, 0.0, -0.026))
    if layout == "B":
        return dict(servo=servo, d_axis=0.0, lengthen=0.0, mass=M[servo], mass_pos=(-0.0386, 0.0, -0.040))
    raise KeyError(layout)
