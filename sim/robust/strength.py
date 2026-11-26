"""출력물(PLA) 주요 부위 굽힘 여유 — 단순 보 계산(MuJoCo 밖).

단면: out20/print_parts_stl 의 STL(출력 방향 그대로, STL z = 쌓는 방향)을 z 평면으로 잘라 감김 수(winding number)로 채움
  (한 STL 안 겹친 덩어리도 합집합으로 셈). 0.1 mm 격자.
출력 설정 반영: 바깥 벽 1.2 mm(3줄)는 꽉 참, 그 안쪽은 채움 20 % → 강성 · 강도 기여 0.2 배 [추정: 밀도 비례].
힘: 그 단면을 지나는 굽힘 모멘트 = 이웃 관절 토크(시뮬 걷기 최대 · 나쁜 조합 최대 · 서보 정지토크 상한) [시뮬 실측 / e-manual].
  모멘트 축(앞뒤 피치 · 옆 롤)이 STL 의 x · y 중 어느 쪽인지 따로 맞추지 않고, 두 축 중 약한 쪽 단면계수를 씀(보수적).
  축 하중(몸무게 약 30 N)은 굽힘에 비해 작아 뺌(단면 300 mm² 에서 0.1 MPa).
허용 응력: 이 부위들은 모두 다리 방향이 쌓는 방향(z)이라 굽힘 응력이 층 사이를 벌린다.
  PLA 층간 인장 강도 15–30 MPa, 대표 20 MPa [추정: 문헌 일반값, 층 0.2 mm · 노즐 0.4 mm]. 반복 하중은 그 1/3 정도를 오래 버틴다고 봄 [추정].
python robust/strength.py → robust/results/strength.json"""
import os, sys, json, glob
import numpy as np
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
STL = os.path.join(ROOT, "out20", "print_parts_stl")
sys.path.insert(0, HERE)
H = 0.1               # 격자 mm
WALL = 1.2            # 벽 mm
INFILL = 0.2
SIG_LAYER = 20.0      # MPa [추정]
# 부품 → (자를 z 범위 mm(끝 8 mm 는 혼 · 나사 자리 · 모따기라 뺌), 지나는 관절 토크)
PARTS = {"R_thigh_frame": ((8, 90), ("R_hip_pitch", "R_knee", "R_hip_roll")),
         "R_shank_frame": ((8, 84), ("R_knee", "R_ankle_pitch", "R_ankle_roll")),
         "R_hip_yaw_bracket": ((8, 74), ("R_hip_roll", "R_hip_pitch", "R_hip_yaw")),
         "R_hip_roll_bracket": ((8, 78), ("R_hip_roll", "R_hip_pitch")),       # 안 A 롤 출력 U 브래킷(피치 서보를 잡음)
         "R_ankle_bracket": ((8, 76), ("R_ankle_pitch", "R_ankle_roll"))}


def load(p):
    b = open(p, "rb").read()
    n = int(np.frombuffer(b[80:84], "<u4")[0])
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return a["v"].astype(float)


def section(T, z):
    """z 평면 단면: 방향 있는 선분 → 감김 수로 채운 격자 mask, 격자 원점"""
    zs = T[:, :, 2] - z
    zs[np.abs(zs) < 1e-9] = 1e-9
    s = np.sign(zs)
    cut = ~((s[:, 0] == s[:, 1]) & (s[:, 1] == s[:, 2]))
    T, zs, s = T[cut], zs[cut], s[cut]
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    segs = []
    for tri, d, sg, nn in zip(T, zs, s, nrm):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if sg[i] != sg[j]:
                t = d[i] / (d[i] - d[j]); pts.append(tri[i] + t * (tri[j] - tri[i]))
        if len(pts) != 2:
            continue
        a, b = pts[0][:2], pts[1][:2]
        dv = b - a
        # 재료가 왼쪽(반시계): 바깥 법선이 진행 방향 오른쪽 (dv × z) · n > 0
        if dv[1] * nn[0] - dv[0] * nn[1] < 0:
            a, b = b, a
        segs.append((a, b))
    if not segs:
        return np.zeros((1, 1), bool), (0.0, 0.0)
    S = np.array(segs)                          # (k, 2, 2)
    x0, y0 = S[:, :, 0].min() - 1, S[:, :, 1].min() - 1
    x1, y1 = S[:, :, 0].max() + 1, S[:, :, 1].max() + 1
    nx, ny = int((x1 - x0) / H) + 1, int((y1 - y0) / H) + 1
    W = np.zeros((ny, nx))
    ya, yb = S[:, 0, 1], S[:, 1, 1]
    for iy in range(ny):
        y = y0 + (iy + 0.5) * H
        c = (np.minimum(ya, yb) <= y) & (np.maximum(ya, yb) > y)
        if not c.any():
            continue
        A, B = S[c, 0], S[c, 1]
        xc = A[:, 0] + (y - A[:, 1]) * (B[:, 0] - A[:, 0]) / (B[:, 1] - A[:, 1])
        sgn = np.where(B[:, 1] > A[:, 1], 1.0, -1.0)
        ix = np.clip(((xc - x0) / H + 0.5).astype(int), 0, nx - 1)
        row = np.zeros(nx + 1); np.add.at(row, ix, sgn)
        W[iy] = np.cumsum(row[:nx])
    return np.abs(W) > 0.5, (x0, y0)


def props(mask, org):
    if mask.sum() < 10:
        return None
    dist = ndimage.distance_transform_edt(mask) * H
    w = np.where(dist <= WALL, 1.0, INFILL) * mask
    ny, nx = mask.shape
    X, Y = np.meshgrid(org[0] + (np.arange(nx) + 0.5) * H, org[1] + (np.arange(ny) + 0.5) * H)
    dA = H * H
    Aw = w.sum() * dA; cx = (w * X).sum() * dA / Aw; cy = (w * Y).sum() * dA / Aw
    Ixx = (w * (Y - cy) ** 2).sum() * dA; Iyy = (w * (X - cx) ** 2).sum() * dA     # mm^4
    cyM = np.abs(Y[mask] - cy).max(); cxM = np.abs(X[mask] - cx).max()
    # 바깥 섬유(벽)에서 응력: σ = M c / I(유효)
    return dict(area_solid_mm2=float(mask.sum() * dA), area_eff_mm2=float(Aw), Sx_mm3=float(Ixx / cyM), Sy_mm3=float(Iyy / cxM),
                Smin_mm3=float(min(Ixx / cyM, Iyy / cxM)))


def torques():
    """시뮬 최대 관절 힘(N·m): 기준 걷기 · 나쁜 조합 걷기 · 전 시험 최대(넘어지기 전)"""
    out = {}
    for g in ("oat", "combo"):
        p = os.path.join(HERE, "results", f"{g}.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["rows"]:
            res = r["res"]
            if "load_max" not in res:
                continue
            fm = {k: v * res["stall"][k] for k, v in res["force_over_stall_max"].items()}
            tag = None
            mt = r["meta"]
            if g == "oat" and mt["axis"] == "기준" and not mt["backpack"] and mt["test"].startswith("걷기"):
                tag = "기준 걷기"
            if g == "combo" and mt["combo"] == "전부" and mt["test"].startswith("걷기"):
                tag = "나쁜 조합 걷기"
            for k, v in fm.items():
                out.setdefault("모든 시험 최대", {}); out["모든 시험 최대"][k] = max(out["모든 시험 최대"].get(k, 0), v)
                if tag:
                    out.setdefault(tag, {}); out[tag][k] = max(out[tag].get(k, 0), v)
            for k, v in res["stall"].items():
                out.setdefault("정지토크", {}); out["정지토크"][k] = max(out["정지토크"].get(k, 0), v) if abs(v - round(v, 2)) < 1e-9 and v in (3.8, 1.75, 1.4) else out["정지토크"].get(k, 0)
            if tag == "기준 걷기":
                out["기준 걷기 99 %"] = {k: v * res["stall"][k] for k, v in res["force_over_stall_p99"].items()}
    return out


if __name__ == "__main__":
    TQ = torques()
    rows = {}
    for part, ((za, zb), joints) in PARTS.items():
        T = load(os.path.join(STL, part + ".stl"))
        best = None; prof = []
        for z in np.arange(za, zb + 0.01, 1.0):
            m, org = section(T, z)
            pr = props(m, org)
            if pr is None:
                continue
            prof.append(dict(z=float(z), **pr))
            if best is None or pr["Smin_mm3"] < best["Smin_mm3"]:
                best = dict(z=float(z), **pr)
        r = dict(weakest=best, joints=joints, profile=prof, stress_MPa={})
        for tag, tq in TQ.items():
            M = max(tq.get(j, 0.0) for j in joints) * 1000.0         # N·mm
            r["stress_MPa"][tag] = dict(M_Nm=M / 1000, sigma=M / best["Smin_mm3"], safety=SIG_LAYER / (M / best["Smin_mm3"]))
        rows[part] = r
        print(part, f"z={best['z']:.0f} A={best['area_solid_mm2']:.0f} mm2 Smin={best['Smin_mm3']:.0f} mm3",
              {k: (round(v["M_Nm"], 2), round(v["sigma"], 1), round(v["safety"], 1)) for k, v in r["stress_MPa"].items()}, flush=True)
    import hashlib, time
    src = {p: dict(sha1=hashlib.sha1(open(os.path.join(STL, p + ".stl"), "rb").read()).hexdigest()[:12],
                   mtime=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(STL, p + ".stl"))))) for p in PARTS}
    json.dump(dict(note=__doc__, stl_source=src, sig_layer_MPa=SIG_LAYER, wall_mm=WALL, infill=INFILL, torques=TQ, parts=rows),
              open(os.path.join(HERE, "results", "strength.json"), "w"), indent=1, ensure_ascii=False)
