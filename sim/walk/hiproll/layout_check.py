"""고관절 롤 배치안 공간 검사(설계 형상 기준, 설계 파일은 읽기만).

설계 형상: code_v17/build18.build_model() (v18). 새 부품은 상자로 근사:
  X430 케이스 W 28.5 × H 46.5 × D 34 mm, 출력축은 D 방향, H 방향 위끝에서 11.2 mm(지금 고관절 피치 서보 위치에서 잼),
  혼 Ø20.4 × 3.9 mm(출력 쪽), 아이들러 Ø20.4 × 3 mm(반대쪽), 브래킷 판 두께 3 mm [추정: 지금 브래킷과 같게].
겹침은 새 부품 · 옮긴 부품의 격자 점(2 mm)이 기존 부품 solid 안(geom4.inside)에 드는 개수와 부피로 셉니다.
오른다리만 보고, 왼다리는 대칭입니다. 다리끼리는 두 다리가 모두 안쪽으로 20° 롤일 때 봅니다.

usage: cd code_v17 && ../sim/.venv/bin/python ../sim/walk/hiproll/layout_check.py  → sim/walk/hiproll/results/layouts.json, layouts.png
"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "code_v17"))
import build18, build4, geom4
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)

links, joints, _ = build18.build_model()
STEP = 2.0
W, H, D, SH = 28.5, 46.5, 34.0, 11.2       # X430 케이스, 축은 위끝에서 SH
HORN, IDL, PL = 3.9, 3.0, 3.0
Y0 = -40.0                                   # 오른 다리 고관절 y
Z_YAW_OUT = 213.0                            # 요 링크 판 윗면: 지금 요 브래킷 판 z 201–213 (그 위는 6805 베어링을 지나는 허브, 몸통 틀)
Z_PITCH = 185.0


def box(x, y, z, name):
    return dict(name=name, lo=np.array([x[0], y[0], z[0]], float), hi=np.array([x[1], y[1], z[1]], float))


def pts_box(b):
    ax = [np.arange(l + STEP / 2, h, STEP) if h - l > STEP else np.array([(l + h) / 2]) for l, h in zip(b["lo"], b["hi"])]
    return np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)


def rotx(P, deg, zc, yc=Y0):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    Q = P.copy(); y, z = P[:, 1] - yc, P[:, 2] - zc
    Q[:, 1] = yc + c * y - s * z; Q[:, 2] = zc + s * y + c * z
    return Q


def roty(P, deg, zc, xc=0.0):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    Q = P.copy(); x, z = P[:, 0] - xc, P[:, 2] - zc
    Q[:, 0] = xc + c * x + s * z; Q[:, 2] = zc - s * x + c * z
    return Q


def solid_pts(sol, dz=0.0):
    P = np.vstack([geom4.samples(s, STEP) for s in sol]); P[:, 2] += dz; return P


def mirror(P):
    Q = P.copy(); Q[:, 1] *= -1; return Q


# 기존 부품
TORSO = {k: v for k, v in links["torso"]["parts"].items()}
THIGH = links["R_thigh"]["parts"]
PITCH_SERVO = links["R_hip_yaw_link"]["parts"]["R_hip_pitch_servo_XM430"]
P_THIGH0 = solid_pts([s for k in ("R_thigh_frame", "R_thigh_shell", "R_knee_servo_XM430", "R_hip_pitch_horn", "R_hip_pitch_idler") for s in THIGH[k]])
P_PSERVO0 = solid_pts(PITCH_SERVO)


def hits(P, parts, skip=()):
    """P 가 들어간 기존 부품: {이름: 점 수}"""
    out = {}
    for nm, sol in parts.items():
        if nm in skip:
            continue
        m = np.zeros(len(P), bool)
        for s in sol:
            b = build4.aabb(s)
            sel = (P[:, 0] > b[0][0]) & (P[:, 0] < b[0][1]) & (P[:, 1] > b[1][0]) & (P[:, 1] < b[1][1]) & (P[:, 2] > b[2][0]) & (P[:, 2] < b[2][1])
            if sel.any():
                idx = np.where(sel)[0]
                m[idx] |= geom4.inside(s, P[idx])
        if m.any():
            out[nm] = int(m.sum())
    return out


def overlap_pts(P, boxes, margin=0.0):
    n = 0
    for b in boxes:
        n += int(np.all((P > b["lo"] - margin) & (P < b["hi"] + margin), axis=1).sum())
    return n


def layout(name):
    """새 상자(고정 = 요 링크에 붙음, 롤 = 롤 출력에 붙음), 롤 축 높이, 다리 늘어남 Δ, 몸통 쪽으로 올린 양 up"""
    if name == "A":
        # 직렬: 요 혼 바로 아래 롤 서보(축 x, 케이스 세로). 롤 출력 U 브래킷이 피치 서보 케이스 윗면을 잡는다.
        top = Z_YAW_OUT - PL; zr = top - SH; bot = top - H
        fixed = [box((-19, 19), (Y0 - 16, Y0 + 16), (top, Z_YAW_OUT), "요 아래 판"),
                 box((-D / 2, D / 2), (Y0 - W / 2, Y0 + W / 2), (bot, top), "롤 서보 케이스"),
                 box((D / 2, D / 2 + HORN), (Y0 - 10.2, Y0 + 10.2), (zr - 10.2, zr + 10.2), "롤 혼"),
                 box((-D / 2 - IDL, -D / 2), (Y0 - 10.2, Y0 + 10.2), (zr - 10.2, zr + 10.2), "롤 아이들러")]
        # 롤 출력: 바닥판이 롤 서보 아래 모서리가 ±20° 돌아도 2 mm 떨어지게 → 반지름
        r_corner = math.hypot(zr - bot, W / 2)
        plate_top = zr - (r_corner + 2.0)
        pitch_top = plate_top - PL
        zp = pitch_top - SH
        roll = [box((D / 2 + HORN + 0.5, D / 2 + HORN + 3.5), (Y0 - 12, Y0 + 12), (plate_top - PL, zr + 10.5), "U 브래킷 앞판"),
                box((-D / 2 - IDL - 3.5, -D / 2 - IDL - 0.5), (Y0 - 12, Y0 + 12), (plate_top - PL, zr + 10.5), "U 브래킷 뒤판"),
                box((-25, 25), (Y0 - 16, Y0 + 16), (plate_top - PL, plate_top), "U 브래킷 바닥판")]
        return dict(fixed=fixed, roll=roll, zr=zr, zp=zp, delta=Z_PITCH - zp, up=0.0)
    if name == "B":
        # 축 교차: 롤 서보를 피치 서보 뒤에 눕혀(축 x) 롤 축 = 피치 축 높이. 요 브래킷을 뒤로 늘려 롤 서보를 잡고,
        # 롤 출력 틀이 피치 서보 케이스를 앞뒤에서 잡는다(앞은 요 브래킷 앞판의 베어링).
        zr = Z_PITCH; xb0 = -14.2 - 3.5 - HORN - D      # 롤 서보 뒤끝
        fixed = [box((xb0 - IDL - 3.5, 25), (Y0 - 16, Y0 + 16), (Z_YAW_OUT - PL, Z_YAW_OUT), "요 아래 판(뒤로 늘림)"),
                 box((xb0, xb0 + D), (Y0 - W / 2, Y0 + W / 2), (zr - (H - SH), zr + SH), "롤 서보 케이스"),
                 box((xb0 + D, xb0 + D + HORN), (Y0 - 10.2, Y0 + 10.2), (zr - 10.2, zr + 10.2), "롤 혼"),
                 box((xb0 - IDL, xb0), (Y0 - 10.2, Y0 + 10.2), (zr - 10.2, zr + 10.2), "롤 아이들러"),
                 box((xb0 - IDL - 3.5, xb0 - IDL), (Y0 - 14, Y0 + 14), (zr - 14, Z_YAW_OUT - PL), "뒤 기둥"),
                 box((21.5, 25), (Y0 - 14, Y0 + 14), (zr - 12, Z_YAW_OUT - PL), "앞 기둥(베어링)")]
        roll = [box((-14.2 - 3.4, -14.2 - 0.4), (Y0 - 12, Y0 + 12), (zr - 14, zr + 14), "롤 틀 뒤판"),
                box((15.5, 18.5), (Y0 - 12, Y0 + 12), (zr - 14, zr + 14), "롤 틀 앞판")]
        return dict(fixed=fixed, roll=roll, zr=zr, zp=Z_PITCH, delta=0.0, up=0.0)
    if name == "C":
        # A 를 몸통 쪽으로 Δ 만큼 올림: 다리 길이 그대로, 요 서보 · 베어링 · 요 혼도 Δ 위로
        a = layout("A"); up = a["delta"]
        sh = lambda bs: [dict(b, lo=b["lo"] + [0, 0, up], hi=b["hi"] + [0, 0, up]) for b in bs]
        return dict(fixed=sh(a["fixed"]), roll=sh(a["roll"]), zr=a["zr"] + up, zp=Z_PITCH, delta=0.0, up=up)


def check(name):
    L = layout(name)
    up, delta, zr, zp = L["up"], L["delta"], L["zr"], L["zp"]
    skip = ()
    torso = dict(TORSO)
    if up:
        # 요 서보 · 베어링을 위로 옮긴 사본으로 바꿈
        for k in ("yaw_servo_R_XC430", "yaw_bearing_R_6805", "yaw_servo_L_XC430", "yaw_bearing_L_6805"):
            torso.pop(k)
    Pf = np.vstack([pts_box(b) for b in L["fixed"]])
    res = {"name": name, "roll_axis_z": zr, "pitch_axis_z": zp, "leg_longer_mm": delta, "pelvis_parts_up_mm": up,
           "hip_y_mm": Y0, "new_boxes": [(b["name"], b["lo"].round(1).tolist(), b["hi"].round(1).tolist()) for b in L["fixed"] + L["roll"]]}
    res["fixed_vs_torso"] = hits(Pf, torso)
    if up:
        ys = {k: solid_pts(TORSO[k], up) for k in ("yaw_servo_R_XC430", "yaw_bearing_R_6805")}
        ys_all = np.vstack(list(ys.values()))
        res["moved_yaw_servo_vs_torso"] = hits(ys_all, {k: v for k, v in torso.items()})
        res["moved_yaw_servo_top_z"] = float(ys["yaw_servo_R_XC430"][:, 2].max())
    # 롤로 도는 묶음: 롤 상자 + 피치 서보 + 허벅지(피치 −80 / 0 / +30°)
    worst = {}
    for phi in (-20, 0, 20):
        for pit in (-80, 0, 30):
            Pr = np.vstack([pts_box(b) for b in L["roll"]])
            Ps = P_PSERVO0 + [0, 0, -delta]
            Pt = roty(P_THIGH0 + [0, 0, -delta], pit, zp)
            Pall = rotx(np.vstack([Pr, Ps, Pt]), phi, zr)
            h = hits(Pall, torso)
            for k, v in h.items():
                worst[k] = max(worst.get(k, 0), v)
            nself = overlap_pts(Pall, L["fixed"], margin=0.0)
            worst["(자기) 고정 상자"] = max(worst.get("(자기) 고정 상자", 0), nself)
            if nself and "self_first" not in res:
                res["self_first"] = (phi, pit, nself)
    res["rolling_vs_torso_worst_pts"] = worst
    # 두 다리 안쪽 20° 롤: 오른 묶음 +20°(안쪽) vs 왼 묶음 −20° 를 거울로
    Pr0 = np.vstack([pts_box(b) for b in L["roll"]] + [P_PSERVO0 + [0, 0, -delta], P_THIGH0 + [0, 0, -delta]])
    R_in = rotx(Pr0, 20, zr)
    L_in = mirror(rotx(Pr0, 20, zr))
    # 최소 간격: 두 점 구름 사이 y 방향 겹침 범위의 최근접 거리(근사, kd-tree)
    from scipy.spatial import cKDTree
    dmin = float(cKDTree(L_in).query(R_in, k=1)[0].min())
    res["leg_leg_gap_both_in20_mm"] = dmin
    R_0 = Pr0; L_0 = mirror(Pr0)
    res["leg_leg_gap_0_mm"] = float(cKDTree(L_0).query(R_0, k=1)[0].min())
    res["leg_leg_gap_R_in20_L0_mm"] = float(cKDTree(L_0).query(R_in, k=1)[0].min())
    # 외형: 새 부품이 반바지 바깥(y 75, x −93..70)으로 나가는지
    allp = np.vstack([Pf, rotx(Pr0, -20, zr), rotx(Pr0, 20, zr)])
    res["outer_envelope_mm"] = {"x": [float(allp[:, 0].min()), float(allp[:, 0].max())],
                                "y_R": [float(allp[:, 1].min()), float(allp[:, 1].max())],
                                "z_top": float(np.vstack([Pf, rotx(np.vstack([pts_box(b) for b in L["roll"]]), 20, zr)])[:, 2].max())}
    res["battery_overlap_pts"] = res["fixed_vs_torso"].get("battery_LB012", 0) + worst.get("battery_LB012", 0)
    # 쓸 수 있는 범위: 롤 φ(−: 바깥, +: 안쪽) × 피치에서 자기 고정 상자 · 몸통과 안 닿는 영역, 다리끼리 3 mm
    Pr = np.vstack([pts_box(b) for b in L["roll"]])
    Ps = P_PSERVO0 + [0, 0, -delta]
    ok = {}
    for pit in range(-80, 31, 10):
        Pt = roty(P_THIGH0 + [0, 0, -delta], pit, zp)
        good = []
        for phi in range(-25, 26, 5):
            Pall = rotx(np.vstack([Pr, Ps, Pt]), phi, zr)
            if overlap_pts(Pall, L["fixed"]) == 0 and not hits(Pall, torso):
                good.append(phi)
        ok[pit] = good
    res["free_roll_by_pitch_deg"] = ok
    for nm_, other in (("in_R_L0", lambda P: mirror(Pr0)), ("both_in", None)):
        lim = None
        for phi in np.arange(0, 25.1, 0.5):
            Rp = rotx(Pr0, phi, zr)
            Lp = mirror(Pr0) if other else mirror(rotx(Pr0, phi, zr))
            if cKDTree(Lp).query(Rp, k=1)[0].min() < 3.0:
                lim = float(phi); break
        res[f"inward_roll_limit_3mm_{nm_}_deg"] = lim if lim is not None else ">25"
    return res, L


# 지금 구조(비교용)
def current():
    P = solid_pts(links["R_hip_yaw_link"]["parts"]["R_hip_yaw_bracket"])
    Pr0 = np.vstack([P_PSERVO0, P_THIGH0])
    from scipy.spatial import cKDTree
    return {"name": "현재", "pitch_axis_z": Z_PITCH, "leg_longer_mm": 0.0,
            "leg_leg_gap_0_mm": float(cKDTree(mirror(Pr0)).query(Pr0, k=1)[0].min()),
            "bracket_vs_torso": hits(P, TORSO)}


if __name__ == "__main__":
    out = {"current": current()}
    Ls = {}
    for nm in ("A", "B", "C"):
        r, L = check(nm); out[nm] = r; Ls[nm] = L
        print(json.dumps({k: v for k, v in r.items() if k != "new_boxes"}, ensure_ascii=False))
    print(json.dumps(out["current"], ensure_ascii=False))
    json.dump(out, open(os.path.join(RES, "layouts.json"), "w"), indent=1, ensure_ascii=False)
    # ---- 그림: 정면(y-z) · 옆(x-z), 지금 구조와 세 안
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = os.path.expanduser("~/.local/share/fonts/NotoSansKR.ttf")
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    ctx = {k: solid_pts(TORSO[k]) for k in ("shorts", "battery_LB012", "pelvis_hatch", "yaw_servo_R_XC430", "yaw_bearing_R_6805", "torso_frame", "shelf_pins", "battery_jetson_2S_1500")}
    col = {"shorts": "#d5d8dc", "battery_LB012": "#e67e22", "pelvis_hatch": "#f5cba7", "yaw_servo_R_XC430": "#5d6d7e",
           "yaw_bearing_R_6805": "#85929e", "torso_frame": "#eaecee", "shelf_pins": "#aab7b8", "battery_jetson_2S_1500": "#f0b27a"}
    fig, ax = plt.subplots(2, 4, figsize=(19, 10))
    titles = {"cur": "현재 (롤 없음)", "A": "안 A 직렬(롤 서보를 요 판 아래 세로로)", "B": "안 B 축 교차(롤 서보를 피치 서보 뒤에)", "C": "안 C 골반 올림(A 를 몸통 쪽으로)"}
    for j, key in enumerate(("cur", "A", "B", "C")):
        for i, (u, lab) in enumerate(((1, "y (mm) — 정면"), (0, "x (mm) — 옆, 앞 = +x"))):
            a = ax[i, j]
            for k, P in ctx.items():
                Pk = P.copy()
                if key == "C" and k.startswith("yaw_"):
                    Pk[:, 2] += Ls["C"]["up"]
                sel = (Pk[:, 1] < 0) | (u == 0)
                a.scatter(Pk[sel, u], Pk[sel, 2], s=1, c=col[k], label=k if (i == 0 and j == 0) else None, rasterized=True)
            if key == "cur":
                Pb = solid_pts(links["R_hip_yaw_link"]["parts"]["R_hip_yaw_bracket"])
                a.scatter(Pb[:, u], Pb[:, 2], s=1, c="#1f618d", rasterized=True)
                a.scatter(P_PSERVO0[:, u], P_PSERVO0[:, 2], s=1, c="#2e4053", rasterized=True)
                a.scatter(P_THIGH0[:, u], P_THIGH0[:, 2], s=1, c="#7fb3d5", rasterized=True)
                zp = Z_PITCH
            else:
                L = Ls[key]
                for b in L["fixed"]:
                    a.add_patch(plt.Rectangle((b["lo"][u], b["lo"][2]), b["hi"][u] - b["lo"][u], b["hi"][2] - b["lo"][2], fc="#c0392b", ec="k", lw=0.5, alpha=0.8))
                for b in L["roll"]:
                    a.add_patch(plt.Rectangle((b["lo"][u], b["lo"][2]), b["hi"][u] - b["lo"][u], b["hi"][2] - b["lo"][2], fc="#27ae60", ec="k", lw=0.5, alpha=0.8))
                Ps = P_PSERVO0 + [0, 0, -L["delta"]]; Pt = P_THIGH0 + [0, 0, -L["delta"]]
                a.scatter(Ps[:, u], Ps[:, 2], s=1, c="#2e4053", rasterized=True); a.scatter(Pt[:, u], Pt[:, 2], s=1, c="#7fb3d5", rasterized=True)
                zp = L["zp"]; a.axhline(L["zr"], color="#27ae60", ls="--", lw=0.8)
            a.axhline(zp, color="#2e4053", ls=":", lw=0.8)
            a.set_xlim((-110, 10) if u == 1 else (-100, 75)); a.set_ylim(80, 300); a.set_aspect("equal"); a.grid(alpha=0.3)
            a.set_xlabel(lab); a.set_ylabel("z (mm)")
            if u == 1:
                a.axvline(-75, color="#999", lw=0.6, ls="--")
            if i == 0:
                extra = "" if key == "cur" else f"\n다리 +{out[key]['leg_longer_mm']:.0f} mm · 요 서보 위로 +{out[key]['pelvis_parts_up_mm']:.0f} mm"
                a.set_title(titles[key] + extra, fontsize=10)
    ax[0, 0].legend(fontsize=7, markerscale=6, loc="lower left")
    dA = out["A"]["leg_longer_mm"]; uC = out["C"]["pelvis_parts_up_mm"]
    fig.suptitle(f"고관절 롤 배치안: 직렬(A)은 다리가 {dA:.0f} mm 길어지고, 축 교차(B)는 다리 길이 그대로지만 롤 서보가 엉덩이 뒤(x −61 mm)로 나가며 허벅지 뒤로 젖히기가 +5° 로 줄고,\n"
                 f"골반 올림(C)은 다리 길이 그대로지만 요 서보를 {uC:.0f} mm 올려 Jetson · 선반과 겹친다 — 빨강: 요 링크에 붙는 새 부품, 초록: 롤과 함께 도는 새 부품, 점선: 롤 축 · 피치 축 (설계 형상 + 상자 근사)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(RES, "layouts.png"), dpi=85)
    print("saved")
