"""키토루 v20 한 장 조립도 (A3 가로) -> out20/assembly_drawing/kitoru_v20_assembly.png (300 dpi) + .pdf

usage (code_v17/):  ../sim/.venv/bin/python assembly_drawing20.py
  - 형상: build20.build_model() (render6 툰 렌더러, 모든 관절 0°). 부품을 이름으로 옮겨 분해도를 만든다
    (혼 · 아이들러는 자식 링크에 있지만 서보 축에 붙어 다니므로 서보와 함께 옮긴다).
  - 나사 · 조립 순서는 out20/LEG_PRINT_KIT_R.md 3절과 out20/BUY_LIST_v20.md 2절(빌드에 체결 부품 정보가 없음).
  - 서보 ID 는 robot/config/joints_v20.yaml (자리표시, 조립 후 확정). 모든 서보는 XL430-W250-T 로 적는다.
  - ASM_CACHE=<dir> 이면 렌더 결과를 그 폴더에 저장 · 재사용한다(배치만 고칠 때). BUILD 환경변수로 빌드 모듈을 바꿀 수 있다.
"""
import os, sys, json, math, inspect, pickle, tempfile, textwrap
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("BUILD", "build20")
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTD = os.path.join(ROOT, "out20", "assembly_drawing")
BUILD_COMMIT = "ac14c70"          # last build20.py change = the leg STL build (LEG_PRINT_KIT_R.md)
DATE = "2026-11-27"
DPI = 300
PX_MM = 25.4 / DPI                # paper mm per image pixel at native resolution
CACHE = os.environ.get("ASM_CACHE")

PARTS = {p["part"]: p for p in json.load(open(os.path.join(ROOT, "out20", "print_parts", "parts.json")))}
SERVO_RGB = np.array([0.13, 0.13, 0.15])
HORN_RGB = np.array([0.62, 0.63, 0.66])
ELECTRONICS = ("imu_", "battery", "opencr", "jetson", "buck_", "switch_", "dc_jack", "camera_pi", "speaker", "amp_", "mic_")

# ---------------------------------------------------------------- leg explode (right leg, part names)
# (key, parts, servo joint or None). Stack top -> bottom; "beside" items share the host's height and move by dx.
LEG_STACK = [
    ("yaw_servo", ["yaw_servo_R_XL430", "R_hip_yaw_horn", "R_hip_yaw_idler"]),
    ("bearing", ["yaw_bearing_R_6805"]),
    ("yaw_bracket", ["R_hip_yaw_bracket"]),
    ("roll_servo", ["R_hip_roll_servo_XM430", "R_hip_roll_horn", "R_hip_roll_idler"]),
    ("roll_bracket", ["R_hip_roll_bracket"]),
    ("pitch_servo", ["R_hip_pitch_servo_XM430", "R_hip_pitch_horn", "R_hip_pitch_idler"]),
    ("thigh_frame", ["R_thigh_frame"]),
    ("knee_servo", ["R_knee_servo_XM430", "R_knee_horn", "R_knee_idler"]),
    ("shank_frame", ["R_shank_frame"]),
    ("ankle_pitch_servo", ["R_ankle_pitch_servo_XC430", "R_ankle_pitch_horn", "R_ankle_pitch_idler"]),
    ("ankle_bracket", ["R_ankle_bracket"]),
    ("foot", ["R_foot_sole"]),
]
LEG_BESIDE = [("thigh_shell", ["R_thigh_shell"], "thigh_frame", 80.0),
              ("shank_shell", ["R_shank_shell"], "shank_frame", 95.0),
              ("ankle_roll_servo", ["R_ankle_roll_servo_XM430", "R_ankle_roll_horn", "R_ankle_roll_idler"], "ankle_bracket", 160.0)]
SPLIT_AFTER = "pitch_servo"       # column A = hip .. hip pitch servo, column B = thigh .. foot
LABEL_W = 56.0                    # label column width next to each leg column (paper mm)
CLR = 20.0                        # mm between stacked items (model)

# labels: printed parts (name, file, colour name, line about screws)
P_LABEL = {
    "yaw_bracket": ("① 고관절 요 브래킷", "R_hip_yaw_bracket.stl", "요 혼 M2×6 ×4 (아래 카운터보어로) · 롤 서보 옆판 M2.5×8 ×8"),
    "roll_bracket": ("② 고관절 롤 U 브래킷", "R_hip_roll_bracket.stl", "롤 혼 M2×6 ×4 · 아이들러 M2×6 ×4 · 피치 서보 M2.5×10 ×4 + M2.5×6 ×2"),
    "thigh_frame": ("③ 허벅지 프레임", "R_thigh_frame.stl", "피치 혼 M2×6 ×4 · 무릎 서보 M2.5×8 ×8"),
    "thigh_shell": ("④ 허벅지 셸 (덮개)", "R_thigh_shell.stl", "흰색 · 걸쇠로 끼움, 나사 없음"),
    "shank_frame": ("⑤ 정강이 프레임", "R_shank_frame.stl", "무릎 혼 M2×6 ×4 · 발목 피치 서보 M2.5×8 ×8"),
    "shank_shell": ("⑥ 정강이 셸 (덮개)", "R_shank_shell.stl", "하늘색 · 무릎을 잇기 전에 위에서 씌움 (빔 걸쇠)"),
    "ankle_bracket": ("⑦ 발목 브래킷", "R_ankle_bracket.stl", "발목 피치 혼 M2×6 ×3 · 발목 롤 서보 M2.5×16 ×4"),
    "foot": ("⑧ 발판", "R_foot_sole.stl", "남색 · 발목 롤 혼 M2×6 ×4 · 고무 패드 Ø12 ×5"),
}
# servos and bought parts: (name, line 2)
S_LABEL = {
    "yaw_servo": ("고관절 요 서보 XL430 · ID 1", "몸통 프레임 안 · M2.5×8 ×2 (혼이 아래)"),
    "bearing": ("6805 베어링 (구매품)", "몸통 프레임에 압입 · ① 허브가 들어감"),
    "roll_servo": ("고관절 롤 서보 XL430 · ID 2", "혼 앞 · 아이들러 뒤, ② 두 판 사이로 내려 넣음"),
    "pitch_servo": ("고관절 피치 서보 XL430 · ID 3", "혼이 바깥쪽, ② 아래에서 두 벽 사이로"),
    "knee_servo": ("무릎 서보 XL430 · ID 4", "③ 아래 칸에 끼움"),
    "ankle_pitch_servo": ("발목 피치 서보 XL430 · ID 5", "⑤ 아래 칸에 끼움"),
    "ankle_roll_servo": ("발목 롤 서보 XL430 · ID 6", "앞에서 ⑦ 에 밀어 넣음 · 아이들러는 뒤판 홈"),
}
# assembly arrows between stacked items: (upper, lower, direction "up"/"down", kit step)
ARROWS = [("yaw_servo", "bearing", "up", 9), ("bearing", "yaw_bracket", "up", 9), ("yaw_bracket", "roll_servo", "up", 10),
          ("roll_servo", "roll_bracket", "down", 8), ("roll_bracket", "pitch_servo", "up", 7),
          ("pitch_servo", "thigh_frame", "up", 7), ("knee_servo", "shank_frame", "up", 6),
          ("shank_frame", "ankle_pitch_servo", "up", 3), ("ankle_pitch_servo", "ankle_bracket", "up", 4),
          ("ankle_bracket", "foot", "up", 2)]
SIDE_ARROWS = [("thigh_shell", 6), ("shank_shell", 5), ("ankle_roll_servo", 1)]
STEPS = ["발목 롤 서보 → 발목 브래킷", "발판 → 발목 롤 혼", "발목 피치 서보 → 정강이 프레임", "발목 브래킷 → 발목 피치 혼",
         "정강이 셸 씌우기", "무릎 서보 → 허벅지 프레임, 허벅지 셸, 정강이 → 무릎 혼", "피치 서보 → U 브래킷, 허벅지 → 피치 혼",
         "롤 서보 → U 브래킷 (다리 한 덩어리 완성)", "요 브래킷 → 몸통 (6805 · 요 혼)", "다리 → 요 브래킷 (안쪽 옆판 나사 먼저)"]


# ---------------------------------------------------------------- geometry helpers
def setup_renderer():
    import render6 as r
    for ln in r.links:                       # zero pose everywhere: every link pose is the identity
        r.CHAINS[ln] = []
    src = inspect.getsource(r.render).replace("def render(", "def render_ids(")
    src = src.replace("return (col * 255).astype(np.uint8)",
                      "return (col * 255).astype(np.uint8), ids, (u0, v1, right, up, scale, pad)")
    exec(compile(src, "render_ids", "exec"), r.__dict__)       # same rasteriser, also returns part ids + projection
    geom = []
    for ln, pn, pid, T, rgb, caps in r.GEOM:
        if any(k in pn for k in ELECTRONICS):
            continue
        rgb = np.array(rgb, float)
        if pn.endswith(("_horn", "_idler")) or "_horn_XC430" in pn or "_idler_XC430" in pn:
            rgb = HORN_RGB
        elif "_servo_" in pn:
            rgb = SERVO_RGB
        geom.append((ln, pn, pid, T, rgb, caps))
    return r, geom


def moved(geom, offs, keep=None):
    """copy of geom with part pn translated by offs[pn]; keep = set of part names to keep (None = all)"""
    out = []
    for ln, pn, pid, T, rgb, caps in geom:
        if keep is not None and pn not in keep:
            continue
        d = np.asarray(offs.get(pn, (0.0, 0.0, 0.0)), float)
        out.append((ln, pn, pid, T + d if len(T) else T, rgb, [(np.asarray(o) + d, [np.asarray(h) + d for h in hs]) for o, hs in caps]))
    return out


def part_pts(geom, names):
    P = [T.reshape(-1, 3) for ln, pn, pid, T, rgb, caps in geom if pn in names and len(T)]
    P += [np.asarray(o) for ln, pn, pid, T, rgb, caps in geom if pn in names for o, hs in caps]
    return np.vstack(P)


def view_axes(elev, azim):
    e, z = math.radians(elev), math.radians(azim)
    view = np.array([math.cos(e) * math.cos(z), math.cos(e) * math.sin(z), math.sin(e)])
    right = np.array([-math.sin(z), math.cos(z), 0.0])
    return right, np.cross(view, right)


def extent(geom, elev, azim):
    right, up = view_axes(elev, azim)
    P = part_pts(geom, {g[1] for g in geom})
    return np.ptp(P @ right), np.ptp(P @ up)


def do_render(r, geom, elev, azim, scale, tag):
    """render geom; returns (image, ids, projector, pid map). Cached under ASM_CACHE when set."""
    if CACHE:
        fp = os.path.join(CACHE, f"r_{tag}.pkl")
        if os.path.exists(fp):
            img, ids, pr = pickle.load(open(fp, "rb"))
            return img, ids, pr
    old = r.GEOM[:]
    r.GEOM[:] = geom
    try:
        img, ids, pr = r.render_ids(elev, azim, {}, scale=scale)
    finally:
        r.GEOM[:] = old
    pr = tuple(np.asarray(x) if isinstance(x, np.ndarray) else x for x in pr)
    if CACHE:
        os.makedirs(CACHE, exist_ok=True)
        pickle.dump((img, ids, pr), open(os.path.join(CACHE, f"r_{tag}.pkl"), "wb"))
    return img, ids, pr


def project(pr, P):
    u0, v1, right, up, scale, pad = pr
    P = np.atleast_2d(P)
    return np.stack([(P @ right - u0) * scale + pad, (v1 - P @ up) * scale + pad], 1)


def anchor_px(geom, ids, pr, names):
    """visible pixel deepest inside the parts' silhouette (falls back to the projected centre)"""
    from scipy.ndimage import distance_transform_edt
    pids = {g[2] for g in geom if g[1] in names}
    m = np.isin(ids, list(pids))
    if m.sum() > 20:
        d = distance_transform_edt(m)
        y, x = np.unravel_index(np.argmax(d), d.shape)
        return np.array([x, y], float)
    return project(pr, part_pts(geom, names).mean(0))[0]


# ---------------------------------------------------------------- explodes
def leg_offsets(geom):
    zr = {}
    for key, names in LEG_STACK:
        P = part_pts(geom, set(names)); zr[key] = (P[:, 2].min(), P[:, 2].max())
    offs, dz, prev_bot = {}, {}, None
    for key, names in LEG_STACK:
        lo, hi = zr[key]
        d = 0.0 if prev_bot is None else (prev_bot - CLR) - hi
        dz[key] = d; prev_bot = lo + d
        for n in names:
            offs[n] = (0.0, 0.0, d)
    for key, names, host, dx in LEG_BESIDE:
        dz[key] = dz[host]
        for n in names:
            offs[n] = (dx, 0.0, dz[host])
    return offs, dz


def whole_offsets(geom):
    offs = {}
    for ln, pn, *_ in geom:
        if ln in ("head", "nod_crank", "nod_rod", "ears", "ear_rod"):
            offs[pn] = (0.0, 0.0, 90.0)
        elif ln in ("R_arm", "R_hand_flap") or pn in ("arm_R_horn_adapter", "arm_R_horn_XC430", "arm_R_idler_XC430", "shoulder_shaft_R"):
            offs[pn] = (0.0, -70.0, 0.0)
        elif ln in ("L_arm", "L_hand_flap") or pn in ("arm_L_horn_adapter", "arm_L_horn_XC430", "arm_L_idler_XC430", "shoulder_shaft_L"):
            offs[pn] = (0.0, 70.0, 0.0)
        elif ln.startswith(("R_", "L_")):
            offs[pn] = (0.0, 0.0, -90.0)
    return offs


# ---------------------------------------------------------------- drawing helpers (one axes in paper mm, y down)
PT = 25.4 / 72                    # mm per point


def fonts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
    src = os.path.expanduser("~/.local/share/fonts/NotoSansKR.ttf")
    d = os.path.join(tempfile.gettempdir(), "kitoru_fonts"); os.makedirs(d, exist_ok=True)
    out = {}
    for w, nm in ((400, "Regular"), (700, "Bold")):
        fp = os.path.join(d, f"NotoSansKR-{nm}.ttf")
        if not os.path.exists(fp):         # the installed font is variable (default weight 100): make static instances
            f = instancer.instantiateVariableFont(TTFont(src), {"wght": w})
            f["name"].setName(f"KitoruKR {nm}", 4, 3, 1, 0x409)
            f.save(fp)
        out[nm] = font_manager.FontProperties(fname=fp)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.unicode_minus"] = False
    return plt, out


class Sheet:
    def __init__(self, plt, F):
        self.plt, self.F = plt, F
        self.fig = plt.figure(figsize=(420 / 25.4, 297 / 25.4), dpi=DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, 420); self.ax.set_ylim(297, 0); self.ax.axis("off")
        self.texts = []

    def text(self, x, y, s, size=7.5, bold=False, color="#191919", ha="left", va="top", z=5, check=True, **kw):
        t = self.ax.text(x, y, s, fontproperties=self.F["Bold" if bold else "Regular"], fontsize=size, color=color,
                         ha=ha, va=va, zorder=z, **kw)
        if check:
            self.texts.append(t)
        return t

    def image(self, img, x, y):
        h, w = img.shape[:2]
        self.ax.imshow(img, extent=(x, x + w * PX_MM, y + h * PX_MM, y), interpolation="lanczos", zorder=1)
        return (x, y, x + w * PX_MM, y + h * PX_MM)

    def line(self, xs, ys, **kw):
        kw.setdefault("color", "#191919"); kw.setdefault("lw", 0.5); kw.setdefault("zorder", 4)
        self.ax.plot(xs, ys, solid_capstyle="round", **kw)

    def rect(self, x, y, w, h, **kw):
        from matplotlib.patches import Rectangle
        kw.setdefault("fill", False); kw.setdefault("lw", 0.6); kw.setdefault("ec", "#191919"); kw.setdefault("zorder", 3)
        self.ax.add_patch(Rectangle((x, y), w, h, **kw))

    def arrow(self, p0, p1, color="#d62728", lw=1.6):
        self.ax.annotate("", xy=p1, xytext=p0, zorder=6,
                         arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.28", color=color, lw=lw, shrinkA=0, shrinkB=0))

    def text_w(self, s, size, bold=False):
        """width of a string in paper mm (measured with the renderer)"""
        t = self.ax.text(0, 0, s, fontproperties=self.F["Bold" if bold else "Regular"], fontsize=size)
        bb = t.get_window_extent(renderer=self.fig.canvas.get_renderer())
        t.remove()
        return bb.width / DPI * 25.4


def wrap(sh, s, size, maxw):
    """split s at ' · ' so every line fits maxw (paper mm)"""
    toks, out, cur = s.split(" · "), [], ""
    for t in toks:
        cand = t if not cur else cur + " · " + t
        if cur and sh.text_w(cand, size) > maxw:
            out.append(cur); cur = t
        else:
            cur = cand
    return out + [cur]


def place_labels(sh, items, x_col, side, y_lo, y_hi, gap=2.2):
    """items: list of dict(anchor=(x, y) mm, lines=[(text, size, bold, colour)]). Labels stacked in a column at x_col
    (side 'L': right-aligned at x_col, 'R': left-aligned at x_col), ordered by anchor height, leader line with elbow."""
    for it in items:
        it["h"] = sum(s * PT * 1.32 for _, s, _, _ in it["lines"])
        it["w"] = max(sh.text_w(t, s, b) for t, s, b, _ in it["lines"])
    items = sorted(items, key=lambda it: it["anchor"][1])
    ys = [it["anchor"][1] - it["h"] / 2 for it in items]
    for i in range(len(items)):                       # push down
        lo = y_lo if i == 0 else ys[i - 1] + items[i - 1]["h"] + gap
        ys[i] = max(ys[i], lo)
    for i in range(len(items) - 1, -1, -1):           # push up from the bottom bound
        hi = (y_hi if i == len(items) - 1 else ys[i + 1] - gap) - items[i]["h"]
        ys[i] = min(ys[i], hi)
    for it, y in zip(items, ys):
        yy = y
        for t, s, b, c in it["lines"]:
            sh.text(x_col, yy, t, size=s, bold=b, color=c, ha="right" if side == "L" else "left")
            yy += s * PT * 1.32
        if it.get("servo"):
            xb_ = x_col - 1.6 if side == "R" else x_col + 0.6
            sh.rect(xb_, y + 0.4, 1.0, it["h"] - 0.8, fill=True, fc="#222222", ec="none", lw=0)
        ax_, ay = it["anchor"]
        ly = y + it["lines"][0][1] * PT * 0.66
        x_end = x_col + (1.2 if side == "L" else -2.4)
        x_elb = x_end + (3.0 if side == "L" else -3.0)
        sh.line([x_end, x_elb, ax_], [ly, ly, ay], color="#444444", lw=0.45)
        sh.ax.plot([ax_], [ay], "o", ms=2.2, color="#191919", zorder=7, mec="white", mew=0.4)
        it["box"] = (x_col - it["w"] if side == "L" else x_col, y, x_col if side == "L" else x_col + it["w"], y + it["h"])
    return items


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUTD, exist_ok=True)
    r, geom = setup_renderer()
    plt, F = fonts()
    sh = Sheet(plt, F)
    INK, GREY, RED = "#191919", "#555555", "#c0392b"

    # ---- assembled extents (height line)
    Pall = part_pts(geom, {g[1] for g in geom})
    z_lo, z_hi = Pall[:, 2].min(), Pall[:, 2].max()
    height = z_hi - z_lo

    # ---- title band
    sh.rect(4, 4, 412, 289, lw=0.9)
    sh.line([4, 416], [15, 15], lw=0.7)
    sh.text(8, 6.2, "키토루 v20 조립도", size=17, bold=True)
    sh.text(62, 7.6, "오른 다리 분해도(크게) · 전체 분해도 · 조립 완성 모습 · 부품 목록", size=9, color=GREY)
    sh.text(412, 6.4, f"{DATE}   빌드 커밋 {BUILD_COMMIT} (code_v17/build20.py)   A3 · 단위 mm", size=8.5, ha="right")
    sh.text(412, 10.6, "모든 서보는 XL430-W250-T · 서보 ID 는 계획 값(robot/config/joints_v20.yaml 자리표시, 조립 후 확정)",
            size=7.5, ha="right", color=RED)

    # ================================================================ 1. right leg exploded (left, big)
    offs, dz = leg_offsets(geom)
    leg_names = {n for _, ns in LEG_STACK for n in ns} | {n for _, ns, _, _ in LEG_BESIDE for n in ns}
    lg = moved(geom, offs, leg_names)
    keys_a = [k for k, _ in LEG_STACK][:[k for k, _ in LEG_STACK].index(SPLIT_AFTER) + 1]
    keys_a += [k for k, _, h, _ in LEG_BESIDE if h in keys_a]
    names_of = {k: ns for k, ns in LEG_STACK}
    names_of.update({k: ns for k, ns, _, _ in LEG_BESIDE})
    set_a = {n for k in keys_a for n in names_of[k]}
    ga = [g for g in lg if g[1] in set_a]
    gb = [g for g in lg if g[1] not in set_a]
    EL, AZ = 18, -38
    box_x0, box_y0, box_x1, box_y1 = 6, 17, 262, 291
    sh.text(box_x0 + 2, box_y0 + 1, "가. 오른 다리 분해도 — 위에서 아래로 붙는 순서, 빨간 화살표 = 끼우는 방향 (숫자 = 조립 순서 단계)",
            size=10, bold=True)
    sh.text(box_x0 + 2, box_y0 + 6.2, "왼 다리는 대칭입니다 — 파일 이름의 R_ 를 L_ 로 (서보 ID 7–12). 서보마다 데이지 체인 케이블을 칸에 넣기 전에 꽂고 다리 뒤쪽으로 뺍니다.",
            size=7.5, color=GREY)
    sh.text(box_x0 + 2, box_y0 + 9.8, "나사 두 종류: M2.5 태핑 나사 = 서보 몸체(플라스틱 케이스)에 바로 박음 · M2×6 = 혼 · 아이들러 나사 (다리 하나 27개). 왼쪽 검은 막대가 붙은 이름 = 서보 · 구매품",
            size=7.5, color=GREY)
    (wa_m, ha_m), (wb_m, hb_m) = extent(ga, EL, AZ), extent(gb, EL, AZ)
    ya = box_y0 + 18.5
    avail_h = (box_y1 - 3) - ya - 3
    avail_w = (box_x1 - box_x0) - 2 * LABEL_W - 14
    pscale = min(avail_h / max(ha_m, hb_m), avail_w / (wa_m + wb_m)) * 0.97   # paper mm per model mm
    scale = pscale / PX_MM                                                    # render px per model mm
    img_a, ids_a, pr_a = do_render(r, ga, EL, AZ, scale, "legA")
    img_b, ids_b, pr_b = do_render(r, gb, EL, AZ, scale, "legB")
    wa, wb = img_a.shape[1] * PX_MM, img_b.shape[1] * PX_MM
    # columns: [image A | labels A] [image B | labels B]
    spare = (box_x1 - box_x0) - wa - wb - 2 * LABEL_W
    xa = box_x0 + spare / 4
    xb = xa + wa + LABEL_W + spare / 2
    rect_a = sh.image(img_a, xa, ya)
    rect_b = sh.image(img_b, xb, ya)
    sh.line([xb - spare / 4, xb - spare / 4], [ya + 4, box_y1 - 4], color="#bbbbbb", lw=0.5, ls=(0, (4, 3)))

    def to_mm(px, rect):
        return (rect[0] + px[0] * PX_MM, rect[1] + px[1] * PX_MM)

    for (gg, ids, pr, rect, keys) in ((ga, ids_a, pr_a, rect_a, keys_a), (gb, ids_b, pr_b, rect_b, [k for k in names_of if k not in keys_a])):
        # centre line of the leg chain (yaw axis x 0, y -40)
        zs = part_pts(gg, {g[1] for g in gg})[:, 2]
        c0, c1 = project(pr, np.array([[0, -40, zs.max() + 8], [0, -40, zs.min() - 8]]))
        p0, p1 = to_mm(c0, rect), to_mm(c1, rect)
        sh.line([p0[0], p1[0]], [p0[1], p1[1]], color="#7f7f7f", lw=0.45, ls=(0, (8, 2, 1.5, 2)), zorder=2)
        items = []
        lw_ = LABEL_W - 6
        for k in keys:
            a = to_mm(anchor_px(gg, ids, pr, set(names_of[k])), rect)
            if k in P_LABEL:
                nm, fn, sc = P_LABEL[k]
                lines = [(nm, 9.0, True, INK), (fn, 7.5, False, "#1f4e9a")] + [(t, 7.2, False, GREY) for t in wrap(sh, sc, 7.2, lw_)]
            else:
                nm, l2 = S_LABEL[k]
                lines = [(nm, 8.6, True, "#000000")] + [(t, 7.2, False, GREY) for t in wrap(sh, l2, 7.2, lw_)]
            items.append(dict(anchor=a, lines=lines, servo=k not in P_LABEL))
        place_labels(sh, items, rect[2] + 5.0, "R", ya, box_y1 - 3, gap=3.0)
        # assembly arrows
        for up_k, lo_k, dirn, step in ARROWS:
            if up_k not in keys or lo_k not in keys:
                continue
            zu = part_pts(gg, set(names_of[up_k]))[:, 2].min()
            zl = part_pts(gg, set(names_of[lo_k]))[:, 2].max()
            # arrow beside the chain at the back (x -32): left on the page, away from the leader lines
            x_ar = -32.0
            q = project(pr, np.array([[x_ar, -40, zl + 1.5], [x_ar, -40, zu - 1.5]]))
            qa, qb = to_mm(q[0], rect), to_mm(q[1], rect)
            if dirn == "down":
                qa, qb = qb, qa
            sh.arrow(qa, qb)
            sh.text(min(qa[0], qb[0]) - 1.2, (qa[1] + qb[1]) / 2, str(step), size=7.5, bold=True, color=RED, va="center", ha="right")
        for k, step in SIDE_ARROWS:
            if k not in keys:
                continue
            P = part_pts(gg, set(names_of[k]))
            host = dict((kk, h) for kk, _, h, _ in LEG_BESIDE)[k]
            zc = (P[:, 2].min() + P[:, 2].max()) / 2
            xmin, xh = P[:, 0].min(), part_pts(gg, set(names_of[host]))[:, 0].max()
            q = project(pr, np.array([[xmin - 3, -40, zc], [xh + 4, -40, zc]]))
            qa, qb = to_mm(q[0], rect), to_mm(q[1], rect)
            sh.arrow(qa, qb)
            sh.text((qa[0] + qb[0]) / 2, min(qa[1], qb[1]) - 0.6, str(step), size=7.5, bold=True, color=RED, ha="center", va="bottom")

    sh.text(rect_b[0], ya - 1.0, "↑ ③ 허벅지 프레임은 왼쪽 칸 맨 아래 고관절 피치 서보의 혼에 붙습니다 (조립 7)", size=7.2, color=RED, va="bottom")
    yy = rect_a[3] + 4
    sh.text(xa, yy, "→ 오른쪽 칸 ③ 허벅지 프레임으로 이어짐", size=7.2, color=RED)
    yy += 7
    sh.text(xa, yy, "다리 조립 순서 (out20/LEG_PRINT_KIT_R.md 3절, 화살표 옆 숫자)", size=8.2, bold=True)
    yy += 4.2
    for i, st in enumerate(STEPS, 1):
        sh.text(xa, yy, f"{i:>2}. {st}", size=7.4, color="#333333")
        yy += 3.4
    assert yy < box_y1, "step list runs past the leg panel"

    # ================================================================ 2. whole robot exploded (right top, simplified)
    RX0, RX1 = 266, 414
    sh.line([264, 264], [15, 293], lw=0.6)
    sh.text(RX0, 17.5, "나. 전체 분해도 — 묶음이 붙는 곳", size=10, bold=True)
    wg = moved(geom, whole_offsets(geom))
    EW, AW = 15, -35
    ew, eh = extent(wg, EW, AW)
    box_w, box_h = 84, 104
    sc_w = min(box_w / ew, box_h / eh) / PX_MM
    img_w, ids_w, pr_w = do_render(r, wg, EW, AW, sc_w, "whole")
    wx = RX0 + 50
    rect_w = sh.image(img_w, wx, 24)
    groups = [
        ("머리 묶음", "머리 뒤 · 앞 · 얼굴판 · 볏 · 귀 · 끄덕임 서보 ID 15", "목 축 Ø8 (688 베어링 ×2) 에 끼움",
         {g[1] for g in wg if g[0] in ("head", "ears")}, "L"),
        ("몸통 묶음", "몸통 프레임 · 선반 · 뚜껑 · 앞뒤 덮개 · 반바지", "고관절 요 서보 ID 1 · 7 이 안에",
         {"torso_cover_front", "torso_cover_back", "lid"}, "L"),
        ("오른 팔 묶음", "팔 서보 ID 13 · 껍데기 · 손", "어깨 축 Ø8 · 혼 어댑터",
         {g[1] for g in wg if g[0] == "R_arm"}, "L"),
        ("왼 팔 묶음", "팔 서보 ID 14", "", {g[1] for g in wg if g[0] == "L_arm"}, "R"),
        ("오른 다리 묶음", "→ 가. 오른 다리 분해도", "요 허브 → 6805 베어링",
         {g[1] for g in wg if g[0].startswith("R_") and "arm" not in g[0] and "flap" not in g[0]}, "L"),
        ("왼 다리 묶음", "대칭 (L_ 파일, ID 7–12)", "", {g[1] for g in wg if g[0].startswith("L_") and "arm" not in g[0] and "flap" not in g[0]}, "R"),
    ]
    li, ri = [], []
    for nm, l2, l3, names, side in groups:
        a = to_mm(anchor_px(wg, ids_w, pr_w, names), rect_w)
        mw = 44 if side == "L" else 30
        lines = [(nm, 8.6, True, INK)] + [(t, 7.0, False, GREY) for x in (l2, l3) if x for t in wrap(sh, x, 7.0, mw)]
        (li if side == "L" else ri).append(dict(anchor=a, lines=lines))
    place_labels(sh, li, rect_w[0] - 1.5, "L", 24, 128)
    place_labels(sh, ri, rect_w[2] + 1.5, "R", 24, 128)

    # ================================================================ 3. assembled front / side (right middle)
    y3 = 133
    sh.line([264, 416], [y3 - 2, y3 - 2], lw=0.6)
    sh.text(RX0, y3, f"다. 조립 완성 — 전체 높이 {height:.0f} mm (발바닥 고무 패드 포함, 설계 형상)", size=10, bold=True)
    ga_all = [g for g in geom]
    fw, fh = extent(ga_all, 0, 0)
    sw_, shh = extent(ga_all, 0, -90)
    hbox = 56
    sc_f = hbox / max(fh, shh) / PX_MM
    img_f, _, pr_f = do_render(r, ga_all, 0, 0, sc_f, "front")
    img_s, _, pr_s = do_render(r, ga_all, 0, -90, sc_f, "side")
    fx = RX0 + 14
    rf = sh.image(img_f, fx, y3 + 7)
    rs = sh.image(img_s, rf[2] + 18, y3 + 7)
    sh.text((rf[0] + rf[2]) / 2, rf[3] + 0.8, "정면", size=7.5, ha="center")
    sh.text((rs[0] + rs[2]) / 2, rs[3] + 0.8, "오른쪽 옆 (앞이 오른쪽)", size=7.5, ha="center")
    # height dimension on the front view
    top = to_mm(project(pr_f, np.array([[0, 0, z_hi]]))[0], rf)[1]
    bot = to_mm(project(pr_f, np.array([[0, 0, z_lo]]))[0], rf)[1]
    xd = rf[0] - 5
    for yy in (top, bot):
        sh.line([xd - 1.5, rf[0] + 2], [yy, yy], lw=0.4, color=GREY)
    sh.ax.annotate("", xy=(xd, top), xytext=(xd, bot), zorder=6, arrowprops=dict(arrowstyle="<|-|>,head_length=0.4,head_width=0.18", lw=0.6, color=INK))
    sh.text(xd - 1.2, (top + bot) / 2, f"{height:.0f}", size=7.5, ha="right", va="center", rotation=90)
    # leg length marks on the side view: hip pitch axis -> ankle axis
    jz = {j["name"]: j["point_mm"][2] for j in r.joints}
    yk = [to_mm(project(pr_s, np.array([[0, -40, jz[n]]]))[0], rs)[1] for n in ("R_hip_pitch", "R_knee", "R_ankle_pitch")]
    xs = rs[2] + 3
    for yy, n in zip(yk, ("고관절 피치 축", "무릎 축", "발목 축")):
        sh.line([rs[2] - 6, xs], [yy, yy], lw=0.35, color=GREY, ls=(0, (2, 1.5)))
        sh.text(xs + 0.8, yy, n, size=7, va="center", color=GREY)
    sh.text(xs + 0.8, (yk[0] + yk[1]) / 2, f"{jz['R_hip_pitch'] - jz['R_knee']:.0f}", size=7, va="center")
    sh.text(xs + 0.8, (yk[1] + yk[2]) / 2, f"{jz['R_knee'] - jz['R_ankle_pitch']:.0f}", size=7, va="center")

    # ================================================================ 4. servo table + other parts list (right bottom)
    y4 = 210
    sh.line([264, 416], [y4 - 2, y4 - 2], lw=0.6)
    sh.text(RX0, y4, "라. 서보 15개 (모두 XL430-W250-T)", size=9, bold=True)
    rows = [("고관절 요", "1", "7"), ("고관절 롤", "2", "8"), ("고관절 피치", "3", "9"), ("무릎", "4", "10"),
            ("발목 피치", "5", "11"), ("발목 롤", "6", "12"), ("팔 흔들기", "13", "14"), ("머리 끄덕임 (크랭크)", "15", "–")]
    cx = [RX0, RX0 + 30, RX0 + 42, RX0 + 54]
    yy = y4 + 5.2
    for i, h in enumerate(("관절", "오른쪽", "왼쪽", "수량")):
        sh.text(cx[i], yy, h, size=7.2, bold=True)
    sh.line([RX0, RX0 + 62], [yy + 3.6, yy + 3.6], lw=0.4)
    yy += 4.2
    for nm, a, b in rows:
        sh.text(cx[0], yy, nm, size=7.2); sh.text(cx[1], yy, a, size=7.2); sh.text(cx[2], yy, b, size=7.2)
        sh.text(cx[3], yy, "1" if b == "–" else "2", size=7.2)
        yy += 3.35
    sh.line([RX0, RX0 + 62], [yy + 0.2, yy + 0.2], lw=0.4)
    sh.text(cx[0], yy + 0.8, "합계", size=7.2, bold=True); sh.text(cx[3], yy + 0.8, "15", size=7.2, bold=True)
    yy += 5.6
    for t in ("구멍 지름: M2.5 몸체 나사 2.7 mm (시험편 실측)", "M2 혼 나사 2.4 mm (추정)", "서보 칸 한쪽 틈 0.35 mm 이상 (공차 5절)",
              "XL430 몸체 · 혼 나사 자리는 실물 확인 전 [추정]"):
        sh.text(cx[0], yy, t, size=7.0, color="#333333")
        yy += 3.2
    # other printed parts (names + files), compact
    ox = RX0 + 66
    sh.text(ox, y4, "마. 몸통 · 팔 · 머리 출력 부품 (39개)", size=9, bold=True)
    groups_txt = [
        ("몸통 (11)", ["몸통 프레임 torso_frame", "전장 선반 electronics_shelf", "선반 핀 shelf_pins", "뚜껑 lid",
                      "앞 덮개 torso_cover_front", "뒤 덮개 torso_cover_back", "반바지 shorts", "목 요크 neck_yoke",
                      "골반 해치 pelvis_hatch", "팔 혼 어댑터 ×2", "  arm_R/L_horn_adapter"]),
        ("팔 (10, 좌우 arm_R_ · arm_L_)", ["안쪽 껍데기 shell_in", "바깥 껍데기 shell_out", "손 hand", "손가락 막대 finger_rod", "손 날개 flap"]),
        ("머리 (18)", ["머리 뒤 head_back", "머리 앞 head_front", "얼굴판 face_plate", "눈 ×2 eye_L/R", "입 mouth",
                      "카메라 클립 camera_clip", "볏 crest", "끄덕임 크랭크 nod_crank", "끄덕임 막대 nod_rod",
                      "귀 ×2 ear_L/R", "귀 원판 ×2 ear_disc_L/R", "안테나 ×2 antenna_L/R", "귀 흔들개 ear_rocker", "귀 막대 ear_rod"]),
    ]
    cols = [[(t, b) for g, items in groups_txt[:2] for t, b in [(g, True)] + [(x, False) for x in items]],
            [(t, b) for g, items in groups_txt[2:] for t, b in [(g, True)] + [(x, False) for x in items]]]
    colw = (RX1 - ox) / 2
    for c, part in enumerate(cols):
        yy = y4 + 5.2
        for t, b in part:
            sh.text(ox + c * colw, yy, t, size=7.0, bold=b, color=INK if b else "#333333")
            yy += 3.3 if b else 3.15
    sh.text(RX1, 289.5, "파일: out20/print_parts/<이름>.stp, 다리는 out20/print_parts_stl/<이름>.stl", size=7.0, color=GREY, ha="right", va="bottom")

    # ---- order list under the leg (fits in the column gap, bottom-left)
    sh.fig.canvas.draw()
    return sh, (rect_a, rect_b, rect_w, rf, rs)


def check_and_save(sh, rects):
    rend = sh.fig.canvas.get_renderer()
    boxes = []
    for t in sh.texts:
        if not t.get_text():
            continue
        bb = t.get_window_extent(renderer=rend)
        # display px -> paper mm (y down)
        x0, x1 = bb.x0 / DPI * 25.4, bb.x1 / DPI * 25.4
        y0, y1 = 297 - bb.y1 / DPI * 25.4, 297 - bb.y0 / DPI * 25.4
        boxes.append((x0, y0, x1, y1, t.get_text()))
    bad = []
    for i in range(len(boxes)):
        a = boxes[i]
        if a[0] < 4 or a[2] > 416 or a[1] < 4 or a[3] > 293:
            bad.append(("off page", a[4]))
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if a[0] < b[2] - 0.1 and b[0] < a[2] - 0.1 and a[1] < b[3] - 0.1 and b[1] < a[3] - 0.1:
                bad.append(("overlap", a[4][:30], b[4][:30]))
    for b in bad:
        print("LAYOUT", *b)
    png = os.path.join(OUTD, "kitoru_v20_assembly.png")
    sh.fig.savefig(png, dpi=DPI, facecolor="white")
    sh.fig.savefig(os.path.join(OUTD, "kitoru_v20_assembly.pdf"), dpi=DPI, facecolor="white")
    print("saved", png, "layout problems", len(bad))


if __name__ == "__main__":
    sh, rects = main()
    check_and_save(sh, rects)
