"""키토루 v21 조립도 (A3 가로, 네 장) -> out20/assembly_drawing/kitoru_v21_assembly_p<N>_<이름>.png (300 dpi)
                                    + kitoru_v21_assembly.pdf (네 장 묶음)

usage (code_v17/):  ASM_CACHE=<dir> ../sim/.venv/bin/python assembly_drawing21.py
  - 형상: build21_arm.build_model() 하나 = 다리 최신(build20: print_robust · no_support) + 몸통 · 머리 2차
    (build21_body) + 팔꿈치 팔(build21_arm). render6 툰 렌더러, 모든 관절 0°. 부품을 이름으로 옮겨 분해도를 만든다.
    부품 STEP / STL / PRINT_v2 는 쓰지 않는다(읽기만: PRINT_v2 STL 이름 52개를 모델과 대조).
  - 장: 1 다리(오른쪽) · 2 몸통 · 3 머리 · 4 팔 + 전체(완성 정면 · 옆, 서보 17개 표, 나사 · 구매품 합계).
  - 조립 순서 · 나사 출처(최신 절 우선):
      다리  out20/LEG_PRINT_KIT_R.md 「새 조립 순서」(11-27) + 11-28 「발판 혼 판 · 발목 롤 나사 8개」(발목 롤 M2.5×16 ×8)
            + 11-28 「조립 뒤 범위 확인」 + 11-28 「서포트 없애기」(발목 서보 케이블 길)
      몸통 · 머리  out20/BODY_HEAD_PRINT_KIT.md 0.0 · 0.5 · 0.6 절
      팔    out21_arm/ARM_ELBOW_v21.md 0절(11-28 내구 보강) + 3 · 4절
  - 서보 ID: robot/config/joints_v20.yaml (자리표시). 팔꿈치 서보는 yaml 에 없어 「미정」. 모든 서보 XL430-W250-T.
  - ASM_CACHE=<dir>: 모델 피클(model.pkl, 빌드 약 90 초)과 렌더(키 = 보는 각 · 배율 · 부품 · 옮긴 거리의 해시)를
    저장 · 재사용한다. 배치만 고칠 때 다시 빌드 · 렌더하지 않는다. 지우는 것은 사용자 몫.
"""
import os, sys, json, math, hashlib, pickle, types, subprocess, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTD = os.path.join(ROOT, "out20", "assembly_drawing")
PRINT_DIR = os.path.join(ROOT, "out20", "PRINT_v2")
CACHE = os.environ.get("ASM_CACHE")
DPI = 300
PX_MM = 25.4 / DPI
PT = 25.4 / 72
INK, GREY, RED, BLUE = "#191919", "#4d4d4d", "#c0392b", "#1f4e9a"
MIN_PT = 7.0                       # smallest text on the sheet (A3 print)


# ---------------------------------------------------------------- model (build once, pickle under ASM_CACHE)
def build_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "log", "-1", "--format=%h %ad", "--date=short", "--",
                                        "code_v17/build20.py", "code_v17/build21_body.py", "code_v17/build21_arm.py",
                                        "code_v17/arm21.py"], text=True).strip()
    except Exception:
        return "?"


def load_model():
    """model pickle under ASM_CACHE, tagged with the build commit: rebuilt when the build code changed"""
    commit = build_commit()
    fp = os.path.join(CACHE, "model.pkl") if CACHE else None
    if fp and os.path.exists(fp):
        d = pickle.load(open(fp, "rb"))
        if isinstance(d, dict) and d.get("commit") == commit:
            return d["model"]
        print("model cache is from another build commit -> rebuilding")
    import build21_arm
    m = build21_arm.build_model()
    if fp:
        os.makedirs(CACHE, exist_ok=True)
        pickle.dump({"commit": commit, "model": m}, open(fp, "wb"))
    return m


MODEL = load_model()
_shim = types.ModuleType("asm21_model")
_shim.build_model = lambda: MODEL
sys.modules["asm21_model"] = _shim
os.environ["BUILD"] = "asm21_model"          # render6 / assembly_drawing20 import the model through this name
import assembly_drawing20 as A20             # noqa: E402  (helpers: Sheet, fonts, place_labels, moved, part_pts, ...)
from assembly_drawing20 import moved, part_pts, extent, project, Sheet  # noqa: E402


def do_render(r, geom, elev, azim, scale, tag):
    """render with a cache key that changes when the geometry placement / view / scale changes"""
    key = repr((tag, elev, azim, round(scale, 6), sorted((g[1], round(float(np.asarray(g[3]).sum()), 3) if len(g[3]) else 0.0)
                                                      for g in geom)))
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    if CACHE:
        fp = os.path.join(CACHE, f"r21_{tag}_{h}.pkl")
        if os.path.exists(fp):
            return pickle.load(open(fp, "rb"))
    old = r.GEOM[:]
    r.GEOM[:] = geom
    try:
        img, ids, pr = r.render_ids(elev, azim, {}, scale=scale)
    finally:
        r.GEOM[:] = old
    pr = tuple(np.asarray(x) if isinstance(x, np.ndarray) else x for x in pr)
    if CACHE:
        pickle.dump((img, ids, pr), open(fp, "wb"))
    return img, ids, pr


FALLBACKS = []


def anchor_px(geom, ids, pr, names, label=""):
    """visible pixel deepest inside the parts' silhouette; prints a warning when the parts are hidden"""
    from scipy.ndimage import distance_transform_edt
    pids = {g[2] for g in geom if g[1] in names}
    m = np.isin(ids, list(pids))
    if m.sum() > 20:
        d = distance_transform_edt(m)
        y, x = np.unravel_index(np.argmax(d), d.shape)
        return np.array([x, y], float)
    FALLBACKS.append(label or sorted(names)[0])
    print("ANCHOR FALLBACK (part hidden):", label, sorted(names)[:3])
    return project(pr, part_pts(geom, names).mean(0))[0]


# ---------------------------------------------------------------- text helpers
def wrap_words(sh, s, size, maxw, bold=False):
    """wrap at spaces so every line fits maxw (paper mm); a single long word stays on its own line"""
    out = []
    for para in s.split("\n"):
        cur = ""
        for w in para.split(" "):
            cand = w if not cur else cur + " " + w
            if cur and sh.text_w(cand, size, bold) > maxw:
                out.append(cur)
                cur = w
            else:
                cur = cand
        out.append(cur)
    return out


def para(sh, x, y, s, size=7.4, maxw=100, color="#262626", bold=False, indent=0.0, lh=1.30):
    """paragraph; first line at x, continuation lines at x + indent. returns the next y"""
    lines = wrap_words(sh, s, size, maxw - indent, bold)
    first = True
    for ln in lines:
        sh.text(x if first else x + indent, y, ln, size=size, bold=bold, color=color)
        y += size * PT * lh
        first = False
    return y


def steps_block(sh, x, y, steps, maxw, size=7.3, num_color=RED):
    """numbered steps: (num, bold head, rest). returns next y"""
    for num, head, rest in steps:
        nw = 5.2
        sh.text(x, y, str(num), size=size + 0.4, bold=True, color=num_color)
        lines = wrap_words(sh, head + (" " + rest if rest else ""), size, maxw - nw)
        # bold head on the first line only when it fits there entirely
        yy = y
        for i, ln in enumerate(lines):
            if i == 0 and ln.startswith(head):
                sh.text(x + nw, yy, head, size=size, bold=True)
                tail = ln[len(head):]
                if tail:
                    sh.text(x + nw + sh.text_w(head, size, True), yy, tail, size=size, color="#262626")
            else:
                sh.text(x + nw, yy, ln, size=size, color="#262626")
            yy += size * PT * 1.30
        y = yy + 0.9
    return y


def table(sh, x, y, cols, rows, size=7.2, head_bold=True, lh=1.36, widths=None, colors=None, wrapw=None):
    """cols: header strings; widths: column widths mm; rows: list of tuples (cells may wrap if wrapw[i]).
    returns next y"""
    xs = [x]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    tw = sum(widths)
    for i, c in enumerate(cols):
        sh.text(xs[i], y, c, size=size, bold=head_bold)
    y += size * PT * lh
    sh.line([x, x + tw], [y - 0.5, y - 0.5], lw=0.4)
    for ri, row in enumerate(rows):
        hmax = 1
        cells = []
        for i, c in enumerate(row):
            ww = widths[i] - 1.2
            ls = wrap_words(sh, str(c), size, ww) if c != "" else [""]
            cells.append(ls)
            hmax = max(hmax, len(ls))
        for i, ls in enumerate(cells):
            col = (colors[ri][i] if colors and colors[ri] and colors[ri][i] else "#262626")
            bold = bool(colors and colors[ri] and colors[ri][i] == "B")
            if bold:
                col = INK
            for k, ln in enumerate(ls):
                if ln:
                    sh.text(xs[i], y + k * size * PT * 1.25, ln, size=size, color=col, bold=bold)
        y += hmax * size * PT * 1.25 + 0.8
        sh.line([x, x + tw], [y - 0.5, y - 0.5], lw=0.2, color="#bbbbbb")
    return y


def box_title(sh, x, y, s, size=9.5):
    sh.text(x, y, s, size=size, bold=True)
    return y + size * PT * 1.45


def header(sh, title, sub, page, commit):
    sh.rect(4, 4, 412, 289, lw=0.9)
    sh.line([4, 416], [15, 15], lw=0.7)
    sh.text(8, 6.0, title, size=17, bold=True)
    sh.text(8 + sh.text_w(title, 17, True) + 5, 7.6, sub, size=9, color=GREY)
    sh.text(412, 6.2, f"{page} / 4 장   2026-11-28   빌드 커밋 {commit} (code_v17/build21_arm.py 모델)   A3 · 단위 mm", size=8.2, ha="right")
    sh.text(412, 10.4, "서보는 모두 XL430-W250-T 17개 · 서보 ID 는 자리표시(robot/config/joints_v20.yaml, 조립 후 확정)",
            size=7.4, ha="right", color=RED)


def to_mm(px, rect):
    return (rect[0] + px[0] * PX_MM, rect[1] + px[1] * PX_MM)


IMAGES = []                       # (sheet tag, rect, ids) for the text-over-part check
SHEET_N = [0]


def new_sheet(plt, F):
    sh = Sheet(plt, F)
    sh.vdiv = []
    SHEET_N[0] += 1
    sh.tag = SHEET_N[0]
    return sh


def place_image(sh, img, ids, x, y):
    rect = sh.image(img, x, y)
    IMAGES.append((sh.tag, rect, ids))
    return rect


def label_lines(sh, name, file_, info, step, maxw, servo=False):
    lines = [(name, 8.4, True, INK)]
    if file_:
        for t in (file_ if isinstance(file_, (list, tuple)) else [file_]):
            lines.append((t, 7.2, False, BLUE))
    for t in wrap_words(sh, info, 7.1, maxw):
        lines.append((t, 7.1, False, GREY))
    if step:
        lines.append((step, 7.2, True, RED))
    return lines


def arrow_model(sh, pr, rect, p0, p1, text=None, text_side="left", color=RED, lw=1.5):
    q = project(pr, np.array([p0, p1]))
    a, b = to_mm(q[0], rect), to_mm(q[1], rect)
    sh.arrow(a, b, color=color, lw=lw)
    if text is not None:
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if text_side == "left":
            sh.text(min(a[0], b[0]) - 1.0, my, str(text), size=7.6, bold=True, color=color, ha="right", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
        elif text_side == "right":
            sh.text(max(a[0], b[0]) + 1.0, my, str(text), size=7.6, bold=True, color=color, ha="left", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
        else:
            sh.text(mx, min(a[1], b[1]) - 0.8, str(text), size=7.6, bold=True, color=color, ha="center", va="bottom",
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.85))
    return a, b


# ---------------------------------------------------------------- generic exploded panel
def stack_offsets(geom, stack, anchor, clr):
    """stack: list top -> bottom of (key, names, (dx, dy)). The anchor item stays; items above go up, items below go
    down, each clr above / below its neighbour. Returns {part: (dx, dy, dz)}, {key: dz}"""
    zr = {k: (lambda P: (P[:, 2].min(), P[:, 2].max()))(part_pts(geom, set(ns))) for k, ns, _ in stack}
    keys = [k for k, _, _ in stack]
    ia = keys.index(anchor)
    dz = {anchor: 0.0}
    top = zr[anchor][1]
    for k in reversed(keys[:ia]):                # going up
        lo, hi = zr[k]
        dz[k] = max(0.0, (top + clr) - lo)
        top = hi + dz[k]
    bot = zr[anchor][0]
    for k in keys[ia + 1:]:                      # going down
        lo, hi = zr[k]
        dz[k] = min(0.0, (bot - clr) - hi)
        bot = lo + dz[k]
    offs = {}
    for k, ns, dxy in stack:
        for n in ns:
            offs[n] = (dxy[0], dxy[1], dz[k])
    return offs, dz


def explode_panel(sh, r, geom, items, offs, elev, azim, box, tag, label_w=(48, 48), img_y=None, label_gap=2.4,
                  max_scale=None):
    """items: list of dict(key, names, side 'L'/'R', lines) -> renders the moved geometry fitted into box between two
    label columns, places labels. returns (rect, ids, pr, gm, scale_mm)"""
    keep = set()
    for it in items:
        keep |= set(it["names"])
    for it in items:
        keep |= set(it.get("also", ()))
    gm = moved(geom, offs, keep)
    x0, y0, x1, y1 = box
    wm, hm = extent(gm, elev, azim)
    avail_w = (x1 - x0) - label_w[0] - label_w[1] - 8
    avail_h = (y1 - y0) - 2
    pscale = min(avail_w / wm, avail_h / hm) * 0.97
    if max_scale:
        pscale = min(pscale, max_scale)
    img, ids, pr = do_render(r, gm, elev, azim, pscale / PX_MM, tag)
    w, h = img.shape[1] * PX_MM, img.shape[0] * PX_MM
    ix = x0 + label_w[0] + 4 + (avail_w - w) / 2
    iy = y0 + (avail_h - h) / 2 if img_y is None else img_y
    rect = place_image(sh, img, ids, ix, iy)
    L, R = [], []
    for it in items:
        if not it.get("lines"):
            continue
        if it.get("anchor_model") is not None:
            a = to_mm(project(pr, np.array([it["anchor_model"]]))[0], rect)
        else:
            a = to_mm(anchor_px(gm, ids, pr, set(it["names"]), it["key"]), rect)
        d = dict(anchor=a, lines=it["lines"], servo=it.get("servo", False))
        (L if it["side"] == "L" else R).append(d)
    if L:
        A20.place_labels(sh, L, rect[0] - 2.0, "L", y0, y1, gap=label_gap)
    if R:
        A20.place_labels(sh, R, rect[2] + 2.0, "R", y0, y1, gap=label_gap)
    return rect, ids, pr, gm, pscale


# ================================================================ sheet 1: right leg
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
# shells go on from the front (+x) now (screws, no snap): drawn in front of their frame at the same height
LEG_BESIDE = [("thigh_shell", ["R_thigh_shell"], "thigh_frame", 78.0),
              ("shank_shell", ["R_shank_shell"], "shank_frame", 82.0),
              ("ankle_roll_servo", ["R_ankle_roll_servo_XM430", "R_ankle_roll_horn", "R_ankle_roll_idler"], "ankle_bracket", 150.0)]
SPLIT_AFTER = "pitch_servo"
LEG_CLR = 20.0
F1 = "1_다리_오른쪽/"
# (name, STL, screws / notes, step)
LEG_P = {
    "yaw_bracket": ("고관절 요 브래킷", F1 + "R_hip_yaw_bracket.stl",
                    "허브 바깥에 2액 에폭시 → 6805 · 아래 카운터보어로 요 혼에 M2×5 ×4 (드라이버 날 22 mm 이상)", "단계 11 (다리는 단계 12)"),
    "roll_bracket": ("고관절 롤 U 브래킷", F1 + "R_hip_roll_bracket.stl", "롤 혼 M2×5 ×4 (앞판 밖) · 아이들러 M2×5 ×4 (뒤판 밖)", "단계 10"),
    "thigh_frame": ("허벅지 프레임", F1 + "R_thigh_frame.stl", "고관절 피치 혼에 M2×5 ×4", "단계 9"),
    "thigh_shell": ("허벅지 셸 (흰색 덮개)", F1 + "R_thigh_shell.stl", "앞에서 대고 M2.5×8 ×2 · 휘어서 끼우지 않음", "단계 7"),
    "shank_frame": ("정강이 프레임", F1 + "R_shank_frame.stl", "무릎 혼에 M2×5 ×4", "단계 8"),
    "shank_shell": ("정강이 셸 (하늘색 덮개)", F1 + "R_shank_shell.stl", "뒤가 열린 U · 앞에서 감싸 M2.5×8 ×2 (셸 앞판 바깥에서)", "단계 5"),
    "ankle_bracket": ("발목 브래킷", F1 + "R_ankle_bracket.stl", "발목 피치 혼에 M2×5 ×3 (카운터보어 6.1 mm 안에서)", "단계 4"),
    "foot": ("발판 (채움 100 %)", F1 + "R_foot_sole.stl", "발목 롤 혼에 M2×5 ×4 (카운터보어 안) · 고무 패드 Ø12×6 ×5", "단계 2"),
}
LEG_S = {
    "yaw_servo": ("고관절 요 서보 XL430 · ID 1", "몸통 프레임 안 · 바깥 옆벽에서 M2.5×8 ×4 (몸통 장에서 먼저)", "몸통 장 단계 3"),
    "bearing": ("6805 베어링 (구매품)", "몸통 골반판에 손으로 넣음 · 요 브래킷 허브가 들어감 (에폭시)", "단계 11"),
    "roll_servo": ("고관절 롤 서보 XL430 · ID 2", "혼 · 아이들러를 달고 U 브래킷 두 판 사이로 위에서 · 요 브래킷 옆판 바깥에서 M2.5×8 ×8 (안쪽 4개는 반대쪽 다리 전에)", "단계 10 · 12"),
    "pitch_servo": ("고관절 피치 서보 XL430 · ID 3", "U 브래킷 아래에서 두 벽 사이로 · 벽 바깥 M2.5×10 ×4 + 바닥판 위 M2.5×6 ×2", "단계 9"),
    "knee_servo": ("무릎 서보 XL430 · ID 4", "허벅지 프레임 아래 칸 · M2.5×8 ×8 (앞 4개는 허벅지 셸 전에)", "단계 6"),
    "ankle_pitch_servo": ("발목 피치 서보 XL430 · ID 5", "정강이 프레임 아래 칸에 아래에서 · M2.5×8 ×8 (뒤 4개는 발목 브래킷 전, 앞 4개는 셸 전)", "단계 3"),
    "ankle_roll_servo": ("발목 롤 서보 XL430 · ID 6", "앞에서 위 · 아래 스페이서 사이로 밀어 넣고 아이들러를 뒤판 포켓에 · M2.5×16 ×8 (위 4개는 귀 바깥에서)", "단계 1"),
}
LEG_ARROWS = [("yaw_servo", "bearing", "up", 11), ("bearing", "yaw_bracket", "up", 11), ("yaw_bracket", "roll_servo", "up", 12),
              ("roll_servo", "roll_bracket", "down", 10), ("roll_bracket", "pitch_servo", "up", 9),
              ("pitch_servo", "thigh_frame", "up", 9), ("thigh_frame", "knee_servo", "up", 6), ("knee_servo", "shank_frame", "up", 8),
              ("shank_frame", "ankle_pitch_servo", "up", 3), ("ankle_pitch_servo", "ankle_bracket", "up", 4),
              ("ankle_bracket", "foot", "up", 2)]
LEG_SIDE_ARROWS = [("thigh_shell", 7), ("shank_shell", 5), ("ankle_roll_servo", 1)]
LEG_STEPS = [
    (1, "발목 롤 서보 → 발목 브래킷:", "앞에서 위 · 아래 스페이서 사이로 밀어 넣고 아이들러를 뒤판 포켓에 맞춤. M2.5×16 ×8 (위 4개는 귀 바깥에서)."),
    (2, "발판 → 발목 롤 혼:", "M2×5 ×4, 카운터보어 안에서."),
    (3, "발목 피치 서보 → 정강이 프레임:", "아래에서 밀어 넣고 M2.5×8 ×8. 뒤 4개는 발목 브래킷을 달기 전에, 앞 4개는 정강이 셸을 대기 전에."),
    (4, "발목 브래킷 → 발목 피치 혼:", "M2×5 ×3 (카운터보어 6.1 mm 안에서)."),
    (5, "정강이 셸:", "앞에서 감싸 대고 M2.5×8 ×2 (셸 앞판 바깥에서). 셸을 휘어서 끼우지 않음."),
    (6, "무릎 서보 → 허벅지 프레임:", "M2.5×8 ×8. 앞 4개는 허벅지 셸을 대기 전에."),
    (7, "허벅지 셸:", "앞에서 대고 M2.5×8 ×2."),
    (8, "정강이 프레임 → 무릎 혼:", "M2×5 ×4."),
    (9, "고관절 피치 서보 → U 브래킷:", "아래에서 두 벽 사이로, 벽 바깥 M2.5×10 ×4 + 바닥판 위 M2.5×6 ×2. 허벅지 프레임 → 피치 혼 M2×5 ×4."),
    (10, "고관절 롤 서보 → U 브래킷:", "혼 · 아이들러를 단 서보를 위에서 내려 넣고 롤 혼 M2×5 ×4 (앞판 밖), 아이들러 M2×5 ×4 (뒤판 밖). 여기까지 다리 한 덩어리."),
    (11, "고관절 요 브래킷 → 몸통:", "허브 바깥에 2액 에폭시를 얇게 바르고 6805 에 끼움. 아래에서 카운터보어로 요 혼에 M2×5 ×4. 굳기 전에 요 서보를 몇 번 돌려 중심을 맞추고, 굳을 때까지 힘 주지 않음."),
    (12, "다리 → 요 브래킷:", "롤 서보를 두 옆판 사이로 아래에서 밀어 올려 옆판 바깥에서 M2.5×8 ×8. 안쪽 옆판 4개는 반대쪽 다리를 달기 전에."),
    (13, "케이블:", "허벅지 뒤 · 정강이 뒤 터널에 케이블 타이(폭 2.5 이하)로 묶음."),
]


def sheet_leg(plt, F, r, geom, commit):
    sh = new_sheet(plt, F)
    header(sh, "키토루 v21 조립도", "1. 다리 (오른쪽을 그림, 왼쪽은 대칭)", 1, commit)
    # ---- exploded leg (two columns) on the left
    zr = {}
    for key, names in LEG_STACK:
        P = part_pts(geom, set(names)); zr[key] = (P[:, 2].min(), P[:, 2].max())
    offs, dz, prev_bot = {}, {}, None
    for key, names in LEG_STACK:
        lo, hi = zr[key]
        d = 0.0 if prev_bot is None else (prev_bot - LEG_CLR) - hi
        dz[key] = d; prev_bot = lo + d
        for n in names:
            offs[n] = (0.0, 0.0, d)
    for key, names, host, dx in LEG_BESIDE:
        dz[key] = dz[host]
        for n in names:
            offs[n] = (dx, 0.0, dz[host])
    leg_names = {n for _, ns in LEG_STACK for n in ns} | {n for _, ns, _, _ in LEG_BESIDE for n in ns}
    lg = moved(geom, offs, leg_names)
    keys_all = [k for k, _ in LEG_STACK]
    keys_a = keys_all[:keys_all.index(SPLIT_AFTER) + 1]
    keys_a += [k for k, _, h, _ in LEG_BESIDE if h in keys_a]
    names_of = {k: ns for k, ns in LEG_STACK}
    names_of.update({k: ns for k, ns, _, _ in LEG_BESIDE})
    set_a = {n for k in keys_a for n in names_of[k]}
    ga = [g for g in lg if g[1] in set_a]
    gb = [g for g in lg if g[1] not in set_a]
    EL, AZ = 18, -38
    bx0, by0, bx1, by1 = 6, 17, 262, 291
    sh.text(bx0 + 2, by0 + 1, "가. 오른 다리 분해도 — 빨간 화살표 = 끼우는 방향, 숫자 = 조립 단계 (오른쪽 칸 목록)", size=10, bold=True)
    sh.text(bx0 + 2, by0 + 6.0, "STL 은 out20/PRINT_v2/ 안. 왼 다리는 2_다리_왼쪽/ 의 L_ 파일, 서보 ID 7–12, 거울 대칭으로 같은 순서. "
            "왼쪽 검은 막대가 붙은 이름 = 서보 · 구매품.", size=7.4, color=GREY)
    LABEL_W = 60.0
    (wa_m, ha_m), (wb_m, hb_m) = extent(ga, EL, AZ), extent(gb, EL, AZ)
    ya = by0 + 13.5
    avail_h = (by1 - 3) - ya - 2
    avail_w = (bx1 - bx0) - 2 * LABEL_W - 12
    pscale = min(avail_h / max(ha_m, hb_m), avail_w / (wa_m + wb_m)) * 0.97
    scale = pscale / PX_MM
    img_a, ids_a, pr_a = do_render(r, ga, EL, AZ, scale, "legA")
    img_b, ids_b, pr_b = do_render(r, gb, EL, AZ, scale, "legB")
    wa, wb = img_a.shape[1] * PX_MM, img_b.shape[1] * PX_MM
    spare = (bx1 - bx0) - wa - wb - 2 * LABEL_W
    xa = bx0 + spare / 4
    xb = xa + wa + LABEL_W + spare / 2
    rect_a = place_image(sh, img_a, ids_a, xa, ya)
    rect_b = place_image(sh, img_b, ids_b, xb, ya)
    sh.line([xb - spare / 4, xb - spare / 4], [ya + 4, by1 - 4], color="#bbbbbb", lw=0.5, ls=(0, (4, 3)))
    sh.line([264, 264], [15, 293], lw=0.6); sh.vdiv.append((264, 15, 293))
    keys_b = [k for k in names_of if k not in keys_a]
    for (gg, ids, pr, rect, keys) in ((ga, ids_a, pr_a, rect_a, keys_a), (gb, ids_b, pr_b, rect_b, keys_b)):
        zs = part_pts(gg, {g[1] for g in gg})[:, 2]
        c0, c1 = project(pr, np.array([[0, -40, zs.max() + 8], [0, -40, zs.min() - 8]]))
        p0, p1 = to_mm(c0, rect), to_mm(c1, rect)
        sh.line([p0[0], p1[0]], [p0[1], p1[1]], color="#7f7f7f", lw=0.45, ls=(0, (8, 2, 1.5, 2)), zorder=2)
        items = []
        lw_ = LABEL_W - 5
        for k in keys:
            a = to_mm(anchor_px(gg, ids, pr, set(names_of[k]), k), rect)
            if k in LEG_P:
                nm, fn, sc, st = LEG_P[k]
                lines = label_lines(sh, nm, fn, sc, st, lw_)
            else:
                nm, l2, st = LEG_S[k]
                lines = label_lines(sh, nm, None, l2, st, lw_)
            items.append(dict(anchor=a, lines=lines, servo=k not in LEG_P))
        A20.place_labels(sh, items, rect[2] + 4.0, "R", ya, by1 - 3, gap=2.4)
        for up_k, lo_k, dirn, step in LEG_ARROWS:
            if up_k not in keys or lo_k not in keys:
                continue
            zu = part_pts(gg, set(names_of[up_k]))[:, 2].min()
            zl = part_pts(gg, set(names_of[lo_k]))[:, 2].max()
            p_lo, p_hi = [-34.0, -40, zl + 1.5], [-34.0, -40, zu - 1.5]
            if dirn == "down":
                p_lo, p_hi = p_hi, p_lo
            arrow_model(sh, pr, rect, p_lo, p_hi, step, "left")
        for k, step in LEG_SIDE_ARROWS:
            if k not in keys:
                continue
            P = part_pts(gg, set(names_of[k]))
            host = dict((kk, h) for kk, _, h, _ in LEG_BESIDE)[k]
            zc = (P[:, 2].min() + P[:, 2].max()) / 2
            if k == "ankle_roll_servo":
                zc = P[:, 2].min() + 6
            xmin, xh = P[:, 0].min(), part_pts(gg, set(names_of[host]))[:, 0].max()
            arrow_model(sh, pr, rect, [xmin - 3, -40, zc], [xh + 4, -40, zc], step, "top")
    sh.text(rect_b[0], ya - 0.8, "↑ 허벅지 프레임은 왼쪽 칸 맨 아래 고관절 피치 서보의 혼에 붙습니다 (단계 9)", size=7.2, color=RED, va="bottom")
    sh.text(xa, rect_a[3] + 3, "→ 오른쪽 칸 허벅지 프레임으로 이어짐", size=7.2, color=RED)
    # what one leg needs (bottom-left, under column A)
    yb = rect_a[3] + 11
    yb = box_title(sh, xa, yb, "다리 하나에 드는 것 (왼 다리도 같음)", 9.2)
    for t in ("M2.5×8 태핑 ×28 — 롤 8 · 무릎 8 · 발목 피치 8 · 셸 4", "M2.5×10 태핑 ×4 · M2.5×6 태핑 ×2 — 고관절 피치",
              "M2.5×16 태핑 ×8 — 발목 롤", "M2×5 혼 나사 ×27 — 혼 · 아이들러 (요 4 포함)", "6805 베어링 ×1 · 고무 패드 Ø12×6 ×5",
              "케이블 타이 (폭 2.5 이하) 2개 이상 · 2액 에폭시 조금", "서보 XL430-W250-T ×6 (요 서보 포함)",
              "요 서보 몸체 나사 M2.5×8 ×4 는 몸통 장에 셈"):
        sh.text(xa, yb, "•", size=7.6, color=RED, bold=True)
        sh.text(xa + 2.6, yb, t, size=7.6, color="#262626")
        yb += 7.6 * PT * 1.32
    assert yb < 291, yb

    # ---- right column: steps, cautions, range check, drill table, screws for one leg
    X0, W = 267, 146
    y = 17.5
    y = box_title(sh, X0, y, "나. 조립 순서 (오른 다리 한쪽, LEG_PRINT_KIT_R.md 최신 절)")
    y = para(sh, X0, y, "미리: 드릴 표대로 구멍을 맞춥니다. 서보 케이블은 칸에 넣기 전에 꽂습니다. 태핑 나사는 한 번 끝까지 넣어 "
             "나사산을 낸 뒤 빼고, 조립할 때 다시 넣어 손힘으로만 조입니다(전동 드라이버 금지).", 7.6, W)
    y += 0.8
    y = steps_block(sh, X0, y, LEG_STEPS, W, size=7.8)
    y += 0.6
    y = box_title(sh, X0, y, "다. 꼭 지킬 것", 9.2)
    notes = [
        "발판(R_foot_sole · L_foot_sole)은 채움 100 % 로 출력합니다. 나머지는 벽 4줄 이상 · 채움 20 % 이상 · 층 0.3 mm.",
        "서보는 칸에 밀어 넣고 몸체에 M2.5 태핑 나사(케이스 안 물림 4 mm 이하). 빡빡하면 칸 안쪽을 줄 · 사포로 갈고, 벽을 벌려 밀지 않습니다.",
        "혼 · 아이들러 나사는 M2×5, 카운터보어 안에서 조입니다. 태핑 나사가 헛돌면 멈추고 M3 로 바꾸지 말고 순간접착제 한 방울 뒤 다시.",
        "고관절 요 허브 ↔ 6805 는 2액 에폭시(한쪽 틈 0.4, 압입 아님). 굳을 때까지 힘 주지 않기.",
        "발목 서보 케이블: 발목 브래킷 안쪽 기둥이 서보 뒤로 돌리는 길을 막습니다 → 스페이서 뒤 틈(위로) · 케이스 케이블 구멍으로 뺍니다.",
    ]
    for t in notes:
        sh.text(X0, y, "•", size=7.6, color=RED, bold=True)
        y = para(sh, X0 + 2.6, y, t, 7.6, W - 2.6)
        y += 0.4
    y += 0.4
    y = box_title(sh, X0, y, "라. 조립 뒤 범위 확인 — 토크를 끄고 손으로 돌려 봅니다", 9.2)
    y = table(sh, X0, y, ["관절", "학습 한계", "형상에서 닿는 각", "닿는 부품 → 할 일"],
              [("고관절 피치", "−115.5 / +111.5°", "−117.6 / +114.5°", "허벅지 셸 ↔ 반바지. 학습 한계까지 안 닿으면 됨"),
               ("무릎", "115.5°", "117.6° (고관절 0°) · 113.8° (고관절 −117°)", "허벅지 ↔ 정강이 프레임. 고관절을 앞으로 다 올린 채 113° 까지"),
               ("발목 피치 위", "+30° 여유 없음", "+29.6 ~ +30°", "정강이 프레임 아래 ↔ 발판. +30° 전에 닿으면 정강이 프레임 아래 모서리를 갈아 +31° 까지"),
               ("발목 피치 아래", "−75°", "−76.4°", "정강이 셸 ↔ 발판")],
              size=7.4, widths=[20, 22, 38, 66], colors=[None, None, [None, None, None, None], None])
    y += 0.6
    y = para(sh, X0, y, "닿으면 닿는 자리를 줄 · 사포로 갈아 냅니다(다시 뽑지 않음). 특히 발목 피치 +30°.", 7.6, W, color=RED)
    y += 1.0
    y = box_title(sh, X0, y, "마. 드릴 표 · 다리 하나 나사", 9.2)
    y = table(sh, X0, y, ["구멍", "출력 → 드릴", "쓰는 것"],
              [("서보 몸체 나사 관통", "2.7 → 2.8", "M2.5 태핑 (서보 몸체에 나사산)"),
               ("혼 · 아이들러 나사 관통 / 카운터보어", "2.4 → 2.5 / 4.4 → 4.5", "M2×5"),
               ("셸 나사 구멍 / 셸 보스 밑구멍", "3.0 / 2.2 → 2.2–2.3 (깊이 6.4)", "M2.5×8 태핑"),
               ("요 허브 바깥 Ø24.2", "손대지 않음", "6805 안지름 Ø25 에 에폭시")],
              size=7.4, widths=[52, 46, 48])
    assert y < 291, f"leg right column overflows: {y:.1f}"
    return sh


# ================================================================ sheet 2: torso
F3 = "3_몸통_머리/"


def bullets(sh, x, y, notes, w, size=7.2):
    for t in notes:
        sh.text(x, y, "•", size=size, color=RED, bold=True)
        y = para(sh, x + 2.6, y, t, size, w - 2.6)
        y += 0.4
    return y


def sheet_torso(plt, F, r, geom, commit):
    sh = new_sheet(plt, F)
    header(sh, "키토루 v21 조립도", "2. 몸통 (몸통 · 머리 키트 0.5 절 단계 1–8)", 2, commit)
    YB = 236                                          # top of the text band
    sh.line([4, 416], [YB - 1, YB - 1], lw=0.6)
    sh.line([187, 187], [15, YB - 1], lw=0.6); sh.vdiv.append((187, 15, YB - 1))
    E, AZ = 20, -40
    lw = 50
    Lb = lambda *a: label_lines(sh, *a, lw - 4)
    frame_set = ["torso_frame", "yaw_bearing_R_6805", "yaw_bearing_L_6805", "shelf_dowels_3mm"]
    # ---------------- A: inside (steps 2-4)
    stack = [("shelf", ["electronics_shelf"], (0, 0)), ("yaw", ["yaw_servo_R_XL430", "yaw_servo_L_XL430"], (0, 0)),
             ("frame", frame_set, (0, 0)), ("hatch", ["pelvis_hatch", "hatch_magnets_6x3"], (0, 0))]
    offs, dz = stack_offsets(geom, stack, "frame", 18.0)
    items = [
        dict(key="shelf", names=["electronics_shelf"], side="R",
             lines=Lb("전자 선반", F3 + "electronics_shelf.stl", "강철 핀 Ø3×12 ×3 을 프레임 받침에 접착하고 얹음 · OpenCR 은 뒷벽 홈에 세움", "단계 4")),
        dict(key="yaw", names=["yaw_servo_R_XL430", "yaw_servo_L_XL430"], side="R", servo=True,
             lines=Lb("고관절 요 서보 XL430 ×2", None,
                      "ID 1 (오른) · 7 (왼) · 프레임 안에 놓고 바깥 옆벽에서 M2.5×8 ×4 씩 (덮개 · 반바지 전) · 그다음 다리 (다리 장 단계 11–12)", "단계 3")),
        dict(key="frame", names=["torso_frame"], side="L",
             lines=Lb("몸통 프레임", F3 + "torso_frame.stl", "골반판 · 옆벽 · 뒷벽 · 레일이 한 부품 · 6805 ×2 를 골반판에 손으로 넣음 · 자석 6×3 ×4 접착", "단계 2–4")),
        dict(key="hatch", names=["pelvis_hatch"], side="L",
             lines=Lb("골반 덮개 (배터리 문)", F3 + "pelvis_hatch.stl", "자석 6×3 ×4 접착 · 배터리를 넣고 아래에 붙임", "단계 2")),
    ]
    for it in items:
        it["also"] = [n for _, ns, _ in stack for n in ns]
    sh.text(8, 17.5, "가. 몸통 안쪽 (단계 2–4)", size=10, bold=True)
    sh.text(8, 22.3, "STL 은 out20/PRINT_v2/3_몸통_머리/ 안", size=7.2, color=GREY)
    rect, ids, pr, gm, ps = explode_panel(sh, r, geom, items, offs, E, AZ, (6, 26, 186, YB - 3), "torsoA", label_w=(lw - 4, lw + 4))
    def zr(g, names):
        P = part_pts(g, set(names)); return P[:, 2].min(), P[:, 2].max()
    names = {k: ns for k, ns, _ in stack}
    for up_k, lo_k, st in (("shelf", "yaw", 4), ("yaw", "frame", 3)):
        arrow_model(sh, pr, rect, [-20, -78, zr(gm, names[up_k])[0] - 1.5], [-20, -78, zr(gm, names[lo_k])[1] + 1.5], st, "left")
    arrow_model(sh, pr, rect, [-20, -30, zr(gm, names["hatch"])[1] + 1.5], [-20, -30, zr(gm, names["frame"])[0] - 1.5], 2, "left")

    # ---------------- B: outside (steps 5-8), everything comes down from above
    base = frame_set + ["electronics_shelf", "yaw_servo_R_XL430", "yaw_servo_L_XL430", "pelvis_hatch", "hatch_magnets_6x3"]
    offs = {n: (0.0, 0.0, 0.0) for n in base}
    D_SH, D_CV, CXS = 45.0, 110.0, 25.0
    offs["shorts"] = (0.0, 0.0, D_SH)
    offs["torso_cover_front"] = (CXS, 0.0, D_CV)
    offs["torso_cover_back"] = (-CXS, 0.0, D_CV)
    d_lid = 336.0 + D_CV + 16.0 - 330.0
    d_yoke = 338.3 + d_lid + 16.0 - 338.3
    offs["lid"] = (0.0, 0.0, d_lid)
    for n in ("neck_yoke", "neck_bearings_688", "nod_anchor_pin"):
        offs[n] = (0.0, 0.0, d_yoke)
    keepB = base + ["shorts", "torso_cover_front", "torso_cover_back", "lid", "neck_yoke", "neck_bearings_688", "nod_anchor_pin"]
    itemsB = [
        dict(key="yoke", names=["neck_yoke"], side="R",
             lines=Lb("목 요크 (남색)", F3 + "neck_yoke.stl", "688 베어링 ×2 를 손으로 밀어 넣고 고정제 · 뚜껑에 위에서 M2.5×10 ×4 (목 축 넣기 전)", "단계 7")),
        dict(key="lid", names=["lid"], side="L",
             lines=Lb("뚜껑 (흰색)", F3 + "lid.stl", "덮개 윗끝을 덮으며 마개를 벽 사이에 내려놓고 위에서 M2.5×14 ×4 · 누르거나 비틀어 끼우지 않음", "단계 8")),
        dict(key="cover_f", names=["torso_cover_front"], side="R",
             lines=Lb("셔츠 덮개 앞 (흰색)", F3 + "torso_cover_front.stl", "옆벽 레일(도브테일)에 위에서 밀어 내림 · 나사 없음 · 헐거운 것이 정상 (틈 0.5, 뚜껑이 잡음)", "단계 6")),
        dict(key="cover_b", names=["torso_cover_back"], side="L",
             lines=Lb("셔츠 덮개 뒤 (흰색)", F3 + "torso_cover_back.stl", "앞 덮개와 같이 레일에 위에서 밀어 내림 · 홈 아래 끝 나팔이 레일 윗끝을 찾아 들어감", "단계 6")),
        dict(key="rail", names=["torso_frame"], side="L", anchor_model=(15.5, -69.0, 300.0),
             lines=Lb("옆벽 레일 (도브테일, 프레임의 일부)", None, "좌우 옆벽에 앞 · 뒤 하나씩 · 덮개 홈이 위에서 끼워짐", "")),
        dict(key="shorts", names=["shorts"], side="R",
             lines=Lb("반바지 (남색)", F3 + "shorts.stl", "다리를 단 뒤 위에서 프레임 둘레로 내려 씌움 · 플랜지가 골반판 위에 얹힘 (틈 0.4) · 셔츠 덮개 전에", "단계 5")),
        dict(key="base", names=["torso_frame"], side="L",
             lines=Lb("가. 를 마친 몸통", None, "프레임 · 요 서보 · 선반 · 골반 덮개 (다리도 이미 달림)", "")),
    ]
    for it in itemsB:
        it["also"] = keepB
    sh.text(190, 17.5, "나. 몸통 바깥 (단계 5–8) — 모두 위에서 내려옵니다", size=10, bold=True)
    sh.text(190, 22.3, "빨간 화살표 = 끼우는 방향, 숫자 = 조립 단계. 반바지는 프레임 둘레로 내려가는 중인 모습입니다.", size=7.2, color=GREY)
    rect, ids, pr, gm, ps = explode_panel(sh, r, geom, itemsB, offs, 12, AZ, (188, 26, 414, YB - 3), "torsoB", label_w=(lw + 4, lw + 4))
    arrow_model(sh, pr, rect, [-80, -80, 237 + D_SH + 6], [-80, -80, 237 + 3], 5, "left")
    for x_r in (15.5 + CXS, -23.5 - CXS):
        arrow_model(sh, pr, rect, [x_r, -76, 237 + D_CV + 4], [x_r, -76, 322], 6, "left")
    arrow_model(sh, pr, rect, [-60, -60, 330 + d_lid - 1.5], [-60, -60, 336 + D_CV + 1.5], 8, "left")
    arrow_model(sh, pr, rect, [-25, -17, 338.3 + d_yoke - 1.5], [-25, -17, 338.3 + d_lid + 1.5], 7, "left")

    # ---------------- text band
    y = YB + 1.5
    steps = [
        (1, "출력 뒤 구멍 맞추기:", "드릴 표대로 모든 나사 구멍을 손드릴로 한 번 훑음. 베어링 자리는 손으로 밀릴 때까지 다듬음 (망치 금지)."),
        (2, "골반:", "자석 8개(프레임 4 · 골반 덮개 4)를 극 방향을 맞춰 접착. 배터리를 넣고 골반 덮개를 붙임."),
        (3, "고관절 요 서보 2개:", "6805 를 골반판에 넣고 서보를 프레임 안에 놓은 뒤 바깥 옆벽에서 M2.5×8 한쪽 4개 (덮개 · 반바지 전). 그다음 다리 (다리 장 단계 11–12)."),
        (4, "전자 선반:", "Ø3 강철 핀 3개를 프레임 받침에 접착하고 선반을 얹음. OpenCR 을 뒷벽 홈에 세움."),
        (5, "반바지:", "위에서 내려 씌움 (틈 0.4)."),
        (6, "셔츠 덮개 앞 · 뒤:", "레일에 위에서 밀어 내림. 나사 없음. 헐거운 것이 정상 (틈 0.5, 뚜껑이 잡아 줌)."),
        (7, "목 요크:", "688 두 개를 손으로 밀어 넣고 고정제. 뚜껑에 놓고 M2.5×10 ×4 로 위에서 (목 축 넣기 전)."),
        (8, "뚜껑:", "덮개 윗끝을 덮으며 마개를 벽 사이에 내려놓고 M2.5×14 ×4 로 위에서. 누르거나 비틀지 않음."),
    ]
    box_title(sh, 8, y, "다. 조립 순서 (몸통 · 머리 키트 0.5 절)", 9.2)
    y1 = y + 9.2 * PT * 1.45
    ya = steps_block(sh, 8, y1, steps[:4], 98, size=7.6)
    yb = steps_block(sh, 108, y1, steps[4:], 98, size=7.6)
    X2 = 209
    yy = box_title(sh, X2, y, "라. 꼭 지킬 것", 9.2)
    yy = bullets(sh, X2, yy, [
        "덮개는 도브테일로만 잡습니다(나사 없음). 위에서 아래로 밀고, 뚜껑 나사 4개가 위에서 눌러 줍니다.",
        "요 서보 몸체 나사는 덮개 · 반바지 전에 바깥 옆벽에서 넣습니다(나중엔 드라이버가 안 들어감).",
        "태핑 나사는 손힘으로만. 빡빡한 칸 · 레일은 줄 · 사포로 갈고 벽을 벌리지 않습니다.",
        "셔츠 덮개는 세워서 브림 권장. 껍데기 벽 5줄(2 mm) 이상이면 속이 꽉 찹니다.",
    ], 94, size=7.4)
    yy += 0.4
    yy = table(sh, X2, yy, ["구멍 (몸통 · 머리 같음)", "드릴"],
               [("태핑 나사 밑구멍 (나사산 내는 쪽)", "2.2"), ("M2.5 나사가 지나는 구멍", "3.0"),
                ("M2 밑구멍 / M2 지나는 구멍", "1.6 / 2.5"), ("선반 핀 받침", "3.0")], size=7.3, widths=[74, 20])
    X3 = 308
    y3 = box_title(sh, X3, y, "마. 몸통 나사 · 구매품 (머리는 3 장)", 9.2)
    y3 = table(sh, X3, y3, ["품목", "개수", "쓰는 곳"],
               [("M2.5×8 태핑", "8", "요 서보 → 바깥 옆벽, 한쪽 4"),
                ("M2.5×10 태핑", "4", "목 요크 → 뚜껑"),
                ("M2.5×14 태핑", "4", "뚜껑 → 프레임 받침 (×12 도 됨)"),
                ("강철 핀 Ø3×12", "3", "선반 받침 (길이 추정)"),
                ("자석 6×3", "8", "골반판 4 · 골반 덮개 4"),
                ("베어링 6805 · 688", "2 · 2", "고관절 요 허브 · 목 축"),
                ("XL430-W250-T", "2", "고관절 요 (ID 1 · 7)"),
                ("접착제 · 고정제", "조금", "자석 · 핀 · 베어링")],
               size=7.3, widths=[30, 12, 64])
    assert max(ya, yb, yy, y3) < 291, f"torso band overflows: {max(ya, yb, yy, y3):.1f}"
    return sh


# ================================================================ sheet 3: head
def sheet_head(plt, F, r, geom, commit):
    sh = new_sheet(plt, F)
    header(sh, "키토루 v21 조립도", "3. 머리 (몸통 · 머리 키트 0.5 절 단계 9–14)", 3, commit)
    YB = 236
    sh.line([4, 416], [YB - 1, YB - 1], lw=0.6)
    nod = ["head_nod_servo_XL430", "head_nod_servo_idler", "nod_servo_horn", "nod_crank", "nod_crank_pin", "nod_ear_pin",
           "nod_rod", "ear_rod", "ear_rocker", "ear_rocker_pin"]
    offs = {}
    def put(names, d):
        for n in names:
            offs[n] = d
    DN = -100.0
    put(nod, (0.0, 0.0, DN))
    put(["nod_rod"], (0.0, 22.0, DN))
    put(["ear_rod"], (0.0, 40.0, DN))
    put(["ear_rocker", "ear_rocker_pin"], (0.0, 58.0, DN))
    X_HF, X_FP, X_EY = 95.0, 175.0, 215.0
    put(["head_front"], (X_HF, 0.0, 0.0))
    put(["face_plate"], (X_FP, 0.0, 0.0))
    put(["camera_clip"], (X_FP, 0.0, -85.0))
    put(["eye_L", "eye_R", "mouth"], (X_EY, 0.0, 0.0))
    put(["crest"], (0.0, 0.0, 50.0))
    put(["ear_L"], (0.0, 60.0, 0.0)); put(["ear_disc_L"], (0.0, 95.0, 0.0)); put(["antenna_L"], (0.0, 60.0, 35.0))
    put(["ear_R"], (0.0, -70.0, 95.0)); put(["ear_disc_R"], (0.0, -105.0, 95.0)); put(["antenna_R"], (0.0, -70.0, 130.0))
    E, AZ = 16, 55
    lw = 62
    L = lambda *a: label_lines(sh, *a, lw - 4)
    items = [
        dict(key="crest", names=["crest"], side="R", lines=L("볏 (하늘색)", F3 + "crest.stl", "둘레 바닥에 순간접착제를 바르고 머리 윗면에 붙임", "단계 14")),
        dict(key="antenna", names=["antenna_L"], side="R",
             lines=L("더듬이 왼 · 오른 (주황색)", [F3 + "antenna_L.stl", F3 + "antenna_R.stl"], "귀 구멍에 접착", "단계 13")),
        dict(key="ear_L", names=["ear_L"], side="R", lines=L("귀 왼쪽 (남색)", F3 + "ear_L.stl", "귀 원판 · 더듬이를 붙인 뒤 귀 축 끝에 끼워 접착", "단계 13")),
        dict(key="disc_L", names=["ear_disc_L"], side="R", lines=L("귀 원판 왼쪽 (하늘색)", F3 + "ear_disc_L.stl", "귀에 접착", "단계 13")),
        dict(key="ear_R", names=["ear_R", "ear_disc_R"], side="L",
             lines=L("귀 오른쪽 · 귀 원판 오른쪽 (뒤쪽, 거울 대칭)", [F3 + "ear_R.stl", F3 + "ear_disc_R.stl"], "왼쪽과 같이 접착", "단계 13")),
        dict(key="head_back", names=["head_back"], side="R",
             lines=L("머리 뒤 (하늘색)", F3 + "head_back.stl", "684 베어링 ×2 를 손으로 넣음 · Ø8 목 축을 목 요크 베어링과 머리 뒤 축 판에 끼움", "단계 9")),
        dict(key="head_front", names=["head_front"], side="L", anchor_model=(13.0 + 95.0 + 30.0, -40.0, 533.0),
             lines=L("머리 앞 (하늘색)", F3 + "head_front.stl", "머리 뒤에 밀어 붙임(윗벽 혀) · 옆에서 M2.5×8 ×2 · 나사 머리가 귀 안에 들어가므로 귀보다 먼저", "단계 12")),
        dict(key="camera_clip", names=["camera_clip"], side="L",
             lines=L("카메라 집게", F3 + "camera_clip.stl", "카메라를 얼굴판 뒤 받침에 놓고 집게를 얹어 M2×6 ×4 (M2×8 은 길어서 안 조여짐)", "단계 11")),
        dict(key="face_plate", names=["face_plate"], side="L",
             lines=L("얼굴판 (흰색)", F3 + "face_plate.stl", "머리 앞 테두리에 넣고 앞에서 M2.5×10 ×2 (나사 머리가 얼굴 양옆에 보임)", "단계 11")),
        dict(key="eyes", names=["eye_L", "eye_R", "mouth"], side="L",
             lines=L("눈 왼 · 오른, 입 (남색)", [F3 + "eye_L.stl · eye_R.stl", F3 + "mouth.stl"], "얼굴판 구멍에 앞에서 끼워 접착 (틈 0.4)", "단계 11")),
        dict(key="nod_servo", names=["head_nod_servo_XL430"], side="L", servo=True,
             lines=L("머리 끄덕임 서보 XL430 · ID 15", None, "머리 뒤 안쪽 바닥 받침에 M2.5×8 ×8", "단계 9")),
        dict(key="nod_crank", names=["nod_crank"], side="R",
             lines=L("끄덕임 크랭크", F3 + "nod_crank.stl", "서보 혼에 M2×5 ×4 (카운터보어) · 크랭크 핀 Ø3 ×2 접착", "단계 10")),
        dict(key="nod_rod", names=["nod_rod"], side="R",
             lines=L("끄덕임 막대", F3 + "nod_rod.stl", "크랭크 핀과 목 요크 고정 핀에 끼움 (돌아감, 뻑뻑하면 3.3 드릴)", "단계 10")),
        dict(key="ear_rod", names=["ear_rod"], side="R",
             lines=L("귀 막대", F3 + "ear_rod.stl", "크랭크의 귀 팔 핀과 귀 흔들개 핀에 끼움", "단계 10")),
        dict(key="ear_rocker", names=["ear_rocker"], side="R",
             lines=L("귀 흔들개", F3 + "ear_rocker.stl", "귀 축 D컷에 접착 (머리 안, 귀 축 위)", "단계 10")),
    ]
    keep = set(nod) | {"head_back", "ear_bearings_684", "ear_axle_4mm", "head_front", "camera_clip", "face_plate", "eye_L", "eye_R",
                       "mouth", "crest", "ear_L", "ear_disc_L", "antenna_L", "ear_R", "ear_disc_R", "antenna_R"}
    for it in items:
        it["also"] = sorted(keep)
    sh.text(8, 17.5, "가. 머리 분해도 (앞 왼쪽에서 봄) — 빨간 화살표 = 끼우는 방향, 숫자 = 조립 단계", size=10, bold=True)
    sh.text(8, 22.3, "STL 은 out20/PRINT_v2/3_몸통_머리/ 안. 끄덕임 장치(서보 · 크랭크 · 막대 · 흔들개)는 머리 아래로 빼서 그렸습니다 — 실제로는 머리 뒤 안쪽 바닥에 붙습니다.",
            size=7.2, color=GREY)
    rect, ids, pr, gm, ps = explode_panel(sh, r, geom, items, offs, E, AZ, (6, 26, 414, YB - 3), "head", label_w=(lw, lw), label_gap=1.4)
    def bb(names):
        P = part_pts(gm, set(names)); return P.min(0), P.max(0)
    lo, hi = bb(["eye_L", "eye_R", "mouth"]); lo2, hi2 = bb(["face_plate"])
    arrow_model(sh, pr, rect, [lo[0] - 2, 37, 476], [hi2[0] + 2, 37, 476], 11, "top")
    lo3, hi3 = bb(["head_front"])
    arrow_model(sh, pr, rect, [lo2[0] - 2, 60, 500], [hi3[0] + 2, 60, 500], 11, "top")
    arrow_model(sh, pr, rect, [lo3[0] - 2, 98, 520], [22.2 + 3, 98, 520], 12, "top")
    lo4, hi4 = bb(["camera_clip"])
    arrow_model(sh, pr, rect, [(lo4[0] + hi4[0]) / 2, 37, hi4[2] + 2], [(lo4[0] + hi4[0]) / 2, 37, 440], 11, "right")
    lo4, hi4 = bb(["crest"])
    arrow_model(sh, pr, rect, [0, 0, lo4[2] - 2], [0, 0, 535 + 3], 14, "right")
    lo5, hi5 = bb(["ear_L"])
    arrow_model(sh, pr, rect, [0, lo5[1] - 2, 482], [0, 100 + 3, 482], 13, "top")
    lo6, hi6 = bb(["ear_disc_L"])
    arrow_model(sh, pr, rect, [0, lo6[1] - 2, 418], [0, hi5[1] + 2, 418], 13, "top")
    lo7, hi7 = bb(["head_nod_servo_XL430"])
    arrow_model(sh, pr, rect, [-20, -10, hi7[2] + 3], [-20, -10, 339 - 3], 9, "left")

    # ---------------- text band
    y = YB + 1.5
    steps = [
        (9, "머리 뒤:", "끄덕임 서보를 바닥 받침에 M2.5×8 ×8. 684 두 개를 손으로 넣음. Ø8 목 축을 목 요크 베어링과 머리 뒤 축 판에."),
        (10, "크랭크 · 막대:", "크랭크를 혼에 M2×5 ×4, 크랭크 핀 2개 접착. 끄덕임 막대 · 귀 막대를 핀에 끼움. 귀 흔들개는 귀 축 D컷에 접착."),
        (11, "얼굴판:", "눈 · 입을 앞에서 접착. 카메라를 받침에 놓고 집게를 얹어 M2×6 ×4. 얼굴판을 테두리에 넣고 앞에서 M2.5×10 ×2."),
        (12, "머리 앞:", "머리 뒤에 밀어 붙임(윗벽 혀). 옆에서 M2.5×8 ×2. 나사 머리가 귀 컵 안에 들어가므로 귀보다 먼저."),
        (13, "귀:", "귀 원판 · 더듬이를 귀에 접착. 귀를 귀 축 끝에 끼워 접착."),
        (14, "볏:", "둘레 바닥에 순간접착제를 바르고 윗면에 붙임."),
    ]
    box_title(sh, 8, y, "나. 조립 순서 (몸통 다음, 몸통 · 머리 키트 0.5 절)", 9.2)
    y1 = y + 9.2 * PT * 1.45
    ya = steps_block(sh, 8, y1, steps[:3], 98, size=7.6)
    yb = steps_block(sh, 108, y1, steps[3:], 98, size=7.6)
    X2 = 209
    yy = box_title(sh, X2, y, "다. 꼭 지킬 것", 9.2)
    yy = bullets(sh, X2, yy, [
        "베어링(684 · 688)은 손으로 밀릴 때까지 자리를 다듬고 고정제로 붙입니다. 망치로 치지 않습니다.",
        "머리 앞을 붙인 다음에 귀를 답니다(이음 나사 머리가 귀 컵 안, 귀가 ±60° 돌아도 약 1 mm 떨어짐).",
        "막대 핀 구멍(Ø3.4)은 돌아가야 합니다 — 뻑뻑하면 3.3 드릴로 훑음. 크랭크 핀(Ø3.2)은 접착.",
        "조립 뒤 끄덕임 −20 ~ +28° 를 손으로 돌려 크랭크 · 막대 · 귀가 걸리지 않는지 봅니다.",
    ], 94, size=7.4)
    yy += 0.3
    yy = para(sh, X2, yy, "머리 부품 18개 (3_몸통_머리/): head_back · head_front · face_plate · eye_L · eye_R · mouth · camera_clip · crest · "
              "nod_crank · nod_rod · ear_L · ear_R · ear_disc_L · ear_disc_R · antenna_L · antenna_R · ear_rocker · ear_rod", 7.2, 94, color=BLUE)
    X3 = 308
    y3 = box_title(sh, X3, y, "라. 머리 나사 · 구매품", 9.2)
    y3 = table(sh, X3, y3, ["품목", "개수", "쓰는 곳"],
               [("M2.5×8 태핑", "8 + 2", "끄덕임 서보 8 · 머리 이음 2"),
                ("M2.5×10 태핑", "2", "얼굴판 → 머리 앞 보스"),
                ("M2×5 (혼 나사)", "4", "끄덕임 크랭크 → 서보 혼"),
                ("M2×6", "4", "카메라 집게"),
                ("베어링 684", "2", "귀 축"),
                ("강철 축 Ø8 · Ø4", "1 · 1", "목 축 · 귀 축"),
                ("강철 핀 Ø3 짧은 것", "4", "크랭크 핀 2 · 요크 고정 핀 · 흔들개 핀 (형상에서 셈)"),
                ("XL430-W250-T", "1", "머리 끄덕임 (ID 15)"),
                ("카메라 v2 · 접착제", "1 · 조금", "얼굴판 받침 · 눈 · 입 · 귀 · 볏")],
               size=7.3, widths=[31, 14, 61])
    assert max(ya, yb, yy, y3) < 291, f"head band overflows: {max(ya, yb, yy, y3):.1f}"
    return sh


# ================================================================ sheet 4: arm + whole
F4 = "4_팔/"


def sheet_arm(plt, F, r, geom, commit):
    sh = new_sheet(plt, F)
    header(sh, "키토루 v21 조립도", "4. 팔꿈치 팔 (오른팔, 왼팔 대칭) · 조립 완성 · 서보 17개 · 나사 합계", 4, commit)
    sh.line([4, 416], [189, 189], lw=0.6)
    sh.line([200, 200], [15, 189], lw=0.6); sh.vdiv.append((200, 15, 189))
    # ---- arm explode
    shoulder = ["arm_R_servo_XL430", "arm_R_horn_XC430", "arm_R_idler_XC430"]
    elbow = ["arm_R_elbow_servo_XL430", "arm_R_elbow_horn", "arm_R_elbow_idler"]
    offs = {}
    for n in shoulder:
        offs[n] = (0.0, 75.0, 0.0)
    offs["arm_R_horn_adapter"] = (0.0, 120.0, 0.0)
    offs["shoulder_shaft_R"] = (0.0, 135.0, 0.0)
    for n in elbow:
        offs[n] = (0.0, -55.0, 0.0)
    offs["arm_R_elbow_bracket_out"] = (0.0, -110.0, -20.0)
    offs["arm_R_elbow_bracket_in"] = (0.0, 52.0, -20.0)
    offs["arm_R_hand_frame"] = (0.0, 0.0, -80.0)
    offs["arm_R_hand_pad"] = (0.0, 0.0, -95.0)
    E, AZ = 14, -28
    lw = 50
    L = lambda *a: label_lines(sh, *a, lw - 3)
    items = [
        dict(key="upper", names=["arm_R_upper_frame"], side="L",
             lines=L("위팔 틀 (하늘색)", F4 + "arm_R_upper_frame.stl", "서보 둘을 잡는 앞뒤 벽 · 상자 · 바깥 덮개 · 윗뚜껑 · 아래판이 한 몸", "")),
        dict(key="shoulder", names=["arm_R_servo_XL430"], side="R", servo=True,
             lines=L("어깨 서보 XL430 · ID 13 (왼 14)", None, "혼이 몸통 쪽 · 안쪽에서 위 칸에 밀어 넣음 · 벽 바깥 M2.5×8 ×8 + 윗뚜껑 위 M2.5×6 ×2", "단계 3")),
        dict(key="adapter", names=["arm_R_horn_adapter"], side="R",
             lines=L("어깨 혼 어댑터 (채움 100 %)", F4 + "arm_R_horn_adapter.stl", "어깨 혼에 M2×5 ×4 (몸통 쪽 카운터보어) · 강철 축 Ø8 에 에폭시 (틈 0.5, 압입 없음)", "단계 4 · 8")),
        dict(key="elbow", names=["arm_R_elbow_servo_XL430"], side="L", servo=True,
             lines=L("팔꿈치 서보 XL430 · ID 미정", None, "혼이 바깥 · 옆에서 아래 칸 앞뒤 벽 사이로 · 벽 바깥 M2.5×8 ×8 + 아래판 밑 M2.5×8 ×2", "단계 1 · 2")),
        dict(key="out", names=["arm_R_elbow_bracket_out"], side="L",
             lines=L("아래팔 바깥 판 (구조색)", F4 + "arm_R_elbow_bracket_out.stl", "팔꿈치 혼에 M2×5 ×4 · 손에 M2.5×8 ×4", "단계 5 · 6")),
        dict(key="in", names=["arm_R_elbow_bracket_in"], side="R",
             lines=L("아래팔 안쪽 판 (구조색)", F4 + "arm_R_elbow_bracket_in.stl", "아이들러에 M2×5 ×4 · 손에 M2.5×6 ×4 (카운터보어)", "단계 5 · 6")),
        dict(key="hand", names=["arm_R_hand_frame"], side="R",
             lines=L("손 (흰색)", F4 + "arm_R_hand_frame.stl", "고리 틀 둘을 아래에서 두 판 사이로 밀어 올림 · 고무 손바닥(1 mm) 접착", "단계 6 · 7")),
    ]
    keep = set(shoulder) | set(elbow) | {"arm_R_upper_frame", "arm_R_horn_adapter", "shoulder_shaft_R", "arm_R_elbow_bracket_out",
                                         "arm_R_elbow_bracket_in", "arm_R_hand_frame", "arm_R_hand_pad"}
    for it in items:
        it["also"] = sorted(keep)
    sh.text(8, 17.5, "가. 오른팔 분해도 — 빨간 숫자 = 조립 단계 (아래 목록)", size=10, bold=True)
    sh.text(8, 22.3, "STL 은 out20/PRINT_v2/4_팔/ 안 (왼팔은 arm_L_ 파일, 거울 대칭). 오른쪽이 몸통 쪽입니다.", size=7.2, color=GREY)
    rect, ids, pr, gm, ps = explode_panel(sh, r, geom, items, offs, E, AZ, (6, 26, 198, 140), "arm", label_w=(lw, lw), label_gap=1.4)
    def bb(names):
        P = part_pts(gm, set(names)); return P.min(0), P.max(0)
    lo, hi = bb(["arm_R_servo_XL430"]); arrow_model(sh, pr, rect, [0, lo[1] - 3, 324], [0, -81 + 3, 324], 3, "top")
    sv_hi = bb(["arm_R_horn_XC430"])[1][1]
    lo, hi = bb(["arm_R_horn_adapter"]); arrow_model(sh, pr, rect, [0, lo[1] - 2, 288], [0, sv_hi + 2, 288], 4, "top")
    lo, hi = bb(["arm_R_elbow_servo_XL430"]); arrow_model(sh, pr, rect, [0, hi[1] + 3, 238], [0, -125 - 3, 238], 2, "top")
    eh_lo = bb(["arm_R_elbow_horn"])[0][1]
    lo, hi = bb(["arm_R_elbow_bracket_out"]); arrow_model(sh, pr, rect, [0, hi[1] + 3, 182], [0, eh_lo - 3, 182], 5, "top")
    lo, hi = bb(["arm_R_elbow_bracket_in"]); arrow_model(sh, pr, rect, [0, lo[1] - 3, 182], [0, -81 + 3, 182], 5, "top")
    lo, hi = bb(["arm_R_hand_frame"]); arrow_model(sh, pr, rect, [0, -102, hi[2] + 3], [0, -102, 146 - 22], 6, "right")
    # arm steps below the explode
    X0, W = 8, 188
    y = 141.5
    steps = [
        (1, "팔꿈치 서보에 짧은 X3P 케이블:", "위쪽이 될 케이블 끝에 꽂음."),
        (2, "팔꿈치 서보 → 위팔 틀 아래 칸:", "혼이 바깥. 앞뒤 벽 바깥 M2.5×8 ×8, 아래판 밑 M2.5×8 ×2. 아래팔 달기 전에 조임."),
        (3, "어깨 서보 → 위 칸:", "상자 안 케이블 · 몸통 케이블을 꽂고 혼이 몸통 쪽을 보게 안쪽에서 밀어 넣음. 벽 바깥 M2.5×8 ×8, 윗뚜껑 위 M2.5×6 ×2."),
        (4, "어깨 혼 어댑터 → 어깨 혼:", "M2×5 ×4. 어댑터 받침이 혼 원판 면에 닿아야 함."),
        (5, "아래팔 판:", "바깥 판 → 팔꿈치 혼 M2×5 ×4, 안쪽 판 → 아이들러 M2×5 ×4."),
        (6, "손:", "고리 틀 둘을 아래에서 두 판 사이로. 바깥 판 밖 M2.5×8 ×4, 안쪽 판 밖 M2.5×6 ×4."),
        (7, "고무 손바닥", "을 손 바닥면에 붙임."),
        (8, "어깨에 달기:", "어댑터 구멍 Ø9 ↔ 강철 축 Ø8 사이를 에폭시로 채우고, 축 끝이 혼 가운데 돌기에 닿을 때까지 밀어 넣고 굳힘."),
    ]
    half = 4
    y1 = steps_block(sh, X0, y, steps[:half], W / 2 - 3, size=7.1)
    y2 = steps_block(sh, X0 + W / 2 + 1, y, steps[half:], W / 2 - 3, size=7.1)
    yy = max(y1, y2) + 0.2
    para(sh, X0, yy, "주의: 어깨 혼 어댑터는 채움 100 % 로 출력 · 서보 칸이 빡빡하면 칸을 갈고 벽을 벌리지 않음 · 영점 = 팔을 곧게 늘어뜨린 자세(어깨 0° · 팔꿈치 0°).",
         7.1, W, color=RED)

    # ---- assembled front / side (right top)
    RX0 = 204
    sh.text(RX0, 17.5, "나. 조립 완성 (설계 형상, 모든 관절 0°)", size=10, bold=True)
    gall = [g for g in geom]
    Pall = part_pts(gall, {g[1] for g in gall})
    z_lo, z_hi = Pall[:, 2].min(), Pall[:, 2].max()
    height = z_hi - z_lo
    hbox = 156
    fw, fh = extent(gall, 0, 0)
    sw_, shh = extent(gall, 0, -90)
    sc = hbox / max(fh, shh) / PX_MM
    img_f, ids_f, pr_f = do_render(r, gall, 0, 0, sc, "front")
    img_s, ids_s, pr_s = do_render(r, gall, 0, -90, sc, "side")
    rf = place_image(sh, img_f, ids_f, RX0 + 22, 24)
    rs = place_image(sh, img_s, ids_s, rf[2] + 22, 24)
    sh.text((rf[0] + rf[2]) / 2, rf[3] + 0.6, "정면", size=7.4, ha="center")
    sh.text((rs[0] + rs[2]) / 2, rs[3] + 0.6, "오른쪽 옆 (앞이 오른쪽)", size=7.4, ha="center")
    top = to_mm(project(pr_f, np.array([[0, 0, z_hi]]))[0], rf)[1]
    bot = to_mm(project(pr_f, np.array([[0, 0, z_lo]]))[0], rf)[1]
    xd = rf[0] - 6
    for yy_ in (top, bot):
        sh.line([xd - 1.5, rf[0] + 2], [yy_, yy_], lw=0.4, color=GREY)
    sh.ax.annotate("", xy=(xd, top), xytext=(xd, bot), zorder=6, arrowprops=dict(arrowstyle="<|-|>,head_length=0.4,head_width=0.18", lw=0.6, color=INK))
    sh.text(xd - 1.2, (top + bot) / 2, f"전체 높이 {height:.0f} mm (발바닥 고무 패드 ~ 볏 윗면)", size=7.4, ha="right", va="center", rotation=90)
    jz = {j["name"]: j["point_mm"][2] for j in r.joints}
    names_j = ("R_hip_pitch", "R_knee", "R_ankle_pitch")
    yk = [to_mm(project(pr_s, np.array([[0, -40, jz[n]]]))[0], rs)[1] for n in names_j]
    xs = rs[2] + 2
    for yy_, n in zip(yk, ("고관절 피치 축", "무릎 축", "발목 축")):
        sh.line([rs[2] - 8, xs], [yy_, yy_], lw=0.35, color=GREY, ls=(0, (2, 1.5)))
        sh.text(xs + 0.8, yy_, n, size=7.0, va="center", color=GREY)
    sh.text(xs + 0.8, (yk[0] + yk[1]) / 2, f"{jz['R_hip_pitch'] - jz['R_knee']:.0f}", size=7.0, va="center")
    sh.text(xs + 0.8, (yk[1] + yk[2]) / 2, f"{jz['R_knee'] - jz['R_ankle_pitch']:.0f}", size=7.0, va="center")
    arm_len = jz["R_arm_swing"] - jz["R_elbow"]
    ya_ = [to_mm(project(pr_s, np.array([[0, -102, jz[n]]]))[0], rs)[1] for n in ("R_arm_swing", "R_elbow")]
    xl = rs[0] - 2
    for yy_, n in zip(ya_, ("어깨 축", "팔꿈치 축")):
        sh.line([xl, rs[0] + 10], [yy_, yy_], lw=0.35, color=GREY, ls=(0, (2, 1.5)))
        sh.text(xl - 0.8, yy_, n, size=7.0, va="center", ha="right", color=GREY)
    sh.text(xl - 0.8, (ya_[0] + ya_[1]) / 2, f"{arm_len:.0f}", size=7.0, va="center", ha="right")

    # ---- servo table (bottom left)
    y = 192
    X0 = 8
    y = box_title(sh, X0, y, "다. 서보 17개 — 모두 XL430-W250-T", 9.5)
    rows = [("고관절 요", "1", "7", "TTL1 · TTL2", "2"), ("고관절 롤", "2", "8", "TTL1 · TTL2", "2"),
            ("고관절 피치", "3", "9", "TTL1 · TTL2", "2"), ("무릎", "4", "10", "TTL1 · TTL2", "2"),
            ("발목 피치", "5", "11", "TTL1 · TTL2", "2"), ("발목 롤", "6", "12", "TTL1 · TTL2", "2"),
            ("어깨 (팔 흔들기)", "13", "14", "TTL3", "2"), ("팔꿈치 (새로)", "미정", "미정", "TTL3, 어깨에서 이어짐 (추정)", "2"),
            ("머리 끄덕임 (크랭크)", "15", "–", "TTL3", "1")]
    y = table(sh, X0, y, ["관절", "오른쪽 ID", "왼쪽 ID", "버스 단자 (OpenCR, 오른 · 왼)", "개수"], rows, size=7.2,
              widths=[34, 18, 16, 50, 12])
    sh.text(X0, y + 0.2, "합계", size=7.4, bold=True); sh.text(X0 + 118, y + 0.2, "17", size=7.4, bold=True)
    y += 4.4
    y = para(sh, X0, y, "ID 는 자리표시입니다(robot/config/joints_v20.yaml, 조립 뒤 확정). yaml 의 XM430 · XC430 이름은 옛 표기이고 실물은 모두 "
             "XL430-W250-T. 팔꿈치는 yaml 에 아직 없습니다. 버스는 OpenCR 직렬 포트 하나 — ID 17개가 모두 달라야 합니다.", 7.1, 130)
    y += 0.6
    y = para(sh, X0, y, "데이지 체인 케이블은 서보를 칸에 넣기 전에 꽂습니다. 팔꿈치 ↔ 어깨는 짧은 X3P(100 mm 이하) 케이블, 위팔 속 상자 안.", 7.1, 130)
    y += 2.0
    y = box_title(sh, X0, y, "마. 출력 부품 52개 (out20/PRINT_v2/ 폴더별)", 9.5)
    cnt = {}
    for fp in glob.glob(os.path.join(PRINT_DIR, "*", "*.stl")):
        d = os.path.basename(os.path.dirname(fp)); cnt[d] = cnt.get(d, 0) + 1
    where = {"1_다리_오른쪽": "1 장 분해도 (R_ 파일)", "2_다리_왼쪽": "1 장과 거울 대칭 (L_ 파일)",
             "3_몸통_머리": "2 장 몸통 8개 · 3 장 머리 18개", "4_팔": "4 장 (오른팔 5 + 왼팔 5)"}
    assert [cnt.get(k) for k in where] == [8, 8, 26, 10], cnt
    y = table(sh, X0, y, ["폴더", "개수", "그린 곳"], [(k, str(cnt[k]), v) for k, v in where.items()] + [("합계", str(sum(cnt.values())), "")],
              size=7.2, widths=[34, 14, 82], colors=[None] * 4 + [["B", "B", None]])

    # ---- totals (bottom right)
    TX = 142
    y = 192
    y = box_title(sh, TX, y, "라. 나사 · 구매품 합계 (로봇 한 대, 조립 단계에서 셈)", 9.5)
    totals = [
        ("M2.5×8 태핑", "56", "8 + 10", "44", "118"),
        ("M2.5×10 태핑", "8", "4 + 2", "–", "14"),
        ("M2.5×6 태핑", "4", "–", "12", "16"),
        ("M2.5×14 태핑", "–", "4", "–", "4"),
        ("M2.5×16 태핑", "16", "–", "–", "16"),
        ("M2×5 냄비머리 (혼 · 아이들러)", "54", "4", "24", "82"),
        ("M2×6 (카메라 집게)", "–", "4", "–", "4"),
        ("베어링 6805 / 688 / 684", "2 / – / –", "– / 2 / 2", "–", "2 / 2 / 2"),
        ("강철 축 Ø8 / Ø4", "–", "목 1 / 귀 1", "어깨 2 / –", "3 / 1"),
        ("강철 핀 Ø3×12 / Ø3 짧은 것", "–", "3 / 4", "–", "3 / 4"),
        ("자석 6×3", "–", "8", "–", "8"),
        ("고무: 발바닥 패드 Ø12×6 / 손바닥 1 mm 판", "10 / –", "–", "– / 2", "10 / 2"),
        ("짧은 X3P 케이블 · 케이블 타이 (폭 2.5 이하)", "– · 4 이상", "–", "2 · –", "2 · 4 이상"),
        ("카메라 FFC 15핀 1 mm 피치 500 mm (머리 바닥 · 뚜껑 구멍 폭 20)", "–", "1", "–", "1"),
        ("2액 에폭시 · 순간접착제 · 베어링 고정제", "에폭시", "접착 · 고정제", "에폭시", "조금씩"),
    ]
    y = table(sh, TX, y, ["품목", "다리 2개", "몸통 + 머리", "팔 2개", "합계"], totals, size=7.3,
              widths=[88, 34, 50, 40, 60], colors=[[None, None, None, None, "B"]] * len(totals))
    y += 0.4
    y = para(sh, TX, y, "출처: 다리 LEG_PRINT_KIT_R.md 「새 조립 순서」 + 11-28 발목 롤 8개 · 몸통 · 머리 BODY_HEAD_PRINT_KIT.md 0.5 · 0.6 절 · "
             "팔 ARM_ELBOW_v21.md 0 · 4절. 출력 부품은 PRINT_v2 52개: 다리 8 + 8, 몸통 · 머리 26, 팔 10.", 7.0, 272, color=GREY)
    y += 1.2
    y = para(sh, TX, y, "합계에 넣지 않은 것: 서보 상자에 든 혼 · 아이들러 · 케이블(데이지 체인용 X3P 는 서보 수만큼 + 여유), 전장품(OpenCR · Jetson · "
             "배터리 · 카메라 · 스피커 등). 나사는 처음 한 개를 끝까지 조여 보고 길이를 확인합니다(케이스 안 물림 4 mm 이하).", 7.2, 272)
    y += 0.8
    y = para(sh, TX, y, "조립 뒤: 토크를 끄고 관절마다 손으로 범위를 확인합니다(1 장 「라.」 표, 특히 발목 피치 +30°). 팔은 어깨 ±110°, "
             "팔꿈치 ±130° 에서 몸에 닿지 않아야 합니다(형상 기준). 다만 팔을 높이 든 채(어깨 ±70° 넘게) 팔꿈치를 크게 접으면 손이 머리에 닿습니다.", 7.2, 272, color=RED)
    assert y < 291, f"totals overflow: {y:.1f}"
    return sh


# ---------------------------------------------------------------- checks + save
def check_sheet(sh, name):
    rend = sh.fig.canvas.get_renderer()
    boxes = []
    small = []
    for t in sh.texts:
        if not t.get_text():
            continue
        if t.get_fontsize() < MIN_PT - 1e-6:
            small.append(t.get_text()[:30])
        bb = t.get_window_extent(renderer=rend)
        x0, x1 = bb.x0 / DPI * 25.4, bb.x1 / DPI * 25.4
        y0, y1 = 297 - bb.y1 / DPI * 25.4, 297 - bb.y0 / DPI * 25.4
        boxes.append((x0, y0, x1, y1, t.get_text()))
    bad = []
    for i in range(len(boxes)):
        a = boxes[i]
        if a[0] < 4 or a[2] > 416 or a[1] < 4 or a[3] > 293:
            bad.append(("off page", a[4][:40]))
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if a[0] < b[2] - 0.15 and b[0] < a[2] - 0.15 and a[1] < b[3] - 0.15 and b[1] < a[3] - 0.15:
                bad.append(("overlap", a[4][:30], b[4][:30]))
    for x0, y0, x1, y1, s in boxes:
        for xv, ya, yb in getattr(sh, "vdiv", []):
            if x0 < xv < x1 and y0 < yb and y1 > ya:
                bad.append(("crosses divider", s[:40]))
    # text over rendered parts (fraction of the text box covering part pixels)
    over = []
    for sid, rect, ids in IMAGES:
        if sid != sh.tag:
            continue
        for x0, y0, x1, y1, s in boxes:
            if x1 < rect[0] or x0 > rect[2] or y1 < rect[1] or y0 > rect[3]:
                continue
            c0 = max(int((x0 - rect[0]) / PX_MM), 0); c1 = min(int((x1 - rect[0]) / PX_MM), ids.shape[1])
            r0 = max(int((y0 - rect[1]) / PX_MM), 0); r1 = min(int((y1 - rect[1]) / PX_MM), ids.shape[0])
            if c1 <= c0 or r1 <= r0:
                continue
            f = (ids[r0:r1, c0:c1] > 0).mean()
            if f > 0.03 and len(s) > 2:
                over.append((round(f, 2), s[:36]))
    for b in bad:
        print(name, "LAYOUT", *b)
    for o in over:
        print(name, "TEXT ON PART", *o)
    for s in small:
        print(name, "SMALL TEXT", s)
    return len(bad), len(over), len(small)


def check_print_set():
    """PRINT_v2 STL names (52) must all be parts of the model this drawing renders"""
    stls = sorted(glob.glob(os.path.join(PRINT_DIR, "*", "*.stl")))
    names = {os.path.basename(p)[:-4] for p in stls}
    model_parts = {pn for L in MODEL[0].values() for pn in L["parts"]}
    missing = sorted(names - model_parts)
    print("PRINT_v2 STL", len(stls), "missing in model:", missing)
    assert len(stls) == 52 and not missing
    return stls


def main():
    check_print_set()
    os.makedirs(OUTD, exist_ok=True)
    r, geom = A20.setup_renderer()
    plt, F = A20.fonts()
    commit = build_commit()
    only = os.environ.get("ASM_ONLY")           # e.g. "1,3" to redo some sheets while laying out
    sheets = [("p1_leg", sheet_leg), ("p2_torso", sheet_torso), ("p3_head", sheet_head), ("p4_arm_total", sheet_arm)]
    figs = []
    for i, (nm, fn) in enumerate(sheets, 1):
        if only and str(i) not in only.split(","):
            continue
        sh = fn(plt, F, r, geom, commit)
        sh.fig.canvas.draw()
        res = check_sheet(sh, nm)
        png = os.path.join(OUTD, f"kitoru_v21_assembly_{nm}.png")
        sh.fig.savefig(png, dpi=DPI, facecolor="white")
        print("saved", png, "layout / text-on-part / small:", res)
        figs.append(sh.fig)
    if not only:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(os.path.join(OUTD, "kitoru_v21_assembly.pdf")) as pdf:
            for f in figs:
                pdf.savefig(f, dpi=DPI, facecolor="white")
        print("saved pdf")
    print("anchor fallbacks:", FALLBACKS)


if __name__ == "__main__":
    main()
