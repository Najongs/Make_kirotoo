"""v19 = v18 prepared for printing the legs.

  * servo case screw holes: the M2.5 tapped holes of the X430 case (x430_holes.HOLES, read from the official ROBOTIS
    STEP) are mapped into every servo seat; where a printed plate meets the hole (within 1 mm + SERVO_CLR of the case
    face) and the plate is extruded along the hole axis, a through hole of CASE_HOLE_D is cut
  * servo pockets: printed walls that clamp a servo case from both sides (gap <= 0.5 mm on two opposite faces) are
    moved SERVO_CLR further away (the wall face toward the case is shifted; one side each)
  * ankle roll servo: it only touched the ankle bracket with its idler face (8.75 mm air to both side plates), so a
    spacer block is added on each side plate up to SERVO_CLR from the case side; two case screws per side go through
    plate + spacer
  * sole plate pockets filled: OCC/gmsh could not read the 12-pocket plate correctly (open mesh), see sole_fill_pockets
  * horn screw holes (4 x M2 on PCD 16) use HORN_HOLE_D
  * one more rubber pad per foot: on the inner straight edge, x = 0, pad radius + 2 mm inside the edge
  * outer pads moved out to |y_rel| 29 mm (feedback walk test: lateral push 2 N -> 6 N), along x into the rounded
    sole corner where needed
  * assembly STEP: the 13 servo boxes and their horn / idler cylinders are replaced by the official ROBOTIS X430 STEP
    (renumbered and placed at every servo seat as a sub-assembly); the build model keeps the boxes for mass and checks
Variables (environment): KITORU_SERVO_CLR (0.15), KITORU_CASE_HOLE_D (2.9), KITORU_HORN_HOLE_D (2.2)
"""
import os, json, math, re
import numpy as np
import build18 as B18
from build18 import *          # noqa
import build, build4, geom4
from build import IDX, UVI
from build6 import PROFI
import x430_holes as X
import servo_fit as SF

SERVO_CLR = float(os.environ.get("KITORU_SERVO_CLR", "0.15"))
CASE_HOLE_D = float(os.environ.get("KITORU_CASE_HOLE_D", "2.9"))
HORN_HOLE_D = float(os.environ.get("KITORU_HORN_HOLE_D", "2.2"))
MIN_WALL = 1.2
ENCLOSE_GAP = 0.5
MID_PAD_IN = 2.0                 # pad edge this far inside the inner straight edge
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out19")
SERVO_STEP = os.path.join(HERE, "..", "references", "XC-430_idle.stp")
LOG = {}


def _wall_left(part_solids, s, uv, r, wmax=6.0):
    """smallest ring width around the hole that is still fully material (same part, any solid; mid-plate plane);
    also limited by the other holes of this solid"""
    axc, a0, a1 = s[1][0], s[1][1], s[1][2]
    k, (iu, iv) = IDX[axc], UVI[axc]
    t = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    w_hole = min([math.hypot(h[0] - uv[0], h[1] - uv[1]) - h[2] - r for h in s[1][5]] + [1e9])
    for w in np.arange(0.1, wmax + 1e-9, 0.1):
        P = np.zeros((len(t), 3)); P[:, iu] = uv[0] + (r + w) * np.cos(t); P[:, iv] = uv[1] + (r + w) * np.sin(t); P[:, k] = 0.5 * (a0 + a1)
        ok = np.zeros(len(t), bool)
        for x in part_solids:
            ok |= geom4.inside(x, P)
        if not ok.all():
            return round(min(w - 0.1, w_hole), 2)
    return round(min(wmax, w_hole), 2)


def _mass_parts():
    try:
        return json.load(open(os.path.join(HERE, "mass_build18_light.json")))["parts"]
    except Exception:
        return None


def _seg_dist(p, a, b):
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    d = b - a; t = max(0.0, min(1.0, float((p - a) @ d / max(d @ d, 1e-12))))
    return float(np.linalg.norm(p - a - t * d))


def _loop_dist(p, loop):
    n = len(loop)
    return min(_seg_dist(p, loop[i], loop[(i + 1) % n]) for i in range(n))


def widen_pockets(links, rows, frames):
    """shift the case-side face of every wall that clamps a servo from both sides by SERVO_CLR"""
    by = {}
    for r in rows:
        by.setdefault(r["servo"], []).append(r)
    done = []
    for sv, rr in by.items():
        fr = frames[sv]; R = np.c_[fr["X"], fr["Y"], fr["Z"]]
        for ax, (fa, fb) in {0: ("+x", "-x"), 1: ("+y(front)", "-y(back)"), 2: ("+z(horn)", "-z(idler)")}.items():
            A = [r for r in rr if r["face"] == fa and r["gap_mm"] <= ENCLOSE_GAP]
            Bb = [r for r in rr if r["face"] == fb and r["gap_mm"] <= ENCLOSE_GAP]
            if not (A and Bb):
                continue
            for r in A + Bb:
                sg = 1 if r["face"] == fa else -1
                n_s = np.zeros(3); n_s[ax] = sg
                nw = R @ n_s; k = int(np.argmax(np.abs(nw)))
                p_s = np.zeros(3)
                p_s[ax] = {0: 14.25 * sg, 1: 11.25 if sg > 0 else -35.25, 2: 17.0 * sg}[ax]
                face_c = float((fr["O"] + R @ p_s)[k])
                ss = links[r["link"]]["parts"][r["part"]]
                s = ss[r["solid_index"]]
                if s[0] != "rprism" or IDX[s[1][0]] != k:
                    done.append(dict(r, action="not handled (wall not extruded along the face normal)")); continue
                axn, a0, a1, outer, inn, holes = s[1]
                if abs(a0 - face_c) <= abs(a1 - face_c):
                    a0 += SERVO_CLR * (1 if nw[k] > 0 else -1)
                else:
                    a1 += SERVO_CLR * (1 if nw[k] > 0 else -1)
                ss[r["solid_index"]] = ("rprism", (axn, min(a0, a1), max(a0, a1), outer, inn, holes), s[2])
                done.append(dict(r, action=f"case-side face moved {SERVO_CLR} mm (plate {abs(s[1][2] - s[1][1]):.2f} -> {abs(a1 - a0):.2f} mm)"))
    return done


def horn_hole_size(links):
    n = 0
    for ln, L in links.items():
        for pn, ss in L["parts"].items():
            for i, s in enumerate(ss):
                if s[0] != "rprism" or not s[1][5]:
                    continue
                holes = list(s[1][5]); ctr = [h for h in holes if abs(h[2] - 2.5) < 1e-6 or abs(h[2] - 4.0) < 1e-6]
                new = []
                for h in holes:
                    if abs(h[2] - 1.1) < 1e-6 and any(abs(math.hypot(h[0] - c[0], h[1] - c[1]) - 8.0) < 0.05 for c in ctr):
                        new.append((h[0], h[1], HORN_HOLE_D / 2)); n += 1
                    else:
                        new.append(h)
                a = s[1]
                ss[i] = ("rprism", (a[0], a[1], a[2], a[3], a[4], new), s[2])
    return n


def _chain(links, sp, mp, max_d=25.0):
    """printed solids met along the hole axis from the case face outward, until 1 mm of empty space"""
    L = links[sp["link"]]; pw = np.array(sp["p_world"]); nw = np.array(sp["n_world"], float)
    parts = SF.printed_parts(L, mp)
    hits, empty, d = [], 0.0, 0.05
    while d < max_d:
        q = (pw + d * nw)[None]
        here = [(pn, si, s[0], s[1][0] if s[0] in ("rprism", "prism", "prismh") else "") for pn in parts
                for si, s in enumerate(L["parts"][pn]) if geom4.inside(s, q)[0]]
        if here:
            empty = 0.0
            hits += [h for h in here if h not in hits]
        elif hits:
            empty += 0.25
            if empty > 1.0:
                break
        d += 0.25
    return hits


def case_holes(links, frames, mp):
    spots = SF.hole_spots(links, frames, mp, depth=(0.05, 1.0 + SERVO_CLR))
    added, skipped, warn = [], [], []
    r = CASE_HOLE_D / 2
    for sp in spots:
        if not sp["hits"]:
            continue
        nw = np.array(sp["n_world"], float); k = int(np.argmax(np.abs(nw))); axc = "xyz"[k]
        for pn, si, kind, sax in _chain(links, sp, mp):
            ss = links[sp["link"]]["parts"][pn]; s = ss[si]
            if kind != "rprism" or sax != axc:
                skipped.append(dict(servo=sp["servo"], hole=sp["hole"], part=pn, reason=f"solid {kind} axis {sax} not along the hole ({axc})"))
                continue
            iu, iv = UVI[axc]; u, v = sp["p_world"][iu], sp["p_world"][iv]
            a = s[1]
            if any(math.hypot(h[0] - u, h[1] - v) < 0.5 for h in a[5]):
                continue
            # wall left around the hole: material of the same part (any solid) on rings r + w, mid-plate
            wall = _wall_left(ss, s, (u, v), r)
            ss[si] = ("rprism", (a[0], a[1], a[2], a[3], a[4], list(a[5]) + [(u, v, r)]), s[2])
            rec = dict(servo=sp["servo"], hole=sp["hole"], link=sp["link"], part=pn, solid=si, axis=axc,
                       uv=[round(u, 2), round(v, 2)], plate_mm=round(abs(a[2] - a[1]), 2), wall_left_mm=round(wall, 2))
            added.append(rec)
            if wall < MIN_WALL:
                warn.append(rec)
    return added, skipped, warn


SPACER = dict(z=(35.0, 43.0), x=(-68.0, -41.0), r=1.5)   # around the two case holes at y_s -4 (z 39), x -45 / -57; down to the back plate (no overhang when printed +x up)


def roll_servo_spacers(links, frames):
    """the ankle roll servo touches the ankle bracket only with its idler face (8.75 mm air to both side plates):
    add a spacer block on each side plate reaching to SERVO_CLR from the case side face, for the case screws"""
    out = []
    for side in ("R", "L"):
        fr = frames[f"{side}_ankle_roll_servo_XM430"]
        ab = links[f"{side}_ankle_link"]["parts"][f"{side}_ankle_bracket"]
        ys = fr["O"][1]
        plates = [x for x in ab if x[0] == "rprism" and x[1][0] == "y" and x[1][2] - x[1][1] > 3.5]
        for sg in (-1, 1):
            face = ys + sg * 14.25
            cand = [x for x in plates if (min(x[1][1], x[1][2]) - face) * sg > 0]
            pl = min(cand, key=lambda x: abs((x[1][1] if sg > 0 else x[1][2]) - face))
            near = pl[1][1] if sg > 0 else pl[1][2]
            y0, y1 = sorted((face + sg * SERVO_CLR, near))
            outline = [(SPACER["z"][0], SPACER["x"][0], SPACER["r"]), (SPACER["z"][1], SPACER["x"][0], SPACER["r"]),
                       (SPACER["z"][1], SPACER["x"][1], SPACER["r"]), (SPACER["z"][0], SPACER["x"][1], SPACER["r"])]
            ab.append(PROFI("y", y0, y1, outline, [], pl[2]))
            out.append((side, round(y0, 2), round(y1, 2)))
    return out


def sole_fill_pockets(links):
    """STEP / STL workaround: OCC (gmsh) does not rebuild the z 8..12 sole plate with its 12 pocket loops correctly
    (some pockets are dropped, the mesh gets open edges; seen in the v17 and v18 STEP too; each pocket alone is read
    fine). For printable STL the pockets of the sole plate are filled (solid 4 mm plate).
    The one pocket under the shank at the ankle limit (pitch -50, roll -20) is kept: filling it made the shank
    frame touch the plate there (sweep); a plate with a single pocket is read correctly."""
    from matplotlib.path import Path
    from shapes import sample
    n = 0
    for side in ("R", "L"):
        s_ = -1 if side == "R" else 1
        keep_pt = (8.5, 40.0 * s_ - s_ * (-14.3))
        fs = links[f"{side}_foot"]["parts"][f"{side}_foot_sole"]
        for i, x in enumerate(fs):
            if x[0] == "rprism" and x[1][0] == "z" and len(x[1][4]) >= 2:
                ax, a0, a1, outer, inn, holes = x[1]
                kept = [l for l in inn if Path(np.asarray(sample(l))).contains_point(keep_pt)]
                n += len(inn) - len(kept)
                fs[i] = PROFI(ax, a0, a1, outer, kept, x[2], holes)
    return n


OUTER_PAD_IN = -29.0           # outer pads (front, back) at |y_rel| 29 mm (walk test: side push 2 N -> 6 N)


def move_outer_pads(links):
    """outer rubber pads to y_rel = OUTER_PAD_IN; where the r 6 pad would cross the rounded sole corner (plate z 6..8
    outline), the pad is moved along x toward the foot centre until it is inside (0 mm margin)"""
    from matplotlib.path import Path
    out = []
    for side in ("R", "L"):
        s = -1 if side == "R" else 1
        ys = 40.0 * s
        fs = links[f"{side}_foot"]["parts"][f"{side}_foot_sole"]
        plate = next(x for x in fs if x[0] == "rprism" and x[1][0] == "z" and abs(x[1][1] - 6.0) < 1e-6)
        poly = Path(np.asarray(build4.sampled(plate)[1][3]))
        t = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        for i, x in enumerate(fs):
            if not (x[0] == "cyl" and abs(x[1][2] - 6) < 1e-9 and abs(x[1][3] - 6) < 1e-9 and abs(x[1][0][2]) < 1e-9):
                continue
            (cx, cy, cz), a, rr, h = x[1]
            yi = (ys - cy) / s
            if yi > -1e-6:
                continue
            yn = ys - s * OUTER_PAD_IN
            xn = cx
            for _ in range(80):
                ring = np.c_[xn + rr * np.cos(t), yn + rr * np.sin(t)]
                if poly.contains_points(ring, radius=-1e-6).all() or poly.contains_points(ring, radius=1e-6).all():
                    break
                xn -= 0.25 * math.copysign(1.0, cx)
            fs[i] = ("cyl", ((xn, yn, cz), a, rr, h), x[2])
            out.append((side, (cx, round(cy, 2)), "->", (round(xn, 2), round(yn, 2))))
    return out


def add_mid_pads(links):
    out = []
    for side in ("R", "L"):
        s = -1 if side == "R" else 1
        ys = 40.0 * s
        fs = links[f"{side}_foot"]["parts"][f"{side}_foot_sole"]
        pads = [x for x in fs if x[0] == "cyl" and abs(x[1][2] - 6) < 1e-9 and abs(x[1][3] - 6) < 1e-9 and abs(x[1][0][2]) < 1e-9]
        rgb = pads[0][2]
        yi = 35.0 - 6.0 - MID_PAD_IN
        c = (0.0, ys - s * yi, 0.0)
        fs.append(("cyl", (c, (0, 0, 1), 6, 6), rgb))
        out.append((side, c))
    return out


def build_model():
    links, joints, fasteners = B18.build_model()
    mp = _mass_parts()
    frames = X.servo_frames(links)
    LOG["gaps_before"], _ = SF.scan(links, mp)
    rows_s, _ = SF.scan(links, mp, per_solid=True)
    LOG["pockets"] = widen_pockets(links, rows_s, frames)
    LOG["gaps_after"], _ = SF.scan(links, mp)
    LOG["roll_spacers"] = roll_servo_spacers(links, frames)
    LOG["horn_holes_resized"] = horn_hole_size(links)
    LOG["case_holes"], LOG["case_holes_skipped"], LOG["wall_warnings"] = case_holes(links, frames, mp)
    LOG["mid_pads"] = add_mid_pads(links)
    LOG["outer_pads"] = move_outer_pads(links)
    LOG["sole_pockets_filled"] = sole_fill_pockets(links)
    LOG["frames"] = {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in f.items()} for k, f in frames.items()}
    return links, joints, fasteners


# ------------------------------------------------------------------ STEP with the official servo
def servo_step_parts(links):
    """part names left out of the assembly STEP (replaced by the official servo): the case boxes, horn / idler discs"""
    frames = X.servo_frames(links)
    skip = set(frames)
    for ln, L in links.items():
        for pn, ss in L["parts"].items():
            if re.search(r"(_horn|_idler)(_X[CML]430)?$", pn) and all(s[0] == "cyl" for s in ss):
                skip.add(pn)
    return skip, frames


def _official(w, path=SERVO_STEP):
    """append the official STEP entities (renumbered, contexts mapped to ours); returns (root pd, root rep, root origin)"""
    t = open(path).read()
    data = t.split("DATA;")[1].split("ENDSEC;")[0]
    D = {int(m.group(1)): " ".join(m.group(2).split()) for m in re.finditer(r"#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|$)", data, re.S)}
    refs = lambda s: [int(x) for x in re.findall(r"#(\d+)", re.sub(r"'[^']*'", "", s))]
    ctx_map = {}
    for i, v in D.items():
        if v.startswith("MECHANICAL_CONTEXT"):
            ctx_map[i] = w.pc
        elif v.startswith("DESIGN_CONTEXT"):
            ctx_map[i] = w.pdc
        elif v.startswith("APPLICATION_CONTEXT"):
            ctx_map[i] = w.app
    anchors = [i for i, v in D.items() if v.startswith(("SHAPE_DEFINITION_REPRESENTATION", "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
                                                         "SHAPE_REPRESENTATION_RELATIONSHIP", "NEXT_ASSEMBLY_USAGE_OCCURRENCE"))]
    keep, stack = set(), list(anchors)
    while stack:
        i = stack.pop()
        if i in keep or i in ctx_map:
            continue
        keep.add(i); stack += refs(D[i])
    base = w.n
    new = {old: base + k + 1 for k, old in enumerate(sorted(keep))}
    new.update(ctx_map)

    def rewrite(v):
        parts = re.split(r"('[^']*')", v)
        return "".join(p if p.startswith("'") else re.sub(r"#(\d+)", lambda m: f"#{new[int(m.group(1))]}", p) for p in parts)
    for old in sorted(keep):
        v = rewrite(D[old])
        if v.startswith("PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE"):
            v = v.replace(".NOT_KNOWN.", ".MADE.")
        w.lines.append(f"#{new[old]}={v};")
    w.n = base + len(keep)
    # root: product DC11_B01_DUMMY_ASM
    pid = next(i for i, v in D.items() if v.startswith("PRODUCT(") and "DUMMY_ASM" in v)
    pdf = next(i for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_FORMATION") and refs(v)[0] == pid)
    pd = next(i for i, v in D.items() if v.startswith("PRODUCT_DEFINITION(") and refs(v)[0] == pdf)
    pds = next(i for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_SHAPE") and refs(v)[0] == pd)
    sdr = next(i for i, v in D.items() if v.startswith("SHAPE_DEFINITION_REPRESENTATION") and refs(v)[0] == pds)
    rep = refs(D[sdr])[1]; org = refs(D[rep])[0]
    return {"pd": new[pd], "rep": new[rep], "origin": new[org], "name": "ROBOTIS_X430"}, len(keep)


def write(links, path):
    from stepwriter import StepWriter
    skip, frames = servo_step_parts(links)
    w = StepWriter(os.path.basename(path))
    saved = build.to_local; build.to_local = build4.to_local
    try:
        servo, n_off = _official(w)
        subs = []
        for lname, L in links.items():
            o = np.array(L["origin"], float)
            kids = [(w.part(pn, [build4.to_local(s, tuple(o)) for s in solids]), w.axis()) for pn, solids in L["parts"].items() if pn not in skip]
            for sv, fr in frames.items():
                if fr["link"] == lname:
                    kids.append((dict(servo, name=f"{sv}_ROBOTIS"), w.axis(tuple(fr["O"] - o), tuple(fr["Z"]), tuple(fr["X"]))))
            subs.append((_assembly(w, lname, kids), w.axis(tuple(o))))
        _assembly(w, "biped_robot", subs)
    finally:
        build.to_local = saved
    open(path, "w").write(w.text())
    LOG["step"] = dict(official_entities=n_off, servos_placed=len(frames), replaced_parts=sorted(skip))
    return w


def _assembly(w, name, children):
    """like StepWriter.assembly but each child has a full placement (axis id) instead of a translation"""
    pd, pds = w._product(name)
    origin = w.axis()
    rep = w.add("SHAPE_REPRESENTATION('" + name + "',(" + ",".join(f"#{i}" for i in [origin] + [p for _, p in children]) + f"),#{w.ctx})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})")
    for k, (child, place) in enumerate(children):
        nauo = w.add(f"NEXT_ASSEMBLY_USAGE_OCCURRENCE('{name}_{k + 1}','{child['name']}','',#{pd},#{child['pd']},$)")
        ppds = w.add(f"PRODUCT_DEFINITION_SHAPE('Placement','Placement of an item',#{nauo})")
        idt = w.add(f"ITEM_DEFINED_TRANSFORMATION('','',#{child['origin']},#{place})")
        rr = w.add(f"( REPRESENTATION_RELATIONSHIP('','',#{child['rep']},#{rep}) "
                   f"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{idt}) SHAPE_REPRESENTATION_RELATIONSHIP() )")
        w.add(f"CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{rr},#{ppds})")
    return {"pd": pd, "rep": rep, "origin": origin, "name": name}


def _md_tables():
    L = ["# v19 servo seat log", "", f"SERVO_CLR {SERVO_CLR} mm, CASE_HOLE_D {CASE_HOLE_D} mm, HORN_HOLE_D {HORN_HOLE_D} mm", "",
         "## X430 case holes (official STEP, servo frame)", "", X.table(), "",
         "## gaps between servo case faces and printed parts (same link), before -> after", "",
         "| servo | face | part | gap before (mm) | gap after (mm) | face covered | in contact before |", "|---|---|---|---|---|---|---|"]
    after = {(r["servo"], r["face"], r["part"]): r for r in LOG["gaps_after"]}
    for r in LOG["gaps_before"]:
        a = after.get((r["servo"], r["face"], r["part"]), {})
        L.append(f"| {r['servo']} | {r['face']} | {r['part']} | {r['gap_mm']:.3f} | {a.get('gap_mm', float('nan')):.3f} | {100 * r['coverage']:.0f}% | {100 * r['contact_frac']:.0f}% |")
    L += ["", "## pockets widened", ""] + [f"- {p['servo']} {p['face']} {p['part']}[{p['solid_index']}]: {p['action']}" for p in LOG["pockets"]]
    L += ["", f"## case holes cut ({len(LOG['case_holes'])})", "", "| servo | hole # | part | axis | u, v | plate (mm) | wall left (mm) |", "|---|---|---|---|---|---|---|"]
    for h in LOG["case_holes"]:
        L.append(f"| {h['servo']} | {h['hole']} | {h['part']} | {h['axis']} | {h['uv'][0]}, {h['uv'][1]} | {h['plate_mm']} | {h['wall_left_mm']} |")
    L += ["", f"## holes not cut ({len(LOG['case_holes_skipped'])})", ""] + [f"- {s['servo']} #{s['hole']} {s['part']}: {s['reason']}" for s in LOG["case_holes_skipped"]]
    L += ["", f"## wall left < {MIN_WALL} mm ({len(LOG['wall_warnings'])})", ""] + [f"- {h['servo']} #{h['hole']} {h['part']}: {h['wall_left_mm']} mm" for h in LOG["wall_warnings"]]
    L += ["", f"horn screw holes resized to {HORN_HOLE_D}: {LOG['horn_holes_resized']}", "", "mid pads: " + str(LOG["mid_pads"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    links, joints, fasteners = build_model()
    os.makedirs(OUT, exist_ok=True)
    w = write(links, os.path.join(OUT, "biped_robot_v19.stp"))
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners, "theta_max_deg": B18.THETA_MAX,
               "servo_clr_mm": SERVO_CLR, "case_hole_d_mm": CASE_HOLE_D, "horn_hole_d_mm": HORN_HOLE_D,
               "step_replaced_parts": LOG["step"]["replaced_parts"]}, open(os.path.join(OUT, "model.json"), "w"), indent=1)
    json.dump({k: v for k, v in LOG.items()}, open(os.path.join(OUT, "servo_seat_log.json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, "SERVO_SEATS_v19.md"), "w").write(_md_tables())
    print("entities", w.n, "official servo entities", LOG["step"]["official_entities"], "pockets", len(LOG["pockets"]),
          "case holes", len(LOG["case_holes"]), "skipped", len(LOG["case_holes_skipped"]), "wall warnings", len(LOG["wall_warnings"]),
          "horn holes", LOG["horn_holes_resized"])
