"""v21 body = v21 head (build21_head: v20 geometry, nod -20 .. +28) + torso / head print fits opened up (first print
kit for the torso and the head, owner 2026-11-27).  Only torso / head printed parts change; legs and arms stay v20.

Fits (build20 rules: sliding / snap gaps >= 0.25 per side, fits per side >= 0.3 where a part drops over another)
  * snap hooks (build14.hook): every hook arm sat 0.15 from the face its barb hooks behind (head seam hooks, head
    bottom hook, face plate hooks -> catch bars, crest hooks -> top wall slots, lid hooks -> frame wall windows).
    The arm (with its barb) moves SNAP_EXTRA away from that face -> 0.25; the barb catch shrinks 0.85 -> 0.75 and
    the room behind the arm in the crest slots 1.0 -> 0.9 (still more than the 0.75 the barb has to deflect)
  * head seam tongue (back part, under the front part's top wall): 0.15 -> 0.25
  * shorts flange hole round the frame walls (y +-60.2 for the walls at +-60): 0.2 -> 0.3 per side (the 0.07 of
    sim/tests/fit_gaps_v20.py was the flange bottom resting on the pelvis plate top, a seat, not a gap)
  * cover dovetails (frame rail <-> cover channel): flank gap 0.18 / face gap 0.2 -> 0.3 (channel widened)
  * lid plug back edge at x -65 dropped into the frame flush with the back wall face (gap 0): -> -65 + LID_BACK_GAP
Nothing else is moved; parts are written to out20/print_parts/<name>.stp (torso / head only) by __main__.
"""
import os, sys, json, math
import numpy as np
import build21_head as BH
from build21_head import *          # noqa  (build20 names)
import build14 as B14

SNAP_EXTRA = 0.1            # hook arm face -> catch face 0.15 -> 0.25
TONGUE_GAP = 0.25
SHORTS_GAP = 0.3            # shorts flange hole -> frame wall, per side
DOVE_GAP = 0.3              # cover channel -> frame rail, per face
LID_BACK_GAP = 0.3          # lid plug back edge (x -65) -> frame back wall face / OpenCR channel
LOG = {}
BODY_LINKS = ("torso", "head", "ears", "nod_crank", "nod_rod", "ear_rod")
# being printed by the owner (2026-11-27): geometry and STEP stay as v20; fits to them change on the receiving part only
FROZEN = ("eye_L", "eye_R", "mouth", "camera_clip", "ear_disc_L", "ear_disc_R", "antenna_L", "antenna_R", "nod_rod",
          "ear_rod", "shelf_pins", "electronics_shelf")
PART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out20", "print_parts")


def _hook(ins, flex, root, catch, face, w0, w1, rgb, t=B14.HT_, b=B14.HB):
    sf = 1 if flex[0] == "+" else -1
    LOG.setdefault("hooks", []).append(dict(flex=flex, face_before=round(face, 2), face=round(face - sf * SNAP_EXTRA, 2)))
    return _HOOK0(ins, flex, root, catch, face - sf * SNAP_EXTRA, w0, w1, rgb, t=t, b=b)


def _cover_channel(s, c, rgb):
    """build14.cover_channel with DOVE_GAP between the female dovetail and the frame rail on every face"""
    W, poly, box = B14.W, B14.poly, B14.box
    g = DOVE_GAP
    y = lambda d: s * (60.0 + d)
    # rail (build14.frame_rail): face d 8.8, tip d 11.8, flank |x - c| 2.5 -> 4.0 over d 8.8 -> 11.8;
    # channel flank = rail flank moved g along its outward normal (0.894, -0.447)
    a0, a1 = 8.8 + g, 11.8 + g
    fx = lambda d: 2.5 + 0.894 * g + (d - 8.8 + 0.447 * g) * 0.5
    C = [W(x=c - 7, y=y(a0)), W(x=c - fx(a0), y=y(a0)), W(x=c - fx(a1), y=y(a1)), W(x=c + fx(a1), y=y(a1)),
         W(x=c + fx(a0), y=y(a0)), W(x=c + 7, y=y(a0)), W(x=c + 7, y=y(13.8)), W(x=c - 7, y=y(13.8))]
    out = [poly("z", 280, 316, C, rgb),
           box((c - 7, min(y(a0), y(13.8)), 316), (c + 7, max(y(a0), y(13.8)), 318), rgb)]
    for x0, x1 in ((c - 7, c - 4.5), (c + 4.5, c + 7)):
        out.append(poly("x", x0, x1, [W(y=y(a0), z=280), W(y=y(13.8), z=280), W(y=y(13.8), z=275.2 + (a0 - 9.0))], rgb))
    LOG.setdefault("dovetail", []).append(dict(side=s, x=c, face_d=round(a0, 2), tip_d=round(a1, 2),
                                               flank_at_face=round(fx(a0), 3), flank_at_tip=round(fx(a1), 3)))
    return out


_HOOK0 = B14.hook


def tongue(links):
    """head seam tongue (back part, x XS .. XS + 6): top face TOP_IN - 0.15 -> TOP_IN - TONGUE_GAP"""
    hb = links["head"]["parts"]["head_back"]
    top0 = B14.TOP_IN - 0.15
    n = 0
    for i, x in enumerate(hb):
        if x[0] == "rprism" and x[1][0] == "z" and abs(x[1][2] - top0) < 1e-6:
            a = x[1]
            hb[i] = (x[0], (a[0], a[1], B14.TOP_IN - TONGUE_GAP, a[3], a[4], a[5]), x[2]); n += 1
    return dict(solids=n, gap=(0.15, TONGUE_GAP))


def shorts_flange(links):
    """flange hole edges along the frame walls (|y| 60.2) -> 60 + SHORTS_GAP"""
    sh = links["torso"]["parts"]["shorts"]
    n = 0
    for i, x in enumerate(sh):
        if x[0] == "rprism" and x[1][0] == "z" and x[1][4]:
            a = x[1]
            inn = [[(p[0], math.copysign(60.0 + SHORTS_GAP, p[1]) if abs(abs(p[1]) - 60.2) < 1e-6 else p[1]) + tuple(p[2:])
                    for p in l] for l in a[4]]
            n += sum(abs(abs(p[1]) - 60.2) < 1e-6 for l in a[4] for p in l)
            sh[i] = (x[0], (a[0], a[1], a[2], a[3], inn, a[5]), x[2])
    return dict(points=n, gap=(0.2, SHORTS_GAP))


def lid_back(links):
    """lid plug (z 330 .. 336.3, the U outline of build17): outline points at x -65 -> -65 + LID_BACK_GAP"""
    lid = links["torso"]["parts"]["lid"]
    n = 0
    for i, x in enumerate(lid):
        if x[0] == "rprism" and x[1][0] == "z" and x[1][1] >= 329.9 and x[1][2] <= 336.4 and len(x[1][3]) >= 8:
            a = x[1]
            o = [(-65.0 + LID_BACK_GAP if abs(p[0] + 65.0) < 1e-6 else p[0],) + tuple(p[1:]) for p in a[3]]
            n += sum(abs(p[0] + 65.0) < 1e-6 for p in a[3])
            lid[i] = (x[0], (a[0], a[1], a[2], o, a[4], a[5]), x[2])
    return dict(points=n, gap=(0.0, LID_BACK_GAP))


def build_model():
    LOG.clear()
    saved = (B14.hook, B14.cover_channel)
    B14.hook, B14.cover_channel = _hook, _cover_channel
    try:
        links, joints, fasteners = BH.build_model()
    finally:
        B14.hook, B14.cover_channel = saved
    LOG["tongue"] = tongue(links)
    LOG["shorts_flange"] = shorts_flange(links)
    LOG["lid_back"] = lid_back(links)
    return links, joints, fasteners


def export_body(links, names):
    """write the torso / head printed parts like export_parts.py (same print direction as parts.json), update their
    parts.json rows in place"""
    sys.argv = sys.argv[:1] + ["build21_body", PART_DIR]
    import export_parts as EP, printcheck, build4
    from scipy import ndimage
    table = json.load(open(os.path.join(PART_DIR, "parts.json")))
    rows = {r["part"]: r for r in table}
    done = []
    for ln, pn, sol in printcheck.printed_parts(links):
        if pn not in names:
            continue
        up = rows[pn]["up"]
        rs = [EP.rot_solid(s, EP.ROT[up]) for s in sol]
        bbs = [build4.aabb(build4.sampled(s)) for s in rs]
        lo = [min(b[i][0] for b in bbs) for i in range(3)]
        hi = [max(b[i][1] for b in bbs) for i in range(3)]
        rs = [EP.rot_solid(s, np.eye(3), (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])) for s in rs]
        h = 0.6 if max(b - a for a, b in zip(lo, hi)) < 90 else 1.0
        V, _ = printcheck.voxelize(rs, h)
        need, _ = printcheck.overhang(V[ndimage.find_objects(V.astype(int))[0]], h)
        build4.write({pn: {"origin": (0, 0, 0), "parts": {pn: rs}}}, os.path.join(PART_DIR, f"{pn}.stp"))
        rows[pn].update(link=ln, size_mm="x".join(str(round(b - a)) for a, b in zip(lo, hi)), support_mm2=round(need),
                        volume_cm3=round(sum(build4.volume(s) for s in sol) / 1000, 1),
                        colours=sorted({tuple(round(c, 2) for c in s[2]) for s in sol}))
        done.append(pn)
    with open(os.path.join(PART_DIR, "parts.json"), "w") as f:
        json.dump(table, f, indent=1, ensure_ascii=False)
    return done


if __name__ == "__main__":
    links, joints, fasteners = build_model()
    names = [r["part"] for r in json.load(open(os.path.join(PART_DIR, "parts.json")))
             if r["link"] in BODY_LINKS and not r["part"].startswith("arm_") and r["part"] not in FROZEN]
    done = export_body(links, names)
    LOG["written"] = done
    LOG["missing"] = sorted(set(names) - set(done))
    json.dump(LOG, open(os.path.join(PART_DIR, "..", "build21_body_log.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in LOG.items() if k != "hooks"}, indent=1, default=str))
    print("hooks", len(LOG.get("hooks", [])))
