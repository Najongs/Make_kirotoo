"""v6: the v5 robot made printable (PLA, 256 mm bed) and assemblable with M3 bolts + captive hex nuts.

What changes from v5
- Head: a 'TV' shell that prints back-down with no support: straight walls, 45-degree stepped chamfers at
  the back and front edges, front left open. The white face plate is set 8 mm inside the front rim and is
  held by two M3 bolts into nut bosses; the open front is the access for the tilt-horn screws.
  Crest (camera housing) prints top-down and is clamped to the head shell together with the internal head
  frame by four M3 bolts. Ears print lid-down and bolt through the head side wall; antennas sit in a socket
  inside each ear.
- Torso: covers and shorts are one print per half (colour change at z 225), flat side panels carry M3 holes
  into standoffs on the torso side plates; the arms bolt through the same standoffs.
- Legs: thigh / shank shells bolt to a nut block on the front of each frame; the shoe and sole are one print.
- Every ring that must be one piece is a true ring (inner loop), not two halves.
Servo case / horn screw holes are added in a later step from the official ROBOTIS STEP.
"""
import math, os, json
import build4, build5, build
from build4 import RB, SERVO, CYL, DSC, PROF, HA, SL, XC, XM, CAM, RUBBER
from build5 import SKY, NAVY, WHITE, ORANGE, DARK, PAN_Z, ZT
from shapes import rrect, annulus_sector, circle_loop, orient

PRINT = build.PRINT
_V5_TORSO, _V5_NECK, _V5_LEG = build5.torso, build5.neck, build5.right_leg
IDX, UVI = build.IDX, build.UVI

T = 1.2                     # shell wall
NUT_AF, NUT_T = 5.8, 2.6    # M3 hex nut pocket (5.5 AF nut + 0.3 clearance), depth
M3 = 1.7                    # M3 clearance hole radius (3.4 mm)


# ---------------------------------------------------------------- helpers
def PROFI(axis, a0, a1, outer, inners, rgb, holes=()):
    return ("rprism", (axis, min(a0, a1), max(a0, a1), orient(outer, True),
                       [orient(l, False) for l in inners], list(holes)), rgb)


def uv(axis, p):
    iu, iv = UVI[axis]
    return p[iu], p[iv]


def plate(axis, lo, hi, rgb, holes=(), cuts=(), r=0.0):
    """Box lo..hi (world) extruded along `axis`; holes: [(world point, radius)]; cuts: [[world points]]."""
    k, (iu, iv) = IDX[axis], UVI[axis]
    out = rrect(lo[iu], hi[iu], lo[iv], hi[iv], r)
    hs = [(p[iu], p[iv], rr) for p, rr in holes]
    inn = [[(p[iu], p[iv], 0) for p in c] for c in cuts]
    return PROFI(axis, lo[k], hi[k], out, inn, rgb, hs)


def hexloop(u, v, af=NUT_AF):
    R = af / math.sqrt(3)
    return [(u + R * math.cos(math.radians(30 + 60 * i)), v + R * math.sin(math.radians(30 + 60 * i)), 0)
            for i in range(6)]


def nut_block(axis, a_hole, a_nut, a_end, lo, hi, pts, rgb):
    """Block lo..hi (world, its extent along `axis` is given by a_hole..a_end) with M3 holes at the world
    points `pts` running along `axis`; the stretch a_nut..a_end holds hex nut pockets (open at a_end)."""
    k = IDX[axis]
    L1, H1 = list(lo), list(hi)
    L1[k], H1[k] = min(a_hole, a_nut), max(a_hole, a_nut)
    L2, H2 = list(lo), list(hi)
    L2[k], H2[k] = min(a_nut, a_end), max(a_nut, a_end)
    iu, iv = UVI[axis]
    return [plate(axis, L1, H1, rgb, holes=[(p, M3) for p in pts]),
            PROFI(axis, L2[k], H2[k], rrect(L2[iu], H2[iu], L2[iv], H2[iv], 0),
                  [hexloop(p[iu], p[iv]) for p in pts], rgb)]


def box_shell(axis, a0, a1, cu, cv, A, B, R, t, rgb, sides=("+u", "-u", "+v", "-v"), holes=None,
              cuts=None, u_open_at=None):
    """Rounded-rectangle tube along `axis` built as 4 quarter-arc corners + 4 flat panels, so panels can
    carry holes and cut-outs. sides: which panels to build. A corner is built when both of its panels are
    built. u_open_at: for a C open on the -u side, the -u ends of the v-panels (a u coordinate)."""
    holes, cuts = holes or {}, cuts or {}
    k, (iu, iv) = IDX[axis], UVI[axis]
    W = lambda u, v, a: [(u if i == iu else v if i == iv else a) for i in range(3)]
    out = []
    quad = {("+u", "+v"): (1, 1, 0, 90), ("-u", "+v"): (-1, 1, 90, 180),
            ("-u", "-v"): (-1, -1, 180, 270), ("+u", "-v"): (1, -1, 270, 360)}
    for (su, sv), (fu, fv, s0, s1) in quad.items():
        if su in sides and sv in sides:
            out.append(PROF(axis, a0, a1, annulus_sector(cu + fu * (A - R), cv + fv * (B - R), R - t, R, s0, s1), rgb))
    for side in sides:
        f = 1 if side[0] == "+" else -1
        if side[1] == "u":
            u0, u1 = sorted((cu + f * A, cu + f * (A - t)))
            v0, v1 = cv - (B - R), cv + (B - R)
            lo, hi = W(u0, v0, a0), W(u1, v1, a1)
            naxis = [n for n in "xyz" if IDX[n] == iu][0]
        else:
            v0, v1 = sorted((cv + f * B, cv + f * (B - t)))
            uu0 = cu - (A - R) if "-u" in sides else (u_open_at if u_open_at is not None else cu - A)
            uu1 = cu + (A - R) if "+u" in sides else cu + A
            lo, hi = W(uu0, v0, a0), W(uu1, v1, a1)
            naxis = [n for n in "xyz" if IDX[n] == iv][0]
        out.append(plate(naxis, lo, hi, rgb, holes=holes.get(side, ()), cuts=cuts.get(side, ())))
    return out


def ring(axis, a0, a1, cu, cv, A, B, R, t, rgb):
    return PROFI(axis, a0, a1, rrect(cu - A, cu + A, cv - B, cv + B, R),
                 [rrect(cu - A + t, cu + A - t, cv - B + t, cv + B - t, max(R - t, 0.5))], rgb)


# ---------------------------------------------------------------- head (tilt link)
ZH = ZT + 20.0                       # head centre
HW, HD, HH, HR, CH = 100.0, 70.0, 75.0, 50.0, 10   # half width (y), half depth (x), half height, corner R, chamfer
FZ = ZH - 5                          # face centre
FX = 60.0                            # face plate back face (x)
TOP_IN = ZH + HH - T                 # inside of the head top wall
CZ0, CZ1 = ZH + HH, ZH + HH + 26     # crest bottom / top


def head():
    P = {}
    sh = []
    # chamfer steps (1 mm per 1 mm = 45 deg): back x -70..-60, front 60..70
    for i in range(CH):
        o = CH - i
        sh.append(ring("x", -HD + i, -HD + i + 1, 0, ZH, HW - o, HH - o, HR - o, T, SKY))
        sh.append(ring("x", HD - CH + i, HD - CH + i + 1, 0, ZH, HW - i - 1, HH - i - 1, HR - i - 1, T, SKY))
    sh.append(PROF("x", -HD, -HD + T, rrect(-(HW - CH), HW - CH, ZH - (HH - CH), ZH + (HH - CH), HR - CH), SKY))
    crest_bolts = [(x, y, 0) for x in (15, 40) for y in (-26, 26)]
    sh += box_shell("x", -HD + CH, HD - CH, 0, ZH, HW, HH, HR, T, SKY, holes={
        "+v": [((x, y, CZ0), M3) for x, y, _ in crest_bolts],
        "+u": [((x, HW, ZH - 5), M3) for x in (-12, 12)],
        "-u": [((x, -HW, ZH - 5), M3) for x in (-12, 12)]},
        cuts={"+v": [[(28, -10, CZ0), (31, -10, CZ0), (31, 10, CZ0), (28, 10, CZ0)]],          # camera FFC slot
              "-v": [[(-38, -20, 0), (30, -20, 0), (50, 0, 0), (30, 20, 0), (-38, 20, 0)]]})   # neck opening
    # face bolt bosses (nut pocket faces the face plate) + 45-degree gussets below them in print direction
    for s in (1, -1):
        y0, y1 = sorted((s * 74, s * 99))
        sh += nut_block("x", FX - 7, FX - NUT_T, FX, (0, y0, FZ - 6), (0, y1, FZ + 6), [(0, s * 80, FZ)], SKY)
        sh.append(PROFI("z", FZ - 6, FZ + 6, [(FX - 7, s * 74, 0), (FX - 7, s * 99, 0), (FX - 32, s * 99, 0)], [], SKY))
    P["head_shell"] = sh

    # face plate (white), eyes and smile (navy, glued on)
    P["face_plate"] = [PROF("x", FX, FX + 2, rrect(-88, 88, FZ - 60, FZ + 60, 45), WHITE,
                            holes=[(y, FZ, M3) for y in (-80, 80)])]
    P["face_eyes"] = [PROF("x", FX + 2, FX + 3.2, rrect(y - 9, y + 9, FZ + 13 - 18, FZ + 13 + 18, 9), NAVY) for y in (-34, 34)]
    P["face_mouth"] = [PROF("x", FX + 2, FX + 3.2, annulus_sector(0, FZ - 7, 13, 16.5, 205, 335), NAVY)]

    # ears: cup (ring with a slot on top for the antenna) + lid with two bolt holes; sky disc hides the bolts
    ez = ZH - 5
    for s, nm in ((1, "L"), (-1, "R")):
        ya, yb = (HW, HW + 16.8) if s > 0 else (-HW - 16.8, -HW)
        la, lb = (HW + 16.8, HW + 18) if s > 0 else (-HW - 18, -HW - 16.8)
        P[f"ear_{nm}"] = [PROF("y", ya, yb, annulus_sector(ez, 0, 28.8, 30, 8, 352), NAVY),
                          PROF("y", la, lb, circle_loop(ez, 0, 30), NAVY, holes=[(ez, x, M3) for x in (-12, 12)]),
                          plate("z", (-7, min(s * 102, s * 116.8), ez + 10), (7, max(s * 102, s * 116.8), ez + 26),
                                NAVY, holes=[((0, s * 108, 0), 4.15)])]                    # antenna socket
        P[f"ear_disc_{nm}"] = [CYL((0, s * (HW + 18), ez), "y" if s > 0 else "-y", 19, 1.5, SKY)]
        P[f"antenna_{nm}"] = [CYL((0, s * 108, ez + 12), "z", 4, 95, ORANGE)]

    # crest: printed top-down; walls + roof, camera window, four bosses with side nut slots
    cu, cv, A, B = 30.0, 0.0, 22.0, 34.0
    cr = box_shell("z", CZ0, CZ1 - T, cu, cv, A, B, 10, T, SKY,
                   holes={"+u": [((cu + A, 0, (CZ0 + CZ1) / 2), 7.5)]})
    cr.append(PROF("z", CZ1 - T, CZ1, rrect(cu - A, cu + A, cv - B, cv + B, 10), SKY))
    for x, y, _ in crest_bolts:
        sq = (x - 4.5, x + 4.5, y - 4.5, y + 4.5)
        cr.append(plate("z", (sq[0], sq[2], CZ0), (sq[1], sq[3], CZ0 + 4), SKY, holes=[((x, y, 0), M3)]))
        sgn = 1 if y < 0 else -1                       # slot opens toward the crest centre line
        c0, c1, rh = x - NUT_AF / 2, x + NUT_AF / 2, NUT_AF / math.sqrt(3)
        if sgn > 0:      # C-shape: square minus a nut channel from the bolt axis to the +y side
            poly = [(sq[0], sq[2]), (sq[1], sq[2]), (sq[1], sq[3]), (c1, sq[3]), (c1, y - rh), (c0, y - rh),
                    (c0, sq[3]), (sq[0], sq[3])]
        else:
            poly = [(sq[0], sq[2]), (c0, sq[2]), (c0, y + rh), (c1, y + rh), (c1, sq[2]), (sq[1], sq[2]),
                    (sq[1], sq[3]), (sq[0], sq[3])]
        cr.append(PROF("z", CZ0 + 4, CZ0 + 4 + NUT_T, [(a, b, 0) for a, b in poly], SKY))
        cr.append(plate("z", (sq[0], sq[2], CZ0 + 4 + NUT_T), (sq[1], sq[3], CZ1 - T), SKY, holes=[((x, y, 0), M3)]))
    P["crest"] = cr
    th = math.radians(build5.VIEW)
    dv = (math.cos(th), 0.0, -math.sin(th))
    cmid = (CZ0 + CZ1) / 2
    P["camera_small"] = [RB(36, 46, -12, 12, cmid - 9, cmid + 9, CAM, "x", 2)]
    P["camera_lens"] = [("cyl", ((cu + A - 6, 0.0, cmid), dv, 6.0, 7.0), CAM)]

    # head frame: bridge under the roof (bolted with the crest) + two side plates down to the tilt horns
    fb = [((x, y, 0), M3) for x, y, _ in crest_bolts]
    hf = [plate("z", (-15, -32, TOP_IN - 4), (45, 32, TOP_IN), DARK, holes=fb)]
    for s in (1, -1):
        y0, y1 = sorted((s * 19, s * 22))
        hf.append(PROF("y", y0, y1, [(ZT - 15, -15, 0), (ZT - 15, 15, 0), (TOP_IN - 4, 45, 0), (TOP_IN - 4, -15, 0)], DARK))
    hf.append(DSC((0, 18, ZT), "y", 8))                                       # child disc of head_tilt
    P["head_frame"] = hf
    return P


# ---------------------------------------------------------------- torso
COVER_PTS = {"front": [(10, 255)], "back": [(-22, 310), (-22, 255)]}       # cover-only bolts (nut in side plate)
ARM_PTS = [(0, 318), (0, 300)]                                              # arm bolts (nut in the arm)


def torso():
    P, F = _V5_TORSO()
    for k in ("shorts_band", "torso_cover_front", "torso_cover_back"):
        P.pop(k, None)
    cu, A, Y, R = (build4.COV_XF + build4.COV_XB) / 2, (build4.COV_XF - build4.COV_XB) / 2, build4.COV_Y, build4.COV_R
    z0, z1 = 205.0, 336.0                     # shorts (navy) z 205-225, shirt (white) above: colour change at 225
    allpts = COVER_PTS["front"] + COVER_PTS["back"] + ARM_PTS
    for half, sgn in (("front", 1), ("back", -1)):
        sides = ("+u", "+v", "-v") if sgn > 0 else ("-u", "+v", "-v")
        hp = [(x, z) for x, z in allpts if (x > cu) == (sgn > 0)]
        shell = box_shell("z", z0, 225.0, cu, 0, A, Y, R, T, NAVY, sides=sides) + \
            box_shell("z", 225.0, z1, cu, 0, A, Y, R, T, WHITE, sides=sides,
                      holes={"+v": [((x, Y, z), M3) for x, z in hp], "-v": [((x, -Y, z), M3) for x, z in hp]})
        # the v-panels stop at the split line
        out = []
        for s_ in shell:
            k_, a_, rgb_ = s_
            if k_ == "rprism" and a_[0] == "y":                    # side panel: trim to this half
                ax, ya0, ya1, outer, inn, hs = a_
                us = [p[1] for p in outer]                          # (u,v) = (z,x) for axis y
                lo_x = cu + 0.25 if sgn > 0 else min(us)
                hi_x = max(us) if sgn > 0 else cu - 0.25
                outer2 = rrect(min(p[0] for p in outer), max(p[0] for p in outer), lo_x, hi_x, 0)
                s_ = ("rprism", (ax, ya0, ya1, outer2, inn, [h for h in hs if lo_x < h[1] < hi_x]), rgb_)
            out.append(s_)
        P[f"torso_cover_{half}"] = out
    # side plates: hole layer + hex-nut layer (cover-only bolts), standoffs out to the cover
    for s, nm in ((-1, "R"), (1, "L")):
        old = P[f"torso_side_{nm}"]
        yo, yi = s * 60.0, s * 55.0                               # outer / inner faces
        cov = COVER_PTS["front"] + COVER_PTS["back"]
        pts_all = [(x, s * 57, z) for x, z in cov + ARM_PTS]
        main = plate("y", (-65, min(yo, s * 57.4), 235), (50, max(yo, s * 57.4), 330), PRINT,
                     holes=[(p, M3) for p in pts_all], r=12)
        # inner layer: hex pockets for cover-only bolts, round holes for arm bolts (bolt head sits here)
        k, (iu, iv) = IDX["y"], UVI["y"]
        inner = PROFI("y", s * 57.4, yi, rrect(235, 330, -65, 50, 12),
                      [hexloop(z, x) for x, z in cov], PRINT, [(z, x, M3) for x, z in ARM_PTS])
        stand = [plate("y", (x - 5, min(yo, s * 73.8), z - 5), (x + 5, max(yo, s * 73.8), z + 5), PRINT,
                       holes=[((x, 0, z), M3)]) for x, z in cov + ARM_PTS]
        # 45-degree gussets under the standoffs (the side plate prints standing, z up)
        gus = [PROFI("x", x - 5, x + 5, [(yo, z - 5, 0), (s * 73.8, z - 5, 0), (yo, z - 5 - 13.8, 0)], [], PRINT)
               for x, z in cov + ARM_PTS]
        # the v4 top tabs (z 324) and shelf ledges (z 268) stick 10 mm out of the inner face: gusset them too
        # (split around each bolt so the bolt and its nut keep a clear channel)
        for x0, x1, zb in ((-65, -59, 324), (-51, -45, 324), (30, 36, 324), (44, 50, 324),
                           (-62, -57, 268), (-49, -44, 268), (20, 25, 268), (33, 38, 268)):
            gus.append(PROFI("x", x0, x1, [(yi, zb, 0), (s * 45.0, zb, 0), (yi, zb - 10, 0)], [], PRINT))
        P[f"torso_side_{nm}"] = [main, inner] + old[1:] + stand + gus
        # arm: rounded-square sleeve (flat inner panel carries the bolt holes) + shoulder dome,
        # nut block inside the sleeve, 45-degree gusset under it
        y = s * 91.0
        inner_side = "-v" if s > 0 else "+v"
        arm = box_shell("z", 262, 325, 0, y, 16, 16, 12, T, SKY,
                        holes={inner_side: [((0, s * 75, z), M3) for _, z in ARM_PTS]})
        arm.append(("scap", ((0, y, 325), (0, 0, 1), 17.7, 16.5, 0.0), SKY))
        arm += nut_block("y", s * 76.2, s * 81.4, s * 84.0, (-7, 0, 292), (7, 0, 324),
                         [(0, 0, z) for _, z in ARM_PTS], SKY)
        arm.append(PROFI("x", -7, 7, [(s * 76.2, 292, 0), (s * 84.0, 292, 0), (s * 76.2, 284.2, 0)], [], SKY))
        P[f"arm_{nm}_sleeve"] = arm
        P.pop(f"arm_{nm}_shoulder", None)
    return P, F


def neck():
    return _V5_NECK()


# ---------------------------------------------------------------- legs
def right_leg():
    L, F, J = _V5_LEG()
    ys = -40.0
    # thigh side plates: square bottom corners so they join the servo walls (one printed piece)
    tf = L["thigh"]["thigh_frame"]
    yp = lambda y0, y1: PROF("y", y0, y1, [(146, -18.5, 0), (196, -18.5, 10), (196, 18.5, 10), (146, 18.5, 0)], PRINT)
    L["thigh"]["thigh_frame"] = [yp(ys - 22, ys - 18), yp(ys + 18, ys + 22)] + tf[2:]
    L["thigh"]["thigh_frame"] += nut_block("x", 18.5, 19.4, 22.0, (0, ys - 6, 119), (0, ys + 6, 141),
                                           [(0, ys, 124), (0, ys, 136)], PRINT)
    L["shank"]["shank_frame"] += nut_block("x", 18.5, 19.4, 22.0, (0, ys - 6, 52), (0, ys + 6, 76),
                                           [(0, ys, 58), (0, ys, 70)], PRINT)
    # shells: C open at the back, flat front panel with two M3 holes
    L["thigh"]["thigh_shell"] = box_shell("z", 118, 150, 0, ys - 0.5, 23.5, 27, 12, 1.5, WHITE,
                                          sides=("+u", "+v", "-v"), u_open_at=-20,
                                          holes={"+u": [((23.5, ys, z), M3) for z in (124, 136)]})
    L["shank"]["shank_shell"] = box_shell("z", 44, 98, 0, ys, 23.5, 30.5, 12, 1.5, SKY,
                                          sides=("+u", "+v", "-v"), u_open_at=-20,
                                          holes={"+u": [((23.5, ys, z), M3) for z in (58, 70)]}) + \
        box_shell("z", 98, 104, 0, ys, 23.5, 30.5, 12, 1.5, NAVY, sides=("+u", "+v", "-v"), u_open_at=-20)
    L["shank"].pop("shank_band", None)               # navy band = filament change at z 98 on the shank shell
    # foot: sole + shoe wall in one print
    sole = L["foot"]["foot_sole"]
    shoe = L["foot"].pop("shoe")
    xs = [p[0] for s_ in shoe for p in s_[1][3]]
    hx = min(xs)
    out = [(hx, ys - 35, 3), (60, ys - 35, 30), (60, ys + 35, 30), (hx, ys + 35, 3)]
    inn = [(hx + 1.6, ys - 33.4, 1.4), (58.4, ys - 33.4, 28.4), (58.4, ys + 33.4, 28.4), (hx + 1.6, ys + 33.4, 1.4)]
    L["foot"]["foot_sole"] = [(k, a, NAVY) if k == "rprism" else (k, a, rgb) for k, a, rgb in sole] + \
        [PROFI("z", 12, 28, out, [inn], NAVY)]
    return L, F, J


# ---------------------------------------------------------------- assembly
def build_model():
    saved = (build5.torso, build5.neck, build5.head, build5.right_leg)
    build5.torso, build5.neck, build5.head, build5.right_leg = torso, neck, head, right_leg
    try:
        return build5.build_model()
    finally:
        build5.torso, build5.neck, build5.head, build5.right_leg = saved


write = build4.write
sampled, aabb, volume, to_local = build4.sampled, build4.aabb, build4.volume, build4.to_local

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out6"
    os.makedirs(out, exist_ok=True)
    w = write(links, f"{out}/biped_robot_v6.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))
