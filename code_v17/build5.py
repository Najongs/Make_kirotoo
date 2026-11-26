"""v5: KIRO-mascot styling on the v4 mechanism (5+5 legs, pan/tilt head, no actuated arms).

Look: big rounded 'TV' head (sky blue) with a white face plate, navy eyes and smile, navy ear pods with
orange antennas and a top crest that houses the camera; white shirt torso, navy shorts band, fixed
decorative arms (sky sleeve, white hand), white thigh shells, sky shank shells with a navy band, navy shoes.

All shells are thin walls (1.2-1.6 mm) built from single-loop profiles (rings are split in two C halves),
so the existing checkers (which ignore inner loops) see them exactly.
"""
import math, os, json
import build4
from build4 import RB, SERVO, CYL, DSC, PROF, HA, SL, XC, XM, DISC, CAM, RUBBER, STEEL
from shapes import rrect, annulus_sector, circle_loop, orient
import build

_V4_TORSO, _V4_LEG = build4.torso, build4.right_leg

SKY = (0.00, 0.68, 0.94)
NAVY = (0.05, 0.19, 0.38)
WHITE = (0.97, 0.97, 0.97)
ORANGE = (0.91, 0.29, 0.10)
DARK = (0.20, 0.24, 0.32)        # structural prints that stay visible between shells
PRINT = build.PRINT

PAN_Z = 347.0                    # pan servo horn face (servo sunk into the torso top)
ZT = 420.0                       # head tilt axis
ZH = ZT + 20.0                   # head shell centre
HW, HD, HH = 100.0, 75.0, 75.0   # head half width (y), half depth (x), half height (z)
HP, HN, HT = 2.6, 20, 1.2        # superellipse exponent, number of slices, wall
VIEW = 20.0                      # camera look-down angle


# ---------------------------------------------------------------- profile helpers
def ring_halves(axis, a0, a1, cu, cv, A, B, R, Ai, Bi, Ri, rgb, gap=0.25, back=True, Rb=None, Rib=None):
    """Rounded-rectangle ring (outer A,B,R / inner Ai,Bi,Ri half sizes about (cu,cv)) as two C halves split
    across u. back=False keeps only the +u half (a C open toward -u)."""
    def half(s):
        u0 = cu + s * gap
        r, ri = (R, Ri) if s > 0 else (Rb if Rb is not None else R, Rib if Rib is not None else Ri)
        pts = [(u0, cv - B, 0), (cu + s * A, cv - B, r), (cu + s * A, cv + B, r), (u0, cv + B, 0),
               (u0, cv + Bi, 0), (cu + s * Ai, cv + Bi, ri), (cu + s * Ai, cv - Bi, ri), (u0, cv - Bi, 0)]
        return PROF(axis, a0, a1, orient(pts, True), rgb)
    return [half(1), half(-1)] if back else [half(1)]


def c_open_back(axis, a0, a1, cu, cv, A, B, R, Ai, Bi, Ri, u_end, rgb):
    """C shell: +u wall and both side walls, open on the -u side at u = u_end."""
    pts = [(u_end, cv - B, 0), (cu + A, cv - B, R), (cu + A, cv + B, R), (u_end, cv + B, 0),
           (u_end, cv + Bi, 0), (cu + Ai, cv + Bi, Ri), (cu + Ai, cv - Bi, Ri), (u_end, cv - Bi, 0)]
    return PROF(axis, a0, a1, orient(pts, True), rgb)


def tube(axis, a0, a1, cu, cv, r_out, r_in, rgb):
    return [PROF(axis, a0, a1, annulus_sector(cu, cv, r_in, r_out, 0.5, 179.5), rgb),
            PROF(axis, a0, a1, annulus_sector(cu, cv, r_in, r_out, 180.5, 359.5), rgb)]


def se(z):
    """Superellipse scale of the head at height z from its centre."""
    q = min(abs(z) / HH, 1.0)
    return (1 - q ** HP) ** (1 / HP)


# ---------------------------------------------------------------- torso
def torso():
    P, F = _V4_TORSO()
    P["head_pan_servo_XC430"] = [SERVO(HA - SL, HA, -14.25, 14.25, PAN_Z - 34, PAN_Z, "z", XC)]
    M4 = build.BOLT["M4"]["hole"]
    hz4 = [((x, y, 0), M4) for x in (-55, 40) for y in (-50, 50)]
    P["top_plate"] = [RB(-65, 50, -60, 60, 330, 335, WHITE, "z", 20, hz4 + [((-12, 0, 0), 37.0)])]
    P["joint_discs_torso"] = [DSC((0, 0, PAN_Z), "z", 10)]
    # shirt (white) = the v4 covers recoloured; shorts (navy) band below them, same outline
    P["torso_cover_front"] = [(k, a, WHITE) for k, a, _ in P["torso_cover_front"]]
    P["torso_cover_back"] = [(k, a, WHITE) for k, a, _ in P["torso_cover_back"]]
    t, r, Y = 1.2, build4.COV_R, build4.COV_Y
    xf, xb = build4.COV_XF, build4.COV_XB
    cu, A = (xf + xb) / 2, (xf - xb) / 2
    P["shorts_band"] = ring_halves("z", 205, 225, cu, 0, A, Y, r, A - t, Y - t, r - t, NAVY, gap=0.5)
    # decorative fixed arms: sleeve tube, shoulder dome, hand ball
    for s, nm in ((1, "L"), (-1, "R")):
        y = s * 91.0
        P[f"arm_{nm}_sleeve"] = tube("z", 262, 325, 0, y, 16, 14.8, SKY)
        P[f"arm_{nm}_shoulder"] = [("scap", ((0, y, 325), (0, 0, 1), 16, 14.8, 0.0), SKY)]
        P[f"arm_{nm}_hand"] = [("scap", ((0, y, 250), (0, 0, 1), 19, 17.8, -15.0), WHITE)]
    return P, F


# ---------------------------------------------------------------- neck (pan link)
def neck():
    return {
        "neck_column": [RB(-9, 7, -9, 9, PAN_Z + 2, ZT - 24, NAVY, "z", 4),
                        RB(-30, 8, -16, 16, ZT - 24, ZT - 14.25, NAVY, "z", 6),
                        DSC((0, 0, PAN_Z + 1), "z", 8)],                        # child disc of head_pan
        "head_tilt_servo_XC430": [SERVO(HA - SL, HA, -17, 17, ZT - 14.25, ZT + 14.25, "y", XC)],
        "joint_disc_neck": [DSC((0, 17, ZT), "y", 10)],                         # parent disc of head_tilt
    }


# ---------------------------------------------------------------- head (tilt link)
def head():
    P = {}
    # rounded 'TV' shell: horizontal slices of a superellipsoid, each a ring; ledges close the steps
    dz = 2 * HH / HN
    edges = [-HH + i * dz for i in range(HN + 1)]
    sc = [se(min(abs(edges[i]), abs(edges[i + 1]))) if edges[i] * edges[i + 1] > 0 else 1.0 for i in range(HN)]
    shell = []
    prof = lambda sc_, off=0.0: (HD * sc_ - off, HW * sc_ - off, max(0.55 * min(HD * sc_, HW * sc_) - off, 2.0))
    for i in range(HN):
        A, B, R = prof(sc[i])
        Ai, Bi, Ri = prof(sc[i], HT)
        shell += ring_halves("z", ZH + edges[i], ZH + edges[i + 1], 0, 0, A, B, R, Ai, Bi, Ri, SKY)
    for i in range(HN - 1):                       # thin ledge where two slices differ in size
        if abs(sc[i] - sc[i + 1]) < 1e-9:
            continue
        big, small = max(sc[i], sc[i + 1]), min(sc[i], sc[i + 1])
        z = ZH + edges[i + 1]
        z0, z1 = (z, z + HT) if sc[i + 1] > sc[i] else (z - HT, z)   # on the bigger slice's side
        A, B, R = prof(big, HT / 2)
        Ai, Bi, Ri = prof(small, HT)
        shell += ring_halves("z", z0, z1, 0, 0, A, B, R, Ai, Bi, Ri, SKY)
    s = sc[-1]
    shell.append(PROF("z", ZH + HH - HT, ZH + HH, rrect(-HD * s, HD * s, -HW * s, HW * s, 0.55 * HD * s), SKY))
    P["head_shell"] = shell

    # white face plate (skin + rim that reaches back into the shell), axis x: u = y, v = z
    fz, fa, fb, fr = ZH - 5, 78.0, 50.0, 45.0
    P["face_plate"] = [PROF("x", 75.4, 77.0, rrect(-fa, fa, fz - fb, fz + fb, fr), WHITE)] + \
        ring_halves("x", 58.0, 75.4, 0, fz, fa, fb, fr, fa - HT, fb - HT, fr - HT, WHITE)
    P["face_eyes"] = [PROF("x", 77.0, 78.2, rrect(y - 9, y + 9, ZH + 8 - 18, ZH + 8 + 18, 9), NAVY) for y in (-34, 34)]
    P["face_mouth"] = [PROF("x", 77.0, 78.2, annulus_sector(0, ZH - 12, 13, 16.5, 205, 335), NAVY)]

    # ear pods (navy rings + lid), sky inner disc, orange antennas
    ez = ZH - 5
    for s, nm in ((1, "L"), (-1, "R")):
        y0, y1 = (HW - 3, HW + 15) if s > 0 else (-HW - 15, -HW + 3)
        lid = (HW + 13.4, HW + 15) if s > 0 else (-HW - 15, -HW - 13.4)
        P[f"ear_{nm}"] = tube("y", y0, y1, ez, 0, 30, 28.4, NAVY) + \
            [PROF("y", lid[0], lid[1], circle_loop(ez, 0, 30), NAVY),
             CYL((0, s * (HW + 15), ez), "y" if s > 0 else "-y", 19, 1.5, SKY)]
        P[f"antenna_{nm}"] = [CYL((0, s * (HW + 6), ez + 22), "z", 4, 85, ORANGE),
                              CYL((0, s * (HW + 6), ez + 22), "z", 6.5, 6, NAVY)]

    # crest on top = camera housing (hollow, axis x: u = y, v = z)
    cx0, cx1, cy, cz0, cz1 = 8.0, 52.0, 34.0, ZH + HH - 4, ZH + HH + 22
    cvz, chz = (cz0 + cz1) / 2, (cz1 - cz0) / 2
    P["crest"] = ring_halves("x", cx0 + HT, cx1 - HT, 0, cvz, cy, chz, 10, cy - HT, chz - HT, 10 - HT, SKY) + \
        [PROF("x", cx1 - HT, cx1, rrect(-cy, cy, cvz - chz, cvz + chz, 10), SKY,
              holes=[(0, cvz + 1, 7.5)]),
         PROF("x", cx0, cx0 + HT, rrect(-cy, cy, cvz - chz, cvz + chz, 10), SKY)]
    th = math.radians(VIEW)
    dv = (math.cos(th), 0.0, -math.sin(th))
    P["camera_small"] = [RB(34, 44, -12, 12, cvz - 9, cvz + 11, CAM, "x", 2)]
    P["camera_lens"] = [("cyl", ((cx1 - 6, 0.0, cvz + 1), dv, 6.0, 9.0), CAM)]

    # inner frame: two plates from the tilt horn / idler up to the lid
    P["head_frame"] = [RB(-15, 15, 19, 22, ZT - 15, ZH + HH - HT, DARK, "z", 3),
                       RB(-15, 15, -22, -19, ZT - 15, ZH + HH - HT, DARK, "z", 3),
                       DSC((0, 18, ZT), "y", 8)]                                # child disc of head_tilt
    return P


# ---------------------------------------------------------------- leg shells (right leg; left is mirrored)
ANK_D = 8.0          # ankle roll servo + brackets moved back by this much (clears the ankle pitch servo)


def right_leg():
    L, F, J = _V4_LEG()
    ys = -40.0
    M3 = build.BOLT["M3"]["hole"]
    # knee: shank side plates end in a semicircle about the knee axis; thigh servo walls narrowed to the
    # servo width below z 140 so the shank side plates pass beside them
    yplate = lambda y0, y1: PROF("y", y0, y1, [(88, -12, 0), (122, -12, 12), (122, 12, 12), (88, 12, 0)], PRINT)
    L["shank"]["shank_frame"] = [yplate(ys - 22, ys - 18), yplate(ys + 18, ys + 22)] + L["shank"]["shank_frame"][2:]
    sf = L["shank"]["shank_frame"]                   # servo clamp walls start at z 30 (sole clearance in roll)
    L["shank"]["shank_frame"] = sf[:4] + [RB(14.5, 18.5, ys - 22, ys + 22, 30, 76, PRINT, "x", 4),
                                          RB(-18.5, -14.5, ys - 22, ys + 22, 30, 76, PRINT, "x", 4)] + sf[6:]
    tf = L["thigh"]["thigh_frame"]
    L["thigh"]["thigh_frame"] = tf[:2] + [
        RB(14.5, 18.5, ys - 17, ys + 17, 98, 140, PRINT, "x", 4), RB(-18.5, -14.5, ys - 17, ys + 17, 98, 140, PRINT, "x", 4),
        RB(14.5, 18.5, ys - 22, ys + 22, 140, 146, PRINT, "x", 2), RB(-18.5, -14.5, ys - 22, ys + 22, 140, 146, PRINT, "x", 2),
    ] + tf[4:]
    # ankle: roll servo, its bracket and the foot's horn bracket moved back by ANK_D; heel lengthened to match
    d = ANK_D
    L["ankle_link"] = {
        "ankle_roll_servo_XM430": [SERVO(-60 - d, -26 - d, ys - 14.25, ys + 14.25, 35 - HA, 35 - HA + SL, "x", XM)],
        "ankle_bracket": [
            RB(-60 - d, 12, ys - 27, ys - 23, 27, 50, PRINT, "y", 10),       # raised off the sole for roll
            RB(-60 - d, 12, ys + 23, ys + 27, 27, 50, PRINT, "y", 10),
            RB(-64 - d, -60 - d, ys - 27, ys + 27, 27, 50, PRINT, "x", 6),
            DSC((0, ys - 23, 35), "y", 8), DSC((-26 - d, ys, 35), "x", 10),
        ],
    }
    ft = [((-40 - d, ys - 16, 0), M3), ((-40 - d, ys + 16, 0), M3)]       # outside the roll servo's sweep
    heel = -60 - d - 6                               # sole runs under the shoe heel
    L["foot"] = {
        "foot_sole": [RB(heel, 60, ys - 35, ys + 35, 6, 12, PRINT, "z", 18, ft)] +
                     [CYL((x, y, 0), "z", 6, 6, RUBBER) for x in (heel + 12, 48) for y in (ys - 23, ys + 23)],
        "foot_bracket": [RB(-25 - d, -22.5 - d, ys - 14, ys + 14, 12, 47, PRINT, "x", 4),
                         RB(-50 - d, -25 - d, ys - 21, ys + 21, 12, 18, PRINT, "z", 4, ft),
                         DSC((-22.5 - d, ys, 35), "x", 8)],
    }
    F = [f for f in F if f[0] != "foot"] + [("foot", "M3", (-40 - d, y, 18), "-z", 16, 12) for y in (ys - 16, ys + 16)]
    # white thigh shell (open at the back where the knee folds)
    L["thigh"]["thigh_shell"] = [c_open_back("z", 118, 150, 0, ys - 0.5, 23.5, 27, 12, 22, 25.5, 10.5, -20, WHITE)]
    # sky shank shell, open at the back, navy band on top
    L["shank"]["shank_shell"] = [c_open_back("z", 44, 98, 0, ys, 23.5, 30.5, 12, 22, 29, 10.5, -20, SKY)]
    L["shank"]["shank_band"] = [c_open_back("z", 98, 104, 0, ys, 24.5, 31.5, 13, 22, 29, 10.5, -20, NAVY)]
    # navy shoe: low wall around the sole; square-ish heel clears the ankle bracket's back plate
    hx = heel                                         # shoe heel, behind the ankle bracket back plate
    cu, A = (60 + hx) / 2, (60 - hx) / 2
    L["foot"]["shoe"] = ring_halves("z", 12, 28, cu, ys, A, 35, 30, A - HT, 35 - HT, 30 - HT, NAVY, Rb=3, Rib=1.4)
    return L, F, J


# ---------------------------------------------------------------- assembly
def build_model():
    saved = (build4.torso, build4.neck, build4.head, build4.right_leg, build4.ZT)
    build4.torso, build4.neck, build4.head, build4.right_leg, build4.ZT = torso, neck, head, right_leg, ZT
    try:
        links, joints, fasteners = build4.build_model()
    finally:
        build4.torso, build4.neck, build4.head, build4.right_leg, build4.ZT = saved
    links["neck"]["origin"] = (0, 0, PAN_Z)
    links["head"]["origin"] = (0, 0, ZT)
    for j in joints:
        if j["name"] == "head_pan":
            j["point_mm"] = (0, 0, PAN_Z)
        if j["name"] == "head_tilt":
            j["point_mm"] = (0, 0, ZT)
    LIM = {"hip_pitch": (-80, 30), "knee": (0, 100), "ankle_pitch": (-50, 22), "ankle_roll": (-20, 20)}
    for j in joints:                            # collision-free ranges found by sweep5 / ankle_range
        key = j["name"].split("_", 1)[1] if j["name"][:2] in ("R_", "L_") else None
        if key in LIM:
            j["lower_deg"], j["upper_deg"] = LIM[key]
    for ln in links.values():                   # structural prints that show between shells: dark
        for pn, ss in ln["parts"].items():
            ln["parts"][pn] = [(k, a, DARK if rgb == PRINT else rgb) for k, a, rgb in ss]
    return links, joints, fasteners


write = build4.write
sampled, aabb, volume, to_local = build4.sampled, build4.aabb, build4.volume, build4.to_local

if __name__ == "__main__":
    links, joints, fasteners = build_model()
    out = "/home/claude/biped/out5"
    os.makedirs(out, exist_ok=True)
    w = write(links, f"{out}/biped_robot_v5.stp")
    json.dump({"links": {k: {"origin": v["origin"], "parts": list(v["parts"])} for k, v in links.items()},
               "joints": joints, "fasteners": fasteners}, open(f"{out}/model.json", "w"), indent=1)
    print("entities", w.n, "links", len(links), "parts", sum(len(v["parts"]) for v in links.values()))
