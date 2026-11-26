"""Export every printed part as its own STEP, rotated to its print orientation and placed on the bed
(lowest point z = 0, centred on x = y = 0), plus a CSV/markdown parts table."""
import os, sys, csv, json, importlib, numpy as np
import build, build4, printcheck
from build import IDX, UVI

mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "build6")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/biped/out6/print_parts"
os.makedirs(OUT, exist_ok=True)

# rotation taking the chosen 'up' direction to +z (proper rotations, det = +1)
ROT = {"+z": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "-z": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
       "+x": [[0, 1, 0], [0, 0, 1], [1, 0, 0]], "-x": [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
       "+y": [[0, 0, 1], [1, 0, 0], [0, 1, 0]], "-y": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]}
FORCE = {"shank_shell": "+z", "thigh_shell": "+z", "torso_cover": "+z", "antenna": "+z", "face_plate": "+x",
         "torso_side": "+z", "foot_sole": "+z", "hip_yaw_bracket": "+z"}
COLOR = {}
NOTE = {
    "torso_cover": "z 20 mm에서 필라멘트 교체: 남색(반바지) → 흰색(셔츠)",
    "shank_shell": "높이 54 mm에서 필라멘트 교체: 하늘색 → 남색(띠)",
    "head_shell": "뒷면을 바닥에. 브림 권장",
    "crest": "윗면을 바닥에",
    "head_frame": "윗판을 바닥에",
    "antenna": "세워서 출력, 브림 5 mm",
    "sleeve": "평평한 안쪽 면을 바닥에",
    "hand": "평평한 면을 바닥에, 소매 아래에 접착",
    "ear_L": "덮개를 바닥에", "ear_R": "덮개를 바닥에",
}


def rot_solid(s, R, t=(0, 0, 0)):
    R = np.array(R, float); t = np.array(t, float)
    kind, args, rgb = s
    if kind == "loft":
        import loft; return ("loft", loft.rotate(args, R, t), rgb)
    if kind in ("cyl", "scap"):
        c = R @ np.array(args[0], float) + t
        a = R @ np.array(args[1], float)
        return (kind, (tuple(c), tuple(a)) + tuple(args[2:]), rgb)
    axis, a0, a1, outer, inners, holes = args
    k = IDX[axis]
    kn = int(np.argmax(np.abs(R[:, k])))
    sgn = R[kn, k]
    naxis = "xyz"[kn]
    iu, iv = UVI[axis]
    nu, nv = UVI[naxis]

    def mp(u, v):
        p = np.zeros(3); p[iu], p[iv] = u, v
        q = R @ p + t
        return q[nu], q[nv]
    loop = lambda l: [(*mp(u, v), r) for u, v, r in l]
    b0, b1 = sorted((sgn * a0 + t[kn], sgn * a1 + t[kn]))
    return ("rprism", (naxis, b0, b1, loop(outer), [loop(l) for l in inners],
                       [(*mp(u, v), r) for u, v, r in holes]), rgb)


def main():
    global FORCE, NOTE
    FORCE = getattr(mod, "FORCE", FORCE)
    NOTE = getattr(mod, "NOTE", NOTE)
    links, _, _ = mod.build_model()
    rows = {(r[1]): r for r in printcheck.analyse(mod)}
    table = []
    for ln, pn, sol in printcheck.printed_parts(links):
        up = next((d for key, d in FORCE.items() if key in pn), rows[pn][3] if pn in rows else "+z")
        R = ROT[up]
        rs = [rot_solid(s, R) for s in sol]
        bbs = [build4.aabb(build4.sampled(s)) for s in rs]
        lo = [min(b[i][0] for b in bbs) for i in range(3)]
        hi = [max(b[i][1] for b in bbs) for i in range(3)]
        shift = (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])
        rs = [rot_solid(s, np.eye(3), shift) for s in rs]
        # support check in the chosen orientation
        V, _ = printcheck.voxelize(rs, 0.6 if max(h - l for l, h in zip(lo, hi)) < 90 else 1.0)
        from scipy import ndimage
        A = V[ndimage.find_objects(V.astype(int))[0]]
        need, worst = printcheck.overhang(A, 0.6 if max(h - l for l, h in zip(lo, hi)) < 90 else 1.0)
        name = pn
        w = build4.write({name: {"origin": (0, 0, 0), "parts": {name: rs}}}, f"{OUT}/{name}.stp")
        vol = sum(build4.volume(s) for s in sol) / 1000
        colours = sorted({tuple(round(c, 2) for c in s[2]) for s in sol})
        table.append(dict(part=name, link=ln, up=up, size_mm="x".join(str(round(h - l)) for l, h in zip(lo, hi)),
                          support_mm2=round(need), volume_cm3=round(vol, 1),
                          note=next((v for k, v in NOTE.items() if k in pn), ""), colours=colours))
    with open(f"{OUT}/parts.json", "w") as f:
        json.dump(table, f, indent=1, ensure_ascii=False)
    for r in table:
        print(f"{r['part']:24s} {r['up']:>3s} {r['size_mm']:>12s} sup {r['support_mm2']:5d}  {r['volume_cm3']:6.1f} cm3  {r['note']}")


if __name__ == "__main__":
    main()
