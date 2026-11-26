"""Collision-free ranges of the leg joints (1-2 deg steps), using the sweep5 machinery."""
import numpy as np, sys
sys.argv = ["x"]
src = open("/home/claude/biped/sweep5.py").read().split("bad = []")[0]
exec(src)
S = "R"
n = [f"{S}_hip_yaw", f"{S}_hip_pitch", f"{S}_knee", f"{S}_ankle_pitch", f"{S}_ankle_roll"]
ch = ["torso", f"{S}_hip_yaw_link", f"{S}_thigh", f"{S}_shank", f"{S}_ankle_link", f"{S}_foot"]
def free(test, lo, hi, step):
    ok = [a for a in np.arange(lo, hi + 1e-9, step) if not test(a)]
    return ok
def first_bad(test, start, stop, step):
    a = start
    while (a <= stop if step > 0 else a >= stop):
        if test(a): return a
        a += step
    return None
# knee: shank vs thigh
t = lambda k: bool(hits(ch[3], R(n[2], k), ch[2]))
print("knee first hit going up:", first_bad(t, 0, 130, 2))
# ankle pitch at knee 0: ankle link + foot vs shank, thigh
def tp(a, k=0, r=0):
    Mk = R(n[2], k); Ma = Mk @ R(n[3], a); Mf = Ma @ R(n[4], r)
    return bool(hits(ch[4], Ma, ch[3], Mk) or hits(ch[5], Mf, ch[3], Mk) or hits(ch[4], Ma, ch[2]) or hits(ch[5], Mf, ch[2]))
for k in (0, 60, 100, 130):
    print("knee", k, "pitch + first hit:", first_bad(lambda a: tp(a, k), 0, 45, 1), " - first hit:", first_bad(lambda a: tp(a, k), 0, -60, -1))
def tr(r, a=0):
    Ma = R(n[3], a); Mf = Ma @ R(n[4], r)
    return bool(hits(ch[5], Mf, ch[4], Ma) or hits(ch[5], Mf, ch[3]))
for a in (-30, 0, 20):
    print("pitch", a, "roll + first hit:", first_bad(lambda r: tr(r, a), 0, 30, 1), " - first hit:", first_bad(lambda r: tr(r, a), 0, -30, -1))
