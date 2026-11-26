import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_tests import BASE, push_t
from hr_cfg import hr
from sweep import sweep
t0 = push_t(6, "ss")
cfgs = []
for nm, h in (("base", False), ("B-XM", hr("B"))):
    for kl, db, li in ((1.0, 0.01, False), (2.0, 0.01, False), (1.0, 0.02, False), (2.0, 0.02, False)):
        tag = f"{nm} k{kl} db{db}"
        cfgs.append(dict(BASE, name=tag + " walk76", hip_roll=h, k_lat=kl, lat_db=db, lat_in=li, stride=0.05))
        for F in (-2, -4, -6, -8, 4, 8, 10):
            cfgs.append(dict(BASE, name=f"{tag} push{F}", hip_roll=h, k_lat=kl, lat_db=db, lat_in=li, push=(t0, 0.2, F, "y"), T_end=t0 + 4.0))
for n, b, o in sweep(cfgs, procs=40):
    print(f"{n:30s} {b[:110]}")
