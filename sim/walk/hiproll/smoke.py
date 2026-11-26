import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_tests import BASE, push_t
from hr_cfg import hr
from sweep import sweep
t0 = push_t(6, "ss")
cfgs = [dict(BASE, name="base s30", stride=0.03),
        dict(BASE, name="A-XM s30", stride=0.03, hip_roll=hr("A")),
        dict(BASE, name="B-XM s30", stride=0.03, hip_roll=hr("B")),
        dict(BASE, name="B-XC s30", stride=0.03, hip_roll=hr("B", "XC430-W240")),
        dict(BASE, name="B-XM s50", stride=0.05, hip_roll=hr("B")),
        dict(BASE, name="A-XM s50", stride=0.05, hip_roll=hr("A")),
        ]
for k in (0.0, 1.0, 2.0):
    for nm, h in (("base", False), ("B-XM", hr("B")), ("A-XM", hr("A"))):
        for F in (-4, -6, -8):
            cfgs.append(dict(BASE, name=f"{nm} klat{k} push{F}", hip_roll=h, k_lat=k, push=(t0, 0.2, F, "y"), T_end=t0 + 4.0))
for n, b, o in sweep(cfgs):
    print(f"{n:28s} {b}")
