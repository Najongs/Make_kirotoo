"""예행: (1) 흐트러짐 0 에서 rb_common.run_rb 가 kw_sim.run 과 같은 결과인지, (2) 백래시 · 전압 · 마찰 한 개씩 돌아가는지.
python robust/smoke.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from rb_common import ROB0, STRIDE, run_rb, brief, one


def ref(_):
    from kw_sim import run
    o, R, m = run(**dict(ROB0, stride=STRIDE))
    return o


if __name__ == "__main__":
    jobs = [("rb 기준 걷기", dict(stride=STRIDE)), ("rb 서 있기", dict(t_walk=99, T_end=8.0)),
            ("rb 백래시 0.5°", dict(stride=STRIDE, backlash_deg=0.5)), ("rb 10.5 V", dict(stride=STRIDE, volt=10.5)),
            ("rb 내부마찰 5%", dict(stride=STRIDE, frictionloss_frac=0.05)), ("rb 질량 +15% 무게중심 +10mm 배낭", dict(stride=STRIDE, mass_f=1.15, com_dx_m=0.01, backpack=True))]
    with Pool(len(jobs) + 1) as p:
        r0 = p.apply_async(ref, (0,))
        out = p.map(one, jobs)
        o0 = r0.get()
    print("kw_sim.run 기준:", o0["fallen"], o0["steps_done"], o0["cop_margin_mm"], max(o0["u_max"].values()), o0["u_p99"]["R_knee"], o0["sat_frac"]["R_knee"])
    o1 = out[0][2]
    print("run_rb  기준:", o1["fallen"], o1["steps_done"], o1["cop_margin_mm"], max(o1["u_max"].values()), o1["u_p99"]["R_knee"], o1["sat_frac"]["R_knee"])
    for meta, c, o in out:
        print(f"{meta:30s} {brief(o)}", o.get("bl_absmax_deg"))
