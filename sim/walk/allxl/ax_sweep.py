"""모든 관절 XL430 걷기 묶음. usage: cd sim/walk/allxl && ../../.venv/bin/python ax_sweep.py base [fix combo] → results/<묶음>.json
한 구성마다: 서기 12 s 1판, 0.05 m/s(보폭 32.5 mm) 3판, 속도 격자 4 걸음 빠르기 × 10 보폭 × 마찰 3 = 120판."""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ax_common as A
from kw_common import jdump

XL12 = {k: "XL430e12" for k in A.ALLXL}
K2, KG = "2xXL430e", "XL430e_g2"
GROUPS = {
    "base": {"지금 v20A (XM · XC)": A.cfg(A.V20A), "모두 XL 11.1 V": A.cfg(), "모두 XL 12 V": A.cfg(XL12)},
    "fix": {"(a) 허벅지 · 정강이 +10": A.cfg(seg=0.01), "(a) 허벅지 · 정강이 0": A.cfg(seg=0.0),
            "(b) 무릎 XL 두 개": A.cfg(dict(knee=K2)), "(c) 무릎 2:1": A.cfg(dict(knee=KG)),
            "(d) 웅크림 25": A.cfg(crouch=0.025), "(d) 몸무게 −10 %": A.cfg(mass_f=0.9), "(d) 몸무게 −20 %": A.cfg(mass_f=0.8),
            "(d) 배낭 250 g": A.cfg(pack_=True),
            "(e) 무릎 2:1 + 발목 피치 2:1": A.cfg(dict(knee=KG, ankle_pitch=KG)),
            "(e) 무릎 XL 두 개 + 발목 피치 XL 두 개": A.cfg(dict(knee=K2, ankle_pitch=K2))},
}


if __name__ == "__main__":
    for g in sys.argv[1:]:
        if g in ("combo", "combo2"):
            import ax_combo
            GROUPS["combo"], GROUPS["combo2"] = ax_combo.COMBO, ax_combo.COMBO2
        t = time.time()
        jobs = [j for n, c in GROUPS[g].items() for j in A.jobs_for(n, c)]
        out = A.run_jobs(jobs)
        rows = [dict(meta=m_, cfg=c, res=r) for m_, c, r in out]
        jdump(dict(note=A.__doc__, rows=rows), os.path.join(A.RES, f"{g}.json"))
        print(f"{g}: {len(rows)} runs, {sum(1 for r in rows if r['res'].get('fallen'))} fell, "
              f"{sum(1 for r in rows if 'error' in r['res'])} errors, {time.time() - t:.0f}s", flush=True)
