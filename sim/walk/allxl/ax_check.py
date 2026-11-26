"""점검 묶음(속도 격자 없이 서기 · 0.05 · 0.077 m/s만):
 (1) 감속이 나빴던 것이 위치 이득(kp × 감속비² × 효율로 커짐) 탓인지 — 출력 kp 를 XM 과 같은 17.5 로 낮춰 다시.
 (2) 추천 후보 D6 의 회전자 관성 민감도 — XL 은 코어 모터라 관성이 더 클 수 있음: 모든 XL 관절 armature × 2(제어기 모델도 같이).
usage: cd sim/walk/allxl && ../../.venv/bin/python ax_check.py → results/check.json"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ax_common as A
import kw_common as C
from kw_common import jdump
from ax_combo import LIGHT
C.SERVO_SIM["XL430e_g2_kp"] = C.SERVO_SIM["XL430e_g2"][:2] + (17.5,) + C.SERVO_SIM["XL430e_g2"][3:]
C.SERVO_SIM["2xXL430e_g2_kp"] = C.SERVO_SIM["2xXL430e_g2"][:2] + (17.5,) + C.SERVO_SIM["2xXL430e_g2"][3:]
for k in ("XL430e_g2_kp", "2xXL430e_g2_kp"):
    C.SERVO_G[k] = C.SERVO_G[k[:-3]]; A.HEAT[k] = A.HEAT[k[:-3]]
for k in ("XL430e", "2xXL430e"):          # 회전자 관성 × 2 (build_model 의 armature 배수는 servo_map 이 덮어써서 서보 표로 넣음)
    v = C.SERVO_SIM[k]; C.SERVO_SIM[k + "_a2"] = v[:3] + (2 * v[3],); C.SERVO_G[k + "_a2"] = C.SERVO_G[k]; A.HEAT[k + "_a2"] = A.HEAT[k]
K2 = "2xXL430e"
CHECK = {"(c) 무릎 2:1, 출력 kp 17.5": A.cfg(dict(knee="XL430e_g2_kp")),
         "무릎 XL 둘 + 2:1 (출력 kp 17.5), 롤 XL 둘": A.cfg(dict(knee="2xXL430e_g2_kp", hip_roll=K2)),
         "D6 회전자 관성 × 2": A.cfg(dict({k: "XL430e_a2" for k in A.ALLXL}, knee=K2 + "_a2", hip_roll=K2 + "_a2"), crouch=0.035, mass_scale=LIGHT),
         "D6 + 배낭 250 g": A.cfg(dict(knee=K2, hip_roll=K2), crouch=0.035, mass_scale=LIGHT, pack_=True),
         "D6 를 12 V 로": A.cfg(dict({k: "XL430e12" for k in A.ALLXL}, knee="2xXL430e12", hip_roll="2xXL430e12"), crouch=0.035, mass_scale=LIGHT)}
if __name__ == "__main__":
    t = time.time()
    jobs = [j for n, c in CHECK.items() for j in A.jobs_for(n, c, speed_grid=False)]
    out = A.run_jobs(jobs)
    rows = [dict(meta=m_, cfg=c, res=r) for m_, c, r in out]
    jdump(dict(note=__doc__, rows=rows), os.path.join(A.RES, "check.json"))
    print(f"check: {len(rows)} runs, {sum(1 for r in rows if r['res'].get('fallen'))} fell, {sum(1 for r in rows if 'error' in r['res'])} errors, {time.time() - t:.0f}s")
