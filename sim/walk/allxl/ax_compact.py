"""큰 결과 json 을 커밋용으로 줄인다(그 자리에 덮어씀): ax_summary · ax_fig 가 쓰는 값만. usage: ../../.venv/bin/python ax_compact.py results/{base,fix,combo,combo2}.json"""
import json, sys
KEEP = ("fallen", "fall_t", "steps_done", "speed_mps", "cop_margin_mm", "mass_kg", "joints", "error", "torso_roll_deg")
CFG = ("stride", "Ts", "Td", "friction", "crouch", "seg", "t_walk", "T_end", "servo_map", "mass_scale", "extra_mass")


def rnd(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rnd(v) for v in x]
    return x


for p in sys.argv[1:]:
    D = json.load(open(p))
    for r in D["rows"]:
        r["res"] = rnd({k: (v[-300:] if k == "error" else v) for k, v in r["res"].items() if k in KEEP})
        hr = r["cfg"].get("hip_roll")
        r["cfg"] = dict({k: r["cfg"][k] for k in CFG if k in r["cfg"]}, hip_roll_servo=hr.get("servo") if isinstance(hr, dict) else None)
    json.dump(D, open(p, "w"), ensure_ascii=False, separators=(",", ":"))
    print(p)
