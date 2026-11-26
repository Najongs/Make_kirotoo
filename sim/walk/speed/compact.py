"""큰 결과 json 을 커밋용으로 줄인다(그 자리에 덮어씀): 요약 · 그림에 쓰는 값만 남김. usage: ../../.venv/bin/python compact.py results/*.json"""
import json, sys
KEEP = ("fallen", "fall_t", "steps_done", "speed_mps", "cop_margin_mm", "split", "mass_kg", "com_z0", "torso_roll_deg", "torso_pitch_deg", "error")
CFG = ("stride", "Ts", "Td", "friction", "crouch", "seg", "push", "t_walk", "T_end")
for p in sys.argv[1:]:
    D = json.load(open(p))
    if not isinstance(D, dict) or "rows" not in D:
        continue
    for r in D["rows"]:
        r["res"] = {k: v for k, v in r["res"].items() if k in KEEP}
        hr = r["cfg"].get("hip_roll")
        r["cfg"] = dict({k: r["cfg"][k] for k in CFG if k in r["cfg"]}, roll_pitch_lengthen=hr.get("lengthen") if isinstance(hr, dict) else None)
        r["res"].get("split", {}).pop("hip_yaw", None)
        for j in r["res"].get("split", {}).values():
            for k in list(j):
                j[k] = round(j[k], 3) if isinstance(j[k], float) else j[k]
    json.dump(D, open(p, "w"), ensure_ascii=False, separators=(",", ":"))
    print(p)
