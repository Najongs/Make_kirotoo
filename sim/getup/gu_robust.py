"""찾은 키프레임이 조건이 조금 바뀌어도 되는지(시뮬 실측): 바닥 마찰 · 동작 빠르기 · 위치 이득 · 팔 질량을 바꿔 다시 돌림.
궤적이 매우 예민하므로(관절 범위 0.0003 rad 차이에도 갈라짐), 한 번 성공은 실기 성공을 뜻하지 않는다.
usage: ../.venv/bin/python gu_robust.py <tag> <팔 이름> <start> → results/robust_<tag>.json"""
import sys, os, math, json, copy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import mujoco
import gu_search as G
from gu_model import ARMS, Sim, RES, jdump


def variants():
    V = [("기준", {})]
    V += [(f"마찰 {f}", dict(friction=f)) for f in (0.5, 0.65, 1.0, 1.2)]
    V += [(f"동작 시간 ×{s}", dict(tscale=s)) for s in (0.9, 0.95, 1.05, 1.1)]
    V += [(f"위치 이득 ×{k}", dict(kp=k)) for k in (0.75, 1.25)]
    V += [(f"팔 질량 ×{a}", dict(armmass=a)) for a in (0.8, 1.2)]
    V += [(f"처음 가라앉는 시간 +{h} s", dict(settle=h)) for h in (0.05, 0.2)]
    return V


def run_v(r, v):
    G.replay(r)                                   # NK · 범위 맞춤
    hm = math.radians(r["hip_min_deg"]) if "hip_min_deg" in r else -1.396
    km = math.radians(r["knee_max_deg"]) if "knee_max_deg" in r else 1.745
    S = Sim(ARMS[r["arm"]], knee_max=max(km, 1.745), hip_min=min(hm, -1.396)); m = S.m
    if "friction" in v:
        m.geom_friction[:, 0] = v["friction"]
    if "kp" in v:
        m.actuator_gainprm[:, 0] *= v["kp"]; m.actuator_biasprm[:, 1] *= v["kp"]
    if "armmass" in v:
        for b in range(m.nbody):
            if m.body(b).name.endswith(("_arm", "_forearm")):
                m.body_mass[b] *= v["armmass"]; m.body_inertia[b] *= v["armmass"]
    keys = G.to_keys(r["best_x"])
    if "tscale" in v:
        keys = [(d * v["tscale"], a) for d, a in keys]
    st = "prone" if r["start"] in ("pike", "tuck", "headpike") else ("supine" if r["start"] == "supine6" else r["start"])
    res, H, _ = S.run(keys, start=st, settle=0.8 + v.get("settle", 0.0))
    return res


if __name__ == "__main__":
    tag, arm, start = sys.argv[1:4]
    r = next(x for x in json.load(open(os.path.join(RES, f"{tag}.json"))) if x["arm"] == arm and x["start"] == start)
    from multiprocessing import Pool
    V = variants()
    with Pool(len(V)) as P:
        R = P.starmap(run_v, [(r, v) for _, v in V])
    out = [dict(name=n, ok=x["ok"], z_end=x["z_end"], tilt_end=x["tilt_end"]) for (n, _), x in zip(V, R)]
    path = os.path.join(RES, f"robust_{tag}.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old[f"{arm}/{start}"] = out; jdump(old, path)
    for o in out:
        print(o["name"], o["ok"], round(o["z_end"] * 1e3), round(o["tilt_end"]))
    print("성공", sum(o["ok"] for o in out), "/", len(out))
