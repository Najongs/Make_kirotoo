"""여러 설정을 병렬로 돌려 한 줄 요약(개발용). python sweep.py '[{"name":..., ...}, ...]'"""
import sys, json, os
from multiprocessing import Pool


def one(cfg):
    from kw_sim import run, brief
    name = cfg.pop("name", "")
    try:
        o, R, m = run(**cfg)
        return name, brief(o), o
    except Exception as e:
        import traceback
        return name, "ERROR " + repr(e) + traceback.format_exc()[-400:], None


def sweep(cfgs, procs=None):
    with Pool(procs or min(len(cfgs), 24)) as p:
        return p.map(one, cfgs)


if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    for name, b, o in sweep(cfgs):
        print(f"{name:28s} {b}")
