"""최적화 결과 보기: results/<tag>.json 에서 팔 · 시작 자세의 가장 좋은 키프레임을 다시 돌려 옆 모습 프레임 묶음(png) 또는 영상(mp4)을 만든다.
usage: ../.venv/bin/python gu_show.py <tag> <팔 이름> <prone|supine> [png|mp4] [출력 경로]"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID"); os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "2")
import numpy as np
import mujoco
import imageio.v3 as iio
from gu_model import ARMS, Sim, RES, side_cam
from gu_search import to_keys


def load(tag, arm, start):
    for r in json.load(open(os.path.join(RES, f"{tag}.json"))):
        if r["arm"] == arm and r["start"] == start:
            return r
    raise KeyError((arm, start))


if __name__ == "__main__":
    tag, arm, start = sys.argv[1:4]; kind = sys.argv[4] if len(sys.argv) > 4 else "png"
    out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(RES, f"show_{tag}_{start}.{kind}")
    r = load(tag, arm, start)
    from gu_search import replay
    if kind == "png":
        S0 = Sim(ARMS[arm]); ren = mujoco.Renderer(S0.m, 300, 400)
        S, res, H, fr = replay(r, render=(ren, side_cam(0.8, -5), 2))
        rows = [np.concatenate(fr[i:i + 5] + [np.zeros_like(fr[0])] * (5 - len(fr[i:i + 5])), 1) for i in range(0, len(fr), 5)]
        iio.imwrite(out, np.concatenate(rows, 0))
    else:
        S0 = Sim(ARMS[arm]); ren = mujoco.Renderer(S0.m, 480, 640)
        S, res, H, fr = replay(r, render=(ren, side_cam(0.85, -8, 60), 30))
        iio.imwrite(out, np.stack(fr), fps=30, codec="libx264", quality=6, macro_block_size=8)
    print(out, {k: res[k] for k in ("ok", "z_end", "tilt_end", "score", "touch_end")})
