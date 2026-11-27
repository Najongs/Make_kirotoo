"""키프레임 최적화(교차 엔트로피 방법): 팔 후보 × 시작 자세(엎드림 · 누움)마다 키프레임 4개 + 서기 1개의 관절 목표 · 시간을 찾는다.
- 관절 목표는 좌우 같음: 팔(어깨) · 팔꿈치(있을 때) · 고관절 피치 · 무릎 · 발목 피치. 나머지(요 · 롤 · 발목 롤 · 머리)는 0.
- 범위는 설계값 안: 무릎 0–100°, 고관절 −80 – +30°, 발목 −50 – +22°, 어깨 ±120°, 팔꿈치 ±130° [가정]. 서보 힘은 정지토크에서 잘림.
- 비용: gu_model.Sim.score(끝 1 s 기울기 · 높이 부족 − 중간 최고 높이 부분 점수). 성공 판정은 getup.py 와 같음.
usage: cd sim/getup && ../.venv/bin/python gu_search.py [iters pop arms...] → results/search.json"""
import sys, os, math, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from gu_model import ARMS, Sim, RES, jdump

LO = dict(arm=-2.094, elbow=-2.27, hip=-1.396, knee=0.0, ankle=-0.873)
HI = dict(arm=2.094, elbow=2.27, hip=0.524, knee=1.745, ankle=0.384)
JN = ["arm", "elbow", "hip", "knee", "ankle"]
NK = int(os.environ.get("NK", "4"))       # 키프레임 수(서기 키 빼고). 4 보다 크면 처음 평균은 앞 키를 늘려 채움
# 처음 평균(손으로 짠 동작). 엎드림: 팔로 가슴을 들며 다리 접기 → 팔을 머리 쪽으로 더 돌려 몸을 뒤로 → 쪼그림 → 섬.
# 누움: 팔을 뒤로 돌려 바닥을 밀어 앉기 → 다리 접기 → 팔로 엉덩이를 앞으로 밀기 → 섬.
INIT = {"prone": [(0.6, [-1.2, -0.5, -0.6, 0.8, 0.0]), (0.8, [-1.6, -0.5, -1.3, 1.7, -0.8]), (0.8, [-2.0, 0.0, -1.3, 1.2, -0.6]),
                  (0.6, [-2.09, 0.0, -1.0, 1.0, -0.4])],
        "supine": [(0.6, [1.6, 0.5, -0.8, 0.5, 0.0]), (0.8, [2.0, 0.5, -1.3, 1.7, -0.8]), (0.8, [2.09, 0.0, -1.396, 1.745, -0.87]),
                   (0.6, [1.0, 0.0, -1.0, 1.2, -0.4])]}
# 두 번째 묶음: 엎드림에서 「팔꿈치로 팔굽혀펴기 → 다리를 편 채 고관절을 접어 엉덩이를 올림(ㅅ자) → 팔로 몸을 뒤로 → 섬」
INIT["pike"] = [(0.6, [-0.3, -2.4, -0.3, 0.3, 0.0]), (0.8, [-1.4, -0.6, -1.0, 0.2, 0.2]), (0.8, [-1.9, 0.0, -1.396, 0.0, 0.38]),
                (0.6, [-2.09, 0.0, -1.2, 0.3, 0.0])]
# 세 번째 묶음(NK=6): 팔꿈치 140+140 첫 묶음에서 팔굽혀펴기 → 판자 자세까지 간 앞 3 키 + 다리를 몸 밑으로 당기기 · 팔로 뒤로 밀기 2 키 + 펴기 1 키
INIT["tuck"] = [(0.68, [-0.65, -0.49, 0.05, 1.29, 0.1]), (1.03, [-0.45, -0.13, -1.04, 1.51, -0.22]), (0.54, [-1.55, 1.0, -1.4, 0.65, -0.87]),
                (0.8, [-2.0, 0.5, -1.396, 1.745, -0.87]), (0.6, [-2.09, 0.0, -1.2, 1.6, -0.87]), (0.6, [-1.5, 0.0, -0.9, 1.2, -0.6])]
INIT["supine6"] = INIT["supine"][:2] + [(0.6, [2.09, 0.3, -1.396, 1.745, -0.87]), (0.6, [2.09, 0.0, -1.396, 1.745, -0.87]),
                                         (0.6, [1.5, 0.0, -1.2, 1.6, -0.87]), (0.6, [0.5, 0.0, -0.9, 1.2, -0.6])]
# 머리 받침 묶음(HEAD21=1): 엎드림에서 이마를 바닥에 대고(머리 +28° 숙임) 고관절을 접어 머리 · 발로 버티는 ㅅ자 → 팔로 마지막 무게 옮기기
INIT["headpike"] = [(0.6, [-0.6, -1.5, -0.5, 0.3, 0.0]), (0.8, [-1.2, -0.6, -1.2, 0.1, 0.2]), (0.8, [-1.5, 0.0, -2.0, 0.0, 0.38]),
                    (0.5, [-1.92, 0.0, -2.0, 0.6, 0.38]), (0.6, [-1.6, 0.0, -1.0, 1.6, -0.3]), (0.6, [-1.0, 0.0, 0.2, 1.9, 0.1])]
INIT_HEAD = {"headpike": [28, 28, 28, 20, 0, 0]}     # 키마다 머리 끄덕임 목표 °(+ 숙임), 없으면 0
T_STAND = 1.5
_SIMS = {}
# 민감도: 고관절 피치 굽힘 한계(설계 −80°)와 무릎 한계(설계 100°)를 환경 변수로 넓힘 — 형상 확인 필요
HIP_MIN = math.radians(float(os.environ.get("HIP_MIN_DEG", "-80")))
KNEE_MAX = math.radians(float(os.environ.get("KNEE_MAX_DEG", "100")))
LO["hip"] = HIP_MIN; HI["knee"] = KNEE_MAX
# 어깨 범위(설계 ±120°). 머리 끄덕임 조사(v21)가 팔 첫 닿음 때문에 ±110° 를 추천해 그 범위로도 돌려 봄
ARM_MAX = math.radians(float(os.environ.get("ARM_MAX_DEG", "120")))
LO["arm"] = -min(ARM_MAX, 2.094); HI["arm"] = min(ARM_MAX, 2.094)
# 머리 끄덕임도 키마다 찾음(HEAD21=1, v21 머리 · XL 크랭크 위치 제어). 벡터 끝에 키 수만큼 머리 각을 덧붙임(앞 묶음 벡터 배치는 그대로)
HEAD = bool(os.environ.get("HEAD21"))
LO["head"], HI["head"] = math.radians(-20.0), math.radians(28.0)


def to_vec(keys, head=None):
    keys = list(keys)
    while len(keys) < NK:
        keys.insert(-1, (0.5, list(keys[-2][1])))
    v = [v for dur, a in keys for v in [dur] + list(a)]
    if HEAD:
        v += list(np.radians(head)) if head is not None else [0.0] * NK
    return np.array(v)


def to_keys(x):
    x = np.asarray(x); n = NK * (1 + len(JN))
    hd = x[n:] if HEAD else None
    x = x[:n].reshape(NK, 1 + len(JN)); keys = []
    for k, row in enumerate(x):
        tg = {j: float(np.clip(row[1 + i], LO[j], HI[j])) for i, j in enumerate(JN)}
        if HEAD:
            tg["head"] = float(np.clip(hd[k], LO["head"], HI["head"]))
        keys.append((float(np.clip(row[0], 0.15, 1.5)), tg))
    keys.append((T_STAND, dict(arm=0.0, elbow=0.0, hip=0.0, knee=0.0, ankle=0.0, head=0.0)))
    return keys


def ev(job):
    name, start, x = job
    start = "prone" if start in ("pike", "tuck", "headpike") else ("supine" if start == "supine6" else start)
    if name not in _SIMS:
        _SIMS[name] = Sim(ARMS[name], knee_max=max(KNEE_MAX, 1.745), hip_min=min(HIP_MIN, -1.396))
    if os.environ.get("ROBUST"):                 # 조건을 바꿔 여러 번 돌려 비용은 평균, 성공은 모두 성공일 때만
        return ev_robust(_SIMS[name], x, start)
    try:
        r, _, _ = _SIMS[name].run(to_keys(x), start=start)
    except Exception as e:
        r = dict(ok=False, score=99.0, error=repr(e))
    if "score" not in r:
        r["score"] = 99.0
    return r


PERT = [{}, dict(friction=0.6), dict(friction=1.0), dict(tscale=0.95), dict(tscale=1.05)]


def ev_robust(S, x, start):
    m = S.m; f0 = m.geom_friction[:, 0].copy(); rs = []
    for v in PERT:
        keys = to_keys(x)
        if "tscale" in v:
            keys = [(d * v["tscale"], a) for d, a in keys]
        if "friction" in v:
            m.geom_friction[:, 0] = v["friction"]
        try:
            r, _, _ = S.run(keys, start=start)
        except Exception as e:
            r = dict(ok=False, score=99.0, error=repr(e))
        finally:
            m.geom_friction[:, 0] = f0
        rs.append(r)
    out = dict(rs[0]); out["score"] = float(np.mean([r.get("score", 99.0) for r in rs])); out["ok_all"] = all(r.get("ok") for r in rs)
    out["ok_n"] = sum(bool(r.get("ok")) for r in rs); out["ok"] = out["ok_all"]
    return out


def cem(names, starts, iters=20, pop=40, elite=8, seed=0, procs=40):
    from multiprocessing import Pool
    rng = np.random.default_rng(seed)
    combos = [(n, s) for n in names for s in starts]
    sig0 = np.array(([0.3] + [0.6] * len(JN)) * NK + ([0.3] * NK if HEAD else []))
    st = {c: dict(mu=to_vec(INIT[c[1]], INIT_HEAD.get(c[1])), sig=sig0.copy(), best=None, hist=[]) for c in combos}
    if os.environ.get("INIT_FROM"):            # tag:팔:시작 — 이미 찾은 키프레임을 처음 평균으로(다른 팔 · 서보로 옮겨 보기)
        t, a, s0 = os.environ["INIT_FROM"].split(":")
        x0 = np.array(next(r for r in json.load(open(os.path.join(RES, f"{t}.json"))) if r["arm"] == a and r["start"] == s0)["best_x"])
        for c in combos:
            st[c]["mu"] = x0.copy(); st[c]["sig"] = sig0 * float(os.environ.get("SIG_SCALE", "0.4"))
    with Pool(procs) as P:
        for it in range(iters):
            t0 = time.time(); jobs = []; X = {}
            for c in combos:
                S = st[c]; X[c] = S["mu"] + S["sig"] * rng.standard_normal((pop, len(S["mu"])))
                if S["best"] is not None:
                    X[c][0] = S["best"][0]          # 가장 좋은 것은 남김
                jobs += [(c[0], c[1], x) for x in X[c]]
            R = P.map(ev, jobs, chunksize=2); k = 0
            for c in combos:
                S = st[c]; rr = R[k:k + pop]; k += pop
                sc = np.array([r["score"] for r in rr]); o = np.argsort(sc)
                E = X[c][o[:elite]]; S["mu"] = E.mean(0); S["sig"] = np.maximum(E.std(0), 0.03)
                if S["best"] is None or sc[o[0]] <= S["best"][1]["score"]:
                    S["best"] = (X[c][o[0]].copy(), rr[o[0]])
                S["hist"].append(float(sc[o[0]])); S["n_ok"] = S.get("n_ok", 0) + sum(r.get("ok", False) for r in rr)
            print(f"iter {it} {time.time() - t0:.0f}s " + " | ".join(f"{c[0]}/{c[1][0]} {st[c]['best'][1]['score']:.2f}{'*' if st[c]['best'][1]['ok'] else ''}" for c in combos), flush=True)
    out = []
    for c in combos:
        S = st[c]
        out.append(dict(arm=c[0], start=c[1], best_x=S["best"][0].tolist(), keys=to_keys(S["best"][0]), res=S["best"][1], hist=S["hist"],
                        n_ok_total=S["n_ok"], n_eval=iters * pop))
    return out


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    pop = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    names = sys.argv[3:] or list(ARMS)
    tag = os.environ.get("TAG", "search")
    starts = os.environ.get("STARTS", "prone,supine").split(",")
    out = cem(names, starts, iters=iters, pop=pop, seed=int(os.environ.get("SEED", "0")), procs=int(os.environ.get("PROCS", "40")))
    for r in out:
        r.update(hip_min_deg=math.degrees(HIP_MIN), knee_max_deg=math.degrees(KNEE_MAX), arm_max_deg=math.degrees(ARM_MAX), selfcol=bool(os.environ.get("SELFCOL")),
                 head21=HEAD, nk=NK)
    jdump(out, os.path.join(RES, f"{tag}.json"))


def replay(r, render=None):
    """저장된 결과 r 을 같은 모델 · 같은 범위로 다시 돌림. 첫 묶음(search.json)은 범위를 −1.396 · 1.745 rad 로 돌렸다(키가 없음).
    궤적이 관절 범위 0.0003 rad 차이에도 갈라질 만큼 예민하므로 범위를 똑같이 맞춰야 같은 결과가 나온다."""
    hm = math.radians(r["hip_min_deg"]) if "hip_min_deg" in r else -1.396
    km = math.radians(r["knee_max_deg"]) if "knee_max_deg" in r else 1.745
    global NK, HEAD
    HEAD = bool(r.get("head21"))                 # v21 머리 + 머리 각 키(머리 받침 묶음)
    if HEAD:
        os.environ["HEAD21"] = "1"
    else:
        os.environ.pop("HEAD21", None)
    if r.get("selfcol"):                         # 다리 ↔ 몸통 · 머리 충돌을 켜고 찾은 묶음(h115*)
        os.environ["SELFCOL"] = "1"
    else:
        os.environ.pop("SELFCOL", None)
    am = math.radians(r.get("arm_max_deg", 120)); LO["arm"] = -min(am, 2.094); HI["arm"] = min(am, 2.094)
    LO["hip"] = hm; HI["knee"] = km; NK = r.get("nk") or len(r["best_x"]) // (1 + len(JN))
    S = Sim(ARMS[r["arm"]], knee_max=max(km, 1.745), hip_min=min(hm, -1.396))
    st = "prone" if r["start"] in ("pike", "tuck", "headpike") else ("supine" if r["start"] == "supine6" else r["start"])
    res, H, fr = S.run(to_keys(r["best_x"]), start=st, render=render)
    return S, res, H, fr
