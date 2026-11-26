"""XL430-W250 공식 STEP(ref/XL-430_new.stp, ROBOTIS download.php?no=773 → Dropbox XL-430_new.stp, 2021-12-22 Creo)과
지금 설계가 쓰는 XC430 STEP(references/XC-430_idle.stp)의 원통면(구멍 · 혼 · 아이들러)을 같은 방식으로 뽑아 견준다.
code_v17/x430_holes.py 와 같은 STEP 읽기(CYLINDRICAL_SURFACE 반지름 · 축 원점 · 방향), 제품(PRODUCT)별로 묶음.
assembly 변환은 적용하지 않는다(x430_holes 와 같은 가정) — 대신 제품마다 점 범위를 찍어 서보 좌표계(혼 축 원점, 혼 +z,
긴 축 +y, 케이스 x ±14.25 · y −35.25..11.25 · z ±17)에 맞는지 확인한다.
usage: cd sim/walk/allxl && ../../.venv/bin/python xl_geom_check.py → results/xl_geom.json"""
import os, re, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
STEPS = {"XL430": os.path.join(HERE, "ref", "XL-430_new.stp"), "XC430": os.path.join(REPO, "references", "XC-430_idle.stp")}


def parse(path):
    t = open(path).read()
    data = t.split("DATA;")[1].split("ENDSEC;")[0]
    D = {int(m.group(1)): " ".join(m.group(2).split()) for m in re.finditer(r"#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|$)", data, re.S)}
    refs = lambda s: [int(x) for x in re.findall(r"#(\d+)", re.sub(r"'[^']*'", "", s))]
    nums = lambda s: [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)]
    tup = lambda i: np.array(nums(D[i].split("(", 1)[1].split("(", 1)[1].split(")")[0]))
    pname = {i: re.match(r"PRODUCT\(\s*'([^']*)'", v).group(1) for i, v in D.items() if v.startswith("PRODUCT(")}
    form = {i: refs(v)[0] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_FORMATION")}
    pdef = {i: pname[form[refs(v)[0]]] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION(")}
    pds = {i: pdef[refs(v)[0]] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_SHAPE") and refs(v)[0] in pdef}
    rep2name = {}
    for v in D.values():
        if v.startswith("SHAPE_DEFINITION_REPRESENTATION"):
            a, b = refs(v)[:2]
            if a in pds:
                rep2name[b] = pds[a]
    for _ in range(3):
        for v in D.values():
            if v.startswith("SHAPE_REPRESENTATION_RELATIONSHIP"):
                a, b = refs(v)[:2]
                if a in rep2name and b not in rep2name:
                    rep2name[b] = rep2name[a]
                if b in rep2name and a not in rep2name:
                    rep2name[a] = rep2name[b]
    reps = {i for i, v in D.items() if "SHAPE_REPRESENTATION" in v and not v.startswith(("SHAPE_DEFINITION", "CONTEXT")) and "RELATIONSHIP" not in v}
    out = {}
    for r in reps:
        nm = rep2name.get(r)
        if nm is None:
            continue
        cyl, pts, seen = [], [], set()

        def walk(i):
            if i in seen:
                return
            seen.add(i); v = D[i]
            if v.startswith("CARTESIAN_POINT"):
                pts.append(tup(i)); return
            if v.startswith("CYLINDRICAL_SURFACE"):
                a = refs(v)[0]; rr = nums(v.split(",")[-1])[0]; ra = refs(D[a])
                cyl.append((rr, tup(ra[0]), tup(ra[1])))
            if v.startswith(("DIRECTION", "AXIS2_PLACEMENT")):
                return
            for j in refs(v):
                if j not in reps:
                    walk(j)
        for it in refs(D[r])[:-1]:
            if D[it].startswith(("MANIFOLD_SOLID_BREP", "BREP_WITH_VOIDS")):
                walk(it)
        if pts:
            P = np.array(pts)
            o = out.setdefault(nm, dict(cyl=[], lo=P.min(0), hi=P.max(0)))
            o["cyl"] += cyl; o["lo"] = np.minimum(o["lo"], P.min(0)); o["hi"] = np.maximum(o["hi"], P.max(0))
    return out


def holes(prod, rmin, rmax):
    s = set()
    for r, o, z in prod["cyl"]:
        if rmin - 1e-6 <= r <= rmax + 1e-6:
            k = int(np.argmax(np.abs(z)))
            s.add((round(r, 3),) + tuple(np.round(o, 2)) + ("xyz"[k],))
    return sorted(s)


if __name__ == "__main__":
    res = {}
    for k, p in STEPS.items():
        P = parse(p)
        res[k] = {nm: dict(lo=v["lo"].round(2).tolist(), hi=v["hi"].round(2).tolist(),
                           radii=sorted({round(r, 3) for r, _, _ in v["cyl"]}),
                           small_holes=holes(v, 0.7, 1.35)) for nm, v in P.items()}
        print(f"== {k} {os.path.basename(p)}")
        for nm, v in res[k].items():
            print(f"  {nm:28s} lo {v['lo']} hi {v['hi']} radii {v['radii'][:14]}")
    # 케이스 구멍(r 1.05 = Ø2.1) 비교: XC 는 DC11_B01_CASE_M_DUMMY, XL 은 모든 제품에서 r 1.05
    def case_holes(prods):
        s = set()
        for nm, v in prods.items():
            for h in v["small_holes"]:
                if abs(h[0] - 1.05) < 1e-3:
                    s.add(h[1:])
        return sorted(s)
    xl, xc = case_holes(res["XL430"]), case_holes(res["XC430"])
    print("XC r1.05 holes", len(xc)); print("XL r1.05 holes", len(xl))
    sxl = {h[:3] for h in xl}; sxc = {h[:3] for h in xc}
    print("in XC not XL:", sorted(sxc - sxl)); print("in XL not XC:", sorted(sxl - sxc))
    res["case_holes"] = dict(XC=xc, XL=xl, only_XC=sorted(sxc - sxl), only_XL=sorted(sxl - sxc))
    for k in ("XL430", "XC430"):
        for nm, v in res[k].items():
            hs = [h for h in v["small_holes"] if abs(h[0] - 1.05) > 1e-3]
            if hs:
                print(k, nm, "other small holes:", hs[:16])
    json.dump(res, open(os.path.join(HERE, "results", "xl_geom.json"), "w"), indent=1, ensure_ascii=False, default=str)
