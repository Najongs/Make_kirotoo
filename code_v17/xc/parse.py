import re, sys, numpy as np, collections
t = open(sys.argv[1]).read()
data = t.split("DATA;")[1].split("ENDSEC;")[0]
D = {}
for m in re.finditer(r"#(\d+)\s*=\s*(.*?);\s*(?=#\d+\s*=|$)", data, re.S):
    D[int(m.group(1))] = " ".join(m.group(2).split())
refs = lambda s: [int(x) for x in re.findall(r"#(\d+)", re.sub(r"'[^']*'", "", s))]
def nums(s): return [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)]
def pt(i): return np.array(nums(re.search(r"\(\((.*)\)\)", D[i]).group(1)) if False else nums(D[i].split("(",1)[1].split("(",1)[1].split(")")[0]))
def dirv(i): return np.array(nums(D[i].split("(",1)[1].split("(",1)[1].split(")")[0]))
def ax2(i):
    r = refs(D[i]); o = pt(r[0]); z = dirv(r[1]) if len(r) > 1 else np.array([0,0,1.]); x = dirv(r[2]) if len(r) > 2 else np.array([1.,0,0])
    z = z/np.linalg.norm(z); x = x - x.dot(z)*z; x/=np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3,0]=x; M[:3,1]=y; M[:3,2]=z; M[:3,3]=o; return M
# representation -> product name
pd2name = {}
for i, v in D.items():
    if v.startswith("PRODUCT("):
        name = re.match(r"PRODUCT\(\s*'([^']*)'", v).group(1); pd2name.setdefault("p", {})[i] = name
pname = pd2name["p"]
form = {i: refs(v)[0] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_FORMATION")}
pdef = {i: pname[form[refs(v)[0]]] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION(")}
pds = {i: pdef[refs(v)[0]] for i, v in D.items() if v.startswith("PRODUCT_DEFINITION_SHAPE") and refs(v)[0] in pdef}
rep2name = {}
for i, v in D.items():
    if v.startswith("SHAPE_DEFINITION_REPRESENTATION"):
        a, b = refs(v)[:2]
        if a in pds: rep2name[b] = pds[a]
# transforms: child rep placed in parent rep
parent = {}
for i, v in D.items():
    if "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION" in v:
        r = refs(v)
        # (REPRESENTATION_RELATIONSHIP('','',#rep1,#rep2) REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#idt) SHAPE_...)
        rep1, rep2, idt = r[0], r[1], r[2]
        a1, a2 = refs(D[idt])[:2]
        T = ax2(a2) @ np.linalg.inv(ax2(a1))
        parent.setdefault(rep1, []).append((rep2, T))
def world(rep):
    if rep not in parent: return [np.eye(4)]
    out = []
    for p, T in parent[rep]:
        for W in world(p): out.append(W @ T)
    return out
# points per rep (recursive through items, skip into other reps)
reps = {i for i, v in D.items() if "SHAPE_REPRESENTATION" in v and not v.startswith("SHAPE_DEFINITION") and not v.startswith("CONTEXT") and "RELATIONSHIP" not in v}
def collect(i, acc, seen):
    if i in seen: return
    seen.add(i); v = D[i]
    if v.startswith("CARTESIAN_POINT"): acc.append(pt(i)); return
    if v.startswith(("AXIS2_PLACEMENT","DIRECTION")) : return
    for j in refs(v):
        if j in reps: continue
        collect(j, acc, seen)
res = collections.defaultdict(list)
for r in reps:
    nm = rep2name.get(r)
    acc = []
    for it in refs(D[r])[:-1]:
        if D[it].startswith(("MANIFOLD_SOLID_BREP","BREP_WITH_VOIDS","SHELL_BASED")):
            collect(it, acc, set())
    if not acc: continue
    P = np.array(acc)
    # find reps that are geometric for a product: rep may be linked by SHAPE_REPRESENTATION_RELATIONSHIP to product rep
    res[r] = P
# map geometric reps to product reps through SHAPE_REPRESENTATION_RELATIONSHIP (no transform)
link = {}
for i, v in D.items():
    if v.startswith("SHAPE_REPRESENTATION_RELATIONSHIP"):
        a, b = refs(v)[:2]; link[b] = a; link.setdefault(a, a)
for r, P in res.items():
    pr = r if r in rep2name else link.get(r, r)
    if pr not in rep2name:
        for a, b in [(refs(v)[0], refs(v)[1]) for v in D.values() if v.startswith("SHAPE_REPRESENTATION_RELATIONSHIP")]:
            if b == r: pr = a
            if a == r: pr = b
    nm = rep2name.get(pr, f"rep{r}")
    for W in world(pr):
        Q = (np.c_[P, np.ones(len(P))] @ W.T)[:, :3]
        sc = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        print(f"{nm:28s} min {np.round(Q.min(0)*sc,2)} max {np.round(Q.max(0)*sc,2)}")
