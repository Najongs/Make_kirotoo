# Re-read the STEP, resolve assembly transforms, compare every solid's world bbox to the model.
import re, collections
from build import build_model, aabb
t=open("out/biped_robot.stp").read().split("DATA;")[1].split("ENDSEC;")[0]
D={int(m.group(1)):m.group(2) for m in re.finditer(r"#(\d+)=(.*?);\n",t+"\n")}
refs=lambda s:[int(x) for x in re.findall(r"#(\d+)",re.sub(r"'[^']*'","",s))]
def pt(i):
    return tuple(float(x) for x in re.search(r"\(([^()]*)\)\)$",D[i]).group(1).split(","))
def axis_loc(i): return pt(refs(D[i])[0])
parent={}  # child rep -> (parent rep, translation)
for i,v in D.items():
    if v.startswith("( REPRESENTATION_RELATIONSHIP"):
        r=refs(v); child,par,idt=r[0],r[1],r[2]
        a,b=refs(D[idt]); oa,ob=axis_loc(a),axis_loc(b)
        parent[child]=(par,tuple(y-x for x,y in zip(oa,ob)))
def world_off(rep):
    o=(0,0,0)
    while rep in parent:
        rep,tr=parent[rep]; o=tuple(a+b for a,b in zip(o,tr))
    return o
def collect_pts(i,acc,seen):
    if i in seen: return
    seen.add(i); v=D[i]
    if v.startswith("CARTESIAN_POINT"): acc.append(pt(i)); return
    if v.startswith(("CIRCLE","CYLINDRICAL_SURFACE")):
        r=float(v.rsplit(",",1)[1].rstrip(")")); ax=refs(v)[0]; c=axis_loc(ax); z=pt(refs(D[ax])[1])
        for k in range(3):
            if abs(z[k])<0.5:
                for s in (-1,1):
                    p=list(c); p[k]+=s*r; acc.append(tuple(p))
        return
    if v.startswith(("PLANE","AXIS2","DIRECTION","VECTOR","LINE")): return
    for j in refs(v): collect_pts(j,acc,seen)
got=[]
for i,v in D.items():
    if v.startswith("ADVANCED_BREP_SHAPE_REPRESENTATION"):
        off=world_off(i)
        for b in refs(v)[1:-1]:
            if D[b].startswith("MANIFOLD"):
                ps=[];collect_pts(b,ps,set())
                bb=tuple((round(min(p[k] for p in ps)+off[k],3),round(max(p[k] for p in ps)+off[k],3)) for k in range(3))
                got.append(bb)
links,_,_=build_model()
exp=[tuple((round(a,3),round(b,3)) for a,b in aabb(s)) for L in links.values() for sol in L["parts"].values() for s in sol]
print("solids in STEP",len(got),"expected",len(exp))
print("all world bboxes match:", collections.Counter(got)==collections.Counter(exp))
allp=[c for bb in got for c in bb]
print("overall bbox:",[ (min(bb[k][0] for bb in got), max(bb[k][1] for bb in got)) for k in range(3)])
cg,ce=collections.Counter(got),collections.Counter(exp)
print("only in STEP:",list((cg-ce).items())[:5]); print("only in model:",list((ce-cg).items())[:5])
