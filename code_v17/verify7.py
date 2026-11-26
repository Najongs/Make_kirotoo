# Re-read the STEP, resolve assembly transforms, compare every solid's world bbox to the model.
import re, collections
from build7 import build_model; from build4 import aabb
t=open("out7/biped_robot_v7.stp").read().split("DATA;")[1].split("ENDSEC;")[0]
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
    if v.startswith("EDGE_CURVE"):
        r_=refs(v)
        if r_[0]!=r_[1]:
            for j in r_[:2]: collect_pts(j,acc,seen)
            cv=r_[2]
            if D[cv].startswith("CIRCLE"):          # sample the arc between its end vertices
                ax=refs(D[cv])[0]; c=pt(refs(D[ax])[0]); zd=pt(refs(D[ax])[1]); xd=pt(refs(D[ax])[2])
                yd=(zd[1]*xd[2]-zd[2]*xd[1], zd[2]*xd[0]-zd[0]*xd[2], zd[0]*xd[1]-zd[1]*xd[0])
                rr=float(D[cv].rsplit(",",1)[1].rstrip(")"))
                import math as _m
                def ang(p):
                    d=[p[i]-c[i] for i in range(3)]
                    return _m.atan2(sum(d[i]*yd[i] for i in range(3)), sum(d[i]*xd[i] for i in range(3)))
                pa,pb=pt(refs(D[r_[0]])[0]),pt(refs(D[r_[1]])[0]); t0,t1=ang(pa),ang(pb)
                sense=v.rstrip(")").endswith(".T.")
                if sense:
                    while t1<=t0: t1+=2*_m.pi
                else:
                    while t1>=t0: t1-=2*_m.pi
                for k in range(1,64):
                    t=t0+(t1-t0)*k/64
                    acc.append(tuple(c[i]+rr*(_m.cos(t)*xd[i]+_m.sin(t)*yd[i]) for i in range(3)))
            return
    if v.startswith("CYLINDRICAL_SURFACE"): return
    if v.startswith("SPHERICAL_SURFACE"):
        # a sphere face is bounded by its cut circle; add each axis pole that lies on the kept side
        R=float(v.rsplit(",",1)[1].rstrip(")")); ax=refs(v)[0]; c=axis_loc(ax); a=pt(refs(D[ax])[1])
        cut=[w for w in D.values() if False]
        acc.append(("SPH",c,a,R)); return
    if v.startswith("CIRCLE"):
        r=float(v.rsplit(",",1)[1].rstrip(")")); ax=refs(v)[0]; c=axis_loc(ax); z=pt(refs(D[ax])[1])
        for k in range(3):
            w=r*max(0.0,1-z[k]**2)**0.5
            for s in (-1,1):
                p=list(c); p[k]+=s*w; acc.append(tuple(p))
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
                sph=[p for p in ps if p[0]=="SPH"]; ps=[p for p in ps if p[0]!="SPH"]
                for _,c,a,R in sph:          # pole along +-axis k is on the face if it lies on the cut-plane side of the circles
                    for k in range(3):
                        for sgn in (-1,1):
                            q=list(c); q[k]+=sgn*R
                            # kept side: the side of the cut plane containing the far pole c + R a
                            if sgn*a[k]*R >= min(sum((p[j]-c[j])*a[j] for j in range(3)) for p in ps)-1e-6:
                                ps.append(tuple(q))
                bb=tuple((round(min(p[k] for p in ps)+off[k],3),round(max(p[k] for p in ps)+off[k],3)) for k in range(3))
                got.append(bb)
links,_,_=build_model()
exp=[tuple((round(a,3),round(b,3)) for a,b in aabb(s)) for L in links.values() for sol in L["parts"].values() for s in sol]
print("solids in STEP",len(got),"expected",len(exp))
def close(a,b,tol=0.05): return all(abs(a[i][j]-b[i][j])<=tol for i in range(3) for j in range(2))
unmatched=list(exp); bad=[]
for g_ in got:
    m=[e for e in unmatched if close(g_,e)]
    if m: unmatched.remove(m[0])
    else: bad.append(g_)
print("all world bboxes match within 0.05 mm:", not bad and not unmatched, "| unmatched:", len(bad))
allp=[c for bb in got for c in bb]
print("overall bbox:",[ (min(bb[k][0] for bb in got), max(bb[k][1] for bb in got)) for k in range(3)])
