# Inside each link: fastener solids must not overlap other solids, except a shank passing through a coaxial clearance hole.
import itertools
from build import build_model, aabb, IDX, UVI
links,_,_=build_model()
def ov(b1,b2,t=0.01): return all(min(b1[i][1],b2[i][1])-max(b1[i][0],b2[i][0])>t for i in range(3))
issues=[]
for ln,L in links.items():
    sol=[(pn,s) for pn,ss in L["parts"].items() for s in ss]
    fast=[(pn,s) for pn,s in sol if "_bolt_" in pn or "_nut_" in pn]
    for (pf,f),(po,o) in itertools.product(fast,sol):
        if pf==po: continue
        if not ov(aabb(f),aabb(o)): continue
        ok=False
        if f[0]=="cyl" and o[0]=="prism":
            c,a,r,h=f[1]; axis=o[1][0]; k=IDX[axis]; iu,iv=UVI[axis]
            if abs(a[k])>0.9:
                for hu,hv,hr in o[1][4]:
                    if abs(hu-c[iu])<1e-6 and abs(hv-c[iv])<1e-6 and hr>=r-1e-9: ok=True
        if f[0]=="prism" and o[0]=="cyl" and "_bolt_" in po:  # nut on its own bolt shank
            c,a,r,h=o[1]; axis=f[1][0]; k=IDX[axis]; iu,iv=UVI[axis]
            if any(abs(hu-c[iu])<1e-6 and abs(hv-c[iv])<1e-6 and hr>=r-1e-9 for hu,hv,hr in f[1][4]): ok=True
        if not ok: issues.append((ln,pf,po))
print("intra-link fastener issues:",len(issues))
for i in issues[:20]: print(" ",i)
