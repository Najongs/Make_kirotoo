import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from build import build_model, IDX, UVI
links,joints,_=build_model()
def prism_faces(axis,a0,a1,poly,holes):
    k,(iu,iv)=IDX[axis],UVI[axis]
    def P(u,v,a):
        p=[0,0,0]; p[iu]=u;p[iv]=v;p[k]=a; return p
    b=np.array([P(u,v,a0) for u,v in poly]); t=np.array([P(u,v,a1) for u,v in poly]); n=len(poly)
    return [b,t]+[np.array([b[i],b[(i+1)%n],t[(i+1)%n],t[i]]) for i in range(n)]
def cyl_faces(c,a,r,h,n=18):
    a=np.array(a,float); c=np.array(c,float)
    ref=np.array([0,0,1.]) if abs(a[2])<.9 else np.array([1.,0,0]); ref-=ref.dot(a)*a; ref/=np.linalg.norm(ref); bb=np.cross(a,ref)
    th=np.linspace(0,2*np.pi,n,endpoint=False)
    ring=np.array([c+r*(np.cos(t)*ref+np.sin(t)*bb) for t in th]); top=ring+a*h
    return [ring,top]+[np.array([ring[i],ring[(i+1)%n],top[(i+1)%n],top[i]]) for i in range(n)]
def draw(ax,elev,azim,title,lim=None,only=None,parts_filter=None,alpha=1.0):
    for ln,L in links.items():
        if only and ln not in only: continue
        for pn,sol in L["parts"].items():
            if parts_filter and not parts_filter(pn): continue
            steel=("_bolt_" in pn or "_nut_" in pn)
            for kind,args,rgb in sol:
                fs=prism_faces(*args) if kind=="prism" else cyl_faces(*args)
                ax.add_collection3d(Poly3DCollection(fs,facecolors=[(*rgb,1.0 if steel else alpha)],edgecolors=(0,0,0,0.3),linewidths=0.25))
    (x0,x1),(y0,y1),(z0,z1)=lim or ((-130,130),(-130,130),(0,380))
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_zlim(z0,z1)
    ax.set_box_aspect((x1-x0,y1-y0,z1-z0)); ax.view_init(elev,azim); ax.set_title(title,fontsize=11)
    ax.set_xlabel("X");ax.set_ylabel("Y");ax.set_zlabel("Z")
fig=plt.figure(figsize=(16,7))
draw(fig.add_subplot(141,projection="3d"),20,-55,"Isometric")
draw(fig.add_subplot(142,projection="3d"),0,0,"Front (from +X)")
draw(fig.add_subplot(143,projection="3d"),0,-90,"Side (from -Y)")
draw(fig.add_subplot(144,projection="3d"),0,0,"R thigh/shank M3 through-bolts (front)",
     lim=((-30,45),(-80,-5),(60,170)),only={"R_thigh","R_shank"},
     parts_filter=lambda p: "servo" not in p,alpha=0.35)
plt.tight_layout(); plt.savefig("out/biped_preview.png",dpi=110)
