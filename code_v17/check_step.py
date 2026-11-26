import re,sys,collections
t=open(sys.argv[1]).read()
data=t.split("DATA;")[1].split("ENDSEC;")[0]
ents=[l for l in data.strip().split("\n") if l]
defs={}
for l in ents:
    m=re.match(r"#(\d+)=(.*);$",l)
    assert m, l[:80]
    i=int(m.group(1)); assert i not in defs; defs[i]=m.group(2)
    body=m.group(2)
    # paren balance outside strings
    s=re.sub(r"'[^']*'","",body)
    assert s.count("(")==s.count(")"), l[:100]
refs=set(int(x) for l in defs.values() for x in re.findall(r"#(\d+)",re.sub(r"'[^']*'","",l)))
missing=refs-set(defs)
types=collections.Counter(re.match(r"\(?\s*([A-Z_0-9]+)",v).group(1) for v in defs.values())
print("entities",len(defs),"missing refs",len(missing))
for k in ["PRODUCT","NEXT_ASSEMBLY_USAGE_OCCURRENCE","MANIFOLD_SOLID_BREP","CYLINDRICAL_SURFACE","ADVANCED_FACE","STYLED_ITEM","CONTEXT_DEPENDENT_SHAPE_REPRESENTATION"]:
    print(k,types[k])
# every EDGE_CURVE used exactly twice by ORIENTED_EDGE (manifold check)
use=collections.Counter()
for v in defs.values():
    m=re.match(r"ORIENTED_EDGE\('',\*,\*,#(\d+),\.(T|F)\.\)",v)
    if m: use[(int(m.group(1)),m.group(2))]+=1
ec=[i for i,v in defs.items() if v.startswith("EDGE_CURVE")]
bad=[e for e in ec if use[(e,'T')]!=1 or use[(e,'F')]!=1]
print("edge curves",len(ec),"non-manifold/orientation issues",len(bad))
