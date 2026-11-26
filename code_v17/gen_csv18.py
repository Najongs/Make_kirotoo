import json, csv
m=json.load(open('../out18/model.json')); ms=json.load(open('mass_build18_light.json'))
SPEC={'XM430-W350':(3.8,4.5),'XC430-W240':(1.75,6.81),'XL430-W250':(1.4,5.45)}
with open('../out18/joints.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['joint','parent_link','child_link','axis','x_mm','y_mm','z_mm','lower_deg','upper_deg','actuator','stall_Nm_11.1V','no_load_rad_s_11.1V'])
    for j in m['joints']:
        s=next((v for k,v in SPEC.items() if j['actuator'].startswith(k)),('',''))
        p=j['point_mm']; w.writerow([j['name'],j['parent'],j['child'],j['axis'],*[round(x,2) for x in p],j['lower_deg'],j['upper_deg'],j['actuator'],*s])
with open('../out18/link_mass_v18.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['link','mass_kg','com_x_m','com_y_m','com_z_m','Ixx','Iyy','Izz','Ixy','Ixz','Iyz','note'])
    for n in m['links']:
        L=ms['links'].get(n)
        if not L: print('missing',n); continue
        c=L['com_local']; I=L['inertia_gmm2']
        w.writerow([n,round(L['mass_g']/1000,4),*[round(x/1000,5) for x in c],*['%.3e'%(I[a][b]*1e-9) for a,b in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]],'CoM from link origin, inertia about CoM, world axes at zero pose'])
print('total',ms['total_g'],ms['com'])
