import math
g=9.81
base = {
 "servo":12*65,"print_links":230,"feet":60,"torso":120,"battery_3S_2200":180,
 "jetson_orin_nano_devkit":174,"camera(est)":70,"u2d2x2":18,"imu":5,
 "power_conv(12V buckboost+hub)":40,"cables":80}
def report(name, M, xm):
    # servo stall per joint
    L=0.075; h=0.85*2*L; d=math.sqrt(L*L-(h/2)**2)
    shank_foot = (2*(82 if xm else 65)+40+30)/1000  # ankle servos + shank + foot
    knee_s = (M-shank_foot)*g*d; knee_d=knee_s*1.8
    leg = (6*65 + (3*17 if xm else 0) + 115+30+17)/1000
    upper = M-2*leg
    roll_s = upper*g*0.04 + leg*g*0.08; roll_d=roll_s*1.5
    ank_p = M*g*0.06; ank_r=M*g*0.035
    I=leg*0.17**2/3; w=2*math.pi*1.5; A=0.35
    hp = I*A*w*w + leg*g*0.085*math.sin(A)
    rows=[("knee",knee_s,knee_d,xm),("hip_roll",roll_s,roll_d,xm),("ankle_pitch",None,ank_p,xm),
          ("ankle_roll",None,ank_r,False),("hip_pitch",hp/2,hp,False)]
    print(f"== {name}: M={M:.3f} kg, leg={leg:.3f}")
    for j,s,dyn,isxm in rows:
        stall=4.1 if isxm else 1.9
        print(f"{j:12s} static {s if s is None else round(s,2)} dyn {dyn:.2f} -> {100*dyn/stall:.0f}% of {stall} ; cont20%={0.2*stall:.2f}")
M_A=sum(base.values())/1000
report("A all XC430 + Jetson",M_A,False)
report("B XM430-W350 at knee/ankle pitch/hip roll",M_A+6*0.017,True)
print("total A g", sum(base.values()))
# battery runtime
for wh,lab in [(24.4,"3S 2200")]:
    for P in [45,60]:
        print(lab,P,"W ->",round(wh*0.8/P*60),"min (80% usable)")
