import math
g=9.81
# mass budget (g)
mass = {
 "XC430-W240 x12": 12*65,
 "3D printed brackets/links (PETG/PA, 12 joints)": 190,
 "Feet x2 (incl. rubber)": 60,
 "Pelvis/torso frame": 80,
 "Battery 3S 1000mAh": 85,
 "SBC (Raspberry Pi 5 + cooler)": 55,
 "U2D2 x2": 18,
 "IMU (BNO085 breakout)": 5,
 "5V 5A buck + power hub": 25,
 "Cables/screws/horns": 70,
}
M=sum(mass.values())/1000
for k,v in mass.items(): print(f"{k}: {v} g")
print("TOTAL kg", M)
leg_mass = (6*65 + 190/2 + 30 + 35/2)/1000  # one leg approx
print("one leg kg", leg_mass)
L1=L2=0.075
for crouch in [0.95,0.90,0.85,0.80]:
    h=crouch*(L1+L2)
    d=math.sqrt(L1**2-(h/2)**2)
    m_above_knee = M - (65*2+40+30)/1000  # minus shank+foot of stance leg
    tk = m_above_knee*g*d
    print(f"crouch {crouch:.2f}: hip-ankle {h*1000:.0f}mm knee offset {d*1000:.1f}mm knee static {tk:.2f} Nm  x1.8 dyn {tk*1.8:.2f}")
# hip roll single support: upper body CoM lateral offset ~ half hip width; swing leg further out
hip_w=0.080
upper = M - 2*leg_mass
t_roll = upper*g*(hip_w/2) + leg_mass*g*hip_w
print("upper body kg", upper, "hip roll static", round(t_roll,2), "dyn", round(t_roll*1.5,2))
# ankle: max torque when ZMP at foot edge
foot_L=0.12; foot_W=0.07
print("ankle pitch max (ZMP at toe, from center) ", round((M)*g*foot_L/2,2))
print("ankle roll max (ZMP at edge)", round(M*g*foot_W/2,2))
# hip pitch swing leg acceleration: leg as rod 0.18m, swing 0.5rad amplitude at 1.5Hz
I = leg_mass*(0.17**2)/3
w=2*math.pi*1.5; A=0.35
print("hip pitch swing inertial", round(I*A*w**2,2), "+gravity", round(leg_mass*g*0.085*math.sin(A),2))
print("max joint speed needed rad/s", round(A*w,2), "vs no-load 70rpm", round(70*2*math.pi/60,2))
# bus timing
def t_bytes(n,baud): return n*10/baud
sw = 14 + 12*(1+4)  # sync write header+crc ~14
print("sync write 12x pos us @4M", round(t_bytes(sw,4e6)*1e6))
