"""진단 그림(개발용): python diag.py out.png '{"kp_scale":2,...}'"""
import sys, json
import numpy as np
from kw_sim import run, brief
from kw_common import korean_font

cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
o, R, m = run(**cfg)
print(brief(o))
plt = korean_font()
fig, ax = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
t5 = np.arange(len(R["com"])) * 5e-3; tc = R["t"]
ax[0].plot(t5, np.degrees(R["rpy"][:, 0]), label="roll"); ax[0].plot(t5, np.degrees(R["rpy"][:, 1]), label="pitch")
ax[0].plot(t5, np.degrees(R["rpy"][:, 2]), label="yaw"); ax[0].plot(tc, np.degrees(R["ik_roll"]), label="IK roll", ls="--"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(t5, R["com"][:, 1], label="true com y"); ax[1].plot(tc, R["c_est"][:, 1], label="est y")
ax[1].plot(tc, R["c_ref"][:, 1], label="ref y"); ax[1].plot(tc, R["ik_com"][:, 1], label="IK com y", ls="--"); ax[1].plot(tc, R["p_d"][:, 1], label="p_d y", lw=.8)
ax[1].plot(t5, R["cop"][:, 1], label="cop y", lw=.6); ax[1].plot(tc, R["p_ref"][:, 1], label="p_ref y", ls=":"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
ax[2].plot(t5, R["com"][:, 0], label="true com x"); ax[2].plot(tc, R["c_est"][:, 0], label="est x")
ax[2].plot(tc, R["c_ref"][:, 0], label="ref x"); ax[2].plot(tc, R["p_d"][:, 0], label="p_d x", lw=.8)
ax[2].plot(t5, R["cop"][:, 0], label="cop x", lw=.6); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
ax[3].plot(t5, R["margin"] * 1e3, label="margin mm"); ax[3].set_ylim(-20, 40); ax[3].grid(alpha=.3)
sup = np.array([0 if s == "R" else 1 for s in R["support"]]); ax[3].plot(tc, sup * 10, label="support L=10"); ax[3].legend()
for i, nm in enumerate(m.actuator(j).name for j in range(m.nu)):
    if nm.startswith("R_"):
        ax[4].plot(t5, R["u"][:, i], label=nm, lw=.8)
ax[4].legend(fontsize=7); ax[4].grid(alpha=.3); ax[4].set_ylim(0, 2.2)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=80)
