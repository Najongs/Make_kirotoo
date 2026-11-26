"""Check one joint's range slowly with a low PWM, then return to the start pose.

  python tools/range_check.py --joint R_knee
  python tools/range_check.py --joint R_knee --lo -5 --hi 30 --pwm 150 --yes

Only the chosen joint gets torque; every other servo stays limp, so hang the robot on
a stand. Each step waits until the joint arrives; if it stops short (a part touches, or
the low PWM cannot lift the link), that direction ends there and the angle is reported.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import _common
from kitoru_hw.control_table import PWM_UNIT_PERCENT, TICK_RAD

MAX_STEP_DEG = 5.0          # same as the robot's per-command jump guard
MAX_PWM = 885               # PWM Limit(36) factory value
MAX_PROFILE_VELOCITY = 50   # 50 x 0.229 rev/min = 1.2 rad/s; 0 would mean "no limit" and is refused
UNFILLED_RANGE_DEG = 30.0   # joints without a YAML range: --lo/--hi must stay within +-30 deg


def move_to(robot, name, angle, timeout, tolerance_rad):
    robot.command_positions({name: angle})
    deadline = time.monotonic() + timeout
    peak = 0
    while True:
        state = robot.read_state([name])[name]
        peak = max(peak, abs(state["load_or_current"]))
        if abs(state["position_rad"] - angle) <= tolerance_rad:
            return True, state["position_rad"], peak
        if time.monotonic() >= deadline:
            return False, state["position_rad"], peak
        time.sleep(0.02)


def sweep(robot, name, start, end, step_rad, timeout, tolerance_rad):
    """Step from start toward end. Returns (reached the end, last measured angle, peak load)."""
    direction = 1 if end >= start else -1
    angle, measured, peak = start, start, 0
    while (end - angle) * direction > 1e-9:
        angle = end if abs(end - angle) <= step_rad else angle + direction * step_rad
        ok, measured, load = move_to(robot, name, angle, timeout, tolerance_rad)
        peak = max(peak, load)
        if not ok:
            return False, measured, peak
    return True, measured, peak


def main(argv=None, confirm=input, **robot_kwargs):
    parser = _common.parser(__doc__.splitlines()[0])
    parser.add_argument("--joint", required=True)
    parser.add_argument("--lo", type=float, help="lower end in deg (default: software limit)")
    parser.add_argument("--hi", type=float, help="upper end in deg (default: software limit)")
    parser.add_argument("--step", type=float, default=2.0, help="step in deg")
    parser.add_argument("--pwm", type=int, default=200, help="Goal PWM while testing (885 = 100 %%)")
    parser.add_argument("--profile-velocity", type=int, default=10, help="0.229 rev/min units")
    parser.add_argument("--timeout", type=float, default=3.0, help="seconds per step")
    parser.add_argument("--yes", action="store_true", help="do not ask before moving")
    args = parser.parse_args(argv)
    args.joints = args.joint
    if not 0 < args.step <= MAX_STEP_DEG:
        parser.error(f"--step must be in (0, {MAX_STEP_DEG}] deg")
    if not 1 <= args.pwm <= MAX_PWM:
        parser.error(f"--pwm must be 1..{MAX_PWM}")
    if not 1 <= args.profile_velocity <= MAX_PROFILE_VELOCITY:
        parser.error(f"--profile-velocity must be 1..{MAX_PROFILE_VELOCITY} (0 means unlimited speed)")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        with _common.make_robot(args, **robot_kwargs) as robot:
            joint = robot.joints[args.joint]
            soft = joint.soft_limits_rad
            lo = math.radians(args.lo) if args.lo is not None else soft[0] if soft else None
            hi = math.radians(args.hi) if args.hi is not None else soft[1] if soft else None
            if lo is None or hi is None:
                parser.exit(1, f"Error: {args.joint} has no software limits; give --lo and --hi\n")
            if soft and not (soft[0] <= lo <= hi <= soft[1]):
                parser.exit(1, "Error: --lo/--hi must stay inside the software limits "
                            f"{math.degrees(soft[0]):.1f}..{math.degrees(soft[1]):.1f} deg\n")
            if lo > hi:
                parser.exit(1, "Error: --lo must not be above --hi\n")
            if soft is None and max(abs(lo), abs(hi)) > math.radians(UNFILLED_RANGE_DEG):
                parser.exit(1, f"Error: {args.joint} has no range in the YAML yet; keep --lo/--hi within "
                            f"+-{UNFILLED_RANGE_DEG:.0f} deg\n")
            if soft is None:
                # Joint without a range yet (e.g. hip roll before the v20 check): use --lo/--hi for this run only.
                robot.joints[args.joint] = replace(joint, soft_limits_rad=(lo, hi))
                print(f"{args.joint}: no software limits in the YAML, using --lo/--hi for this run only")
            print(f"{args.joint}: {math.degrees(lo):.1f} .. {math.degrees(hi):.1f} deg, step {args.step} deg, "
                  f"Goal PWM {args.pwm} ({args.pwm * PWM_UNIT_PERCENT:.0f} %), profile velocity {args.profile_velocity}")
            if not args.yes and confirm("Robot on a stand, hands clear? Type yes: ").strip() != "yes":
                parser.exit(1, "Cancelled\n")
            robot.arm([args.joint], profile_velocity=args.profile_velocity, goal_pwm=args.pwm)
            try:
                start = robot.read_state([args.joint])[args.joint]["position_rad"]
                step, tol = math.radians(args.step), 3 * TICK_RAD
                results = []
                for label, end in (("lower", lo), ("upper", hi)):
                    ok, reached, peak = sweep(robot, args.joint, start, end, step, args.timeout, tol)
                    results.append((label, end, ok, reached, peak))
                    # Goal = where the joint really is (no jump if it stopped short), then step back.
                    robot.hold()
                    back, now, _ = sweep(robot, args.joint, reached, start, step, args.timeout, tol)
                    if not back:
                        raise RuntimeError(f"{args.joint}: could not return to the start, stopped at "
                                           f"{math.degrees(now):.1f} deg")
                for label, end, ok, reached, peak in results:
                    print(f"  {label}: target {math.degrees(end):7.1f} deg  reached {math.degrees(reached):7.1f} deg  "
                          f"{'ok' if ok else 'STOPPED SHORT'}  peak load/current raw {peak}")
            finally:
                robot.disarm()
            print("Torque off verified")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
