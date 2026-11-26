"""Read-only pose print, or --demo: move about 5 degrees and return (legacy/XL430_robot_base.py).

  python tools/read_demo.py                          # read only, bench config (ID 6, 7, 12)
  python tools/read_demo.py --demo --port /dev/ttyACM0
  python tools/read_demo.py --config config/joints_v20.yaml --joints R_knee
  python tools/read_demo.py --rate 500               # how many sync reads per second (torque stays off)

The demo is for unloaded bench motors. Configure the joints before attaching links.
"""

from __future__ import annotations

import math
import time

import _common
from kitoru_hw.control_table import TICK_RAD


def print_positions(label, state, started):
    positions = ", ".join(f"{name}={math.degrees(joint['position_rad']):.2f} deg "
                          f"({joint['position_tick']} ticks)" for name, joint in state.items())
    print(f"{time.monotonic() - started:.3f}s {label}: {positions}", flush=True)


def wait_for_pose(robot, pose, timeout=2.0, period=0.05):
    deadline = time.monotonic() + timeout
    while True:
        state = robot.read_state(list(pose))
        if all(abs(state[name]["position_rad"] - angle) <= 3 * TICK_RAD for name, angle in pose.items()):
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError("One or more joints did not reach the requested pose")
        time.sleep(period)


def run_demo(robot, started, period=0.05):
    robot.arm()
    try:
        initial = robot.read_state()
        pose_a = {name: joint["position_rad"] for name, joint in initial.items()}
        pose_b = {}
        for index, (name, angle) in enumerate(pose_a.items()):
            delta = math.radians(5 if index % 2 == 0 else -5)
            for candidate in (angle + delta, angle - delta):
                try:
                    robot.target_tick(name, candidate)
                except ValueError:
                    continue
                pose_b[name] = candidate
                break
            else:
                raise RuntimeError(f"{name}: no room for a 5 degree move within the limits")

        print_positions("A", initial, started)
        for label, pose in (("B", pose_b), ("A", pose_a)):
            goal = ", ".join(f"{name}={math.degrees(angle):.2f} deg" for name, angle in pose.items())
            print(f"{time.monotonic() - started:.3f}s goal {label}: {goal}", flush=True)
            robot.command_positions(pose)
            print_positions(f"measured {label}", wait_for_pose(robot, pose, period=period), started)
    finally:
        robot.disarm()
    print("Torque off verified for all motors")


def main(argv=None, **robot_kwargs):
    parser = _common.parser(__doc__.splitlines()[0], default_config=_common.BENCH_CONFIG)
    parser.add_argument("--demo", action="store_true", help="move the unloaded bench motors A -> B -> A")
    parser.add_argument("--rate", type=int, default=0, help="time this many state reads and print Hz")
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        with _common.make_robot(args, **robot_kwargs) as robot:
            if args.demo:
                run_demo(robot, started)
            elif args.rate:
                t0 = time.monotonic()
                for _ in range(args.rate):
                    robot.read_state()
                dt = time.monotonic() - t0
                print(f"{args.rate} sync reads of {len(robot.joints)} joints: {dt / args.rate * 1000:.2f} ms each, "
                      f"{args.rate / dt:.0f} Hz", flush=True)
            else:
                print_positions("position", robot.read_state(), started)
    except (OSError, RuntimeError, ValueError) as error:  # DxlError is a RuntimeError
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
