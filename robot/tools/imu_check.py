"""Check the OpenCR IMU (bridge firmware virtual ID): ping, live axes, read timing. Torque is never touched.

  python tools/imu_check.py --port /dev/kitoru_opencr                 # PING 200, then 10 Hz for 10 s
  python tools/imu_check.py --port /dev/kitoru_opencr --no-servos     # servo power off: IMU only
  python tools/imu_check.py --port /dev/kitoru_opencr --timing 500    # servo sync read vs IMU read, ms
  python tools/imu_check.py --port /dev/kitoru_opencr --calibrate     # re-measure gyro bias (keep still)

Each line shows gravity and angular velocity in the body frame (what the policy gets) and,
in brackets, the chip-frame values (to fix imu.r_board_chip in the YAML): g_chip from the filter,
a_chip = accelerometer direction (points UP, no filter lag). README 3절.
"""

from __future__ import annotations

import math
import time

import _common
from kitoru_hw.opencr_imu import OpenCRImu


def fmt(v, digits=2):
    return "(" + ", ".join(f"{x:+.{digits}f}" for x in v) + ")"


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def main(argv=None, **robot_kwargs):
    parser = _common.parser(__doc__.splitlines()[0])
    parser.add_argument("--hz", type=float, default=10.0, help="print rate")
    parser.add_argument("--seconds", type=float, default=10.0, help="how long to print (0: only ping)")
    parser.add_argument("--timing", type=int, default=0, help="time this many servo reads and IMU reads")
    parser.add_argument("--calibrate", action="store_true", help="re-measure the gyro bias first (keep still)")
    parser.add_argument("--no-servos", action="store_true", help="do not ping or read servos (servo power off)")
    args = parser.parse_args(argv)
    try:
        robot = _common.make_robot(args, **robot_kwargs)
        bus = next(iter(robot.buses.values()))
        bus.open(check_models=not args.no_servos)
        try:
            imu = OpenCRImu.from_config(bus, robot.config.imu).open()
            print(f"ID {imu.virtual_id}: OpenCR IMU firmware {imu.firmware_version}, {imu.config}", flush=True)
            if args.calibrate:
                print("gyro calibration: keep the robot still ...", flush=True)
                print("  done" if imu.calibrate_gyro() else "  still running after 6 s", flush=True)
            if args.timing:
                imu.diagnostics(reset=True)
                servo_ms, imu_ms = [], []
                for _ in range(args.timing):
                    if not args.no_servos:
                        t0 = time.perf_counter()
                        robot.read_state()
                        servo_ms.append((time.perf_counter() - t0) * 1000)
                    t0 = time.perf_counter()
                    imu.read_raw()
                    imu_ms.append((time.perf_counter() - t0) * 1000)
                for label, ms in (("servo sync read", servo_ms), ("IMU read", imu_ms)):
                    if ms:
                        print(f"{label}: mean {sum(ms) / len(ms):.2f} ms, max {max(ms):.2f} ms ({len(ms)} reads)")
                print(f"firmware: {imu.diagnostics()}  (loop_max_us = longest relay pause)", flush=True)
            end = time.monotonic() + args.seconds
            while time.monotonic() < end:
                s = imu.read()
                roll, pitch = (math.degrees(a) for a in s["roll_pitch"])
                print(f"g_body {fmt(s['gravity_b'])} w_body {fmt(s['ang_vel_b'])} rad/s "
                      f"roll {roll:+6.1f} pitch {pitch:+6.1f} deg "
                      f"[g_chip {fmt(s['gravity_chip'])} a_chip {fmt(unit(s['acc_chip']))} w_chip {fmt(s['gyro_chip'])}]"
                      f"{' STALE' if s['stale'] else ''}", flush=True)
                time.sleep(max(0.0, 1.0 / args.hz - s["read_s"]))
        finally:
            bus.close()
    except (OSError, RuntimeError, ValueError) as error:  # DxlError is a RuntimeError
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
