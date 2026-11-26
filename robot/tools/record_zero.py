"""Record zero offsets: put the robot in the zero pose with torque off, read, save.

  python tools/record_zero.py                    # print offsets only
  python tools/record_zero.py --save             # write zero_offset into the YAML
  python tools/record_zero.py --joints R_knee --save

zero_offset = Present Position - 2048 (JOINT_RANGES.md section 4). Torque stays off; the
tool refuses to run if any selected servo has torque on.
"""

from __future__ import annotations

import math
from dataclasses import replace

import _common
from kitoru_hw import control_table as ct
from kitoru_hw.joints import update_zero_offsets

LARGE_OFFSET = round(30 / 360 * 4096)  # more than 30 deg: probably a horn one spline off


def main(argv=None, **robot_kwargs):
    parser = _common.parser(__doc__.splitlines()[0])
    parser.add_argument("--save", action="store_true", help="write the offsets into --config")
    parser.add_argument("--samples", type=int, default=5, help="readings to average")
    args = parser.parse_args(argv)
    try:
        with _common.make_robot(args, **robot_kwargs) as robot:
            on = [n for n in robot.joints if robot.read_register(n, "TORQUE_ENABLE")]
            if on:
                parser.exit(1, f"Error: torque is on for {', '.join(on)}; turn it off first\n")
            samples = [robot.read_state() for _ in range(max(1, args.samples))]
            offsets, rows, blocking = {}, [], []
            for name, joint in robot.joints.items():
                ticks = [s[name]["position_tick"] for s in samples]
                tick = round(sum(ticks) / len(ticks))
                offset = tick - ct.CENTER_TICK
                offsets[name] = offset
                new = replace(joint, zero_offset=offset)
                limits = new.dxl_position_limits()
                note = []
                if max(ticks) - min(ticks) > 2:
                    note.append(f"moving ({min(ticks)}..{max(ticks)})")
                    blocking.append(name)
                if abs(offset) > LARGE_OFFSET:
                    note.append("offset > 30 deg, check the horn")
                if limits and not (0 <= limits[0] and limits[1] <= 4095):
                    note.append("limits leave 0..4095")
                    blocking.append(name)
                degrees = math.degrees(offset * ct.TICK_RAD)
                rows.append(f"  {name:14s} tick {tick:5d}  offset {offset:+5d} ({degrees:+6.1f} deg)"
                            f"  Min/Max Position Limit {limits if limits else '-'}  {'; '.join(note)}")
            print("\n".join(rows))
        if args.save:
            if blocking:
                parser.exit(1, f"Error: not saved; fix {', '.join(sorted(set(blocking)))} first "
                            "(joint moving, or limits outside 0..4095)\n")
            update_zero_offsets(args.config, offsets)
            print(f"saved {len(offsets)} offsets to {args.config}")
            print("Next: check each joint's direction, then set calibrated: true in the YAML by hand")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
