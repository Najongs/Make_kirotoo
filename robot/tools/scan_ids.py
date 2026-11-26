"""Find every servo on the bus with one broadcast ping, and compare with the YAML.

  python tools/scan_ids.py --port /dev/ttyACM0
  python tools/scan_ids.py --port /dev/ttyACM0 --bauds 57600,1000000,2000000

New X430 servos come at ID 1, 57600 baud. Plug in one new servo at a time to give it an ID.
"""

from __future__ import annotations

import _common
from dynamixel_sdk import PortHandler
from kitoru_hw.dxl_bus import scan
from kitoru_hw.joints import load_config


def compare(found, config):
    expected = {j.motor_id: j for j in config.joints.values()}
    lines = []
    for motor_id, joint in sorted(expected.items()):
        if motor_id not in found:
            lines.append(f"  missing  ID {motor_id:3d} {joint.name} ({joint.model})")
        elif found[motor_id][0] != joint.model:
            lines.append(f"  MODEL?   ID {motor_id:3d} {joint.name}: YAML {joint.model}, servo {found[motor_id][0]}")
    for motor_id in sorted(set(found) - set(expected)):
        lines.append(f"  extra    ID {motor_id:3d} {found[motor_id][0]} (not in YAML)")
    return lines


def main(argv=None, port_factory=PortHandler):
    parser = _common.parser(__doc__.splitlines()[0])
    parser.add_argument("--bauds", default="", help="comma separated baud rates to try (default: YAML baud)")
    args = parser.parse_args(argv)
    config = _common.select(load_config(args.config), args.joints)
    bus = next(iter(config.buses.values()))
    port_name = args.port or bus["port"]
    bauds = [int(b) for b in args.bauds.split(",") if b] or [args.baud or int(bus.get("baud", 1_000_000))]
    port = port_factory(port_name)
    try:
        for baud in bauds:
            if not port.setBaudRate(baud):
                parser.exit(1, f"Error: could not open {port_name} at {baud}\n")
            found = scan(port)
            print(f"{port_name} @ {baud}: {len(found)} servo(s)")
            for motor_id, (model, firmware) in found.items():
                print(f"  ID {motor_id:3d}  {model}  firmware {firmware}")
            if baud == int(bus.get("baud", 1_000_000)):
                problems = compare(found, config)
                print("  matches the YAML" if not problems else "\n".join(problems))
    except OSError as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        if port.is_open:
            port.closePort()


if __name__ == "__main__":
    main()
