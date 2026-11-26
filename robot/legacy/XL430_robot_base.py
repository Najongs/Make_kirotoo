"""OpenCR + XL430 position control, shared by a manual demo or a policy.

  python XL430_robot_base.py                 # Read only
  python XL430_robot_base.py --demo          # Move about 5 degrees and return
  python XL430_robot_base.py --port /dev/ttyACM0

Install the dependency with: python -m pip install dynamixel-sdk
The demo is for unloaded bench motors. Configure JOINTS before attaching links.
"""

import argparse
import math
import time
from dataclasses import dataclass

from dynamixel_sdk import COMM_SUCCESS, GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler

TICK_RAD = math.tau / 4096
VELOCITY_UNIT_RAD_S = 0.229 * math.tau / 60
MODEL = 1060

# XL430-W250 control table (address, size is specified where used).
DRIVE_MODE, OPERATING_MODE = 10, 11
MAX_POSITION, MIN_POSITION = 48, 52
TORQUE, STATUS_LEVEL, HARDWARE_ERROR, WATCHDOG = 64, 68, 70, 98
PROFILE_ACCELERATION, PROFILE_VELOCITY, GOAL_POSITION = 108, 112, 116
PRESENT_VELOCITY, PRESENT_POSITION = 128, 132


@dataclass(frozen=True)
class Joint:
    motor_id: int
    zero_tick: int = 2048  # Placeholder until the mechanical zero is measured.
    direction: int = 1    # Set to -1 if the robot's positive direction is reversed.
    limits_rad: tuple[float, float] | None = None

    def angle(self, tick):
        return self.direction * (tick - self.zero_tick) * TICK_RAD

    def tick(self, angle):
        return self.zero_tick + round(self.direction * angle / TICK_RAD)


# Order is stable and can also be used by a simulator or policy.
JOINTS = {"joint_6": Joint(6), "joint_7": Joint(7), "joint_12": Joint(12)}


def signed32(value):
    return value - (1 << 32) if value >= 1 << 31 else value


class XL430Bus:
    """One owner of the serial port. Position commands use named joints in radians."""

    def __init__(self, port="COM3", baud=1_000_000, joints=None):
        self.joints = dict(JOINTS if joints is None else joints)
        ids = [joint.motor_id for joint in self.joints.values()]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Configure at least one joint and use unique motor IDs.")
        for name, joint in self.joints.items():
            if not 0 <= joint.motor_id <= 252 or joint.direction not in (-1, 1):
                raise ValueError(f"Invalid motor ID or direction for {name}")
            if joint.limits_rad is not None:
                lo, hi = joint.limits_rad
                if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                    raise ValueError(f"Invalid software limits for {name}")

        self.port = PortHandler(port)
        self.baud = baud
        self.packet = PacketHandler(2.0)
        self.sync_read = GroupSyncRead(self.port, self.packet, PRESENT_VELOCITY, 8)
        self.sync_write = GroupSyncWrite(self.port, self.packet, GOAL_POSITION, 4)
        for motor_id in ids:
            if not self.sync_read.addParam(motor_id):
                raise RuntimeError(f"Could not register ID {motor_id} for group read")
        self._limits = {}
        self._saved = {}
        self._armed = False

    def __enter__(self):
        if not self.port.setBaudRate(self.baud):
            raise RuntimeError(f"Could not open {self.port.port_name} at {self.baud} baud")
        try:
            time.sleep(0.25)
            for name, joint in self.joints.items():
                model, result, error = self.packet.ping(self.port, joint.motor_id)
                self._check(result, error, f"Ping {name}")
                if model != MODEL:
                    raise RuntimeError(f"{name}: expected XL430 model {MODEL}, got {model}")
        except BaseException:
            self.port.closePort()
            raise
        return self

    def __exit__(self, *_):
        try:
            self.disarm()
        finally:
            self.port.closePort()

    def _check(self, result, error=0, operation="DYNAMIXEL"):
        if result != COMM_SUCCESS:
            raise RuntimeError(f"{operation}: {self.packet.getTxRxResult(result)}")
        if error:
            raise RuntimeError(f"{operation}: {self.packet.getRxPacketError(error)}")

    def _read(self, name, address, size=1):
        reader = self.packet.read1ByteTxRx if size == 1 else self.packet.read4ByteTxRx
        value, result, error = reader(self.port, self.joints[name].motor_id, address)
        self._check(result, error, f"Read {name} at {address}")
        return value

    def _write(self, name, address, value, size=1):
        writer = self.packet.write1ByteTxRx if size == 1 else self.packet.write4ByteTxRx
        result, error = writer(self.port, self.joints[name].motor_id, address, value)
        self._check(result, error, f"Write {name} at {address}")

    def read_state(self):
        """Return one sample with per-joint position and velocity in radians and rad/s."""
        self._check(self.sync_read.txRxPacket(), operation="Group read")
        state = {}
        for name, joint in self.joints.items():
            motor_id = joint.motor_id
            if not self.sync_read.isAvailable(motor_id, PRESENT_VELOCITY, 8):
                raise RuntimeError(f"No position/velocity feedback for {name} (ID {motor_id})")
            tick = signed32(self.sync_read.getData(motor_id, PRESENT_POSITION, 4))
            velocity = signed32(self.sync_read.getData(motor_id, PRESENT_VELOCITY, 4))
            state[name] = {
                "position_rad": joint.angle(tick),
                "velocity_rad_s": joint.direction * velocity * VELOCITY_UNIT_RAD_S,
                "position_tick": tick,
            }
        return state

    def _target_tick(self, name, angle):
        joint = self.joints[name]
        if not math.isfinite(angle):
            raise ValueError(f"{name}: angle must be finite")
        if joint.limits_rad is not None and not joint.limits_rad[0] <= angle <= joint.limits_rad[1]:
            raise ValueError(f"{name}: angle is outside the configured joint limits")
        tick = joint.tick(angle)
        lo, hi = self._limits[name]
        if not lo <= tick <= hi:
            raise ValueError(f"{name}: target {tick} ticks is outside motor limits {lo}..{hi}")
        return tick

    def _send_ticks(self, targets):
        try:
            for name, tick in targets.items():
                data = list(tick.to_bytes(4, byteorder="little"))
                if not self.sync_write.addParam(self.joints[name].motor_id, data):
                    raise RuntimeError(f"Could not register target for {name}")
            self._check(self.sync_write.txPacket(), operation="Group write")
        finally:
            self.sync_write.clearParam()

    def arm(self):
        """Enable torque in existing Position Control Mode and hold the measured pose."""
        if self._armed:
            raise RuntimeError("Torque is already enabled by this controller")
        saved = {}
        state = self.read_state()
        for name, motor in state.items():
            mode, drive, torque, status, watchdog, fault = [self._read(name, address) for address in
                (OPERATING_MODE, DRIVE_MODE, TORQUE, STATUS_LEVEL, WATCHDOG, HARDWARE_ERROR)]
            if mode != 3 or drive not in (0, 1) or torque or status != 2 or watchdog or fault:
                raise RuntimeError(f"{name}: test requires position mode, velocity profile, torque off, "
                                   "Status Return Level 2, and no watchdog or hardware error")
            lo = self._read(name, MIN_POSITION, 4)
            hi = self._read(name, MAX_POSITION, 4)
            if not 0 <= lo <= motor["position_tick"] <= hi <= 4095:
                raise RuntimeError(f"{name}: current position is outside the single-turn limits")
            self._limits[name] = (lo, hi)
            self._target_tick(name, motor["position_rad"])
            registers = (PROFILE_ACCELERATION, PROFILE_VELOCITY, GOAL_POSITION)
            saved[name] = {address: self._read(name, address, 4) for address in registers}
            if not lo <= saved[name][GOAL_POSITION] <= hi:
                raise RuntimeError(f"{name}: original goal position cannot be restored")

        self._saved = saved
        try:
            for name, motor in state.items():
                self._write(name, PROFILE_ACCELERATION, 5, 4)
                self._write(name, PROFILE_VELOCITY, 15, 4)
                # Confirm the hold goal individually before turning on torque.
                self._write(name, GOAL_POSITION, motor["position_tick"], 4)
            for name in self.joints:
                self._write(name, TORQUE, 1)
            armed_state = self.read_state()
            for name, motor in state.items():
                if abs(armed_state[name]["position_tick"] - motor["position_tick"]) > 3:
                    raise RuntimeError(f"{name}: position shifted when torque was enabled")
            self._armed = True
        except BaseException:
            self.disarm()
            raise

    def command_positions(self, positions_rad):
        """Send a complete pose in a single packet. Verify actual motion via read_state()."""
        if not self._armed:
            raise RuntimeError("Call arm() before commanding motion")
        if positions_rad.keys() != self.joints.keys():
            raise ValueError(f"Provide a target for every joint: {', '.join(self.joints)}")
        targets = {name: self._target_tick(name, angle) for name, angle in positions_rad.items()}
        self._send_ticks(targets)

    def disarm(self):
        """Disable torque on all armed joints and restore their original RAM settings."""
        self._armed = False
        failures = []
        for name in tuple(self._saved):
            try:
                self._write(name, TORQUE, 0)
                if self._read(name, TORQUE):
                    raise RuntimeError("torque did not turn off")
            except Exception as error:
                failures.append(f"{name}: {error}")
                continue
            try:
                for address, value in self._saved[name].items():
                    self._write(name, address, value, 4)
            except Exception as error:
                failures.append(f"{name}: could not restore settings: {error}")
            else:
                del self._saved[name]
        if failures:
            raise RuntimeError("Cleanup failed: " + "; ".join(failures))


def print_positions(label, state, started):
    positions = ", ".join(f"{name}={math.degrees(joint['position_rad']):.2f} deg "
                          f"({joint['position_tick']} ticks)" for name, joint in state.items())
    print(f"{time.monotonic() - started:.3f}s {label}: {positions}", flush=True)


def wait_for_pose(bus, pose, timeout=2.0):
    deadline = time.monotonic() + timeout
    while True:
        state = bus.read_state()
        if all(abs(state[name]["position_rad"] - angle) <= 3 * TICK_RAD for name, angle in pose.items()):
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError("One or more joints did not reach the requested pose")
        time.sleep(0.05)


def run_demo(bus, started):
    bus.arm()
    try:
        initial = bus.read_state()
        pose_a = {name: joint["position_rad"] for name, joint in initial.items()}
        pose_b = {}
        for index, (name, angle) in enumerate(pose_a.items()):
            delta = math.radians(5 if index % 2 == 0 else -5)
            for candidate in (angle + delta, angle - delta):
                try:
                    bus._target_tick(name, candidate)
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
            bus.command_positions(pose)
            print_positions(f"measured {label}", wait_for_pose(bus, pose), started)
    finally:
        bus.disarm()
    print("Torque off verified for all motors")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="COM3", help="COM port or Linux serial device")
    parser.add_argument("--baud", default=1_000_000, type=int)
    parser.add_argument("--demo", action="store_true", help="move the unloaded bench motors A -> B -> A")
    args = parser.parse_args()
    started = time.monotonic()
    try:
        with XL430Bus(args.port, args.baud) as bus:
            if args.demo:
                run_demo(bus, started)
            else:
                print_positions("position", bus.read_state(), started)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
