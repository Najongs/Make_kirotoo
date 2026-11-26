"""Whole-robot controller: joints in radians on one or more Dynamixel buses.

Arming keeps the checks of the bench script (legacy/XL430_robot_base.py): position mode,
torque off, Status Return Level 2, no watchdog or hardware error, current pose inside
the limits, hold goal written before torque on, original RAM settings restored on disarm.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional

from . import control_table as ct
from .dxl_bus import DxlBus
from .joints import RobotConfig
from .safety import Event, SafetyMonitor, SafetyStop, apply_pair_limits, clip_to_limits

SAVED_REGISTERS = ("PROFILE_ACCELERATION", "PROFILE_VELOCITY", "GOAL_POSITION", "GOAL_PWM")


class Robot:
    def __init__(self, config: RobotConfig, port_handlers: Optional[Mapping[str, object]] = None,
                 settle_s: float = 0.25, port_override: Optional[str] = None, baud_override: Optional[int] = None):
        self.config = config
        self.joints = config.joints
        self.buses: Dict[str, DxlBus] = OrderedDict()
        port_handlers = port_handlers or {}
        for bus_name, bus_cfg in config.buses.items():
            motors = {j.name: (j.motor_id, j.model) for j in self.joints.values() if j.bus == bus_name}
            if not motors:
                continue
            port = port_override or bus_cfg["port"]
            baud = baud_override or int(bus_cfg.get("baud", 1_000_000))
            self.buses[bus_name] = DxlBus(port, baud, motors, port_handler=port_handlers.get(bus_name), settle_s=settle_s)
        self._limits: Dict[str, tuple] = {}
        self._saved: Dict[str, Dict[str, int]] = {}
        self.armed: List[str] = []
        self.monitor: Optional[SafetyMonitor] = None

    # ----- lifecycle ------------------------------------------------------
    def __enter__(self):
        opened = []
        try:
            for bus in self.buses.values():
                bus.open()
                opened.append(bus)
        except BaseException:
            for bus in opened:
                bus.close()
            raise
        return self

    def __exit__(self, *_):
        try:
            self.disarm()
        finally:
            for bus in self.buses.values():
                bus.close()

    def bus_of(self, name: str) -> DxlBus:
        return self.buses[self.joints[name].bus]

    def _group(self, names: Iterable[str]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = OrderedDict()
        for name in names:
            groups.setdefault(self.joints[name].bus, []).append(name)
        return groups

    # ----- reading --------------------------------------------------------
    def read_state(self, names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, float]]:
        """Position (rad), velocity (rad/s), raw tick and raw load/current for each joint."""
        names = list(self.joints if names is None else names)
        state = {}
        for bus_name, group in self._group(names).items():
            raw = self.buses[bus_name].read_state_raw(group)
            for name in group:
                joint, r = self.joints[name], raw[name]
                state[name] = {
                    "position_rad": joint.angle(r["position_tick"]),
                    "velocity_rad_s": joint.velocity(r["velocity"]),
                    "position_tick": r["position_tick"],
                    "load_or_current": joint.direction * r["load_or_current"],
                }
        return OrderedDict((name, state[name]) for name in names)

    def read_health(self, names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, float]]:
        names = list(self.joints if names is None else names)
        health = {}
        for bus_name, group in self._group(names).items():
            health.update(self.buses[bus_name].read_health(group))
        return health

    def read_register(self, name: str, item_name: str) -> int:
        return self.bus_of(name).read(name, item_name)

    def write_register(self, name: str, item_name: str, value: int):
        self.bus_of(name).write(name, item_name, value)

    # ----- safety ---------------------------------------------------------
    def start_monitor(self) -> List[Event]:
        """Read Temperature Limit(31) / Max Voltage Limit(32) from every servo and build the monitor."""
        from .safety import check_voltage_limits
        temps = {n: self.read_register(n, "TEMPERATURE_LIMIT") for n in self.joints}
        max_v = {n: self.read_register(n, "MAX_VOLTAGE_LIMIT") * ct.VOLTAGE_UNIT_V for n in self.joints}
        self.monitor = SafetyMonitor(temps, self.config.safety)
        return check_voltage_limits(max_v, self.config.safety.get("full_charge_v", 12.6))

    def safety_step(self, tilt_deg: Optional[float] = None, read_health: bool = True) -> List[Event]:
        """Evaluate the rules; lower Goal PWM on a fall, stop on low voltage / heat / hardware error."""
        if self.monitor is None:
            raise RuntimeError("Call start_monitor() first")
        events = self.monitor.evaluate(self.read_health() if read_health else None, tilt_deg)
        if any(e.kind == "fall" for e in events) and self.armed:
            self.set_goal_pwm(self.monitor.fall.reduced_pwm)
        stops = [e for e in events if e.level == "stop"]
        if stops:
            self.emergency_stop()
            raise SafetyStop("; ".join(e.message for e in stops))
        return events

    def set_goal_pwm(self, value: int, names: Optional[Iterable[str]] = None):
        """Goal PWM(100) limits the output in position mode; one sync write per bus."""
        names = list(self.armed if names is None else names)
        for bus_name, group in self._group(names).items():
            self.buses[bus_name].sync_write("GOAL_PWM", {name: value for name in group})

    def reset_fall(self):
        """After the robot is upright again: clear the latch and give back the normal Goal PWM."""
        self.monitor.fall.reset()
        if self.armed:
            self.set_goal_pwm(self.monitor.fall.normal_pwm)

    def emergency_stop(self) -> List[str]:
        """Broadcast torque off on every bus, then check each servo as far as the bus allows."""
        problems = []
        for bus_name, bus in self.buses.items():
            try:
                bus.torque_off_broadcast()
            except Exception as error:
                problems.append(f"{bus_name}: {error}")
        self.armed = []
        for name in self.joints:
            try:
                if self.read_register(name, "TORQUE_ENABLE"):
                    problems.append(f"{name}: torque still on")
            except Exception as error:
                problems.append(f"{name}: {error}")
        return problems

    # ----- targets --------------------------------------------------------
    def target_tick(self, name: str, angle: float) -> int:
        """Strict check used by tools: raises if outside software or servo position limits."""
        joint = self.joints[name]
        if not math.isfinite(angle):
            raise ValueError(f"{name}: angle must be finite")
        limits = joint.soft_limits_rad
        if limits is None and self.config.require_soft_limits:
            raise ValueError(f"{name}: no software limits configured")
        if limits is not None and not limits[0] <= angle <= limits[1]:
            raise ValueError(f"{name}: angle is outside the configured joint limits")
        tick = joint.tick(angle)
        lo, hi = self._limits[name]
        if not lo <= tick <= hi:
            raise ValueError(f"{name}: target {tick} ticks is outside motor limits {lo}..{hi}")
        return tick

    def command_positions(self, positions_rad: Mapping[str, float], use_work: bool = False) -> List[str]:
        """Clip to software and pair limits, then send one Sync Write per bus. Returns clipped names."""
        if not self.armed:
            raise RuntimeError("Call arm() before commanding motion")
        if set(positions_rad) != set(self.armed):
            raise ValueError(f"Provide a target for every armed joint: {', '.join(self.armed)}")
        clipped, changed = clip_to_limits(positions_rad, self.joints, use_work, self.config.require_soft_limits)
        clipped, pairs = apply_pair_limits(clipped, self.config.pair_limits, self.joints)
        ticks = {name: self.target_tick(name, angle) for name, angle in clipped.items()}
        for bus_name, group in self._group(ticks).items():
            self.buses[bus_name].sync_write("GOAL_POSITION", {name: ticks[name] for name in group})
        return changed + pairs

    # ----- arming ---------------------------------------------------------
    def arm(self, names: Optional[Iterable[str]] = None, profile_velocity: int = 15,
            profile_acceleration: int = 5, goal_pwm: Optional[int] = None):
        """Enable torque in existing Position Control Mode and hold the measured pose."""
        names = list(self.joints if names is None else names)
        if self.armed:
            raise RuntimeError("Torque is already enabled by this controller")
        saved = {}
        state = self.read_state(names)
        for name, motor in state.items():
            mode, drive, torque, status, watchdog, fault = [self.read_register(name, item) for item in (
                "OPERATING_MODE", "DRIVE_MODE", "TORQUE_ENABLE", "STATUS_RETURN_LEVEL", "BUS_WATCHDOG",
                "HARDWARE_ERROR_STATUS")]
            if mode != ct.POSITION_MODE or drive not in (0, 1) or torque or status != 2 or watchdog or fault:
                raise RuntimeError(f"{name}: test requires position mode, velocity profile, torque off, "
                                   "Status Return Level 2, and no watchdog or hardware error")
            lo = self.read_register(name, "MIN_POSITION_LIMIT")
            hi = self.read_register(name, "MAX_POSITION_LIMIT")
            if not 0 <= lo <= motor["position_tick"] <= hi <= 4095:
                raise RuntimeError(f"{name}: current position is outside the single-turn limits")
            self._limits[name] = (lo, hi)
            self.target_tick(name, motor["position_rad"])
            saved[name] = {item: self.read_register(name, item) for item in SAVED_REGISTERS}
            if not lo <= saved[name]["GOAL_POSITION"] <= hi:
                raise RuntimeError(f"{name}: original goal position cannot be restored")

        self._saved = saved
        self.armed = list(names)
        try:
            for name, motor in state.items():
                self.write_register(name, "PROFILE_ACCELERATION", profile_acceleration)
                self.write_register(name, "PROFILE_VELOCITY", profile_velocity)
                if goal_pwm is not None:
                    self.write_register(name, "GOAL_PWM", goal_pwm)
                # Confirm the hold goal individually before turning on torque.
                self.write_register(name, "GOAL_POSITION", motor["position_tick"])
            for name in names:
                self.write_register(name, "TORQUE_ENABLE", 1)
            armed_state = self.read_state(names)
            for name, motor in state.items():
                if abs(armed_state[name]["position_tick"] - motor["position_tick"]) > 3:
                    raise RuntimeError(f"{name}: position shifted when torque was enabled")
        except BaseException:
            self.disarm()
            raise

    def disarm(self):
        """Disable torque on all armed joints and restore their original RAM settings."""
        self.armed = []
        failures = []
        for name in tuple(self._saved):
            try:
                self.write_register(name, "TORQUE_ENABLE", 0)
                if self.read_register(name, "TORQUE_ENABLE"):
                    raise RuntimeError("torque did not turn off")
            except Exception as error:
                failures.append(f"{name}: {error}")
                continue
            try:
                for item, value in self._saved[name].items():
                    self.write_register(name, item, value)
            except Exception as error:
                failures.append(f"{name}: could not restore settings: {error}")
            else:
                del self._saved[name]
        if failures:
            raise RuntimeError("Cleanup failed: " + "; ".join(failures))
