"""Safety layer: command clipping, pair limits, fall PWM hook, voltage and temperature.

Thresholds come from sim/walk/battery6000/BATTERY_6000.md:
  - body tilt > 25 deg -> Goal PWM 885 * 0.3 = 265 (Goal PWM(100) is RAM and limits the
    PWM in position mode; PWM Limit(36) is EEPROM and cannot change while torque is on)
  - stop below 10.8 V (3.6 V per cell)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .joints import Joint, PairLimit


class SafetyStop(RuntimeError):
    """Raised after the robot was stopped by a safety rule."""


@dataclass
class Event:
    level: str      # "warn", "reduce" or "stop"
    kind: str       # "fall", "voltage", "temperature", "hardware"
    message: str


def clip_to_limits(targets: Mapping[str, float], joints: Mapping[str, Joint],
                   use_work: bool = False, require: bool = True) -> Tuple[Dict[str, float], List[str]]:
    """Clamp each target to its software (or work) range. Returns (clipped, names that changed)."""
    out, changed = {}, []
    for name, angle in targets.items():
        if not math.isfinite(angle):
            raise ValueError(f"{name}: angle must be finite")
        joint = joints[name]
        limits = joint.work_limits_rad if use_work and joint.work_limits_rad else joint.soft_limits_rad
        if limits is None:
            if require:
                raise ValueError(f"{name}: no software limits configured, refusing to command it")
            out[name] = angle
            continue
        clipped = min(max(angle, limits[0]), limits[1])
        if clipped != angle:
            changed.append(name)
        out[name] = clipped
    return out, changed


def apply_pair_limits(targets: Dict[str, float], pairs: Iterable[PairLimit],
                      joints: Mapping[str, Joint], iterations: int = 3) -> Tuple[Dict[str, float], List[str]]:
    """Move both joints of a violated pair by the smallest amount, then re-clip to single limits."""
    out = dict(targets)
    hit = []
    for _ in range(iterations):
        moved = False
        for pair in pairs:
            if not pair.enabled or pair.a not in out or pair.b not in out:
                continue
            ca, cb = pair.coef
            value = ca * out[pair.a] + cb * out[pair.b]
            lo, hi = pair.range_rad
            excess = value - hi if value > hi else value - lo if value < lo else 0.0
            if excess == 0.0:
                continue
            norm = ca * ca + cb * cb
            out[pair.a] -= excess * ca / norm
            out[pair.b] -= excess * cb / norm
            moved = True
            if pair.name not in hit:
                hit.append(pair.name)
        out, _ = clip_to_limits(out, joints, require=False)
        if not moved:
            break
    for pair in pairs:
        if pair.enabled and pair.a in out and pair.b in out:
            value = pair.coef[0] * out[pair.a] + pair.coef[1] * out[pair.b]
            if not pair.range_rad[0] - 1e-9 <= value <= pair.range_rad[1] + 1e-9:
                raise ValueError(f"pair limit {pair.name} cannot be met inside the single-joint limits")
    return out, hit


def tilt_deg(roll_rad: float, pitch_rad: float) -> float:
    """Angle between the body z axis and vertical."""
    c = math.cos(roll_rad) * math.cos(pitch_rad)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class FallGuard:
    """Reports a fall every cycle until the caller confirms the low Goal PWM was written.

    `fallen` is set by mark_applied() only after the write succeeded, so a failed write
    is retried on the next cycle. It stays set until reset().
    """

    def __init__(self, threshold_deg: float = 25.0, reduced_pwm: int = 265, normal_pwm: int = 885):
        self.threshold_deg = threshold_deg
        self.reduced_pwm = reduced_pwm
        self.normal_pwm = normal_pwm
        self.fallen = False

    def update(self, tilt: Optional[float]) -> Optional[Event]:
        if tilt is None or self.fallen or tilt <= self.threshold_deg:
            return None
        return Event("reduce", "fall", f"tilt {tilt:.1f} deg > {self.threshold_deg} deg, "
                     f"Goal PWM -> {self.reduced_pwm}")

    def mark_applied(self):
        self.fallen = True

    def reset(self):
        self.fallen = False


class VoltageGuard:
    """Median of all servo input voltages; stop after `debounce` low samples in a row."""

    def __init__(self, stop_v: float = 10.8, warn_v: float = 11.1, debounce: int = 3):
        self.stop_v, self.warn_v, self.debounce = stop_v, warn_v, debounce
        self.low_count = 0

    def update(self, voltages: Iterable[float]) -> Optional[Event]:
        voltages = list(voltages)
        if not voltages:
            return None
        median = statistics.median(voltages)
        if median < self.stop_v:
            self.low_count += 1
            if self.low_count >= self.debounce:
                return Event("stop", "voltage", f"battery {median:.1f} V < {self.stop_v} V "
                             f"for {self.low_count} samples")
            return Event("warn", "voltage", f"battery {median:.1f} V below stop level "
                         f"({self.low_count}/{self.debounce})")
        self.low_count = 0
        if median < self.warn_v:
            return Event("warn", "voltage", f"battery {median:.1f} V < {self.warn_v} V")
        return None


def check_voltage_limits(max_limits_v: Mapping[str, float], full_charge_v: float = 12.6) -> List[Event]:
    """Max Voltage Limit(32) below a full pack makes servos raise an input voltage error."""
    return [Event("warn", "voltage", f"{name}: Max Voltage Limit {limit:.1f} V < full charge {full_charge_v} V")
            for name, limit in max_limits_v.items() if limit < full_charge_v]


class TemperatureGuard:
    """Per-servo Temperature Limit(31) read from the servo (XL430 72 C, XC/XM 80 C by default)."""

    def __init__(self, limits_c: Mapping[str, float], warn_margin_c: float = 10.0, stop_margin_c: float = 3.0):
        self.limits_c = dict(limits_c)
        self.warn_margin_c, self.stop_margin_c = warn_margin_c, stop_margin_c

    def update(self, temperatures: Mapping[str, float]) -> List[Event]:
        events = []
        for name, temp in temperatures.items():
            limit = self.limits_c[name]
            if temp >= limit - self.stop_margin_c:
                events.append(Event("stop", "temperature", f"{name}: {temp} C, limit {limit} C"))
            elif temp >= limit - self.warn_margin_c:
                events.append(Event("warn", "temperature", f"{name}: {temp} C, limit {limit} C"))
        return events


class SafetyMonitor:
    def __init__(self, temperature_limits_c: Mapping[str, float], settings: Optional[Mapping] = None):
        s = dict(settings or {})
        self.fall = FallGuard(s.get("fall_tilt_deg", 25.0), s.get("fall_goal_pwm", 265), s.get("normal_goal_pwm", 885))
        self.voltage = VoltageGuard(s.get("low_voltage_stop_v", 10.8), s.get("low_voltage_warn_v", 11.1),
                                    s.get("voltage_debounce", 3))
        self.temperature = TemperatureGuard(temperature_limits_c, s.get("temp_warn_margin_c", 10.0),
                                            s.get("temp_stop_margin_c", 3.0))

    def evaluate(self, health: Optional[Mapping[str, Mapping[str, float]]] = None,
                 tilt: Optional[float] = None) -> List[Event]:
        events = []
        fall = self.fall.update(tilt)
        if fall:
            events.append(fall)
        if health:
            voltage = self.voltage.update(h["voltage_v"] for h in health.values())
            if voltage:
                events.append(voltage)
            events += self.temperature.update({n: h["temperature_c"] for n, h in health.items()})
            events += [Event("stop", "hardware", f"{n}: hardware error status {h['hardware_error']}")
                       for n, h in health.items() if h.get("hardware_error")]
        return events
