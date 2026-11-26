"""Joint table: names, motor IDs, direction, zero offset and limits, loaded from YAML.

Angle convention (sim/walk/ranges/JOINT_RANGES.md section 4):
    raw tick = 2048 + zero_offset + direction * angle_deg * 4096 / 360
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

from . import control_table as ct

Limits = Optional[Tuple[float, float]]


@dataclass(frozen=True)
class Joint:
    name: str
    motor_id: int
    model: str
    bus: str = "opencr"
    connector: str = ""          # wiring label only (TTL1/TTL2/TTL3 share one OpenCR bus)
    direction: int = 1           # -1 if the robot's positive direction is reversed
    zero_offset: int = 0         # ticks; zero pose reads 2048 + zero_offset
    soft_limits_rad: Limits = None
    work_limits_rad: Limits = None

    @property
    def zero_tick(self) -> int:
        return ct.CENTER_TICK + self.zero_offset

    def angle(self, tick: int) -> float:
        return self.direction * (tick - self.zero_tick) * ct.TICK_RAD

    def tick(self, angle: float) -> int:
        return self.zero_tick + round(self.direction * angle / ct.TICK_RAD)

    def velocity(self, raw: int) -> float:
        return self.direction * raw * ct.VELOCITY_UNIT_RAD_S

    def dxl_position_limits(self) -> Optional[Tuple[int, int]]:
        """Min/Max Position Limit(52/48) register values for the software limits."""
        if self.soft_limits_rad is None:
            return None
        a, b = (self.tick(angle) for angle in self.soft_limits_rad)
        return (min(a, b), max(a, b))


@dataclass
class PairLimit:
    """lo <= coef[0] * q[a] + coef[1] * q[b] <= hi (radians)."""
    name: str
    a: str
    b: str
    coef: Tuple[float, float]
    range_rad: Tuple[float, float]
    enabled: bool = False


@dataclass
class RobotConfig:
    path: str
    buses: Dict[str, Dict]
    joints: "OrderedDict[str, Joint]"
    pair_limits: List[PairLimit] = field(default_factory=list)
    safety: Dict = field(default_factory=dict)
    require_soft_limits: bool = True


def _limits(value, label) -> Limits:
    if value is None:
        return None
    lo, hi = (float(v) for v in value)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError(f"{label}: invalid limits {value}")
    return (lo, hi)


def load_config(path: str) -> RobotConfig:
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    buses = raw.get("buses") or {}
    joints: "OrderedDict[str, Joint]" = OrderedDict()
    for name, item in (raw.get("joints") or {}).items():
        joint = Joint(
            name=name,
            motor_id=int(item["id"]),
            model=item["model"],
            bus=item.get("bus", "opencr"),
            connector=item.get("connector", ""),
            direction=int(item.get("direction", 1)),
            zero_offset=int(item.get("zero_offset", 0)),
            soft_limits_rad=_limits(item.get("soft_limit_rad"), f"{name}.soft_limit_rad"),
            work_limits_rad=_limits(item.get("work_limit_rad"), f"{name}.work_limit_rad"),
        )
        joints[name] = joint
    pairs = [PairLimit(name=p["name"], a=p["a"], b=p["b"], coef=tuple(p["coef"]),
                       range_rad=_limits(p["range_rad"], p["name"]), enabled=bool(p.get("enabled", False)))
             for p in raw.get("pair_limits") or []]
    config = RobotConfig(path=path, buses=buses, joints=joints, pair_limits=pairs,
                         safety=raw.get("safety") or {},
                         require_soft_limits=bool(raw.get("require_soft_limits", True)))
    validate(config)
    return config


def validate(config: RobotConfig):
    if not config.joints:
        raise ValueError(f"{config.path}: no joints")
    seen: Dict[Tuple[str, int], str] = {}
    for joint in config.joints.values():
        if joint.bus not in config.buses:
            raise ValueError(f"{joint.name}: unknown bus {joint.bus!r}")
        port = config.buses[joint.bus]["port"]
        if (port, joint.motor_id) in seen:
            raise ValueError(f"{joint.name}: ID {joint.motor_id} already used by {seen[port, joint.motor_id]} on {port}")
        seen[port, joint.motor_id] = joint.name
        if not 0 <= joint.motor_id <= 252 or joint.direction not in (-1, 1):
            raise ValueError(f"{joint.name}: invalid motor ID or direction")
        if joint.model not in ct.MODELS:
            raise ValueError(f"{joint.name}: unknown model {joint.model!r}")
        soft, work = joint.soft_limits_rad, joint.work_limits_rad
        if work is not None and soft is not None and not (soft[0] <= work[0] and work[1] <= soft[1]):
            raise ValueError(f"{joint.name}: work range must be inside the software limits")
        if soft is not None:
            lo, hi = joint.dxl_position_limits()
            if not (0 <= lo and hi <= 4095):
                raise ValueError(f"{joint.name}: limits map to ticks {lo}..{hi}, outside 0..4095; "
                                 "re-mount the horn or change zero_offset")
    for pair in config.pair_limits:
        for name in (pair.a, pair.b):
            if name not in config.joints:
                raise ValueError(f"pair limit {pair.name}: unknown joint {name}")


def update_zero_offsets(path: str, offsets: Dict[str, int], note: str = "record_zero.py"):
    """Rewrite only the `zero_offset:` line of each joint, keeping comments and layout."""
    with open(path, encoding="utf-8") as file:
        lines = file.read().split("\n")
    pending = dict(offsets)
    current = None
    for index, line in enumerate(lines):
        header = re.match(r"^  ([A-Za-z_][\w]*):\s*(#.*)?$", line)
        if header:
            current = header.group(1)
            continue
        if re.match(r"^\S", line):
            current = None
            continue
        match = re.match(r"^(\s{4}zero_offset:\s*)(-?\d+)(.*)$", line)
        if match and current in pending:
            lines[index] = f"{match.group(1)}{int(pending.pop(current))}  # {note}"
    if pending:
        raise ValueError(f"zero_offset line not found for: {', '.join(pending)}")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
