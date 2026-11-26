"""X430 series (Protocol 2.0) control table used by Kitoru.

Values were checked on the ROBOTIS e-manual pages for XL430-W250, XC430-W240 and
XM430-W350 on 2026-11-26. The three models share every address used here except
126 (XL/XC: Present Load in 0.1 %, XM: Present Current in 2.69 mA) and the current
registers 38 / 102, which only XM430 has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

# Units.
TICK_RAD = math.tau / 4096                 # Present/Goal Position, 1 pulse
VELOCITY_UNIT_RAD_S = 0.229 * math.tau / 60  # Present Velocity, 0.229 rev/min
VOLTAGE_UNIT_V = 0.1                       # Present Input Voltage, voltage limits
PWM_UNIT_PERCENT = 0.113                   # Goal PWM, PWM Limit, Present PWM
LOAD_UNIT_PERCENT = 0.1                    # XL430/XC430 Present Load(126)
CURRENT_UNIT_A = 0.00269                   # XM430 Present Current(126), Current Limit(38)

CENTER_TICK = 2048
BROADCAST_ID = 254

# name: (address, size in bytes, signed, eeprom)
# EEPROM items can only be written while Torque Enable(64) is 0.
ITEMS: Dict[str, Tuple[int, int, bool, bool]] = {
    "MODEL_NUMBER": (0, 2, False, True),
    "FIRMWARE_VERSION": (6, 1, False, True),
    "ID": (7, 1, False, True),
    "BAUD_RATE": (8, 1, False, True),
    "RETURN_DELAY_TIME": (9, 1, False, True),   # 2 us per unit, default 250 (= 500 us)
    "DRIVE_MODE": (10, 1, False, True),
    "OPERATING_MODE": (11, 1, False, True),
    "HOMING_OFFSET": (20, 4, True, True),
    "TEMPERATURE_LIMIT": (31, 1, False, True),
    "MAX_VOLTAGE_LIMIT": (32, 2, False, True),
    "MIN_VOLTAGE_LIMIT": (34, 2, False, True),
    "PWM_LIMIT": (36, 2, False, True),
    "CURRENT_LIMIT": (38, 2, False, True),      # XM430 only
    "VELOCITY_LIMIT": (44, 4, False, True),
    "MAX_POSITION_LIMIT": (48, 4, False, True),
    "MIN_POSITION_LIMIT": (52, 4, False, True),
    "TORQUE_ENABLE": (64, 1, False, False),
    "LED": (65, 1, False, False),
    "STATUS_RETURN_LEVEL": (68, 1, False, False),
    "HARDWARE_ERROR_STATUS": (70, 1, False, False),
    "BUS_WATCHDOG": (98, 1, True, False),       # 20 ms per unit, reads -1 after it fired
    "GOAL_PWM": (100, 2, True, False),          # limits PWM in position mode too
    "GOAL_CURRENT": (102, 2, True, False),      # XM430 only
    "PROFILE_ACCELERATION": (108, 4, False, False),
    "PROFILE_VELOCITY": (112, 4, False, False),
    "GOAL_POSITION": (116, 4, False, False),
    "PRESENT_PWM": (124, 2, True, False),
    "PRESENT_LOAD_OR_CURRENT": (126, 2, True, False),
    "PRESENT_VELOCITY": (128, 4, True, False),
    "PRESENT_POSITION": (132, 4, True, False),
    "PRESENT_INPUT_VOLTAGE": (144, 2, False, False),
    "PRESENT_TEMPERATURE": (146, 1, False, False),
}

# Fast loop block: Present Load/Current(126) .. Present Position(132+4).
STATE_BLOCK = (126, 10)
# Slow loop block: Present Input Voltage(144) .. Present Temperature(146+1).
HEALTH_BLOCK = (144, 3)

POSITION_MODE = 3
OPERATING_MODES = {0: "current", 1: "velocity", 3: "position", 4: "extended position",
                   5: "current-based position", 16: "PWM"}

# Baud Rate(8) register value for each bit rate (same on all three models).
BAUD_RATE_INDEX = {9600: 0, 57600: 1, 115200: 2, 1_000_000: 3, 2_000_000: 4,
                   3_000_000: 5, 4_000_000: 6, 4_500_000: 7}

# [확인 필요] bit names not re-checked on the e-manual this time (only used in messages).
HARDWARE_ERROR_BITS = {0: "input voltage", 2: "overheating", 3: "motor encoder",
                       4: "electrical shock", 5: "overload"}


@dataclass(frozen=True)
class Model:
    name: str
    number: int
    has_current: bool          # Present Current(126) instead of Present Load
    temperature_limit_c: int   # factory default of Temperature Limit(31)
    max_voltage_v: float       # factory default of Max Voltage Limit(32)
    min_voltage_v: float       # factory default of Min Voltage Limit(34)
    velocity_limit: int        # factory default of Velocity Limit(44), 0.229 rev/min


MODELS = {
    m.name: m for m in (
        Model("XL430-W250", 1060, False, 72, 14.0, 6.0, 265),
        Model("XC430-W240", 1080, False, 80, 16.0, 6.0, 306),
        Model("XM430-W350", 1020, True, 80, 16.0, 9.5, 200),
    )
}
MODELS_BY_NUMBER = {m.number: m for m in MODELS.values()}


def item(name: str, model: str = "") -> Tuple[int, int, bool, bool]:
    if name in ("CURRENT_LIMIT", "GOAL_CURRENT") and model and not MODELS[model].has_current:
        raise ValueError(f"{model} has no {name}")
    return ITEMS[name]


def to_signed(value: int, size: int) -> int:
    bits = 8 * size
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


def to_unsigned(value: int, size: int) -> int:
    bits = 8 * size
    if not -(1 << (bits - 1)) <= value < (1 << bits):
        raise ValueError(f"{value} does not fit in {size} bytes")
    return value & ((1 << bits) - 1)


def describe_hardware_error(status: int) -> str:
    names = [label for bit, label in HARDWARE_ERROR_BITS.items() if status >> bit & 1]
    return ", ".join(names) if names else "none"
