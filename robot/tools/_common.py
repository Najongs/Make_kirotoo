"""Shared argument handling for the tools (run from robot/: python tools/<name>.py)."""

from __future__ import annotations

import argparse
import os
import sys

ROBOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROBOT_DIR not in sys.path:
    sys.path.insert(0, ROBOT_DIR)

from kitoru_hw.joints import RobotConfig, load_config  # noqa: E402
from kitoru_hw.robot import Robot  # noqa: E402

V20_CONFIG = os.path.join(ROBOT_DIR, "config", "joints_v20.yaml")
BENCH_CONFIG = os.path.join(ROBOT_DIR, "config", "bench_legacy.yaml")


def parser(description: str, default_config: str = V20_CONFIG) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=default_config, help="joint YAML file")
    p.add_argument("--port", help="override the port in the YAML (COM3, /dev/ttyACM0, ...)")
    p.add_argument("--baud", type=int, help="override the baud rate in the YAML")
    p.add_argument("--joints", help="comma separated subset of joints")
    return p


def select(config: RobotConfig, joints_arg) -> RobotConfig:
    if not joints_arg:
        return config
    names = [n.strip() for n in joints_arg.split(",") if n.strip()]
    unknown = [n for n in names if n not in config.joints]
    if unknown:
        raise ValueError(f"unknown joints: {', '.join(unknown)}")
    config.joints = type(config.joints)((n, config.joints[n]) for n in names)
    config.pair_limits = [p for p in config.pair_limits if p.a in names and p.b in names]
    return config


def make_robot(args, **kwargs) -> Robot:
    config = select(load_config(args.config), args.joints)
    return Robot(config, port_override=args.port, baud_override=args.baud, **kwargs)
