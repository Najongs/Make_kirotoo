import os
import sys

import pytest

ROBOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROBOT_DIR, os.path.join(ROBOT_DIR, "tools"), os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from fake_dxl import FakePort, FakeServo  # noqa: E402
from kitoru_hw.joints import load_config  # noqa: E402
from kitoru_hw.robot import Robot  # noqa: E402

CONFIG_DIR = os.path.join(ROBOT_DIR, "config")
V20 = os.path.join(CONFIG_DIR, "joints_v20.yaml")
BENCH = os.path.join(CONFIG_DIR, "bench_legacy.yaml")
REPO_DIR = os.path.dirname(ROBOT_DIR)


def servos_for(config, **overrides):
    """One virtual servo per joint, at the joint's zero tick unless overridden by name."""
    servos = []
    for joint in config.joints.values():
        kwargs = dict(model=joint.model, position=joint.zero_tick)
        kwargs.update(overrides.get(joint.name, {}))
        servos.append(FakeServo(joint.motor_id, **kwargs))
    return servos


@pytest.fixture
def v20_robot():
    config = load_config(V20)
    port = FakePort(servos_for(config))
    with Robot(config, port_handlers={"opencr": port}, settle_s=0) as robot:
        yield robot, port
