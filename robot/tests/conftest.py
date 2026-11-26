import os
import sys
from dataclasses import replace

ROBOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROBOT_DIR, os.path.join(ROBOT_DIR, "tools"), os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

from fake_dxl import FakeServo  # noqa: E402

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


def calibrated(config):
    """Mark every joint calibrated, as a person would after checking ID, direction and zero."""
    for name, joint in list(config.joints.items()):
        config.joints[name] = replace(joint, calibrated=True)
    return config


def calibrated_yaml(tmp_path):
    """Copy of joints_v20.yaml with calibrated: true everywhere (for tools that read a file)."""
    path = tmp_path / "joints_calibrated.yaml"
    path.write_text(open(V20, encoding="utf-8").read().replace("calibrated: false", "calibrated: true"),
                    encoding="utf-8")
    return str(path)
