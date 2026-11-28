"""Observation / action interface for a learned walking policy.

The joint order is the order of `joints:` in the YAML file, so a simulator can use
the same order. Angles are radians in the robot convention (after direction and
zero offset), the same convention as the MuJoCo joints.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

from .robot import Robot


class NullImu:
    """No IMU: read() returns None.

    With the bridge firmware robot/firmware/opencr_bridge_imu on the OpenCR, use
    kitoru_hw.opencr_imu.OpenCRImu instead (same read() interface).
    """

    def read(self) -> Optional[Dict[str, Sequence[float]]]:
        return None


class TimingLog:
    """Seconds spent per part of a cycle (servo sync read, IMU read): count, mean, max."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.stats: Dict[str, List[float]] = {}  # key: [count, total, max]

    def add(self, key: str, seconds: float):
        s = self.stats.setdefault(key, [0, 0.0, 0.0])
        s[0] += 1
        s[1] += seconds
        s[2] = max(s[2], seconds)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {k: {"count": int(c), "mean_ms": t / c * 1000 if c else 0.0, "max_ms": m * 1000}
                for k, (c, t, m) in self.stats.items()}


class PolicyIO:
    def __init__(self, robot: Robot, imu=None, clip_to_work: bool = True):
        self.robot = robot
        self.imu = imu or NullImu()
        self.clip_to_work = clip_to_work
        self.joint_names: List[str] = list(robot.joints)
        self.timing = TimingLog()

    def observe(self) -> Dict[str, object]:
        """One synchronous sample: q, dq in joint order and the IMU (None if not wired).

        obs["timing"] has the seconds of this cycle's servo read and IMU read; self.timing
        keeps count / mean / max over the run (print self.timing.summary()).
        """
        stamp = time.monotonic()
        t0 = time.perf_counter()
        state = self.robot.read_state(self.joint_names)
        t1 = time.perf_counter()
        imu = self.imu.read()
        t2 = time.perf_counter()
        self.timing.add("servo_read", t1 - t0)
        self.timing.add("imu_read", t2 - t1)
        return {
            "time": stamp,
            "joint_names": self.joint_names,
            "q": [state[n]["position_rad"] for n in self.joint_names],
            "dq": [state[n]["velocity_rad_s"] for n in self.joint_names],
            # OpenCRImu: {"gravity_b", "ang_vel_b", "acc_b", "quat", "gyro", "roll_pitch", ...}; NullImu: None
            "imu": imu,
            "timing": {"servo_read_s": t1 - t0, "imu_read_s": t2 - t1},
        }

    def act(self, q_target: Sequence[float]) -> List[str]:
        """Write target angles in joint order. Returns the joints that were clipped."""
        if len(q_target) != len(self.joint_names):
            raise ValueError(f"expected {len(self.joint_names)} targets, got {len(q_target)}")
        return self.robot.command_positions(dict(zip(self.joint_names, q_target)), use_work=self.clip_to_work)
