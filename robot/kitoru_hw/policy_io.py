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
    """Placeholder until an IMU is wired to the Jetson.

    The OpenCR has an ICM-20648 on board, but in USB-to-Dynamixel bridge mode the
    host only sees the Dynamixel bus [확인 필요: a bridge firmware that also streams IMU].
    """

    def read(self) -> Optional[Dict[str, Sequence[float]]]:
        return None


class PolicyIO:
    def __init__(self, robot: Robot, imu=None, clip_to_work: bool = True):
        self.robot = robot
        self.imu = imu or NullImu()
        self.clip_to_work = clip_to_work
        self.joint_names: List[str] = list(robot.joints)

    def observe(self) -> Dict[str, object]:
        """One synchronous sample: q, dq in joint order and the IMU (None if not wired)."""
        stamp = time.monotonic()
        state = self.robot.read_state(self.joint_names)
        return {
            "time": stamp,
            "joint_names": self.joint_names,
            "q": [state[n]["position_rad"] for n in self.joint_names],
            "dq": [state[n]["velocity_rad_s"] for n in self.joint_names],
            "imu": self.imu.read(),  # {"quat": [w,x,y,z], "gyro": [x,y,z], "roll_pitch": [r,p]} or None
        }

    def act(self, q_target: Sequence[float]) -> List[str]:
        """Write target angles in joint order. Returns the joints that were clipped."""
        if len(q_target) != len(self.joint_names):
            raise ValueError(f"expected {len(self.joint_names)} targets, got {len(q_target)}")
        return self.robot.command_positions(dict(zip(self.joint_names, q_target)), use_work=self.clip_to_work)
