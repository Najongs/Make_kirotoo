import math

import pytest

from conftest import V20, calibrated, servos_for
from fake_dxl import SYNC_WRITE, FakePort
from kitoru_hw.joints import load_config
from kitoru_hw.policy_io import PolicyIO
from kitoru_hw.robot import Robot
from kitoru_hw.safety import SafetyStop

R = math.radians
LEGS = ["R_knee", "R_ankle_pitch", "L_knee"]


def leg_robot(**overrides):
    config = calibrated(load_config(V20))
    port = FakePort(servos_for(config, **overrides))
    robot = Robot(config, port_handlers={"opencr": port}, settle_s=0)
    return robot, port


def test_arm_command_disarm_restores_registers():
    robot, port = leg_robot(R_knee={"position": 2048 + 200})
    knee = port.servos[robot.joints["R_knee"].motor_id]
    knee.set("PROFILE_VELOCITY", 77)
    with robot:
        robot.arm(LEGS)
        assert knee.get("TORQUE_ENABLE") == 1 and knee.get("PROFILE_VELOCITY") == 15
        port.tx_log.clear()
        clipped = robot.command_positions({"R_knee": R(120), "R_ankle_pitch": R(-10), "L_knee": R(30)},
                                          max_step_rad=math.inf)
        assert clipped == ["R_knee"]
        assert len(port.instructions(SYNC_WRITE)) == 1  # one packet for the whole pose
        state = robot.read_state(LEGS)
        assert state["R_knee"]["position_rad"] == pytest.approx(R(98.2), abs=R(0.1))
        assert state["L_knee"]["position_rad"] == pytest.approx(R(30), abs=R(0.1))
    assert knee.get("TORQUE_ENABLE") == 0 and knee.get("PROFILE_VELOCITY") == 77


def test_arm_refuses_pose_outside_limits_and_unlimited_joint():
    robot, _ = leg_robot(R_knee={"position": 2048 - 200})  # knee bent backwards, below 0 deg
    with robot:
        with pytest.raises(ValueError, match="outside the configured joint limits"):
            robot.arm(["R_knee"])
        assert robot.armed == []
        with pytest.raises(ValueError, match="no software limits"):
            robot.arm(["R_hip_roll"])  # [v20 후 채움] range not filled yet


def test_arm_refuses_wrong_mode():
    robot, port = leg_robot()
    port.servos[robot.joints["R_knee"].motor_id].set("OPERATING_MODE", 16)
    with robot, pytest.raises(RuntimeError, match="position mode"):
        robot.arm(["R_knee"])


def test_fall_hook_lowers_goal_pwm_on_armed_joints():
    robot, port = leg_robot()
    with robot:
        assert robot.start_monitor() == []
        robot.arm(LEGS)
        port.tx_log.clear()
        events = robot.safety_step(tilt_deg=30)
        assert [e.kind for e in events] == ["fall"]
        ids = [robot.joints[n].motor_id for n in LEGS]
        assert [port.servos[i].get("GOAL_PWM") for i in ids] == [265] * 3
        assert len(port.instructions(SYNC_WRITE)) == 1
        robot.reset_fall()
        assert [port.servos[i].get("GOAL_PWM") for i in ids] == [885] * 3


def test_low_voltage_stops_robot():
    robot, port = leg_robot()
    with robot:
        robot.start_monitor()
        robot.arm(LEGS)
        for servo in port.servos.values():
            servo.set("PRESENT_INPUT_VOLTAGE", 105)
        robot.safety_step()
        robot.safety_step()
        with pytest.raises(SafetyStop, match="10.5 V"):
            robot.safety_step()
        assert all(s.get("TORQUE_ENABLE") == 0 for s in port.servos.values())
        assert robot.armed == []


def test_overheat_stops_robot():
    robot, port = leg_robot(R_arm_swing={"temperature_c": 70})  # XL430 limit 72
    with robot:
        robot.start_monitor()
        with pytest.raises(SafetyStop, match="R_arm_swing"):
            robot.safety_step()


def test_policy_io_order_and_clip():
    config = calibrated(load_config(V20))
    for name in ("R_hip_roll", "L_hip_roll"):  # give the unfilled joints a temporary range for this test
        j = config.joints[name]
        config.joints[name] = type(j)(**{**j.__dict__, "soft_limits_rad": (R(-10), R(10))})
    knee = config.joints["R_knee"]
    port = FakePort(servos_for(config, R_knee={"position": knee.tick(R(92))}))
    with Robot(config, port_handlers={"opencr": port}, settle_s=0) as robot:
        io = PolicyIO(robot)
        obs = io.observe()
        assert obs["joint_names"] == list(config.joints) and len(obs["q"]) == 15 and obs["imu"] is None
        robot.arm()
        target = list(obs["q"])
        target[obs["joint_names"].index("R_knee")] = R(97)  # from 92: inside soft (98.2), outside work (93.2)
        assert io.act(target) == ["R_knee"]
        assert robot.read_state(["R_knee"])["R_knee"]["position_rad"] == pytest.approx(R(93.2), abs=R(0.1))
        with pytest.raises(ValueError):
            io.act(target[:-1])
