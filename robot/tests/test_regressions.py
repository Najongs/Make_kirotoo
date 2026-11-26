"""Regression tests for the defects found in the review of 354a3a6 (numbers = review items)."""

import math
import os
import shutil

import pytest

from conftest import BENCH, V20, calibrated, calibrated_yaml, servos_for
from fake_dxl import SYNC_WRITE, FakePort
from kitoru_hw import joints as joints_module
from kitoru_hw.joints import load_config, update_zero_offsets
from kitoru_hw.robot import Robot
from kitoru_hw.safety import SafetyStop

import range_check
import record_zero

R = math.radians


class CtrlCPort(FakePort):
    """Raises KeyboardInterrupt once, in the middle of reading a reply."""
    ctrl_c = False

    def readPort(self, length):
        if self.ctrl_c:
            self.ctrl_c = False
            raise KeyboardInterrupt
        return super().readPort(length)


def make(port_cls=FakePort, config=None, **overrides):
    config = config or calibrated(load_config(V20))
    port = port_cls(servos_for(config, **overrides))
    return Robot(config, port_handlers={"opencr": port}, settle_s=0), port, config


def servo(port, config, name):
    return port.servos[config.joints[name].motor_id]


# 1 ---------------------------------------------------------------------------
def test_1_ctrl_c_mid_packet_then_exit_still_turns_torque_off():
    robot, port, config = make(CtrlCPort)
    with pytest.raises(KeyboardInterrupt):
        with robot:
            robot.arm(["R_knee"])
            port.ctrl_c = True
            robot.read_state(["R_knee"])
    assert servo(port, config, "R_knee").get("TORQUE_ENABLE") == 0
    assert port.is_using is False


def test_1_emergency_stop_frees_a_port_left_locked():
    robot, port, config = make()
    with robot:
        robot.arm(["R_knee"])
        port.is_using = True  # as if an earlier call had been cut short outside the bus
        assert robot.emergency_stop() == []
        assert servo(port, config, "R_knee").get("TORQUE_ENABLE") == 0


# 2 ---------------------------------------------------------------------------
def test_2_failed_fall_pwm_write_is_retried_next_cycle():
    robot, port, config = make()
    with robot:
        robot.start_monitor()
        robot.arm(["R_knee"])
        good = port.writePort
        port.writePort = lambda packet: 0  # transmit failure
        events = robot.safety_step(tilt_deg=40, read_health=False)
        assert any("will retry" in e.message for e in events) and not robot.monitor.fall.fallen
        port.writePort = good
        robot.safety_step(tilt_deg=40, read_health=False)
        assert robot.monitor.fall.fallen
        assert servo(port, config, "R_knee").get("GOAL_PWM") == 265


def test_2_stop_wins_over_a_failing_fall_pwm_write(monkeypatch):
    robot, port, config = make()
    with robot:
        robot.start_monitor()
        robot.arm(["R_knee"])
        robot.monitor.voltage.low_count = 2
        for s in port.servos.values():
            s.set("PRESENT_INPUT_VOLTAGE", 100)

        def failing(*args, **kwargs):
            raise RuntimeError("Sync write GOAL_PWM failed")
        monkeypatch.setattr(robot, "set_goal_pwm", failing)
        with pytest.raises(SafetyStop, match="10.0 V"):
            robot.safety_step(tilt_deg=40)
        assert servo(port, config, "R_knee").get("TORQUE_ENABLE") == 0 and robot.armed == []


# 3 ---------------------------------------------------------------------------
def test_3_arm_refuses_placeholder_calibration():
    robot, port, config = make(config=load_config(V20))  # YAML as shipped: calibrated: false
    assert not any(j.calibrated for j in config.joints.values())
    with robot, pytest.raises(RuntimeError, match="Not calibrated"):
        robot.arm(["R_knee"])
    assert servo(port, config, "R_knee").get("TORQUE_ENABLE") == 0
    assert all(j.calibrated for j in load_config(BENCH).joints.values())


def test_3_range_check_refuses_uncalibrated_joint():
    config = load_config(V20)
    with pytest.raises(SystemExit):
        range_check.main(["--joint", "R_knee", "--yes"],
                         port_handlers={"opencr": FakePort(servos_for(config))}, settle_s=0)


# 4 ---------------------------------------------------------------------------
def test_4_watchdog_set_after_arm_and_cleared_on_disarm():
    robot, port, config = make()
    knee = servo(port, config, "R_knee")
    with robot:
        robot.arm(["R_knee"], watchdog_ms=100)
        assert knee.get("BUS_WATCHDOG") == 5  # 5 x 20 ms
        knee.set("BUS_WATCHDOG", -1)          # the watchdog fired: goal writes are refused now
        goal = knee.get("GOAL_POSITION")
        robot.command_positions({"R_knee": R(3)})  # sync write has no reply, the servo just ignores it
        assert knee.get("GOAL_POSITION") == goal
    assert knee.get("BUS_WATCHDOG") == 0 and knee.get("TORQUE_ENABLE") == 0
    assert knee.get("PROFILE_VELOCITY") == 0 and knee.get("GOAL_PWM") == 885  # settings restored


def test_4_watchdog_range_checked():
    robot, _, _ = make()
    with robot, pytest.raises(ValueError, match="watchdog_ms"):
        robot.arm(["R_knee"], watchdog_ms=5000)


# 5 ---------------------------------------------------------------------------
def test_5_emergency_stop_retries_single_servo_and_reports_stuck_one():
    robot, port, config = make(R_knee={"ignore_broadcast": True}, L_knee={"stuck_torque": True})
    with robot:
        robot.start_monitor()
        robot.arm(["R_knee", "L_knee"])
        for s in port.servos.values():
            s.set("PRESENT_TEMPERATURE", 79)  # XM limit 80 -> stop
        with pytest.raises(SafetyStop, match=r"emergency stop problems: L_knee: torque still on"):
            robot.safety_step()
        assert servo(port, config, "R_knee").get("TORQUE_ENABLE") == 0  # missed broadcast, fixed one by one
        for s in port.servos.values():
            s.stuck_torque = False
            s.set("PRESENT_TEMPERATURE", 35)


# 6 ---------------------------------------------------------------------------
def test_6_command_jump_is_limited_per_command():
    robot, port, config = make()
    with robot:
        robot.arm(["R_knee"])
        assert robot.max_step_rad == pytest.approx(R(5))
        assert robot.command_positions({"R_knee": R(60)}) == ["R_knee"]
        assert robot.read_state(["R_knee"])["R_knee"]["position_rad"] == pytest.approx(R(5), abs=R(0.1))
        robot.command_positions({"R_knee": R(60)})
        assert robot.last_command["R_knee"] == pytest.approx(R(10))
        assert robot.command_positions({"R_knee": R(12)}) == []  # small step passes unchanged


# 7 ---------------------------------------------------------------------------
def test_7_reset_fall_restores_the_goal_pwm_given_at_arm():
    robot, port, config = make()
    with robot:
        robot.start_monitor()
        robot.arm(["R_knee"], goal_pwm=200)
        robot.safety_step(tilt_deg=40, read_health=False)
        assert servo(port, config, "R_knee").get("GOAL_PWM") == 265
        robot.reset_fall()
        assert servo(port, config, "R_knee").get("GOAL_PWM") == 200 and not robot.monitor.fall.fallen


# 8 ---------------------------------------------------------------------------
@pytest.mark.parametrize("extra", [["--step", "0"], ["--step", "6"], ["--pwm", "0"], ["--pwm", "900"],
                                   ["--profile-velocity", "0"], ["--profile-velocity", "500"]])
def test_8_range_check_rejects_bad_arguments(extra, tmp_path):
    config = load_config(V20)
    port = FakePort(servos_for(config))
    with pytest.raises(SystemExit):
        range_check.main(["--config", calibrated_yaml(tmp_path), "--joint", "R_knee", "--yes", *extra],
                         port_handlers={"opencr": port}, settle_s=0)
    assert port.instructions(SYNC_WRITE) == []


def test_8_unfilled_joint_range_capped_at_30_deg(tmp_path):
    config = load_config(V20)
    port = FakePort(servos_for(config))
    with pytest.raises(SystemExit):
        range_check.main(["--config", calibrated_yaml(tmp_path), "--joint", "R_hip_roll", "--lo", "-31", "--hi", "5",
                          "--yes"], port_handlers={"opencr": port}, settle_s=0)
    assert servo(port, config, "R_hip_roll").get("TORQUE_ENABLE") == 0


# 9 ---------------------------------------------------------------------------
def test_9_arm_refuses_while_earlier_settings_are_not_restored():
    robot, port, config = make()
    with robot:
        robot.arm(["R_knee"])
        robot.emergency_stop()
        with pytest.raises(RuntimeError, match="call disarm"):
            robot.arm(["L_knee"])
        assert list(robot._saved) == ["R_knee"]
        robot.disarm()
        robot.arm(["L_knee"])
    assert servo(port, config, "L_knee").get("TORQUE_ENABLE") == 0


# 10 --------------------------------------------------------------------------
def test_10_record_zero_refuses_to_save_bad_offsets(tmp_path):
    path = tmp_path / "joints.yaml"
    shutil.copy(V20, path)
    before = path.read_text(encoding="utf-8")
    config = load_config(V20)
    servos = servos_for(config, R_arm_swing={"position": 2048 + 1000})  # +-120 deg then leaves 0..4095
    with pytest.raises(SystemExit):
        record_zero.main(["--config", str(path), "--save"], port_handlers={"opencr": FakePort(servos)}, settle_s=0)
    assert path.read_text(encoding="utf-8") == before


def test_10_yaml_write_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "joints.yaml"
    shutil.copy(V20, path)
    before = path.read_text(encoding="utf-8")

    def crash(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(joints_module.os, "replace", crash)
    with pytest.raises(OSError):
        update_zero_offsets(str(path), {"R_knee": 37})
    assert path.read_text(encoding="utf-8") == before
    monkeypatch.undo()
    update_zero_offsets(str(path), {"R_knee": 37})
    assert not os.path.exists(f"{path}.tmp") and load_config(str(path)).joints["R_knee"].zero_offset == 37
