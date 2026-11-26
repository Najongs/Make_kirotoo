import math
import shutil

import pytest

from conftest import V20, calibrated_yaml, servos_for
from fake_dxl import FakePort, FakeServo
from kitoru_hw.joints import load_config
from kitoru_hw.timing import cycle_time

import range_check
import record_zero
import scan_ids


def test_scan_reports_missing_and_extra(capsys):
    config = load_config(V20)
    servos = [s for s in servos_for(config) if s.motor_id != 4] + [FakeServo(40)]
    scan_ids.main(["--port", "fake"], port_factory=lambda name: FakePort(servos))
    out = capsys.readouterr().out
    assert "14 servo(s)" not in out and "15 servo(s)" in out
    assert "missing  ID   4 R_knee" in out and "extra    ID  40" in out


def test_record_zero_saves_offsets(tmp_path, capsys):
    path = tmp_path / "joints.yaml"
    shutil.copy(V20, path)
    config = load_config(V20)
    servos = servos_for(config, R_knee={"position": 2048 + 37}, L_knee={"position": 2048 - 500})
    port = FakePort(servos)
    record_zero.main(["--config", str(path), "--save"], port_handlers={"opencr": port}, settle_s=0)
    out = capsys.readouterr().out
    saved = load_config(str(path)).joints
    assert saved["R_knee"].zero_offset == 37 and saved["L_knee"].zero_offset == -500
    assert "check the horn" in out
    assert all(s.get("TORQUE_ENABLE") == 0 for s in servos)


def test_record_zero_refuses_with_torque_on(tmp_path):
    config = load_config(V20)
    servos = servos_for(config)
    servos[0].set("TORQUE_ENABLE", 1)
    with pytest.raises(SystemExit):
        record_zero.main([], port_handlers={"opencr": FakePort(servos)}, settle_s=0)


def test_range_check_stops_short_at_obstacle(capsys, tmp_path):
    config = load_config(V20)
    knee = config.joints["R_knee"]
    blocked_hi = knee.tick(math.radians(60))  # a part touches at 60 deg
    servos = servos_for(config, R_knee={"position": knee.tick(math.radians(20)),
                                         "blocked": (0, blocked_hi)})
    port = FakePort(servos)
    range_check.main(["--config", calibrated_yaml(tmp_path), "--joint", "R_knee", "--step", "5", "--timeout", "0.05",
                      "--yes"],
                     port_handlers={"opencr": port}, settle_s=0)
    out = capsys.readouterr().out
    lower, upper = [line for line in out.splitlines() if line.strip().startswith(("lower", "upper"))]
    assert "ok" in lower and "STOPPED SHORT" in upper and "reached    60.0 deg" in upper
    servo = port.servos[knee.motor_id]
    assert servo.get("TORQUE_ENABLE") == 0 and servo.get("GOAL_PWM") == 885
    assert abs(servo.get("PRESENT_POSITION") - knee.tick(math.radians(20))) <= 3  # back at the start


def test_range_check_needs_limits_for_unfilled_joint(tmp_path):
    config = load_config(V20)
    with pytest.raises(SystemExit):
        range_check.main(["--config", calibrated_yaml(tmp_path), "--joint", "R_hip_roll", "--yes"],
                         port_handlers={"opencr": FakePort(servos_for(config))}, settle_s=0)


def test_range_check_unfilled_joint_with_explicit_range(capsys, tmp_path):
    config = load_config(V20)
    port = FakePort(servos_for(config))
    range_check.main(["--config", calibrated_yaml(tmp_path), "--joint", "R_hip_roll", "--lo", "-5", "--hi", "5",
                      "--step", "5", "--timeout", "0.05", "--yes"],
                     port_handlers={"opencr": port}, settle_s=0)
    out = capsys.readouterr().out
    assert "for this run only" in out and out.count(" ok ") == 2
    servo = port.servos[config.joints["R_hip_roll"].motor_id]
    assert servo.get("TORQUE_ENABLE") == 0 and servo.get("PRESENT_POSITION") == 2048
    assert load_config(V20).joints["R_hip_roll"].soft_limits_rad is None  # YAML untouched


def test_cycle_time_numbers():
    # 15 servos, 1 Mbps: (14+15) + 15*(11+10) + (14+15*5) = 433 bytes -> 4.33 ms
    t = cycle_time(15, 1_000_000)
    assert t.wire_ms == pytest.approx(4.33)
    assert cycle_time(15, 1_000_000, return_delay_us=500).total_ms == pytest.approx(4.33 + 7.5)
    assert cycle_time(17, 4_000_000).rate_hz > cycle_time(17, 2_000_000).rate_hz > cycle_time(17, 1_000_000).rate_hz


def test_read_demo_main_read_and_rate(capsys):
    import read_demo
    servos = [FakeServo(i, "XL430-W250", 2048) for i in (6, 7, 12)]
    read_demo.main([], port_handlers={"opencr": FakePort(servos)}, settle_s=0)
    read_demo.main(["--rate", "5"], port_handlers={"opencr": FakePort(servos)}, settle_s=0)
    out = capsys.readouterr().out
    assert "position: joint_6=0.00 deg (2048 ticks)" in out and "5 sync reads of 3 joints" in out
