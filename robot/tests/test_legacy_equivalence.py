"""New tools against the original bench script, both talking to the same virtual servos."""

import importlib.util
import io
import os
import re
from contextlib import redirect_stdout

from conftest import BENCH, ROBOT_DIR
from fake_dxl import SYNC_WRITE, FakePort, FakeServo
from kitoru_hw.joints import load_config
from kitoru_hw.robot import Robot

import read_demo

START = {6: 2000, 7: 2100, 12: 2300}


def load_legacy():
    path = os.path.join(ROBOT_DIR, "legacy", "XL430_robot_base.py")
    spec = importlib.util.spec_from_file_location("legacy_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_port():
    return FakePort([FakeServo(i, "XL430-W250", t) for i, t in START.items()])


def strip_times(text):
    return re.sub(r"^\d+\.\d+s ", "", text, flags=re.M)


def run_legacy(demo, monkeypatch):
    legacy, port = load_legacy(), fresh_port()
    monkeypatch.setattr(legacy, "PortHandler", lambda name: port)
    monkeypatch.setattr(legacy.time, "sleep", lambda s: None)  # restored after the test
    out = io.StringIO()
    with redirect_stdout(out), legacy.XL430Bus("COM3") as bus:
        if demo:
            legacy.run_demo(bus, 0.0)
        else:
            legacy.print_positions("position", bus.read_state(), 0.0)
    return port, strip_times(out.getvalue())


def run_new(demo):
    port = fresh_port()
    out = io.StringIO()
    with redirect_stdout(out), Robot(load_config(BENCH), port_handlers={"opencr": port}, settle_s=0) as robot:
        if demo:
            read_demo.run_demo(robot, 0.0, period=0)
        else:
            read_demo.print_positions("position", robot.read_state(), 0.0)
    return port, strip_times(out.getvalue())


def test_read_only_prints_the_same(monkeypatch):
    (_, old), (_, new) = run_legacy(False, monkeypatch), run_new(False)
    assert old == new
    assert "joint_6=-4.22 deg (2000 ticks)" in new


def test_demo_sends_identical_goal_packets_and_restores(monkeypatch):
    (old_port, old), (new_port, new) = run_legacy(True, monkeypatch), run_new(True)
    assert old == new
    old_goals, new_goals = old_port.instructions(SYNC_WRITE), new_port.instructions(SYNC_WRITE)
    assert len(new_goals) == 2 and old_goals == new_goals  # byte for byte: pose B, then pose A
    for port in (old_port, new_port):
        for motor_id, tick in START.items():
            servo = port.servos[motor_id]
            assert servo.get("TORQUE_ENABLE") == 0
            assert servo.get("PRESENT_POSITION") == tick
            assert servo.get("PROFILE_VELOCITY") == 0 and servo.get("GOAL_POSITION") == tick
