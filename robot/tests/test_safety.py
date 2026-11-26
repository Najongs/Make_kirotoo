import math

import pytest

from kitoru_hw.joints import Joint, PairLimit
from kitoru_hw.safety import (FallGuard, SafetyStop, TemperatureGuard, VoltageGuard, apply_pair_limits,
                              check_voltage_limits, clip_to_limits, tilt_deg)

R = math.radians
JOINTS = {
    "a": Joint("a", 1, "XL430-W250", soft_limits_rad=(R(-20), R(20)), work_limits_rad=(R(-15), R(15))),
    "b": Joint("b", 2, "XL430-W250", soft_limits_rad=(R(-20), R(20))),
    "free": Joint("free", 3, "XL430-W250"),
}


def test_clip_to_soft_and_work_limits():
    out, changed = clip_to_limits({"a": R(30), "b": R(-5)}, JOINTS)
    assert out["a"] == pytest.approx(R(20)) and out["b"] == pytest.approx(R(-5)) and changed == ["a"]
    out, _ = clip_to_limits({"a": R(18), "b": R(18)}, JOINTS, use_work=True)
    assert out["a"] == pytest.approx(R(15)) and out["b"] == pytest.approx(R(18))  # b has no work range


def test_clip_refuses_missing_limits_and_nan():
    with pytest.raises(ValueError, match="no software limits"):
        clip_to_limits({"free": 0.0}, JOINTS)
    assert clip_to_limits({"free": 1.0}, JOINTS, require=False)[0] == {"free": 1.0}
    with pytest.raises(ValueError, match="finite"):
        clip_to_limits({"a": float("nan")}, JOINTS)


def test_pair_limit_moves_both_joints_minimally():
    pair = PairLimit("yaw", "a", "b", (1.0, -1.0), (R(-10), R(10)), enabled=True)
    out, hit = apply_pair_limits({"a": R(12), "b": R(-8)}, [pair], JOINTS)
    assert out["a"] - out["b"] == pytest.approx(R(10))
    assert out["a"] == pytest.approx(R(7)) and out["b"] == pytest.approx(R(-3))
    assert hit == ["yaw"]
    disabled = PairLimit("yaw", "a", "b", (1.0, -1.0), (R(-10), R(10)), enabled=False)
    assert apply_pair_limits({"a": R(12), "b": R(-8)}, [disabled], JOINTS)[1] == []


def test_pair_limit_with_single_limit_active():
    pair = PairLimit("yaw", "a", "b", (1.0, -1.0), (R(-10), R(10)), enabled=True)
    out, _ = apply_pair_limits({"a": R(20), "b": R(-20)}, [pair], JOINTS)
    assert abs(out["a"] - out["b"]) <= R(10) + 1e-9
    assert R(-20) <= out["a"] <= R(20) and R(-20) <= out["b"] <= R(20)


def test_tilt_and_fall_guard_latch():
    assert tilt_deg(R(25), 0) == pytest.approx(25)
    assert tilt_deg(R(20), R(15)) == pytest.approx(24.81, abs=0.01)  # more than either axis alone
    guard = FallGuard(25, 265)
    assert guard.update(None) is None
    assert guard.update(24.9) is None
    event = guard.update(25.1)
    assert event.level == "reduce" and "265" in event.message
    assert guard.update(40) is None and guard.fallen  # latched, one event only
    guard.reset()
    assert not guard.fallen


def test_low_voltage_debounce_and_median():
    guard = VoltageGuard(stop_v=10.8, warn_v=11.1, debounce=3)
    assert guard.update([12.0, 12.1, 9.0]) is None              # one bad reading does not count
    assert guard.update([11.0, 11.0, 11.0]).level == "warn"
    assert guard.update([10.7] * 3).level == "warn"
    assert guard.update([10.7] * 3).level == "warn"
    assert guard.update([10.6] * 3).level == "stop"
    guard.update([12.0] * 3)
    assert guard.low_count == 0


def test_max_voltage_limit_check():
    assert check_voltage_limits({"xl": 14.0, "xm": 16.0}) == []
    (event,) = check_voltage_limits({"xl": 12.0})
    assert "12.0 V" in event.message


def test_temperature_guard_uses_servo_limits():
    guard = TemperatureGuard({"xl": 72, "xm": 80})
    assert guard.update({"xl": 61, "xm": 69}) == []
    assert [e.level for e in guard.update({"xl": 62, "xm": 70})] == ["warn", "warn"]
    assert [e.level for e in guard.update({"xl": 69, "xm": 70})] == ["stop", "warn"]


def test_safety_stop_is_runtime_error():
    assert issubclass(SafetyStop, RuntimeError)
