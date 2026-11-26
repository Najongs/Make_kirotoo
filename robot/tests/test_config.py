import math
import shutil

import pytest

from conftest import BENCH, V20
from kitoru_hw import control_table as ct
from kitoru_hw.joints import load_config, update_zero_offsets


def test_v20_has_15_servo_joints_with_unique_ids():
    config = load_config(V20)
    assert len(config.joints) == 15
    assert len({j.motor_id for j in config.joints.values()}) == 15
    assert {j.connector for j in config.joints.values()} == {"TTL1", "TTL2", "TTL3"}
    models = {n: j.model for n, j in config.joints.items()}
    assert models["R_knee"] == "XM430-W350" and models["R_hip_roll"] == "XM430-W350"
    assert models["R_ankle_pitch"] == "XC430-W240"
    assert models["R_hip_yaw"] == models["R_arm_swing"] == models["nod_crank"] == "XL430-W250"


def test_v20_limits_are_conservative():
    j = load_config(V20).joints
    assert j["R_hip_roll"].soft_limits_rad is None and j["L_hip_roll"].soft_limits_rad is None
    assert math.degrees(j["R_hip_pitch"].soft_limits_rad[1]) < 85.5  # v19 value does not fit v20
    assert math.degrees(j["R_knee"].soft_limits_rad[1]) == pytest.approx(98.2, abs=1e-3)
    for joint in j.values():
        if joint.soft_limits_rad and joint.work_limits_rad:
            (s_lo, s_hi), (w_lo, w_hi) = joint.soft_limits_rad, joint.work_limits_rad
            assert s_lo <= w_lo <= w_hi <= s_hi


def test_bench_matches_legacy_joints():
    config = load_config(BENCH)
    assert [(n, j.motor_id, j.zero_tick, j.direction, j.soft_limits_rad) for n, j in config.joints.items()] == [
        ("joint_6", 6, 2048, 1, None), ("joint_7", 7, 2048, 1, None), ("joint_12", 12, 2048, 1, None)]
    assert config.require_soft_limits is False


def test_model_numbers():
    assert {m.name: m.number for m in ct.MODELS.values()} == {
        "XL430-W250": 1060, "XC430-W240": 1080, "XM430-W350": 1020}
    with pytest.raises(ValueError):
        ct.item("GOAL_CURRENT", "XL430-W250")


def test_duplicate_id_rejected(tmp_path):
    text = open(BENCH, encoding="utf-8").read().replace("id: 12", "id: 7")
    path = tmp_path / "dup.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="already used"):
        load_config(str(path))


def test_update_zero_offsets_keeps_comments(tmp_path):
    path = tmp_path / "joints.yaml"
    shutil.copy(V20, path)
    before = path.read_text(encoding="utf-8")
    update_zero_offsets(str(path), {"R_knee": 37, "L_knee": -12}, note="test")
    after = path.read_text(encoding="utf-8")
    config = load_config(str(path))
    assert config.joints["R_knee"].zero_offset == 37 and config.joints["L_knee"].zero_offset == -12
    assert config.joints["R_hip_pitch"].zero_offset == 0
    assert after.count("\n") == before.count("\n")
    assert "# 짝 관절 제한" in after
    with pytest.raises(ValueError):
        update_zero_offsets(str(path), {"no_such_joint": 1})
