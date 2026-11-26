import csv
import math
import os

import pytest

from conftest import REPO_DIR
from kitoru_hw import control_table as ct
from kitoru_hw.joints import Joint


def joint(direction=1, offset=0, soft_deg=None):
    soft = tuple(math.radians(v) for v in soft_deg) if soft_deg else None
    return Joint("j", 1, "XM430-W350", direction=direction, zero_offset=offset, soft_limits_rad=soft)


def test_formula_matches_joint_ranges_example():
    # JOINT_RANGES.md section 4: right knee, d = +1, off = +37 -> Min 1755, Max 3202
    j = joint(1, 37, (-29.0, 98.2))
    assert j.dxl_position_limits() == (1755, 3202)


def test_reverse_direction_swaps_min_and_max():
    j = joint(-1, 37, (-29.0, 98.2))
    lo, hi = j.dxl_position_limits()
    assert (lo, hi) == (2048 + 37 - round(98.2 * 4096 / 360), 2048 + 37 + round(29.0 * 4096 / 360))
    assert j.angle(hi) == pytest.approx(math.radians(-29.0), abs=ct.TICK_RAD)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("offset", [0, -300, 411])
def test_tick_angle_round_trip(direction, offset):
    j = joint(direction, offset)
    for tick in range(0, 4096, 97):
        assert j.tick(j.angle(tick)) == tick
    for deg in (-120.0, -17.6, 0.0, 33.3, 98.2):
        assert j.angle(j.tick(math.radians(deg))) == pytest.approx(math.radians(deg), abs=ct.TICK_RAD / 2 + 1e-12)


def test_velocity_unit_and_sign():
    assert joint(1).velocity(100) == pytest.approx(100 * 0.229 * 2 * math.pi / 60)
    assert joint(-1).velocity(100) == pytest.approx(-joint(1).velocity(100))


def test_matches_v19_csv_raw_columns_within_one_tick():
    path = os.path.join(REPO_DIR, "sim", "walk", "ranges", "joint_ranges_v19.csv")
    if not os.path.exists(path):
        pytest.skip("v19 range table not in this checkout")
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["dxl_min_raw"]]
    assert rows
    for r in rows:
        lo, hi = joint(1, 0, (float(r["soft_lower_deg"]), float(r["soft_upper_deg"]))).dxl_position_limits()
        # The csv angles are rounded to 0.1 deg, the raw columns were computed before rounding.
        assert abs(lo - int(r["dxl_min_raw"])) <= 1, r["joint"]
        assert abs(hi - int(r["dxl_max_raw"])) <= 1, r["joint"]


def test_signed_helpers():
    assert ct.to_signed(ct.to_unsigned(-5, 4), 4) == -5
    assert ct.to_signed(0xFFFF, 2) == -1
    with pytest.raises(ValueError):
        ct.to_unsigned(1 << 16, 2)
