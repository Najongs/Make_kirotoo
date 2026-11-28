"""OpenCR bridge firmware core (C++, through ctypes) and the Jetson-side OpenCRImu.

The core tests need a host C++ compiler (g++); without one they are skipped.
"""

import math
import random
import struct

import pytest
from dynamixel_sdk import COMM_SUCCESS, PacketHandler
from dynamixel_sdk.protocol2_packet_handler import Protocol2PacketHandler

from conftest import V20, calibrated, servos_for
from fake_dxl import FakePort, FakeServo
from fake_opencr import BridgePort, Core, build_core, compiler, f32, u8
from kitoru_hw import opencr_imu as oi
from kitoru_hw.dxl_bus import DxlBus, DxlError
from kitoru_hw.joints import load_config
from kitoru_hw.policy_io import PolicyIO
from kitoru_hw.robot import Robot

import imu_check

needs_cxx = pytest.mark.skipif(compiler() is None, reason="no host C++ compiler for the firmware core")
PH = Protocol2PacketHandler()
STUFF_FLOAT = struct.unpack("<f", b"\xFF\xFF\xFD\x3F")[0]  # ~1.984, little endian bytes FF FF FD 3F


@pytest.fixture(scope="session")
def so(tmp_path_factory):
    if compiler() is None:
        pytest.skip("no host C++ compiler")
    return build_core(tmp_path_factory.mktemp("kbridge"))


def bridge_bus(so, servos=(), **imu):
    core = Core(so)
    core.set_imu(**imu) if imu else core.set_imu()
    port = BridgePort(list(servos), core)
    return core, port


def packet(motor_id, instruction, params=()):
    """A Protocol 2.0 instruction packet built by the SDK's own stuffing and CRC."""
    length = len(params) + 3
    p = [0xFF, 0xFF, 0xFD, 0x00, motor_id, length & 0xFF, length >> 8, instruction] + list(params) + [0, 0]
    p = PH.addStuffing(p)
    crc = PH.updateCRC(0, p, len(p) - 2)
    p[-2:] = [crc & 0xFF, crc >> 8]
    return bytes(p)


# ---- firmware core: CRC, stuffing, gravity -------------------------------------------
@needs_cxx
def test_crc_and_stuffing_match_the_sdk(so):
    rng = random.Random(0)
    for _ in range(200):
        params = [rng.choice([0xFF, 0xFD, 0x00, rng.randrange(256)]) for _ in range(rng.randrange(0, 40))]
        raw = [0xFF, 0xFF, 0xFD, 0x00, 200, (len(params) + 3) & 0xFF, 0, 0x02] + params + [0, 0]
        out = u8(b"\0" * 256)[0]
        buf, n = u8(raw)
        m = so.kb_add_stuffing(buf, n, out, 256)
        expected = PH.addStuffing(list(raw))
        assert bytes(out[:m]) == bytes(expected)
        buf, n = u8(expected[:-2])
        assert so.kb_crc16(buf, n) == PH.updateCRC(0, expected, len(expected) - 2)


@needs_cxx
def test_gravity_from_quat_points_down_in_sensor_axes(so):
    def g_of(q):
        g = f32([0, 0, 0])
        so.kb_gravity_from_quat(f32(q), g)
        return list(g)

    assert g_of([1, 0, 0, 0]) == pytest.approx([0, 0, -1])
    rng = random.Random(1)
    for _ in range(50):
        q = [rng.gauss(0, 1) for _ in range(4)]
        n = math.sqrt(sum(c * c for c in q))
        q = [c / n for c in q]
        # quaternion maps sensor -> world; gravity in sensor axes = R^T (0, 0, -1)
        r = oi.quat_to_matrix(q)
        assert g_of(q) == pytest.approx(oi.matvec(oi.transpose(r), [0, 0, -1]), abs=1e-5)


@needs_cxx
def test_filter_start_from_accel(so):
    """The firmware starts its Madgwick filter here; gravity must equal -accel direction at once."""
    rng = random.Random(2)
    cases = [[0, 0, 9.8], [0, -9.8, 0], [9.8, 0, 0], [0, 0, -9.8], [0.001, 0, -9.8]]
    cases += [[rng.gauss(0, 5) for _ in range(3)] for _ in range(50)]
    for a in cases:
        q, g = f32([0, 0, 0, 0]), f32([0, 0, 0])
        assert so.kb_quat_from_accel(f32(a), q) == 1
        assert sum(c * c for c in q) == pytest.approx(1.0, abs=1e-5)
        so.kb_gravity_from_quat(q, g)
        n = math.sqrt(sum(c * c for c in a))
        assert list(g) == pytest.approx([-c / n for c in a], abs=2e-3)  # within 0.1 deg (upside-down snap)
    assert so.kb_quat_from_accel(f32([0, 0, 0]), f32([0, 0, 0, 0])) == 0


# ---- firmware core: the virtual ID through the real SDK -------------------------------
@needs_cxx
def test_ping_and_read_virtual_id_never_touch_the_bus(so):
    quat = [0.9238795, 0.3826834, 0.0, 0.0]  # 45 deg about chip x
    core, port = bridge_bus(so, [FakeServo(1)], timestamp_us=123456, count=7, status=0x07, quat=quat,
                            gyro=[0.1, -0.2, 0.3], acc=[0.0, 6.9, 6.9])
    port.setBaudRate(1_000_000)
    ph = PacketHandler(2.0)
    model, result, error = ph.ping(port, 200)
    assert (result, error, model) == (COMM_SUCCESS, 0, oi.MODEL_NUMBER)
    data, result, error = ph.readTxRx(port, 200, oi.SAMPLE_ADDR, oi.SAMPLE_LEN)
    assert result == COMM_SUCCESS and error == 0
    s = oi.decode_sample(bytes(data))
    assert (s["timestamp_us"], s["sample_count"], s["status"]) == (123456, 7, 0x07)
    assert s["quat_chip"] == pytest.approx(quat)
    assert s["gyro_chip"] == pytest.approx([0.1, -0.2, 0.3])
    assert s["gravity_chip"] == pytest.approx([0, -math.sin(math.pi / 4), -math.cos(math.pi / 4)], abs=1e-6)
    assert port.bus_log == []  # the servo bus never saw ID 200
    model, result, _ = ph.ping(port, 1)  # a servo still answers through the bridge
    assert result == COMM_SUCCESS and model == 1060


@needs_cxx
def test_reply_with_stuffing_decodes(so):
    core, port = bridge_bus(so, gyro=[STUFF_FLOAT, 0.0, -STUFF_FLOAT])
    port.setBaudRate(1_000_000)
    data, result, error = PacketHandler(2.0).readTxRx(port, 200, oi.SAMPLE_ADDR, oi.SAMPLE_LEN)
    assert result == COMM_SUCCESS and error == 0
    assert oi.decode_sample(bytes(data))["gyro_chip"] == pytest.approx([STUFF_FLOAT, 0.0, -STUFF_FLOAT])
    assert b"\xFF\xFF\xFD\xFD" in port.host_log[-1]  # the reply really carried a stuffed byte


@needs_cxx
def test_virtual_id_errors_and_writes(so):
    core, port = bridge_bus(so)
    port.setBaudRate(1_000_000)
    ph = PacketHandler(2.0)
    _, result, error = ph.readTxRx(port, 200, 120, 16)          # past the end of the table
    assert result == COMM_SUCCESS and error == 0x04
    result, error = ph.write1ByteTxRx(port, 200, 40, 1)         # STATUS is read only
    assert result == COMM_SUCCESS and error == 0x07
    assert so.kb_take_gyro_calibrate_request(core.h) == 0
    result, error = ph.write1ByteTxRx(port, 200, oi.ADDR_GYRO_CALIBRATE, 1)
    assert (result, error) == (COMM_SUCCESS, 0)
    assert so.kb_take_gyro_calibrate_request(core.h) == 1
    bad = bytearray(packet(200, 0x02, [32, 0, 64, 0]))
    bad[-1] ^= 0xFF
    core.feed(bytes(bad), 10_000)
    reply = core.take_host()
    assert reply[4] == 200 and reply[8] == 0x03                 # CRC error status
    core.feed(packet(200, 0x08), 20_000)                         # REBOOT is not supported
    assert core.take_host()[8] == 0x02
    assert port.bus_log == [] and core.take_bus() == b""


@needs_cxx
def test_virtual_packet_split_over_two_usb_transfers(so):
    core = Core(so)
    core.set_imu(count=5)
    p = packet(200, 0x02, [36, 0, 4, 0])
    core.feed(p[:3], 1000)
    core.feed(p[3:], 1100)
    reply = core.take_host()
    assert struct.unpack("<I", reply[9:13])[0] == 5 and core.take_bus() == b""


# ---- firmware core: servo traffic passes unchanged ------------------------------------
@needs_cxx
def test_servo_packets_pass_byte_for_byte(so):
    config = calibrated(load_config(V20))
    core = Core(so)
    port = BridgePort(servos_for(config), core)
    with Robot(config, port_handlers={"opencr": port}, settle_s=0) as robot:
        robot.read_state()
        robot.arm(["R_knee", "L_knee"])
        robot.command_positions({"R_knee": 0.05, "L_knee": -0.05})
    sent = b"".join(port.tx_log)
    assert b"".join(port.bus_log) == sent  # every servo byte reached the bus, nothing added or dropped
    assert port.host_log == []


@needs_cxx
def test_broadcast_passes_at_once(so):
    core = Core(so)
    stop = packet(254, 0x03, [64, 0, 0])  # torque off to every servo
    core.feed(stop, 5000)
    assert core.take_bus() == stop  # no poll needed: nothing is held back


@needs_cxx
def test_stalled_header_is_released_after_timeout(so):
    core = Core(so)
    core.feed(b"\xFF\xFF", 1000)
    assert core.take_bus() == b""         # could be the start of an ID-200 packet: held
    core.poll(1100)
    assert core.take_bus() == b""
    core.poll(1000 + 300)
    assert core.take_bus() == b"\xFF\xFF"  # released unchanged
    core.feed(b"\x12\xFF\x34", 2000)       # not a header: goes straight through
    assert core.take_bus() == b"\x12\xFF\x34"
    p = packet(200, 0x02, [112, 0, 4, 0])  # HOLD_TIMEOUTS counter
    core.feed(p, 10_000)
    assert struct.unpack("<I", core.take_host()[9:13])[0] >= 1


@needs_cxx
def test_garbage_then_servo_packet(so):
    core = Core(so)
    servo = packet(3, 0x01)
    core.feed(b"\x00\xFF\x7E" + servo, 1000)
    core.poll(2000)
    assert core.take_bus() == b"\x00\xFF\x7E" + servo


# ---- Jetson side: frames ---------------------------------------------------------------
def test_body_from_board_matches_the_design():
    r = oi.check_rotation(oi.R_BODY_BOARD_V17)
    assert oi.det(r) == pytest.approx(1.0)
    assert oi.matvec(r, [1, 0, 0]) == [0, 1, 0]    # board x -> body +y
    assert oi.matvec(r, [0, 1, 0]) == [0, 0, -1]   # board y -> body -z
    assert oi.matvec(r, [0, 0, 1]) == [-1, 0, 0]   # board z (component face) -> body -x
    # standing upright: gravity (down, body -z) is board +y; the accelerometer reads board -y
    assert oi.matvec(r, [0, 1, 0]) == [0, 0, -1]
    assert oi.matvec(r, [0, -9.8, 0]) == pytest.approx([0, 0, 9.8])


def test_rotation_checks_refuse_mirrors_and_skew():
    mirror = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    with pytest.raises(ValueError, match="det"):
        oi.check_rotation(mirror)
    with pytest.raises(ValueError, match="orthonormal"):
        oi.check_rotation([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])


def test_roll_pitch_signs():
    assert oi.roll_pitch_from_gravity([0, 0, -1]) == pytest.approx([0, 0])
    a = math.radians(20)
    # nose down (positive pitch about +y): gravity gains +x in the body frame
    assert oi.roll_pitch_from_gravity([math.sin(a), 0, -math.cos(a)]) == pytest.approx([0, a])
    # right side down (positive roll about +x): gravity gains -y
    assert oi.roll_pitch_from_gravity([0, -math.sin(a), -math.cos(a)]) == pytest.approx([a, 0])


def test_yaml_imu_section_loads():
    config = load_config(V20)
    assert config.imu["virtual_id"] == 200
    assert config.imu["r_body_board"] == oi.R_BODY_BOARD_V17
    bus = DxlBus("fake", 1_000_000, {"R_knee": (4, "XL430-W250")}, port_handler=FakePort([]))
    imu = oi.OpenCRImu.from_config(bus, config.imu)
    assert imu.r_body_chip == oi.R_BODY_BOARD_V17


def test_virtual_id_must_not_be_a_servo():
    bus = DxlBus("fake", 1_000_000, {"x": (200, "XL430-W250")}, port_handler=FakePort([]))
    with pytest.raises(ValueError, match="also servo"):
        oi.OpenCRImu(bus)


def test_stock_bridge_gives_a_clear_error():
    bus = DxlBus("fake", 1_000_000, {"R_knee": (4, "XL430-W250")}, port_handler=FakePort([FakeServo(4)]),
                 settle_s=0)
    with bus:
        with pytest.raises(DxlError, match="opencr_bridge_imu"):
            oi.OpenCRImu(bus).open()


# ---- Jetson side through the firmware core ---------------------------------------------
@needs_cxx
def test_opencr_imu_reads_body_frame_values(so):
    # robot pitched 20 deg nose down; values the chip would see with identity chip->board
    a = math.radians(20)
    r = oi.R_BODY_BOARD_V17
    g_body = [math.sin(a), 0.0, -math.cos(a)]
    w_body = [0.5, -1.0, 2.0]
    rt = oi.transpose(r)
    g_chip, w_chip = oi.matvec(rt, g_body), oi.matvec(rt, w_body)
    acc_chip = [-9.80665 * c for c in g_chip]
    # a chip quaternion whose gravity matches g_chip: world <- chip = (world <- body)(body <- chip)
    q_body = [math.cos(a / 2), 0.0, math.sin(a / 2), 0.0]
    q_chip = oi.matrix_to_quat(oi.matmul(oi.quat_to_matrix(q_body), r))
    core, port = bridge_bus(so, [FakeServo(4)], timestamp_us=10, count=3, status=0x07, quat=q_chip,
                            gyro=w_chip, acc=acc_chip)
    bus = DxlBus("fake", 1_000_000, {"R_knee": (4, "XL430-W250")}, port_handler=port, settle_s=0)
    with bus:
        imu = oi.OpenCRImu(bus).open()
        assert imu.firmware_version == 1 and imu.config == {"imu_rate_hz": 200, "gyro_fs_dps": 2000, "acc_fs_g": 2}
        out = imu.read()
        assert out["gravity_b"] == pytest.approx(g_body, abs=1e-5)
        assert out["ang_vel_b"] == pytest.approx(w_body, abs=1e-5)
        assert out["acc_b"] == pytest.approx([-9.80665 * c for c in g_body], abs=1e-4)
        assert out["quat"] == pytest.approx(q_body, abs=1e-5)
        assert out["roll_pitch"] == pytest.approx([0.0, a], abs=1e-5)
        assert out["stale"] is False and imu.read()["stale"] is True  # same sample read twice
        assert out["read_s"] >= 0
        diag = imu.diagnostics()
        assert diag["virtual_packets"] >= 4


@needs_cxx
def test_not_ready_imu_is_refused(so):
    core, port = bridge_bus(so, [FakeServo(4)], status=0x03)  # gyro calibrated, filter not started
    bus = DxlBus("fake", 1_000_000, {"R_knee": (4, "XL430-W250")}, port_handler=port, settle_s=0)
    with bus:
        with pytest.raises(DxlError, match="not ready"):
            oi.OpenCRImu(bus).open(wait_s=0.1)
        imu = oi.OpenCRImu(bus, require_calibrated=False).open()
        with pytest.raises(DxlError, match="not ready"):
            oi.OpenCRImu(bus).read()
        assert imu.read_raw()["status"] == 0x03


@needs_cxx
def test_filter_disagreeing_with_accel_is_refused(so):
    # filter says chip z up, accelerometer says chip -y up (the Kitoru mounting): not converged
    core, port = bridge_bus(so, [FakeServo(4)], status=0x07, quat=[1, 0, 0, 0], acc=[0, -9.8, 0])
    bus = DxlBus("fake", 1_000_000, {"R_knee": (4, "XL430-W250")}, port_handler=port, settle_s=0)
    with bus:
        with pytest.raises(DxlError, match="filter vs accel 1.41"):
            oi.OpenCRImu(bus).open(wait_s=0.1)
        core.set_imu(status=0x07, quat=oi.matrix_to_quat(oi.R_BODY_BOARD_V17), acc=[0, -9.8, 0])
        oi.OpenCRImu(bus).open(wait_s=0.1)  # agrees: board y down -> upright robot


@needs_cxx
def test_policy_io_with_servos_and_imu_on_one_port(so):
    config = load_config(V20)
    core = Core(so)
    core.set_imu(count=1)
    port = BridgePort(servos_for(config), core)
    with Robot(config, port_handlers={"opencr": port}, settle_s=0) as robot:
        imu = oi.OpenCRImu.from_config(robot.buses["opencr"], config.imu).open()
        io = PolicyIO(robot, imu=imu)
        obs = io.observe()
        assert len(obs["q"]) == 15 and set(obs["timing"]) == {"servo_read_s", "imu_read_s"}
        io.observe()
        summary = io.timing.summary()
        assert summary["servo_read"]["count"] == 2 and summary["imu_read"]["count"] == 2
    # identity chip quaternion with identity chip->board: board z up -> body -x up
    assert obs["imu"]["gravity_b"] == pytest.approx(oi.matvec(oi.R_BODY_BOARD_V17, [0, 0, -1]), abs=1e-6)


@needs_cxx
def test_imu_check_tool(so, capsys):
    config = load_config(V20)
    core = Core(so)
    core.set_imu(count=1)
    port = BridgePort(servos_for(config), core)
    imu_check.main(["--seconds", "0.02", "--hz", "200", "--timing", "3"], port_handlers={"opencr": port}, settle_s=0)
    out = capsys.readouterr().out
    assert "ID 200: OpenCR IMU firmware 1" in out and "IMU read: mean" in out and "servo sync read: mean" in out
    # chip at rest with z up and identity chip->board: board z is body -x, so gravity is body +x
    assert "g_body (+1.00, " in out and "[g_chip (" in out
    assert all(s.get("TORQUE_ENABLE") == 0 for s in port.servos.values())
    imu_check.main(["--seconds", "0", "--no-servos"], port_handlers={"opencr": BridgePort([], core)}, settle_s=0)
    assert "ID 200" in capsys.readouterr().out
