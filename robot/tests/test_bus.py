import pytest

from fake_dxl import SYNC_READ, SYNC_WRITE, FakePort, FakeServo
from kitoru_hw.dxl_bus import DxlBus, DxlCommError, DxlDisconnected, DxlError, scan

MOTORS = {"a": (1, "XL430-W250"), "b": (2, "XC430-W240"), "c": (3, "XM430-W350")}


def make_bus(servos=None, **kwargs):
    servos = servos or [FakeServo(1, "XL430-W250", 2000), FakeServo(2, "XC430-W240", 2100),
                        FakeServo(3, "XM430-W350", 2200)]
    port = FakePort(servos)
    return DxlBus("fake", 1_000_000, MOTORS, port_handler=port, settle_s=0, **kwargs), port


def test_sync_read_one_packet_for_all_motors():
    bus, port = make_bus()
    port.servos[2].set("PRESENT_VELOCITY", -7)
    port.servos[3].set("PRESENT_LOAD_OR_CURRENT", -120)
    with bus:
        port.tx_log.clear()
        state = bus.read_state_raw()
    assert [s["position_tick"] for s in state.values()] == [2000, 2100, 2200]
    assert state["b"]["velocity"] == -7 and state["c"]["load_or_current"] == -120
    (packet,) = port.tx_log
    assert packet[7] == SYNC_READ
    # start address 126, length 10, then the three IDs
    assert list(packet[8:15]) == [126, 0, 10, 0, 1, 2, 3]


def test_sync_write_packet_bytes_and_effect():
    bus, port = make_bus()
    with bus:
        for motor_id in (1, 2, 3):
            port.servos[motor_id].set("TORQUE_ENABLE", 1)
        port.tx_log.clear()
        bus.sync_write("GOAL_POSITION", {"a": 1000, "b": 2048, "c": 3000})
    (packet,) = port.tx_log
    assert packet[4] == 0xFE and packet[7] == SYNC_WRITE
    assert list(packet[8:12]) == [116, 0, 4, 0]
    body = list(packet[12:-2])
    assert body == [1, *1000 .to_bytes(4, "little"), 2, *2048 .to_bytes(4, "little"), 3, *3000 .to_bytes(4, "little")]
    assert [port.servos[i].get("PRESENT_POSITION") for i in (1, 2, 3)] == [1000, 2048, 3000]


def test_sync_write_refuses_eeprom():
    bus, _ = make_bus()
    with bus, pytest.raises(DxlError, match="EEPROM"):
        bus.sync_write("PWM_LIMIT", {"a": 265})


def test_health_block():
    bus, port = make_bus()
    port.servos[1].set("PRESENT_INPUT_VOLTAGE", 108)
    port.servos[3].set("HARDWARE_ERROR_STATUS", 1 << 2)
    with bus:
        health = bus.read_health()
    assert health["a"]["voltage_v"] == pytest.approx(10.8)
    assert health["b"]["temperature_c"] == 35
    assert health["c"]["hardware_error"] == 4


def test_model_mismatch_on_open():
    bus, port = make_bus([FakeServo(1, "XL430-W250"), FakeServo(2, "XM430-W350"), FakeServo(3, "XM430-W350")])
    with pytest.raises(DxlError, match="expected XC430-W240"):
        bus.open()
    assert not port.is_open


def test_missing_servo_named_in_error():
    bus, port = make_bus()
    with bus:
        del port.servos[2]
        with pytest.raises(DxlCommError, match=r"b \(ID 2\)"):
            bus.read_state_raw()


def test_lost_replies_then_disconnected():
    bus, port = make_bus(max_consecutive_failures=2)
    with bus:
        port.drop_replies = 1
        with pytest.raises(DxlCommError):
            bus.read("a", "TORQUE_ENABLE")
        assert bus.read("a", "TORQUE_ENABLE") == 0  # a good packet resets the counter
        port.drop_replies = 10
        with pytest.raises(DxlCommError):
            bus.read("a", "TORQUE_ENABLE")
        with pytest.raises(DxlDisconnected):
            bus.read("a", "TORQUE_ENABLE")


def test_unplugged_usb_raises_disconnected_and_frees_port():
    bus, port = make_bus()
    with bus:
        port.unplugged = True
        with pytest.raises(DxlDisconnected, match="serial port failed"):
            bus.sync_write("GOAL_PWM", {"a": 265})
        assert port.is_using is False
        port.unplugged = False
        assert bus.read("a", "MODEL_NUMBER") == 1060


def test_broadcast_torque_off():
    bus, port = make_bus()
    with bus:
        for servo in port.servos.values():
            servo.set("TORQUE_ENABLE", 1)
        port.tx_log.clear()
        bus.torque_off_broadcast()
    assert len(port.tx_log) == 1 and port.tx_log[0][4] == 0xFE
    assert all(s.get("TORQUE_ENABLE") == 0 for s in port.servos.values())


def test_scan_broadcast_ping():
    port = FakePort([FakeServo(6), FakeServo(12, "XM430-W350"), FakeServo(7, "XC430-W240")])
    port.setBaudRate(1_000_000)
    assert scan(port) == {6: ("XL430-W250", 46), 7: ("XC430-W240", 46), 12: ("XM430-W350", 46)}
