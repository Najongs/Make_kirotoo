"""One Dynamixel Protocol 2.0 serial bus (OpenCR in USB bridge mode, or a U2D2).

The bus only knows motor names, IDs, models and raw register values. Angles,
limits and safety live in joints.py, safety.py and robot.py.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from dynamixel_sdk import COMM_SUCCESS, GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler

from . import control_table as ct


class DxlError(RuntimeError):
    """Servo answered with an error, or the bus is in a state we refuse to use."""


class DxlCommError(DxlError):
    """A packet was lost or corrupted (timeout, CRC, missing ID)."""


class DxlDisconnected(DxlError):
    """The serial port failed or too many packets in a row were lost."""


class DxlBus:
    """Owns one serial port. Motors are {name: (motor_id, model_name)}."""

    def __init__(self, port: str, baud: int = 1_000_000,
                 motors: Optional[Mapping[str, Tuple[int, str]]] = None,
                 port_handler: Optional[PortHandler] = None,
                 max_consecutive_failures: int = 3, settle_s: float = 0.25):
        self.motors = dict(motors or {})
        ids = [motor_id for motor_id, _ in self.motors.values()]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{port}: configure at least one motor and use unique IDs")
        for name, (motor_id, model) in self.motors.items():
            if not 0 <= motor_id <= 252:
                raise ValueError(f"{name}: invalid motor ID {motor_id}")
            if model not in ct.MODELS:
                raise ValueError(f"{name}: unknown model {model!r}, known: {', '.join(ct.MODELS)}")

        self.port = port_handler if port_handler is not None else PortHandler(port)
        self.baud = baud
        self.packet = PacketHandler(2.0)
        self.max_consecutive_failures = max_consecutive_failures
        self.settle_s = settle_s
        self.failures = 0
        self.is_open = False
        self._readers: Dict[tuple, GroupSyncRead] = {}

    # ----- open / close -------------------------------------------------
    def open(self, check_models: bool = True):
        if not self._io(self.port.setBaudRate, self.baud):
            raise DxlDisconnected(f"Could not open {self.port.port_name} at {self.baud} baud")
        self.is_open = True
        try:
            time.sleep(self.settle_s)
            if check_models:
                for name, (motor_id, model) in self.motors.items():
                    number, result, error = self._io(self.packet.ping, self.port, motor_id)
                    self._check(result, error, f"Ping {name} (ID {motor_id})")
                    if number != ct.MODELS[model].number:
                        found = ct.MODELS_BY_NUMBER.get(number)
                        raise DxlError(f"{name} (ID {motor_id}): expected {model} ({ct.MODELS[model].number}), "
                                       f"got {found.name if found else number}")
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if self.is_open:
            self.is_open = False
            try:
                self.port.closePort()
            except OSError:
                pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()

    # ----- error handling -------------------------------------------------
    def _io(self, fn: Callable, *args):
        """Run one SDK call; a serial exception means the USB cable or board is gone."""
        try:
            return fn(*args)
        except OSError as error:  # serial.SerialException is an OSError
            # The SDK leaves is_using set when writePort raises, which would block every later call.
            self.port.is_using = False
            self.failures = self.max_consecutive_failures
            raise DxlDisconnected(f"{self.port.port_name}: serial port failed: {error}") from error

    def _check(self, result: int, error: int = 0, operation: str = "DYNAMIXEL"):
        if result != COMM_SUCCESS:
            self.failures += 1
            message = f"{operation}: {self.packet.getTxRxResult(result)}"
            if self.failures >= self.max_consecutive_failures:
                raise DxlDisconnected(f"{message} ({self.failures} failures in a row, bus treated as lost)")
            raise DxlCommError(message)
        self.failures = 0
        if error:
            raise DxlError(f"{operation}: {self.packet.getRxPacketError(error)}")

    # ----- single register access ----------------------------------------
    def motor_id(self, name: str) -> int:
        return self.motors[name][0]

    def read(self, name: str, item_name: str) -> int:
        address, size, signed, _ = ct.item(item_name, self.motors[name][1])
        reader = {1: self.packet.read1ByteTxRx, 2: self.packet.read2ByteTxRx, 4: self.packet.read4ByteTxRx}[size]
        value, result, error = self._io(reader, self.port, self.motor_id(name), address)
        self._check(result, error, f"Read {item_name} of {name}")
        return ct.to_signed(value, size) if signed else value

    def write(self, name: str, item_name: str, value: int):
        address, size, _, _ = ct.item(item_name, self.motors[name][1])
        writer = {1: self.packet.write1ByteTxRx, 2: self.packet.write2ByteTxRx, 4: self.packet.write4ByteTxRx}[size]
        result, error = self._io(writer, self.port, self.motor_id(name), address, ct.to_unsigned(value, size))
        self._check(result, error, f"Write {item_name}={value} to {name}")

    # ----- group access ---------------------------------------------------
    def sync_read(self, address: int, length: int, names: Optional[Iterable[str]] = None) -> Dict[str, bytes]:
        """Read the same block from many motors with one Sync Read instruction."""
        names = list(self.motors if names is None else names)
        key = (address, length, tuple(names))
        reader = self._readers.get(key)
        if reader is None:
            reader = GroupSyncRead(self.port, self.packet, address, length)
            for name in names:
                if not reader.addParam(self.motor_id(name)):
                    raise DxlError(f"Could not register {name} for group read")
            self._readers[key] = reader
        for motor_id in reader.data_dict:
            reader.data_dict[motor_id] = []  # no stale data from the previous cycle
        result = self._io(reader.txRxPacket)
        if result != COMM_SUCCESS:
            # The SDK reads answers in ID order and stops at the first missing one.
            missing = [f"{n} (ID {self.motor_id(n)})" for n in names
                       if len(reader.data_dict.get(self.motor_id(n)) or []) < length]
            self._check(result, operation=f"Sync read {address}+{length}, no answer from {missing[0] if missing else '?'}")
        self.failures = 0
        data = {}
        for name in names:
            motor_id = self.motor_id(name)
            if not reader.isAvailable(motor_id, address, length):
                raise DxlCommError(f"Sync read {address}+{length}: no data from {name} (ID {motor_id})")
            data[name] = bytes(reader.data_dict[motor_id][:length])
        return data

    def sync_write(self, item_name: str, values: Mapping[str, int]):
        """Write one register on many motors with one Sync Write instruction (no reply)."""
        address, size, _, eeprom = ct.ITEMS[item_name]
        if eeprom:
            raise DxlError(f"{item_name} is an EEPROM item; write it per motor with torque off")
        writer = GroupSyncWrite(self.port, self.packet, address, size)
        for name, value in values.items():
            data = list(ct.to_unsigned(int(value), size).to_bytes(size, "little"))
            if not writer.addParam(self.motor_id(name), data):
                raise DxlError(f"Could not register {item_name} for {name}")
        result = self._io(writer.txPacket)
        self._check(result, operation=f"Sync write {item_name}")

    def read_state_raw(self, names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, int]]:
        """Position tick, velocity and load/current raw values for each motor (one packet)."""
        address, length = ct.STATE_BLOCK
        state = {}
        for name, block in self.sync_read(address, length, names).items():
            state[name] = {
                "load_or_current": ct.to_signed(int.from_bytes(block[0:2], "little"), 2),
                "velocity": ct.to_signed(int.from_bytes(block[2:6], "little"), 4),
                "position_tick": ct.to_signed(int.from_bytes(block[6:10], "little"), 4),
            }
        return state

    def read_health(self, names: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, float]]:
        """Input voltage, temperature and hardware error status for each motor (two packets)."""
        names = list(self.motors if names is None else names)
        address, length = ct.HEALTH_BLOCK
        blocks = self.sync_read(address, length, names)
        errors = self.sync_read(ct.ITEMS["HARDWARE_ERROR_STATUS"][0], 1, names)
        return {name: {
            "voltage_v": int.from_bytes(blocks[name][0:2], "little") * ct.VOLTAGE_UNIT_V,
            "temperature_c": blocks[name][2],
            "hardware_error": errors[name][0],
        } for name in names}

    def torque_off_broadcast(self):
        """One packet to ID 254: every servo on the wire turns torque off, no reply needed."""
        address = ct.ITEMS["TORQUE_ENABLE"][0]
        result = self._io(self.packet.write1ByteTxOnly, self.port, ct.BROADCAST_ID, address, 0)
        self._check(result, operation="Broadcast torque off")


def scan(port_handler: PortHandler) -> Dict[int, Tuple[str, int]]:
    """Broadcast ping on an open port. Returns {id: (model name or number, firmware)}."""
    packet = PacketHandler(2.0)
    found, result = packet.broadcastPing(port_handler)
    if result != COMM_SUCCESS and not found:
        return {}
    named = {}
    for motor_id, (number, firmware) in sorted(found.items()):
        model = ct.MODELS_BY_NUMBER.get(number)
        named[motor_id] = (model.name if model else f"model {number}", firmware)
    return named
