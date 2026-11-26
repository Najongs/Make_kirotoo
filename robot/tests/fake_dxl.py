"""Virtual X430 servos behind a fake serial port, for running the real dynamixel-sdk offline.

FakePort subclasses the SDK PortHandler and answers Protocol 2.0 packets byte by byte
(PING, READ, WRITE, SYNC READ, SYNC WRITE, broadcast), so GroupSyncRead/GroupSyncWrite
packets built by the SDK are checked as they would go on the wire.
"""

from __future__ import annotations

import serial
from dynamixel_sdk import PortHandler
from dynamixel_sdk.protocol2_packet_handler import Protocol2PacketHandler

from kitoru_hw import control_table as ct

PING, READ, WRITE, SYNC_READ, SYNC_WRITE = 0x01, 0x02, 0x03, 0x82, 0x83
ACCESS_ERROR = 0x07
EEPROM_END = 64


class FakeServo:
    def __init__(self, motor_id, model="XL430-W250", position=2048, voltage_v=12.0, temperature_c=35,
                 blocked=None):
        m = ct.MODELS[model]
        self.memory = bytearray(256)
        self.blocked = blocked  # (lo_tick, hi_tick): a part stops the horn outside this
        for name, value in (("MODEL_NUMBER", m.number), ("FIRMWARE_VERSION", 46), ("ID", motor_id),
                            ("BAUD_RATE", 3), ("RETURN_DELAY_TIME", 250), ("OPERATING_MODE", 3),
                            ("TEMPERATURE_LIMIT", m.temperature_limit_c),
                            ("MAX_VOLTAGE_LIMIT", round(m.max_voltage_v * 10)),
                            ("MIN_VOLTAGE_LIMIT", round(m.min_voltage_v * 10)), ("PWM_LIMIT", 885),
                            ("VELOCITY_LIMIT", m.velocity_limit), ("MAX_POSITION_LIMIT", 4095),
                            ("STATUS_RETURN_LEVEL", 2), ("GOAL_PWM", 885), ("GOAL_POSITION", position),
                            ("PRESENT_POSITION", position), ("PRESENT_INPUT_VOLTAGE", round(voltage_v * 10)),
                            ("PRESENT_TEMPERATURE", temperature_c)):
            self.set(name, value)

    @property
    def motor_id(self):
        return self.memory[7]

    def set(self, name, value):
        address, size, _, _ = ct.ITEMS[name]
        self.memory[address:address + size] = ct.to_unsigned(value, size).to_bytes(size, "little")

    def get(self, name):
        address, size, signed, _ = ct.ITEMS[name]
        value = int.from_bytes(self.memory[address:address + size], "little")
        return ct.to_signed(value, size) if signed else value

    def write(self, address, data):
        if address < EEPROM_END and self.get("TORQUE_ENABLE"):
            return ACCESS_ERROR
        self.memory[address:address + len(data)] = bytes(data)
        if self.get("TORQUE_ENABLE"):
            goal = self.get("GOAL_POSITION")
            if self.blocked:
                goal = min(max(goal, self.blocked[0]), self.blocked[1])
            self.set("PRESENT_POSITION", goal)  # an ideal servo reaches the goal at once
        return 0


class FakePort(PortHandler):
    def __init__(self, servos, port_name="fake"):
        super().__init__(port_name)
        self.servos = {s.motor_id: s for s in servos}
        self.handler = Protocol2PacketHandler()
        self.rx = []
        self.tx_log = []
        self.unplugged = False
        self.drop_replies = 0

    # serial side -------------------------------------------------------
    def setupPort(self, cflag_baud):
        self.is_open = True
        self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
        return True

    def closePort(self):
        self.is_open = False

    def clearPort(self):
        self.rx = []

    def getBytesAvailable(self):
        return len(self.rx)

    def readPort(self, length):
        data, self.rx = self.rx[:length], self.rx[length:]
        return data

    def writePort(self, packet):
        if self.unplugged:
            raise serial.SerialException("device disconnected")
        packet = list(packet)
        self.tx_log.append(bytes(packet))
        self._handle(self.handler.removeStuffing(list(packet)))
        return len(packet)

    # servo side --------------------------------------------------------
    def instructions(self, instruction):
        return [p for p in self.tx_log if p[7] == instruction]

    def _status(self, motor_id, error=0, params=()):
        if self.drop_replies:
            self.drop_replies -= 1
            return
        length = len(params) + 4
        packet = [0xFF, 0xFF, 0xFD, 0x00, motor_id, length & 0xFF, length >> 8, 0x55, error] + list(params) + [0, 0]
        crc = self.handler.updateCRC(0, packet, len(packet) - 2)
        packet[-2:] = [crc & 0xFF, crc >> 8]
        self.rx += packet

    def _handle(self, p):
        motor_id, instruction = p[4], p[7]
        length = p[5] | p[6] << 8
        params = p[8:8 + length - 3]
        targets = list(self.servos.values()) if motor_id == ct.BROADCAST_ID else \
            [self.servos[motor_id]] if motor_id in self.servos else []
        word = lambda i: params[i] | params[i + 1] << 8  # noqa: E731
        if instruction == PING:
            for servo in targets:
                m = servo.memory
                self._status(servo.motor_id, 0, [m[0], m[1], m[6]])
        elif instruction == READ:
            address, size = word(0), word(2)
            for servo in targets:
                self._status(servo.motor_id, 0, servo.memory[address:address + size])
        elif instruction == WRITE:
            address, data = word(0), params[2:]
            for servo in targets:
                error = servo.write(address, data)
                if motor_id != ct.BROADCAST_ID:
                    self._status(servo.motor_id, error)
        elif instruction == SYNC_READ:
            address, size = word(0), word(2)
            for target_id in params[4:]:
                if target_id in self.servos:
                    self._status(target_id, 0, self.servos[target_id].memory[address:address + size])
        elif instruction == SYNC_WRITE:
            address, size = word(0), word(2)
            body = params[4:]
            for i in range(0, len(body), size + 1):
                if body[i] in self.servos:
                    self.servos[body[i]].write(address, body[i + 1:i + 1 + size])
