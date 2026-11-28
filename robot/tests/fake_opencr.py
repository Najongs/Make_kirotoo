"""The OpenCR bridge firmware core (C++) between the real dynamixel-sdk and the fake servos.

build_core() compiles robot/firmware/opencr_bridge_imu/bridge_core.cpp with the host
compiler into a shared library (extern "C" API, -DKBRIDGE_C_API). BridgePort is a FakePort
whose bytes first go through that core: bytes the core sends to the bus are split into
packets for the virtual servos; bytes the core sends to the host become the SDK's rx.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
import subprocess

from fake_dxl import FakePort

ROBOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(ROBOT_DIR, "firmware", "opencr_bridge_imu")


def compiler():
    return shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")


def build_core(out_dir) -> ctypes.CDLL:
    cxx = compiler()
    if cxx is None:
        raise RuntimeError("no C++ compiler")
    lib = os.path.join(str(out_dir), "libkbridge.so")
    subprocess.run([cxx, "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "-DKBRIDGE_C_API",
                    os.path.join(CORE_DIR, "bridge_core.cpp"), "-o", lib], check=True)
    so = ctypes.CDLL(lib)
    u8p, f32p, sz, vp = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_void_p
    so.kb_create.restype, so.kb_create.argtypes = vp, [ctypes.c_uint8]
    so.kb_destroy.argtypes = [vp]
    so.kb_feed_host.argtypes = [vp, u8p, sz, ctypes.c_uint32]
    so.kb_poll.argtypes = [vp, ctypes.c_uint32]
    so.kb_take_bus.restype, so.kb_take_bus.argtypes = sz, [vp, u8p, sz]
    so.kb_take_host.restype, so.kb_take_host.argtypes = sz, [vp, u8p, sz]
    so.kb_set_imu.argtypes = [vp, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint8, f32p, f32p, f32p]
    so.kb_take_gyro_calibrate_request.restype, so.kb_take_gyro_calibrate_request.argtypes = ctypes.c_int, [vp]
    so.kb_note_loop_us.argtypes = [vp, ctypes.c_uint32]
    so.kb_crc16.restype, so.kb_crc16.argtypes = ctypes.c_uint16, [u8p, sz]
    so.kb_add_stuffing.restype, so.kb_add_stuffing.argtypes = sz, [u8p, sz, u8p, sz]
    so.kb_gravity_from_quat.argtypes = [f32p, f32p]
    return so


def u8(data):
    data = bytes(data)
    return (ctypes.c_uint8 * max(len(data), 1)).from_buffer_copy(data or b"\0"), len(data)


def f32(values):
    return (ctypes.c_float * len(values))(*values)


class Core:
    """Thin Python handle around one kbridge::Bridge."""

    def __init__(self, so, virtual_id=200):
        self.so = so
        self.h = so.kb_create(virtual_id)

    def __del__(self):
        try:
            self.so.kb_destroy(self.h)
        except Exception:
            pass

    def feed(self, data, now_us):
        buf, n = u8(data)
        self.so.kb_feed_host(self.h, buf, n, now_us)

    def poll(self, now_us):
        self.so.kb_poll(self.h, now_us)

    def _take(self, fn):
        out = (ctypes.c_uint8 * 8192)()
        n = fn(self.h, out, len(out))
        return bytes(out[:n])

    def take_bus(self):
        return self._take(self.so.kb_take_bus)

    def take_host(self):
        return self._take(self.so.kb_take_host)

    def set_imu(self, timestamp_us=0, count=0, status=0x03, quat=(1, 0, 0, 0), gyro=(0, 0, 0), acc=(0, 0, 9.80665)):
        self.so.kb_set_imu(self.h, timestamp_us, count, status, f32(quat), f32(gyro), f32(acc))


def split_packets(stream: bytes):
    """Cut a Protocol 2.0 byte stream into whole packets (header + LEN), leftovers dropped."""
    packets, i = [], 0
    while i + 7 <= len(stream):
        if stream[i:i + 4] != b"\xFF\xFF\xFD\x00":
            i += 1
            continue
        length = stream[i + 5] | stream[i + 6] << 8
        end = i + 7 + length
        if end > len(stream):
            break
        packets.append(stream[i:end])
        i = end
    return packets


class BridgePort(FakePort):
    """FakePort with the firmware core in front of the servos (the OpenCR in the middle)."""

    def __init__(self, servos, core: Core, step_us=50):
        super().__init__(servos)
        self.core = core
        self.now_us = 1000
        self.step_us = step_us
        self.bus_log = []    # byte runs the core put on the TTL bus
        self.host_log = []   # byte runs the core answered itself

    def writePort(self, packet):
        if self.unplugged:
            import serial
            raise serial.SerialException("device disconnected")
        packet = bytes(packet)
        self.tx_log.append(packet)
        self.now_us += self.step_us
        self.core.feed(packet, self.now_us)
        self.pump()
        return len(packet)

    def pump(self):
        bus = self.core.take_bus()
        if bus:
            self.bus_log.append(bus)
            for p in split_packets(bus):
                self._handle(self.handler.removeStuffing(list(p)))
        host = self.core.take_host()
        if host:
            self.host_log.append(host)
            self.rx += list(host)


def sample_bytes(timestamp_us, count, status, quat, gravity, gyro, acc) -> bytes:
    return struct.pack("<IIB3x4f3f3f3f", timestamp_us, count, status, *quat, *gravity, *gyro, *acc)
