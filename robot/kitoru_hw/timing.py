"""How fast one control cycle (sync read state + sync write goal) can run on one bus.

Protocol 2.0 packet sizes (header 3 + reserved 1 + ID 1 + length 2 + instruction 1 + CRC 2):
  Sync Read instruction   14 + n            (address 2, length 2, one byte per ID)
  each status packet      11 + L            (adds the error byte)
  Fast Sync Read answer   8 + n * (L + 4)   (one packet; error, ID, data, CRC per servo) [approximate]
  Sync Write instruction  14 + n * (1 + L)  (no reply)
One byte on the wire is 10 bits (8N1). Every servo waits Return Delay Time before it answers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CycleTime:
    wire_ms: float
    return_delay_ms: float
    host_ms: float

    @property
    def total_ms(self) -> float:
        return self.wire_ms + self.return_delay_ms + self.host_ms

    @property
    def rate_hz(self) -> float:
        return 1000.0 / self.total_ms


def cycle_time(n: int, baud: int, read_len: int = 10, write_len: int = 4, return_delay_us: float = 0.0,
               usb_ms: float = 0.0, per_servo_host_ms: float = 0.0, fast_sync_read: bool = False) -> CycleTime:
    read_tx = 14 + n
    read_rx = 8 + n * (read_len + 4) if fast_sync_read else n * (11 + read_len)
    write_tx = 14 + n * (1 + write_len)
    wire_ms = (read_tx + read_rx + write_tx) * 10 / baud * 1000
    return CycleTime(wire_ms, n * return_delay_us / 1000, usb_ms + n * per_servo_host_ms)
