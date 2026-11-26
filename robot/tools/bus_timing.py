"""Print how many Hz one sync read + sync write cycle allows (no hardware needed).

  python tools/bus_timing.py
"""

from __future__ import annotations

import _common  # noqa: F401  (sets sys.path)
from kitoru_hw.timing import cycle_time

BAUDS = (1_000_000, 2_000_000, 4_000_000)
CASES = (
    ("wire only, Return Delay 0", dict(return_delay_us=0)),
    ("Return Delay default 250 (500 us)", dict(return_delay_us=500)),
    ("Return Delay 0 + USB 2 ms + host 0.1 ms/servo [estimate]", dict(usb_ms=2.0, per_servo_host_ms=0.1)),
    ("same, Fast Sync Read [estimate]", dict(usb_ms=2.0, per_servo_host_ms=0.1, fast_sync_read=True)),
)


def table(counts=(15, 17)):
    lines = []
    for n in counts:
        lines.append(f"servos {n}: read 10 bytes (126..135) + write Goal Position 4 bytes")
        lines.append("  case" + "".join(f" | {b // 1_000_000} Mbps" for b in BAUDS))
        for label, kwargs in CASES:
            cells = []
            for baud in BAUDS:
                t = cycle_time(n, baud, **kwargs)
                cells.append(f" | {t.total_ms:5.2f} ms {t.rate_hz:4.0f} Hz")
            lines.append(f"  {label}" + "".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
