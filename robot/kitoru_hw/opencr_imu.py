"""OpenCR on-board IMU (ICM-20648) read through the bridge firmware's virtual Dynamixel ID.

Firmware: robot/firmware/opencr_bridge_imu (README 3절). It answers PING / READ / WRITE for
one virtual ID (default 200) from a small control table; servo packets pass unchanged.
The IMU shares the servo serial port, so OpenCRImu borrows an open DxlBus instead of
opening the port a second time.

Frames
  chip  : axes of the IMU values the firmware sends (ROBOTIS IMU library, identity remap)
  board : OpenCR STEP frame (z out of the component face, x / y in the board plane)
  body  : robot torso frame, the same axes as the MuJoCo "torso" body (x forward, z up)

  v_body = R_body_board @ R_board_chip @ v_chip

R_body_board comes from the design (code_v17/build17.py oc_box: board z -> body -x,
board x -> body +y, board y -> body -z). R_board_chip is identity until checked on the
real board [확인 필요]. Both can be overridden in the YAML `imu:` section.

The policy (kitoru-rl envs/kitoru_walk/mdp.py) uses
  meas_gravity_b : gravity direction in the body frame, unit vector, (0, 0, -1) standing
  meas_ang_vel_b : body angular velocity in the body frame, rad/s
which read() returns as "gravity_b" and "ang_vel_b".
"""

from __future__ import annotations

import math
import struct
import time
from typing import Dict, List, Mapping, Optional, Sequence

from .dxl_bus import DxlBus, DxlError

# ---- control table of the virtual ID (firmware bridge_core.h) -------------------------
MODEL_NUMBER = 0x4B49
DEFAULT_VIRTUAL_ID = 200
ADDR_FIRMWARE_VERSION = 6
ADDR_CONFIG = 10                  # IMU_RATE_HZ u16, GYRO_FS_DPS u16, ACC_FS_G u8
SAMPLE_ADDR, SAMPLE_LEN = 32, 64  # TIMESTAMP_US .. ACC_Z, one READ
DIAG_ADDR, DIAG_LEN = 100, 16     # LOOP_MAX_US, IMU_UPDATE_MAX_US, VIRTUAL_PACKETS, HOLD_TIMEOUTS
ADDR_RESET_DIAG = 116
ADDR_GYRO_CALIBRATE = 117
STATUS_IMU_FOUND = 0x01
STATUS_GYRO_CALIBRATED = 0x02
STATUS_ATTITUDE_READY = 0x04
READY = STATUS_IMU_FOUND | STATUS_GYRO_CALIBRATED | STATUS_ATTITUDE_READY
# timestamp u32, count u32, status u8, pad x3, quat f32x4, gravity f32x3, gyro f32x3, acc f32x3
_SAMPLE = struct.Struct("<IIB3x4f3f3f3f")
assert _SAMPLE.size == SAMPLE_LEN

Matrix = List[List[float]]
IDENTITY: Matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# Columns are the board axes written in body coordinates: x_b -> +y, y_b -> -z, z_b -> -x.
R_BODY_BOARD_V17: Matrix = [[0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0]]
IMU_POS_BODY_M = (-0.070, 0.000, 0.280)  # design position, for reference only (not used)


# ---- small 3x3 helpers (no numpy on the Jetson test path) -----------------------------
def matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def matvec(a: Matrix, v: Sequence[float]) -> List[float]:
    return [a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2] for i in range(3)]


def transpose(a: Matrix) -> Matrix:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def det(a: Matrix) -> float:
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))


def check_rotation(a, label: str = "rotation") -> Matrix:
    """A proper rotation (orthonormal, det +1). A mirror (det -1) flips one axis sign silently."""
    m = [[float(x) for x in row] for row in a]
    if len(m) != 3 or any(len(row) != 3 for row in m):
        raise ValueError(f"{label}: need a 3x3 matrix")
    p = matmul(m, transpose(m))
    if any(abs(p[i][j] - (1.0 if i == j else 0.0)) > 1e-6 for i in range(3) for j in range(3)):
        raise ValueError(f"{label}: rows are not orthonormal")
    if det(m) < 0:
        raise ValueError(f"{label}: det = -1 (a mirror, one axis has the wrong sign)")
    return m


def quat_to_matrix(q: Sequence[float]) -> Matrix:
    """Hamilton w,x,y,z -> R that turns vectors of the rotated frame into the reference frame."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def matrix_to_quat(m: Matrix) -> List[float]:
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    return q if q[0] >= 0 else [-c for c in q]


def roll_pitch_from_gravity(g: Sequence[float]) -> List[float]:
    """ZYX roll / pitch (rad) from a body-frame gravity direction; upright (0, 0, -1) -> (0, 0)."""
    gx, gy, gz = g
    return [math.atan2(-gy, -gz), math.atan2(gx, math.hypot(gy, gz))]


def gravity_mismatch(sample: Mapping) -> float:
    """|filter gravity - (-accel direction)|; near 0 while still and converged."""
    acc = sample["acc_chip"]
    n = math.sqrt(sum(a * a for a in acc))
    if n == 0:
        return 2.0
    return math.sqrt(sum((g + a / n) ** 2 for g, a in zip(sample["gravity_chip"], acc)))


def decode_sample(data: bytes) -> Dict[str, object]:
    ts, count, status, *rest = _SAMPLE.unpack(bytes(data))
    return {"timestamp_us": ts, "sample_count": count, "status": status,
            "quat_chip": list(rest[0:4]), "gravity_chip": list(rest[4:7]),
            "gyro_chip": list(rest[7:10]), "acc_chip": list(rest[10:13])}


class OpenCRImu:
    """Same interface as NullImu: read() -> dict (or None when disabled).

    read() keys
      gravity_b  [3]  gravity direction in the body frame (unit, points down)
      ang_vel_b  [3]  rad/s in the body frame
      acc_b      [3]  m/s^2 in the body frame (reads about +9.8 along body +z standing still)
      quat       [4]  w,x,y,z body orientation from the Madgwick filter (yaw drifts, no magnetometer)
      gyro       [3]  same as ang_vel_b (key kept from the NullImu comment)
      roll_pitch [2]  rad, for safety.tilt_deg
      timestamp_us, sample_count, status, stale (same sample as last read), read_s (seconds)
      gravity_chip, gyro_chip, acc_chip  the same values before rotation (to find r_board_chip on the real board)
    """

    def __init__(self, bus: DxlBus, virtual_id: int = DEFAULT_VIRTUAL_ID,
                 r_body_board: Sequence[Sequence[float]] = R_BODY_BOARD_V17,
                 r_board_chip: Sequence[Sequence[float]] = IDENTITY,
                 require_calibrated: bool = True):
        if not 0 <= virtual_id <= 252:
            raise ValueError(f"virtual ID {virtual_id} out of range")
        used = {motor_id: name for name, (motor_id, _) in bus.motors.items()}
        if virtual_id in used:
            raise ValueError(f"virtual IMU ID {virtual_id} is also servo {used[virtual_id]}; pick another ID "
                             f"(firmware VIRTUAL_ID) or renumber the servo")
        self.bus = bus
        self.virtual_id = virtual_id
        self.r_body_board = check_rotation(r_body_board, "imu.r_body_board")
        self.r_board_chip = check_rotation(r_board_chip, "imu.r_board_chip")
        self.r_body_chip = matmul(self.r_body_board, self.r_board_chip)
        self.require_calibrated = require_calibrated
        self.firmware_version: Optional[int] = None
        self.config: Dict[str, int] = {}
        self._last_count: Optional[int] = None
        self.last_read_s = 0.0

    @classmethod
    def from_config(cls, bus: DxlBus, imu_cfg: Optional[Mapping] = None, **kwargs) -> "OpenCRImu":
        """Build from the YAML `imu:` section (keys: virtual_id, r_body_board, r_board_chip)."""
        cfg = dict(imu_cfg or {})
        return cls(bus, virtual_id=int(cfg.get("virtual_id", DEFAULT_VIRTUAL_ID)),
                   r_body_board=cfg.get("r_body_board") or R_BODY_BOARD_V17,
                   r_board_chip=cfg.get("r_board_chip") or IDENTITY, **kwargs)

    # ----- low level ------------------------------------------------------
    def _read(self, address: int, length: int, what: str) -> bytes:
        data, result, error = self.bus._io(self.bus.packet.readTxRx, self.bus.port, self.virtual_id, address, length)
        self.bus._check(result, error, f"Read {what} of OpenCR IMU (ID {self.virtual_id})")
        if len(data) != length:
            raise DxlError(f"OpenCR IMU: asked {length} bytes at {address}, got {len(data)}")
        return bytes(data)

    def _write1(self, address: int, value: int, what: str):
        result, error = self.bus._io(self.bus.packet.write1ByteTxRx, self.bus.port, self.virtual_id, address, value)
        self.bus._check(result, error, f"Write {what} of OpenCR IMU (ID {self.virtual_id})")

    # ----- public ---------------------------------------------------------
    def open(self, wait_s: float = 3.0, max_mismatch: float = 0.1) -> "OpenCRImu":
        """PING the virtual ID, check the firmware and wait until the attitude is usable.

        Usable: gyro bias measured, filter started, and the filter's gravity agrees with the
        accelerometer within `max_mismatch` (unit-vector distance, 0.1 ~ 5.7 deg) while still.
        Skipped when require_calibrated is False. The bus must already be open."""
        number, result, error = self.bus._io(self.bus.packet.ping, self.bus.port, self.virtual_id)
        try:
            self.bus._check(result, error, f"Ping OpenCR IMU (ID {self.virtual_id})")
        except DxlError as e:
            raise DxlError(f"{e}. Is robot/firmware/opencr_bridge_imu on the OpenCR? "
                           f"The stock usb_to_dxl bridge has no ID {self.virtual_id}.") from e
        if number != MODEL_NUMBER:
            raise DxlError(f"ID {self.virtual_id} answered model {number}, expected the OpenCR IMU ({MODEL_NUMBER})")
        self.firmware_version = self._read(ADDR_FIRMWARE_VERSION, 1, "firmware version")[0]
        rate, gyro_fs, acc_fs = struct.unpack("<HHB", self._read(ADDR_CONFIG, 5, "config"))
        self.config = {"imu_rate_hz": rate, "gyro_fs_dps": gyro_fs, "acc_fs_g": acc_fs}
        sample = decode_sample(self._read(SAMPLE_ADDR, SAMPLE_LEN, "sample"))
        if not sample["status"] & STATUS_IMU_FOUND:
            raise DxlError("OpenCR firmware runs, but its IMU did not start (cIMU::begin found no chip)")
        if not self.require_calibrated:
            return self
        deadline = time.monotonic() + wait_s
        while True:
            mismatch = gravity_mismatch(sample)
            if sample["status"] & READY == READY and mismatch <= max_mismatch:
                return self
            if time.monotonic() >= deadline:
                raise DxlError(f"OpenCR IMU not ready after {wait_s:.1f} s (status 0x{sample['status']:02x}, "
                               f"filter vs accel {mismatch:.2f}): keep the robot still after power-up and retry")
            time.sleep(0.05)
            sample = decode_sample(self._read(SAMPLE_ADDR, SAMPLE_LEN, "sample"))

    def read_raw(self) -> Dict[str, object]:
        start = time.perf_counter()
        sample = decode_sample(self._read(SAMPLE_ADDR, SAMPLE_LEN, "sample"))
        self.last_read_s = time.perf_counter() - start
        sample["read_s"] = self.last_read_s
        return sample

    def read(self) -> Dict[str, object]:
        s = self.read_raw()
        if self.require_calibrated and s["status"] & READY != READY:
            raise DxlError(f"OpenCR IMU is not ready (status 0x{s['status']:02x}: gyro bias not calibrated or "
                           f"filter not started). Keep the robot still after power-up, or call calibrate_gyro()")
        r = self.r_body_chip
        gravity_b = matvec(r, s["gravity_chip"])
        ang_vel_b = matvec(r, s["gyro_chip"])
        acc_b = matvec(r, s["acc_chip"])
        # quat_chip maps chip vectors to the filter's world frame; body -> world = (chip -> world)(body -> chip)
        quat_b = matrix_to_quat(matmul(quat_to_matrix(s["quat_chip"]), transpose(r)))
        stale = s["sample_count"] == self._last_count
        self._last_count = s["sample_count"]
        return {
            "gravity_b": gravity_b, "ang_vel_b": ang_vel_b, "acc_b": acc_b, "quat": quat_b, "gyro": ang_vel_b,
            "roll_pitch": roll_pitch_from_gravity(gravity_b),
            "timestamp_us": s["timestamp_us"], "sample_count": s["sample_count"], "status": s["status"],
            "stale": stale, "read_s": s["read_s"],
            "gravity_chip": s["gravity_chip"], "gyro_chip": s["gyro_chip"], "acc_chip": s["acc_chip"],
        }

    def calibrate_gyro(self, wait_s: float = 6.0, poll_s: float = 0.1) -> bool:
        """Ask the firmware to re-measure the gyro bias. Keep the robot still. True when done."""
        self._write1(ADDR_GYRO_CALIBRATE, 1, "gyro calibrate")
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            time.sleep(poll_s)
            if self._read(ADDR_GYRO_CALIBRATE, 1, "gyro calibrate")[0] == 0:
                return True
        return False

    def diagnostics(self, reset: bool = False) -> Dict[str, int]:
        """Firmware timing counters (µs) since the last reset; for the relay-delay check (README 3절)."""
        loop_max, imu_max, packets, timeouts = struct.unpack("<IIII", self._read(DIAG_ADDR, DIAG_LEN, "diagnostics"))
        if reset:
            self._write1(ADDR_RESET_DIAG, 1, "reset diagnostics")
        return {"loop_max_us": loop_max, "imu_update_max_us": imu_max,
                "virtual_packets": packets, "hold_timeouts": timeouts}


def timestamp_delta_us(newer: int, older: int) -> int:
    """Firmware micros() wraps every 2^32 µs (~71.6 min)."""
    return (newer - older) % (1 << 32)
