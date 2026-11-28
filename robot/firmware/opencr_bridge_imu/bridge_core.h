// Board-independent part of the Kitoru OpenCR bridge (no Arduino headers).
//
// The OpenCR passes Dynamixel Protocol 2.0 bytes between USB (host) and the TTL bus,
// like the ROBOTIS usb_to_dxl example. This core watches the host->bus stream and takes
// out only the packets addressed to one virtual ID (default 200). Those packets never go
// on the bus; the core answers them from a small control table filled with IMU data.
//
// Host->bus forwarding is cut-through: at most the first 5 bytes of a packet
// (FF FF FD 00 ID) are held until the ID is known, then the rest passes byte by byte.
// Broadcast (ID 254) and every servo ID pass at once. If a held header stalls for
// kHoldTimeoutUs the held bytes go to the bus unchanged, so a confused parser never
// blocks servo traffic.
//
// The same file builds on a PC (g++) for unit tests; define KBRIDGE_C_API to get a
// small extern "C" API used by robot/tests/test_firmware_core.py through ctypes.

#ifndef KITORU_BRIDGE_CORE_H_
#define KITORU_BRIDGE_CORE_H_

#include <stddef.h>
#include <stdint.h>

namespace kbridge {

// ---- Protocol 2.0 constants --------------------------------------------------
const uint8_t kBroadcastId = 0xFE;
const uint8_t kInstPing = 0x01;
const uint8_t kInstRead = 0x02;
const uint8_t kInstWrite = 0x03;
const uint8_t kInstStatus = 0x55;

const uint8_t kErrResultFail = 0x01;
const uint8_t kErrInstruction = 0x02;
const uint8_t kErrCrc = 0x03;
const uint8_t kErrDataRange = 0x04;
const uint8_t kErrDataLength = 0x05;
const uint8_t kErrAccess = 0x07;

// ---- Virtual device control table (little endian, like Dynamixel) ------------
const uint16_t kModelNumber = 0x4B49;   // "KI"; not a ROBOTIS model number
const uint8_t kFirmwareVersion = 1;
const uint8_t kTableVersion = 1;
const uint8_t kDefaultVirtualId = 200;

enum Address : uint16_t {
  ADDR_MODEL_NUMBER = 0,        // u16  R
  ADDR_FIRMWARE_VERSION = 6,    // u8   R
  ADDR_ID = 7,                  // u8   R  (set at compile time)
  ADDR_TABLE_VERSION = 8,       // u8   R
  ADDR_IMU_RATE_HZ = 10,        // u16  R  IMU.begin() rate
  ADDR_GYRO_FS_DPS = 12,        // u16  R  chip full scale (2000)
  ADDR_ACC_FS_G = 14,           // u8   R  chip full scale (2)
  // ---- one READ of 32..95 (64 bytes) gives a whole sample -------------------
  ADDR_TIMESTAMP_US = 32,       // u32  R  micros() when the sample was taken (wraps ~71.6 min)
  ADDR_SAMPLE_COUNT = 36,       // u32  R  +1 per IMU sample (same value twice = stale)
  ADDR_STATUS = 40,             // u8   R  bit0 IMU found, bit1 gyro bias calibrated, bit2 attitude started
  ADDR_QUAT_W = 44,             // f32 x4 R  Madgwick quaternion w,x,y,z (chip -> world, yaw drifts)
  ADDR_GRAVITY_X = 60,          // f32 x3 R  gravity direction in chip axes, unit vector, points DOWN
  ADDR_GYRO_X = 72,             // f32 x3 R  angular velocity in chip axes, rad/s
  ADDR_ACC_X = 84,              // f32 x3 R  acceleration in chip axes, m/s^2 (reads +9.8 UP at rest)
  // ---- diagnostics -----------------------------------------------------------
  ADDR_LOOP_MAX_US = 100,       // u32  R  longest loop() pass since reset
  ADDR_IMU_UPDATE_MAX_US = 104, // u32  R  longest IMU.update() since reset
  ADDR_VIRTUAL_PACKETS = 108,   // u32  R  packets answered by the virtual ID
  ADDR_HOLD_TIMEOUTS = 112,     // u32  R  held headers released by timeout
  ADDR_RESET_DIAG = 116,        // u8   W  write 1: zero 100..115
  ADDR_GYRO_CALIBRATE = 117,    // u8   RW write 1: re-measure gyro bias (keep still ~3 s); reads 1 while running
  kTableSize = 128
};

const uint16_t kSampleBlockAddr = ADDR_TIMESTAMP_US;
const uint16_t kSampleBlockLen = 64;

const uint8_t kStatusImuFound = 0x01;
const uint8_t kStatusGyroCalibrated = 0x02;
const uint8_t kStatusAttitudeReady = 0x04;  // filter started from the measured gravity

const uint32_t kHoldTimeoutUs = 300;      // release a stalled 1..4 byte header
const uint32_t kPacketTimeoutUs = 3000;   // give up a packet whose body stopped arriving
const size_t kCaptureMax = 192;            // largest virtual-ID packet we take in

uint16_t crc16(uint16_t crc_accum, const uint8_t* data, size_t size);
// Byte stuffing as in the ROBOTIS SDK (add/remove FD after FF FF FD from the instruction on).
// `packet` is a whole packet with CRC; returns the new total size. `out` needs room.
size_t add_stuffing(const uint8_t* packet, size_t size, uint8_t* out, size_t out_cap);
size_t remove_stuffing(const uint8_t* packet, size_t size, uint8_t* out, size_t out_cap);
// Gravity direction (points down, unit) in sensor axes from a Madgwick quaternion w,x,y,z.
void gravity_from_quat(const float q[4], float g[3]);
// Starting quaternion (yaw 0) whose gravity_from_quat() equals -a/|a| (a = accelerometer, any scale).
// Returns false for a zero vector. Without this the filter starts at (1,0,0,0) and needs ~23 s
// (beta 0.1, 200 Hz) to turn 90 deg, and never leaves an exact 180 deg start.
bool quat_from_accel(const float a[3], float q[4]);

struct ImuSample {
  uint32_t timestamp_us;
  uint32_t sample_count;
  uint8_t status;
  float quat[4];
  float gyro_rad_s[3];
  float acc_m_s2[3];
};

typedef void (*WriteFn)(void* ctx, const uint8_t* data, size_t size);

class Bridge {
 public:
  explicit Bridge(uint8_t virtual_id = kDefaultVirtualId);
  void set_outputs(WriteFn to_bus, WriteFn to_host, void* ctx);

  // Bytes that came from the host over USB.
  void feed_host(const uint8_t* data, size_t size, uint32_t now_us);
  // Call often: releases a stalled header / drops a stalled packet.
  void poll(uint32_t now_us);

  void set_imu_config(uint16_t rate_hz, uint16_t gyro_fs_dps, uint8_t acc_fs_g);
  void set_imu(const ImuSample& s);
  void note_loop_us(uint32_t us);
  void note_imu_update_us(uint32_t us);
  // The host asked for a gyro bias re-measurement (cleared by this call).
  bool take_gyro_calibrate_request();
  void set_gyro_calibrating(bool running);

  uint8_t virtual_id() const { return virtual_id_; }
  const uint8_t* table() const { return table_; }

 private:
  enum Mode { HUNT, PASS, CAPTURE };
  void out_bus(const uint8_t* d, size_t n);
  void out_host(const uint8_t* d, size_t n);
  void flush_bus();
  void hunt_byte(uint8_t b, uint32_t now_us);
  void release_held();
  void handle_capture();
  void reply(uint8_t error, const uint8_t* params, size_t n);
  void put_u16(uint16_t addr, uint16_t v);
  void put_u32(uint16_t addr, uint32_t v);
  void put_f32(uint16_t addr, float v);
  uint32_t get_u32(uint16_t addr) const;

  uint8_t virtual_id_;
  uint8_t table_[kTableSize];
  WriteFn to_bus_;
  WriteFn to_host_;
  void* ctx_;

  Mode mode_;
  uint8_t held_[8];
  size_t held_n_;
  uint32_t pass_left_;     // PASS: body bytes still to forward (UINT32_MAX = LEN not known yet)
  uint8_t pass_len_[2];
  size_t pass_len_n_;
  uint8_t cap_[kCaptureMax];
  size_t cap_n_;
  size_t cap_total_;       // 0 until LEN known
  uint32_t last_byte_us_;
  bool gyro_cal_request_;
  uint8_t bus_buf_[256];   // bytes for the bus, written in runs (fewer TX enable toggles)
  size_t bus_n_;
};

}  // namespace kbridge

#endif  // KITORU_BRIDGE_CORE_H_
