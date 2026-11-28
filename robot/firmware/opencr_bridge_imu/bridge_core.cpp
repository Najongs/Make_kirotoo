// See bridge_core.h. No Arduino headers here: the same file is unit tested on a PC.

#include "bridge_core.h"

#include <string.h>

namespace kbridge {

namespace {
const uint8_t kHeader[4] = {0xFF, 0xFF, 0xFD, 0x00};
const size_t kPktId = 4, kPktLenL = 5, kPktLenH = 6, kPktInst = 7, kPktParam = 8;
}  // namespace

// CRC-16 (polynomial 0x8005, init 0, not reflected), same values as the SDK's table.
uint16_t crc16(uint16_t crc, const uint8_t* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; ++b) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x8005) : (uint16_t)(crc << 1);
  }
  return crc;
}

size_t add_stuffing(const uint8_t* p, size_t size, uint8_t* out, size_t cap) {
  if (size < 10 || size > cap) return 0;
  memcpy(out, p, kPktInst);
  size_t o = kPktInst;
  for (size_t i = kPktInst; i < size - 2; ++i) {
    if (o + 4 > cap) return 0;
    out[o++] = p[i];
    if (p[i] == 0xFD && p[i - 1] == 0xFF && p[i - 2] == 0xFF) out[o++] = 0xFD;
  }
  out[o++] = p[size - 2];
  out[o++] = p[size - 1];
  size_t len = o - kPktInst;
  out[kPktLenL] = (uint8_t)(len & 0xFF);
  out[kPktLenH] = (uint8_t)(len >> 8);
  return o;
}

size_t remove_stuffing(const uint8_t* p, size_t size, uint8_t* out, size_t cap) {
  if (size < 10 || size > cap) return 0;
  memcpy(out, p, kPktInst);
  size_t o = kPktInst;
  for (size_t i = kPktInst; i < size - 2; ++i) {
    if (p[i] == 0xFD && i + 1 < size - 2 && p[i + 1] == 0xFD && p[i - 1] == 0xFF && p[i - 2] == 0xFF) continue;
    out[o++] = p[i];
  }
  out[o++] = p[size - 2];
  out[o++] = p[size - 1];
  size_t len = o - kPktInst;
  out[kPktLenL] = (uint8_t)(len & 0xFF);
  out[kPktLenH] = (uint8_t)(len >> 8);
  return o;
}

// Madgwick (updateIMU) drives its estimate of the accelerometer direction
//   v = (2(q1q3 - q0q2), 2(q0q1 + q2q3), 1 - 2(q1^2 + q2^2))
// towards the measured acceleration, which points UP at rest. Gravity points down: g = -v.
void gravity_from_quat(const float q[4], float g[3]) {
  g[0] = -2.0f * (q[1] * q[3] - q[0] * q[2]);
  g[1] = -2.0f * (q[0] * q[1] + q[2] * q[3]);
  g[2] = -(1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]));
}

Bridge::Bridge(uint8_t virtual_id)
    : virtual_id_(virtual_id), to_bus_(0), to_host_(0), ctx_(0), mode_(HUNT), held_n_(0),
      pass_left_(0), pass_len_n_(0), cap_n_(0), cap_total_(0), last_byte_us_(0), gyro_cal_request_(false),
      bus_n_(0) {
  memset(table_, 0, sizeof(table_));
  put_u16(ADDR_MODEL_NUMBER, kModelNumber);
  table_[ADDR_FIRMWARE_VERSION] = kFirmwareVersion;
  table_[ADDR_ID] = virtual_id;
  table_[ADDR_TABLE_VERSION] = kTableVersion;
  float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 4; ++i) put_f32(ADDR_QUAT_W + 4 * i, q[i]);
  float g[3];
  gravity_from_quat(q, g);
  for (int i = 0; i < 3; ++i) put_f32(ADDR_GRAVITY_X + 4 * i, g[i]);
}

void Bridge::set_outputs(WriteFn to_bus, WriteFn to_host, void* ctx) {
  to_bus_ = to_bus;
  to_host_ = to_host;
  ctx_ = ctx;
}

// ---- table helpers -------------------------------------------------------------
void Bridge::put_u16(uint16_t a, uint16_t v) {
  table_[a] = (uint8_t)v;
  table_[a + 1] = (uint8_t)(v >> 8);
}
void Bridge::put_u32(uint16_t a, uint32_t v) {
  for (int i = 0; i < 4; ++i) table_[a + i] = (uint8_t)(v >> (8 * i));
}
uint32_t Bridge::get_u32(uint16_t a) const {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= (uint32_t)table_[a + i] << (8 * i);
  return v;
}
void Bridge::put_f32(uint16_t a, float v) {
  uint32_t bits;
  memcpy(&bits, &v, 4);  // IEEE 754 single; both Cortex-M7 and x86 are little endian
  put_u32(a, bits);
}

void Bridge::set_imu_config(uint16_t rate_hz, uint16_t gyro_fs_dps, uint8_t acc_fs_g) {
  put_u16(ADDR_IMU_RATE_HZ, rate_hz);
  put_u16(ADDR_GYRO_FS_DPS, gyro_fs_dps);
  table_[ADDR_ACC_FS_G] = acc_fs_g;
}

void Bridge::set_imu(const ImuSample& s) {
  put_u32(ADDR_TIMESTAMP_US, s.timestamp_us);
  put_u32(ADDR_SAMPLE_COUNT, s.sample_count);
  table_[ADDR_STATUS] = s.status;
  float g[3];
  gravity_from_quat(s.quat, g);
  for (int i = 0; i < 4; ++i) put_f32(ADDR_QUAT_W + 4 * i, s.quat[i]);
  for (int i = 0; i < 3; ++i) {
    put_f32(ADDR_GRAVITY_X + 4 * i, g[i]);
    put_f32(ADDR_GYRO_X + 4 * i, s.gyro_rad_s[i]);
    put_f32(ADDR_ACC_X + 4 * i, s.acc_m_s2[i]);
  }
}

void Bridge::note_loop_us(uint32_t us) {
  if (us > get_u32(ADDR_LOOP_MAX_US)) put_u32(ADDR_LOOP_MAX_US, us);
}
void Bridge::note_imu_update_us(uint32_t us) {
  if (us > get_u32(ADDR_IMU_UPDATE_MAX_US)) put_u32(ADDR_IMU_UPDATE_MAX_US, us);
}
bool Bridge::take_gyro_calibrate_request() {
  bool r = gyro_cal_request_;
  gyro_cal_request_ = false;
  return r;
}
void Bridge::set_gyro_calibrating(bool running) { table_[ADDR_GYRO_CALIBRATE] = running ? 1 : 0; }

// ---- output --------------------------------------------------------------------
void Bridge::out_bus(const uint8_t* d, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (bus_n_ == sizeof(bus_buf_)) flush_bus();
    bus_buf_[bus_n_++] = d[i];
  }
}

void Bridge::out_host(const uint8_t* d, size_t n) {
  if (to_host_) to_host_(ctx_, d, n);
}

void Bridge::flush_bus() {
  if (bus_n_ && to_bus_) to_bus_(ctx_, bus_buf_, bus_n_);
  bus_n_ = 0;
}

// ---- host stream ------------------------------------------------------------------
void Bridge::release_held() {
  if (held_n_) {
    out_bus(held_, held_n_);
    held_n_ = 0;
  }
}

void Bridge::hunt_byte(uint8_t b, uint32_t now_us) {
  held_[held_n_++] = b;
  // Keep the held bytes a prefix of FF FF FD 00; drop (forward) leading bytes until they are.
  for (;;) {
    bool ok = true;
    for (size_t i = 0; i < held_n_ && i < 4; ++i) {
      if (held_[i] != kHeader[i]) {
        ok = false;
        break;
      }
    }
    if (ok) break;
    out_bus(held_, 1);
    memmove(held_, held_ + 1, --held_n_);
  }
  if (held_n_ < 5) return;
  uint8_t id = held_[kPktId];
  if (id == virtual_id_) {
    memcpy(cap_, held_, 5);
    cap_n_ = 5;
    cap_total_ = 0;
    held_n_ = 0;
    mode_ = CAPTURE;
  } else {
    release_held();  // servo ID or broadcast: no more holding
    pass_len_n_ = 0;
    pass_left_ = 0;
    mode_ = PASS;
  }
  last_byte_us_ = now_us;
}

void Bridge::feed_host(const uint8_t* data, size_t size, uint32_t now_us) {
  poll(now_us);
  for (size_t i = 0; i < size; ++i) {
    uint8_t b = data[i];
    switch (mode_) {
      case HUNT:
        hunt_byte(b, now_us);
        break;
      case PASS:
        out_bus(&b, 1);
        if (pass_len_n_ < 2) {
          pass_len_[pass_len_n_++] = b;
          if (pass_len_n_ == 2) {
            pass_left_ = (uint32_t)pass_len_[0] | ((uint32_t)pass_len_[1] << 8);
            if (pass_left_ == 0) mode_ = HUNT;
          }
        } else if (--pass_left_ == 0) {
          mode_ = HUNT;
        }
        break;
      case CAPTURE:
        cap_[cap_n_++] = b;
        if (cap_n_ == 7) {
          size_t len = (size_t)cap_[kPktLenL] | ((size_t)cap_[kPktLenH] << 8);
          if (len < 3 || len + 7 > kCaptureMax) {
            // Not something we can answer: let it through like any other packet.
            out_bus(cap_, cap_n_);
            pass_len_[0] = cap_[kPktLenL];
            pass_len_[1] = cap_[kPktLenH];
            pass_len_n_ = 2;
            pass_left_ = (uint32_t)len;
            cap_n_ = 0;
            mode_ = len ? PASS : HUNT;
            break;
          }
          cap_total_ = len + 7;
        }
        if (cap_total_ && cap_n_ == cap_total_) {
          handle_capture();
          cap_n_ = 0;
          cap_total_ = 0;
          mode_ = HUNT;
        }
        break;
    }
  }
  last_byte_us_ = now_us;
  flush_bus();
}

void Bridge::poll(uint32_t now_us) {
  uint32_t idle = now_us - last_byte_us_;
  if (mode_ == HUNT && held_n_ && idle >= kHoldTimeoutUs) {
    release_held();
    put_u32(ADDR_HOLD_TIMEOUTS, get_u32(ADDR_HOLD_TIMEOUTS) + 1);
  } else if (mode_ == PASS && idle >= kPacketTimeoutUs) {
    mode_ = HUNT;  // the rest of that packet never came; resync on the next header
  } else if (mode_ == CAPTURE && idle >= kPacketTimeoutUs) {
    cap_n_ = 0;  // half a packet for the virtual ID: nobody on the bus wants it
    cap_total_ = 0;
    mode_ = HUNT;
  }
  flush_bus();
}

// ---- virtual device -----------------------------------------------------------------
void Bridge::reply(uint8_t error, const uint8_t* params, size_t n) {
  uint8_t raw[kTableSize + 16];
  uint8_t out[sizeof(raw) * 2];
  size_t len = n + 4;  // instruction + error + params + CRC
  memcpy(raw, kHeader, 4);
  raw[kPktId] = virtual_id_;
  raw[kPktLenL] = (uint8_t)(len & 0xFF);
  raw[kPktLenH] = (uint8_t)(len >> 8);
  raw[kPktInst] = kInstStatus;
  raw[kPktParam] = error;
  if (n) memcpy(raw + kPktParam + 1, params, n);
  size_t total = kPktParam + 1 + n + 2;
  raw[total - 2] = raw[total - 1] = 0;
  size_t m = add_stuffing(raw, total, out, sizeof(out));
  if (!m) return;
  uint16_t crc = crc16(0, out, m - 2);
  out[m - 2] = (uint8_t)(crc & 0xFF);
  out[m - 1] = (uint8_t)(crc >> 8);
  out_host(out, m);
  put_u32(ADDR_VIRTUAL_PACKETS, get_u32(ADDR_VIRTUAL_PACKETS) + 1);
}

void Bridge::handle_capture() {
  uint16_t crc = crc16(0, cap_, cap_total_ - 2);
  if ((uint8_t)(crc & 0xFF) != cap_[cap_total_ - 2] || (uint8_t)(crc >> 8) != cap_[cap_total_ - 1]) {
    reply(kErrCrc, 0, 0);
    return;
  }
  uint8_t p[kCaptureMax];
  size_t n = remove_stuffing(cap_, cap_total_, p, sizeof(p));
  if (!n) return;
  size_t nparam = n - kPktParam - 2;
  const uint8_t* param = p + kPktParam;
  switch (p[kPktInst]) {
    case kInstPing: {
      uint8_t r[3] = {table_[ADDR_MODEL_NUMBER], table_[ADDR_MODEL_NUMBER + 1], table_[ADDR_FIRMWARE_VERSION]};
      reply(0, r, 3);
      break;
    }
    case kInstRead: {
      if (nparam != 4) {
        reply(kErrDataLength, 0, 0);
        break;
      }
      uint32_t addr = param[0] | (param[1] << 8), len = param[2] | (param[3] << 8);
      if (len == 0 || addr + len > kTableSize) {
        reply(kErrDataRange, 0, 0);
        break;
      }
      reply(0, table_ + addr, len);
      break;
    }
    case kInstWrite: {
      if (nparam < 3) {
        reply(kErrDataLength, 0, 0);
        break;
      }
      uint32_t addr = param[0] | (param[1] << 8), len = nparam - 2;
      const uint8_t* d = param + 2;
      bool ok = true;
      for (uint32_t i = 0; i < len; ++i) {
        uint32_t a = addr + i;
        if (a != ADDR_RESET_DIAG && a != ADDR_GYRO_CALIBRATE) ok = false;
      }
      if (!ok) {
        reply(kErrAccess, 0, 0);
        break;
      }
      for (uint32_t i = 0; i < len; ++i) {
        uint32_t a = addr + i;
        if (a == ADDR_RESET_DIAG && d[i] == 1) memset(table_ + ADDR_LOOP_MAX_US, 0, ADDR_RESET_DIAG - ADDR_LOOP_MAX_US);
        if (a == ADDR_GYRO_CALIBRATE && d[i] == 1) {
          gyro_cal_request_ = true;
          table_[ADDR_GYRO_CALIBRATE] = 1;
        }
      }
      reply(0, 0, 0);
      break;
    }
    default:
      reply(kErrInstruction, 0, 0);
      break;
  }
}

}  // namespace kbridge

// ---- extern "C" API for PC tests (ctypes) ----------------------------------------------
#ifdef KBRIDGE_C_API
namespace {
struct Handle {
  kbridge::Bridge bridge;
  uint8_t bus[4096];
  size_t bus_n;
  uint8_t host[4096];
  size_t host_n;
  explicit Handle(uint8_t id) : bridge(id), bus_n(0), host_n(0) {}
};
void to_bus(void* ctx, const uint8_t* d, size_t n) {
  Handle* h = static_cast<Handle*>(ctx);
  for (size_t i = 0; i < n && h->bus_n < sizeof(h->bus); ++i) h->bus[h->bus_n++] = d[i];
}
void to_host(void* ctx, const uint8_t* d, size_t n) {
  Handle* h = static_cast<Handle*>(ctx);
  for (size_t i = 0; i < n && h->host_n < sizeof(h->host); ++i) h->host[h->host_n++] = d[i];
}
size_t take(uint8_t* src, size_t* n, uint8_t* out, size_t cap) {
  size_t m = *n < cap ? *n : cap;
  memcpy(out, src, m);
  memmove(src, src + m, *n - m);
  *n -= m;
  return m;
}
}  // namespace

extern "C" {
void* kb_create(uint8_t virtual_id) {
  Handle* h = new Handle(virtual_id);
  h->bridge.set_outputs(to_bus, to_host, h);
  h->bridge.set_imu_config(200, 2000, 2);
  return h;
}
void kb_destroy(void* h) { delete static_cast<Handle*>(h); }
void kb_feed_host(void* h, const uint8_t* d, size_t n, uint32_t now_us) {
  static_cast<Handle*>(h)->bridge.feed_host(d, n, now_us);
}
void kb_poll(void* h, uint32_t now_us) { static_cast<Handle*>(h)->bridge.poll(now_us); }
size_t kb_take_bus(void* h, uint8_t* out, size_t cap) {
  Handle* p = static_cast<Handle*>(h);
  return take(p->bus, &p->bus_n, out, cap);
}
size_t kb_take_host(void* h, uint8_t* out, size_t cap) {
  Handle* p = static_cast<Handle*>(h);
  return take(p->host, &p->host_n, out, cap);
}
void kb_set_imu(void* h, uint32_t ts, uint32_t count, uint8_t status, const float* quat, const float* gyro,
                const float* acc) {
  kbridge::ImuSample s;
  s.timestamp_us = ts;
  s.sample_count = count;
  s.status = status;
  memcpy(s.quat, quat, sizeof(s.quat));
  memcpy(s.gyro_rad_s, gyro, sizeof(s.gyro_rad_s));
  memcpy(s.acc_m_s2, acc, sizeof(s.acc_m_s2));
  static_cast<Handle*>(h)->bridge.set_imu(s);
}
int kb_take_gyro_calibrate_request(void* h) { return static_cast<Handle*>(h)->bridge.take_gyro_calibrate_request(); }
void kb_note_loop_us(void* h, uint32_t us) { static_cast<Handle*>(h)->bridge.note_loop_us(us); }
uint16_t kb_crc16(const uint8_t* d, size_t n) { return kbridge::crc16(0, d, n); }
size_t kb_add_stuffing(const uint8_t* p, size_t n, uint8_t* out, size_t cap) {
  return kbridge::add_stuffing(p, n, out, cap);
}
void kb_gravity_from_quat(const float* q, float* g) { kbridge::gravity_from_quat(q, g); }
}
#endif
