// Kitoru OpenCR 1.0 firmware: USB <-> Dynamixel bridge + on-board IMU as a virtual Dynamixel ID.
//
// - Every Protocol 2.0 packet from USB goes to the TTL bus and every answer comes back,
//   as in the ROBOTIS example "10. Etc/usb_to_dxl" (same Serial3, same byte-by-byte
//   flush, same "follow the USB baud rate" rule).
// - Packets for VIRTUAL_ID (200) never reach the bus. The OpenCR answers PING, READ and a
//   small WRITE itself from a control table (bridge_core.h) holding the ICM-20648 data.
// - SYNC_READ / BULK_READ that list VIRTUAL_ID are NOT answered: read the IMU with its own
//   READ (see robot/README.md 3절).
// - IMU.begin() blocks for up to ~5 s at power-up (gyro bias, keep the robot still);
//   the bridge does not answer during that time.
//
// Build: Arduino IDE or arduino-cli with the OpenCR board package (FQBN OpenCR:OpenCR:OpenCR,
// checked with package 1.5.3). Upload steps: robot/README.md 3절.

#include <DynamixelSDK.h>  // kept from usb_to_dxl (board drivers, drv_dxl_tx_enable)
#include <IMU.h>           // ROBOTIS cIMU (ICM-20648 on OpenCR 1.0, Madgwick filter)

#include "bridge_core.h"

#define CMD_PORT Serial   // USB
#define DXL_PORT Serial3  // Dynamixel TTL / RS-485
#define DXL_BAUD 1000000

#define VIRTUAL_ID 200          // must not be a servo ID on the bus
#define IMU_RATE_HZ 200         // cIMU::begin() default; Madgwick runs at this rate
#define IMU_MAX_DEFER_US 2000   // IMU update waits for a quiet bus, but never longer than this
#define DXL_RX_BUFFER 1024

#define DXL_POWER_ENABLE() digitalWrite(BDPIN_DXL_PWR_EN, HIGH)

// ICM-20648 setup in the ROBOTIS driver (ICM20648.cpp): gyro +-2000 deg/s, accel +-2 g.
// Note: IMU.cpp uses aRes = 8 g / 32768 for imu.ax..az, which does not match +-2 g; the
// Madgwick filter normalises the accel so the attitude is fine, but we scale accData here.
static const float GYRO_RAD_S_PER_LSB = (2000.0f / 32768.0f) * 0.01745329252f;
static const float ACC_M_S2_PER_LSB = 9.80665f / 16384.0f;  // [확인 필요] +-2 g -> 16384 LSB/g

cIMU imu;
kbridge::Bridge bridge(VIRTUAL_ID);

static uint8_t usb_rx[256];
static uint8_t dxl_rx[DXL_RX_BUFFER];
static uint32_t sample_count = 0;
static uint32_t last_imu_us = 0;

static void write_bus(void*, const uint8_t* data, size_t size) {
  drv_dxl_tx_enable(TRUE);
  for (size_t i = 0; i < size; i++) {
    DXL_PORT.write(data[i]);
    DXL_PORT.flush();
  }
  drv_dxl_tx_enable(FALSE);
}

static void write_host(void*, const uint8_t* data, size_t size) {
  CMD_PORT.write((uint8_t*)data, size);
}

static void update_imu() {
  uint32_t t0 = micros();
  uint16_t ran = imu.update();  // computes only when 1/IMU_RATE_HZ has passed (returns 0 otherwise)
  uint32_t t1 = micros();
  if (!ran) return;
  last_imu_us = t1;
  bridge.note_imu_update_us(t1 - t0);

  kbridge::ImuSample s;
  s.timestamp_us = t0;
  s.sample_count = ++sample_count;
  s.status = (imu.bConnected ? kbridge::kStatusImuFound : 0) |
             (imu.SEN.gyro_cali_get_done() ? kbridge::kStatusGyroCalibrated : 0);
  for (int i = 0; i < 4; i++) s.quat[i] = imu.quat[i];
  for (int i = 0; i < 3; i++) {
    s.gyro_rad_s[i] = imu.gyroData[i] * GYRO_RAD_S_PER_LSB;
    s.acc_m_s2[i] = imu.accData[i] * ACC_M_S2_PER_LSB;
  }
  bridge.set_imu(s);
}

void setup() {
  CMD_PORT.begin(115200);
  DXL_PORT.begin(DXL_BAUD);

  pinMode(BDPIN_DXL_PWR_EN, OUTPUT);
  drv_dxl_tx_enable(FALSE);
  DXL_POWER_ENABLE();

  bridge.set_outputs(write_bus, write_host, 0);
  bridge.set_imu_config(IMU_RATE_HZ, 2000, 2);

  imu.begin(IMU_RATE_HZ);  // blocks: SPI setup + gyro bias (512 samples, up to 5 s)
  update_imu();
}

void loop() {
  uint32_t t0 = micros();

  //-- USB -> bridge core -> DXL (or a virtual-ID answer back to USB)
  int length = CMD_PORT.available();
  if (length > 0) {
    if (length > (int)sizeof(usb_rx)) length = sizeof(usb_rx);
    for (int i = 0; i < length; i++) usb_rx[i] = CMD_PORT.read();
    bridge.feed_host(usb_rx, length, micros());
  } else {
    bridge.poll(micros());
  }

  //-- DXL -> USB, unchanged from usb_to_dxl
  length = DXL_PORT.available();
  if (length > 0) {
    if (length > DXL_RX_BUFFER) length = DXL_RX_BUFFER;
    for (int i = 0; i < length; i++) dxl_rx[i] = DXL_PORT.read();
    CMD_PORT.write(dxl_rx, length);
  }

  if (CMD_PORT.getBaudRate() != DXL_PORT.getBaudRate()) {
    DXL_PORT.begin(CMD_PORT.getBaudRate());
  }

  //-- IMU: run when no byte is waiting on either side, so servo answers are not delayed;
  //   if the bus never goes quiet, run anyway after IMU_MAX_DEFER_US.
  uint32_t now = micros();
  bool due = (now - last_imu_us) >= (1000000UL / IMU_RATE_HZ);
  bool quiet = CMD_PORT.available() == 0 && DXL_PORT.available() == 0;
  bool overdue = (now - last_imu_us) >= (1000000UL / IMU_RATE_HZ) + IMU_MAX_DEFER_US;
  if (due && (quiet || overdue)) update_imu();

  if (bridge.take_gyro_calibrate_request()) imu.SEN.gyro_cali_start();
  bridge.set_gyro_calibrating(!imu.SEN.gyro_cali_get_done());

  bridge.note_loop_us(micros() - t0);
}
