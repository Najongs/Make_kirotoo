# 키토루 서보 제어 기초 코드 (`robot/`)

PC 나 Jetson Nano 에서 OpenCR(USB) 을 거쳐 Dynamixel X430 서보(XL430 · XC430 · XM430)를 움직이는 코드입니다.
소유자가 PC 에서 확인한 `XL430_robot_base.py`(지금 `legacy/` 에 그대로 있음)를 나눠서, 관절 15개(나중에 17개)를 같은 코드로 다룰 수 있게 했습니다.

- **이 코드는 걷기 제어기가 아닙니다.** 관절 각을 읽고, 목표 각을 안전하게 보내는 아래층입니다. 걷기는 나중에 정책(강화학습)이 이 위에서 돕니다(8절).
- **이 머신(DGX_1)에는 로봇이 없어서, 실제 서보로는 한 번도 안 돌렸습니다.** 시험은 가짜 서보(10절)로만 했습니다.

## 1. 구조

| 파일 | 하는 일 |
| --- | --- |
| `kitoru_hw/control_table.py` | X430 컨트롤 테이블 주소 · 단위 · 모델 번호(XL430-W250 1060, XC430-W240 1080, XM430-W350 1020) |
| `kitoru_hw/dxl_bus.py` | 직렬 포트 하나. 한 번에 여러 서보 읽기(Sync Read) · 쓰기(Sync Write), 통신 오류 · 연결 끊김 처리, 브로드캐스트 토크 끄기, ID 스캔 |
| `kitoru_hw/joints.py` | 관절 표(YAML) 읽기, 각도 ↔ tick 변환, 영점 오프셋을 YAML 에 다시 쓰기 |
| `kitoru_hw/safety.py` | 한계로 자르기, 짝 관절 제한, 넘어짐 감지 → PWM 낮추기, 저전압 정지, 온도 감시 |
| `kitoru_hw/robot.py` | 관절 이름 · 라디안으로 쓰는 로봇 전체. 토크 켜기 전 점검(원본과 같음), 비상정지 |
| `kitoru_hw/policy_io.py` | 정책 연결 자리: 관측(관절 각 · 속도 · IMU) 읽기, 목표 각 쓰기, 서보 읽기 · IMU 읽기 시간 기록 |
| `kitoru_hw/opencr_imu.py` | OpenCR 보드 IMU 를 가상 ID 200 으로 읽어 몸통 기준 중력 방향 · 각속도로 바꿈(3.1절) |
| `firmware/opencr_bridge_imu/` | OpenCR 펌웨어: 서보 패킷은 그대로 넘기고, ID 200 만 IMU 값으로 직접 답함(3.1절) |
| `kitoru_hw/timing.py` | 제어 주기 계산식 |
| `config/joints_v20.yaml` | v20 관절 15개: 이름 · ID · 모델 · 단자 · 방향 · 영점 · 한계 |
| `config/bench_legacy.yaml` | 원본 시험대와 똑같은 설정(XL430 ID 6 · 7 · 12) |
| `tools/read_demo.py` | 읽기만 / `--demo`(약 5° 움직였다 돌아옴) / `--rate`(읽기 속도 재기). 원본 기능 그대로 |
| `tools/scan_ids.py` | 브로드캐스트 ping 으로 버스의 서보를 모두 찾고 YAML 과 대조 |
| `tools/record_zero.py` | 영점 자세에서 현재 값을 읽어 `zero_offset` 을 YAML 에 저장 |
| `tools/range_check.py` | 한 관절씩 느린 속도 · 낮은 PWM 으로 범위 확인 |
| `tools/bus_timing.py` | 통신 속도별 제어 주기 표 출력(하드웨어 필요 없음) |
| `tools/imu_check.py` | IMU 펌웨어 확인: PING 200, 몸통 · 칩 축 값 실시간 출력, 서보 읽기 · IMU 읽기 시간(토크는 안 건드림) |
| `legacy/XL430_robot_base.py` | 원본(내용 그대로) |
| `tests/` | 가짜 서보로 도는 pytest |

## 2. 설치와 실행

필요한 것은 `dynamixel-sdk` 와 `pyyaml` 뿐입니다(`requirements.txt`). Python 3.8 이상에서 돌게 썼습니다(문법만 확인, 실제로는 3.12 에서 시험).

**PC (Windows · Linux)**

```bash
cd robot
python -m pip install -r requirements.txt
python tools/read_demo.py --port COM3                    # 원본과 같은 시험대(ID 6, 7, 12), 읽기만
python tools/read_demo.py --port COM3 --demo             # 원본 --demo 와 같음
python tools/scan_ids.py --port /dev/ttyACM0             # Linux
```

**Jetson Nano 2GB**

- JetPack 4.6 은 Ubuntu 18.04 이고 기본 Python 이 3.6 이라 이 코드가 안 돕니다. 둘 중 하나로 3.8 이상을 씁니다.
  - `uv python install 3.12` 뒤에 `uv venv --python 3.12` [확인 필요: Jetson(aarch64, glibc 2.27)에서 uv 의 Python 이 도는지]
  - 또는 `sudo apt install python3.8 python3.8-venv` 뒤에 `python3.8 -m venv .venv`
- 그다음은 PC 와 같습니다: `pip install -r requirements.txt`, `python tools/scan_ids.py --port /dev/kitoru_opencr`.
- 포트 이름과 권한은 4절에서 정합니다.

모든 도구는 `--config`(기본 `config/joints_v20.yaml`, `read_demo.py` 만 기본이 `config/bench_legacy.yaml`), `--port`, `--baud`, `--joints R_knee,L_knee` 를 받습니다.

## 3. OpenCR 을 USB-다이나믹셀 다리로 쓰는 조건

원본이 이미 이 방식으로 PC 에서 돌았습니다. 지금 OpenCR 에 올라간 펌웨어를 바꾸지 않으면 Jetson 에 꽂아도 똑같이 돕니다.

- **원본이 도는 방식**
  - PC 는 OpenCR 을 USB 가상 직렬 포트(Windows `COM3`, Linux `/dev/ttyACM0`)로 봅니다.
  - PC 의 dynamixel-sdk 가 Protocol 2.0 패킷을 그 포트로 보냅니다.
  - OpenCR 의 펌웨어가 그 바이트를 Dynamixel 단자(TTL · RS-485)로 그대로 넘기고, 답을 다시 USB 로 돌려줍니다. OpenCR 자신은 서보를 제어하지 않습니다.
- **펌웨어:** OpenCR Arduino 예제의 USB-다이나믹셀 다리 스케치(`usb_to_dxl`)로 봅니다 [확인 필요: 지금 보드에 올라간 것이 그 예제인지, 아니면 TurtleBot3 펌웨어인지]. TurtleBot3 펌웨어는 자기 프로토콜로 바퀴를 직접 제어해서 이 코드와 안 맞을 수 있습니다 [확인 필요].
- **통신 속도:** PC 에서 포트를 1 Mbps 로 열면 다리 스케치가 Dynamixel 쪽도 1 Mbps 로 맞추는 것으로 봅니다 [확인 필요: 다리 스케치가 2 · 4 Mbps 도 넘기는지]. 서보의 Baud Rate(8) 와 YAML `baud` 가 같아야 합니다.
- **서보 전원은 USB 에서 안 나옵니다.** OpenCR 입력(5–24 V 배터리 · SMPS)에 12 V 를 넣고 보드 전원 스위치를 켜야 합니다.
  - USB 5 V 로는 XL430 최저 전압(6 V 쪽) 아래라 서보가 안 돕니다.
  - 공식 문서에는 「서보 전원이 어디서 오는지」 문장이 없습니다 [확인 필요].
  - OpenCR 입력 퓨즈는 10 A(LittleFuse 0453010, 공식 문서)입니다. 넘어질 때 순간 14.5 A(시뮬, `HIPROLL_STUDY`)가 이 퓨즈에 걸리는지는 [확인 필요].
- **TTL 단자 셋:** 공식 문서에 Dynamixel 포트는 `Serial3` 하나로 나옵니다. 그래서 TTL1(오른다리) · TTL2(왼다리) · TTL3(팔 + 머리)는 한 버스에 나란히 붙은 것으로 보고, YAML 의 `connector` 는 배선 이름으로만 씁니다 [확인 필요: 「나란히 붙는다」는 문장은 문서에 없음]. 그래서 **ID 15개는 모두 달라야 합니다.**
- **IMU:** OpenCR 보드의 ICM-20648 을 쓰려면 `usb_to_dxl` 대신 `firmware/opencr_bridge_imu` 를 올립니다(3.1절, 소유자 결정 2026-11-28 「A. OpenCR 펌웨어」). 안 되면 Jetson I2C 에 따로 붙인 IMU 가 대비책(B)입니다.

### 3.1 IMU 펌웨어 `firmware/opencr_bridge_imu`

**하는 일:** OpenCR 이 지금처럼 서보 패킷을 그대로 넘기면서, **ID 200 앞으로 온 패킷만 버스에 안 보내고 직접 답합니다.** 예: Jetson 이 `READ ID 200, 32 번지, 64 바이트` 를 보내면 OpenCR 이 USB 로 IMU 한 표본(시각 · 쿼터니언 · 중력 방향 · 각속도 · 가속도)을 돌려줍니다. 서보 버스에는 아무것도 안 나갑니다.

- **무엇이 아닌가:** OpenCR 이 서보를 제어하는 것이 아닙니다. 서보 패킷은 앞 5 바이트(`FF FF FD 00 ID`)만 보고 바로 넘깁니다. 브로드캐스트(ID 254, 비상정지 토크 끄기)는 붙잡지 않습니다. 앞머리가 반쯤 온 채 300 µs 멈추면 잡아 둔 바이트도 그대로 넘깁니다.
- **읽는 법은 따로 READ 입니다.** 서보 Sync Read 목록에 200 을 넣지 마세요. 펌웨어가 Sync Read · Bulk Read · 브로드캐스트 PING 에는 답하지 않습니다.
  - 이유: Sync Read 는 모든 ID 가 같은 번지 · 길이를 읽고(서보 126 번지 10 바이트 창에 IMU 가 안 들어감), SDK 가 목록 순서대로만 답을 받으며, 서보가 앞 ID 의 답을 기다리는지 모릅니다 [확인 필요].
  - 그래서 `scan_ids.py` 에도 ID 200 은 안 나옵니다.
  - 값: 한 주기에 USB 왕복 한 번이 더 듭니다(약 1–2 ms [추정], 50 Hz 한 주기 20 ms 안).
- **IMU 갱신:** ROBOTIS `IMU` 라이브러리(`cIMU`, ICM-20648 + Madgwick 필터, 200 Hz)를 그대로 씁니다. 양쪽 포트에 기다리는 바이트가 없을 때만 갱신하고, 2 ms 넘게 밀리면 그냥 갱신합니다. 갱신 한 번이 막는 시간은 `imu_update_max_us` 로 보입니다.
- **전원을 켠 뒤 약 3 초 가만히 둡니다.** `IMU.begin()` 이 자이로 영점(512 표본, 최대 5 초)을 재는 동안 다리도 답하지 않습니다. USB 포트를 열 때 보드가 다시 시작하는지는 [확인 필요]입니다(그렇다면 프로그램을 띄울 때마다 3 초).
- **라이브러리 값의 성질:** 자이로는 영점을 빼고 3 표본 이동 평균을 낸 뒤 ±3 LSB(±0.18 °/s)를 0 으로 자른 값입니다. 시뮬과 다른 점으로 남습니다. 지자기는 없어서 쿼터니언의 yaw 는 흐릅니다(정책은 중력 방향 · 각속도만 씀).

**가상 ID 200 의 control table** (리틀 엔디언, `bridge_core.h`)

| 번지 | 크기 | 이름 | 읽기/쓰기 | 뜻 · 단위 |
| --- | --- | --- | --- | --- |
| 0 | 2 | MODEL_NUMBER | R | `0x4B49`(19273). ROBOTIS 모델 번호가 아님 |
| 6 | 1 | FIRMWARE_VERSION | R | 1 |
| 7 | 1 | ID | R | 200(펌웨어 `VIRTUAL_ID`, 컴파일 때 정함) |
| 8 | 1 | TABLE_VERSION | R | 1 |
| 10 | 2 | IMU_RATE_HZ | R | 200 |
| 12 | 2 | GYRO_FS_DPS | R | 2000(칩 설정 ±2000 °/s) |
| 14 | 1 | ACC_FS_G | R | 2(칩 설정 ±2 g) |
| **32** | 4 | TIMESTAMP_US | R | 표본을 잰 `micros()`. 약 71.6 분마다 0 으로 돌아감 |
| 36 | 4 | SAMPLE_COUNT | R | 표본마다 +1. 두 번 같으면 새 값이 아님 |
| 40 | 1 | STATUS | R | bit0 IMU 찾음, bit1 자이로 영점 끝 |
| 44 | 16 | QUAT_W..Z | R | float32 ×4, Madgwick 쿼터니언(칩 축 → 세계, yaw 흐름) |
| 60 | 12 | GRAVITY_X..Z | R | float32 ×3, 칩 축 중력 방향 단위벡터. **아래를 가리킴**(칩 z 가 위면 (0, 0, −1)) |
| 72 | 12 | GYRO_X..Z | R | float32 ×3, 칩 축 각속도 rad/s |
| 84 | 12 | ACC_X..Z | R | float32 ×3, 칩 축 가속도 m/s². 가만히 있으면 **위쪽**으로 +9.8 |
| 100 | 4 | LOOP_MAX_US | R | `loop()` 한 바퀴 최장 시간(중계가 멈춘 최장 시간) |
| 104 | 4 | IMU_UPDATE_MAX_US | R | IMU 갱신 한 번 최장 시간 |
| 108 | 4 | VIRTUAL_PACKETS | R | ID 200 이 답한 패킷 수 |
| 112 | 4 | HOLD_TIMEOUTS | R | 잡아 둔 앞머리를 시간 초과로 넘긴 횟수 |
| 116 | 1 | RESET_DIAG | W | 1 을 쓰면 100–115 를 0 으로 |
| 117 | 1 | GYRO_CALIBRATE | RW | 1 을 쓰면 자이로 영점을 다시 잼(약 3 초 가만히), 재는 동안 1 |

- 한 표본은 **32 번지부터 64 바이트를 한 번 READ** 합니다(답 75 바이트 안팎).
- 답하는 명령: PING, READ, WRITE(116 · 117 만, 나머지는 Access Error 0x07). 표 밖 READ 는 Data Range Error 0x04, CRC 가 틀리면 CRC Error 0x03, 그 밖의 명령은 Instruction Error 0x02.
- 값은 float32 라 바이트 스터핑이 실제로 생깁니다(예: 1.984 rad/s 의 바이트가 `FF FF FD 3F`). 펌웨어 답도 규격대로 스터핑합니다.
- 가속도 눈금은 칩 설정 ±2 g → 16384 LSB/g 로 펌웨어가 계산합니다 [확인 필요: 실물에서 가만히 놓고 9.8 이 나오는지]. 라이브러리의 `imu.ax` 는 ±8 g 눈금으로 계산해서 4배 작게 나오므로 쓰지 않습니다.

**축(좌표)** — `config/joints_v20.yaml` 의 `imu:`

```
v_몸통 = r_body_board · r_board_chip · v_칩
r_body_board = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]   # 보드 x → 몸통 +y, 보드 y → 몸통 −z, 보드 z(부품 면) → 몸통 −x
r_board_chip = 단위행렬                                # [확인 필요] 아래 ②③ 에서 정함
```

- `r_body_board` 는 설계 형상(`code_v17/build17.py` `oc_box`, 보드 중심 몸통 좌표 약 (−0.070, 0, 0.280) m)에서 왔습니다. 몸통 축은 MuJoCo `torso` body 와 같은 축(x 앞, z 위)으로 봅니다(MJCF `torso` 에 회전 없음).
- 똑바로 서면 중력은 보드 +y 쪽이라 `gravity_b` ≈ (0, 0, −1) 이어야 합니다. 정책 `meas_gravity_b` 와 같은 뜻입니다.
- det 가 −1 인 행렬(거울)이나 직교가 아닌 행렬은 `OpenCRImu` 가 거부합니다.

**펌웨어 올리기** (OpenCR 보드 패키지 1.5.3 으로 컴파일 확인. 컴파일러가 x86 용이라 Jetson 이 아니라 PC 에서 올립니다)

- Arduino IDE: 환경설정 「추가 보드 관리자 URL」에 `https://raw.githubusercontent.com/ROBOTIS-GIT/OpenCR/master/arduino/opencr_release/package_opencr_index.json` → 보드 관리자에서 OpenCR 설치 → 보드 「OpenCR Board」, 포트 선택 → `firmware/opencr_bridge_imu/opencr_bridge_imu.ino` 를 열고 업로드.
- arduino-cli:

  ```bash
  arduino-cli core update-index --additional-urls https://raw.githubusercontent.com/ROBOTIS-GIT/OpenCR/master/arduino/opencr_release/package_opencr_index.json
  arduino-cli core install OpenCR:OpenCR --additional-urls https://raw.githubusercontent.com/ROBOTIS-GIT/OpenCR/master/arduino/opencr_release/package_opencr_index.json
  arduino-cli compile --fqbn OpenCR:OpenCR:OpenCR robot/firmware/opencr_bridge_imu
  arduino-cli upload  --fqbn OpenCR:OpenCR:OpenCR -p /dev/ttyACM0 robot/firmware/opencr_bridge_imu   # Windows 는 -p COM3
  ```

- 올리기가 안 되면 OpenCR 을 부트로더로 띄운 뒤 다시 합니다(PUSH SW2 를 누른 채 RESET) [확인 필요: ROBOTIS e-manual 「OpenCR Arduino IDE」 절]. Linux 는 e-manual 의 udev 규칙도 필요할 수 있습니다 [확인 필요].
- 원래대로 돌리려면 예제 `10. Etc/usb_to_dxl` 을 다시 올립니다.

**실물 첫 확인 순서** — 서보 토크는 끈 채로 합니다. `imu_check.py` 는 토크 레지스터를 건드리지 않고, `arm()` 을 부르지 않습니다. 비상정지(전원 스위치)는 손이 닿는 곳에 둡니다(6절).

0. 펌웨어를 올리고 전원을 켠 뒤 **3 초 가만히** 둡니다. 서보 전원 없이 USB 만으로 ①–③ 을 해도 됩니다(`--no-servos`).
1. **PING 200:** `python tools/imu_check.py --port /dev/kitoru_opencr --no-servos --seconds 0`. `ID 200: OpenCR IMU firmware 1` 이 나와야 합니다. 「Is robot/firmware/opencr_bridge_imu on the OpenCR?」가 나오면 펌웨어가 안 올라간 것입니다.
2. **중력 축 부호:** `--no-servos --seconds 60` 으로 띄우고 값을 봅니다.
   - 보드만 책상에 부품 면을 위로 놓으면 `g_chip` ≈ (0, 0, −1) 이어야 합니다(칩 z = 보드 z 인지).
   - 몸통에 단 채 똑바로 세우면 `g_body` ≈ (0, 0, −1).
   - 앞으로 숙이면 `g_body` x 가 + 로, `pitch` 가 + 로. 오른쪽으로 기울이면 `g_body` y 가 − 로, `roll` 이 + 로.
   - 다르면 `g_chip` 이 어느 축으로 움직였는지 보고 `r_board_chip` 을 고칩니다(열 = 칩 축을 보드 좌표로 쓴 것).
3. **각속도 축:** 똑바로 든 채 손으로 돌립니다. 위에서 볼 때 반시계(왼쪽으로 돌기)면 `w_body` z 가 +, 앞으로 숙이는 동안 y 가 +, 오른쪽으로 기우는 동안 x 가 + 이어야 합니다(오른손 법칙). 중력 축을 고친 행렬로 각속도도 맞아야 합니다. 안 맞으면 칩 축이 왼손계라는 뜻이니 멈추고 알려 주세요.
4. **서보 중계 지연 전 · 후:** 서보 전원을 켜고(토크는 끈 채) 같은 포트에서 잽니다.
   - 전: `usb_to_dxl` 을 올린 상태에서 `python tools/read_demo.py --config config/joints_v20.yaml --rate 500`.
   - 후: 이 펌웨어로 바꾸고 같은 명령, 그리고 `python tools/imu_check.py --timing 500 --seconds 0`.
   - 서보 읽기 ms 가 거의 같아야 하고, `loop_max_us` 가 수십 µs 대여야 합니다 [추정]. IMU 읽기 ms 가 한 주기 시간에 더해집니다.

**Jetson 쪽 코드**

```python
from kitoru_hw.opencr_imu import OpenCRImu
with Robot(config) as robot:                       # 포트는 서보와 같이 한 번만 엽니다
    imu = OpenCRImu.from_config(robot.buses["opencr"], config.imu).open()   # PING 200 + 상태 확인
    io = PolicyIO(robot, imu=imu)
    obs = io.observe()     # obs["imu"]["gravity_b"], obs["imu"]["ang_vel_b"], obs["timing"]
    print(io.timing.summary())   # 서보 읽기 · IMU 읽기 시간 평균 · 최대(ms)
```

- `read()` 키: `gravity_b`(몸통 중력 방향, 단위벡터) · `ang_vel_b`(rad/s) · `acc_b`(m/s²) · `quat`(몸통 w,x,y,z) · `gyro`(= `ang_vel_b`) · `roll_pitch`(rad, `safety.tilt_deg` 용) · `stale` · `read_s` · `gravity_chip` · `gyro_chip`.
- 자이로 영점이 안 끝났으면 `read()` 가 거부합니다(`calibrate_gyro()` 로 다시 잴 수 있음).
- 정책 학습 때 지연 10–30 ms 를 흔들었습니다. 실물 지연(표본 시각 → 정책 입력)은 `timestamp_us` 와 `read_s` 로 잴 수 있습니다.

## 4. Jetson: 고정 포트 이름과 권한

1. OpenCR 을 꽂고 번호를 확인합니다: `udevadm info -a -n /dev/ttyACM0 | grep -E 'idVendor|idProduct|serial'`. OpenCR 은 보통 `0483:5740` 입니다 [확인 필요: 위 명령으로].
2. `/etc/udev/rules.d/99-kitoru-opencr.rules` 에 한 줄을 씁니다.

   ```
   SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", SYMLINK+="kitoru_opencr", GROUP="dialout", MODE="0660", ENV{ID_MM_DEVICE_IGNORE}="1"
   ```

   - `ID_MM_DEVICE_IGNORE` 는 ModemManager 가 `ttyACM` 을 모뎀으로 잡고 바이트를 보내는 것을 막습니다.
   - 같은 VID:PID 장치가 둘이면 `ATTRS{serial}=="…"` 를 더합니다.
3. `sudo udevadm control --reload-rules && sudo udevadm trigger` 를 실행하고 다시 꽂습니다. `ls -l /dev/kitoru_opencr` 로 확인합니다.
4. 권한: `sudo usermod -aG dialout $USER` 뒤에 다시 로그인합니다.
5. YAML `buses.opencr.port` 를 `/dev/kitoru_opencr` 로 바꾸거나, 도구에 `--port /dev/kitoru_opencr` 를 줍니다.

참고: U2D2(FTDI, `/dev/ttyUSB0`)로 바꾸면 `/sys/bus/usb-serial/devices/ttyUSB0/latency_timer` 를 16 → 1 로 낮춰야 빠릅니다. OpenCR(`ttyACM`)에는 이 값이 없습니다.

## 5. 통신 속도와 제어 주기

한 주기는 「모든 관절 상태를 한 번에 읽기(Sync Read, 126–135 번지 10 바이트: 부하 · 속도 · 위치)」와 「목표 각을 한 번에 쓰기(Sync Write, Goal Position 4 바이트)」입니다.

**계산 근거 (`kitoru_hw/timing.py`)**

- Protocol 2.0 패킷 크기:
  - 읽기 요청: 14 + n 바이트
  - 서보마다 답: 11 + 10 바이트
  - 쓰기: 14 + n × (1 + 4) 바이트
- 한 바이트는 선 위에서 10 bit(8N1)입니다.
- 서보는 답하기 전에 Return Delay Time(9) 만큼 기다립니다. 공장값은 250 × 2 µs = 500 µs 입니다.
- 예: 서보 15개, 1 Mbps → (14 + 15) + 15 × 21 + (14 + 75) = 433 바이트 = 4.33 ms.

**`python tools/bus_timing.py` 결과** (한 주기 시간과 그때 가능한 최대 주파수)

| 경우 | 서보 수 | 1 Mbps | 2 Mbps | 4 Mbps |
| --- | --- | --- | --- | --- |
| 선 위 시간만, Return Delay 0 | 15 | 4.33 ms · 231 Hz | 2.17 ms · 462 Hz | 1.08 ms · 924 Hz |
| Return Delay 공장값(500 µs) | 15 | 11.83 ms · 85 Hz | 9.66 ms · 103 Hz | 8.58 ms · 117 Hz |
| Return Delay 0 + USB 2 ms + 컴퓨터 0.1 ms/서보 [추정] | 15 | 7.83 ms · **128 Hz** | 5.67 ms · **177 Hz** | 4.58 ms · **218 Hz** |
| 선 위 시간만, Return Delay 0 | 17 | 4.87 ms · 205 Hz | 2.44 ms · 411 Hz | 1.22 ms · 821 Hz |
| Return Delay 공장값(500 µs) | 17 | 13.37 ms · 75 Hz | 10.94 ms · 91 Hz | 9.72 ms · 103 Hz |
| Return Delay 0 + USB 2 ms + 컴퓨터 0.1 ms/서보 [추정] | 17 | 8.57 ms · **117 Hz** | 6.13 ms · **163 Hz** | 4.92 ms · **203 Hz** |

**이 표가 말하는 것**

- **Return Delay Time 을 0 으로 바꾸는 것이 통신 속도를 올리는 것보다 먼저입니다.**
  - 공장값이면 서보 15개에서 기다림만 7.5 ms 라, 4 Mbps 로 올려도 117 Hz 에서 막힙니다.
  - Return Delay Time 은 EEPROM 이라 토크를 끈 채 서보마다 한 번 씁니다. 방법은 아래 코드 또는 Dynamixel Wizard 입니다.
- **USB 2 ms 와 컴퓨터 0.1 ms/서보는 추정입니다.**
  - USB full-speed 는 1 ms 단위로 오갑니다.
  - Jetson Nano 에서 Python SDK 가 답을 푸는 시간은 재 본 적이 없습니다.
  - 실물에서는 `python tools/read_demo.py --rate 500` 으로 읽기만 잰 값을 이 표와 맞춰 보세요.
- **전압 · 온도 · 오류 읽기(144–146, 70 번지)는 따로 두 패킷입니다.** 한 주기마다 읽지 말고 5–10 Hz 로 읽으세요.
- **Fast Sync Read**(답을 한 패킷으로 묶음)를 쓰면 1 Mbps 에서 약 146 Hz(15개)까지 더 갑니다. SDK 4.1 에는 있지만, X430 e-manual 에서는 지원 펌웨어 버전을 못 찾았습니다 [확인 필요]. 지금 코드는 일반 Sync Read 만 씁니다.
- **4.5 Mbps 는 못 씁니다.** 서보는 지원하지만, dynamixel-sdk 의 포트 설정 목록에 없습니다.

**추천**

- 처음에는 1 Mbps(원본 확인값) + Return Delay 0 으로 시작합니다. 약 120 Hz [추정]라 정책 50 Hz 에는 충분합니다.
- 시뮬의 피드백 걷기 제어기(200 Hz)를 그대로 돌리려면 4 Mbps 가 필요하고, 그것도 빠듯합니다. 이때 서보 Baud Rate(8) 를 모두 바꾸고 OpenCR 다리가 그 속도를 넘기는지 확인해야 합니다 [확인 필요].

```python
# Return Delay Time 을 0 으로 (토크 끈 상태에서, 한 번만)
from kitoru_hw.dxl_bus import DxlBus
with DxlBus("/dev/kitoru_opencr", 1_000_000, {"R_knee": (4, "XM430-W350")}) as bus:
    bus.write("R_knee", "RETURN_DELAY_TIME", 0)
```

## 6. 처음 켤 때 안전 순서

비상정지는 「배터리 커넥터 뽑기 · OpenCR 전원 스위치」입니다. 손이 닿는 곳에 둡니다. 소프트웨어 비상정지(`Robot.emergency_stop()`, ID 254 로 토크 끄기 한 패킷)는 그다음입니다.

- **물리 비상정지(전원 스위치)가 토크를 끄는 유일한 보장입니다.**
  - USB 가 빠지거나 프로그램이 죽으면, 소프트웨어는 토크를 끌 수 없습니다.
  - Bus Watchdog(7절)은 움직임을 멈출 뿐이고, 토크를 끈다고는 e-manual 에 적혀 있지 않습니다.
- **YAML 의 `calibrated: false` 관절은 토크를 켜지 않습니다**(`arm()` · `range_check.py` 거부). 지금 v20 YAML 은 모두 false 입니다. 아래 1–4 단계를 마친 관절만 사람이 손으로 `calibrated: true` 로 바꿉니다.

1. **서보 하나씩 ID 정하기.**
   - 새 X430 서보는 ID 1, 57600 bps 로 옵니다.
   - 하나만 꽂고 `python tools/scan_ids.py --bauds 57600,1000000` 으로 찾습니다.
   - Dynamixel Wizard 로 ID · Baud Rate(1 Mbps) · Return Delay Time(0) 을 씁니다.
   - 같은 때 Max Voltage Limit(32)도 봅니다. XL430 공장값 14.0 V 는 만충 12.6 V 보다 높아서 괜찮습니다.
2. **버스 전체 확인.** 모두 꽂고 `python tools/scan_ids.py` 를 돌려 YAML 과 「matches the YAML」인지 봅니다. 모델이 다르면 `MODEL?` 이 뜹니다.
3. **토크 끈 채 읽기.**
   - `python tools/read_demo.py --config config/joints_v20.yaml` 를 돌립니다.
   - 관절을 손으로 + 쪽(MuJoCo 와 같은 방향)으로 돌려, 값이 커지는지 봅니다.
   - 반대면 그 관절 `direction: -1` 로 바꿉니다.
4. **영점 기록.**
   - 로봇을 영점 자세(MuJoCo 관절 0)로 맞추고 `python tools/record_zero.py` 로 값을 봅니다.
   - 괜찮으면 `--save` 를 붙입니다.
   - `offset > 30 deg` 가 뜨면 혼을 한 칸 옮겨 끼우는 것이 낫습니다.
   - 같이 나오는 `Min/Max Position Limit` 값을 Wizard 로 서보 EEPROM 에 쓰면, 서보 자신도 그 밖으로 안 갑니다(두 번째 울타리).
   - 관절이 움직이는 중이거나(`moving`) 한계가 0..4095 밖이면 `--save` 가 저장을 거부합니다. YAML 은 임시 파일에 쓴 뒤 바꿔치기해서, 도중에 죽어도 반쪽 파일이 남지 않습니다.
   - **그다음 사람이 `calibrated: true` 를 켭니다.** ID(2단계) · 방향(3단계) · 영점(4단계)을 모두 확인한 관절만입니다. 도구는 이 값을 자동으로 켜지 않습니다.
5. **한 관절씩 범위 확인.**
   - 로봇을 거치대에 겁니다. 다른 서보는 힘이 없습니다.
   - `python tools/range_check.py --joint R_knee` 를 돌립니다. 기본값은 Goal PWM 200(23 %), 느린 속도, 2° 씩입니다.
   - 인자 범위: `--step` 0 초과 5° 이하, `--pwm` 1–885, `--profile-velocity` 1–50(0 은 「속도 제한 없음」이라 거부).
   - 멈춘 자리를 목표로 다시 잡은 뒤(`Robot.hold()`) 같은 간격으로 처음 자세에 돌아옵니다.
   - `STOPPED SHORT` 가 나오면 무언가 닿거나 힘이 모자란 것입니다. 그 각을 기록합니다.
   - 범위가 비어 있는 관절(고관절 롤)은 `--lo -5 --hi 5` 처럼 직접 범위를 줍니다. ±30° 를 넘으면 거부합니다. 그 실행에서만 쓰고 YAML 은 안 바꿉니다.
6. **전체 토크 켜기.** 거치대에서 `Robot.arm()` 으로 지금 자세를 잡아 봅니다(`read_demo.py --demo --config config/joints_v20.yaml --joints …` 로 작은 움직임).
7. 그다음에 바닥 · 줄에 매단 상태에서 정책을 돌립니다.

## 7. 안전 규칙 (`safety.py`, 값은 `joints_v20.yaml` 의 `safety:`)

| 규칙 | 하는 일 | 근거 |
| --- | --- | --- |
| 한계로 자르기 | 명령 각을 `soft_limit_rad` 로 자릅니다. 한계가 없으면(`null`) 그 관절은 명령을 거부합니다 | `joint_ranges_v19.csv` ∩ `out20/joints.csv` |
| 서보 한계 재검사 | 자른 뒤 tick 이 서보의 Min/Max Position Limit 밖이면 거부합니다(원본과 같음) | 원본 |
| 짝 관절 | `lo ≤ a·q₁ + b·q₂ ≤ hi` 를 넘으면 두 관절을 가장 적게 옮깁니다. 지금 자리(두 고관절 요, ±40°)만 있고 꺼져 있습니다 [확인 필요] | `JOINT_RANGES.md` 2절 |
| 넘어짐 | 몸통 기울기 > 25° 이면 모든 관절 Goal PWM 을 265(30 %)로 한 패킷에. 다시 서면 `reset_fall()` | `BATTERY_6000.md` |
| 저전압 | 서보 전압의 가운데값 < 10.8 V 가 연속 3번이면 토크 끄고 멈춤. 11.1 V 미만은 경고 [추정] | 셀당 3.6 V |
| 온도 | 서보에서 Temperature Limit(31)을 읽어, 한계 − 10 °C 경고 · 한계 − 3 °C 멈춤(XL430: 62 · 69 °C) | e-manual 공장값 XL 72 · XC/XM 80 |
| 하드웨어 오류 | Hardware Error Status(70)가 0 이 아니면 멈춤 | — |
| 연결 끊김 | 직렬 포트 예외, 또는 패킷 3개 연속 분실이면 `DxlDisconnected` | — |
| 한 번에 움직이는 양 | 명령마다 직전 명령에서 `max_step_deg`(기본 5°) 넘게 못 가게 자릅니다 | XM430 무부하 4.5 rad/s × 20 ms(50 Hz) ≈ 5.2°: 서보가 한 주기에 실제로 갈 수 있는 것보다 큰 점프는 버그 |
| Ctrl-C | SDK 호출 도중 끊겨도 포트 잠금(`is_using`)을 풀고, `disarm()` · `emergency_stop()` 첫머리에서도 풉니다. 그래서 `with Robot(...)` 를 나갈 때 토크가 꺼집니다 | — |
| 정지 우선 | 같은 주기에 넘어짐과 정지가 함께 나오면 정지(토크 끄기)를 먼저 합니다. 넘어짐 PWM 쓰기가 실패하면 다음 주기에 다시 씁니다 | — |
| 비상정지 확인 | 브로드캐스트 뒤 서보마다 읽어 보고, 켜져 있으면 한 대씩 다시 끕니다. 그래도 남은 것은 `SafetyStop` 메시지에 붙입니다 | — |
| Bus Watchdog(선택) | `arm(watchdog_ms=100)` 이면 토크를 켠 뒤 서보마다 98 번지(20 ms 단위)를 씁니다. 그 시간 동안 명령 패킷이 안 오면 서보가 움직임을 멈춥니다. `disarm()` 에서 0 으로 되돌립니다(0 은 「발동됨」 상태도 풂). 기본은 끔 | e-manual: 「DYNAMIXEL will stop」, 토크를 끄는지는 안 적힘, 발동 뒤 Goal 값은 읽기 전용 |

- **넘어짐 훅에 Goal PWM 을 쓰는 이유:** 배터리 문서는 「Goal PWM 이나 PWM Limit」입니다. PWM Limit(36)은 EEPROM 이라 토크를 켠 채로는 못 씁니다(e-manual). Goal PWM(100)은 RAM 이고, 위치 모드에서도 출력 PWM 의 마지막 제한으로 쓰입니다(e-manual). 그래서 Goal PWM 으로 정했습니다.
- **SDK 의 응답 기다림:** 서보 하나가 답을 안 하면 SDK 는 약 34 ms(`LATENCY_TIMER = 16` × 2 + 2 ms) 기다린 뒤 실패를 돌려줍니다. 그 주기는 그만큼 늦어집니다.
- 저전압 · 온도에서 「멈춤」은 토크를 끄는 것이라 로봇이 주저앉습니다. 걷는 중 순간 처짐으로 멈추지 않게 연속 3번을 봅니다.
- **Watchdog 을 쓸 때:** 매 주기 읽기나 쓰기가 오가므로 따로 살려 두는 패킷은 필요 없습니다. XM430 문서는 「모든 명령 패킷」이 통신으로 친다고 적었고, XL430 문서는 어떤 패킷이 시계를 되돌리는지 적지 않았습니다 [확인 필요: Sync Write 만으로 XL430 이 살아 있는지].
- 발동하면 Bus Watchdog 이 −1 이 되고 `arm()` 이 거부합니다. `disarm()` 이나 Wizard 로 0 을 쓰면 풀립니다.

## 8. 정책(강화학습) 연결 자리

```python
from kitoru_hw.joints import load_config
from kitoru_hw.robot import Robot
from kitoru_hw.policy_io import PolicyIO

with Robot(load_config("config/joints_v20.yaml")) as robot:
    robot.start_monitor()          # 서보에서 온도 · 전압 한계를 읽음
    robot.arm(profile_velocity=100, watchdog_ms=100)  # 100 × 0.229 rpm = 2.4 rad/s 상한, 100 ms 끊기면 멈춤
    io = PolicyIO(robot, imu=my_imu)
    for step in range(...):
        obs = io.observe()         # {"q": [15], "dq": [15], "imu": {...} 또는 None, "time": …}
        io.act(policy(obs))        # 목표 각 15개, YAML 순서, 라디안. 작업 범위로 잘림
        if step % 20 == 0:
            robot.safety_step(tilt_deg=tilt_from(obs["imu"]))  # 넘어짐 · 전압 · 온도
```

- **`profile_velocity=0`(속도 제한 없음)으로 시작하지 마세요.**
  - 정책이 튀는 목표를 내면 서보가 최고 속도로 갑니다.
  - 처음에는 위처럼 상한을 두고, 시뮬에서 정책이 쓰는 최대 관절 속도를 본 뒤 그 값에 맞춰 올립니다.
  - 명령마다 5° 상한(7절)은 그 밖의 두 번째 울타리입니다.
- **관절 순서는 YAML `joints:` 순서입니다.** 시뮬도 같은 순서로 맞추면 됩니다.
- **각도는 로봇 기준 라디안입니다(방향 · 영점 적용 뒤).** MuJoCo 관절 각과 같은 뜻이 되게 방향 · 영점을 맞춥니다.
- **IMU 는 `OpenCRImu`(3.1절)를 넘깁니다.** 넘기지 않으면 `NullImu` 라 `obs["imu"]` 가 None 입니다. 넘어짐 감시에는 `tilt_deg(*obs["imu"]["roll_pitch"])` 를 씁니다.
- `obs["timing"]` 에 그 주기의 서보 읽기 · IMU 읽기 초가, `io.timing.summary()` 에 평균 · 최대가 남습니다.

## 9. 원본(`legacy/XL430_robot_base.py`)과 비교

| 항목 | 원본 | 새 코드 | 같은가 |
| --- | --- | --- | --- |
| 읽기만 | `python XL430_robot_base.py` | `python tools/read_demo.py` | 같음. 출력 글자까지 같음(시험 `test_read_only_prints_the_same`) |
| `--demo` | 약 5° → 원래 자리, 0.05 초마다 도착 확인 | `python tools/read_demo.py --demo` | 같음. 보내는 목표 패킷이 바이트까지 같음(`test_demo_sends_identical_goal_packets_and_restores`) |
| 기본 포트 | `COM3` | `bench_legacy.yaml` 의 `COM3`(v20 YAML 은 `/dev/ttyACM0`) | 같음 |
| 관절 설정 | 코드 안 `JOINTS`(ID 6 · 7 · 12) | `config/bench_legacy.yaml` | 같은 값 |
| 모델 확인 | ping 해서 1060(XL430)만 허용 | 관절마다 YAML 모델 번호와 대조 | 넓힘 |
| 토크 켜기 전 점검 | 위치 모드 · 토크 꺼짐 · Status Return Level 2 · 워치독 0 · 오류 0 · 지금 위치가 한계 안 | 같음 | 같음 |
| 토크 켤 때 | 가속 5 · 속도 15 → 지금 위치를 목표로 → 토크 켜기 → 3 tick 넘게 밀리면 취소 | 같음(가속 · 속도 · Goal PWM 을 인자로 바꿀 수 있음) | 같음(기본값) |
| 끌 때 되돌리는 값 | 가속 · 속도 · 목표 위치 | 같은 셋 + Goal PWM | 하나 더 |
| 범위 밖 명령 | 거부(`ValueError`) | 먼저 소프트웨어 한계로 **자르고**, 그 tick 이 서보 위치 한계 밖이면 거부 | 다름(스펙: 자르기). 데모의 「5° 갈 자리가 있나」 검사는 원본처럼 거부 방식(`target_tick`) |
| 한계 없는 관절 | 서보 위치 한계만 검사 | `require_soft_limits: false`(시험대)면 원본과 같음. v20 은 거부 | 시험대는 같음 |
| 읽는 값 | 속도 · 위치(128 번지 8 바이트) | 부하 · 전류 + 속도 · 위치(126 번지 10 바이트) | 넓힘 |
| 여러 관절 쓰기 | Sync Write 한 패킷 | 같음(버스마다 한 패킷) | 같음 |
| 통신 오류 | `RuntimeError` | `DxlCommError` · `DxlDisconnected`(둘 다 `RuntimeError`), 끊긴 서보 이름 표시 | 넓힘 |

## 10. 하드웨어 없이 시험

```bash
cd robot
mkdir -p tmp/pytest
uv run --no-project --python 3.12 --with pytest --with pyyaml --with dynamixel-sdk python -m pytest -q -p no:cacheprovider --basetemp=tmp/pytest
rm -rf tmp && find . -name __pycache__ -prune -exec rm -rf {} +
```

- `--basetemp` 를 주지 않으면 pytest 가 `/tmp` 에 씁니다.
- `tests/test_regressions.py` 는 2026-11-26 검토에서 나온 결함 10개(Ctrl-C 뒤 포트 잠금, 넘어짐 PWM 재시도, 교정 안 된 관절 거부, Watchdog, 비상정지 재시도, 한 번 움직이는 양, PWM 되돌리기, 범위 확인 인자, 설정 덮어쓰기, 영점 저장)를 다시 막는 시험입니다.

- `tests/fake_dxl.py` 는 SDK 의 `PortHandler` 를 바꿔 끼운 가짜 직렬 포트입니다.
  - 그 뒤에 가상 X430 서보가 있어서, 진짜 dynamixel-sdk 가 만든 패킷 바이트를 받아 답합니다.
  - 답하는 명령: PING, READ, WRITE, SYNC READ, SYNC WRITE, 브로드캐스트.
  - 토크가 켜져 있으면 EEPROM 쓰기를 거부합니다.
- 원본 스크립트도 같은 가짜 서보에 붙여서 새 도구와 출력 · 패킷을 비교합니다.
- `tests/test_opencr_imu.py` 는 펌웨어의 보드와 무관한 부분(`bridge_core.cpp`)을 PC 의 g++ 로 공유 라이브러리로 빌드해(`--basetemp` 안), 진짜 dynamixel-sdk → 펌웨어 코어 → 가짜 서보로 패킷을 왕복시킵니다(`tests/fake_opencr.py`). g++ 가 없으면 그 시험은 건너뜁니다.
  - 확인하는 것: CRC · 스터핑이 SDK 와 같음, PING · READ 200 이 버스에 안 나감, 스터핑된 답, 오류 답, 서보 패킷이 바이트 하나 안 바뀌고 넘어감, 브로드캐스트가 바로 넘어감, 멈춘 앞머리를 시간 초과로 넘김, 회전행렬 부호, `PolicyIO` 시간 기록.
- 펌웨어 전체(`.ino`)는 `arduino-cli compile --fqbn OpenCR:OpenCR:OpenCR` 로 컴파일만 확인했습니다(2026-11-28, 보드 패키지 1.5.3, 128,992 바이트). 실물 보드에서는 안 돌렸습니다.

## 11. [확인 필요] 모음

- OpenCR 에 올라간 펌웨어가 `usb_to_dxl` 다리인지, 2 · 4 Mbps 를 넘기는지, TTL 셋이 한 버스인지, 서보 전원 경로 · 10 A 퓨즈와 넘어질 때 순간 전류.
- OpenCR USB VID:PID(`0483:5740` 로 가정).
- Jetson 에서 uv Python 3.12 가 도는지.
- Fast Sync Read 지원 펌웨어 버전.
- `joints_v20.yaml`:
  - ID · 방향 · 영점은 모두 자리표시입니다(조립 후 확정).
  - 고관절 롤 범위는 비어 있습니다(v20 범위 검사 후 채움).
  - 고관절 피치는 「v19 표 ∩ v20 설계 범위」라 v20 검사로 다시 봐야 합니다.
  - `nod_crank` 는 크랭크 각 범위(−70~40°)이고, 머리 각 한계(−20.8~28°)를 크랭크 각으로 바꾸지 않았습니다.
  - 짝 관절 제한은 자리만 있고 꺼져 있습니다.
  - 모든 관절이 `calibrated: false` 입니다.
- Bus Watchdog 이 발동할 때 토크가 켜진 채 붙잡는지(e-manual 에 없음), XL430 에서 Sync Write 가 시계를 되돌리는지.
- 저전압 경고 11.1 V, 온도 여유 10 · 3 °C 는 정한 값이 아니라 출발점입니다.
- 제어 주기 표의 USB 2 ms · 컴퓨터 0.1 ms/서보는 추정입니다. `read_demo.py --rate` 로 실측해야 합니다.
- IMU 펌웨어(3.1절): 칩 축 ↔ 보드 축(`r_board_chip`), 가속도 눈금 16384 LSB/g, 서보가 Sync Read 에서 앞 ID 답을 기다리는지, USB 포트를 열 때 OpenCR 이 다시 시작하는지, 업로드에 부트로더 모드 · udev 규칙이 필요한지, 실물 중계 지연(`loop_max_us`)과 IMU 읽기 시간, 몸통 축이 MuJoCo `torso` 축과 같은지.
