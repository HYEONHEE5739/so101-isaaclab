# Joint Precision Calibration

2026-10-12. **HARDWARE_NOT_TESTED**. 기존 PyQt Real/Sim UI에 관절별 Calibration 탭을 추가했다.
자동 joint 이동, LeRobot Calibration 파일 덮어쓰기, Homing/Range/Drive register 변경은 구현하지 않는다.
Scene/Builder/Workspace/v5–v8/사용자 Snapshot·사진·HDF5·checkpoint는 변경하지 않았다.

## 구조와 실제 source 확인

`scripts/real2sim/real2sim_ui.py` → `real2sim.ui.Window` → `precision_ui.PrecisionPanel`
→ 기존 RuntimeLauncher/file IPC → 기존 `real2sim_sim.py --backend calibration`
→ `Runtime.command('precision.*')` → `precision.Calibration` → `hardware.CalibrationDevice`.

새 launcher/독립 앱은 없다. Calibration worker는 Isaac/카메라를 시작하지 않는다.
MuJoCo는 사용자가 **URDF current/reference preview**를 누를 때만 import/render한다.
UI 시작/탭 전환만으로 serial·카메라·모터를 연결하지 않는다.

기존 구현에서 확인한 부분:

- repository `devices/lerobot/so101_leader.py`: Leader 5 arm motor는 DEGREES, gripper는 RANGE_0_100.
- repository `devices/lerobot/motors/motors_bus.py:_normalize/_unnormalize`와 설치된 LeRobot 0.4.1의 동명 함수: DEGREES는 Range 중간값, RANGE_0_100은 range endpoints를 사용한다. DEGREES 경로에서 drive_mode 반전은 적용하지 않는다.
- Feetech `read_calibration`, `_get_half_turn_homings`, encoding tables: raw `Present_Position`은 이미 firmware Homing_Offset이 적용된 register 값이다. encoder resolution은 motor model table에서 읽는다. STS3215는 4096 counts/turn, 기존 degree normalization의 분모는 4095다.
- 설치된 Follower `connect/calibrate/configure`: 기본 connect/configure는 토크·control register 설정을 포함한다. Calibration은 이를 호출하지 않고 bus read-only connect만 사용한다.
- `assets/SO101/urdf/so101_isaaclab.urdf`: 6 named revolute joints의 parent/child/origin/rpy/axis/limits를 읽는다. URDF 파일/mesh는 수정하지 않는다.
- 기존 `so101_joint_mapping.py`는 calibrated degree/percent → model rad의 legacy mapping이다. 활성 Precision Profile 없이 기존 모드를 유지한다.

Raw 읽기는 `read('Present_Position', joint, normalize=False)`, 기존 값 읽기는 `normalize=True`로 분리한다.
sequential raw/native 읽기 사이 값이 3 counts보다 달라지거나 normalization clipping이 감지되면 reference를 받지 않는다.
Host read timestamp·device identity·joint·calibration hash·motor spec을 기록한다. 이는 hardware exposure/동기화 시간 인증이 아니다.

## UI 실행

기존 환경과 기존 진입점을 그대로 쓴다. dependency를 새로 설치하지 않았다.

```bash
cd /home/hyeonhee/soarm_isaaclab/soarm101_lab
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

**4 · Precision Calibration** 탭. 오른쪽 기존 연결 설정의 Leader/Follower port와 각 Calibration file/ID/directory를 사용한다.
worker Python은 기존 Workspace Pipeline의 실행 Python을 사용한다. 로컬에서는
`/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python`.
기존 LeRobot calibration 파일 및 hardware register가 서로 일치해야 read-only 연결된다.
파일이 없거나 register가 다르면 자동으로 초기 calibration을 만들지 않는다.

Capture/Real-Sim/Workflow가 장치를 사용 중이면 먼저 사용자가 Stop/Disconnect해야 한다.
Calibration Connect가 기존 Teleop worker를 자동 중지하거나 장치를 빼앗지 않는다.

## Follower 한 관절 절차 — 손으로 움직이며 확인

1. Robot `follower`와 Joint를 선택하고 오른쪽 port/ID/기존 LeRobot calibration 파일을 확인한다.
2. **1. 연결 · 위치만 읽기**. 모터 숫자(raw)와 맨 위 **현재 각도**가 나온다.
   해당 관절의 저장·적용한 Profile이 있으면 **저장 보정 적용**이라고 표시하고 보정 각도를 사용한다.
   아래 **보정 전 현재각**은 이번 재보정 직전의 현재 적용값이다. 이미 저장된 Precision이 있으면 그 각도를 쓰며 새 계산 전에는 보정 후와 같다.
   자동 goal/torque/register 변경 없음.
3. 로봇을 지지하고 support를 체크한 뒤 **2. 선택 관절 힘 풀기**의 승인 창을 확인한다.
4. 손으로 알려진 기준 자세에 맞춰 **손으로 맞춘 실제 기준각**에 각도를 입력하고 **3. 지금 자세 기록**.
   모든 새 기록은 `train`(보정 계산용)으로 저장하며 용도 선택창은 없다.
   표에는 입력 기준각, raw, 현재 저장 보정으로 계산한 보정 전 각도/차이와 보정 후 각도/차이가 나온다. 새 계산 전에는 현재 적용된 변환을 양쪽에 사용한다.
5. **4. 보정 계산**. 표의 보정 후 각도와 기준각 차이가 채워진다.
   한 점이면 기존 방향·비율을 유지한 영점(offset) 보정이다.
   다른 알려진 기준각을 추가하면 방향·비율도 계산한다. raw 차이 ≥20 counts, 각도 차이 ≥5° 필요.
6. **5. 자세 검증 · 손으로 움직여 확인**.
   이제 손으로 관절을 움직이며 현재 raw, 보정 전 현재각, 보정 후 현재각,
   보정 전·후 차이를 실시간으로 확인한다. 별도 validation 자세의 기록은 요구하지 않는다.
   입력 기준각 대비 차이도 표시한다. 사용자가 실제로 그 기준 자세에 맞췄을 때만 의미가 있다.
   이 버튼은 goal/torque/활성 pointer를 쓰거나 로봇을 움직이지 않는다.
7. 화면이 원하는 자세와 대응하는지 직접 확인한 뒤 **6. 보정 파일 저장** → **7. 확인한 보정 적용**의 승인 창 확인.
   소프트웨어 Profile만 활성화한다. 기존 LeRobot JSON/register는 변경하지 않는다.
   현재 세션에서 확인한 다른 관절의 Candidate도 함께 저장한다. 이미 적용한 관절은 유지한다.
   관절별로 저장·적용해도 되고, 여러 관절을 확인한 뒤 마지막에 함께 저장·적용해도 된다.
8. **보정 작업 종료**로 연결을 닫는다. 이미 해제한 관절 힘을 자동으로 다시 켜지 않는다.

허용 오차 기본값은 **0.5°**다. **새 보정식으로 계산한 각도 − 사용자가 입력한 기준각**의
최대 절댓값에 적용한다. 보정 계산은 끝났더라도 이 값이 초과하면 review가 NEEDS_MORE_DATA가 되어 적용하지 못한다.
이 값은 raw counts 차이 자체나 0°와 90° 사이의 각도 차이에 대한 제한이 아니다.

현재 fit은 영점과 각도 비율을 함께 계산하므로 서로 다른 **두 점이면 두 점을 통과하는 직선**을 만든다.
두 점의 재현 오차는 거의 0°가 되어 raw 간격이 이상한지 0.5°로 판단할 수 없다.
세 점 이상이면 직선으로 함께 설명되지 않는 입력이 있어 오차 검사가 유효하다.
수동 검토 결과는 REVIEWED_MANUAL/independent_validation=False/physical_verified=False이며 실측 정확도 인증이 아니다.

기준 표 아래 raw 간격 비교는 현재 저장 비율로 예상한 두 기준점의 raw 차이와 실제 raw 차이를 보여준다.
기존 변환 기준 두 각도 간격과 입력 기준각 간격의 차이도 degree로 표시한다.
예를 들어 기존 0°/90° raw가 2000/3024라면 새 2020/3044는 두 점 모두 +20 counts로 영점만 달라진 경우다.
새 2020/3080이면 +20/+56 counts로 간격도 달라져 비율 변화 또는 수동 자세 오차가 있다.
두 점만으로 어느 원인인지 자동 식별하지 않으며 raw 간격 비교값을 0.5°의 적용 차단 조건으로 쓰지 않는다.
그리퍼는 기존 joint mapping 비율을 사용하므로 모든 관절에 90°=1024 counts를 강제하지 않는다.

별도 검증 데이터를 사용한 이전 `VALIDATED_MANUAL` Profile은 그대로 읽는다.
새 worker는 `precision.review`만 검토 경로로 사용하며 이전 `precision.validate` 실행 명령은 제거했다.
`precision.record`에는 손으로 맞춘 `angle_rad`를 전달한다. worker가 `split=train`,
`reference_kind=angle`을 고정한다. validation/closed/open/opening_m 입력은 새 기록으로 받지 않는다.
기존 파일의 해당 필드는 원본 그대로 읽고 증거를 재검사한다.

보정 후 현재각은 Candidate 계산값이며 활성화 전까지 **검토용·미적용**으로 표시한다.
0°와 90°는 보정식의 영점·비율을 계산하는 기준점이며 사용 범위의 양 끝이 아니다.
계산한 보정식은 **URDF 관절 범위 전체**에 사용한다. 음수 및 90° 이후도 해당 joint의 limit 안이면
현재각에 표시한다. UI에 선택 joint의 URDF 표시 범위를 보여준다.
Candidate의 기록 구간 밖은 **기록 범위 밖·보정식으로 계산**이라고 표시하지만 각도를 숨기지 않는다.
한 점만 기록했을 때도 영점 보정은 전체 URDF 범위에 사용하되 비율은 기존 변환을 유지한다.
이는 범위 전체의 실물 정확도를 검증했다는 뜻이 아니다. 실제 follower goal은 기존 모터 calibration
범위도 확인하므로 그 범위가 URDF보다 좁으면 명령을 거절한다. 모터 range/register를 자동 변경하지 않는다.
저장 보정은 재기록·재저장 없이 사용한다. 기존 worker에는 이전 코드가 남으므로 **보정 작업 종료 →
UI 재시작 → 연결·위치만 읽기**로 새 코드를 적용한다. 자동 연결/이동은 없다.

현재각의 `보정 전`은 현재 적용한 저장 보정값이며, 선택 joint에 Precision이 없을 때만 기존 LeRobot native→legacy URDF 값을 사용한다. 새 Candidate를 계산하면 기존 적용값과 새 계산값을 비교하고, 적용 후에는 양쪽이 같아진다. 이동 중 live read는 같은 raw packet을 SDK로
정규화하여 서로 다른 시각의 raw/native를 혼합하지 않는다. **기준 기록**은 기존 별도 raw/native 읽기 및
연속 두 read의 안정성 검사를 유지하므로 움직이는 중에는 기록하지 않는다.

재시작 후 명시적으로 같은 장치에 연결하면 worker가 `active.json`의 Profile을 읽는다.
새 session Candidate가 비어 있어도 **보정 후 현재각**은 현재 저장·적용 값과 같게 표시되며
적용했던 관절을 다시 기록·계산할 필요는 없다. 저장만 하고 적용하지 않은 파일은 자동 활성화하지 않는다.
Profile이 다른 관절에만 있는 경우 선택 관절에는 **적용한 보정 없음**이라고 표시한다.
저장 보정 범위 밖/변환 실패이면 현재각을 추측하지 않고 그 상태를 표시한다.
실험 Candidate의 추정값과 실제 적용한 runtime mapping은 구분한다.
같은 device identity/URDF의 활성 보정이 있으면 해당 기준 기록을 새 세션 표에 불러온다.
이 기록은 과거 측정의 복사본이며 새로 측정한 자료가 아니다. 새 journal에 loaded_profile_id를 남긴다.
원본 Profile/과거 journal은 변경하지 않는다. 기록을 삭제·추가하면 다시 계산·확인·저장·적용한다.
부분 관절 파일을 적용할 때도 기존 다른 관절을 유지하며, 합칠 필요가 있으면 원본을 덮어쓰지 않고 새 Profile을 만든다.

URDF preview/JSON/device hash/Fake/복구 설정은 접힌 상세 영역에 있다. Camera/Sim은 preview 버튼을
누르기 전까지 시작하지 않는다. 그리퍼도 기준각만 기록하며 open/closed/간격 입력은 없다.
음수 기준각의 − 부호를 확인한다. 입력 오류가 있는 이전 세션은 자동 수정/삭제하지 않는다.

### 잘못 기록한 자세 삭제·재기록

1. 기록 표에서 잘못된 **행을 클릭**한다.
2. **선택 자세 삭제** → 확인 창에서 승인한다.
3. 로봇을 올바른 자세에 맞춰 실제 기준각을 입력하고 **지금 자세 기록**을 누른다.
4. **보정 계산 → 자세 검증 → 저장 → 적용**을 다시 수행한다.

삭제는 현재 관절의 계산/표에서 해당 기록을 제외한다. 다른 관절과 이미 적용한 Profile은 유지한다.
기준이 바뀌었으므로 해당 관절의 미적용 Candidate와 UI의 저장 경로/검토 상태는 초기화한다.
모든 행을 제외했다면 새 기준을 기록하기 전에는 계산할 수 없다. 새로 기록한 자세는 새로운 ID를 갖는다.
이 기능은 현재 연결 세션 표의 기록(활성 Profile에서 불러온 복사본 포함)을 대상으로 한다.
과거 원본 Profile/journal의 기록은 변경하지 않는다.

저장 journal의 `references`에는 **모든 원래 측정값**을 보관한다.
`exclusions`에는 reference_id/joint/제외 이유/시각을 추가한다. journal을 읽어 계산할 때에는 이 ID들을
제외해야 한다. Runtime/UI의 status.references와 Candidate에는 이미 필터된 계산용 기록만 전달한다.
파일 저장에 실패하면 제외 상태와 기존 Candidate를 그대로 유지한다. 삭제는 모터에 명령을 보내지 않는다.

## Leader 절차 / 두 로봇 비교

Follower부터 보정한다. Leader는 Robot `leader`, 동일 Joint를 선택하고 자신의 port/leader Calibration file을 사용해 위 절차를 별도로 수행한다.
raw/homing offset 숫자가 Follower와 같을 필요는 없다. 같은 실제 angle이 같은 URDF coordinate로 변환돼야 한다.

두 로봇을 각자 read-only 연결하고 해당 Joint Profile을 활성화한 뒤, 사용자가 실제로 같은 정적인 자세에 맞췄음을 확인하여 **Compare Leader / Follower**를 실행한다.
각 장치의 timestamp/measurement와 공통 q 차이를 표시한다. 이 버튼은 로봇을 정렬/이동하지 않는다.

그리퍼도 다른 관절과 동일하게 **손으로 맞춘 실제 기준각**만 입력한다.
한 점은 영점 보정, 두 개 이상의 서로 다른 기준각은 방향·비율 보정이며 간격(mm)은 요구하지 않는다.
입력 각도는 URDF `gripper` joint frame의 각도다. 닫힌 상태를 자동으로 URDF 0°라고 정하지 않는다.
새 reference는 `reference_kind=angle`, opening_m 필드 없이 저장하고 새 Candidate는
`reference_mode=angle`로 저장한다. 기존 LeRobot gripper의 RANGE_0_100 정규화/모터 register는
유지하며 기존 SDK의 percent↔raw를 통해 URDF rad로 변환한다. 퍼센트를 degree로 간주하지 않는다.
URDF limits/기존 모터 goal range 및 승인 검사를 유지하며 기록 raw 범위는 증거로만 보존한다.

이미 저장된 endpoint 기반 Profile과 증거는 변경하지 않는다. `reference_mode`가 없는 이전 Profile은
`legacy_endpoints` 규칙으로 재검사하여 기존 동작을 보존한다. 이 legacy 경로는 새 UI 입력이 아니다.

## 수학과 코드 경로

`precision.fit_joint`는 train raw/reference angle만 사용한다:

```text
q_model_rad = rad_per_count × (raw_unwrapped − anchor_raw) + reference_angle_rad
```

sign, scale, zero_raw를 따로 저장한다. 한 점이면 기존 slope를 유지한 offset만 계산하며 방향/비율은 새로 식별하지 않는다. `review_joint`는 기록 기준의 최대 재현 오차를 확인하고 live 수동 검토를 시작한다. 별도 holdout을 요구하지 않으며 물리 정합을 자동 인증하지 않는다. `validate_joint`는 이전 saved Profile의 증거 재검증에만 유지한다.
raw는 이미 Homing이 적용된 Present_Position이다. Homing_Offset을 다시 더하거나 빼지 않는다.
native degree/percent를 raw로 되돌릴 때 기존 range-mid/range-endpoint/drive-mode 규칙을 사용한다.

`PrecisionMapper.raw_to_urdf`는 실제 raw+native에서 calibrated joints만 변환한다.
`calibrated_to_urdf`는 기존 LeRobot native 값에서 동일 raw coordinate를 복원한다.
`urdf_to_calibrated`는 역변환으로 실제 follower goal의 기존 normalized input을 만든다.
device identity/calibration hash/normalization/URDF hash가 다르면 적용을 거절한다.

fit의 reference branch는 기존 half-turn 미만 조건을 유지한다. 실시간 변환은 nearest-anchor unwrap이
아니라 URDF limits의 역변환 구간에서 유일한 encoder branch를 선택한다. 따라서 Wrist roll처럼
전체 관절 폭이 반 바퀴보다 넓어도 구간을 식별할 수 있으면 계산한다. 가능한 branch가 여러 개면
추측하지 않고 변환 오류를 표시한다. 동일한 규칙을 UI/Calibration/Runtime/Compare에서 사용한다.
`raw_domain`은 실제 기록한 증거 범위이며 사용 제한이 아니다. 원본 Profile/Reference를 덮어쓰지 않는다.
goal은 URDF limits와 기존 calibrated motor range를 검사하고 유일한 branch의 정수 raw로 변환한다.
limit의 소수 raw를 반올림할 때 limit 밖으로 나가지 않는 정수값을 사용한다(최대 한 count 차이).
잘못된 goal을 범위 안으로 implicit clipping하지 않는다.
Profile의 fallback joints는 legacy라고 명시한다. VALIDATED_MANUAL은 physical_verified가 아니다.

## 저장·복구 계약

기본 `<UI workspace>/joint_calibration/`:

```text
joint_calibration/
  follower/
    references/references_<uuid>.json  # so101.joint-references/1
    profiles/profile_<uuid>.json      # so101.precision-calibration/1
    active.json                       # profile hash + previous pointer + 승인 시각
  leader/...
  fake/follower/...                   # Fake 결과는 실제 active pointer와 분리
  fake/leader/...
```

Profile에는 role/device/calibration/URDF hash, selected per-joint coefficients와 모든 reference/validation evidence가 들어간다.
한 관절의 후보 저장 시 같은 device의 현재 Profile 나머지 joints를 유지한다.
inactive candidate 저장은 가능하지만 기록 오차 검사를 통과하지 않은 Profile 활성화는 거절한다. `REVIEWED_MANUAL`과 기존 `VALIDATED_MANUAL`은 명시 승인으로 활성화할 수 있다.
기존 saved profile을 덮어쓰거나 삭제하지 않는다. **Restore Previous Profile**은 별도 승인으로 이전 pointer 또는 legacy를 복구한다.
Fake와 실제 장치를 한 session에 섞지 못하며 Fake active Profile은 실제 Real/Sim에서 사용하지 않는다.

## Real/Sim 적용

Calibration session 종료 후 사용자가 다시 Connect한다. UI는 저장된 active pointer가 있으면
**저장된 관절 보정 사용 (다음 Connect)**과 **URDF 관절 기준 사용 · 예전 Scene 영점 보정 제외**를 기본 선택한다.
실제 보정 Apply 성공 시에도 다음 Connect 설정을 선택한다. Fake 보정은 실제 설정에 적용하지 않는다.
Follower만 먼저 완료했다면 static measured-q 비교/캡처는 가능하다.
Precision Teleop Start는 **Leader/Follower 양쪽에 같은 joint set의 승인된 수동 보정 Profile**이 있어야 허용한다.
partial Profile의 다른 joints는 기존 legacy 모드임을 metadata에 남긴다.

```text
Leader raw → 기존 native degree/percent → Leader PrecisionMapper → common URDF q
→ Follower PrecisionMapper inverse → follower native goal → 기존 send_action
Follower actual raw/native → Follower PrecisionMapper → static Sim state / tracking record
```

Real/Sim이 STOPPED일 때 실제 follower measurement를 변환해 Sim FK에 반영한다.
RUNNING에서는 같은 common target을 follower와 Sim에 전달하되 실제 plant response가 같다고 가정하지 않는다.
`snapshot_data.tracking`에 Leader command, Follower command/measured, Sim measured, tracking error, timestamp를 별도 기록한다.
이 오차로 calibration zero를 조정하지 않는다. Capture state/history/camera-aligned q도 활성 mapping을 사용하고 Profile hash/evidence/native/raw를 보존한다.
새 mapping은 `so101.precision-joint-mapping/1`; 기존 capture와 mapping은 그대로 읽는다. mapping_verified는 False다.

Scene의 `simulation_joint_offsets_rad`와 Precision을 중복 적용하지 않는다. 예전 Scene offset을
제외하는 설정을 끈 경우 nonzero offset이 있으면 연결을 거절한다. URDF 관절 기준 사용 설정은:

- MuJoCo: session profile 사본에서 Sim offsets만 제외.
- Isaac: 기존 immutable workspace의 original robot USD를 session spawn 설정으로 선택.
- 원본 Workspace/export/profile 파일은 수정하지 않는다. execution flag를 Capture state에 기록한다.
- Real-only Capture는 Sim이 없으므로 이 conflict가 적용되지 않는다.

Scene base/camera/geometry의 추정 오차까지 해결하는 기능이 아니다.

### 전체 실행 경로의 공통 보정

Real2Sim Capture의 Real-only/STOPPED 상태는 모터 구동과 분리한다. URDF 범위 밖의 유한한
측정각도 reference encoder branch가 구별되면 그대로 화면 갱신/스냅샷 저장에 사용한다.
`precision_observation_only`, `joint_limit_warnings`에 범위 초과를 기록하며 raw/native/q를 자르지 않는다.
현재 구현에서 ±half-turn branch가 모호하면 여전히 캡처를 거절한다. 원본을 임의로 0°로 바꾸지 않는다.
Teleop Start/실행/역변환, Real/Sim의 실제 physics 목표는 기존 URDF/모터 범위 검사를 유지한다.
범위 밖 스냅샷이 저장되었다고 해서 해당 자세를 모터 목표나 physics 재생으로 허용한 것은 아니다.

LeRobot 파일은 기존 Homing/Range/Drive 설정과 SDK degree/percent 입출력에 사용한다.
Precision은 homed raw값과 사용자가 기록한 URDF 기준각의 관계를 계산한다. 기존 각도에
보정각을 두 번 더하지 않으며 LeRobot 파일/register를 새로 만들거나 덮어쓰지 않는다.
두 점은 영점·비율, 한 점은 기존 비율에 영점을 맞춘다. LeRobot DEGREES는 고정 4095 counts/360°이므로
임의로 추정한 새 비율을 JSON의 Homing/Range만 변경해 정확히 표현할 수 없다.

- `hardware.Leader`/`Follower`는 Connect 때 승인된 active Profile을 자동 로드하고
  device/calibration/model/normalization을 확인한다. 없는 경우 기존 변환을 유지한다.
  잘못된 Profile은 자동 legacy fallback하지 않는다. Real/Sim은 자신의 선택 설정/저장 루트를 사용한다.
- standalone `SO101Leader.connect/advance`, Isaac Source/Teleop 및 MuJoCo Source도 같은 보정을 사용한다.
  실제 motor 입력에만 보정을 적용하며 이미 URDF rad인 Sim action/Replay/IK에는 재적용하지 않는다.
- 새 Precision Source는 원본 URDF 관절 frame을 사용한다. Isaac은 original USD,
  MuJoCo는 run-local MJCF에서 예전 추정 joint offsets를 제외한다. immutable Workspace는 보존한다.
- Source의 좌표 계약에는 실제 수집 변환 `hardware_mapping`과 `simulation_joint_frame=urdf`를 추가한다.
  `mapping`은 기존 nominal model-coordinate 계약을 보존하고 실제 device 변환은 별도 기록한다.
  MuJoCo는 project_provenance.json의 coordinate_contract에도 저장한다.
- Annotation/Datagen/Convert/Train/Sim Eval은 이 수집 출처와 Sim frame을 이어받는다.
  현재 active 값을 과거 데이터에 덮어씌우지 않는다. 새 Precision 데이터는 원본 USD/MJCF/IK URDF,
  과거 데이터는 당시 Scene frame을 재사용한다. Source resume은 보정 변경 시 서로 섞지 못한다.
- real_eval은 현재 Follower 보정으로 측정값→model rad와 policy rad→motor goal을 한 번씩 변환한다.
  활성화 전 현재 자세의 왕복 변환을 확인하며 기존 명시적 enable_motion 요구와 안전 제어를 유지한다.
  기존 checkpoint의 model-radian/robot/workspace/camera 계약 검사는 그대로 유지한다.
  예전 nonzero Scene offset 기준으로 학습한 policy를 현재 URDF 보정 실물에 바로 적용하는 것은
  모터 활성화 전에 거절한다. 새 기준의 Source로 재학습하거나 명시적인 별도 migration이 필요하다.

기본 Profile 경로는 `outputs/real2sim/joint_calibration`이다. 기존 클래스의 선택적
precision_root/model_hash 및 SO101LeaderCfg의 precision_profile_root/urdf_path로 실행 경계를 지정할 수 있다.
별도 calibration launcher/새 package/전역 DEFAULT_MAPPER 변경은 하지 않았다.
Calibration worker가 열려 있으면 UI Source/real_eval 실행을 차단한다. 실제 calibration serial fd는 Linux TIOCEXCL로 추가 open을 제한한다.

## 검증과 실제 하드웨어의 한계

Fake Motor + 기존 worker subprocess + offscreen PyQt로 reference/fit/validate/save/activate/restore 전체 절차를 실행했다.
실제 MuJoCo EGL에서 current/reference preview 생성 및 no-goal/no-Isaac-import를 확인했다.
tests/real2sim/test_precision_calibration.py, test_precision_ui.py와 기존 UI/Runtime/Joint/Launcher/Snapshot/Workflow 관련 회귀를 실행했다.
최종 수치와 명령은 `outputs/workflow_validation/precision_calibration/REPORT.md` 및 regression.log 참조.

실제 serial bus·Leader/Follower·지그/각도계·torque release/readback·TIOCEXCL·실물 Teleop은 **HARDWARE_NOT_TESTED**다.
실물 gain/sign/zero 정합, 실제 gripper geometry/opening correspondence, encoder boundary 동작,
안전한 정지/해제/접속·USB latency는 사용자가 직접 통제하는 별도 검증이 필요하다.
소프트웨어 단위/Fake 통과를 실제 물리 좌표 일치나 안전 인증이라고 주장하지 않는다.

## 반복 저장과 선택 관절 수정

같은 device/model/관절 기록·계수·검토 상태를 다시 저장하면 기존 파일을 재사용한다.
같은 활성 파일을 다시 적용해도 active.json의 이전 이력을 추가로 쌓지 않는다.
실제로 기준 기록이나 계산값이 바뀌면 이전 원본을 보존하고 새로운 버전 하나를 만든다.
선택 관절만 수정하면 다른 관절의 저장 보정은 유지된다. 기존 사용자 파일을 자동 삭제하지 않는다.
