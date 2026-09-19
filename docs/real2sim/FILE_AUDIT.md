# 기존 구현 분석 및 정리 결과

대상 프로젝트 commit: `24c7ff9637c24434cf6111270abf1ef9b6494ca4`.
원격 HEAD도 동일함을 확인했다. 작업 사본에는 이전 대화에서 추가한 V2/V3/desktop 파일이 있었으며 원격 기본 코드와 구분했다.
참고 구현 commit: `736da5c2c6da040a653e581b8f8052ae5559ad51`.

원본을 삭제하기 전에 분류와 이유를 제시했다. 이전 생성 소스는 작업 폴더 밖
`real2sim-legacy-preserved.zip`에 보존한 다음 새 source tree를 구성했다.
사용자 로컬에만 있는 변경은 이 보관본에 들어 있지 않다. 로컬 적용 시 기존 폴더를 먼저 별도 보관해야 한다.

| 기존 경로 (`scripts/real2sim/` 기준) | 판정 | 이유 / 새 위치 |
|---|---|---|
| `teleop_reference.py` | REPLACE | `real2sim_sim.py` 단일 runtime entry point |
| `render_reference.py` | MERGE | UI의 preview, runtime의 measured-pose replay |
| `compare.py` | MERGE | `calibration.image_metrics/evaluate`, UI에서 실행 |
| `calibrate_camera.py` | MERGE | `calibration.calibrate_intrinsics`: 보드 검출 + held-out 검사 |
| `align_workspace.py` | MERGE | workspace landmark SE(3) 최적화 |
| `fit_reference.py` | REPLACE | 사진의 임의 초기값 대신 실측 profile + 한 그룹씩 최적화 |
| `inspect_projection.py` | MERGE | raw 사진 landmark 지정 + 재투영 평가 |
| `reference_scene.json` | REPLACE | 출처가 불명확한 provisional 값 사용 중단, 명시적 synthetic 측정 템플릿 제공 |
| `preflight.py` | MERGE | Connect 시 실제 버전·소스·보정·관절·카메라 검증 |
| `check_offline.py`, `check_v3.py`, `check_usd.py` | REPLACE | `tests/real2sim/` regression suite |
| `v3/core.py` | KEEP/MERGE | 검증된 단위 변환·timestamp·joint error·stability를 내부 core에 보존 |
| `v3/cameras.py` | KEEP/MERGE | LeRobot 카메라 reader와 수신시각 기록을 내부 cameras에 보존 |
| `v3/hardware.py`, `api_manifest.json` | MERGE | 실제 검사한 API 보존. 자동 calibration 금지, Connect와 Start 분리 |
| `v3/capture.py` | REPLACE | atomic session reservation + COMPLETE commit marker |
| `v3/sim.py` | MERGE | runtime snapshot: camera matrices, 모든 body FK, measured joints |
| `v3/comparison.py` | REPLACE | 왜곡 보정된 appearance + raw landmark geometric metric |
| `v3/runtime.py` | REPLACE | UI command 상태기계, 비동기 저장, replay/update/export |
| `v3/viewer.py` | REMOVE | HTTP/browser server 제거, PyQt 네 타일 사용 |
| `scripts/real2sim_desktop/*` | REMOVE/MERGE | V3 복사본 제거. 필요한 PyQt/IPC 기능을 단일 구현으로 대체 |
| `source/.../real2sim/frames.py` | MERGE | profile의 엄격한 SE(3) 검증과 변환 |
| `source/.../real2sim/geometry.py` | REPLACE | 실측 primitive USD 생성기. 컵은 바닥 + 개방 벽체 |
| `source/.../real2sim/reset.py` | REMOVE | Real2Sim 보정 중 random reset 금지. 원래 task reset 코드는 그대로 사용 |
| `source/.../real2sim/scene.py` | REPLACE | opt-in 환경 wrapper + USD builder + stopped scene update |
| `docs/real2sim/README_V3.md`, `regions_template.json` | REPLACE | 새 README, 측정/landmark 계약과 테스트 안내 |
| `*.before_*`, `__pycache__` | REMOVE | 현재 전달 소스에 포함하지 않음. 검사한 작업 사본에 before 파일은 없었음 |
| 기존 `assets`, `devices/lerobot`, `tasks`, teleop/Mimic/datagen/policy | KEEP | API 참조만 했으며 재작성하지 않음 |

최종 내부 책임:

| 모듈 | 책임 |
|---|---|
| `profile.py` | 좌표/단위/schema/측정값 및 revision identity |
| `storage.py` | 파일 IPC, 세션, atomic capture와 revision |
| `core.py` | 관절 단위, 시간·안정성·일치 검사 |
| `hardware.py` + manifest | 검사한 leader/follower API adapter |
| `cameras.py` | 실물 카메라 읽기 |
| `scene.py` | static USD와 기존 Isaac 환경 wrapper |
| `calibration.py` | 보드 내부값, bounded SE(3) 최적화, 정량 metric |
| `agent.py` | 제한된 GPT 결정·appearance와 반복 실행 |
| `runtime.py` | Isaac·장비 소유 프로세스 |
| `ui.py` | PyQt 설정·viewer·측정 입력·작업 실행 |

# GPT6-real2sim에서 가져온 것과 바꾼 것

`real2sim_pipeline.md`, `real2sim_ep0/refine_calibration.py`, `real2sim_ep2/fit_scene.py`,
`real2sim_microphones/calibrate.py`를 분석했다. 참고 저장소 자체도 semi-manual workflow이며
자동 end-to-end reconstruction이나 정확한 물성 복원을 입증한 구현이 아니라고 설명한다.

| 참고 방식 | SO-101 대응 |
|---|---|
| 다중 시점·joint state·공통 좌표계 | paired capture metadata + 동일 measured pose replay |
| scipy robust least-squares | 측정 landmark reprojection 최적화 + capture 단위 holdout |
| `mj_forward`, MuJoCo body transform | Isaac articulation body pose / FK readback |
| MJCF primitive / mesh scene | USD primitive / 개방 container collision geometry |
| MuJoCo joint state replay | `write_joint_state_to_sim` 후 Isaac renderer로 재촬영 |
| Blender 렌더링 경로 | 기존 Isaac Lab Camera sensor |
| agent의 mismatch 해석과 다음 trial 선택 | structured action/group/reason → 코드 optimizer → accept/reject → render |
| mass/friction identification | 이번 범위에서 제외 |

# API 분석 근거

- Repository leader: `devices/lerobot/so101_leader.py`의 six motor names, 팔 DEGREES,
  gripper RANGE_0_100, `advance()`가 6개 모두 pi/180 변환하는 기존 인터페이스를 확인.
  Gripper 결과를 실제 턱 각도라고 해석하지 않는다.
- Leader 생성자의 파일 누락 시 자동 `calibrate()` 경로를 override하여 항상 오류 처리.
- LeRobot 참조 소스: v0.4.1의 SO101Follower가 SO100Follower 동작을 상속.
  `connect(calibrate=True)` 기본값에 의존하지 않고 bus 연결 → read-only calibration 검사.
  Start에서 현재 위치를 Goal_Position에 넣고 readback 확인 후 기존 configure 사용.
- 실제 사용자의 설치 환경은 이 컨테이너에 없다. 설치 버전 및 검사 대상 소스의 AST hash를
  Connect 시 확인하며 다르면 연결을 차단한다. 임의 버전 업데이트나 mismatch 무시는 하지 않는다.
- Isaac Lab API 기준: 2.3.0. 실제 GPU 실행 검증은 하지 못했다.

공식/원본 자료:

- https://github.com/HYEONHEE5739/so101-isaaclab
- https://github.com/lingxiao-guo/GPT6-real2sim/blob/main/real2sim_pipeline.md
- https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sensors/camera/camera.html
- https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/images-vision
