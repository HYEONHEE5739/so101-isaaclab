> 과거 설계·실험 기록입니다. 현재 구조와 실행 경로는 [ARCHITECTURE](../../real2sim/ARCHITECTURE.md)와 [DATA_FLOW](../../real2sim/DATA_FLOW.md)를 따릅니다. 본문의 단계·제한·명령은 작성 당시 기준입니다.

# Phase 2: SO-101 독립 MuJoCo 최소 환경

2026-10-06 구현/검증. Phase 2.5 VLA gate는 **미실행**이며 UI 연결 조건은 아직 충족하지 않았다.
기존 Isaac 실행 코드/UI/teleoperation/annotation/datagen은 이번 작업에서 수정하지 않았다.

## 구현 결과

- 6개 SO-101 model-radian position target, CPU physics, env count=1.
- URDF link/joint origins, axes, inertias, limits, STL 13개 재사용.
- 기존 workspace profile의 robot base/table/cube/target/camera transform 재사용.
- Table/cube/support는 box, bin은 primitive cylinder 바닥+box 벽, 열린 top.
- side RGB camera 1개, renderer 지연 생성, EGL/OSMesa/software EGL을 별도 worker로 선택.
- reset(seed)/step, measured joint state, commanded target, workspace EEF/object pose,
  RGB uint8, contained-and-settled 최소 성공 조건, MuJoCo state 저장/복원.
- MuJoCo worker에서 `isaaclab`, `omni`, `pxr`가 import되지 않는 것을 clean subprocess로 검증.
- Blender 사용 없음. 입력 workspace나 URDF/STL/USD 파일을 수정하지 않는다.

## 파일과 데이터 흐름

| 파일 | 역할 |
|---|---|
| `source/soarm101_lab/soarm101_lab/simulation/base.py` | engine-independent 단일 env 계약 |
| `simulation/mjcf.py` (동일 패키지) | 명시적으로 선택한 Workspace/Profile → run-local robot/scene MJCF |
| `simulation/mujoco_backend.py` | CPU physics, joint/body/site API, renderer, state restore |
| `scripts/mujoco/check_env.py` | 독립 worker smoke, placement probe, RAM/GPU 스냅샷/FPS 보고서 |
| `scripts/mujoco/requirements.txt` | 독립 worker dependency. MuJoCo 3.3.7 고정 |
| `tests/real2sim/test_mujoco_backend.py` | 독립 FK, 물리/성공/복원/좌표/camera/import 검사 |

```text
기존 package.load → profile/task/URDF/STL (읽기 전용)
  → run/scene/so101.xml + scene.xml + scene_metadata.json
  → MjModel/MjData (CPU)
  → joint/state/poses + optional side RGB
  → run/report.json + rgb.png
```

실제 package source hash, URDF/mesh SHA256, joint order/limits, camera K,
home/task 설정, timestep/decimation, 물성/형상 근사와 출처를 metadata에 남긴다.
고정된 repository robot XML 대신 workspace마다 base transform을 반영한 `so101.xml`을
실행 출력 폴더에 생성한다. scene과 robot XML은 분리되어 있다.

## 설치와 실행

프로젝트 루트에서 실행한다. 이번 작업에서 `.venv-mujoco`를 생성하고 의존성을 설치했다.
기존 LeRobot/Isaac Python 환경은 변경하지 않았다. 다른 머신에서 처음 설치할 경우:

```bash
python3 -m venv .venv-mujoco
.venv-mujoco/bin/python -m pip install -r scripts/mujoco/requirements.txt pytest
```

Physics only (OpenGL context 생성 없음):

```bash
.venv-mujoco/bin/python scripts/mujoco/check_env.py \
  --workspace outputs/real2sim/environments/workspace_001/v2 \
  --renderer none --output outputs/workflow_validation/mujoco_my_physics_run
```

실제 GPU EGL:

```bash
.venv-mujoco/bin/python scripts/mujoco/check_env.py \
  --workspace outputs/real2sim/environments/workspace_001/v2 \
  --renderer egl --width 320 --height 240 \
  --output outputs/workflow_validation/mujoco_my_egl_run
```

CPU software EGL (Mesa llvmpipe 지원 시스템):

```bash
.venv-mujoco/bin/python scripts/mujoco/check_env.py \
  --workspace outputs/real2sim/environments/workspace_001/v2 \
  --renderer software-egl --width 320 --height 240 \
  --output outputs/workflow_validation/mujoco_my_cpu_render_run
```

OSMesa를 사용할 경우 위 명령의 renderer를 `osmesa`로 변경한다.
시스템의 `libosmesa6` 공유 라이브러리가 필요하다. 이번 머신에는 해당 라이브러리가
없어 실패했으며 시스템 package를 설치하지 않았다. software EGL/llvmpipe는 실제 통과했다.
각 모드는 **별도 프로세스**에서 실행한다. 이미 MuJoCo를 import한 프로세스에서 renderer를
바꾸면 실패한다. CPU rendering을 요청했는데 software renderer로 확인되지 않으면 거절한다.

`--workspace`는 필수다. 옛 bootstrap scene 기본 입력은 제거했다.
`--output`은 새 폴더를 지정해야 한다. 기존 출력과 published workspace 내부 쓰기는 거절한다.
기본 검증은 control step 150회와 camera frame 30회이고, `--steps`, `--frames`, `--seed`로 조정한다.

## 확인할 화면과 수치

출력 폴더의 `rgb.png`: 기존 side pose에서 table, red cube, 열린 target cup,
SO-101 yellow/black visual이 보여야 한다. v2의 기존 side camera는 작업 영역 중심이어서
robot 일부가 오른쪽에 잘릴 수 있다. camera pose를 보기 좋게 임의 변경하지 않았다.
이 worker는 창을 띄우지 않는다. 독립 offscreen 이미지와 JSON으로 검증한다.

`report.json`:

- `status=PASSED`, `forbidden_imports=[]`, `reset_restore_verified=true`.
- `measured_joint_rad`/`commanded_joint_rad`는 각각 6개; 단위 rad.
- `initial_eef_pose_workspace`/`initial_object_pose_workspace`는 4×4.
- 시간은 기본 150 steps × .02s = 3s; physics timestep=.002s, decimation=10.
- 초기 cube는 success=false. cube를 bin 안에 놓고 안정화한 `placement_probe.success=true`.
  이 probe는 manipulation 성공 데모가 아니라 최소 성공 판정/충돌 검증이다.
- `render_context.vendor/renderer`로 실제 NVIDIA 또는 llvmpipe를 확인한다.
  renderer=`egl`만으로 GPU rendering 성공을 판단하지 않는다.
- `camera_render_fps`는 render+readback만 포함한다. physics/PNG 저장/policy 시간을 제외한다.
- `samples`의 RSS와 GPU 값은 단계별 스냅샷. GPU 값은 전체 device 점유량이고
  process VRAM이나 CUDA allocator/전체 실행 peak가 아니다.
- `phase25_gate=NOT_RUN`: Qwen/Action Head forward를 아직 수행하지 않았다.

## 이번 실행 결과

독립 환경 Python 3.10, MuJoCo 3.3.7, 320×240, 30 camera frames.
실제 GPU: NVIDIA GeForce RTX 4060 Laptop GPU, device total 8,188 MiB.

| 실행 | 결과 | RAM RSS | GPU memory | camera render FPS |
|---|---|---|---|---|
| provisional physics only | PASSED | 로드 약 424 MiB | sandbox에서는 측정 불가 | 해당 없음 |
| workspace v2 physics only (실제 장치 접근) | PASSED | 물리 약 424 MiB | 1,912→1,923 MiB (전체 device) | 해당 없음 |
| workspace v2 software EGL / llvmpipe | PASSED | 물리 약 458 MiB, 렌더 약 600 MiB | sandbox에서는 측정 불가 | 약 59.8 |
| workspace v2 NVIDIA EGL | PASSED | 물리 약 459 MiB, 렌더 약 562 MiB | 물리 1,906 MiB, 렌더 1,941 MiB (전체 device) | 약 3,537 |
| OSMesa | FAILED / 미지원 의존성 | 해당 없음 | 해당 없음 | `libOSMesa` 없음 |

NVIDIA EGL의 render-only median latency는 약 .235ms였다. 매우 작은 정적 장면의 짧은
측정으로, 실시간 VLA loop의 처리량을 뜻하지 않는다. 관측된 renderer 활성화 전후 GPU
점유 차이는 약 35 MiB이며, 모델 load/forward peak를 보증하지 않는다. 다른 GPU process의
점유 변화도 영향을 주므로 Phase 2.5에서 process/device/allocator 수치를 함께 측정해야 한다.
GPU 검사에는 sandbox 밖 device 접근이 필요했고 승인된 worker로 실행했다.
별도의 physics-only worker는 GL context를 만들지 않았지만 전체 GPU snapshot은 11 MiB
변동했다. 이를 MuJoCo physics의 GPU allocation으로 해석하지 않는다. 비교 실행 간에도
background GPU 점유량이 달라져 snapshot을 engine별 고정 VRAM 비용으로 사용할 수 없다.

raw 결과:

- `outputs/mujoco/phase2_physics/report.json`
- `outputs/mujoco/phase2_nvidia_physics/report.json`
- `outputs/mujoco/phase2_software_egl/report.json`, `rgb.png`
- `outputs/mujoco/phase2_nvidia_egl/report.json`, `rgb.png`
- `outputs/mujoco/phase2_osmesa/report.json` (설치되지 않은 library로 인한 실패 증거)

NVIDIA v2 physics 150 step 후 최대 joint target 오차 약 .00436 rad,
cube speed 약 3.76e-16 m/s. 모든 state는 finite였고 initial state 복원이 통과했다.

## 테스트

```bash
.venv-mujoco/bin/python -m pytest -q tests/real2sim/test_mujoco_backend.py
```

최종 실행 결과: **7 passed in 3.44s**. 7개 integration tests:

1. MuJoCo model compile(6 actuators, 1 camera, robot 6 DoF + object free joint),
   home+20 random poses의 독립 URDF FK와 EEF 4×4 오차 <1e-9, camera ROS→GL 변환.
2. 관절 이동/3s finite physics, cube settling, 동일 seed reset, state restore,
   잘못된 이름/shape/NaN/limit/state 계약 거절.
3. 열린 bin에 넣은 cube의 settling 성공과 speed/wall/height 실패 판정.
4. 잘못된 transform/resolution과 output overwrite 거절.
5. 회전된 workspace frame에서 object pose set/get 및 world 변환 검사.
6. clean subprocess에서 llvmpipe RGB 생성과 red cube centroid의 pinhole projection 오차 <4px.
7. clean worker의 forbidden import 부재와 엔진 import 후 rendering mode 변경 거절.

기존 Isaac 경로는 수정하지 않았고 Isaac runtime을 실행하지 않았다.

## 제한과 다음 수정 위치

Phase 2는 선택된 cube 1개와 target 1개, table/support, robot, side camera만 포함한다.
다른 cube/target/background/texture/Blender asset은 제외하고 metadata에 명시했다.
RGB 근사이며 photometric calibration 동일성을 주장하지 않는다.

URDF collision mesh는 MuJoCo convex hull이며 robot self-collision은 disabled다.
gripper grasp/contact 품질과 실제 teleop는 아직 검증하지 않았다. 손가락 collision은
Phase 4 이전에 검토해야 한다. actuator gain은 기존 코드 기반 configured 값,
force limit=10은 URDF 값이고 검증된 실제 motor torque가 아니다.

성공 조건은 eight corners containment + 순간 선속도이며 지속 시간/각속도 기반
안정 grasp 조건은 아니다. 추가 물리 검증은 `mujoco_backend.success_details`와
`mjcf.robot_xml`/`build_scene`에서 수행한다. 본 모델의 frame/state 계약은 version 1이다.

다음 단계는 `DUAL_BACKEND_PLAN.md`의 **Phase 2.5 VLA VRAM Feasibility Gate**.
Qwen3-VL-2B 4-bit + 실제 RGB + `[B,N,2048]` hidden features + 작은 head의 동시 forward를
실제 8GB GPU에서 검증한 후에만 UI 연결로 진행한다.

API 근거: [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html),
[MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html).

현재 독립 check_env의 기본 출력은 `outputs/workflow_validation/mujoco_phase2_<날짜>/`다.
위 검증 결과 목록은 당시 실행 기록이며 최신 파일 존재 여부를 뜻하지 않는다.
