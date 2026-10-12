# Snapshot Capture

2026-10-08. 기존 UI의 **현재 상태 캡처** 한 번으로 원본 Side/Wrist PNG, Follower 실측
관절값, 카메라별 timestamp/정렬된 q, calibration/model/hash를 자동 저장합니다.
Record/Stop Recording/Save/Event/Inspect 화면은 제거했습니다.
Teleoperation Start/Stop, 장치 연결, Profile 백엔드와 Workspace Pipeline은 유지합니다.

```bash
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
```

1. UI는 처음에 대기 상태이며 카메라/로봇/Sim을 자동 연결하지 않습니다. 오른쪽의 공통 **연결 설정** 패널에서
   Profile, 실제 Leader/Follower 포트와 calibration, follower ID, 서로 다른 Side/Wrist
   카메라를 설정합니다. Workspace Pipeline의 실행 Python은 기존 lerobot-arena를 선택합니다.
2. **Capture 탭의 Start · 카메라/로봇 연결**을 누릅니다. 이 버튼은 항상 real-only이며
   Sim을 실행하지 않습니다. 장치 연결과 로봇 추종을 분리하여 새 목표를 보내지 않습니다.
   같은 탭에서 Start Teleop을 누르면 기존 조작을 시작합니다.
3. 두 REAL 화면을 확인합니다. 오른쪽 Snapshot 저장 설정에서 이름/폴더를 선택하고 원하는 자세로
   이동한 뒤 약 0.6초 이상 정지하고 **현재 상태 캡처**를 누릅니다.
4. Saved/경로/개수/품질 경고를 확인하고 다른 자세에서 반복합니다. 실패는 개수에
   포함되지 않습니다. 물체/base 배치는 유지하며 변경 시 Scene state를 바꿉니다.
5. **한 번의 Start부터 Disconnect까지가 한 세션**입니다. 같은 세션의 캡처는 같은 폴더에
   추가됩니다. Stop Teleop/탭 변경은 세션을 끝내지 않습니다. 처음 8–15개 다양한 자세를
   권장하지만 성공 보장은 아닙니다. 독립 자세를 validation으로 선택합니다.
   저장 위치는 세션들을 담는 상위 폴더입니다. 첫 캡처 후 저장 위치/이름은 고정되고,
   바꾸려면 Disconnect 후 Start합니다. 기존 세션에 자동으로 이어 저장하지 않습니다.
6. Builder에서 아래처럼 import합니다. 실제 경로/ID로 교체하세요.

```bash
cd /home/hyeonhee/soarm_isaaclab/real2sim-builder
python -m app.cli new 'SO101 snapshots'
PROJECT='projects/so101-snapshots-REPLACE_ID'
python -m app.cli import-snapshots "$PROJECT" /absolute/path/to/snapshot_20261008_213000_ID \
  --scene-state static_001 --validation snapshot_000009 snapshot_000012
python -m app.cli inspect "$PROJECT"
```

환경별 실행 경로는 Builder `docs/LOCAL_RUN.md`, 데이터 계약/보정/Codex 명령/전체
export 절차는 [MULTIPOSE_CAPTURE.md](/home/hyeonhee/soarm_isaaclab/real2sim-builder/docs/MULTIPOSE_CAPTURE.md)를 보세요.

`real2sim.snapshot/1`: 공통 capture_manifest.yaml / robot_model / calibration과
snapshots/snapshot_000001/{side.png,wrist.png,state.json,metadata.json,COMPLETE}를 저장합니다.
폴더는 Start 시각으로 이름을 정하며 첫 캡처 때 실제 생성합니다. 공통 robot model과
보정은 세션 안에 한 번만 저장합니다. 각 샷에는 원본 이미지와 해당 순간의 상태/메타데이터만
저장합니다. manifest의 `session`에 연결 시작/종료 Unix ns가 기록됩니다.

```text
outputs/real2sim/snapshots/
  snapshot_20261008_213000_a1b2c3d4/
    capture_manifest.yaml
    calibration/{cameras,sensors}.json
    robot_model/robot.urdf + meshes
    snapshots/
      snapshot_000001/{side,wrist}.png + state.json + metadata.json + COMPLETE
      snapshot_000002/...
    COMPLETE
    .capture.lock
  snapshot_20261008_220000_e5f6a7b8/    # 다음 Start 후 새 세션
```

기존 `snapshot_dataset_*` 데이터는 이동하거나 수정하지 않으며 Builder에서 계속 읽습니다.
명시적 도구용 `SnapshotDataset.resume` 호환 API는 유지하지만 마감된 새 세션에는 쓸 수
없습니다. Disconnect는 이미 확보한 샷의 background 저장 뒤에 세션 마감을 예약하며
저장 완료를 기다리느라 장치 정지/닫기를 막지 않습니다.
원자적 manifest index에 있는 완료된 snapshot만 유효합니다. 카메라별 host read interval을
보존하며 exposure timestamp를 주장하지 않습니다. stale/skew/missing/moving은 거부하고
blur/textureless는 저장 경고와 fitting 제외로 처리합니다. measured q와 command는 분리합니다.

물리 관절 대응은 아직 legacy_unverified입니다. 이것을 실제 검증 없이 calibrated로
바꾸면 안 됩니다. 실제 USB/Leader/Follower 통합은 NOT_TESTED입니다. Capture는 움직임을
명령하지 않으며 탭 전환은 장치를 재연결하지 않습니다. Sim 없이 실측 수집이 가능하고,
Real/Sim 탭은 상단에서 Isaac/MuJoCo를 선택한 뒤 Connect하여 4-view를 사용합니다.
모드 변경은 이전 runtime을 종료한 뒤 연결하며 장치를 중복 연결하지 않습니다. 오래된
BUILDING_ENVIRONMENT 상태는 선택을 잠그지 않으며, 실제 장치가 없는 engine 초기화만
검증한 프로세스 경로에서 취소할 수 있습니다.

**Builder package 직접 Real/Sim 로딩은 아직 미지원**입니다. 이 UI는 Profile/Workspace
계약을 사용합니다. Builder XML/USD는 기하와 카메라를 export하지만 실제 제어 가능한
robot/task로 연결하는 downstream adapter는 별도입니다.

옛 continuous/paired recorder, 저장·읽기·변환 API와 내부 Reference/보정/API agent는
제거했습니다. 현재 UI 수집 경로는 `SnapshotDataset` 하나입니다. 독립 Builder는 별도
프로젝트로 유지하며 Snapshot import/보정/render/export를 수행합니다.
학습 Source/Annotation/Datagen pipeline은 이 실측 Snapshot 수집과 별개로 유지합니다.
