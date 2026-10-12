# Real/Sim Pair 관측 → 기존 Builder 개선

2026-10-10. UI는 관측 저장만 한다. Calibration/Geometry/Candidate 개선은 독립 Builder와 외부 Codex 책임이다.

## 수집

1. Pipeline에서 Builder Workspace를 선택하고 Real/Sim의 Pipeline Workspace 사용을 켠다.
2. Isaac 또는 MuJoCo를 선택하고 실제 Follower·Side/Wrist 카메라와 Sim을 Connect한다.
3. 기존 Teleop으로 자세를 바꾸고 안정될 때 **현재 상태 Pair 캡처**를 누른다.
4. 경로/경고는 버튼 아래와 로그에 표시된다. 저장 위치/Scene state는 Capture 탭의 저장 설정을 공유한다.
5. 물체 배치와 base를 고정한다. 변경 시 Scene state를 바꾼다. 물체 이동 이후 자료를 동일 정적 scene fitting에 합치지 않는다.

버튼은 새로운 장치 read/send/step 명령을 실행하지 않는다. 마지막 완전한 runtime 관측의 원본 RGB와
상태를 복사하여 기존 background writer에 전달한다. UI resize/letterbox/JPEG preview를 저장하지 않는다.
Builder workspace checksum 검증은 Connect 때 수행하고 context를 캐시하여 클릭마다 큰 asset을 해시하지 않는다.

## 재사용한 계약

새 독립 recorder 없이 `SnapshotDataset.save(..., paired=True)`와 기존 real2sim.snapshot/1을 확장한다.
Start→Disconnect 세션, 공통 calibration/robot_model, atomic index/COMPLETE/hash를 그대로 사용한다.

```text
snapshot_<시각>_<id>/
  capture_manifest.yaml
  calibration/ robot_model/       # 세션당 한 번
  snapshots/snapshot_000001/
    side.png wrist.png            # Real RGB
    sim_side.png sim_wrist.png     # 역사적 runtime RGB
    state.json metadata.json pair.json COMPLETE
```

pair.json schema real2sim.paired-observation/1: engine, workspace ID/version/content hash,
Builder manifest/scene hash, profile, runtime camera K/optical world transform/resolution,
actual sim q, measured feedback와의 rad 차이, sim render host interval, real camera metadata,
mapping, quality. Real q_native/q_measured/q_command와 측정 history는 기존 state.json에 별도로 보존한다.
그리퍼 rad는 기존 mapper의 제어 좌표이며 실제 jaw angle 측정이라고 주장하지 않는다.

기존 stale/moving/missing/해상도 불일치는 저장 실패로 보고한다.
Pair q 차이 >0.02 rad 또는 100ms 시간 불일치/200ms stale 및 기존 runtime quality 실패는
경고와 함께 저장하지만 calibration에서 제외한다. 이 threshold는 보수적 heuristic이다.
Legacy unverified mapping, host-read timestamp/exposure 불확실성은 검증 완료로 승격하지 않는다.

## Builder 사용

```bash
cd ../real2sim-builder
PROJECT='projects/so101-scene1-snapshot-reconstruction-2202744f'
python -m app.cli import-paired "$PROJECT" /absolute/new/session \
  --scene-state static_001 --validation snapshot_000009 snapshot_000012
python -m app.cli inspect "$PROJECT"
python -m app.cli candidate "$PROJECT" create paired_refinement_01
# 기존 candidate edit/fit/blender-author/blender-sync로 한 parameter group씩 변경
python -m app.cli candidate "$PROJECT" evaluate paired_refinement_01 --engine blender --validation
python -m app.cli candidate "$PROJECT" evaluate paired_refinement_01 --engine mujoco --renderer software-egl --validation
# 개선·독립 holdout nonregression 확인 후만 promote
python -m app.cli candidate "$PROJECT" promote paired_refinement_01
python -m app.cli validate "$PROJECT" --engine blender
python -m app.cli export "$PROJECT"
```

정확한 fit evidence/schema는 Builder docs/AGENTIC_TOOLKIT.md와 MULTIPOSE_CAPTURE.md 참조.
Import는 기존 scene을 초기화하거나 runtime camera 값으로 canonical camera를 덮어쓰지 않는다.
기존 camera ID/parent/K/distortion/resolution 계약이 달라지면 명시적 reconciliation이 필요하다.
추정 runtime mount pose 차이는 보존한다. 동일/근접 자세의 train/validation 중복은 기존 importer가 거절한다.
Real-only와 Pair를 같은 세션에 혼합했다면 import-snapshots도 확장 metadata를 읽는다.
import-paired는 선택한 Scene state의 모든 관측이 Pair여야 한다.

## 비교와 진단

기존 FK는 새 Real observation의 측정 q로 Candidate를 pose하여 다시 렌더한다.
metrics.json의 images[].paired_diagnostics에 real_runtime와 candidate_runtime metric 및 overlay/difference를 추가한다.
기존 Real↔Candidate metric만 optimization/nonregression의 기준이다. 과거 Sim은 ground truth가 아니다.
Builder-render도 Real과 다르면 camera/geometry 가설을 조사한다. Builder는 맞는데 Runtime만 다르면
Adapter transform/FK/asset/renderer를 조사한다. MAE만으로 인과를 자동 판정하지 않는다.
정합 불량은 fitting/validation에서 제외되며 과거 영상·metadata는 immutable 입력 사본으로 보존한다.

Manifest의 observations.runtime_baseline, dataset_provenance와 provenance/reconstruction.yaml에
Pair workspace/calibration/hash 출처가 이어진다. export의 기존 validation/history 계약은 유지한다.
개선 export는 새 workspace version으로 import한다. 옛 Workspace/HDF5/model identity를 변경하지 않는다.

## 검증 범위

Snapshot·UI·launcher 회귀 82 passed/1 skipped.
Builder core/export/candidate/multipose 30 passed/1 skipped(기존 USD/Qt 경로 사용).
실제 software-EGL Candidate rerender 및 train/heldout 비교 synthetic smoke 통과:
outputs/workflow_validation/paired_capture/smoke_report.json.
별도 Pair save/import/hash corruption/불일치 제외/무장치 명령 검사를 포함한다.
실제 USB/로봇/카메라 Pair 수집, Isaac Pair 렌더 및 실측 Scene 개선은 NOT_TESTED.
synthetic runtime baseline을 Real과 동일하게 만든 smoke는 pipeline 연결 검사이며 reconstruction 성공이 아니다.
