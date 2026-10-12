# Scene1 Real/Sim 보정 작업 인수인계

## 최신 — 2026-10-11 / v7 새 2 Pair → v8 Wrist K 보정

`outputs/workflow_validation/v7_wrist_camera_refinement/REPORT.md`를 먼저 읽는다.
새 `snapshot_20261011_222612_a9610eee` 2자세(v7) train, 이전6자세(v6) 회귀 비교.
새 Builder project `scene1-v7-wrist-camera-refinement-20261011-58823cd8`,
선택 `wrist_view_review`: Wrist K만 fx/fy[560.8899,558.3139], cx/cy[326.8775,262.6275].
원본 q/geometry/Workspace축/robot/base/joint zero/mount/Side/task/물성은 그대로다.
새 paper IoU .8290→.9459, .8207→.8938; 이전6도 모두 개선.
영상전체 MAE 및 일부 center 회귀로 공식 nonregression FAIL, OPEN/미 promote/UNVALIDATED.
그리퍼 벌어짐은 여전히 다르다. 영점 probe는 닫힌 자세 개선/열린 자세 악화로 적용하지 않았다.
pose+focal 대안은 실제 camera body 내부 가림으로 제외. 모든 대안은 REJECTED로 닫았다.
review XML/USD export와 sibling 소비 Workspace `outputs/real2sim/environments/scene1_builder/v8` 생성.
MuJoCo 16RGB/raw-q/FK/camera/reset/200step, Blender Wrist3렌더, export USD load/구조 확인.
Isaac v8 runtime/contact/실물 calibration/독립 blind validation 미검증. 장치 접근 없음.
새 실험의 disposable Blender payload/mask 약1.11GB만 제거; source/Scene/PNG/metrics 유지.
기존 v7/원본/canonical/evidence 보존, production/dependency 변경 없음.
UI 선택은 자동 변경하지 않았다. 사용자가 v8 선택 후 직접 재연결하고 Pair version/hash 확인.
상세 실패 로그·보존·실험 recipes/·다음 finger mapping 검증은 위 REPORT.md 참조.


## 최신 — 2026-10-11 / v6 새 6 Pair → v7 Wrist 작업공간 검토 후보

`outputs/workflow_validation/v6_wrist_workspace_refinement/REPORT.md`를 먼저 읽는다.
새 `snapshot_20261011_145450_d2cf22b8`는 전부 v6이며 6자세/12이미지 모두 사용했다.
새 Builder `scene1-v6-wrist-workspace-refinement-20261011-b3a03339`의
`paper_pose_guarded`에서 종이/테두리만 XY[-3.380,+1.292]mm, yaw+1.448° 수정했다.
원본 q, robot/base/zero, camera K/mount, 컵·블록·table의 world pose/치수/물성 유지.
Wrist 종이-only 진단 IoU는 6자세 모두 개선했으나 Side 일부 mask와 RGB 회귀가 남아
공식 nonregression FAIL, **OPEN / 미 promote / UNVALIDATED**다.
자세5/6은 fitting 제외 비교지만 같은 session이며 독립 validation 인증이 아니다.
`paper_pose_wrist_priority`는 Side 회귀로 REJECTED. camera-only 대안은 적용하지 않았다.
review XML/USD export와 `outputs/real2sim/environments/scene1_builder/v7` 생성.
MuJoCo 12RGB/raw-q/EEF/camera/reset/200step, Blender 두 자세4렌더 완료.
v7 Isaac runtime/실물 contact/독립 session validation 미검증. actual hardware 접근 없음.
Adapter의 Workspace 축은 수정한 support yaw를 따르므로 runtime-local 좌표는 달라지지만
나머지 물체 world pose는 동일하다. Source Reset 영역은 새 종이를 따라 이동/회전한다.
원본/canonical/evidence/기존 v6/export 보존, production/dependency 변경 없음.
이번 실험의 disposable Blender payload/mask 약1.39GB만 정리했다. recipes/ 재현 가능.
UI는 자동 변경하지 않았다. 사용자가 v7 선택 후 직접 Disconnect→Connect하고
새 Pair version/hash를 확인한다. 상세 metrics/보존/다음 단계는 위 REPORT.md 참조.


## 최신 — 2026-10-11 / 새 v5 컵 주변 4 Pair 보정

우선 `outputs/workflow_validation/v5_new_capture_refinement/REPORT.md`와
`review_summary.json`, `workspace_smoke.json`, `final_preservation.json`을 읽는다.
새 원본 `snapshot_20261011_140108_47734fec` 4자세/8이미지 전부 실제 v5이며 모두 사용했다.
새 Builder project `../real2sim-builder/projects/scene1-v5-cup-approach-refinement-20261011-56afacbf`.
canonical/evidence는 seed v5/import 이후 불변. 보정은 `scratch/agent/approach_contact_review`.
카메라-only 대안 `wrist_mount_review`는 이전 정합 악화로 REJECTED다.

작은 Sim zero/base 후보는 새 컵 rim 추정 RMSE52.94→45.50px,
새 Side tip28.90→26.96px, 이전 Side/Wrist 평균 오차 개선.
개별 이전 추정 point2px 증가 허용; 새 Wrist RGB MAE .17206→.17332 악화.
최종 **NEEDS_MORE_DATA / NOT PROMOTED**이고 미확인 물성을 인증하지 않았다.
원본 q/물체/카메라 geometry는 잠금. 새 model tip의 월드 X 변위 약 +4mm는
네 자세의 계산값이지 물리적인 전 자세 calibration 인증이 아니다.
저장 q tracking 차이만으로 FK 끝단5.42–9.92mm 차이도 있으므로 분리해서 해석한다.

review export는 새 project의 `exports/...-approach_contact_review`,
UI 검토용 import는 `outputs/real2sim/environments/scene1_builder/v6`.
MuJoCo 8영상 raw-q/EEF/camera/reset/200step 실제 검사, Blender 두 자세 렌더 완료.
v6 Isaac runtime, 물리/실물 grasp/place, 독립 holdout는 미검증이다.
현재 UI runtime/선택을 자동 변경하지 않았다. 사용자는 직접 v6 선택/재연결할 수 있다.
기존 v1–v5, 이전 project/hash, Snapshot 전부 보존. dependency/production 코드 변경 없음.
새 generated Blender 반복 payload/mask 약1.39GB만 제거; result/PNG/Scene은 유지.
실험 재현 script는 REPORT.md의 recipes/; create recipe 중복 실행은 금지한다.
다음은 블록 중앙/컵 rim의 사용자 정지 Pair + 별도 validation session이며
실제 로봇은 자동 연결·정지·이동시키지 않는다.

## 최신 진단 — 2026-10-11 / 새 7 Pair의 실제 버전

`outputs/workflow_validation/v5_root_cause_20261011/REPORT.md`를 먼저 읽는다.
새 원본 `snapshot_20261011_121308_65bb7b1f`의 7쌍은 모두 **v1**이다.
version/content/scene hash/전체 runtime_profile/저장 optical 변환을 각각 대조했다.
v5를 선택한 캡처로 취급하지 않는다. 7쌍 모두 원본 이미지 확인과 camera-aligned measured-q
재렌더에 사용했다. v5 offset-only 비교에서 추정 Side 끝단 RMSE30.62→12.87px,
14개 결합 변수의 Side Jacobian rank10. 새 Wrist/object 대응 및 독립 holdout 부족으로
**NEEDS_MORE_DATA**, 추가 fitting/promote/export 없음. canonical/후보/evidence hash 불변.
Runtime v1/v5 각각14장, Builder offset 조건3개×14장 완료. v5 EEF/camera 행렬은
Builder↔Runtime 약1e-12로 일치하지만 실측 기구/물성/전체 영상 정합 인증은 아니다.
다음 사용자는 UI에서 v5를 선택한 뒤 직접 Disconnect→Connect하고 새 Pair identity부터
확인해야 한다. 에이전트는 현재 사용자 runtime을 자동 정지·연결하지 않는다.
구체적인 train/독립 validation 자세, 부족한 annotation, 다음 명령은 새 REPORT.md의 F/G.

## 최신 재개 — 2026-10-11 / Builder V2

최신 실행 가능한 검토용 Workspace는 **`outputs/real2sim/environments/scene1_builder/v5`**.
아래 v3 기록은 과거 시점이다. 사용자 요청으로 v2 컵 배치에서 새 project
`../real2sim-builder/projects/so101-scene1-v2-joint-refinement-224805-256692a7`를 만들었다.
12자세/24장 모두 사용하고 q는 보존, Sim zero-frame/base/Wrist/cube XY를 추정 보정했다.
RMSE31.41→15.95px. 독립 holdout이 없어 review-only export, 미 promote 상태다.
결과와 UI 조작은 `outputs/workflow_validation/builder_v2_scene_refinement/REPORT.md`.
같은 project의 `scratch/agent/geometry_review_export/`와 export를 읽는다.
v4는 원본 USD hash 충돌을 발견한 중간 실패판이므로 선택하지 않는다. v5는 원본 USD 보존/
별도 override layer로 실제 MuJoCo4자세/Isaac2자세 RGB/FK smoke를 통과했다.
실물/접촉 물성은 미검증이다. 추정값을 인증된 calibration으로 바꾸지 말고 새 candidate로 이어간다.


## 재개 후 최신 상태 — 2026-10-10

사용자가 보정 재개를 요청하여 아래 인계 시점 이후 실험을 추가했다. 현재 최신
Workspace는 **`outputs/real2sim/environments/scene1_builder/v3`**이다.
v2 컵 배치는 채택하지 않았으며 v3는 기존 baseline geometry의 표면 색상 개선이다.
v1/v2/export/원본 Snapshot은 보존했다. 아래 본문의 “추가 fitting 중단”,
“current_scene은 baseline”은 인계 당시 기록이다. 현재 canonical은 색상 후보가 promote된 상태다.

작업 project는 동일한 `so101-scene1-paired-layout-20261010-b9e9cc4d`이며:

- `wrist_mount_focal_silhouette`: train-only cube centroid/extent로 mount+focal 보정.
  광학 pose만 변경해 시점이 기존 카메라 외형에 가려졌다. REJECTED.
- `wrist_mount_focal_coupled`: 광학 pose와 `frames.wrist_camera_mount` 및 시각 mount arm을
  함께 변경해 가림 해결. 대응점 RMSE는 train38.96→20.01px, holdout23.98→23.29px지만
  holdout MAE0.13643→0.14542로 악화해 REJECTED. 이 카메라 보정은 runtime에 적용하지 않았다.
- `train_surface_colors`: train1–6의 eroded matched regions에서 linear color gain을
  median으로 추정하고65%만 반영. marker/table/robot_gripper_1/robot_jaw_0의 color만 변경.
  모든 geometry/K/pose/q/dimensions/physics 불변을 비교 검사했다.
  train MAE0.14690263→0.13589986, holdout0.13642631→0.12364413.
  holdout4장 모두 nonregression PASS 후 **PROMOTED**. 최종 geometry acceptance는 아니다.

GPU EGL끼리 비교한 수치다. 이전 software-EGL 수치와 baseline에 작은 차이가 있다.
holdout은 반복 실험에서 이미 확인했던 같은 세션의 자세이며 새로운 독립 검증이라 주장하지 않는다.
새 export는 project의 `exports/so101-scene1-paired-layout-20261010-b9e9cc4d`이고,
기존 `...-cups-review`와 별개다. 상태는 UNVALIDATED,
`VISUAL_APPEARANCE_IMPROVED_GEOMETRY_UNVERIFIED`, `PHYSICS_NOT_VALIDATED`다.
Blender는 `working/blender/surface_refinement/scene.blend`, shot3 두 view preview 생성.
XML/USD 구조 parity PASS. 실제 MuJoCo v3 smoke 결과/경로는
`outputs/workflow_validation/paired_refinement_160851/surface_refinement_result.json`.
비교 이미지 `surface_refinement_holdout_comparison.jpg`는 Real/baseline/수정후 순서다.
실제 hardware/Isaac runtime/full manipulation workflow는 이번 검증에서 미실행.

추가 진단 `contact_geometry_diagnostic.json`: 원본 measured q의 URDF visual vertices와
canonical cube box 사이 최소 vertex 거리(shot2/5/6)의 기록이다. Surface의 실제 최근접
거리보다 크거나 같은 값이므로 physical contact gap으로 해석하지 않는다. 실제 로봇의
영점 보정이나 물체 접촉을 이 수치만으로 확정하지 않는다.

다음 기하 fitting에서는 **camera.pose와 동일한 wrist_camera_mount frame을 함께 관리**한다.
현재 set_camera/fit API는 다른 frame을 자동 동기화하지 않는다. 시점만 변경한 검은 화면을
카메라 fitting의 물리적 실패로 오해하지 않는다. 다음에는 robot-fixed landmark와 그리퍼
윤곽을 함께 제약해 여러 자세에서 mount/기구 오차를 분리해야 한다.
후보 폴더의 `experiment_recipe.py`, `fit.json`은 실행 이력이며 같은 이름으로 다시 실행하면
기존 후보 보호를 위해 실패한다. 새 후보명/경로를 지정해서 재사용한다.

기준일: 2026-10-10, Asia/Seoul. 이전 대화를 모르는 Codex를 위한 현재 작업 기록이다.
전체 프로젝트 설명은 `CODEX_HANDOFF.md`, Builder 쪽은
`../real2sim-builder/CODEX_HANDOFF.md`를 먼저 읽는다. 이번 문서는 Scene 보정의
실제 자료, 실패한 가설, 새 검토용 export, 다음 시작점을 설명한다.

## 1. 사용자가 원하는 결과와 작업 범위

- 실제 SO-101 작업대 사진과 이미 수집한 Snapshot/Real-Sim Pair로 Scene을 만든다.
- 외부 Codex가 reconstruction agent이고 Builder는 offline author/fit/render/compare 도구다.
- Wrist 그리퍼 정렬과 Side/Wrist의 동일 자세 일관성이 중요하다. Real 그리퍼가
  바닥/블록에 닿았는데 Sim에서는 떠 있는 차이도 진단해야 한다.
- Leader/Follower 모터 캘리브레이션이 되어 있어도 URDF 영점/축/그리퍼 변환과
  실제 기구가 같다고 가정하지 않는다. 원본 measured q는 보존한다.
- 새 Pair 8쌍은 사용자가 **제외 없이 사용하도록 명시적으로 요청**했다. 경고는 남기되
  soft quality flag만으로 자료를 버리지 않는다. 파일 무결성/좌표 계약 검사는 유지한다.
- 물체 배치까지 수정한 Scene XML/USD 및 metadata를 export하고 SO-101 UI에서 쓰게 한다.
- 최신 지시: 진행 중이던 후보 평가와 검토용 export/Workspace를 마무리하고 인수인계한다.
  이번 턴 이후 추가 fitting, 새 후보 생성, 실제 로봇 구동은 수행하지 않는다.

## 2. 두 프로젝트의 역할과 실제 흐름

```text
soarm101_lab UI → 실제 장치에서 Snapshot/Pair 저장
  ↓ 이미지 / measured q / calibration / robot model / hashes
real2sim-builder → Codex 이미지 해석 → canonical/candidate 작성
  → pose replay → render → compare → fit/edit → evaluate → promote/reject
  → scene.yaml / scene.xml / scene.usd / scene_manifest.yaml / SCENE.md
  ↓ Builder package import
soarm101_lab real2sim.workspaces.builder.import_scene
  → immutable Workspace (profile/task/physics/provenance/USD wrappers)
  → 기존 Real/Sim UI 및 Source → Annotation → Datagen → Convert → Train/Eval
```

Builder export의 로봇은 reference/FK geometry다. SO-101 Adapter가 제어 articulation을
연결한다. 정적 Builder 비교와 실제 런타임 actuator/contact 결과를 동일하다고 주장하지 않는다.
문서: `docs/real2sim/BUILDER_WORKSPACE.md`, `PAIRED_CAPTURE.md`, `CAPTURE_DATASET.md`.
Builder 도구/schema: `../real2sim-builder/docs/AGENTIC_TOOLKIT.md`, `MULTIPOSE_CAPTURE.md`.
새 camera/object/task는 metadata로 전달하지만 지원 범위는 기존 SO-101 pick/place recipe다.
MuJoCo Datagen bounded IK와 Isaac MimicGen은 같은 알고리즘이 아니다.

## 3. 정확한 데이터와 프로젝트 경로

모든 상대 경로는 SO-101 root `/home/hyeonhee/soarm_isaaclab/soarm101_lab` 기준이다.

| 구분 | 경로 |
|---|---|
| 최초 사진/실측 설명 | `../real2sim-builder/inputs/scene1/scene_info.md` 및 같은 폴더 사진 |
| 기존 제작 프로젝트 | `../real2sim-builder/projects/so101-scene1-snapshot-reconstruction-2202744f` |
| 기존 runtime Workspace | `outputs/real2sim/environments/scene1_builder/v1` |
| 새 Pair 세션 | `outputs/real2sim/snapshots/snapshot_20261010_160851_00c94af5` |
| 그 직전 Pair | `outputs/real2sim/snapshots/snapshot_20261010_152443_b4ee565c` |
| 기존 영점/카메라 진단 | `outputs/workflow_validation/paired_refinement_160851` |
| 최신 배치 프로젝트 | `../real2sim-builder/projects/so101-scene1-paired-layout-20261010-b9e9cc4d` |
| 최신 검토용 export | 위 프로젝트의 `exports/so101-scene1-paired-layout-20261010-b9e9cc4d-cups-review` |
| 최신 배치/색상 후보 | 위 프로젝트의 `scratch/agent/cups_layout_white_paper` |
| 최신 UI용 Workspace | `outputs/real2sim/environments/scene1_builder/v2` |

최신 export와 Workspace의 실제 검사 결과/경로는
`outputs/workflow_validation/paired_refinement_160851/layout_export_result.json`에서 확인한다.
존재/결과를 확인하지 않고 이 문서의 예상 경로만으로 실행 성공을 판단하지 않는다.

실측 정보: 컵 높이65mm, 윗지름70mm, 컵 중심간90mm, 블록24mm 정육면체,
표시 영역150×150mm. 컵 바닥지름/벽두께/질량/마찰은 실측이 아니다.

새 세션은 8 shots × Side/Wrist = 16 images. 1–6 train, 7–8 holdout.
이 holdout은 같은 수집 세션의 다른 자세이며 독립 수집 세션 validation이 아니다.
모든 16 images를 eligible로 가져왔고 원본 warnings/measured q/hash를 유지했다.

## 4. 파일 계약과 보정 시 주의사항

- `real2sim.snapshot/1`: 세션 공통 `robot_model/`, `calibration/`, manifest와
  `snapshots/snapshot_00000N/{side,wrist,sim_side,sim_wrist}.png`, state/metadata/pair/COMPLETE.
- named measured q는 rad, native follower feedback와 command는 별도로 저장한다.
  Sim measured q와 real mapped q가 다르면 추적/동역학/매핑 진단 자료이지 physical truth가 아니다.
- host timestamps는 실제 exposure timestamp와 다르다.
- `real2sim.scene/1`: m, +Z up, column-vector transform; optical +X right/+Y down/+Z forward.
- `real2sim.package/1`: `scene_manifest.yaml`의 모든 파일 hash를 검사한다.
- `so101.workspace/1`: published runtime 파일은 immutable. 수정 대신 새 version import.
- joint mapping은 `legacy_unverified`. 수치 fitting만으로 physical calibration 완료라 표시하지 않는다.
- Builder candidate pose replay는 measured q를 고정해 FK와 `base × FK × wrist_mount`를 계산한다.
- 런타임 replay는 reset이 물체를 랜덤화한다. 비교 전에 저장된
  `pair.runtime_profile.objects[name].T_workspace`로 물체 pose를 복원해야 한다.

`app.core.capture.import_paired(..., include_flagged_reason='사용자 승인 이유')`는
explicit quality override를 지원한다. `capture_quality_reasons`, `quality_override`가
evidence에 남고 raw dataset은 바뀌지 않는다. **현재 CLI에는 이 이유 옵션이 없다.**
`reconciliation`은 raw K와 canonical의 추정 K가 달랐던 사유를 명시하는 API 옵션이다.
실측/calibrated K를 임의 변경하는 허가가 아니다.

## 5. 지금까지 실행한 보정과 실제 결과

### 관절 영점/카메라 가설

기존 프로젝트에서 `pair_all_160851_wrist_mount`를 평가했으나 REJECTED.
기존 `multipose.fit`은 holdout 악화 시 **원래 scene을 반환**한다. 따라서 이 후보의
전체 렌더는 개선된 mount가 아니라 baseline이다.

12개의 색상 silhouette 중심 correspondence를 estimated/low-weight로 기록했다.
이들은 실측 코너가 아니며 가려짐/blur/3D 중심과 silhouette 중심 차이에 영향을 받는다.

`joint_silhouette_diagnostic.json`의 train-only offline 가설:

| 가설 | Train px 전→후 | Holdout px 전→후 |
|---|---|---|
| shoulder/elbow 영점 | 31.41 → 21.61 | 22.82 → 27.36 |
| 영점 + wrist mount | 31.41 → 13.83 | 22.82 → 23.69 |

추정 영점을 URDF/controller에 적용하지 않았다. 이 결과는 실제 기구 차이를 부정하지
않으며 현재 대응점으로 오차 원인이 충분히 분리되지 않았다는 뜻이다.
`REPORT.md`, `joint_zero_diagnostic.json`, `joint_silhouette_diagnostic.json`에 상세 결과가 있다.

### 최신 컵 배치 후보

최신 Pair 전용 새 프로젝트를 만들고 기존 canonical/assets를 baseline으로 복사했다.
기존 프로젝트/export/v1과 사용자 원본은 보존했다. block/table/robot/camera는 이번 후보에서
수정하지 않았다. 이동한 것은 컵과 컵에 붙은 빨간 인쇄 visual이다.

컵 입구 ellipse는 train shot3 Side/Wrist에서 직접 해석한 **추정 evidence**다.
`cups_latest_layout/layout_fit.json`에 목표 pixel 중심/ellipse와 파라미터가 있다.
고정 실측 크기/높이와 90mm 중심간격을 유지해 공유 평면배치를 맞췄다.

| 물체 | 기존 XY(m) | 후보 XY(m) |
|---|---|---|
| cup_a | [0.24497205, -0.19727896] | [0.25323881, -0.15967502] |
| cup_b | [0.15755338, -0.17587859] | [0.16324022, -0.15916952] |

둘 다 center Z=0.0325m, 높이0.065m, 지름0.07m. cup_a 이동 약38.5mm,
cup_b 약17.7mm. `cups_layout_white_paper`는 추가로 종이컵 색을 선형 RGB
[0.92,0.90,0.83]으로 추정했다. 렌더 색은 measured albedo가 아니다.

| 전체 영상 MAE (0–1, 낮을수록 좋음) | Baseline | 배치+색상 후보 |
|---|---|---|
| Train 12 images | 0.14672549 | 0.14885090 |
| Holdout 4 images | 0.13635893 | 0.13944194 |

선택된 Wrist 컵 입구 중심은 가까워졌지만 전체 영상과 Side/holdout은 악화됐다.
따라서 **이 후보는 승격/검증 성공이 아니다**. 사용자 요청에 따라 XML/USD를
`UNVALIDATED`, `VISUAL_REVIEW_ONLY_NONREGRESSION_FAILED`, `PHYSICS_NOT_VALIDATED`로
export하여 UI에서 직접 비교할 수 있게 한다.

새 프로젝트의 `working/current_scene.yaml`은 baseline으로 유지된다.
검토용 export의 source는 **candidate_scene.yaml**이다. `working/review_export.json`이
그 관계를 기록한다. 둘을 혼동해서 다음 export에 baseline을 사용하지 않는다.

## 6. 도구와 이어서 작업하는 방법

기존 Python: `/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python` (3.11).
기존 Blender: `/home/hyeonhee/apps/blender/blender`. 설치/conda 버전 변경 금지.
MuJoCo GPU EGL/software-egl은 새 process로 비교하고 pxr/Isaac import와 분리한다.
USD/Qt 경로는 명령에만 지정한다. 상세 `docs/real2sim/TESTING.md` 및 Builder `LOCAL_RUN.md`.
다른 머신에서는 아래 `/home/hyeonhee` 경로를 실제 checkout/environment에 맞춰 바꾼다.

Builder에서 현재 자료 확인:

```bash
cd /home/hyeonhee/soarm_isaaclab/real2sim-builder
PY=/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python
PROJECT=projects/so101-scene1-paired-layout-20261010-b9e9cc4d
export REAL2SIM_BLENDER=/home/hyeonhee/apps/blender/blender
"$PY" -B -m app.cli inspect "$PROJECT"
```

다음 개발자가 새 후보를 실행할 때는 baseline으로 시작하는 `candidate create` 후,
원하는 review candidate의 scene과 base/evidence signature를 먼저 확인하고 변경을 명시한다.
Rejected candidate는 다시 edit하지 말고 새 이름으로 만든다. promote 검사를 우회하지 않는다.

```bash
"$PY" -B -m app.cli candidate "$PROJECT" create next_shared_calibration
"$PY" -B -m app.cli candidate "$PROJECT" edit next_shared_calibration --edits /absolute/edits.json
"$PY" -B -m app.cli candidate "$PROJECT" evaluate next_shared_calibration --engine mujoco --renderer software-egl --validation
# 개선과 holdout nonregression을 실제 통과한 경우에만 promote.
```

UI 입력은 Builder 프로젝트/후보 폴더가 아니라 실행용 Workspace다.
`layout_export_result.json`의 workspace를 Workspace Pipeline에서 선택하고,
Real/Sim에서 **Pipeline Workspace 사용**을 켜고 backend 선택 후 Connect한다.
UI를 켜거나 문서를 읽는 것만으로 로봇을 연결/구동하지 않는다.
Source reset은 설정에 따라 block을 랜덤화하므로 Real 배치 비교와 혼동하지 않는다.

새 export를 import하려면 SO-101 root에서:

```bash
"$PY" -B scripts/real2sim/workflow.py import-builder /absolute/export \
  --workspace-id scene1_builder --version NEW_UNUSED_VERSION
```

이 명령의 USD wrapper 생성은 기존 pxr 환경 경로가 필요할 수 있다. 아래 실제 검사 결과를
확인하고 문서의 오래된 optional dependency 상태를 최신 성공으로 오해하지 않는다.

이번 import에 사용한 기존 USD 경로:

```bash
USDROOT=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
KITROOT=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/kit
PYTHONPATH=source/soarm101_lab:"$USDROOT" LD_LIBRARY_PATH="$USDROOT/bin:$KITROOT" \
  "$PY" -B scripts/real2sim/workflow.py import-builder /absolute/export \
  --workspace-id scene1_builder --version NEW_UNUSED_VERSION
```

경로는 기존 설치 확인 후 사용하며 MuJoCo worker에는 이 USD 환경을 전달하지 않는다.
Adapter의 컵 target ID `cup_a`/`cup_b`는 profile object
`cup_a_visual`/`cup_b_visual`로 매핑된다. `task.json.targets`를 따라 읽는다.

검토 후보의 export는 기존 `export_package(..., allow_unvalidated=True)`를 재사용했다.
`app.cli export PROJECT`는 baseline canonical을 내보내므로 해당 명령만으로 이번 후보를
재현할 수 없다. 이번 호출 형태는 아래와 같다(Builder cwd, 같은 기존 USD 환경).
기존 export를 덮어쓰려는 목적이 없다면 재실행할 필요가 없다.

```python
from types import SimpleNamespace
from app.core.project import Project, read
from app.core.candidates import Candidate
from app.exporters.package import export_package
p = Project('projects/so101-scene1-paired-layout-20261010-b9e9cc4d')
c = Candidate(p, 'cups_layout_white_paper')
report = read(c.root / 'review_export_report.json')
assert not report['passed']
assert report['input_signature'] == p.signature(c.scene)
view = SimpleNamespace(
    root=p.root, scene=c.scene, evidence=p.evidence,
    info={**p.info, 'id': p.info['id'] + '-cups-review'},
    state={**p.state, 'candidate': c.name},
    signature=lambda scene=None: p.signature(c.scene if scene is None else scene))
export_package(view, report, allow_unvalidated=True)
```

이는 사용자가 요구한 명시적인 후보 비교용 UNVALIDATED export다. 새로운 permanent
exporter/launcher는 추가하지 않았다. Candidate rejection/acceptance 상태는 보존한다.

## 7. 검증과 남은 일

- 기존 Pair 회귀 4 passed: override/default, 원본 보존, 경고 유지, hash 훼손 거절.
- 두 최신 후보 각각 실제 MuJoCo 16 images + baseline 16 images 평가 완료.
- 이번 검토용 export: XML model load PASS(330 geoms), USD load PASS(482 prims),
  XML/USD structural parity PASS, export/Workspace hash 검증 PASS.
- 새 v2 runtime: 실제 RTX4060 EGL, Side/Wrist RGB 각각480×640×3, finite step10회 PASS.
  MuJoCo worker의 isaaclab/omni/pxr import 없음. Workflow Workspace artifact로 등록했다.
- 실제 Blender5.2.2에서 `candidate.blend` 생성과 measured-q shot3 두 카메라 preview 완료.
  Blender 전체16 observations의 metric 평가는 이번에 수행하지 않았다.
- 두 엔진 파일 생성/structural parity/Workspace import/런타임 smoke의 이번 최종 결과는
  `layout_export_result.json`이 authoritative record다.
- Blender canonical review file/shot3 preview와 per-shot MuJoCo overlay/difference는
  후보 폴더에 남긴다. `renders/mujoco/metrics.json`에서 수치를 읽는다.
- 실제 hardware, Isaac runtime, grasp/Source/Annotation/Datagen/학습은 이번 보정 검증이 아니다.

이번 실행에서는 runtime target을 바로 profile의 `cup_a`로 조회한 검사 스크립트가
KeyError로 한 번 중단됐다. 실제 mapping은 `task.targets['cup_a']='cup_a_visual'`이다.
이에 맞춰 조회를 고쳐 두 컵의 새로운 pose와 Workspace hash를 확인했다.
Scene importer/runtime의 오류였던 것은 아니다.

관련 회귀 재현 명령(SO-101 root):

```bash
PYTHONPATH=source/soarm101_lab "$PY" -B -m pytest \
  tests/real2sim/test_snapshots.py -k 'pair and not command' -q -p no:cacheprovider
```

다음 작업은 새로운 임의 영점을 확정하는 것이 아니라, 기존 8쌍에서 robot-fixed
landmark/그리퍼 끝단/확실한 contact label을 확인해 mount와 기구 오차를 분리하는 것이다.
컵 입구의 크기까지 다르므로 camera intrinsics/depth와 geometry의 관계도 확인해야 한다.
holdout을 반복 선택의 학습자료처럼 쓰지 말고, 독립 session 검증이 없다는 제한을 보존한다.
보정값은 원본 q에 덮어쓰지 않고 별도 derived calibration으로 기록한다.

## 8. 다른 계정에 전달할 것과 금지 사항

- 두 프로젝트 전체와 ignored 사용자 data를 함께 전달한다. SO-101 Git clone만으로
  Builder나 outputs/Snapshot/assets/checkpoint를 복구할 수 없다. Builder는 Git repo가 아니다.
- `.git`에 많은 미커밋 수정/삭제/untracked가 있다. `git status`, 해당 diff를 확인하고
  reset/clean/일괄 checkout/자동 commit을 하지 않는다.
- 원본 snapshots/photos/HDF5/checkpoint, 기존 scene1/reference project 및 Workspace v1은 보호한다.
- shared robot model/calibration은 세션당 한 번 저장한다. 옛 continuous capture 시스템을
  다시 만들거나 중복 launcher/새 reconstruction framework를 만들지 않는다.
- `/tmp` 실험 스크립트는 환경 재시작으로 이미 사라졌다. 다음 계정이 존재를 가정하면 안 된다.
  필요한 수치/파라미터/서명/metric은 project와 workflow_validation에 보존되어 있다.
- 사용자 credential, 외부 계정 token, 실제 장치 자동 구동은 인수인계에 포함하지 않는다.

새 Codex에게 첫 메시지 예:

> AGENTS.md, CODEX_HANDOFF.md, SCENE_REFINEMENT_HANDOFF.md와 sibling Builder 인수인계를
> 읽고 실제 파일 상태부터 확인해. 최신 cups-review export는 검증 성공이 아닌 비교용 후보야.
> 8 Pair의 measured q와 원본을 유지하고, 기존 candidate/render/compare 도구로
> Wrist/Side와 실제 접촉 차이를 함께 줄이는 보정을 이어가. 기존 로봇을 자동 구동하지 마.
