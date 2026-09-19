# SO-101 Real2Sim Control / Measurement

PyQt UI와 Isaac runtime은 별도 프로세스이며 같은 `--workspace`의 file IPC를 사용합니다.
일반 UI 실행에는 OpenAI API key, model 선택, 네트워크 요청이 필요하지 않습니다.

```bash
# 기존 GUI 환경
.venv-real2sim-ui/bin/python scripts/real2sim/real2sim_ui.py
# 기존 Isaac Lab Python 환경에서
python scripts/real2sim/real2sim_sim.py
```

기존 environment, motor calibration, teleoperation, Mimic/datagen/policy 설정은 유지합니다.
`agent.py`는 UI에서 분리되어 있지만 외부 orchestration을 나중에 검토하기 위해 보존합니다.
해당 모듈을 직접 실행하는 사용자의 코드는 여전히 API를 호출할 수 있습니다.

## Bootstrap 사용

1. Reference에서 batch view를 선택하고 실제 사진을 여러 장 등록합니다. Sim 이미지는 입력으로 요구하지 않습니다.
2. 외부 Codex가 references, 기존 SO-101 asset, 알려진 치수를 검토하여 provisional profile을 준비합니다. 자동 reconstruction/agent는 이번 UI에 구현하지 않았습니다.
3. Real / Sim에서 Profile 열기로 결과를 선택하고 Sim 미리보기 또는 장비 Connect를 실행합니다.
4. 장비 설정 후 기존 Start/Stop으로 자세를 제어하고 같은 Real / Sim 탭에서 Capture 품질을 확인해 여러 자세를 저장합니다.
5. Inspect에서 개선된 profile을 열어 적용하거나 Advanced 측정/optimizer/rerender 도구를 사용합니다.

Inspect → Advanced → Profile JSON의 알려진 치수 템플릿은 기존 bootstrap을 편집/저장하는 선택 도구입니다.
체커보드 도구는 Advanced에 보존했으며 reference 등록의 선행 조건이 아닙니다.
기존 optimizer 자체의 intrinsic 보정 조건은 바꾸지 않았습니다.

## Reference 입력

새 import는 실제 정지 사진(JPG/PNG) 전용이며 batch 전체에 side/wrist/overview/auxiliary/unknown 중 하나를 직접 지정합니다.
파일명에서 view를 추론하지 않습니다. 기존 영상 import UI는 사진 중심 workflow로 대체했으며 과거 영상 추출 파일은 보존합니다.
`references/import_<...>/sources.json`은 `so101.references/1` schema의 이미지별 manifest입니다.
각 item은 UUID id, manifest 기준 상대 stored_path, original_filename, source_path, source_type,
role=real_reference, view, optional camera_id(null 허용), classification_source를 기록합니다.
상위에는 import_id와 UTC imported_at이 있습니다. imported_at은 촬영 시각이 아닙니다.
UI 시작 및 목록 새로고침 시 manifest를 다시 읽습니다. 불완전 import는 경고로 표시합니다.
Legacy manifest는 읽기만 하며 unclassified/unknown으로 표시하고 새 real reference 수에 포함하지 않습니다.
사진 원본/기존 manifest는 이동·삭제·덮어쓰지 않습니다.

references는 실제 입력, revisions는 수치 scene state, renders는 simulation 출력,
captures는 measured state/quality가 연결된 paired evidence입니다. Capture를 references에 복사하지 않습니다.
Real / Sim의 Capture availability는 최근 관측 기준이고, 최종 capture eligibility는 기존 runtime이 다시 검사합니다.

전체 문서를 실측으로 승격하는 checkbox는 없습니다.
Bootstrap scene에는 기존 robot과 provisional cameras/light가 사용됩니다.
치수/위치가 미정인 table, 위치가 미정인 cube, 형상이 미완성인 cups는 렌더하지 않습니다.
150 mm marker 크기는 numerical reference로 보존하며 이번 변경에서 새 marker mesh는 만들지 않습니다.
빈 workspace USD도 유효한 generated representation입니다.

## Profile schema 2

`schema = so101.real2sim/2`. 수치 배열의 기존 위치를 유지하고 각 block에
`provenance: {필드명: 출처}`를 둡니다. 지원 출처는 다음과 같습니다.

- `user_measured`: 사용자가 제공한 직접 측정값
- `model_asset`: repository의 기존 robot asset
- `derived`: 다른 값에서 계산된 값
- `numerically_fitted`: 수치 보정 결과
- `provisional`: preview/solver 초기값이며 측정값이 아님
- `unmeasured`: 미측정값. 값은 `null`

`geometry_source` 같은 전체 profile 출처는 schema 2에서 거부합니다.
숫자가 필요한 camera/base/workspace/light 초기값은 provisional로 명시합니다.
Camera K와 0 distortion은 rendering initialization이며 physical intrinsic/distortion ground truth가 아닙니다.
provisional camera/base/workspace를 calibrated=true로 표시하면 validation에서 거부합니다.
색과 roughness도 현재는 provisional이며 측정된 physical material이 아닙니다.

알려진 값:

| 필드 | 값 | 출처 |
|---|---|---|
| robot_asset | 기존 SO101 URDF/USD | model_asset |
| objects.cube.dimensions_m | [0.024, 0.024, 0.024] | user_measured |
| workspace.size_m | [0.150, 0.150] | user_measured |
| 각 cup.top_outer_diameter_m | 0.070 | user_measured |
| 각 cup.wall_thickness_m | 0.002 | user_measured |
| 각 cup.height_m | 0.065 | user_measured |
| relations.cup_spacing.center_to_center_distance_m | 0.090 | user_measured |

90 mm는 cup_a/cup_b 사이의 상대 거리만 나타냅니다. 절대 위치나 방향을 결정하지 않습니다.
`profile.derived_measurements()`가 top inner diameter 0.066 m와 top edge gap 0.020 m를
`source=derived`, formula와 함께 계산합니다. 이 값들은 JSON에 독립 측정값으로 중복 저장하지 않습니다.

Cup의 70 mm는 **윗부분 외경**이며 opening 지름이 아닙니다.
`bottom_outer_diameter_m`, `bottom_inner_diameter_m`, `bottom_thickness_m`, `taper_profile`,
`T_workspace`는 null/unmeasured입니다. 현재 partial `shape=cup`은 렌더 및 metric fitting에서 제외됩니다.
새 bootstrap에는 cylindrical cup approximation이 없습니다.

`render_enabled=false`는 해당 object를 USD에서 생략합니다. Box/cylinder의 preview를 켜려면
양의 dimensions와 명시적인 T_workspace가 필요합니다. pose를 초기값으로 넣는 경우 출처는
provisional로 두며, 이것을 실측 위치로 해석하지 않습니다. Render 여부 변경은 reconnect가 필요합니다.
Unknown/provisional dimensions 및 cup proxy는 metric landmark constraint로 사용하지 않습니다.

## 기존 profile/revision 읽기

`load()`는 schema 1을 메모리에서 schema 2로 보수적으로 변환합니다.
원본 파일과 기존 revision directory는 수정하지 않습니다.

- 기존 숫자는 보존하되 전역 user_measured 표시는 신뢰하지 않습니다.
- geometry/camera/appearance 숫자는 provisional로 표시합니다.
- 기존 calibration flag는 false로 내려 evidence 재검토를 요구합니다.
- `migration.parent_hash`로 원본을 연결합니다.
- 기존 원통형 cup은 `cup_proxy`로 명시합니다. 원통 반지름을 바닥까지 연장하고
  wall_m를 바닥 두께에도 쓰던 기존 preview 동작을 보존하지만, 둘 다 provisional 가정이며
  metric fitting에서는 거부합니다.
- 사용자가 확인한 측정/보정 근거를 필드별로 입력한 뒤 새 revision으로 저장할 수 있습니다.

Migration은 과거 실제 측정 여부를 복원하지 않습니다. 숫자가 예제와 같거나
전역 user_measured 문자열이 있다는 이유로 측정값을 인증하지 않습니다.

## 유지된 기능과 이번 범위

4-camera viewer, Connect/Start/Stop/Disconnect, paired Capture, landmark 클릭,
수치 optimizer, metrics, rerender, file IPC는 유지합니다.
Capture는 기존 measured-state/timestamp/quality 검사와 COMPLETE 저장 방식을 사용합니다.
Stop은 새 physical target 전송을 중단하고, Disconnect는 기존 adapter로 torque를 해제합니다.

수치 fitting의 목적함수, 파라미터 수, 초기화, bounds, holdout/acceptance 규칙은 변경하지 않았습니다.
변경한 fitting 연결은 unknown/proxy geometry 입력 거부와 accepted transform의 provenance 기록뿐입니다.
체커보드 intrinsic 보정 기능과 두 카메라 intrinsic prerequisite도 현재 그대로입니다.
따라서 새 bootstrap이 곧바로 calibrated profile이 되거나 camera matching을 수행하지는 않습니다.
Effective pose+focal fitting, PnP, silhouette/appearance fitting, external agent protocol은 후속 작업입니다.

## 검증

`tests/real2sim`은 synthetic fitting, capture/storage/IPC, mock hardware guards,
PyQt offscreen UI와 실제 USD authoring/reopen 검증을 포함합니다.
`test_provenance.py`는 이번 schema/측정/누락 geometry/API-free UI 회귀 검증입니다.
물리 장비 연결이나 simulation application을 시작하지 않습니다.

`VALIDATION.json`, `FILE_AUDIT.md`는 기존 기록이며 이번 변경의 실행 결과를 뜻하지 않습니다.
GPU/RTX rendering과 실제 USB/motor 동작은 별도로 확인해야 합니다.

## Reviewed real-reference bootstrap

현재 실제 reference Side 1장, Wrist 1장, Overview 3장을 검토한 장면 가설은
`docs/real2sim/bootstrap/reviewed_scene.json`입니다. 이는 범용 자동 reconstruction이 아닙니다.
다른 환경에서는 해당 환경의 real references를 검토한 새로운 hypothesis가 필요합니다.
원본 filename이 아닌 manifest의 role/view로 입력을 선택하고, reviewed item ID/view/image SHA256을 검증합니다.
Legacy/unknown은 primary target으로 사용하지 않습니다. 기존 sim screenshot은 필요하지 않습니다.

```bash
PYTHONPATH=source/soarm101_lab .venv-real2sim-ui/bin/python -m soarm101_lab.real2sim.bootstrap \
  --workspace outputs/real2sim \
  --hypothesis docs/real2sim/bootstrap/reviewed_scene.json
```

명령이 출력하는 새 profile 경로를 UI Real / Sim → Profile 열기로 선택하고
새로 실행한 Isaac runtime에서 Sim 미리보기를 누르세요. 기존 environment 재생성 문제를
피하려면 실행 중인 장비와 분리한 새 runtime을 사용하세요. Bootstrap 명령 자체는 장비를 열지 않습니다.

현재 권장 결과:
`outputs/real2sim/revisions/revision_20260918_132346_be877f679f6c4450b31a81718f58b1da/profile.json`

Table, paper marker와 검은 테두리, 세 색상 cube, 두 cup visual proxy, 검은 원통 배경과
단순 벽을 기존 primitive builder로 만듭니다. Robot은 기존 asset을 그대로 사용합니다.
Cube 위치는 사용자 요청에 따라 150 mm workspace 내부에 임시 배치했습니다.
Randomizer는 변경하지 않았습니다. 컵 원본 측정 record는 부분 geometry 상태로 유지하고,
별도 `cup_a_visual`/`cup_b_visual`은 provisional cylindrical proxy로서 metric fitting에서 제외됩니다.
Proxy bottom과 taper는 측정값이 아닙니다. 컵 중심 거리는 90 mm를 유지합니다.
Wrist reference는 흐리고 joints가 없으므로 기존 mount seed를 provisional로 유지합니다.
Robot default joint pose도 reference measured pose가 아닙니다.

Profile/scene schema나 scene builder 변경은 필요하지 않았습니다.
현재 generated primitive들은 기존 builder의 static collision geometry이며 movable task objects가 아닙니다.
향후 task randomizer 연결은 별도 단계입니다. 기존 task machinery를 재사용하지만
training/Mimic에 바로 적용했다는 의미는 아닙니다.

비하드웨어 USD 생성/재로딩 및 동일 profile의 deterministic 출력은 검사했습니다.
현재 검증 환경에서 NVIDIA driver에 접근하지 못했으므로 Side/Wrist GPU render 및
visual similarity는 검증하지 않았습니다. revision의 `report.json`에 기록된 `render_verified=false`를
렌더 성공으로 해석하지 마세요.

## Protocol-based rebootstrap candidate (2026-09-18)

Bootstrap/iteration 요청은 `REAL2SIM_AGENT_PROTOCOL.md`를 먼저 읽습니다.
새 candidate:
`outputs/real2sim/revisions/revision_20260918_143142_0cae6f473d334b4b933dcbeccec13b9a/profile.json`

변경은 profile에 기록했습니다: 기본 asset FK 방향에 근거한 provisional base yaw,
Side marker 4개 영상점으로 fixed-focal pose 초기화, -Z tool 방향을 보는 wrist optical frame,
측정 90mm 컵 간격을 유지하는 provisional 배치, tapered visual cup과 사진 기반 print texture,
gripper에 종속된 provisional camera body/bracket/lens입니다.
실측값은 그대로이며 cup bottom OD 52mm는 visual hypothesis일 뿐입니다.
원본 side 이미지를 수정 없이 texture asset으로 보존하고 UV rectangle로 인쇄 영역을 사용합니다.
이는 반사율 복원이 아니고 사진의 조명/원근이 포함된 임시 texture입니다.

기존 builder에 optional cup_proxy.bottom_diameter_m/print_texture와 wrist_mount만 추가했습니다.
Tapered cups와 mount는 visual-only이며 metric cup constraints 또는 physics reconstruction이 아닙니다.
Mount/topology/texture 변경은 reconnect가 필요합니다. 원본 robot USD는 수정하지 않습니다.
Runtime은 mount USD를 gripper 아래에 생성하고 optical camera도 같은 gripper frame을 사용합니다.

`report.json`의 Side RMSE는 수동으로 지정한 동일 4개 marker corner의 **좌표 성분별**
재투영 잔차(89.40→4.89 px)입니다. Held-out metric이나 전체 이미지 유사도 점수가 아닙니다.
COMPLETE capture가 없어 reference robot pose는 알 수 없습니다.
GPU render는 이 실행 환경에서 불가능했으므로 새 Side/Wrist 렌더 비교는 아직 미검증입니다.
새 runtime에서 위 candidate를 선택하고 장비 없는 Sim 미리보기로 확인하세요.
