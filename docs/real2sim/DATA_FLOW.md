> Historical / Archived — 2026-10-22: 아래 저장/Registry 설명은 과거 기록입니다. 현재 데이터 흐름은 [SYSTEM_FLOW](../architecture/SYSTEM_FLOW.md)를 따릅니다.

# SO-101 Real2Sim Data Flow

## 단계별 저장 경로·입력 선택 — 2026-10-19

모든 Workflow 화면 상단의 **저장 이름 (필수)**에 현재 단계의 결과 이름을 입력한다. 단계별 이름을 따로
보관하며 입력한 이름을 결과 폴더와 Registry `display_name`에 사용한다. 비운 채 시작하면 이름 입력
오류 알림을 표시하고 worker를 실행하지 않는다. Annotation은 선택한 Source 이름에 `annotation_`을 붙여
기본값을 채우며 직접 수정할 수 있다. 입력 Source를 바꾸면 자동 이름만 갱신하고 사용자 지정 이름은 유지한다.
모든 단계의 `저장 경로(output)` editor와 이전 UI 설정값의 전달은 제거했다. 현재 UI는
필수 이름으로만 새 저장 위치를 정하며 LeRobot 병합은 전용 선택 창에서 이름을 입력한다. CLI의 명시적 저장 이름도
계속 지원한다. 한글/영문/숫자/밑줄/하이픈, 100자·UTF-8 200바이트 이내이며
경로 구분자·점·공백·장치 예약 이름은 거절한다. Worker 내부의 데이터 형식과 파일 구조를 유지하며,
Source는 자기 origin·선별·분할을 묶는다. 이후 단계는 각 저장소의 `<저장 이름>/`에 바로 저장한다.
입력 Source/Annotation/DataGen 이름·split·모델 종류로 상위 폴더를 자동 생성하지 않는다.

```text
outputs/workflow/
  sources/<사용자가 입력한 Source 이름>/
    origin/                  # Isaac: source.hdf5 / MuJoCo: dataset.hdf5 또는 과거 dataset/
    <사용자가 입력한 선별 이름>/
      dataset.hdf5           # Source 일부를 제외한 선별본; 영상·계약 포함
    <사용자가 입력한 분할 이름>/
      train/dataset.hdf5     # 과거 Canonical 입력이면 train/ 자체가 Dataset
      val/dataset.hdf5
      source_split.json
    <다른 분할 이름>/         # 같은 origin, 다른 비율/Seed/선택
      train/
      val/
      source_split.json
  annotations/<저장 이름>/     # 기본 annotation_<선택한 Source 이름>; HDF5·MP4·보고서
  datagen/<저장 이름>/          # dataset.hdf5·dataset.videos·report
  datasets/<저장 이름>/         # 변환/병합한 LeRobot의 meta/data/videos 등
  models/<저장 이름>/           # 기존 정책 형식의 checkpoints 또는 policy 내부 구조 유지
  evaluations/<저장 이름>/      # 평가/Replay/업로드 결과; 형식별 report/result JSON 등
  curation/filtered/<이름>.hdf5  # Annotation/DataGen 일반 선별; Source는 자기 묶음에 저장
  runs/<고유 Run ID>/         # request.json, state.json, process.log, presentation/
  artifacts/<고유 ID>.json    # 데이터가 아닌 Registry 인덱스
```

Annotation에는 Source/origin/split/train/val 경로를 다시 중첩하지 않는다. 예: Source `테스트_1`이면
`annotations/annotation_테스트_1/dataset.hdf5`다. `train`/`val`처럼 일반적인 입력 이름은 Source·분할 이름을
붙여 서로 구분하고 같은 이름이 이미 있으면 새 이름을 입력해야 한다. 실제 Train/Val 역할·계보는 기존
HDF5의 Source Partition과 Registry 부모 ID로 유지한다. 기존 Annotation 폴더/Registry는 이동·변경하지 않고
계속 읽는다. 모델의 실제 학습 Dataset은 Registry 부모 ID와
기존 학습 config/provenance에 기록한다. `split_N`은 데이터의 Train/Val 분할이며 모델의 분할을 뜻하지 않는다.
필요한 폴더는 새 저장 시 생성한다.
Worker에는 실제 결과 경로를 전달하고 request/state.plan.output에
동일하게 기록한다. 직접 지정한 `output`은 유지한다. 저장 이름도 지정했다면 경로의 마지막 이름(파일은
stem 또는 포함 폴더)과 일치해야 한다. 파일형 Source/Evaluation은 HDF5/JSON 확장자를 검사한다.
기존 결과가 있으면 거절하고, 동시 실행은 저장 경계 옆의 배타적 예약 파일로 중복 사용을 차단한다.
중단 결과도 삭제/덮어쓰지 않는다. DataGen/Source 이어 수집·MiniVLA Resume은 기존 입력을 선택하고
새 결과에 저장한다. UI Source는 이 경우에도 새 Source 이름을 입력한다. ACT/SmolVLA는 기존 Resume 출력 계약을 유지한다.
CLI에서 ACT/SmolVLA 재개의 새 이름을 지정하려면 일치하는 새 `output`도 명시해야 한다.
강제 kill로 예약 파일이 남았다면 자동 삭제하지 않고 해당 파일에 기록된 Run의 종료 여부를 먼저 확인한다.

Source 이름은 확장자 없이 입력한다. 예: `테스트_1` → `sources/테스트_1/origin/`.
worker의 HDF5 basename은 기존대로 MuJoCo `dataset.hdf5`, Isaac `source.hdf5`이고, 과거 canonical
Source는 `origin/dataset/` 형식을 유지한다. 기존 자동 이름의 Source는 목록에서 기록된 Task ID와
수집 시각을 먼저 보여 준다. 이 표시 변경은 기존 폴더·Registry JSON을 수정하거나 이동하지 않는다.

**이어 수집할 데이터 선택 (HDF5 / 폴더)**은 엔진과 무관하게 기존 Curation의 `DatasetPicker`를 사용한다.
새 HDF5 파일과 과거 canonical 데이터셋 폴더를 한 창에서 고른다. Source 선택창은 `sources`에서,
DataGen은 `datagen`에서 시작한다. `origin` 안의 HDF5 또는 원래 `dataset` 폴더를 고른다.
평가 폴더는 이어 수집 입력으로 취급하지 않으며 분할한 Source Train/Val은 잠겨 있다.
Source의 **총 목표 Episodes (기존 포함)**는 누적 목표다. 기존20개에5개 추가는25다. 선택할 때 현재
목표가 기존 개수 이하이면 기존+1로 올린다. MuJoCo도 기존 `data_resume.validate`를 계획 시점에 호출해
잘못된 역할/Task·Scene·총 목표를 장치 연결 전에 거부한다. MuJoCo worker에는 총 목표를, 기존 Isaac
worker에는 추가할 개수를 전달하는 현재 계약을 유지한다.

Source의 이어 수집과 DataGen의 이어 생성은 화면 위젯을 재사용하되 `resume_inputs`에 **단계별로 따로**
보관한다. Source에서 고른 경로가 DataGen으로 넘어가지 않는다. DataGen의 `기존 DataGen 결과 (선택)`는
새 생성이면 비워 둔다. 이것은 기본 Annotation 입력을 고르는 곳이 아니라 이전 생성 결과를 이어 만드는 입력이다.
`data_resume.inspect_resume_input`은 `.videos` 같은 영상 보관 폴더와 Source/Annotation을 이전 생성 결과로
사용하지 못하게 한다. `unified_plan`은 같은 엔진·Workspace·Task·Annotation SHA와 총 목표도 검사한다.
직접 worker 요청 역시 `unified.execute` 입구에서 잘못된 Resume 경로를 출력/엔진 생성 전에 거부한다.
기존 worker의 상세 seed/IK/정책/좌표·모델 Resume 검사는 그대로 유지한다.

새 Annotation은 지정한 위치에 등록하고 Source를 부모로 연결하여 원본을 보존한다.
기존 명시적 `Registry.promote_annotation`의 검증·rollback API는 유지한다. Curation에서 Source의 선별·분할을
저장할 때는 모두 저장 이름을 직접 입력한다. 선별본은 `sources/<Source>/<선별 이름>/dataset.hdf5`,
분할본은 `sources/<Source>/<분할 이름>/{train,val}/dataset.hdf5`다. 과거 canonical 선별본은 이름 폴더에
manifest와 episode 폴더를 바로 저장한다. Source의 별도 파일 저장창과 자동 `next_split` 번호 코드는 제거했다.
`origin` 이름 재사용과 기존 이름 덮어쓰기는 거부한다. Annotation/DataGen 일반 선별의 filtered 제안/저장창은
유지한다. 저장 후 현재 Replay/Annotation/변환 화면의 입력만 새 결과 폴더로
갱신할 수 있으며 다른 단계의 선택은 바꾸지 않는다. Source에서 저장해도 Annotation을 자동 변경하지 않는다.
Train/Val은 따로 등록하며 기존 Hash/Inventory/Assignments·중복 그룹 분리 금지·분할 잠금을 그대로 사용한다.
이미 나눈 Train/Val에서 재분할하지 않는다. 새 분할은 미분할 origin이나 Source 선별본에서 만든다. Source의 최신 기록이 분할 결과라면
Curation의 기본 열기는 같은 묶음 origin을 사용하고, 명시적으로 선택한 Train/Val 검토·선별은 기존대로 잠금을 유지한다.
관리 밖의 이전 Source는 첫 선별/분할 저장 때 관리 경로의 origin으로 복사한다. 관리되는 이름 폴더의 Source는
현재 선택한 입력을 그대로 사용한다. 기존 파일은 이동/수정하지 않는다.
HDF5 복사본에는 기존 Source 계보/그룹을 명시해 경로 변경으로 그룹이 달라지는 것을 막으며 숫자/영상은 보존한다.
기존 `export_source_splits` 명시적 출력 API는 호환되는 평면 형식도 지원하고 UI는 `nested=True`를 사용한다.

입력 선택기는 날짜/episode/partition/backend/짧은 ID를 표시하며 현재 단계에 호환되는 유형만 선택할 수 있다.
검색·종류·Task 필터와 경로/부모 계보 설명 영역은 제거했다. 실제 선택 경로는 입력 dropdown tooltip에 남는다.
기존 `입력 폴더 열기`/`데이터 묶음 선택` 버튼·처리 함수는 삭제하고 **데이터 선택** 하나로 교체했다.
선택창은 `Registry.input_directory`가 정한 아래 저장소에서 매번 시작한다.

| 단계 | 처음 여는 폴더 | 고를 대상 |
|---|---|---|
| Source | `outputs/real2sim/environments` | `scene2_builder/v8` 같은 READY Workspace 버전 폴더 |
| Replay/Annotation | `outputs/workflow/sources` | Source 폴더. `origin`, 선별 이름, `분할 이름/train`, `분할 이름/val`을 목록에 표시 |
| Mimic/DataGen | `outputs/workflow/annotations` | 사용할 Annotation 폴더 |
| LeRobot 변환 | `outputs/workflow/datagen` | 사용할 DataGen 폴더 |
| 모델 학습/Dataset 업로드 | `outputs/workflow/datasets` | 변환된 Dataset 폴더 |
| Sim/Real 평가·Policy 업로드 | `outputs/workflow/models` | 학습 결과 또는 명시적 checkpoint 폴더 |

선택 폴더 아래의 호환 입력을 기존 Registry 검증으로 찾아 `사용할 데이터/모델` 목록에 표시한다.
하나이면 자동 선택하고, 여러 개면 목록에서 고른다. Source의 origin은 기본 우선 선택한다.
파일 basename 대신 데이터의 폴더 이름을 표시하고 실제 경로는 tooltip에 남긴다.
폴더 범위는 `input_folders`, 선택 Artifact ID와 Train의 Val ID는 `input_selections`에 단계별로 기억한다.
단계 전환 때 공용 combo의 이전 값을 재사용하지 않으며 UI 재시작 후에도 각 단계의 선택을 복원한다.
다른 단계의 Scene 변경으로 현재 단계의 목록을 바꾸지 않고 해당 단계의 선택된 입력/폴더를 기준으로 읽는다.
빈 폴더는 빈 목록으로 남기며 다른 묶음으로 자동 대체하지 않는다.
선택 폴더의 미등록 HDF5/Canonical/LeRobot/완료 정책은 기존 Registry 검사를 거쳐 등록한다.
정책 폴더 탐색은 기존 provenance가 있는 모델을 읽는다. 기록된 Workflow checkpoint를 직접 고른 경우의
명시적 metadata 연결은 기존 `import_checkpoint`를 사용한다. Source의 데이터 선택은 Builder 원본을 가져오지 않고
READY 버전 폴더만 받는다. 기존 Real/Sim 화면의 **Scene 선택 / Import**는 `Window.browse_scene`로 옮겨 유지했다.
선택/완료 후 목록을 자동 갱신하며 별도 새로고침 버튼은 없다.
등록되지 않은 Episode는 기록된 생성/주석 증거로 유형을 판별하고 기존 Registry 검사를 거친다.
MiniVLA는 Train Dataset 폴더를 먼저 선택한 뒤, 같은 데이터 선택 버튼으로 별도 Val 폴더를 고를 수 있다.
실제 Train/Val 계약을 검사하고 기존 Train과 폴더 범위를 유지한 채 Validation 선택기에 연결한다.
명시적으로 고른 Val만 다른 폴더에서도 유지한다. 변환과 Real 평가는 선택한 Sim backend 때문에 호환 입력을
숨기지 않으며 실제 좌표·Task·학습/실물 준비 검사는 기존 실행 서비스에서 유지한다.
Workspace/Python 수동 editor도 없다. 입력의 Scene identity에 맞는 경로를 찾고 기존 backend별 Python 설정/기본값을
worker에 전달한다. Real/Sim도 같은 내부 Scene 상태를 공유하며, 선택만으로 worker/장치를 시작하지 않는다.
`Task provenance` 설명 행과 문구 생성/갱신 코드는 삭제했다. Task 상속·객체/Task 충돌 검사와 데이터 파일의
provenance는 실행·분할 누수 검증에 필요하므로 유지한다. Scene/Task 준비 오류는 기존 `NOT_READY` 상태에 표시한다.
Scene objects/Active Objects와 Task preset/Pick object/Target cup/Language instruction 편집 영역은
Source 단계에만 표시한다. 다른 단계에서는 영역 전체를 숨기며 입력 데이터의 Task/Scene 상속과 기존 평가 처리는 유지한다.
등록 모델은 Resume과 모델 학습의 Loss Graph 목록에서도 선택한다. 그래프는 현재 실행/선택 정책/
Train 입력의 등록 모델에서 metrics.jsonl을 자동 탐색하고 수동 JSONL 선택도 유지한다.
Checkpoint import는 모델의 상위 폴더 또는 Registry의 Run request에서 기록된 출력 관계를 검증한다.

기존 실제 저장 자료도 2026-10-17에 이 구조로 이동했다. Source20 묶음은
`sources/20261016_094817_source_cube_red_to_cup_a_656da738/origin/dataset`이고, 기존 Train14/Val6은
같은 묶음의 `split_1/train`, `split_1/val`이다. 당시 Annotation·DataGen·LeRobot·MiniVLA도 각 단계 폴더에서
같은 Source/분할을 공유한다. Registry의 ID/부모/내용 SHA는 유지하고 실제 path/collection만 갱신했다.
기존 파일에 기록된 절대 경로와 Resume identity를 보존하기 위해 예전 위치에는 새 경로를 가리키는 symlink를 남겼다.
원래 runs의 request/state/process.log/tasks/presentation은 실행 이력으로 유지한다. 실패 실행의 남은 자료는
각 단계 `.incomplete/<Run ID>`에 보관하며 정상 입력으로 등록하지 않는다. 모델/숫자/영상/manifest는 재작성하지 않았다.
이 당시 이동된 폴더는 그대로 읽는다. 현재 새 저장은 위 평평한 이름 구조를 따르며 기존 폴더를 자동 이동하지 않는다.
MiniVLA로 모델을 바꾸면 Train/Val 목록도 다시 필터링하여 Val을 Train 입력으로 남기지 않는다.

2026-10-18에 최상위 `soarm101_lab/datasets`의 호환 링크도 제거했다. 이 경로가 없는 상태에서 Source 원본/분할,
Curation 저장, Train/Val 로딩과 MiniVLA Resume 계약을 확인했다. 기존 `annotated_from` 등 옛 경로 문자열은
과거 출처 기록이며 해당 디렉터리를 열어야 하는 실행 입력이 아니다. 기록/내용 SHA를 바꾸려고 재작성하지 않는다.
새 Source는 계속 `outputs/workflow/sources/<묶음>/origin`에 저장한다. 현재 Curation의 선별·분할은 같은 묶음 아래
사용자가 입력한 이름을 쓰며, 기존 `split_N` 경로와 metadata도 계속 읽는다.
검증 결과는 `outputs/workflow_validation/legacy_datasets_removal/REPORT.md`다.

2026-10-18의 후속 사용자 요청으로 기존 DataGen/변환/모델/평가10개 경로를 당시의 바로 앞 입력 구조로 이동했다.
Registry9개의 위치만 갱신했고666개 파일 내용·ID·부모·등록 SHA를 유지했다. 옛 Checkpoint/config의 절대 경로를
보존하기 위해 이전 위치에는 호환 링크가 남는다. 모델의 실제 파일은 act/minivla에 있고 smolvla 저장 폴더도 준비했다.
기존 평가 report1개의 Workspace metadata 불일치는 이동 전 레코드에서도 재현되어 임의 수정하지 않았다.
검증은 `outputs/workflow_validation/artifact_input_layout/REPORT.md`를 따른다.

앞의 명시적인 정리 작업 외에는 기존 결과를 자동으로 이동/수정하지 않는다. 이름 없는 Registry 항목은 읽을 때 경로 기반 이름을 보여주며
ID/Path/SHA256/Parent는 그대로 사용한다. `workflow.py run|ready --storage-name 이름`,
`register --display-name 이름`도 같은 경계를 사용한다. 명시적 `worker --output` 경로는 그대로 유지한다.

2026-10-19부터 MuJoCo worker의 첫 화면 `rgb.png`, `sim_overview.png`, `sim_side.png`, `sim_wrist.png`
저장 코드는 제거했다. 학습용 Side/Wrist MP4와 Preview가 있는 단계의 runs/presentation JPG는 유지한다.
Annotation은 진행 상태만 전송하므로 미리보기 JPG를 생성하지 않는다.
이미 존재하는 사용자 PNG는 삭제하지 않는다. MuJoCo의 생성된 `scene/`, `scene.xml`, `so101.xml`,
`scene_metadata.json` 저장 코드도 제거했다. `mjcf.build_scene`은 XML 문자열과 include bytes를 반환하고
`MjModel.from_xml_string`이 메모리에서 컴파일한다. Source/Replay/Annotation/DataGen/Sim 평가와 Runtime
미리보기 모두 이 Backend를 사용한다. 임시 XML 폴더도 만들지 않는다. Workspace의 원래 자산을 공유해 읽으며
데이터 내부의 씬 설정·모델 hash는 유지한다. 같은 XML 직렬화를 써서 기존 model hash/Resume 계약을 보존한다.
기존 결과의 scene 폴더는 자동 삭제하지 않는다. CLI/직접 API의 명시적 데이터 output 계약과 Source 자동 이름은 유지하며
GUI Source만 필수 이름과 관리 경로를 사용한다.

## 통합 Episode 계약 — 2026-10-16

새 Workspace Source / Annotation / DataGen은 두 엔진 모두 `so101.common-episode/2`를 사용한다.
숫자·주석·메타데이터는 HDF5, 영상은 상대 경로로 연결한 Side/Wrist **640×480 H.264 MP4**다.
Backend는 `data.attrs.env_args.backend`에 기록하며, 좌표·모델 계약은 원본을 상속한다.
기존 Canonical NPY+MP4와 Isaac HDF5는 계속 읽는다. Workflow가 필요한 Legacy 입력을 새 결과 폴더에
자동 Import하므로 엔진을 바꾸기 위해 사용자가 파일을 수동 변환할 필요가 없다.

```text
기존 NPY/Isaac HDF5 ── 읽기 전용 Import ─┐
새 MuJoCo/Isaac Source ───────────────┼─ 공통 HDF5 + MP4
                                     ├─ Source Partition → Annotation
                                     └─ 공식 Mimic 공통 코드 → 물리 Adapter → 중앙 Writer
                                                          ↓
                           Curation → 기존 LeRobot Converter → ACT / MiniVLA → 양 엔진 평가
```

`workflow.dataset.EpisodeDataset` / `DatasetWriter` / `validate_dataset`가 계약을 담당한다.
Curation의 기존 `EpisodeReader`는 같은 HDF5 키와 external video resolver를 읽으며,
전체 RGB를 RAM에 적재하지 않는다. 영상·숫자 복제는 Import/선택 저장/새 결과 작성 시에만 수행한다.

### DataGen IK 선택

기존 SettingsForm의 DataGen `IK Solver`에서 `scipy_ls`(기본) / `pinocchio_dls`를 선택한다.
MuJoCo와 Isaac 모두 같은 옵션을 사용한다. UI 드롭다운 자체가 `options.ik_solver`의 편집기이며
별도의 JSON 덮어쓰기는 없다. 실행 중/완료 상태에도 선택값이 표시된다.
공통 CLI `workflow.py worker --backend mujoco|isaac --stage datagen ... --ik-solver pinocchio_dls`를
사용할 수 있다. `--ik-solver`가 `--options` JSON보다 우선하며, immutable `--request` 사용 시에는
request의 options 안에 설정해야 한다. Workspace Isaac CLI는 `--ik_solver`를 받는다.
잘못된 이름은 엔진 초기화 전에 거부한다. Source를 자동 물리 재생하는 단계는 추가하지 않는다.

`Mimic → Adapter → simulation.ik.solve_selected → 기존 solve_pose / SO101Kinematics.inverse_kinematics`
순으로 실행한다. 기존 두 계산 본문과 공식 Mimic은 변경하지 않는다. DLS는 원래 Isaac 설정
(position/orientation weight 1/.01, 200 iterations, tolerance .001, damping .001, step .3)이다.
MuJoCo world 목표를 기록된 robot base로 변환하고, 기록된 simulation model의 URDF를 사용한다.
관절 영점은 URDF/MJCF에 이미 반영되므로 별도 보정을 더하지 않는다. Pinocchio Model/Data는 worker별 소유다.

SciPy는 이전 명령, DLS는 측정 관절에서 시작한다. 별도 `IK Failure Policy`는 기본 `strict`이며
`hold` / `best_effort`를 선택할 수 있다. Strict는 기존 SciPy 3 mm / .12 rad 탈락, DLS native 미수렴 hold를
보존한다. Hold는 native 미수렴 또는 pose 오차 초과 시 측정 팔 각도를 유지하고 다음 목표로 진행한다.
Best Effort는 수렴 여부와 무관하게 유효한 candidate를 적용하고 실제 grasp/place로 최종 성공을 판단한다.
SciPy의 `reject_on_error=False`는 candidate 반환만 허용하며 least_squares 수치 알고리즘은 그대로다.

공통 candidate의 5축 순서·finite·관절 제한을 검사하고 gripper는 원래 명령을 유지한다.
Hold/Best Effort는 이전 팔 명령 대비 어느 관절이든 1 step에 0.5 rad를 넘으면 attempt를 중단한다.
이 값은 급격한 명령을 검출하는 고정 기준이며 Strict에서는 기존 동작 보존을 위해 진단만 기록한다.
잘못된 candidate나 제한 밖의 measured hold는 적용하지 않는다. clipping/solver fallback/제한 완화는 없다.
CLI는 `--ik-failure-policy best_effort`, Workspace Isaac CLI는 `--ik_failure_policy best_effort`다.
UI → options → worker → Adapter.configure_ik → solve_selected → 기존 physics 경로를 사용한다.

`generation_configuration`, report/attempts와 episode attrs에 solver/config 및 `ik_failure_policy`,
`ik_policy_configuration`을 기록한다. optional `ik/*`에는 `ik_converged`, candidate 위치(mm)/방향(deg) 오차,
`ik_candidate_applied`, `ik_held`, `ik_invalid_candidate`, 시간(ms)과 실패 이유를 저장한다.
Native 수렴과 `common_pose_passed`는 독립이다. `ik_unconverged_applied` 집계는 native 실패 **또는** pose gate
실패 후보의 적용 수다. invalid candidate와 invalid hold command도 별도 집계한다. 실패 시도는 attempts에 남는다.
`ik_motion_evidence`는 시도당 최대 8개의 실패 시점 candidate/실제 명령/물리 전후 측정을 기록한다.
물리 추종 오차와 candidate 오차는 구분하며 기존 grasp/place 성공 조건을 바꾸지 않는다.
Resume은 solver·policy·설정 불일치를 복사 전에 거부한다. 옛 생략값은 SciPy/Strict로 해석한다.
실제 비교 및 변환 검사는 `outputs/workflow_validation/ik_failure_policy/REPORT.md`에 있다.

### 키·단위·시점

각 episode는 `data/demo_N`이고 `data.attrs.env_args`와 episode attributes는 JSON metadata다.
`schema_contract`에서 JOINTS 순서, rad, robot_base, quaternion 순서, action semantics를 검증한다.

| 경로 | 의미 / shape / 단위 |
|---|---|
| `obs/policy/joint_pos` | 명령 전 측정 관절 `[T,6]`; JOINTS 순서, gripper 포함 model rad |
| `joint_targets` | 관측 t 직후 적용한 절대 관절 목표 `[T,6]`, model rad |
| `actions` | `[T,7]`: 이전 명령 EE 목표에서 현재 목표까지 base-frame translation m + 좌측 rotation delta rotvec rad + raw gripper rad. 첫 delta는 초기 측정 EE 기준. 정책 학습의 action은 `joint_targets` |
| `obs/datagen_info/eef_pose/tool0` | 측정 EE `[T,4,4]`, robot_base SE(3), m |
| `obs/datagen_info/target_eef_pose/tool0` | 제어 EE 목표 `[T,4,4]`, robot_base. 관절 전용 Legacy 입력은 기록된 목표의 FK로 복원; 다음 측정 상태를 목표로 대신 쓰지 않음 |
| `obs/datagen_info/object_pose/<id>` | 명령 전 동적/목표 물체의 base-frame SE(3) |
| `obs/datagen_info/subtask_term_signals/grasp` | `[T,1]`, 0→1의 단일 증가 경계. Source/실패 평가에는 없을 수 있고 Mimic 입력에는 실제 경계가 필수 |
| `obs/events/grasp`, `obs/events/place` | 관측한 이벤트; 최종 action 뒤 배치 성공은 마지막 action frame에 표시하고 annotation metadata에 명시 |
| `obs/timestamp`, `action_timestamp` | 같은 제어 tick의 observation/action 초. 관측을 먼저 읽고 명령을 적용. 생성/평가는 episode-relative, Legacy Source의 시작 offset은 보존 |
| `obs/camera_timestamps/{side,wrist}` | 대응 observation timestamp와 동일. MP4 PTS는 `frame/fps`, 원본 control Hz 유지 |
| `obs/object_poses/<id>` | world-frame xyz+wxyz `[T,7]`, Curation telemetry |
| `initial_state/{articulation,rigid_object}` | 초기 root pose xyz+wxyz/velocity와 robot joint position/velocity. Legacy 미기록 velocity는 0 복원임을 표시하고 물리 replay로 별도 검증 |
| `legacy/*` | Import 원본 숫자 배열을 dtype/값 그대로 보존. Mimic pose만 공식 런타임에 맞춘 float32; 원본은 여기 남음 |
| `mimic_plan/<subtask>/*` | IK 전 목표 EE pose / gripper / noise, 선택 원본·경계는 episode의 `mimic_source_segments` |
| `generation/attempts_json` (root) | 압축된 시도 이력. 새 생성 파일을 이동해도 Resume의 누적 시도/실패 통계 유지 |

`initial_state`는 각 episode의 첫 관측과 일치해야 한다. action/gripper/pose/clock/MP4의 frame 수,
실제 H.264 codec, 모든 packet PTS, checksum과 상대 경로를 검사한다. numeric/video를 함께 복사·이동한다.
`external_videos`에는 파일 상대경로·shape·FPS·frame 수·codec·SHA256이 있으며, 파일 부모 밖을 가리키는
참조는 읽기에서 거절한다. 새 저장/Import의 입력 원본 경로만 명시적으로 허용한 뒤 결과 내부에 복사한다.

Workspace/Task/active objects/scene/reset, Coordinate/Calibration/robot model, seed/engine version은
`env_args`에 보존한다. episode에는 `source_group`, `source_lineage`, success와 판정 근거/실패 이유,
annotation을 기록한다. Source/operator 저장 라벨과 물리 검증 성공은 구분한다.

### 공통 Mimic / 병렬 실행

`workflow.mimic_reference`는 설치된 Isaac Lab Mimic **1.0.15**에서 이식했다. 다섯 생성 모듈의
실행 본문은 그대로이고 import 경로만 바꿨다. 라이선스/원본 hash는 해당 폴더의 NOTICE/UPSTREAM.json에 있다.
공식 HDF5 Pool의 group 순서(`demo_0, demo_1, demo_10…`)도 유지한다. UI 표시 순서는 별도로 숫자 순서다.
SO101 설정은 object nearest-neighbor `nn_k=3`, grasp end offset `(0,10)`, 마지막 `(0,0)`,
interpolation5 / fixed0 / noise0, subtask별/arm별 원본 재선택 false, 마지막 command에서 보간이다.
SO101 단일 arm 두 subtask를 지원하며, SkillGen·multi-arm coordination은 이 adapter 범위에 포함하지 않는다.

`workflow.adapters`만 FK/IK/상태/접촉/이미지/물리 엔진에 의존한다. 공통 bounded 5축 IK는
위치3mm·회전0.12rad를 각각 검사하고 seed/관절 한계/목표 frame/두 잔차를 기록한다.
6D 목표의 모든 자세가 5축 로봇에서 도달 가능한 것은 아니다. IK·계획 탈락, 물리 실행 오류,
집기·배치 실패, 시간 초과, 성공을 분리한다. 허용 오차를 늘리거나 탈락을 성공으로 바꾸지 않는다.

Isaac은 한 batch clock에서 환경별 coroutine/action/reset mailbox를 사용한다. episode 종료와 reset은
각 환경이 독립적으로 수행한다. reset이 다른 환경의 q/rigid state를 바꾸지 않았는지 실제 검사한다.
Kit의 별도 asyncio loop 때문에 physics/reset은 generator loop 바깥에서 실행한다.
MuJoCo는 `MjModel` **한 개**와 worker별 `MjData`/FK scratch/IK scratch를 사용한다. ThreadPool worker가
독립 실행하며 GL 생성·사용·종료는 소유 thread에서, render/GL 호출만 lock으로 직렬화한다.
MP4는 카메라별 bounded8frame encoder queue로 흘린다. physics/IK를 렌더 lock으로 감싸지 않는다.

Source Pool은 worker 초기화 때 한 번 로드하고 읽기 전용으로 재사용한다. Source 선택·경계·RNG·trace는 매 시도의 Generator가 소유한다.
`GenerationScheduler`가 상태 lock에서 성공 슬롯을 예약하고 별도 Writer lock에서 완료 episode만 commit한다.
commit 실패는 예약을 취소하며 성공 수를 늘리지 않는다. 저장 중에도 다른 환경의 상태 조회·claim은 계속된다.
먼저 끝난 환경은 즉시 다음 시도를 가져간다. 목표를 넘는 완료는 추가 등록하지 않는다.
중지 시 완료 데이터를 보존하며 미완성 영상은 버린다. Resume은 Source Pool byte hash·좌표·seed·엔진을
검사하고 새 출력으로 이어 쓴다. max_attempts는 재개 전 시도까지 포함한 전체 한도다.
성공률은 성공/(성공+분류된 실패), 목표 도달로 중단한 다른 환경과 진행 중 시도는 제외한다.
새 HDF5의 시도 이력이 없거나 모순되면 재개를 거절한다. 구형 파일의 이력이 없으면 누적 성공률은 UNKNOWN이다.

새 Source는 양 엔진 모두 처음부터 공통 Writer에 저장한다. Isaac의 `EpisodeData`는 메모리상의 native API 경계이며
중간 Legacy HDF5를 만들지 않는다. finalized `VideoEpisode`의 private staging은 검증 후 UUID 영상 폴더로 이동한다.
기존 입력·Resume 영상은 복사한다. 두 경로 모두 해상도/FPS/H264/전체 PTS/SHA 검사와 원자적 publish를 유지한다.

### 다른 엔진의 입력 / 계보

DataGen은 입력 엔진과 관계없이 모델·단위·frame·초기 q/object·시간·주석 계약을 검사한 뒤
환경을 준비하고 곧바로 선택한 환경 수로 병렬 생성한다. Source 전체를 자동으로 물리 재생하지 않으며,
교차 Replay 성공이나 캐시 유무를 생성 시작 조건으로 사용하지 않는다. 계약 검사 통과는 물리 성공을 뜻하지 않는다.
새 생성 episode의 성공 여부는 해당 실행에서 측정한 집기·배치 조건으로 계속 판정한다.

명시적으로 선택한 Replay/Annotation은 **기록 목표 관절을 실제 재생**하고 집기와 최종 배치를
`cross_replay.json`에 남긴다. Annotation은 통과한 episode만 별도 결과에 저장하고 제외 이유/개수를
함께 반환한다. 원본을 교체하지 않는다. 개발용 교차 재생·positive proof 캐시 helper는 별도 검사에만 사용한다.

자동 Annotation의 실행 구현은 `workflow/annotation.py:run`이다. 기존 `unified.replay` 인터페이스는 유지하되
에피소드 복원·명령·접촉·판정·주석 구성은 `annotation.replay_episode` 한 구현을 Replay와 공유한다.
MuJoCo는 한 model/독립 MjData worker, Isaac은 하나의 vector scene을 사용하며 빈 슬롯에 다음 Source를 즉시
배정한다. 에피소드 묶음이 끝날 때까지 기다리지 않는다. 완료한 숫자/영상은 Source 순서로 단일 Writer에 저장해
후속 Mimic Pool 순서를 유지한다. UI의 기존 실행 설정에서 동시 주석 환경 수1/2/4/8을 선택한다.
수동 mark/skip의 대상은 기존 Env0이며 수동 모드는1환경만 허용한다. 여러 환경을 조용히1개로 바꾸지 않는다.

Annotation 화면은 양 엔진 모두 `view.AnnotationProgress`를 사용한다. 카메라 영역 대신 진행률,
처리 완료/저장 성공/실패/저장 대기, 환경별 Source·frame·결과, 실패 유형, 경과 시간을 표시한다.
경과 시간은 환경 준비와 일시정지를 포함하며 완료 시 고정한다. 작업 상태는 `annotation.run`의
`progress_snapshot` → 기존 Presentation 상태 IPC → `annotation_status_text`로 전달한다.
순서 보존 Writer를 기다리는 완료 episode는 저장 성공에 미리 더하지 않는다.
Annotation 전용 preview 큐·RGB 수집·JPEG publish와 Isaac Overview sensor 생성은 제거했다.
결과 MP4에 필요한 Side/Wrist 렌더/인코딩·접촉 Physics와 수동 mark/skip/pause는 유지한다.
수동 Developer 모드의 기존 native 창과 일시정지 중 창 갱신도 유지한다. 다른 단계의 CameraView/
Presentation.publish/attach와 Preview는 그대로 사용한다. 실제 비교·화면은
`outputs/workflow_validation/annotation_job_view/REPORT.md`에 있다.

Annotation은 realtime 옵션과 관계없이 실행 중 제어 주기의 sleep을 하지 않는다. pause/step은 유지한다.
각 환경은 원래 joint_targets를 정확히 한 번씩, 원래 dt/decimation으로 적용한다. 입력 계약·전체 MP4 PTS/SHA는
실행 전에 한 번 검사하며 입력 RGB 프레임은 디코딩하지 않는다. 기록은 실제 재생 관측의 Side/Wrist를 각각 한 번
H.264로 인코딩해 private staging을 이동한다. 다른 엔진에서 재생한 관측을 입력 MP4로 대체하지 않는다.
`report.json/cross_replay.json`의 annotation_profile은 입력 읽기/검사, 물리, 렌더, 인코딩/대기, 저장을 구분한다.
worker 합계·background 인코딩은 wall과 겹치므로 더하지 않는다. assignments에는 슬롯별 Source 시작/완료가 남는다.
Isaac physics_step_s는 SimulationContext.step의 렌더 제외 시간이고 physics_s는 native control-step의
렌더 제외 시간으로 observation manager 등을 포함한다. physics_substeps/batch_steps로 decimation도 확인한다.
Isaac의 환경 수를 바꾸면 PhysX의 측정 궤적·배치 이벤트 frame에 수치 차이가 생길 수 있다. 판정식/시간축을
변경하거나 기존 경계로 강제 보정하지 않으며, 실제 비교 오차는 annotation_throughput 보고서에 공개한다.

Isaac vector scene 생성 직후 전체 환경을 reset해 Task 초기 관절/물체 상태를 적용한다.
명시 Replay에서 대기 로봇은 최초 hold 목표를 유지한다. 매 tick 측정 관절을 새 목표로 덮어쓰지 않는다.
MuJoCo는 환경별 MjData를 따로 step하지만 Isaac은 대기 환경까지 공통 physics step을 진행하므로 이 구분이 필요하다.
DataGen/Sim 평가 영상은 기존 Overall 및 Env 0의 Side/Wrist 세 뷰를 유지한다.
Source/Replay는 기존 Side/Wrist, Annotation은 위의 진행 상태 화면을 사용한다.

Train14/Val6는 새 split을 만들지 않고 원래 partition/Source ID를 상속한다. 생성 episode는
각 subtask의 실제 Source ID·경계를 `mimic_segments`로 기록한다. 미분할 생성 자료를 학습 분리할 때는
A+B / A+C처럼 원본을 공유한 조합을 연결 성분으로 묶어 같은 split에 놓는다. 추적이 UNKNOWN이면
독립 split을 허용하지 않는다. 원래 Partition 파일과 원본 데이터는 수정하지 않는다.

### UI / CLI / 학습 / 평가

Workflow에서 입력과 실행 Backend를 선택하고 성공 데이터 수, 환경 수1/2/4/8, 생성 seed(미지정=Auto),
전체 최대 시도 수를 설정한다. 상세 설명은 기존 설명 버튼에 있다. 실행 중 환경별 상태와 실패 종류,
짧은 IK 오차를 보여준다. 원본 로그와 `attempts.json`에는 전체 진단을 보존한다.

두 엔진의 DataGen 진행 UI는 같은 `Presentation`의 Scheduler provider와 `datagen_status_text`를 사용한다.
환경 준비/생성을 표시하며, 전체 성공/목표·시도·실패·성공률과 환경별 상태/시도/성공/실패/step,
최근 결과의 환경 번호를 표시한다. DataGen에는 Source 사전 재생 단계가 없다.
실패 원인은 전체/환경별 집기 실패·이동 중 놓침·컵 배치 실패 및 IK/계획/저장 오류로 표시한다.
놓침은 안정 집기 확인 뒤 release 명령 전에 양 접촉 소실+상대 위치6mm 이탈이0.2초 지속된 진단이다.
기존 성공 조건은 그대로이며 attempts의 failure_diagnostics로 판정 증거를 확인한다.
생성 경과는 wall_seconds, 목표 저장 완료 시간은 마지막 성공 commit 시 고정한 target_completed_seconds다.
분·초로 표시하며 환경 준비 제외/일시정지 포함, Resume이면 이번 실행 구간만 표시한다.
종료/일시정지에서도 provider의 최신 snapshot을 반영한다. 성공0개에서 사용자 중지는 STOPPED이며 빈 artifact를 등록하지 않는다.

```bash
# 필요할 때만 명시 Import. 기존 입력은 읽기 전용이며 새 경로가 필요하다.
PYTHONPATH=source/soarm101_lab python scripts/real2sim/workflow.py migrate /path/to/legacy /path/to/new.hdf5
# 기존 worker JSON 경로: backend, stage, workspace, input, output, options.
PYTHONPATH=source/soarm101_lab python scripts/real2sim/workflow.py worker --request request.json
# 기존 Isaac CLI도 Workspace일 때 같은 공통 core를 사용한다. --output_file 경로는 그대로 보존한다.
PYTHONPATH=source/soarm101_lab python scripts/mimic/generate_dataset.py --workspace /path/to/workspace \
  --input_file /path/to/annotated.hdf5 --output_file /path/to/generated.hdf5 \
  --num_envs 4 --generation_num_trials 100 --max_attempts 10000 --headless
```

기존 LeRobot Converter 하나가 양 엔진의 HDF5를 처리한다. `joint_targets` model rad/JOINTS6와
Source lineage/Partition을 보존하고 FPS를 변경하지 않는다. 현재 LeRobot은 정수 FPS만 지원하므로
분수 FPS를 임의 반올림/재샘플링하지 않고 거절한다. LeRobot 저장 codec은 기존 라이브러리 설정을 유지한다.
ACT/MiniVLA는 기존 관측/6D action/정규화/체크포인트를 사용한다. MiniVLA의 현재 고정 모델은 Side 하나를
입력으로 쓰며 저장 영상은 Side/Wrist 모두 남는다. 별도 Val 입력과 train-only statistics를 유지한다.

두 엔진의 Workspace Sim 평가에는 `so101.evaluation-episode/1` summary와 공통 HDF5+두 MP4를 추가한다.
초기 물체 배치, checkpoint, grasp/place, timeout/controller/IK, steps/duration/failure reason 및 실제
state/action trace를 보존한다. 기존 배치 성공 조건과 안정 집기 측정은 별도 필드다. 미완성 평가는
정식 episode로 저장하지 않는다. Legacy 단일 camera/review scene과 실제 로봇 평가 경로는 유지한다.
검증 보고서: `outputs/workflow_validation/unified_mimic/REPORT.md`.

### 병렬 Sim 평가와 성공/실패 검토 — 2026-10-18

Sim 평가 실행 설정의 `num_envs`는 1/2/4/8이며, `episodes`는 **Task당 모든 환경을 합친 전체 평가 횟수**다.
예를 들어 100회/환경4는 총100회다. DataGen처럼 성공 개수를 채우기 위해 실패를 재시도하지 않는다.
MuJoCo `mujoco.execute(stage='rollout')`와 Isaac `evaluation.evaluate(stage='sim_eval')`은
공통 `evaluation.evaluate_sim`→`policy.batch_actions`→기존 backend/Adapter→`EvaluationRecorder`를 사용한다.
MuJoCo는 한 MjModel/독립 MjData를 thread로 step하고, Isaac은 vector scene의 action batch를 step한다.
정책은 한 번 로드한다. ACT/SmolVLA는 공식 batch queue를 사용하고 MiniVLA는 기존 predict를 공유하면서
환경별 chunk/ActionGuard를 분리한다. 같은 batch의 완료 환경은 목표 자세를 유지하고 다음 batch에서 함께 reset한다.
MiniVLA의 모델·추론 계산, 충돌·물성·성공 조건을 변경하거나 속도 향상을 보장하지 않는다.

```text
outputs/workflow/evaluations/<act 또는 smolvla 또는 minivla>/<학습 실행>/<평가 실행>/
  report.json                         # MuJoCo 실행 보고서, Isaac도 동일 요약 생성
  result.json                         # Isaac 기존 file-output 계약
  evaluation_datasets.json            # 결과/성공·실패 HDF5 상대 경로 목록
  episode_0000.json ...               # 각 평가 요약과 시점별 object metrics
  success/dataset.hdf5 + dataset.videos/
  failure/dataset.hdf5 + dataset.videos/
  scene/                              # MuJoCo 실제 씬
```

여러 Task를 평가하면 Task별 하위 `tasks/<task_id>/`에 scene과 success/failure를 둔다.
한쪽 결과가0개이면 빈 HDF5를 만들지 않는다. 중단/예외 시 완료된 episode만 보존하며 진행 중 episode는 버린다.
평가의 기존 성공 조건은 컵 안 배치/정지다. 집기 측정과 별도로 유지한다. 실패는 기존 Adapter 진단으로
집기 실패/이동 중 놓침/컵 배치 실패를 구분하고, timeout 및 높이·컵 벽·정지 판정도 따로 기록한다.
영상 위에는 전체 완료·성공·실패·실패 유형·환경별 집계·시간을 표시한다.
UI에서 시작한 시간은 Workflow 실행 시작부터 계산하여 환경 준비와 일시정지를 포함한다.

완료한 `sim_eval` Artifact 아래 HDF5를 `evaluation_dataset`으로 별도 등록한다.
Replay에서 이 입력을 선택할 수 있고, 실패 기록의 재생 완료를 task 성공으로 바꾸지 않는다.
MuJoCo 기록은 native 초기 integration state도 남겨 같은 모델 재생에 사용한다.
Data Curation의 데이터 열기에서 **평가 실행 폴더**를 선택하면 성공/실패를 함께 읽어 초록/빨강 분포를 비교한다.
성공 라벨·실패 유형·환경 필터, Side/Wrist 영상 재생과 EE/물체 경로를 기존 reader로 확인한다.
개별 HDF5도 열 수 있다. 평가는 읽기 전용 검토이며 Source Train/Val 분할·학습 입력으로 자동 전환하지 않는다.
예전 평가의 에피소드별 HDF5 폴더도 schema를 읽어 합쳐 열며 원본 파일을 재작성하지 않는다.
기존 단일 카메라/다른 해상도 환경은 예전처럼 요약만 기록하며, 상세 HDF5 녹화 불가 이유를 결과에 명시한다.

실제 검증/제한: `outputs/workflow_validation/parallel_sim_eval/REPORT.md`.

## MiniVLA Train/Val과 학습 기록 — 2026-10-17

Registry의 Train/Validation LeRobot 입력→`episodes.validate_training_inputs`(메타데이터 사전검사)→
`mujoco.train_from_lerobot`(기존 Parquet/MP4를 읽는 작은 index)→train-only 통계→기존 MiniVLA trainer로 이어진다.
검증된 Source Partition/Lineage를 최우선 사용하며 Train 전용 데이터를 다시 나누지 않는다. 일반 UI에
`source_groups` 입력은 없고 기록과 충돌하는 매핑은 거절한다. Partition 없는 Legacy는 검증된 독립 Group과
고급 호환 옵션이 있어야 기존 `split_episodes`를 사용할 수 있다. Val 준비 경로와 UI 사용법은
[MiniVLA 통합](MINIVLA_INTEGRATION.md#minivla-학습-화면과-체크포인트--2026-10-17)을 참조한다.

일정 설정은 체크된 UI 값→사용자 JSON→기본 JSON 순서다. MiniVLA UI 초기값50k/검증1k/저장2.5k,
Early Stopping OFF이며 기존 ACT/SmolVLA 설정과 분리한다. 매 step train loss, 검증 시에만 val loss를 기록한다.
검증은 episode당 최대8개의 고정 시점 평균을 episode별 동일 가중 평균한다. 실제 표본 수는
`validation_protocol.json`에 남긴다. 저장 주기는 검증과 독립이며 번호별/best/last를 보존한다.
Resume은 원본 모델·통계·양쪽 dataset identity를 검사하고 새 출력에 저장한다. Sim 평가의 명시적
체크포인트 파일 선택과 실제 step을 두 엔진의 결과에 기록한다. 모델·손실·전처리·Mimic 본문은 유지한다.

## MuJoCo 손가락 collision — 2026-10-18

기본 `isaac_matched`의 `simulation.mjcf.robot_xml`은 gripper/jaw의 오목한 collision STL을
단일 convex hull 대신 정적 `assets/SO101/mujoco/finger_convex_parts.json`의 convex 조각으로 구성한다.
원래 URDF origin/scale과 접촉 mask를 사용하며 visual STL, cube collision, 질량/관성, 관절 범위,
구동기 gain/force와 마찰은 유지한다. Source/DataGen/Sim 평가는 기존 공통 compiler로 연결된다.
명시적 `legacy_urdf`는 이전 collision을 유지하고 Isaac의 collision 구성은 바꾸지 않는다.

`scripts/mujoco/export_finger_collisions.py`는 자산 재생성용 오프라인 도구다. 실행 worker는 CoACD를
import하지 않는다. STL SHA가 export와 다르면 재생성을 요구하며, export SHA도 기존 scene/model SHA에
포함되어 Replay 증거 캐시가 새 물리 모델을 구분한다. 기존 데이터/Checkpoint는 재작성하지 않으므로
이전 collision에서 생성·학습한 궤적의 새 물리 결과는 달라질 수 있다. 접촉 표본 검사와 정책 결과의 범위는
[교정 보고서](../../outputs/workflow_validation/gripper_contact_fix/REPORT.md)를 따른다.

## Real 평가의 연결 설정

`WorkflowPanel.real_evaluation_defaults`가 현재 연결 입력 또는 기존
`outputs/real2sim/ui_settings.json`에서 Follower 포트·보정 ID·보정 폴더와 Side/Wrist 장치를 읽어
Real 평가 설정에 채운다. 사용자가 수정한 값은 단계 이동 중 유지한다. 이것은 장치를 여는 동작이 아니다.
`Real 평가 시작` 버튼으로 실행할 UI 요청은 `stage_options`에서 내부 `enable_motion=true`를 지정하며,
별도의 모터 체크는 없다. CLI/직접 Worker 요청의 명시 opt-in과 기존 좌표·보정·freshness·range 검사는 유지한다.

`evaluation.evaluate`는 Follower 연결 후 모터 활성화 전에 `real_start_pose`로 측정 native/raw를 model-rad로
변환하고 명령 역변환 가능성을 float64로 검사한다. 정책의 float32 관측은 기존대로 사용하지만 encoder
왕복 검증에는 사용하지 않는다. 실행 폴더의 `hardware_preflight.json`은 활성화 전 측정값·통과/실패를 기록한다.
실제 보정 범위 밖의 자세는 계속 거부하며 Follower 자세를 고친 뒤 명시적으로 다시 시작해야 한다.

일반 두 카메라는 `side_device`/`wrist_device`를 사용한다. 추가 카메라 표 `camera_devices`는
Workspace의 카메라 이름을 실제 장치 경로/인덱스에 연결하며, 같은 이름의 개별 장치 입력보다 우선한다.
예를 들어 `front` → `/dev/cam_side`는 모델의 front 관측에 해당 USB 카메라 영상을 제공한다.
표의 이름은 모델/Workspace의 관측 이름과 같아야 한다. 자동으로 카메라를 탐색하거나 이름을 바꾸지는 않는다.

## 변환된 LeRobot 데이터셋 병합

`LeRobot 변환` 화면의 **변환된 LeRobot 데이터셋 병합**에서 변환본을 2개 이상 체크하고 새 이름으로
저장한다. 목록에 없으면 선택 창의 **데이터셋 폴더 추가**로 `meta/info.json`이 있는 폴더를 등록한다.
HDF5를 먼저 합칠 필요는 없다. 목록에 표시된 순서와 각 입력의 episode 순서를 유지한다.

```text
WorkflowPanel.select_merge_datasets → DatasetMergeDialog
  → WorkflowPanel.execute(stage='merge') → Operations.plan/run → 기존 workflow.py worker
  → lerobot_merge.merge → 공식 LeRobot aggregate_datasets(roots=로컬 입력)
  → SO-101 좌표·episode 계보 저장 → 새 dataset Registry 등록 (모든 입력이 parents)
```

기본 결과는 `outputs/workflow/datasets/merged/<병합 이름>/`이다. 이력·요청·로그는 기존 `runs`에 남는다.
결과는 Parquet/MP4와 재집계된 통계, `so101_coordinates.json`, `workflow_provenance.json`,
`merge_report.json`을 포함한다. Hub 업로드 없이 원본을 보존한다. 실패·중단 시 이번 병합의 private
staging을 제거하고 미완성 결과를 등록하지 않는다. 학습 단계에서 결과 폴더를 입력으로 고르거나
`datasets/merged` 묶음에서 선택한다. 기존에 범위를 좁혀 둔 학습 목록은 그 범위를 유지한다.

현재 지원 범위는 원본 Source 계보가 기록된 LeRobot v3 변환본이다. **미분할끼리 / Train끼리 / Val끼리**
병합한다. 미분할은 `source_partition=None`을 보존하며 Train/Val 역할을 자동 부여하지 않는다.
미분할과 명시적 Train/Val의 혼합은 거부한다. Workspace·엔진·좌표/Gripper·로봇/충돌 모델·카메라
feature·FPS·robot_type이 같아야 한다. 미분할도 각 episode의 원본 Source/그룹 식별을 확인한다.
다른 Source 집합은 원본 inventory/Train·Val 배정의 충돌과 group 누수를 검사한 뒤 합친다.
`episode_source_partitions`/`episode_source_partition_inputs`가 각 episode의 원래 증거를 보존한다.
`episodes.partition_evidence`를 준비 검사와 MiniVLA index에서 공통 사용하므로 원래 lineage hash를
새 partition hash로 덮어쓰지 않는다. 기존 변환본은 이 필드가 없으면 이전 top-level 증거를 사용한다.
원본과 이미 그 원본을 포함한 병합본의 중복 선택은 거부한다. frame 내용의 유사성을 판정하거나 자동
중복 제거하지는 않는다. 서로 다른 Task는 같은 ID의 정의가 충돌하지 않으면 함께 저장한다.
색상별 Task 선택은 물리 모델 불일치로 취급하지 않는다. Scene의 Task/pick/target 선택을 제외한
active objects·reset·Workspace·모델 설정은 같아야 하며 원래 모델/Scene hash도 검사한다.
Task 선택이 다른 병합본의 공통 모델은 `scene_configuration=None`과 그 모델의 새 hash를 저장한다.
episode별 Scene과 `merge_inputs.coordinates`에는 원래 Task·좌표 계약을 보존한다. 이미 병합한 결과를
추가 병합할 때도 같은 검사를 적용한다. MiniVLA의 기존 Train/Val 사전 검사는 변경하지 않는다.

설치된 LeRobot 0.4.1에 공식 merge CLI/API가 있지만 일반 병합은 SO-101 sidecar를 처리하지 않는다.
Workflow에서는 공식 `aggregate_datasets`의 로컬 `roots`를 사용하고 위 계약을 추가로 유지한다.
기존 CLI로도 `run --stage merge --artifact <첫 ID> --options <JSON>`을 사용한다. JSON 예:
`{"merge_datasets":["첫 ID","둘째 ID"],"storage_name":"combined_train"}`.
`repo_id`는 생략하면 `local/so101_merged`이며 업로드 동작은 없다.

## 관련 계약과 이력

- 실시간 관절: [JOINT_CONTROL.md](JOINT_CONTROL.md), 보정: [PRECISION_CALIBRATION.md](PRECISION_CALIBRATION.md).
- Snapshot: [CAPTURE_DATASET.md](CAPTURE_DATASET.md), Builder 입력: [BUILDER_WORKSPACE.md](BUILDER_WORKSPACE.md).
- 학습/평가: [MINIVLA_INTEGRATION.md](MINIVLA_INTEGRATION.md), 모듈 지도: [ARCHITECTURE.md](ARCHITECTURE.md).
- 통합 이전 NPY/Recorder/운영 설명은 [보존 기록](../archive/real2sim/DATA_FLOW_BEFORE_COMMON.md)이다. 현재 실행 경로의 기준으로 사용하지 않는다.
