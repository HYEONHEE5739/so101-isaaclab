# SO-101 Real2Sim Testing Guide

## Workflow UI 응답 지연 수정 — 2026-10-19

목록 표시는 `Registry.get/all(metadata_only=True)`의 변경 감지 캐시를 사용한다. 전체 get/all/verify는
기존 데이터를 계속 읽으며, UI readiness/execute는 worker에서 전체 Registry record를 다시 읽는다.
MiniVLA Val 후보 검증도 기존 readiness executor에서 실행하고 이전 generation 결과를 버린다.
Workspace는 화면 표시만 재사용하며 실행 전 자산/계보 검증은 생략하지 않는다.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_workflow_ui.py tests/real2sim/test_dataset_collections.py tests/real2sim/test_artifact_storage.py tests/real2sim/test_workflow_view.py tests/real2sim/test_live_scene_selection.py tests/real2sim/test_presentation.py tests/real2sim/test_settings_form.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_source_partitions.py tests/real2sim/test_training_resume.py tests/real2sim/test_common_mimic.py tests/real2sim/test_runtime_liveness.py -q
```

최종 UI206/계약90PASS. 이후 중복 등록과 Val 제거 시 자동 대체 방지를 보강한 부분5PASS이며 겹침은 합산하지 않는다.
초기 suite의 캐시 재사용 예상·get mock keyword·기존 merge/enable_motion schema 기대를 현재 계약에 맞췄다.
빠른 동일 크기 이미지 덮어쓰기의 timestamp tick fixture도 명시적 utime으로 고쳐 새 이미지의 실제 색을 확인했다.

`outputs/workflow_validation/ui_responsiveness/measure_ui.py <새 label>`은 실제 Registry31개/136MiB를
임시 Registry로 복사해 UI만 측정한다. 일반 readiness는 fixture이며 새 물리/장치 worker를 실행하지 않는다.
기존 before/after JSON·profile을 덮어쓰지 말고 새 label을 사용한다. 같은 runner에서 단계 전환2.35–4.86초→
0.062–0.136초, 목록 refresh2.07–2.21초→0.068–0.075초. 각1회/동일 session의 cProfile 포함 측정이다.
`verify_interactions.py`는 실제 Train100/Val30의 일반 readiness도 그대로 검사했고 READY 및 UI heartbeat49회,
최대 간격0.160초를 확인했다. 두 runner는 원 Registry/UI byte 보존을 검사한다. 실물·새 학습·장시간 평가는 미실행이다.

## 데이터 선택 버튼·평평한 이름 저장 — 2026-10-19

기존 두 버튼/처리 함수를 제거하고 `browse_data/select_data`로 교체했다. Source 버전 선택·Source family의
origin/선별/Train·Val 목록·단일 입력 자동 선택·단계별 복원·MiniVLA 별도 Val 연결을 검사한다.
신규 출력은 각 저장소의 `<필수 이름>/`이고 Source만 `<이름>/origin`이다. CLI 명시 경로/Resume·분할 계약은 유지한다.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_workflow_ui.py tests/real2sim/test_artifact_storage.py tests/real2sim/test_dataset_collections.py tests/real2sim/test_presentation.py tests/real2sim/test_live_scene_selection.py tests/real2sim/test_settings_form.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_source_partitions.py tests/real2sim/test_lerobot_merge.py tests/real2sim/test_training_resume.py tests/real2sim/test_parallel_sim_eval.py tests/real2sim/test_common_mimic.py -k 'not test_real_video_merge_worker_registration_and_training_reader' -q
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_lerobot_merge.py::test_real_video_merge_worker_registration_and_training_reader -q
```

각각 UI183/계약115/native reader1개가 통과했다. 이후 Source origin 기본 이름·모델 종류 검사6개와
선택 부분30개도 통과했다. 부분 suite는 겹치므로 합산하지 않는다. 첫 계약 suite의 영상 reader는 Qt를
같은 process에 import한 뒤 GLib symbol 문제로 실패해 native Python에서 분리 통과했다. 의존성 변경 없음.
UI 처음 실패는 origin/source_demo 파일 표시를 origin으로 정리한 최종 로그와 구분한다.

`outputs/workflow_validation/workflow_data_selection/capture_ui.py`는 위 USD/Qt 환경에서 원 Registry metadata만
임시 Registry로 복사하고 실제 자료를 선택한다. 9개 화면·Source 원본/분할 목록·Train100/Val30의 실제 계약,
DataGen 단일 입력 자동 선택을 확인한다. capture의 일반 readiness는 fixture로 대체하며 worker/장치는 시작하지 않는다.
그 snapshot의 입력/옵션은 native 환경의 `verify_runtime.py mujoco|isaac`로 실제 생성/변환/등록을 확인한다.
기존 검증 폴더는 덮어쓰지 않으므로 재실행 시 runner의 새 root 이름을 사용한다.
두 엔진 각2환경 목표1개·414/362frame·LeRobot 변환과 HDF5/MP4 계약·계보·원본/Registry 보존을 검사했다.
모든11stage×2backend의 flat 경로와 삭제한 함수/위젯 검사는 final_audit.json, 화면은 display_audit.json에 있다.
새 정책 학습·실물·실제 업로드/새 Source 수집은 하지 않았다. 이력·표준 모델/영상 내부 폴더는 유지한다.

## 미사용 출력 인수·공통 필수 이름·평평한 Annotation 경로 — 2026-10-19

MujocoBackend/PolicyEnvironment의 미사용 출력 인수를 삭제하고 테스트30개 호출도 Workspace 입력만
전달하도록 고쳤다. UI는 모든10단계의 필수 저장 이름을 전달하며 Annotation 기본 경로는
`annotations/annotation_<Source 이름>/`다. 기존 데이터/분할/계보와 다른 단계 경로는 유지한다.
최종 core147/native IK27/구성32/USD·UI182개, 마지막 UI 부분84개가 통과했다. 겹치는 범위는 합산하지 않는다.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_backend.py tests/real2sim/test_minivla_integration.py tests/real2sim/test_rgb_video.py tests/real2sim/test_parallel_annotation.py tests/real2sim/test_common_mimic.py tests/real2sim/test_parallel_sim_eval.py tests/real2sim/test_runtime_liveness.py -q
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_ik_selection.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_active_objects.py tests/real2sim/test_simulation_calibration.py -q
```

USD/UI는 아래 Annotation 작업 화면의 USD/Qt 환경으로
`test_mujoco_workflow_ui.py test_artifact_storage.py test_dataset_collections.py test_presentation.py test_builder_workspace.py`
를 별도 process에서 실행한다. MuJoCo native와 같은 process에 USD를 import하지 않는다.
처음에는 Pinocchio import와 여러 suite를 합쳐 native segfault, USD/MuJoCo 혼합으로 guard 실패가 발생했다.
위처럼 프로세스를 분리한 최종 로그만 완료 근거다. 의존성을 설치/변경하지 않았다.
구성 검사의 warning1개는 legacy fixture의 미확인 provenance를 그대로 보존한다는 기존 안내다.

실제 실행은 `outputs/workflow_validation/storage_names_cleanup/verify_runtime.py mujoco|isaac`다.
기존 출력이 있으면 거부하므로 재실행 시 runner의 root를 새 검증 이름으로 정한다. 임시 검증 Registry에서
기존 Source1개432frame을 실제 Annotation/저장/등록하고 MP4 계약·JPG0/Scene0·원본 보존을 검사한다.
MuJoCo는 처음 Scene 제거 전 baseline의 model hash·23배열·30step trace도 대조한다.
`capture_ui.py`는 위 USD/Qt 환경에서 기존 artifact metadata를 임시 Registry로 복사해10개 화면을
실행 없이 캡처한다. 필수 이름 row의 동일 위치/편집 가능·사용자 Registry 보존을 검사한다.
결과는 REPORT/final_audit/native_*/result/display_audit.json에 있다. 실제 장치/Hub 업로드/새 학습은 하지 않았다.

## Annotation 작업 화면·미리보기 제거 — 2026-10-19

MuJoCo/Isaac Annotation의 preview 큐·JPEG 발행·GUI 이미지 읽기·Isaac Overview sensor를 제거했다.
상태 전용 진행률/처리 완료/저장/실패/환경·시간과 수동 mark/skip/pause는 기존 IPC를 사용한다.
최종 core107개와 별도 USD/UI70개가 통과했다. scheduling 실패 fixture·1/2/4/8환경·pause/step/resume,
실패 집계·저장 대기·시간 고정, stale JPG를 읽지 않는 UI와 다른 Stage Preview 복원을 검사한다.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_parallel_annotation.py tests/real2sim/test_common_mimic.py tests/real2sim/test_annotation_diagnostics.py tests/real2sim/test_settings_form.py tests/real2sim/test_parallel_sim_eval.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_presentation.py tests/real2sim/test_mujoco_workflow_ui.py -q
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/annotation_job_view/audit.py
```

실행 결과는 `outputs/workflow_validation/annotation_job_view`에 있다. `verify_runtime.py before|after mujoco|isaac`
는 새 출력에 실제 Source4개/2환경을 실행하며 기존 측정 폴더를 덮어쓰지 않는다. `--manual --count 1`은
실제 Source1개에 scripted IPC pause→mark→resume을 보낸다. 재실행하려면 runner의 별도 출력 이름을 정한다.
before snapshot은 검증 전용이며 production 분기나 launcher로 사용하지 않는다. `audit.py`는37배열×4개와
주석/계보 정확 일치, 영상 계약, JPEG0, Isaac sensor·dt·substep, 수동 표시 저장을 검사한다.
`capture_ui.py`는 위 USD/Qt 환경에서 실제 상태 기록을 임시 Registry/세션에 넣어5개 화면을 캡처한다.
사람의 키보드 주석·Developer native 창·실물·여러 seed/반복 성능은 미검증이다.

## Annotation 최대 속도·동적 슬롯 — 2026-10-19

기존 단일환경 구현은 검증 폴더 before_unified.txt에만 보관한다. Production은 공유 episode 상태 머신과
새 Annotation 실행 구현만 사용한다. 다음 회귀는 실제 엔진/저장 검사와 CPU scheduling/신호 fixture를 포함한다.
132개가 통과했다. 별도 SettingsForm/Source Partition/diagnostics26개와 USD Presentation16개는 겹치는 범위가
있으므로 합산하지 않는다. 최초 GUI/USD 경로 누락과 synthetic manual buffer의 events 누락은 최종 로그와 구분한다.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_parallel_annotation.py tests/real2sim/test_common_mimic.py tests/real2sim/test_parallel_sim_eval.py tests/real2sim/test_mujoco_backend.py tests/real2sim/test_source_partitions.py tests/real2sim/test_minivla_integration.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_settings_form.py tests/real2sim/test_annotation_diagnostics.py tests/real2sim/test_source_partitions.py -q
```

`test_presentation.py`는 아래 기존 Source 이름 검사 명령의 USD 환경에서 별도 실행한다. 이전 테스트의
operator/developer 비교는 같은 storage_name을 지정해 원래 무작위 새 이름이 비교를 깨뜨리지 않게 했다.
회귀는1/2/4/8 슬롯의 즉시 재배정·대기 hold·원본 순서 저장·pause/step/resume, 자동 모드의 realtime 대기 없음,
수동 mark·집기/배치 gate와 DataGen/Evaluation/기존 MiniVLA CPU 계약을 검사한다. 8환경은 CPU fixture 범위다.

실제 비교는 기존 Green Source4개(432/351/403/338frame)를 양 엔진에서 이전1환경·개선1/2/4환경으로 실행한다.
GPU process는 순서대로 실행하고 기존 Registry를 사용/수정하지 않는다. 다음은 새 tag를 사용한 재측정 예다.

```bash
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/annotation_throughput/run_measurements.py --tag rerun_1 mujoco:1:before mujoco:1:verified mujoco:2:verified mujoco:4:verified isaac:1:before isaac:1:verified isaac:2:verified isaac:4:verified
PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/annotation_throughput/audit.py
```

audit는 최종 검증의 tag 없는 verified 출력과 before 출력을 비교한다. 재측정은 별도 tag로 보존한다.
원본 Action/clock·계보·frame/MP4 계약과 전체 numeric/event 배열, 카메라별3개 RGB 표본을 비교한다.
MuJoCo 전체 numeric/주석과 Isaac1 결과는 정확 일치한다. Isaac 다환경의 물리 측정과 배치 frame 차이는
숫자 그대로 audit.json에 남기며, 주석 경계를 강제로 기존 값으로 고치지 않는다. 상세 시간/분모·차이는
annotation_throughput의 REPORT/BENCHMARK에 있다. 실물·새 학습·native8환경·여러 seed는 검사하지 않았다.
상세 physics probe는 `run_measurements.py --tag physics isaac:4:verified`이며 기존 tag는 덮어쓰지 않는다.
SimulationContext.step 계측과 전체 native control-step 계측을 구분하고3,456substep/432tick 및 numeric/주석
일치를 audit에서 검사한다. 이후 최종 Annotation/공통 부분 suite68개도 통과했다.

## MuJoCo Scene 파일 생성 제거 — 2026-10-19

`mjcf.build_scene`과 MujocoBackend는 파일을 쓰지 않고 XML/VFS로 모델을 컴파일한다. 후속 정리에서
사용되지 않는 MujocoBackend.output_dir와 PolicyEnvironment.output 인수도 삭제했다. 생성자에 Workspace만
전달하고 디렉터리를 만들거나 이전 XML을 읽지 않는지, Source·DataGen·평가·renderer/물리·Resume이
보존되는지 확인한다. Runtime의 MuJoCo preview도 임시 Scene 폴더 없이 동작한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_backend.py tests/real2sim/test_common_mimic.py tests/real2sim/test_parallel_sim_eval.py tests/real2sim/test_minivla_integration.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_runtime_liveness.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/mujoco_memory_scene/verify_runtime.py
```

변경 전에 기록한 model hash·compiled 배열23개·실제 물리30step trace와의 정확 일치를 검사한다.
마지막 runner는 기존 Source/Annotation을 읽고 격리 Registry에서 Replay·Annotation·DataGen을,
측정 자세 유지 fixture로 짧은2환경 Sim 평가를 실행한다. 출력에 scene 디렉터리/XML/별도 씬 metadata
파일이 없는지 확인한다. Source는 앞선 저장 runner를 재사용하여 fake Leader/실제 MuJoCo3frame과
데이터 보존을 확인한다. 원본·Registry를 보존하고 임시 payload를 제거한다. 실물·새 학습·정책 품질은 대상 밖이다.

## 필수 Source 이름·Curation 이름 폴더·PNG 제거 — 2026-10-19

기존 UI/SettingsForm/Source Partition/Artifact 저장/묶음 선택 suite154개가 통과했다. 빈 Source 이름의
시작 오류 알림·worker 실행0, 한글 이름과 양 엔진 출력 계획, Source output editor/이전 값 제거,
Source의 이름별 선별·분할·중복 이름 차단·원본 보존·Annotation 입력 연결을 확인한다. USD가 필요한
묶음 fixture는 설치된 기존 USD 경로를 그 명령에만 추가한다. 의존성을 설치하지 않는다.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_workflow_ui.py tests/real2sim/test_settings_form.py tests/real2sim/test_source_partitions.py tests/real2sim/test_artifact_storage.py tests/real2sim/test_dataset_collections.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/source_named_storage/verify_runtime.py
```

runner는 실제 Common/과거 Canonical Source20을 임시 관리 경로로 복사하고17개 선별·14/6 분할을 검증한다.
Common17개의629배열·MP4 34개와 Canonical17개 episode 파일을 비교한다. 별도로 실제 Workspace/MuJoCo/EGL에서
가상 Leader로1episode/3frame만 수집해 PNG0개·MP42개·실시간 발행을 확인한다. 후속 변경으로 현재 runner는
Scene 파일0개와 메모리 모델 실행을 검사한다.
실제 장치 연결·사람 시연·새 학습을 수행하지 않는다. 원본 hash/Registry byte 보존 후 임시 payload를 제거한다.
`--ui`는 위 GUI/USD 경로의 별도 process에서 이름 알림·Source editor·선별/분할 미리보기 경로를 캡처한다.
근거는 `outputs/workflow_validation/source_named_storage/REPORT.md`와 JSON/로그에 있다.

## Source/DataGen Resume 입력 격리 — 2026-10-19

사용자 Green DataGen의 두 실패는 파란 Source `dataset.videos`를 Resume 데이터로 넘긴 경우다.
Source→Annotation→DataGen 이동에서 이어 수집 경로가 공유된 것이 원인이었다. UI 회귀에 단계별 복원과
신규 생성의 빈 값·역할 안내 문구를 추가했다. Data Resume 검사는 잘못된 폴더/Source 입력의 거부와
직접 worker가 출력 폴더나 엔진을 만들기 전 차단되는지 확인한다.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_workflow_ui.py tests/real2sim/test_data_resume.py tests/real2sim/test_common_mimic.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/datagen_resume_isolation/verify_runtime.py
```

실제 runner는 실패한 Green Annotation17개를 그대로 읽는다. 원래 잘못된 경로는 NOT_READY, Resume를
뺀 새 계획은 READY를 확인한다. 별도 임시 Registry/출력에서 같은 IK/정책·4환경·목표1개를 실제 생성하고
그 결과의 유효한 이어 생성 계획도 READY인지 검사한 뒤 임시 payload를 제거한다. 사용자 요청50개를 대신
생성하거나 원래 Registry에 새 검증 결과를 넣지 않는다. 모터/Leader/실물 카메라를 사용하지 않는다.
보고서와 사용자 실패 근거는 `outputs/workflow_validation/datagen_resume_isolation/`에 있다.

## Source 이어 수집 선택·저장 이름 — 2026-10-19

Qt UI 검사에서 양 엔진의 HDF5/기존 폴더 선택·취소·Source 분할 잠금과 키보드 이름 입력→worker 전달,
다른 단계에 이름이 섞이지 않는지 확인한다. 기존 저장/Workspace 검사에는 선택 Task를 사용한 자동 이름,
기존 목록 이름의 읽기 전용 표시와 양 엔진 Resume 총 목표/추가 개수 계약 검사를 포함한다.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_workflow_ui.py tests/real2sim/test_curation.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/source_resume_naming/verify_runtime.py
```

실제 검사 runner는 최신 common Source20을 읽고 `EpisodeWriter.resume`로 임시 출력에 복사한다.
숫자740배열·MP4 SHA40개와 원본 hash/Registry 보존을 검사하고 임시 출력은 제거한다. 실제 새 시연을
수집하거나 Leader/모터/카메라를 연결하지 않는다. UI capture도 별도 임시 Registry를 사용한다.
결과와 한계는 `outputs/workflow_validation/source_resume_naming/REPORT.md`에 있다.

## LeRobot 변환본 병합 — 2026-10-19

미분할 Source로 생성한 색상별 변환본도 병합한다. 미분할/Train·Val 혼합, 깨진 partition/model/Scene hash,
Source 계보 누락·물리 모델 변경을 차단하며 재병합과 per-episode Task/원본 증거 보존을 검사한다.
`outputs/workflow_validation/lerobot_merge_unpartitioned/verify_runtime.py`는 사용자 빨강·초록·파랑 각50개를
임시 Registry/출력에서 기존 worker로 병합한다.150개/60,942frame의 수치 전체 일치·RGB18표본·Task·계보,
MiniVLA index 읽기와 원본 hash를 확인하고 임시 payload를 제거한다. `--ui`는 같은 입력의 체크 선택과
미분할 요약/버튼 활성화를 offscreen에서 확인하고 캡처한다. 원래 Registry에 결과를 등록하지 않는다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/lerobot_merge_unpartitioned/verify_runtime.py
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/lerobot_merge_unpartitioned/verify_runtime.py --ui
```

`test_lerobot_merge.py`는 독립 Source Partition의 병합, 중복/Train·Val 혼합/좌표·모델·카메라·FPS
불일치와 중단 정리를 검사한다. 실제 설치된 LeRobot으로 작은 영상 포함 데이터를 생성하고 기존
Operations worker를 통해 병합·등록한 뒤 수치/영상/통계와 MiniVLA CPU reader를 확인한다.
`test_mujoco_workflow_ui.py`의 추가 검사는 변환 화면 전용 버튼, 체크 선택·역할 차단·이름·worker 옵션과
실행 중 재병합 차단을 확인한다. GUI 검사와 native 영상 검사는 의존성 경로를 나누어 실행한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_lerobot_merge.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_workflow_ui.py tests/real2sim/test_source_partitions.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/lerobot_merge/verify_runtime.py
```

마지막 runner는 기존 Train100의 서로 다른 두 episode를 임시 subset으로 만들고 병합한다.
숫자738frame과 두 카메라12개 RGB 표본의 병합 전후 일치, 기존 Val30과 계보 검증, MiniVLA CPU 로딩,
원본/Registry/UI 보존을 확인한 뒤 임시 데이터는 제거한다. 기존 사용 데이터를 병합 결과로 바꾸지 않는다.
보고서와 로그는 `outputs/workflow_validation/lerobot_merge/`에 있다. GPU 학습·Hub 업로드·실물 검사는 아니다.

## MuJoCo 손가락 collision / Real 시작 자세 — 2026-10-18

범위와 실제 증거는 [교정 보고서](../../outputs/workflow_validation/gripper_contact_fix/REPORT.md)를 따른다.
기존 visual/모터/질량·물체 collision 보존과 실제 접촉 집기·상승, Real 활성화 전 encoder 경계 검사를 확인한다.
프로젝트 루트에서 native 검사와 USD 검사는 별도 프로세스로 실행한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_backend.py tests/real2sim/test_eval_logging.py tests/real2sim/test_parallel_sim_eval.py tests/real2sim/test_common_mimic.py -q
```

별도 USD 보정 검사는 기존 Qt/설치된 USD 명령별 경로를 설정한 프로세스에서
`python -m pytest tests/real2sim/test_simulation_calibration.py -k real_eval -q`다.
회귀88PASS/1skip, 별도 보정2PASS. synthetic/fake 장치 검사이며 실제 하드웨어를 열지 않는다.
실제 ACT/MuJoCo4 평가4회는 성공1/집기 실패3이고 성공·실패 HDF5/MP4를 검사했다.
오프라인 convex export를 다시 만들 때만 CoACD가 필요하며 학습/평가 worker에 설치하지 않는다.

2026-10-18 저장 경로 개선: `test_artifact_storage.py`, `test_mujoco_workflow_ui.py`,
`test_source_partitions.py`, `test_training_resume.py`의113개 검사와 기존 실제 Train/Val CPU 로딩·모델별 UI 선택을
확인했다. 정확한 환경/명령/기존 실패와 이동 byte 검사는
`outputs/workflow_validation/artifact_input_layout/REPORT.md`를 따른다. 실제 학습/정책/장치 실행은 하지 않았다.

테스트의 목표는 "실행된다"를 넘어서 **변경이 맞다는 증거를 만드는 것**이다.

2026-10-18 전체 구조 감사의 변경 전/후 suite·실행 환경·기존 실패·UI 비교와 재현 명령은
[REFACTOR_VALIDATION](../architecture/REFACTOR_VALIDATION.md)에 있다.
`test_architecture_contracts.py`는 worker request/파일 identity의 기존 interface와 import 방향을 검사한다.

## 병렬 Sim 평가·성공/실패 기록 — 2026-10-18

보고서/실제 ACT 실행/화면은 `outputs/workflow_validation/parallel_sim_eval/REPORT.md`.
`test_parallel_sim_eval.py`는 전체 quota·실패 집계, batched processor, MiniVLA 환경별 queue/guard,
success/failure 영상·EE 읽기, 과거 평가 폴더 원본 보존, Source gate와 실패 Playback의 분리를 검사한다.
실제 양 엔진4환경·마지막부분묶음·pause/step/resume/Registry는 검증 전용 Registry와 실제 기존 ACT를 사용했다.
사용자 실행을 중단하거나 기존 모델을 재학습하지 않는다. 아래 suite는 서로 중복되므로 합산하지 않는다.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_parallel_sim_eval.py tests/real2sim/test_eval_logging.py tests/real2sim/test_common_mimic.py tests/real2sim/test_mujoco_backend.py tests/real2sim/test_minivla_integration.py tests/real2sim/test_architecture_contracts.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_parallel_sim_eval.py::test_evaluation_setting_and_curation_review tests/real2sim/test_settings_form.py -q
PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/parallel_sim_eval/audit_runtime.py
```

첫 native 실행121PASS/1skip 뒤 추가한 Playback/strict gate 계약은 최종67PASS/1skip으로 재검사했다.
USD/UI139PASS는 기존 `test_modes_same_source_environment`(자동 UUID 출력 경로 비교)만 제외했고,
테스트별 USD/PyQt 경로는 해당 로그의 명령을 사용한다. MuJoCo 물리를 pxr가 import된 process에 섞지 않는다.
실제 재실행 runner는 새 output이 필요하며 기존 증거를 덮어쓰지 않는다. Isaac smoke8-step은 task 품질 측정이 아니다.

## 공통 경로 정리·성능 검사 (2026-10-16)

`test_common_mimic.py`는 공식 본문 parity, Pool 재사용/선택 subset, RNG 격리, 성공 예약/느린 Writer 중 상태 진행,
실패 rollback, private 영상 이동/무결성, 교차 재생 캐시의 전체 identity와 실패 비캐시를 검사한다.
`test_curation.py`는 한 Converter로의 전달과 원본 보존을, 별도 학습 suite는 과거 Checkpoint/normalization을 검사한다.
신호를 주입한 Fixture는 물리 성공 증거가 아니다. 실제 1/2/4환경·정지/재개·양방향 실행은 별도 GPU probe로 확인한다.

검증 기록은 `outputs/workflow_validation/architecture_cleanup/`에 둔다. `before_benchmarks.json`은 기준 커밋8da97ca,
`after_benchmarks.json`은 정리 후 결과이며 pool/lock/commit/copy/영상 검사 시간과 CPU/GPU/RSS 표본을 보존한다.
GPU 비교 중 다른 Isaac 작업이 있으면 정상 표본으로 사용하지 않는다. OOM 실행 로그도 남긴다.
GUI PyQt와 학습 TorchCodec의 GLib 경로를 섞지 않고, USD·MuJoCo 검사는 별도 프로세스로 실행한다.

완료 결과는 [구조 정리 보고서](../../outputs/workflow_validation/architecture_cleanup/REPORT.md)와
[성능 측정](../../outputs/workflow_validation/architecture_cleanup/BENCHMARK.md)을 본다.
core364PASS/6skip, USD68PASS, learning49PASS/1skip, common40PASS는 중복이 있어 합산하지 않는다.
양 엔진 실제1/2/4환경·SIGINT/Resume, 설치된 공식14-Source parity, 실제 LeRobot 변환4경로,
positive 물리 replay cache hit와 실패 Source 차단을 별도로 확인했다.
최초 before CPU병행·after Mu4 사용자GPU부하가 있어 전체 wall 차이를 코드효과로 단정하지 않는다.
사람의 수동 annotation,8환경,실물,새 정책 품질은 미검증이다.

## 1. 테스트 층

### A. Pure/unit tests
Isaac renderer나 실제 serial 장비 없이 검증 가능한 로직.

예:
- joint mapping degree/% ↔ model rad
- profile schema/provenance
- transform math
- artifact/workflow state
- workspace package metadata

### B. Isaac integration tests
실제 Isaac Lab/Isaac Sim API 경계가 필요한 검증.

예:
- articulation state/target 적용
- camera prim/config
- workspace scene compile
- reset/physics 적용
- render/replay

### C. Hardware integration tests
Leader/follower/camera 실제 장비가 필요한 검증.

예:
- calibration identity
- follower measured state
- leader→follower command
- capture synchronization

### D. End-to-end workflow
Demo → replay → annotation → datagen → conversion → train/eval artifact 흐름.

## 2. 현재 테스트 위치

핵심 regression suite는 `tests/real2sim/`에 있다.

주요 예:
- `test_joint_mapping.py`: hardware/model coordinate 경계
- `test_camera_prim.py`: camera config/prim
- `test_pipeline.py`: file IPC와 USD scene 생성
- `test_snapshots.py`: Snapshot 계약/저장/세션/UI와 독립 Builder import·FK·fit·rollback
- `test_provenance.py`: measured/provisional provenance
- `test_runtime_liveness.py`: UI/runtime IPC와 liveness
- `test_workspaces.py`: workspace package/environment
- `test_workflow.py`, `test_workflow_view.py`: workflow operation/UI
- `test_task_definitions.py`: task definition contract
- `test_provenance.py` (기존 Profile/visual 검사 통합): 현재 기본 Profile/scene geometry와 visual

옛 paired/continuous capture와 Reference/API-agent/fitting 전용 테스트는 제거했다.
MuJoCo는 import 후 GL 모드를 바꿀 수 없으므로 Builder rendering 검사와
`test_mujoco_backend.py`/`test_minivla_integration.py`의 physics-only 검사는 별도
프로세스에서 실행한다. USD를 import하는 tests도 MuJoCo backend와 같은 process에 넣지 않는다.
Renderer/engine guard를 테스트 때문에 우회하지 않는다. 최신 분리 suite 결과 및 실행 목록은
`outputs/workflow_validation/unified_joint_control/SIMPLIFICATION_REPORT.md`를 따른다.

## Active Object Set 검사 (2026-10-15)

`test_active_objects.py`는 imported Builder fixture의 실제 MuJoCo body/geom/joint/state/step,
1/2/3개·비연속 조합·새 Dynamic ID, reset/overlap/seed, 기록 상속·Resume·mixed curation을 검사한다.
기존 Task/UI/Source/Precision/두 엔진 회귀와 함께 실행한다. 상세 실행 로그/테스트 분리 목록,
실제 Isaac GPU 렌더·20-frame HDF5 Replay·HDF5/LeRobot 변환 및 초기 실패 기록은
`outputs/workflow_validation/active_scene_objects/REPORT.md` 참조. 모델 학습·물리 grasp 성공과 구분한다.

## Data Curation 동기 재생 (2026-10-15)

`test_curation.py`는 단일/다중 active object, 임의 Builder ID와 치수, 회전·이동된 base의
Robot FK/EE/object 등록, canonical/HDF5 pre/post/initial-only, grasp/transport/place,
timestamp 오류 거절, lazy RGB 및 UI worker/seek/필터, composition/group 선별과
Export→Registry→Convert provenance/원본 보존을 검사한다. 대형 HDF5는 10000-frame sparse
fixture이며 실제 18GiB RGB를 메모리에 적재하지 않는다. 파일 크기와 scalar frame read를 검사한다.

MuJoCo/모델 group과 USD group은 독립 process로 실행한다. 실제 두 엔진의 pre-step 기록,
Qt 저장 frame 재생, 영상/Parquet 변환과 MiniVLA 입력 adapter 실행은 별도 probe다.
로그·명령·화면·최초 실패·실행 한계: `outputs/workflow_validation/curation_object_playback/REPORT.md`.
검증용 home-hold/합성 fixture를 실제 성공 demonstration으로 등록하거나 학습하지 않는다.

## 3. Joint 변경 시 최소 검증

- known degree → expected rad
- rad → hardware representation round trip이 허용 오차 내인지
- joint name/order가 유지되는지
- gripper가 arm joint와 다른 normalization 경로를 유지하는지
- sim action target과 measured state를 혼동하지 않는지
- legacy demo를 새 mapping으로 자동 오인하지 않는지

중요: repository/model 구조만으로 physical zero/sign이 완전히 증명되지 않을 수 있다.
실제 camera/follower 측정 데이터가 부족하면 `NEEDS_MORE_DATA`로 남겨야 한다.

## 4. Camera/Real2Sim 변경 시 최소 검증

가능하면 train/fit data와 held-out validation을 분리한다.

추천 metric:
- landmark/keypoint reprojection error (px)
- silhouette/mask overlap (예: IoU)
- object center/corner reprojection error
- camera calibration reprojection error
- paired-capture joint/state mismatch
- timestamp/receipt spread

결과는 가능하면 before → after 수치로 남긴다.

예:
```text
camera reprojection: 31.4 px -> 6.8 px
robot keypoint:      42.0 px -> 9.1 px
held-out views:      no regression
```

## 5. Optimizer에서 반드시 보는 실패 조건

- camera extrinsic 오차를 robot base transform이 대신 보상
- object pose 오차를 camera pose가 대신 보상
- training frames만 좋아지고 held-out frames가 악화
- wrist/side 중 한 camera만 과적합
- provisional parameter가 verified/measured로 승격
- joint mismatch 상태에서 image metric만 최적화

따라서 parameter group을 나누고, 한 그룹을 바꿀 때 다른 그룹을 고정하거나 prior/constraint를 둔다.

## 6. Dataset/replay 변경 시 최소 검증

```text
record
  → save
  → reload
  → restore initial state
  → replay
  → compare trajectory/state
```

확인 항목:
- field 존재 여부
- dtype/shape
- coordinate contract/version
- initial state
- action semantics
- observation semantics
- replay target field
- task/workspace metadata

## 7. AI/Codex 작업 완료 기준

"테스트 통과" 한 줄이 아니라 다음을 보고해야 한다.

- 실행한 정확한 test/command
- 어떤 실패 조건을 커버하는지
- 결과
- 실행하지 못한 test와 이유
- 실제 hardware/Isaac validation이 필요한 남은 부분

고위험 변경(joint mapping, transform, dataset schema, physics state write)은 사람이 관련 구현과 테스트를 직접 확인한다.

## 공통 Joint Control 검사 (2026-10-12)

`test_joint_control.py` + 기존 precision/calibration/UI/snapshots/launcher 검사를 실행한다.
MuJoCo/Source/Replay 검사와 USD/Workspace 검사는 별도 프로세스에서 실행한다.
설치된 LeRobot SDK의 normalize/integer encoding을 포트 없는 객체로 검사하므로 모터 이동이 아니다.
정확한 명령/최종 결과/Isaac 렌더 범위/실패 이력은
`outputs/workflow_validation/unified_joint_control/REPORT.md`를 따른다.
공통 계약과 구동 heuristic의 한계는 [JOINT_CONTROL.md](JOINT_CONTROL.md)를 따른다.
