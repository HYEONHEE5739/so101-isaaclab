# 기능 단위 리팩터링 검증 — 2026-10-19 (완료 경계별 기록)

## 2026-10-21 단위7b-1 — Replay/Annotation Path 실행

보고서: [path_execution/REPORT](../../outputs/workflow_validation/functional_refactor/path_execution/REPORT.md).
최신7a dirty baseline을 보존했다. 기존 GUI227PASS, 최종 Path/Window 인터페이스 별도, native180PASS/Qt1skip, 기존7a/Curation/Source64PASS는 중복 합산하지 않는다. native의 Qt 평가/Curation 검사는 qt_excluded_verified.log에서 따로 실행했다. 모든 새 Payload·Registry·Run은 임시 root/evidence에만 생성했다.

Repository root, 기존 Python/Qt/USD 환경의 Path·Window 계약:

```sh
env PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest -q tests/real2sim/test_episode_paths.py tests/real2sim/test_episode_path_ui.py tests/real2sim/test_live_scene_selection.py
```

MuJoCo/Curation/Source I/O 검사는 USD 경로를 섞지 않은 별도 process에서 기존 Qt 경로만 사용한다. import·목록만으로 App/GPU/모델이 초기화되지 않는7a 검사도 재실행했다. No Registry·stale 결과·한글/공백 Path·READY/Run·잘못된 header/asset/Scene·partial·출력 symlink/충돌/Worker 결과 이탈·Cancel/Interrupt cleanup을 검사한다.

실제 CLI 검증은 기존4episode fixture를 읽고 별도 `path_execution/CLI root 한글`에 저장했다. 다음은 같은 종류의 재현 예시이며 output 충돌을 피하려면 새 root/이름을 사용한다. options JSON은 backend=mujoco, mode=operator, episode=demo_0, renderer=egl, generation_seed=1047, realtime=false다.

```sh
env PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python scripts/real2sim/workflow.py run --stage replay --input outputs/workflow_validation/annotation_throughput/verified_mujoco1_timed/dataset.hdf5 --workspace outputs/real2sim/environments/scene2_builder/v8 --root /tmp/so101-path-check --storage-name 검증_한글 --options outputs/workflow_validation/functional_refactor/path_execution/cli_options.json
```

Native A/B 도구: `path_execution/native_run.py before|after replay|annotate <새 label> [mujoco|isaac]`. 기준은 저장한7a Operations/unified를 검증 process 안에서만 읽고 임시 Registry를 사용한다. after는 Registry 없는 root이며 child에서 Registry 메서드를 금지한다. 실제 SDK·Workspace·Dataset·카메라·seed는 유지한다. `SO101_VALIDATION_SOURCE`는 검증 도구만의 명시 입력 경로 선택, `SO101_VALIDATION_CANCEL=1`은 첫 완성 episode 진행 파일 뒤 **소유 child만** 중지하는 검증 옵션이다. Production 설정/새 launcher가 아니다. 각각 독립 process로 실행하고 GPU 부족 시 조건을 바꿔 무리하게 재실행하지 않는다.

`compare_native.py <기준 label> <현재 label>`은 control tick trace·주석·배열·metadata·MuJoCo 영상 bytes를 비교한다. `compare_rgb.py`는 Isaac 기준 반복과 전후3048frame씩 streaming decode하여 PTS와 RGB 변동을 구분한다. `summary.json`은 실제5쌍·Cancel/CLI 근거다. CPU 검사 병행·계측이 포함된 단일 실행시간을 일반 성능 향상으로 해석하지 않는다. 실제 장치/학습·Isaac Replay/4·8환경/장시간은 미검증이다.

`actual_gui_benchmark.py`는 사용자 Registry를 읽기만 하며 UI 저장을 양쪽에서 비활성화한 목록 비교다. `gui_benchmark.py`는 evidence의100/1000개 marker fixture와 임시 UI 설정만 쓴다. 선택 header READY 시간/전체 Window 처리량과 목록 완료시간을 구분한다. `audit.py`는 보호 package139파일, unified 실행/정리 try·CurationDialog AST·기존 untracked 로그와 HEAD를 확인한다. 사용자 원본 전수 Hash 감사는 아니다.

## 2026-10-21 단위7a — Path 탐색/선택 (완료)

보고서: [path_discovery/REPORT](../../outputs/workflow_validation/functional_refactor/path_discovery/REPORT.md).
6c-2d까지 반영한 dirty baseline을 보존했다. 신규17PASS, 기존UI202PASS, Curation/Qt49PASS, 엔진/저장 경계203PASS/1제외는 겹치는 suite이므로 합산하지 않는다. 실제 저장소는 read-only marker/선택 config/별도 offline Registry 비교만 수행했다. 이번에 Isaac/GPU·학습 정책 실행·실물을 검증한 것이 아니다.

Repository root에서 Qt-only 검사(USD 경로를 넣지 않음):

```sh
env PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest -q tests/real2sim/test_path_discovery.py tests/real2sim/test_path_discovery_ui.py
env PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest -q tests/real2sim/test_curation.py
PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python scripts/real2sim/workflow.py discover --path outputs/workflow/models --category policy
PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python scripts/real2sim/workflow.py discover --details --path outputs/workflow/models/model_colorblock_cup_a_150ep/checkpoints/150000/pretrained_model
```

기존 Registry/Workspace GUI suite는 필요한 기존 USD/Qt 환경에서 **별도 process**로 실행했다. MuJoCo를 쓰는 Curation suite에 USD 경로를 섞지 않는다. 로그는 new_verified/registry_ui_verified/curation_native_qt/engine_contracts.log에 분리했다. 초기 혼합 환경 실패 로그도 남겼다.

성능 재현 도구는 `outputs/workflow_validation/functional_refactor/path_discovery/benchmark.py`와 `gui_benchmark.py`다. 전자는 처음 실행 시 이 evidence 안에 fixture_100/fixture_1000을 새로 만들며 기존 경로가 있으면 충돌로 중단한다. 재측정은 benchmark의 measure/stats를 import해 기존 fixture를 읽는다. GUI 도구는 evidence 안의 fixture Registry만 생성/갱신하며 실제 Registry/모델에는 쓰지 않는다. gui_benchmark.json은 교체한 Resume 목록만 비교한다. 전체 Workflow refresh 속도나 checkpoint 실행 가능 인증으로 해석하지 않는다. 초기화되지 않은 앱/모델·영상·HDF5 open 금지 검사는 fresh subprocess와 monkeypatch로 별도 확인했다.

필수 metadata 부족 사례/7b 처리는 [REGISTRY_RESPONSIBILITIES](../../outputs/workflow_validation/functional_refactor/path_discovery/REGISTRY_RESPONSIBILITIES.md). audit.py는 원래6c-2 본문/기존 I/O·Policy AST·남은 Registry 메서드/HEAD 보존을 검사한다. 저장 marker 크기 보존은 전체 payload byte Hash 감사가 아니다.

## 2026-10-19 단위6c-1 — Replay/Annotation 공통 물리 재생

보고서: `outputs/workflow_validation/functional_refactor/playback/REPORT.md`. 최신6b-3 이후 dirty tree를 baseline_files.json과 before/에 보존했다. Source/Reader/Writer·DataGen/평가·물리·IK·보정·학습·UI의 보호 package140파일 bytes와 Annotation scheduler/수동 출력 AST를 확인했다. API/기존 callback 연결만 교체했고 engine/공식 Mimic 알고리즘은 변경하지 않았다.

```bash
task_repo=/home/hyeonhee/soarm_isaaclab/soarm101_lab
task_python=/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python
task_playback_evidence="$task_repo/outputs/workflow_validation/functional_refactor/playback"
cd "$task_playback_evidence/test_cwd"
env PYTHONPATH="$task_repo/source/soarm101_lab" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$task_python" -m pytest "$task_repo/tests/real2sim/test_playback.py" "$task_repo/tests/real2sim/test_common_mimic.py" "$task_repo/tests/real2sim/test_parallel_annotation.py" "$task_repo/tests/real2sim/test_parallel_sim_eval.py" "$task_repo/tests/real2sim/test_recording_io.py" "$task_repo/tests/real2sim/test_data_resume.py" "$task_repo/tests/real2sim/test_annotation_diagnostics.py" -q -rs -k 'not diagnostics_remain_in_raw_log'
task_usd=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
env QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$task_repo/source/soarm101_lab:$task_repo/.venv-real2sim-ui/lib/python3.11/site-packages:$task_usd" LD_LIBRARY_PATH="$task_usd/bin" "$task_python" -m pytest "$task_repo/tests/real2sim/test_presentation.py" "$task_repo/tests/real2sim/test_mujoco_workflow_ui.py" "$task_repo/tests/real2sim/test_workflow_view.py" "$task_repo/tests/real2sim/test_parallel_sim_eval.py::test_evaluation_setting_and_curation_review" "$task_repo/tests/real2sim/test_annotation_diagnostics.py" -q
```

실행 로그는 native121PASS/Qt1skip·Qt1제외, GUI107PASS와 별도진단3PASS다. GUI 결합 명령은 두 로그의 suite를 함께 재현한다. baseline93PASS→동일회귀84PASS인 차이는 실사용 없는 Replay cache 전용9검사를 삭제했기 때문이며 실패를 숨기지 않는다. 주요 회귀 실행 시 새 Playback21개는 command/None 순서·before state 재사용·step exception/Interrupt·Manual mark/missing/skip/pause·실제 iterator를 쓰는1/2/4/8 vector refill/Cancel·Worker 소유 engine 미종료를 검사한다. 기존32조합 scheduler fixture는 pause/single-step/resume/idle hold·입력 순서 저장·실패 집계를 계속 검사한다.

마지막에 독립 Stage import2개를 보강하고 test_playback/test_architecture_contracts의 공개 계약45개를 별도로 통과했다(final_interfaces.log). 앞선 회귀와 합산하지 않는다.

native 실행은 이전 검증의 annotated4episode를 읽기 전용으로 재생했다. 원본 Dataset/MP4 fingerprint·좌표/Scene 계약을 유지했고 새 출력은 이 검증 폴더에만 만든다. 아래 재실행 예시는 **아직 존재하지 않는 고유 label**을 써야 한다. baseline prefix가 snapshot_before이면 보존한 옛 annotation/unified 본문을 검증 process 안에서만 읽고, 이후 최신 구현을 실행한다. Production 구 import·Wrapper를 복구하지 않는다.

```bash
env PYTHONPATH="$task_repo/source/soarm101_lab" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$task_python" "$task_playback_evidence/native_compare.py" snapshot_before_recheck_mujoco2 mujoco 2 annotate
env PYTHONPATH="$task_repo/source/soarm101_lab" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$task_python" "$task_playback_evidence/native_compare.py" after_recheck_mujoco2 mujoco 2 annotate
env PYTHONPATH="$task_repo/source/soarm101_lab" OPENBLAS_NUM_THREADS=1 "$task_python" "$task_playback_evidence/compare_native.py" snapshot_before_recheck_mujoco2 after_recheck_mujoco2
# 실제 Isaac 검증은 별도 process/동일 conda에서 backend=isaac,count=2로 실행했다.
# native.close 전에 trace를 게시하여 Kit 종료 후에도 evidence를 남긴다.
env PYTHONPATH="$task_repo/source/soarm101_lab" OPENBLAS_NUM_THREADS=1 "$task_python" "$task_playback_evidence/benchmark_steps.py"
"$task_python" "$task_playback_evidence/audit.py"
"$task_python" "$task_playback_evidence/dependency_audit.py"
```

MuJoCo Annotation1·2env 각4episode/1524step과 명시Replay432step, Isaac Annotation2env4episode를 비교했다. Native engine trace/numeric·grasp/place/attrs·state 호출 수는 정확 일치한다. MuJoCo MP4도8개씩bytes일치. Isaac은 옛 구현 두 번부터MP4가 달라 baseline/전후3048frame을 전부 streaming decode했다. PTS/count 동일, 영상별 최대평균 절대RGB차0.832/255→0.815/255이며 전후video bitwise일치는 주장하지 않는다. 물리·renderer seed나 성공 기준을 바꿔 영상을 맞추지 않았다.

순수5000frame/6회 iterator 중앙5.350→5.528µs(+0.178µs), 실제 Annotation processing MuJoCo1 12.265→12.391초·순차MuJoCo2 10.724→11.293초·Isaac2 71.147→71.726초다. trace/영상/physics 포함 단일paired값이며 성능 개선/장시간 동등성 보장이 아니다. 다른 검사와 겹친 초기MuJoCo2 측정도 별도로 보존했다. cProfile은 Physics/render/I/O 없는 합성 iterator만 측정한다.

core_initial의 잘못된 test경로/cwd, interfaces의 USD 누락, core_final의 nativeQt1실패는 최종환경검사와 구분해 남겼다. 첫Isaac 비교의 byte일치 실패 후 실제 기존반복/RGB 차이를 별도 기록했다. 기존 SDK visual/seed·unknown provenance 경고를 수정하지 않았다. 실제Isaac Replay1env·양 엔진4/8env·수동developer창·실물·장시간/강제전원종료는 미검증이다. Commit/Push/의존성 설치는 하지 않는다.

## 2026-10-19 단위6b-3 — Source EpisodeWriter 책임 분리

보고서: `outputs/workflow_validation/functional_refactor/source_writer/REPORT.md`. 기존 dirty-tree baseline과 실제 코드/test를 읽은 뒤6b-3만 교체했다. 기존 Reader/Common Writer/Buffer AST·보호 package134파일·Source 외 물리/학습/과거 Partition 함수 보존을 확인했다. 사용자 결과를 검증 출력으로 사용/수정하지 않는다.

아래는 검증용 cwd에서 실행하는 최종 Source 회귀 명령이다. 기존 conda/Qt/USD만 사용하고 dependency는 설치하지 않는다.

```bash
task_repo=/home/hyeonhee/soarm_isaaclab/soarm101_lab
task_python=/home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python
task_source_evidence="$task_repo/outputs/workflow_validation/functional_refactor/source_writer"
cd "$task_source_evidence/test_cwd"
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$task_repo/source/soarm101_lab" "$task_python" -m pytest "$task_repo/tests/real2sim/test_source_recording.py" "$task_repo/tests/real2sim/test_recording_io.py" "$task_repo/tests/real2sim/test_minivla_integration.py" "$task_repo/tests/real2sim/test_active_objects.py" "$task_repo/tests/real2sim/test_rgb_video.py" -q -k 'not ui_filters_objects'
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$task_repo/source/soarm101_lab" "$task_python" -m pytest "$task_repo/tests/real2sim/test_episode_reader.py" "$task_repo/tests/real2sim/test_parallel_annotation.py" "$task_repo/tests/real2sim/test_parallel_sim_eval.py" "$task_repo/tests/real2sim/test_eval_logging.py" "$task_repo/tests/real2sim/test_training_resume.py" "$task_repo/tests/real2sim/test_architecture_contracts.py" -q -rs
task_usd=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
env QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$task_repo/source/soarm101_lab:$task_repo/.venv-real2sim-ui/lib/python3.11/site-packages:$task_usd" LD_LIBRARY_PATH="$task_usd/bin" "$task_python" -m pytest "$task_repo/tests/real2sim/test_mujoco_workflow_ui.py" "$task_repo/tests/real2sim/test_artifact_storage.py" "$task_repo/tests/real2sim/test_parallel_sim_eval.py::test_evaluation_setting_and_curation_review" -q
```

최종 Source/소유권/영상58PASS·Qt1개 제외(별도GUI에서통과), 경계82PASS/Qt1skip(위GUI130PASS에서통과). 별도 Curation/분할/목록과 active object UI76PASS다. 초기 저장/Resume76PASS 이후 새로운 수명/FK/외부 staging 검사를 추가해 최종58개를 실행했다. 겹치는 suite는 합산하지 않는다. 기존 unknown provenance 경고는 유지한다.

`compare_source.py before`는 변경 전 실제 EpisodeWriter로 이미 실행한 baseline이며 현재 삭제된 import를 복구해 실행하지 않는다. 해당 원본 snapshot과 before/after payload를 보존했다. `compare_resume.py`는 검증 process 안에서만 snapshot의 원래 Writer를 읽어 비교하고 production에 옛 export를 만들지 않는다.

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$task_repo/source/soarm101_lab" "$task_python" "$task_source_evidence/verify_results.py"
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$task_repo/source/soarm101_lab" "$task_python" "$task_source_evidence/benchmark_recording.py"
"$task_python" "$task_source_evidence/audit.py"
"$task_python" "$task_source_evidence/dependency_audit.py"
```

실제 MuJoCo/EGL seed1047/1048 각형식2×8step의 numeric·RGB·MP4 bytes/PTS·metadata/순서 비교다. Resume2+1episode의 실제 MuJoCo·합성RGB 비교도 별도 보존한다. append benchmark는 합성64×48/128frame/7회, 기존 bounded encoder 포함·Physics/render/save 제외이며 중앙0.07668→0.07667ms/frame이다. cold EGL 한 번의 저장 시간 차이를 speedup으로 해석하지 않는다.

초기 caller 편집 경로/구 import 실패(source_core.log), A/B harness의 Legacy.close 호출(after_source.log), Qt 없는 실행(ui_native_environment_attempt.log), spy fixture의 camera 목록 누락/잘못된 Qt 제외(source_final.log)는 보존하고 최종 *_verified 로그와 구분한다. audit harness의 상수 정규화는 실제 코드가 아닌 검사만 수정했다. 실제 Isaac PhysX·실물 장치·장시간·디스크 강제 종료·GPU 학습은 미실행이다. 실제 장치 검증은 별도 명시 요청이 필요하다.

## 2026-10-19 단위6b-1/2 — Common Writer / 명시적 recording Buffer

보고서: `outputs/workflow_validation/functional_refactor/dataset_writer/REPORT.md`. 최신 미커밋 기준을 보존하고6a Reader 클래스 AST 동일성을 검사했다. 이번6b의 완료 범위는 Common Writer/validator와 전달값 Buffer까지이며 Source Legacy Writer/FK/normalization 전체 분리는 남아 있다.

```bash
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_recording_io.py tests/real2sim/test_rgb_video.py tests/real2sim/test_common_mimic.py tests/real2sim/test_parallel_annotation.py tests/real2sim/test_parallel_sim_eval.py tests/real2sim/test_data_resume.py -q -rs
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_episode_reader.py tests/real2sim/test_curation.py tests/real2sim/test_source_partitions.py tests/real2sim/test_dataset_collections.py tests/real2sim/test_parallel_sim_eval.py::test_evaluation_setting_and_curation_review -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_workflow_ui.py tests/real2sim/test_artifact_storage.py tests/real2sim/test_settings_form.py tests/real2sim/test_workflow_view.py -q
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/functional_refactor/dataset_writer/benchmark_preparation.py
PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/functional_refactor/dataset_writer/audit.py
```

최종 storage111PASS/Qt1skip(뒤 GUI84개 중 실제 통과), Reader/Curation/평가UI84PASS·WorkflowUI158PASS. 별도 Source/Resume6PASS, Native 경계53PASS와 Qt 누락1실패→해당GUI1PASS. 오래된 provenance 경고는 보존했으며 중복 suite 합산 없음.6b-1/2는 각113PASS/1skip로 단계별 검사한 뒤 다음 변경을 진행했다.

Native 명령은 검증 소유 cwd에서 실행해 기존 MUJOCO_LOG를 보존했다.

```bash
# cwd: outputs/workflow_validation/functional_refactor/dataset_writer/test_cwd
PYTHONPATH=/home/hyeonhee/soarm_isaaclab/soarm101_lab/source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest /home/hyeonhee/soarm_isaaclab/soarm101_lab/tests/real2sim/test_minivla_integration.py -k 'source or recorded_state_actions or resume or front_camera' -q
PYTHONPATH=/home/hyeonhee/soarm_isaaclab/soarm101_lab/source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest /home/hyeonhee/soarm_isaaclab/soarm101_lab/tests/real2sim/test_active_objects.py /home/hyeonhee/soarm_isaaclab/soarm101_lab/tests/real2sim/test_eval_logging.py /home/hyeonhee/soarm_isaaclab/soarm101_lab/tests/real2sim/test_training_resume.py /home/hyeonhee/soarm_isaaclab/soarm101_lab/tests/real2sim/test_architecture_contracts.py -q
```

`verify_storage.py`는 보존한 이전 본문과 실제 새 Writer/Buffer를 실제 MuJoCo·기존 agent 소유 Workspace·검은 RGB fixture로 비교했다.12step/qpos/hash·numeric28개·attrs8개·영상 bytes와 adopt inode가 같다. 비교 payload는 충돌 방지상 같은 경로에 재실행하지 않는다. 원래 결과를 삭제하지 말고 새 검증 위치에서 비교할 것.

준비3문장만의 fixture profile은2만frame/pose10개·7회. float64 peak33.30→12.80MB, 중앙15.19→5.93ms. 전체 저장/Physics/영상 처리량 측정이 아니다. 실제 Isaac PhysX/실물/학습 정책/장시간은 미검증이며 dependency/사용자 데이터/Commit/Push 변경 없음. 초기 호출자 import SyntaxError·renderer none RGB 요청 실패는 수정한 최종 로그와 구분해 보존했다.


## 2026-10-19 기능 단위6a — Curation 공통 Reader

실제 계약/수명·전후 측정은 `outputs/workflow_validation/functional_refactor/dataset_reader/REPORT.md`를 따른다. 추가 구현 품질 기준은 설계안 상단에 반영했다. 전체 Dataset I/O·Registry/Training 전환은 아직 완료하지 않았다.

```bash
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_episode_reader.py tests/real2sim/test_rgb_video.py tests/real2sim/test_parallel_annotation.py tests/real2sim/test_common_mimic.py tests/real2sim/test_parallel_sim_eval.py tests/real2sim/test_data_resume.py tests/real2sim/test_training_resume.py tests/real2sim/test_architecture_contracts.py -q -rs
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_curation.py tests/real2sim/test_source_partitions.py tests/real2sim/test_mujoco_workflow_ui.py tests/real2sim/test_workflow_view.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_parallel_sim_eval.py::test_evaluation_setting_and_curation_review -q
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/functional_refactor/dataset_reader/benchmark.py before
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/functional_refactor/dataset_reader/benchmark.py after
PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python outputs/workflow_validation/functional_refactor/dataset_reader/audit.py
```

계약144PASS/Qt1skip(별도 GUI1PASS)·UI136PASS. 최종 group handle 재사용 뒤 Reader/RGB12PASS·Curation/평가 UI36PASS로 영향을 받는 경로를 다시 확인했다. 순차 측정7회·80프레임 open1520→1·중앙0.383→0.071초·프레임 배열880개 정확 일치. 합성 HDF5/Qt/MP4 fixture이며 사용자 데이터·실물·새 엔진 rollout·학습/GPU/전체 UI cold/warm 성능 검증이 아니다. OS page cache를 통제하지 않았다. 초기/최종 로그를 보존했고 suite를 합산하지 않는다. 의존성을 설치/변경하지 않았다.


전체 리팩터링 완료 검사가 아니다. Control/Guard/Checkpoint 핵심/정규화/Workspace Task만 실제 이동했다.
Registry/Group 정책/전용 MiniVLA 학습 연결은 남아 있다. 공식 ACT/SmolVLA Trainer는 바꾸지 않았다.
상세 근거: `outputs/workflow_validation/functional_refactor/REPORT.md`, `behavior_audit.json`, `model_comparison.json`, `dependency_comparison.json`.

기존 lerobot-arena Python3.11을 사용한다. 프로젝트 root에서 아래 명령을 실행한다. 실제 장치를 열지 않는 fixture 검사다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_parallel_annotation.py tests/real2sim/test_common_mimic.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=source/soarm101_lab /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/minivla/test_data_and_runtime.py tests/minivla/test_act_baseline.py tests/minivla/test_lerobot_direct.py tests/real2sim/test_training_resume.py tests/real2sim/test_architecture_contracts.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_mujoco_workflow_ui.py tests/real2sim/test_settings_form.py tests/real2sim/test_dataset_collections.py tests/real2sim/test_task_definitions.py tests/real2sim/test_presentation.py tests/real2sim/test_live_scene_selection.py tests/real2sim/test_artifact_storage.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH=source/soarm101_lab:.venv-real2sim-ui/lib/python3.11/site-packages:/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311 LD_LIBRARY_PATH=/home/hyeonhee/miniconda3/envs/lerobot-arena/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest tests/real2sim/test_source_partitions.py -k test_curation_ -q
```

최초3단위 검사에서 모듈명/기존인수 충돌5개를 수정해28개 재통과했다. native 통합의188개는 통과했고,
PyQt 없는6개 UI 검사는 위 마지막 명령으로6개 모두 통과했다. UI196개, 전처리/모델 CPU60개는 겹치는 suite이므로 합산하지 않는다.
공통 Task의 MuJoCo 실제 before/after 비교는 같은 임시 Workspace의23배열/30step/hash·Scene이 정확히 같다.
새 Isaac/실물/GPU 학습·실제 pretrained MiniVLA 추론·현재 UI 성능 수치는 측정하지 않았다.

---

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

## 6c-2a 진행 상태 — 2026-10-20

기록 Scene의 Isaac 생성만 `simulation/environment.py:create_isaac`로 이동하고 unified._native_isaac를 삭제했다. Worker가 identity 검사에 사용한 Workspace를 재사용한다. 105개 계약 회귀 통과, 설정/reset/close 본문은 입력 치환 외 AST 동일. 실제 Isaac baseline부터 GPU OOM으로 실패하여 native A/B 및 성능은 미검증이다. MuJoCo 생성, Source/평가의 별도 Isaac 경로, make_config의 기존 Presentation 간접 참조는 남아 있다. **6c-2 전체 완료 아님.** 근거: `outputs/workflow_validation/functional_refactor/environment/REPORT.md`.

## 2026-10-21 — 6c-2a 실제 엔진 검증 완료

동일 Isaac2환경/seed1047/Dataset4episode로 기준2회·현재1회 실행 완료. Workspace load2→1, 생성/reset/close1/5/1, 설정·초기/물체속도 포함29trace·148numeric·Annotation 순서 정확 일치. RGB3048frame PTS/count 동일, pixel 평균차 최대 기준0.83587/255·전후0.84249/255로 bitwise/통계적 비열등성 주장은 하지 않는다. GPU peak 전체약4.96GiB, OOM 없음. 예외 close는 별도13Fixture통과. Production/테스트 파일hash 불변, 사용자자료/설정·Commit/Push 변경 없음. 근거 `outputs/workflow_validation/functional_refactor/environment/verification_20261021/REPORT.md` 및 summary.json. **6c-2a 구현+실제 검증 완료; 6c-2 전체 미완료.** 이전 GPU OOM/검증 보류 문구는 역사 기록이다. 다음 구조 변경은 수행하지 않음.

## 6c-2b — 명시 Preview/관측 설정

근거: [보고서](../../outputs/workflow_validation/functional_refactor/presentation_config/REPORT.md). baseline은 직전6c-2a 최신 dirty tree이며 비교 tool은 해당 검증 폴더에만 있다. native와 기존 Qt/USD process를 분리한다. 의존성 설치는 없다.

```sh
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest -q tests/real2sim/test_environment_factory.py tests/real2sim/test_common_mimic.py tests/real2sim/test_parallel_annotation.py tests/real2sim/test_playback.py tests/real2sim/test_mujoco_backend.py
```

이 suite는 초기122PASS였다. 명시 bool/fresh import/전달 검사 보강 뒤 test_environment_factory+test_common_mimic72PASS, 병렬평가 포함41PASS/Qt1skip도 확인했다. Qt/USD 기존 환경에서 presentation/workflow/mujoco_workflow_ui/environment_factory145PASS, runtime/workspaces42PASS, parallel_sim_eval16PASS. 겹치는 suite는 합산하지 않는다. 초기 native pxr 미지정 collection과 to_dict(partial) 검증도구 실패는 로그에 보존했다.

실제2환경 Annotation A/B는 `presentation_config/verification/run.py <새 label>`을 별도 process에서 순서대로 실행한다. before로 시작하는 label은 해당 검증 폴더 baseline 모듈만 복원한다. 기존 before_b/before_c/after_a가 완료 근거다. 환경수·해상도·Physics 변경은 없다. `compare_native.py before_b after_a`, `compare_rgb.py`, 상위 summarize.py로 숫자/계측/RGB를 구분한다. 기존 결과 폴더는 덮어쓰지 않는다.

`run_evaluation.py before_eval/after_eval`은 실제 Isaac2환경·6step·측정 자세 유지 검증 정책이며 성공률 검사가 아니다. Production 정책/학습모델을 변경하지 않는다. `config_invariance.py`는 실제 SDK 설정만32번 비교하고 native 환경을 생성하지 않는다. MuJoCo `mujoco_compare.py before/after`는 실제EGL/compiled20배열·30step·Side/Wrist/Overview RGB를 비교했다. 재현 시 고정 출력 이름 충돌을 피하도록 검증 폴더를 복사해 사용한다.

새 Production 환경 생성/관측/read/render 호출은 없다. 픽셀 차이는 기준 반복에서도 발생했으며 PTS/count·숫자 동등성과 별도로 보고한다. 수동 developer 창의 실제 조작, 실물, Source 새수집/장시간DataGen·학습정책 품질은 이번 실검증 범위가 아니다.

## 6c-2c — MuJoCo 생성 책임

[보고서](../../outputs/workflow_validation/functional_refactor/mujoco_environment/REPORT.md), 기준 dd935a6. 실제 payload/runner는 검증 폴더에만 존재한다. 새 의존성·장치 연결은 없다. Native와 기존 Qt/USD process를 분리한다.

```sh
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest -q tests/real2sim/test_mujoco_preparation.py tests/real2sim/test_common_mimic.py tests/real2sim/test_source_recording.py tests/real2sim/test_simulation_calibration.py tests/real2sim/test_minivla_integration.py tests/real2sim/test_active_objects.py -k 'not ui_filters_objects'
```

최종 core126PASS/Qt1제외. 기존 Qt/USD 환경에서 해당1개 PASS, Workflow/Presentation/평가UI110PASS. 계약 suite(parallel_annotation/parallel_sim_eval/data_resume/mujoco_backend/playback/training_resume)는95PASS/Qt1skip이며 그Qt는UI suite에서PASS. 초기정책분리34PASS/새준비12PASS, 생성이동43PASS. suite중복합산 금지.

`mujoco_environment/native_compare.py <새 label> mujoco 2 annotate`는 실제EGL/기존4episode·seed1047 A/B, `... mujoco 1 replay`는 실제Replay와접촉digest를 기록한다. before 접두사는 검증 폴더 baseline 모듈만 process안에서 읽는다. Root가 아닌 해당 cwd에서 실행하고 PYTHONPATH는 Repository/source/soarm101_lab 절대 경로를 사용한다. 기존출력폴더 덮어쓰기 금지.

Source source_compare.py는 **가상 Leader+실제 MuJoCo/EGL**이며 source/resume label은 고정 검증입력을 사용한다. 실물에 연결하지 않는다. datagen_compare.py의 controlled label은 검증용 Condition으로 첫 claim만 env1→env0 순서로 맞춘다. Production scheduler 수정이 아니며 자유 병렬 실행 bitwise 재현성을 검증한 것이 아니다. 초기 자유실행 결과차이도 보고서/로그에 보존했다. controlled_resume은 Parent 없이 직접 생성/이어 실행을 검사했다. 목표 성공2개 미달은 기존domain 실패로 남겼다.

실제 평가 evaluation_compare.py는2환경/3episode×6step의 측정자세 유지 정책이며 학습모델 성능평가가 아니다. before/after payload는 compare_payload.py(새 Source UUID/private video 경로/실측 timing 제외), physics/model은 compare_runs.py·resource_comparison.json으로 비교했다. DataGen의 실측 ik_duration_ms 배열은 shape/dtype만 비교한다. None준비3회 benchmark_prepare.py의 단일median을전체Workflow속도보장으로 해석하지 않는다.

기존 CLI smoke는 scripts/real2sim/workflow.py worker --backend mujoco --stage environment --workspace <원본v8> --renderer egl --output <새검증경로> --options '{"steps":3,"seed":1047,"realtime":false}'로3step 성공했다. 기존 check_env의 직접Backend 인터페이스는 바꾸지 않았다. Stage/Worker금지engineimport·무부작용import·close계약은유지했다. 실물/장시간/학습정책품질/4·8환경actual/OSMesa·softwareEGL/새LegacyEGL A/B는이번에실행하지않았다.


## 6c-2d Isaac 생성 경계·Workspace 재사용

[보고서](../../outputs/workflow_validation/functional_refactor/isaac_boundaries/REPORT.md)와 [호출자 계약](../../outputs/workflow_validation/functional_refactor/isaac_boundaries/CALLERS.md). 기존 conda Python3.11 및 Qt/USD 경로를 사용하며 의존성을 설치하지 않는다.

```sh
PYTHONPATH=source/soarm101_lab OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/hyeonhee/miniconda3/envs/lerobot-arena/bin/python -m pytest -q tests/real2sim/test_isaac_boundaries.py tests/real2sim/test_environment_factory.py tests/real2sim/test_mujoco_preparation.py tests/real2sim/test_mujoco_backend.py tests/real2sim/test_common_mimic.py tests/real2sim/test_playback.py tests/real2sim/test_parallel_annotation.py tests/real2sim/test_parallel_sim_eval.py -k 'not evaluation_setting_and_curation_review and not source_partition_evaluation_review'
```

core_complete.log는 최종 Native 회귀, gui_complete.log는 기존 Qt/USD 환경에서 Runtime/launcher/Live Scene/평가/Presentation/Calibration/새입력137PASS, adjacent.log는 Source 녹화/보정/학습Resume/구조57PASS다. 입력6개와 suite를 합산하지 않는다. Native에서 제외한 평가 UI는 GUI suite에서 검사한다.

실제 Isaac은 `outputs/workflow_validation/functional_refactor/isaac_boundaries/run.py <새 label> <eval|runtime|runtime_connect|source|check>`를 Repository root에서 conda Python으로 실행한다. before 접두사는 이 폴더의 시작시점 baseline 코드만 프로세스 안에서 사용한다. 다른 process와 GPU 상태를 먼저 읽고 각 process 종료 후 순차 실행하며 기존 label은 덮어쓰지 않는다. run.py가 1초 간격 GPU전체/RSS를 기록하고, success JSON·정상 close와 비교 파일로 성공을 판단한다. Kit exit0만으로 성공 처리하지 않는다.

Source 모드는 가상 Leader+실제 엔진의 **생성 경계까지만** 실행한다. Recorder/전체 수집/실물 검사가 아니다. eval은 측정자세 유지 정책2환경×6step, runtime_connect는 preview_only로 로봇/카메라 장치를 열지 않는다. check는 기존 물리CLI120step이며 로봇 집기 성공 검사가 아니다. 원본 Workspace/카메라/Physics는 그대로다. compare.py는 정확한 cfg/trace/numeric/PTS와 렌더 RGB 변동을 구분한다. config_invariance.py는 실제 SDK cfg만32회 구성하고 Environment를 생성하지 않는다. 최초 fixture/GUI파일명/검증 helper import 실패 로그도 보존했다.

## Registry 제거 검증 (2026-10-21)

로그는 `outputs/workflow_validation/functional_refactor/registry_removal/`에 있다. `integrated.log`는 관련 Path/저장/Resume/모델/병합/Curation 통합 검사이며, `ui_connections.log`는 Source/Live/비동기 연결 검사다. USD가 로드되는 Workspace Fixture와 실제 MuJoCo/TorchCodec 검사는 별도 process로 실행한다. 환경 충돌 뒤 해당 영역만 다시 검사하며 기존 DataGen/Conversion/Evaluation 실행을 반복하지 않는다. `native_*.log`는 실제 CLI 결과다. 실물 Real Evaluation·Hub 업로드·새 학습은 실행하지 않았다.

기존 CLI는 `workflow.py discover <path>`, `ready/run --stage <stage> --input <path> --workspace <path> --storage-name <name> --options <json-file>`을 사용한다. `register/--artifact/--registry`는 제거됐다.

## Curation 독립화 (2026-10-21)

`tests/real2sim/test_source_partitions.py`, `test_curation.py`, `test_dataset_collections.py`가 자유 배정/역사 증명·저장/취소·Preview를 확인한다. 실제 MuJoCo를 사용하는 Legacy Preview fixture는 USD가 import되지 않은 Qt process에서 실행한다. `tests/minivla/test_partition_training.py`는 기존 Training 계약을 고정 역사 fixture로 유지한다. 결과는 `outputs/workflow_validation/functional_refactor/curation_independent/REPORT.md`에 있다. 이번에는 시뮬레이션 A/B를 반복하지 않았다.


## Training / Offline Validation 구조 개편

공식 Trainer config/Resume·optional Val·형식/누수·취소/충돌 검사는 `tests/real2sim/test_learning_interfaces.py`, `test_training_resume.py`를 사용한다. Qt 연결은 `test_learning_ui.py`와 기존 Workflow UI의 학습 선택 검사를 사용하며 기존 Qt/USD 환경이 필요하다. 실제 ACT/SmolVLA 로컬 Checkpoint CPU 1batch 실행·사용한 입력·수치/미검증 사항은 [보고서](../../outputs/workflow_validation/functional_refactor/training_validation/REPORT.md)에 있다. 새 학습/장시간 Resume·대형 모델 다운로드·시뮬레이션을 회귀 확인만을 위해 시작하지 않는다.
