# SO-101 Project Invariants

이 문서는 AI/Codex가 코드를 바꿀 때도 **절대 조용히 깨뜨리면 안 되는 계약**을 모은다.
변경이 필요하면 이유, 영향 범위, migration/test 계획을 먼저 명시한다.

## 1. Robot / joint

- Isaac Lab 내부 SO-101 model joint position/target은 radians를 사용한다.
- LeRobot calibrated arm 값(degree)과 model radians를 같은 값처럼 사용하지 않는다.
- Gripper hardware normalization(0..100)과 model joint coordinate를 구분한다.
- hardware→model 변환은 공통 mapper 경계를 통해 수행한다.
- motor homing offset을 이미 device/calibration 계층에서 적용한 경우 다시 적용하지 않는다.
- Leader calibration을 follower calibration 대신 사용하지 않는다.
- follower hardware 명령 경로와 sim measured-pose render 경로를 혼동하지 않는다.
- joint order/name mapping은 암묵적 list 위치가 아니라 명시적인 contract를 유지한다.

## 2. Real2Sim

- GPT/LLM이 이미지에서 camera pose, object pose, robot base transform, 물리 파라미터를 숫자로 임의 추측하지 않는다.
- 측정 가능한 값은 측정하고, calibration 가능한 값은 calibration하며, 최적화 가능한 값은 objective/metric으로 최적화한다.
- 모든 중요한 profile 값은 provenance를 가진다: measured / calibrated / optimized / configured / provisional(assumed).
- Real/Sim 이미지 비교는 가능한 한 동일한 robot configuration에서 수행한다.
- camera/robot/object/table 변수가 서로의 오차를 임의로 상쇄하지 않도록 parameter group과 validation을 분리한다.
- Real2Sim calibration용 deterministic scene과 training randomization을 분리한다.
- "화면이 비슷해 보인다"만으로 calibration 성공을 선언하지 않는다.

## 3. Camera / capture

- side/wrist camera identity를 명시적으로 유지한다.
- Real/Sim pair에는 가능한 범위에서 capture timing과 robot state metadata를 저장한다.
- image resolution/intrinsics 적용 관계를 명시한다.
- wrist mount/extrinsic이 provisional이면 verified로 취급하지 않는다.

## 4. Dataset

- observation과 action의 의미를 섞지 않는다.
- action이 commanded target이면 measured state라고 기록하지 않는다.
- replay에 필요한 initial robot/object state와 coordinate contract를 보존한다.
- dataset schema/coordinate 변경은 versioned contract와 compatibility check 없이 적용하지 않는다.
- old demo를 자동으로 "호환됨" 처리하지 않는다. 실제 contract/mapping 정보를 검사한다.

## 5. Existing pipeline preservation

Real2Sim/Workspace 기능 추가만을 이유로 다음 알고리즘을 불필요하게 재작성하지 않는다.

- Mimic
- datagen
- policy architecture
- training algorithm
- 기존 SO-101 robot implementation
- 기존 teleoperation semantics

필요한 연결은 adapter/config/workspace 계층에서 최소 변경으로 수행한다.

## 6. Workspace

- accepted Real2Sim revision과 published workspace version을 구분한다.
- 기존 published workspace를 조용히 덮어쓰지 않는다. 의미 있는 변경은 새 version/revision으로 남긴다.
- mass/friction/collision 등 미측정 물성은 measured 값처럼 표현하지 않는다.
- workspace task 의미를 사진만 보고 확정하지 않는다. 명시적 task definition을 사용한다.

## 7. AI-assisted development

Codex/agent가 non-trivial 변경을 끝냈을 때 최소한 다음 증거를 남긴다.

1. 해결한 문제
2. 접근 방법
3. 변경 파일과 이유
4. 데이터 흐름 변화
5. 실행한 테스트
6. 검증하지 못한 부분
7. 가능한 실패 조건
8. 사람이 직접 봐야 할 고위험 코드
9. 이후 수정 위치

프로그램이 실행된다는 사실만으로 correctness를 주장하지 않는다.
