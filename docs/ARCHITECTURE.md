# 파이프라인 구조

_2026-09-24 · 프로토타입 기준_

## 한 장 요약

```
                         ┌─────────────── 학습 경로 (오프라인, Colab) ───────────────┐
  OmniFall 영상 ──► extract_keypoints.py ──► (T,17,3) .npy ──► retrim/pack ──► train_stage2.py ──► gru.pt
                    YOLOv11n-pose + 정규화                     kps_trim.npz      5-fold · LOO
                         └────────────────────────────────────────────────────────────┘

                         ┌─────────────── 추론 경로 (실시간, 로컬 M3) ───────────────┐
  영상 파일 ──► resize 960 ──► YOLOv11n-pose ──► 최대 박스 1인 ──► normalize()
                                                                      │
                                                     ring buffer (window_sec × fps)
                                                                      │
                                              stride 마다: resample(64) ──► GRU ──► prob[normal,fall,fallen]
                                                                      │
                                                    히스테리시스 ──► 상태 {normal | fall | fallen | no_person}
                                                                      │
                                                  ImmobilityDetector (매 프레임)
                                                    R1  위험 상태 ≥ t_fallen  ──► FALLEN_IMMOBILE  (CRITICAL)
                                                    R2  미동 ∧ 수평 ≥ t_still ──► STILL_HORIZONTAL (WARNING)
                                                    에지 fall 진입 (쿨다운)   ──► FALL
                                                                      │
                               오버레이 · 저장 영상 · events.jsonl ◄──┘
                         └────────────────────────────────────────────────────────────┘
```

## 동작 시나리오 — 프레임 하나가 들어오면

영상이 60fps 라고 하자. 매 프레임마다 1~5 가 돌고, 0.5초마다 6~8 이, 매 프레임 9 가 돈다.

#### 1. 프레임을 읽고 가로 960 픽셀로 줄인다
AI Hub 영상은 4K(3840×2160)다. 그대로 넣으면 YOLO 가 느리다. 키포인트는 나중에 프레임 크기로
나눠 정규화하므로 축소해도 결과는 같다.

#### 2. YOLOv11n-pose 가 사람을 찾고 자세를 잰다
프레임 한 장 → 검출된 사람마다 **바운딩 박스 + 17개 관절 좌표**(코·어깨·팔꿈치·손목·엉덩이·무릎·발목, 각 x·y·신뢰도).
여러 명이 잡힐 수 있다.

#### 3. 가장 큰 바운딩 박스 한 명만 남긴다
학습 데이터가 1인 클립이라 모델도 1인을 전제한다. 카메라에 가까운(큰) 사람을 추적 대상으로 삼는다.
아무도 없으면 17개 관절을 전부 0 으로 채운 빈 벡터를 넣는다 — "없었다" 는 사실도 기록해야 6번에서 쓸 수 있다.

#### 4. 카메라에 무관한 좌표로 바꾼다 (정규화)
- 좌표를 프레임 크기로 나눈다 → 해상도 무관
- **엉덩이 중점을 (0,0) 으로** 옮긴다 → 화면 어디에 있든 무관
- **어깨~엉덩이 거리를 1** 로 맞춘다 → 카메라가 멀든 가깝든 무관

이제 "서 있음" 은 어깨가 (0, −1) 근처, "누움" 은 어깨가 (±1, 0) 근처다. 어느 카메라로 찍었든 같은 숫자가 나온다.
AI Hub 공식 모델이 카메라 위치마다 모델을 8개 둔 문제를 이 단계가 대신한다.

#### 5. 최근 2초(120프레임)를 링 버퍼에 쌓는다
낙상은 한 장으로는 알 수 없다 — "서 있다가 넘어져 바닥에 있다" 는 시간의 변화다.
새 프레임이 들어오면 가장 오래된 프레임이 밀려나간다.

#### 6. 0.5초마다 — 사람이 실제로 있었는지 먼저 확인한다
버퍼 120프레임 중 사람이 검출된 비율이 **50% 미만이면 판정을 건너뛰고 상태를 `no_person` 으로** 둔다.

이 게이트가 없으면 위험하다 — 빈 벡터(전부 0)를 모델에 넣으면 **`fallen` 0.79** 를 낸다는 것을 확인했다.
사람이 화면 밖으로 나가면 "넘어져 있다" 고 오판하는 셈이다.

#### 7. 120프레임을 64프레임으로 리샘플해 GRU 에 넣는다
모델은 항상 64프레임을 받도록 학습됐다. 선형 보간으로 길이를 맞춘다 — 30fps 웹캠이든 60fps 영상이든 같은 입력이 된다.
GRU 는 순서를 따라 읽고 세 확률을 낸다: **normal · fall · fallen**.

#### 8. 확률을 상태로 바꾼다 — 경계에서 떨리지 않게
- 위험 확률(fall + fallen)이 **0.5 를 넘으면** 위험 상태로 들어간다
- 한 번 들어가면 **0.35 아래로 떨어져야** 빠져나온다 (히스테리시스)
- 위험 상태 안에서는 fall 과 fallen 중 큰 쪽이 라벨

진입·해제 임계를 다르게 두는 이유: 0.5 근처에서 확률이 0.48 ↔ 0.52 로 흔들리면 상태가 매 판정마다 뒤집혀 알림이 중복된다. 실측에서 그 문제가 있었다.

#### 9. 매 프레임 — 규칙이 시간을 잰다
모델은 "지금 어떤 상태인가" 만 말한다. "그 상태가 얼마나 지속됐나" 는 규칙이 센다.

| 규칙 | 조건 | 이벤트 |
|---|---|---|
| 낙상 순간 | 상태가 `fall` 로 **막 바뀐** 순간 (2초 안에는 다시 안 울림) | `FALL` |
| **R1 낙상 후 방치** | `fall` 또는 `fallen` 이 **3초 연속** | `FALLEN_IMMOBILE` — CRITICAL |
| **R2 누운 채 미동** | 5초 동안 관절이 거의 안 움직이고 **몸통이 수평** | `STILL_HORIZONTAL` — WARNING |

- R1 이 fall 과 fallen 을 구분하지 않는 이유: 모델의 둘 사이 경계가 흐리고, 어느 쪽이든 "바닥에 있다" 는 뜻이다. 중요한 건 라벨이 아니라 **지속 시간**이다.
- R2 에 "몸통 수평" 이 붙는 이유: 앉아서 가만히 있는 것은 정상이다. 자세 조건 없이 미동만 보면 오탐이 폭발한다.
- 매 프레임 재는 이유: 0.5초 단위로만 재면 임계 근처를 놓친다. 실측에서 2.91초로 3.0초를 0.09초 차이로 못 넘긴 적이 있다.

#### 10. 화면에 그리고, 기록한다
- **오버레이**: 골격 · 상태 라벨(색) · 세 확률 바 · FPS · 하단 타임라인(정답 구간 음영) · 이벤트 시 2초간 배너
- **저장**: `--save-video` 로 오버레이된 mp4
- **로그**: `events.jsonl` — 0.5초마다 상태 한 줄, 이벤트가 나면 이벤트 한 줄

### 실제로 돌리면 — `00003_H_A_FY_C1` (전면낙상, 정답 구간 260~320f)

```
  0 ~ 260f   서서 걷는 중                  normal 0.85          상태 normal
    290f     정답 구간 안, 넘어지는 중       fall   0.74          ★ FALL 이벤트   (정답 시작 +30f = 0.5초)
    319f     바닥에 닿음                    fall   0.92
    348f     확률이 0.50 / 0.44 로 흔들림                          히스테리시스가 상태 유지 → 중복 알림 없음
    406f ~   바닥에 그대로                  fallen 0.91~0.92     상태 fallen
    470f     위험 상태 290f 부터 3초 경과                          ★ FALLEN_IMMOBILE (R1, CRITICAL)
    600f     영상 끝
```

같은 파이프라인이 비낙상 영상(`00047`)에서는: 침대에 엎드리는 116f 에 `FALL` 오탐이 한 번 나지만
232f 에 `normal` 로 돌아와 위험 상태가 2초를 못 채운다 → **R1 은 울리지 않는다.**
순간 오판을 지속 시간 조건이 걸러내는 사례다.

**두 단계로 나눈 이유.** 낙상은 "서있음 → 넘어짐 → 바닥" 이라는 시간에 걸친 변화다.
YOLO 는 각 순간의 자세를 숫자로 바꾸는 센서 역할이고, 행동 판단은 시계열 모델이 한다.
미동은 그 위에서 **시간 누적 규칙**으로 판정한다 — 모델이 "지금 바닥에 있다" 를 말하면,
규칙이 "그 상태가 얼마나 지속됐나" 를 잰다.

## 구성요소

| 단계 | 역할 | 파일 | 입력 → 출력 |
|---|---|---|---|
| 자세 추정 | 사람 검출 + COCO 17 키포인트 | `yolo11n-pose.pt` (ultralytics) | 프레임 → (N, 17, 3) 픽셀 |
| 1인 선택 | 가장 큰 박스 | `scripts/realtime_demo.py` `largest_person()` | (N,17,3) → (17,3) |
| **정규화** | 카메라 불변 표현 | `scripts/pipeline_core.py` `normalize()` | (T,17,3) 픽셀 → 몸통 단위 |
| 리샘플 | 고정 길이 | `pipeline_core.py` `resample()` | (T,17,3) → (64,17,3) |
| 분류 | 프레임 상태 | `runs/models/gru.pt` · `pipeline_core.py` `build_model()` | (64,17,3) → 3 확률 |
| 상태 결정 | 히스테리시스 | `realtime_demo.py` 판정 블록 | 확률 → 상태 |
| **미동 감지** | 시간 누적 규칙 | `realtime_demo.py` `ImmobilityDetector` | 상태 시계열 → 이벤트 |
| 출력 | 오버레이·기록 | `realtime_demo.py` `draw()` | 프레임 + 상태 → mp4 · jsonl |

## 정규화 — 이 연구의 핵심 기여

`pipeline_core.py:normalize()`. 세 단계다.

1. 프레임 크기로 나눈다 → 해상도 무관
2. **엉덩이 중점을 원점으로** → 화면상 위치 무관
3. **몸통 길이(어깨~엉덩이)로 나눈다** → 카메라 거리 무관

스케일 기준을 몸통으로 잡은 이유: 서 있든 누워 있든 몸통 길이는 거의 일정하다.
바운딩 박스 높이로 나누면 누웠을 때 값이 폭증한다.

검증: 해상도를 절반으로 줄이고 위치를 옮겨도 **차이 0.0**. 서 있음(어깨 y=-1)과
누움(어깨 x=-1)이 명확히 구분된다.

| | AI Hub 베이스라인 | 본 파이프라인 |
|---|---|---|
| 특징 추출 | MediaPipe Holistic 1,662차원 (얼굴 468점이 84%) | YOLOv11-pose 17점 × 2 = 34차원 |
| 입력 크기 | 997,200 (600f flatten) | **20,400** (64f × 17 × 3) — 49배 작음 |
| 카메라 대응 | 위치별 RandomForest **8개** | **단일 모델** |
| 시간 처리 | flatten (순서 소실) | 시퀀스 모델 |

실측 효과 — 같은 낙상 장면을 8각도로 찍은 AI Hub 샘플에서 판정 표준편차:
베이스라인 **0.30** (C5 0.98 vs C2 0.23 로 판정 뒤집힘) → 본 파이프라인 **0.213**.
베이스라인이 무너진 C2·C3 에서 0.59·0.89 로 판정을 유지한다. ([results_aihub_pipeline.md](results_aihub_pipeline.md))

### 정규화가 만드는 제약 (알아둘 것)

- **엉덩이의 절대 y 는 항상 0 이다.** "엉덩이가 아래로 떨어졌다" 를 조건으로 쓰면 항상 거짓이다.
  낙하는 **발목-엉덩이 높이차** 같은 관절 간 상대량으로 봐야 한다 (규칙 기반 베이스라인이 이 버그를 겪었다).
- **사람 미검출(전부 0) 입력에 모델이 `fallen` 0.79 를 낸다.** 화면에서 사람이 사라지면
  "넘어져 있다" 로 오판하므로, 검출률 < `min_det` 윈도우는 GRU 를 돌리지 않고 `no_person` 으로 둔다.

## 모델 카드 — `runs/models/gru.pt`

| 항목 | 값 |
|---|---|
| 구조 | GRU 2층 × 128 hidden → Dropout 0.3 → Linear(128, 3) |
| 파라미터 | 약 169K |
| 입력 | (64, 17, 3) — 64프레임 리샘플, COCO 17, (x, y, conf) 정규화 좌표 |
| 출력 | softmax 3 — `normal` · `fall` · `fallen` |
| 학습 데이터 | OmniFall staged 3종 (GMDCSA24 453 · edf 483 · caucafall 258) = 1,193 클립 · 피험자 19명 |
| 클래스 매핑 | fall=1 → `fall` · fallen=2 → `fallen` · 나머지 8클래스 → `normal` |
| 손실 | CrossEntropy + 클래스 가중치(빈도 역수 √) + label smoothing 0.05 |
| 모델 선택 | val **Recall** 기준 (accuracy 아님) · early stop patience 15 |
| 5-fold (피험자 단위, 이진 위험 기준) | Recall **0.867 ± 0.066** · FPR **0.112 ± 0.066** · fallen 포착 **73.7%** |
| 환경 교차 (leave-one-out 3회) | Recall 0.832 · FPR 0.205 |

gru 를 고른 이유: bilstm 과 `fallen` 포착률이 같고(73.7%) FPR 이 가장 낮으며 파라미터가 1/3 이다.
lstm 은 Recall 0.891 로 가장 높지만 `fallen` 을 47.5% 밖에 못 잡아 "넘어진 뒤 방치" 목표에 약하다.
([results_final.md](results_final.md))

**학습 데이터의 알려진 편향** — `lying`(침대에 누움) 이 23개(1.9%)뿐이다. 모델이 수평 자세를
위험으로 배워, 침대에 엎드리는 장면을 낙상으로 오판한다. AI Hub 비낙상 샘플 FPR 0.750 의 원인이다.

## 미동 감지 규칙 — `ImmobilityDetector`

모델 위에 얹는 시간 누적 규칙. `realtime_demo.py`.

| 규칙 | 조건 | 알림 | 기본값 |
|---|---|---|---|
| **R1 낙상 후 방치** | 상태 ∈ {fall, fallen} 이 `t_fallen` 초 이상 연속 | `FALLEN_IMMOBILE` (CRITICAL) | 3.0s |
| **R2 누운 채 미동** | `motion_energy` < `eps` ∧ `torso_horizontality` > `horiz_thr` 가 `t_still` 초 지속 | `STILL_HORIZONTAL` (WARNING) | 5.0s · 0.02 · 1.0 |
| 낙상 순간 | `fall` 진입 에지 (쿨다운 `fall_cooldown`) | `FALL` | 2.0s |

설계 근거:

- **R1 이 fall 과 fallen 을 구분하지 않는 이유.** 모델의 fall↔fallen 경계는 흐리다
  (5-fold 에서 둘의 혼동은 흔하고, 위험 판정 자체는 바꾸지 않는다). 관심사는 정확한 라벨이 아니라
  **바닥에 있는 상태가 얼마나 지속됐나** 다.
- **R2 의 "몸통 수평" 조건.** 앉아서 가만히 있는 것은 정상이다. 자세 조건 없이 미동만 보면 오탐이 폭발한다.
- **검출기는 매 프레임 갱신, 모델은 stride 마다.** 지속 시간을 stride(0.5s) 단위로 양자화하면
  임계 근처에서 놓친다 — 실측: 2.91s 로 3.0s 를 0.09s 차이로 못 넘김.
- **히스테리시스.** 위험 진입 0.5 / 해제 0.35. 경계에서 normal↔fall 이 떨리며 FALL 이 중복 발화하던 것을 막는다.
- **지속 시간이 순간 오판을 걸러낸다.** 비낙상 샘플에서 FALL 오탐이 났지만 위험 상태가 2초 만에 풀려
  R1(3초)은 발화하지 않았다.

기본값은 AI Hub 샘플(낙상 후 4.7초 남음)에 맞춰 짧다. 실서비스에서는 `t_fallen` 을 10~30초로 올려야 한다.
전부 CLI 플래그다.

## 프로토타입 실측 (AI Hub 샘플 C1, M3 MPS)

| 영상 | 정답 낙상 시작 | FALL 감지 | 지연 | R1 발화 | 처리 FPS |
|---|---|---|---|---|---|
| 00003 전면낙상 | 260f | 290f | **+30f (0.5s)** | 470f | 26.4 |
| 00015 측면낙상 | 281f | 290f | **+9f (0.15s)** | 470f | 23.7 |
| 00151 후면낙상 | 375f | 406f | **+31f (0.5s)** | 586f | 30.0 |
| 00047 비낙상 | — | 116f (오탐) | — | **없음** | 26.3 |

낙상 3장면 전부 0.5초 이내 감지. 비낙상은 FALL 오탐 1건, CRITICAL 없음.

## 파일 지도

```
scripts/
  pipeline_core.py        정규화·리샘플·모델빌드·자세헬퍼 — 공통 모듈
  realtime_demo.py        실시간 프로토타입 (산출물 3)
  make_demo_report.py     데모 이벤트 → 시각 보고서
  extract_keypoints.py    학습용 키포인트 추출 (Colab 단독 실행용, normalize 복사본 유지)
  retrim_keypoints.py     edf 자르기 오류 수정
  pack_keypoints.py       .npy 다수 → .npz 하나 (Drive FUSE 병목 회피)
  train_stage2.py         6모델 학습 · k-fold · leave-one-out · --save-model
  build_manifest.py       staged + synthetic manifest
  aihub_infer.py          AI Hub 베이스라인(RandomForest) 재현
  aihub_pipeline_eval.py  AI Hub 샘플 슬라이딩 윈도우 평가
  make_pipeline_report.py AI Hub 평가 시각 보고서
  make_testset_report.py  OmniFall test 시각 보고서
runs/
  models/gru.pt, bilstm.pt     학습된 모델
  demo/*.mp4, *.jsonl          프로토타입 출력
  report/*.html, *.pdf         보고서 4종
  trim/, stage2/               학습 결과 JSON
docs/
  ARCHITECTURE.md   이 문서
  SUMMARY.md        프로젝트 현황 총정리
  results_final.md  학습 최종 수치 (논문 인용용)
  results_aihub_pipeline.md  베이스라인 비교
```

## 실행

```bash
./venv_aihub/bin/python scripts/realtime_demo.py \
  --source <영상.mp4> --model runs/models/gru.pt \
  --save-video runs/demo/out.mp4 --events runs/demo/out.jsonl \
  [--gt-json <AI Hub 라벨.json>]      # 정답 구간 오버레이
  [--no-display]                      # 화면 없이 저장만
  [--t-fallen 3 --t-still 5 --eps 0.02]
```

`events.jsonl` 한 줄: `{"frame", "t_sec", "state", "prob":[normal,fall,fallen], "event", "rule"}`.
stride 마다 상태 행(event=null), 이벤트 발생 시 이벤트 행.

## 알려진 한계 · 범위 밖

| 항목 | 상태 |
|---|---|
| 단일 인물 | 가장 큰 박스 1인. 다인 추적(ByteTrack)은 후속 |
| `lying` 오탐 | 학습 데이터 편향. 합성 데이터 보강이 해결책이나 Colab 세션 제약으로 미완 ([SUMMARY.md](SUMMARY.md) §2) |
| 침대/소파 맥락 | ROI 감점 미구현 |
| 웹캠 입력 | 사용자 결정으로 제외. `cv2.VideoCapture(0)` 으로 열 수 있다 |
| 알림 전송 | 이벤트 로그까지. 푸시·SMS 없음 |
| 평가 표본 | AI Hub 샘플 4장면 — 정성 검증용. 정량 수치는 OmniFall 5-fold |
