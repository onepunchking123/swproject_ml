# 낙상·미동 감지 파이프라인

치매환자·독거노인 모니터링을 위한 영상 기반 이상상황 감지. 단일 카메라 영상에서 **낙상**과
**낙상 후 방치(미동)** 를 실시간으로 잡는다. 졸업연구 + 공모전용 프로토타입.

```
영상 → YOLOv11n-pose → 17관절 좌표 → 카메라 불변 정규화 + 속도 → GRU → normal / fall / fallen
                                                                        ↓
                                                    규칙: 위험 상태 3초 지속 → 낙상 후 방치 (CRITICAL)
```

## 바로 실행

```bash
./venv_aihub/bin/python scripts/realtime_demo.py --source 00003_H_A_FY_C1
```

```bash
python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

데모 영상은 `data/aihub_sample/` 에 있어야 한다 (git 제외 — AI Hub 재배포 금지). 아래 **팀 Drive** 에서 받는다.
이름만 주면 찾고 같은 이름의 정답 라벨도 자동으로 붙는다. 창에 골격·상태·확률·FPS·알림 배너가 뜬다. `q` 로 종료.
임의 영상은 `--source path/to/clip.mp4`. 상세는 [docs/how_to_run.md](docs/how_to_run.md).

### 팀 Drive — git 에 없는 데이터

📁 **[falldata (Google Drive)](https://drive.google.com/drive/folders/1bgFi-Fg1-j8DAMG7fFPNnPsP4XqbxZph?usp=drive_link)** — 팀 내부 공유. 외부 재배포 금지 (AI Hub 이용약관).

| Drive 경로 | 받을 위치 | 용량 | 용도 |
|---|---|---|---|
| `aihub_sample.zip` | `data/` 에서 압축 해제 → `data/aihub_sample/` | 895MB | **데모·평가 영상 32 + 라벨** — 데모에 필수 |
| `kps_trim.npz` · `manifest_full.csv` | `runs/kps/` | 20MB | 재학습용 키포인트 (영상 없이 `train_stage2.py` 실행 가능) |
| `omnifall/` `omnifall_syn/` `fallvision/` `urfd/` | (선택) | 60GB | 원본 영상 zip — 키포인트 재추출·데이터 확장할 때만 |
| `models/` `runs_trim/` | (선택) | — | Colab 학습 산출물 원본 |

```bash
# 브라우저로 aihub_sample.zip 을 받아 data/ 에 풀거나, rclone 이 있으면:
rclone copyto gdrive:falldata/aihub_sample.zip data/aihub_sample.zip && unzip -q data/aihub_sample.zip -d data/
```

| 장면 | 이름 | 정답 낙상 | 데모 결과 (models/gru.pt) |
|---|---|---|---|
| 전면낙상 | `00003_H_A_FY_C1` | 4.3s | FALL 5.3s → CRITICAL 8.3s |
| 측면낙상 | `00015_H_A_SY_C1` | 4.7s | FALL 4.8s → CRITICAL 7.8s |
| 후면낙상 | `00151_H_A_BY_C1` | 6.3s | FALL 6.8s → CRITICAL 9.8s |
| 비낙상 | `00047_H_A_N_C1` | — | 침대에 엎드릴 때 FALL 오탐 1회 (2초 뒤 해제, CRITICAL 없음) |

M3 MPS 에서 36 FPS (4K → 960 축소).

---

## 1. 사용한 데이터셋

### 학습 — OmniFall staged 3종 (1,193 클립 · 피험자 19명 · 50분)

[OmniFall](https://arxiv.org/abs/2505.19889) 은 공개 낙상 데이터셋 8종을 16클래스로 재라벨링한 통합판이다
(Zenodo 17170592). 그중 키포인트 추출을 마친 3종으로 학습했다.

| 데이터셋 | 클립 | 피험자 | 환경 | 라이선스 |
|---|---|---|---|---|
| GMDCSA24 | 453 | 4 | 가정 3곳 (침대·의자) | MIT |
| edf | 483 | 5 | 실내, 2 카메라 동기 | 연구 목적 |
| caucafall | 258 | 10 | 가정 1곳 | 연구 목적 (Mendeley) |

16클래스를 3클래스로 접었다: `fall`(넘어지는 중) · `fallen`(넘어진 채) · 나머지 → `normal`.
클래스 분포 편향: `fallen` 179 (15%) 에 비해 `lying`(정상 누움) 은 **23 (1.9%)** 뿐이다 — 아래 4절.

### 평가 — AI Hub 「낙상사고 위험동작 영상-센서 쌍 데이터」 샘플 32영상

4장면(전면·측면·후면낙상·비낙상) × 8카메라, 병실, 10초 4K 60fps, 낙상 구간 프레임 라벨.
베이스라인 비교·카메라 편차·실시간 데모에 쓴다. 본편 491GB 는 해외 IP 차단으로 미확보.
AI Hub 공식 베이스라인(MediaPipe → RandomForest × 8카메라)은 재현해 두었다 ([docs/aihub_baseline.md](docs/aihub_baseline.md)).

### 확보했지만 아직 안 쓴 것 (팀 Drive `falldata/`)

OmniFall 나머지 4종(LE2I·MCFD·UP_Fall·OCCU, 26GB) · OF-Synthetic 12,000개(AV1, `lying` 1,200) · FallVision 16GB.
Colab 세션 회수로 키포인트 추출을 못 끝냈다. 추가 후보(CMDFALL 7시간, Toyota Smarthome 등)와
우선순위는 [docs/dataset_candidates.md](docs/dataset_candidates.md).

---

## 2. 파이프라인 구조

프레임마다 1~4, 0.5초마다 5~7, 프레임마다 8. 자세한 시나리오는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

| 단계 | 하는 일 | 왜 |
|---|---|---|
| 1. YOLOv11n-pose | 사람 검출 + COCO 17관절 (x, y, conf) | 영상 픽셀은 여기까지만 쓴다. 이후는 좌표만 |
| 2. 1인 선택 | 가장 큰 박스 한 명 | 학습 데이터가 1인 클립. 없으면 전부 0 |
| 3. 정규화 | 엉덩이 중점 → 원점, 몸통 길이 → 1 | **카메라 위치·해상도에 무관한 좌표.** AI Hub 가 카메라별 모델 8개를 둔 문제를 이 단계가 대신한다 |
| 4. 속도 채널 | 관절당 (Δx, Δy)·fps 추가 → 5채널 | 모델이 "모양" 이 아니라 "변화" 를 보게 한다. 엉덩이가 원점이라 발목 Δy = 높이 하락률 |
| 5. 링 버퍼 | 최근 2초(120프레임) 유지 | 낙상은 한 장으로는 알 수 없는 시간의 변화다 |
| 6. 리샘플 → GRU | 64프레임으로 맞춰 (64,17,5) 입력 → 확률 3개 | GRU 2층×128, 182K 파라미터. 시계열을 순서대로 읽는다 |
| 7. 히스테리시스 | 위험(fall+fallen) 확률 0.5 진입 / 0.35 해제 | 경계 근처 떨림 방지 |
| 8. 규칙 (`ImmobilityDetector`) | R1: 위험 상태 3초 연속 → `FALLEN_IMMOBILE` CRITICAL · R2: 몸통 수평 + 미동 5초 → `STILL_HORIZONTAL` WARNING · `fall` 진입 에지 → `FALL` | GRU 는 2초 창만 본다. 긴 시간은 규칙이 센다 |

**모델 = "이 2초에 무슨 일인가", 규칙 = "그 판정이 몇 초 이어지나."** 사람 검출률 50% 미만 창은 GRU 를 건너뛴다
(빈 입력에 `fallen` 0.79 를 내기 때문).

### 결과 요약

| | AI Hub 공식 베이스라인 | 본 파이프라인 |
|---|---|---|
| 입력 차원 | 997,200 (얼굴 랜드마크 84%) | 5,440 |
| 카메라 대응 | 위치별 모델 8개 | 단일 모델 |
| 낙상 Recall (AI Hub 32영상) | 0.833 | **0.917** |
| 비낙상 8영상 위험 윈도우 비율 | — | 0.162 (좌표만 쓸 때 0.346) |
| 5-fold Recall / FPR (OmniFall 피험자 단위) | — | 0.860 ± 0.026 / 0.131 |

수치의 출처: [docs/results_velocity.md](docs/results_velocity.md) (현재 모델) ·
[docs/results_final.md](docs/results_final.md) (좌표만 쓴 이전 모델, 모델 6종 비교) ·
[docs/results_aihub_pipeline.md](docs/results_aihub_pipeline.md) (AI Hub 카메라 편차).

---

## 3. 파일 역할

```
models/
  gru.pt                  배포 모델 (posvel, 5채널). 데모 기본값
  gru_pos.pt              좌표만 쓴 이전 모델 (비교용)

scripts/                  ── 현재 파이프라인 ──
  pipeline_core.py        공통 모듈: normalize · add_velocity · resample · build_model · load_checkpoint
  realtime_demo.py        영상 파일 → 실시간 판정 + 오버레이 + events.jsonl (ImmobilityDetector 포함)
  train_stage2.py         6모델 학습 · 피험자 k-fold · 데이터셋 leave-one-out · --feat pos|posvel · --save-model
  extract_keypoints.py    영상 → YOLO 키포인트 .npy (Colab 단독 실행용, normalize 복사본 유지)
  retrim_keypoints.py     edf 자르기 오류 수정 (fps 역산)
  pack_keypoints.py       .npy 다수 → .npz 하나 (Drive 병목 회피)
  build_manifest.py       OmniFall zip 구조 → manifest.csv
  expand_one.sh           데이터셋 하나씩 해제·추출 (Colab 세션 회수 대비)
  aihub_pipeline_eval.py  AI Hub 32영상 슬라이딩 윈도우 평가 → predictions.json
  aihub_infer.py          AI Hub 공식 베이스라인(RandomForest) 재현
  make_demo_report.py     events.jsonl + mp4 → 데모 보고서 HTML/PDF
  make_pipeline_report.py AI Hub 평가 시각 보고서
  make_testset_report.py  OmniFall test 시각 보고서
  make_report.py          베이스라인 보고서
scripts/archive/          AI Hub 스트리밍 다운로드 시도 · 데이터 확보 스크립트 (완료/중단)

docs/
  ARCHITECTURE.md         구조 · 10단계 동작 시나리오 · 모델 카드 · 규칙 명세 · 한계
  ISSUES.md               서비스 적용 전 문제점 12개 (팀 논의용)         ← 4절
  results_velocity.md     현재 모델: 속도 채널 실험 (pos vs posvel)
  results_final.md        이전 모델: 6모델 비교 · 5-fold · leave-one-out
  results_aihub_pipeline.md  AI Hub 샘플 평가 · 카메라 편차 · FPR 원인
  aihub_baseline.md / baseline_run.md  AI Hub 공식 모델 분석·재현
  dataset_candidates.md   데이터셋 후보와 우선순위
  how_to_run.md           팀원용 실행 가이드: 환경 · 데이터 받기 · 데모 · 평가 · 재학습
  setup.md                Colab CLI · Drive · rclone 함정 (학습 인프라 메모)
  SUMMARY.md              경과 총정리 (9-24 기준)
docs/archive/             계획서 · 중간 결과 (참고용)

configs/datasets.yaml     공개 데이터셋 레지스트리 (URL · 라이선스 · 구조)
requirements.txt          실행 환경 (Python 3.11)
data/aihub_sample/        데모·평가 영상 32 + 라벨       (git 제외)
runs/                     학습 결과 · 데모 출력 · 보고서 · 캐시  (git 제외)
aihub_model/              AI Hub 공식 모델 100MB          (git 제외)
```

**git 에 없는 것** — 위 "팀 Drive" 표. `yolo11n-pose.pt` 는 첫 실행 시 ultralytics 가 자동 다운로드.
`requirements.txt` 로 환경을 만든다 (Python 3.11).

---

## 4. 현재 찾은 문제점

전체는 [docs/ISSUES.md](docs/ISSUES.md) — 항목마다 관찰(실측) → 왜 문제인가 → 선택지 → 팀이 정해야 할 것.

| # | 문제 | 심각도 | 결정 필요 |
|---|---|---|---|
| 1 | 위험 상태가 한 판정만 `normal` 로 빠져도 3초 타이머가 리셋된다 | 높음 | 규칙 설계 — 연속 조건 vs 비율 조건 |
| 2 | 3초는 데모용. 서비스 값(30초? 3분?)은 데이터로 검증 불가 | 높음 | 서비스 정의 — 다단계 알림 |
| 3 | 낙상 직후 사람이 가려지면 "안 보임 = 판정 없음" 이 된다 | 높음 | `FALL_THEN_LOST` 이벤트 |
| 4 | `lying` 1.9% — 침대에 누우면 낙상으로 본다. 속도 채널로 줄였으나(R1 발화 4/8 → 1/8) 남아 있다 | 매우 높음 | 데이터 확장 장소 · 침대 ROI |
| 5 | 한국 노인 가정의 **바닥 생활**이 학습 데이터에 없다 — 정상 바닥 생활이 곧 `fallen` | 매우 높음 | 타깃 환경 (요양원 vs 가정) |
| 6 | 새 환경에서 FPR 0.13 → 0.20 (환경 교차 실측) | 높음 | 인지 |
| 7 | 1인 전제 — 보호자가 다가오면 대상이 바뀐다 | 중간 | 1차 타깃을 독거로 한정? |
| 8 | 클립당 FPR 은 서비스 지표가 아니다. **시간당 오경보**를 잴 긴 정상 영상이 없다 | 높음 | 평가 영상 확보 |
| 9 | RTSP·엣지·다중 카메라 미구현 | 중간 | 배포 형태 |
| 10 | 침실 카메라 프라이버시 — 현 구조는 골격만 저장하면 동작한다 | 중간 | 영상 저장 정책 |
| 11 | 임계값이 4장면에 맞춰짐 · 테스트 코드 없음 · 데이터 확장 미완 | 낮음 | — |
| 12 | YOLOv11 AGPL · 일부 데이터 CC-BY-NC — 공모전은 무관, 서비스화 시 교체 | 중간 | 서비스화 시 |

**지금 바로 할 수 있는 것**: `runs/demo/*.jsonl` 로 R1 규칙 변형 오프라인 비교(1번) · `FALL_THEN_LOST` 추가(3번) ·
`normalize()` 회귀 테스트(11번). **데이터가 있어야 하는 것**: 긴 정상 영상(8번), 좌식 생활 장면(5번), `lying` 대량(4번).
