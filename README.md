# 낙상·미동 감지 파이프라인

치매환자·독거노인 모니터링을 위한 영상 기반 이상상황 감지. **낙상**과 **낙상 후 방치(미동)**를
단일 카메라 영상에서 실시간으로 잡는다.

```
영상 → YOLOv11n-pose → 17 키포인트 → 카메라 불변 정규화 → GRU → normal / fall / fallen
                                                                    ↓
                                            규칙: 위험 상태 3초 지속 → 낙상 후 방치 (CRITICAL)
```

문서 두 개면 전체가 보인다:

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — 구조 · 모델 카드 · 미동 규칙 · 파일 지도
- **[docs/SUMMARY.md](docs/SUMMARY.md)** — 완료된 것 · 미완 · 논문 관점 완결성

## 바로 실행

```bash
./venv_aihub/bin/python scripts/realtime_demo.py --source 00003_H_A_FY_C1
```

데모 영상은 `data/aihub_sample/` 에 있다. 이름만 주면 찾고, 같은 이름의 라벨(정답 낙상 구간)도 자동으로 붙는다.
창이 뜨고 골격·상태·확률·FPS·알림 배너가 오버레이된다. `q` 로 종료. 임의 영상은 `--source path/to/clip.mp4`.

| 장면 | 이름 (C1~C8) | 정답 낙상 시작 |
|---|---|---|
| 전면낙상 | `00003_H_A_FY_C1` | 4.3초 |
| 측면낙상 | `00015_H_A_SY_C1` | 4.7초 |
| 후면낙상 | `00151_H_A_BY_C1` | 6.3초 |
| 비낙상 | `00047_H_A_N_C1` | — (침대에 엎드리는 장면에서 FALL 오탐 1회, CRITICAL 없음) |
환경 구성은 [docs/how_to_run.md](docs/how_to_run.md) §0.

## 결과 한눈에

| | AI Hub 공식 베이스라인 | 본 파이프라인 |
|---|---|---|
| 입력 차원 | 997,200 (얼굴 랜드마크 84%) | **20,400** |
| 카메라 대응 | 위치별 모델 8개 | **단일 모델** |
| 낙상 Recall (AI Hub 샘플 32영상) | 0.833 | **0.917** |
| 카메라 판정 편차 (같은 장면 8각도) | 0.30 | **0.213** |
| 5-fold Recall / FPR (OmniFall, 피험자 단위) | — | 0.867 / 0.112 |
| 실시간 처리 (M3 MPS) | — | 23~30 FPS · 감지 지연 ≤ 0.5초 |

**알려진 한계** — 학습 데이터에 `lying`(침대에 누움)이 1.9% 뿐이라 침대에 엎드리는 장면을
낙상으로 오판한다. 다만 지속 시간 규칙 덕에 CRITICAL 로는 이어지지 않는다.
원인·해결책은 [docs/results_aihub_pipeline.md](docs/results_aihub_pipeline.md) · [docs/dataset_candidates.md](docs/dataset_candidates.md).

## 디렉토리

```
scripts/            현재 파이프라인 (pipeline_core · realtime_demo · train_stage2 · 보고서 생성기)
scripts/archive/    데이터 확보·AI Hub 스트리밍 등 완료/중단된 스크립트
docs/               결과 · 구조 · 실행법 (9개)
docs/archive/       계획서 · 경과 기록 · 오염 데이터 중간 결과
configs/            공개 데이터셋 레지스트리 (datasets.yaml)
data/aihub_sample/  데모·평가용 AI Hub 샘플 32영상 + 라벨  (git 제외)
runs/               모델 · 데모 출력 · 보고서 · 캐시  (git 제외)
aihub_model/        AI Hub 공식 베이스라인 모델 (RandomForest)  (git 제외)
```

## 데이터

| 출처 | 용도 | 상태 |
|---|---|---|
| **OmniFall** (Zenodo, 8종 통합) | 학습 · 5-fold · leave-one-out | 3종 1,194클립 학습 완료 · 4종 + 합성 12k 은 Drive 확보, 추출 미완 |
| **AI Hub 낙상 데이터** 샘플 32영상 | 베이스라인 비교 · 카메라 편차 · 실시간 데모 | 확보. 본편 491GB 는 해외 IP 차단으로 미확보 |
| AI Hub 공식 모델 | 비교 대상 | 재현 완료 ([docs/aihub_baseline.md](docs/aihub_baseline.md) · [docs/baseline_run.md](docs/baseline_run.md)) |

라이선스: Le2i·URFD 는 CC-BY-NC. 서비스 탑재 모델은 비상업 데이터를 빼고 학습해야 한다.

## Colab

학습은 Colab T4 에서 했다. CLI 설치와 의존성 함정은 [docs/setup.md](docs/setup.md).
세션이 예고 없이 회수되므로 **중간 산출물은 Drive 에 바로 쓴다** — 이 원칙이 없어 세 번 잃었다.
