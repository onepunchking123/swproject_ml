# AI Hub 베이스라인 재현

_실행: 2026-09-20_

AI Hub 공식 영상 모델(RandomForest)을 샘플 데이터로 돌려 베이스라인 수치를 얻는다.

## 환경

원 소스는 Python 3.9 / TensorFlow 2.14 를 요구하지만, **영상 RF 만 쓸 경우
TensorFlow 는 불필요하다** (센서 LSTM/CNN 전용). 실제로 필요한 것은 다음뿐이다.

```bash
uv venv --python 3.11 venv_aihub
uv pip install --python venv_aihub/bin/python \
    "scikit-learn==1.3.2" "mediapipe==0.10.14" "numpy==1.26.4" opencv-python joblib
```

### 버전을 고정해야 하는 이유

| 패키지 | 고정 버전 | 이유 |
|---|---|---|
| `scikit-learn` | 1.3.2 | 학습 시 버전. 다르면 `.pkl` 로드 시 경고·오류 |
| `mediapipe` | 0.10.14 | **1.0 에서 `mp.solutions` legacy API 가 제거됨.** 원 코드가 쓰는 `Holistic` 이 여기 있다 |
| `numpy` | 1.26.4 | mediapipe 설치 시 2.x 가 딸려와 sklearn 과 ABI 충돌 (`numpy.dtype size changed`) |

`numpy` 는 mediapipe 설치 **후에** 다시 1.x 로 내려야 한다. 순서가 중요하다.

## 샘플 데이터

897MB. 각 클래스의 같은 장면을 카메라 8대(C1~C8)로 촬영했다.

| 클래스 | 장면 ID | 영상 수 |
|---|---|---|
| FY (전면낙상) | 00003 | 8 |
| SY (측면낙상) | 00015 | 8 |
| N (비낙상) | 00047 | 8 |
| BY (후면낙상) | 00151 | 8 |

**전 영상이 정확히 600프레임 / 60fps / 3840×2160.** 모델이 요구하는
600프레임과 정확히 일치한다 — 데이터셋이 그렇게 설계되어 있다.

장면이 4개뿐이므로 통계적으로 유의한 평가는 불가능하다. 파이프라인 검증과
동작 확인이 목적이다.

## 재현 스크립트

[`scripts/aihub_infer.py`](../scripts/aihub_infer.py)

원 소스(`video_mediapipe.ipynb` + `video_smote_train.ipynb`)의 파이프라인을
그대로 따르되, 하드코딩된 개발자 로컬 경로(`/Users/kimmonica/...`,
`/Volumes/MonicaSSD/...`)를 제거하고 CLI 로 만들었다.
`extract_keypoints()` 구현은 원본과 동일하다.

```bash
./venv_aihub/bin/python scripts/aihub_infer.py \
  --data   "<샘플>/01.원천데이터/영상" \
  --models "<모델>/2. AI학습모델파일/영상" \
  --task fnf          # 또는 fd
```

### 처리 흐름

1. 파일명에서 카메라 번호(`_C1`~`_C8`) 판별 → 해당 `.pkl` 선택
2. MediaPipe Holistic 으로 프레임별 1,662 차원 추출
   (`min_detection_confidence=0.1`, `min_tracking_confidence=0.1` — 원본과 동일)
3. 600 프레임 → `reshape(1, 997200)`
4. `RandomForestClassifier.predict()`

### 구현상 판단

- **4K → 가로 960 리사이즈.** 원본 해상도로 MediaPipe 를 돌리면 매우 느리다.
  랜드마크 좌표는 0~1 정규화되어 반환되므로 리사이즈가 결과에 영향을 주지 않는다.
- **키포인트 캐시** (`runs/keypoints/*.npy`). 추출이 영상당 약 17초로
  전체의 대부분을 차지하므로, 재실행 시 재사용한다.
- **카메라별 배치 추론.** 샘플 하나가 997,200 × 8바이트 ≈ 8MB 이므로
  전체를 한 번에 올리면 메모리를 많이 쓴다. 카메라 단위로 나눠 처리하고
  모델을 즉시 해제한다.

## 성능

| 항목 | 값 |
|---|---|
| 키포인트 추출 | 영상당 약 17초 (M3, 4K→960 리사이즈) |
| 전체 32개 | 약 9분 |
| RF 추론 | 즉시 |

병목은 전적으로 MediaPipe 추출이다.

## 결과

_(전체 실행 완료 후 기록)_

### FNF — 낙상유무 2분류

| 항목 | 값 |
|---|---|
| 정확도 | |
| Recall (낙상) | |
| FPR | |

### FD — 낙상유형 3분류

| 항목 | 값 |
|---|---|
| 정확도 | |

## 주의

샘플이 4개 장면뿐이라 이 수치를 논문에 그대로 쓸 수는 없다.
**본 평가는 Validation 원천데이터(VS.zip, 55GB) 확보 후 다시 수행한다.**
지금은 파이프라인이 동작함을 확인하고, 재현 절차를 확정하는 것이 목적이다.
