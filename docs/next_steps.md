# 다음 작업 계획

_작성: 2026-09-20_

## 오늘까지의 결론

세 가지가 확정됐다.

1. **AI Hub 데이터 다운로드는 한국 IP 에서만 가능하다.** Colab(미국 리전)에서
   직접 받는 경로는 막혔다. 맥북(한국 IP)을 거쳐야 하지만 SSD 여유가 18GB 뿐이다.
2. **데이터는 총 491GB**, Training 원천은 436GB 분할 zip 한 덩어리라 쪼갤 수 없다.
   Validation 원천(VS.zip, 55GB)은 단일 파일이라 상대적으로 다루기 쉽다.
3. **AI Hub 공식 모델은 영상 단독 실행이 가능하다.** 센서 없이도 된다.
   다만 카메라 위치별 RandomForest 8개라 고정 카메라를 전제하며 사람 검출 단계가 없다.

→ 외장 SSD 확보를 기다리는 동안, **웹에서 받을 수 있는 것**으로 베이스라인을 돌린다.

## AI Hub 모델 실행 계획

### 필요한 파일

[데이터셋 페이지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71641)
에서 로그인 후 직접 받는다. API(`aihubshell`)로는 `1.데이터` 만 노출되므로 웹에서 받아야 한다.

| 항목 | 상태 | 용도 |
|---|---|---|
| AI 모델 상세 설명서 | **확보** (`2.모델_doc`) | 사용법 문서 |
| **AI 모델 다운로드** | 다운로드 중 | 소스코드 + 학습된 RF 모델 |
| **샘플(경량) 데이터** | 미확보 | 테스트용 영상 — 외장하드 없이 돌리기 위해 필요 |

샘플 데이터는 민감정보가 마스킹된 경량본이라 맥북 용량으로 감당된다.

### 1단계 — 환경 구성

현재 시스템 Python 은 3.14 라 별도 환경이 필요하다 (모델은 3.9 기준).

```bash
uv venv --python 3.9 ~/Documents/fall-detection/venv_aihub
source ~/Documents/fall-detection/venv_aihub/bin/activate
pip install -r requirements.txt
```

**Apple Silicon 에서 예상되는 문제:**

| 패키지 | 문제 | 대응 |
|---|---|---|
| `tensorflow==2.14.0` | M3 용 빌드가 별도 | 영상 RF 만 쓸 거면 **불필요** — 센서 LSTM/CNN 전용 |
| `datatable` (git 설치) | arm64 휠 부재 가능 | 실패 시 제외하고 진행 |
| `xgboost`, `imbalanced-learn` | 대체로 무난 | |

영상 모델은 **scikit-learn RandomForest** 이므로 실제로 필요한 것은
`scikit-learn`, `numpy`, `pandas`, `opencv`, `joblib` 정도다.
전체 설치가 막히면 최소 구성으로 줄인다.

### 2단계 — 영상 경로만 분리

소스코드에서 센서·앙상블을 제외하고 영상 RF 경로만 실행한다.

- **FNF (낙상유무)** 부터 — 목표가 "낙상인지 아닌지" 판단이므로 이진 분류가 우선
- FD (낙상유형: 전면/후면/측면) 는 그 다음
- `util/util_nia_project.py` 에 공통 모듈이 있다

확인할 것:

- 영상 → 특징 벡터 변환 코드의 위치와 입력 형식
  (RandomForest 는 이미지를 직접 받지 않는다)
- 카메라 위치별 모델 8개 중 샘플 영상에 맞는 것을 고르는 기준

### 3단계 — 추론 실행

문서 권장사항: **test size 500~700개를 넘지 않을 것.**

Docker 도 제공되지만(`fnf_docker.tar`) Apple Silicon 에서는 x86 에뮬레이션으로
느려질 수 있어 소스 직접 실행을 우선한다. Docker 를 쓸 경우:

```bash
docker load -i fnf_docker.tar
docker run -it --oom-kill-disable <이미지명>
```

### 4단계 — 결과 기록

문서상 "confusion matrix 로 마무리되어야 함". 다음을 기록한다.

| 항목 | 비고 |
|---|---|
| Confusion matrix | TP/FP/TN/FN |
| **Recall** | 낙상 감지에서 최우선 지표 |
| **FPR** | 경보 피로(alarm fatigue) 판단용 |
| Precision, F1, Accuracy | |
| 추론 속도 | 프레임당 ms |
| 사용한 카메라 위치 모델 | 8개 중 어느 것 |
| test 샘플 수 | |

이 수치가 **논문 비교표의 베이스라인 행**이 된다.

## 외장 SSD 확보 후

### 받을 것 (우선순위 순)

| 파일 | 크기 | filekey | 우선순위 |
|---|---|---|---|
| TL.zip (Training 라벨) | 126 MB | 531136 | **1 — 즉시** |
| VL.zip (Validation 라벨) | 16 MB | 531138 | **1 — 즉시** |
| VS.zip (Validation 원천) | 55 GB | 531137 | 2 |
| TS.z01~z04 + TS.zip | 436 GB | 531131~531135 | 3 — 판단 후 |

```bash
aihubshell -mode d -datasetkey 71641 -filekey 531136,531138 \
  -aihubapikey "$(cat ~/.aihub_key)"
```

### 판단이 필요한 지점

**Training 436GB 를 전부 받을 것인가, Validation 55GB 로 Stage 1 을 학습할 것인가.**

- 55GB 만 쓰면 인프라 부담이 크게 줄지만 데이터가 적다
- 436GB 는 분할 불가라 한 번에 받아야 하고, 압축 해제에 2~3배 용량이 더 필요하다
  (실질 1TB 이상)

라벨(142MB)을 먼저 분석해 클립 수·다양성·카메라 위치 분포를 확인한 뒤 결정한다.

## 라벨만으로 가능한 작업 (SSD 불필요)

라벨 142MB 는 맥북에서 바로 받을 수 있다. 이것만으로 상당 부분이 진행된다.

- **실제 JSON 스키마 확인** → 추측이 아닌 정확한 전처리 코드 작성
- 클래스 분포, 클립 구조, 키포인트 형식 파악
- **Stage 2 전체** — LSTM / GRU / 1D-CNN / BiLSTM+Attention / ST-GCN 비교.
  입력이 키포인트 좌표뿐이라 이미지가 필요 없고, 모델이 작아 M3 에서 학습된다
- 규칙 기반 베이스라인 (종횡비 + 몸통각도 + 하강속도)

## 병행할 것

- **Google Drive 400GB** 학생 계정 신청 — 전처리 결과·체크포인트 보관용
- **교내 GPU 서버 문의** (창원대 전산원 / 지도교수 연구실) —
  한국 IP + 저장공간 + GPU 를 무료로 해결할 수 있는 최선의 경로
- 공개 데이터셋(Le2i / URFD)으로 YOLOv11-pose + ByteTrack 프로토타입.
  AI Hub 이미지 확보 즉시 학습만 돌리면 되도록

## 정리된 논문 구도

| 방법 | 카메라 독립 | 사람 검출 | 센서 | Recall | FPR |
|---|---|---|---|---|---|
| AI Hub RF (영상) — **오늘 돌릴 것** | ✗ | ✗ | ✗ | | |
| AI Hub 앙상블 | ✗ | ✗ | ✓ | | |
| 제안: YOLOv11-pose + 규칙 | ✓ | ✓ | ✗ | | |
| 제안: YOLOv11-pose + LSTM | ✓ | ✓ | ✗ | | |
| 제안: YOLOv11-pose + ST-GCN | ✓ | ✓ | ✗ | | |

> 센서 없이 영상만으로 시계열을 제대로 다루는 것이 본 연구의 논지다.
> AI Hub 영상 모델이 시간 구조에 약한 것은 센서가 없어서가 아니라
> RandomForest(정적 분류기)를 썼기 때문이며, 키포인트 시퀀스를 LSTM/ST-GCN 으로
> 학습하면 영상만으로도 시간 정보를 활용할 수 있다.
