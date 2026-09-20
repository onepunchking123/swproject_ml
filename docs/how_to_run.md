# 실행 방법

AI Hub 공식 영상 모델(RandomForest) 베이스라인 추론을 돌리는 절차.

## 0. 준비 — 처음 한 번만

### 작업 폴더로 이동

```bash
cd ~/Documents/fall-detection
```

앞으로 **모든 명령은 이 폴더에서** 실행한다.

### 가상환경 만들기

```bash
uv venv --python 3.11 venv_aihub
```

```bash
uv pip install --python venv_aihub/bin/python "scikit-learn==1.3.2" "mediapipe==0.10.14" opencv-python joblib
```

```bash
uv pip install --python venv_aihub/bin/python "numpy==1.26.4"
```

**numpy 설치는 반드시 마지막에, 따로 실행한다.** mediapipe 가 numpy 2.x 를
끌어오는데 그러면 scikit-learn 이 `numpy.dtype size changed` 오류로 깨진다.

### 설치 확인

```bash
./venv_aihub/bin/python -c "import numpy,sklearn,mediapipe,cv2,joblib; print(numpy.__version__, sklearn.__version__, mediapipe.__version__)"
```

`1.26.4 1.3.2 0.10.14` 가 나와야 한다.

## 1. 추론 실행

### 낙상유무 탐지 (FNF) — 2분류

```bash
cd ~/Documents/fall-detection && ./venv_aihub/bin/python scripts/aihub_infer.py --data "/Users/mymoon/Downloads/Sample 2/01.원천데이터/영상" --models "/Users/mymoon/Downloads/낙상사고 위험동작 영상-센서 쌍 데이터/1.모델/2. AI학습모델파일/영상" --task fnf
```

### 낙상유형 분류 (FD) — 3분류

```bash
cd ~/Documents/fall-detection && ./venv_aihub/bin/python scripts/aihub_infer.py --data "/Users/mymoon/Downloads/Sample 2/01.원천데이터/영상" --models "/Users/mymoon/Downloads/낙상사고 위험동작 영상-센서 쌍 데이터/1.모델/2. AI학습모델파일/영상" --task fd
```

FD 는 낙상 유형(전면/후면/측면)만 구분하므로 비낙상(N) 영상은 자동 제외된다.

### 빠르게 확인만 하고 싶을 때

```bash
cd ~/Documents/fall-detection && ./venv_aihub/bin/python scripts/aihub_infer.py --data "/Users/mymoon/Downloads/Sample 2/01.원천데이터/영상" --models "/Users/mymoon/Downloads/낙상사고 위험동작 영상-센서 쌍 데이터/1.모델/2. AI학습모델파일/영상" --task fnf --limit 4
```

`--limit 4` 는 영상 4개만 처리한다. 전체는 약 9분, 4개는 약 1분.

### 시간이 오래 걸릴 때 (백그라운드)

```bash
cd ~/Documents/fall-detection && nohup ./venv_aihub/bin/python scripts/aihub_infer.py --data "/Users/mymoon/Downloads/Sample 2/01.원천데이터/영상" --models "/Users/mymoon/Downloads/낙상사고 위험동작 영상-센서 쌍 데이터/1.모델/2. AI학습모델파일/영상" --task fnf > runs/fnf.log 2>&1 &
```

진행 상황 확인:

```bash
cd ~/Documents/fall-detection && ls runs/keypoints/*.npy | wc -l
```

처리된 영상 수가 나온다 (전체 32개).

## 2. 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--data` | (필수) | 영상 루트. 하위 폴더를 재귀 탐색한다 |
| `--models` | (필수) | `2. AI학습모델파일/영상` 경로 |
| `--task` | `fnf` | `fnf`=낙상유무 2분류, `fd`=낙상유형 3분류 |
| `--limit` | 전체 | 처리할 영상 수 제한 |
| `--resize` | 960 | MediaPipe 입력 가로 크기. `0` 이면 4K 원본 (매우 느림) |
| `--cache` | `runs/keypoints` | 키포인트 캐시 위치 |
| `--out` | `runs/aihub_baseline` | 결과 저장 위치 |

## 3. 결과가 저장되는 곳

```
~/Documents/fall-detection/
└── runs/
    ├── keypoints/                  ← 키포인트 캐시 (.npy)
    │   ├── 00003_H_A_FY_C1.npy         영상당 약 4MB
    │   └── ...                         32개 전부면 약 128MB
    │
    └── aihub_baseline/            ← 추론 결과
        ├── fnf_results.json           낙상유무
        └── fd_results.json            낙상유형
```

### 키포인트 캐시

**한 번 추출하면 재사용된다.** 추출이 영상당 17초로 전체 시간의 대부분이므로,
같은 영상을 다시 돌리면 즉시 끝난다. 로그에 `(캐시)` 로 표시된다.

다시 추출하고 싶으면 지운다:

```bash
rm -rf ~/Documents/fall-detection/runs/keypoints
```

`.gitignore` 로 제외되어 GitHub 에는 올라가지 않는다.

### 결과 JSON

```json
{
  "task": "fnf",
  "n": 32,
  "accuracy": 0.875,
  "results": [
    {
      "file": "00003_H_A_FY_C1",
      "cam": 1,
      "true": "FY",
      "pred_idx": 0,
      "pred": "FALL",
      "prob": [0.83, 0.17]
    }
  ]
}
```

| 필드 | 의미 |
|---|---|
| `file` | 영상 파일명 |
| `cam` | 카메라 번호 (1~8) — 어느 모델을 썼는지 |
| `true` | 실제 라벨 (FY/BY/SY/N) |
| `pred` | 예측 (FALL/NonFall 또는 BY/FY/SY) |
| `prob` | 클래스별 확률 |

## 4. 화면에 나오는 것

```
[*] task=fnf  영상 32개  모델 낙상분류
  [1/32] 00003_H_A_FY_C1  cam=C1  true=FY  (17초)
  [2/32] 00003_H_A_FY_C2  cam=C2  true=FY  (캐시)
  ...
[*] 키포인트 추출 완료 (540초)

[*] C1: 4개 추론 완료
...

파일                          cam     실제        예측         확률
--------------------------------------------------------------------
00003_H_A_FY_C1              C1     FY       FALL      0.83 0.17
...
--------------------------------------------------------------------
정확도: 28/32 = 87.5%

Confusion matrix (행=실제, 열=예측)
              FALL   NonFall
      FALL      22         2
   NonFall       2         6

결과 저장: runs/aihub_baseline/fnf_results.json
```

## 5. 다른 데이터로 돌리기

`--data` 만 바꾸면 된다. 조건은 두 가지다.

- **영상이 600프레임 이상**이어야 한다. 미만이면 건너뛴다
- **파일명에 `_C1` ~ `_C8`** 이 있어야 카메라 모델을 고를 수 있다

Validation 원천데이터(VS.zip) 를 받으면:

```bash
cd ~/Documents/fall-detection && ./venv_aihub/bin/python scripts/aihub_infer.py --data /Volumes/외장하드/VS --models "/Users/mymoon/Downloads/낙상사고 위험동작 영상-센서 쌍 데이터/1.모델/2. AI학습모델파일/영상" --task fnf
```

## 6. 문제가 생기면

### `numpy.dtype size changed`

numpy 가 2.x 로 올라간 것이다.

```bash
cd ~/Documents/fall-detection && uv pip install --python venv_aihub/bin/python "numpy==1.26.4"
```

### `module 'mediapipe' has no attribute 'solutions'`

mediapipe 가 1.x 다. legacy API 가 제거되었으므로 내린다.

```bash
cd ~/Documents/fall-detection && uv pip install --python venv_aihub/bin/python "mediapipe==0.10.14" && uv pip install --python venv_aihub/bin/python "numpy==1.26.4"
```

### 경고 메시지가 너무 많다

MediaPipe 의 정상 출력이다. 무시해도 된다. 걸러내려면:

```bash
cd ~/Documents/fall-detection && ./venv_aihub/bin/python scripts/aihub_infer.py --data "/Users/mymoon/Downloads/Sample 2/01.원천데이터/영상" --models "/Users/mymoon/Downloads/낙상사고 위험동작 영상-센서 쌍 데이터/1.모델/2. AI학습모델파일/영상" --task fnf 2>&1 | grep -v "W0000\|I0000\|absl\|XNNPACK\|feedback"
```

### 영상을 못 찾는다

`--data` 경로에 공백이 있으면 **따옴표로 감싸야 한다.**
`Sample 2`, `낙상사고 위험동작...` 모두 공백이 있다.

## 7. 시각 보고서 만들기

추론 결과를 영상 썸네일과 함께 HTML 로 정리한다. 어떤 장면에서 왜 틀렸는지
눈으로 확인할 수 있다.

```bash
cd ~/Documents/fall-detection && ./venv_aihub/bin/python scripts/make_report.py --data "/Users/mymoon/Downloads/Sample 2/01.원천데이터/영상" --task fnf
```

```bash
cd ~/Documents/fall-detection && ./venv_aihub/bin/python scripts/make_report.py --data "/Users/mymoon/Downloads/Sample 2/01.원천데이터/영상" --task fd
```

저장 위치: `runs/report/fnf_report.html`, `runs/report/fd_report.html`

브라우저로 열기:

```bash
open ~/Documents/fall-detection/runs/report/fnf_report.html
```

### 보고서에 담기는 것

- 정확도 · Recall · Precision · FPR · 놓친 낙상 수
- Confusion matrix (대각선 초록, 오분류 빨강)
- **카메라별 정확도** — 어느 각도가 취약한지
- **영상별 카드** — 시간순 프레임 5장 + 실제/예측 라벨 + 클래스별 확률
  - 오답은 빨간 테두리로 구분된다
  - 같은 장면이 카메라별로 묶여 있어 판정이 갈리는 지점을 바로 볼 수 있다

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--frames` | 5 | 영상당 추출할 프레임 수 |
| `--width` | 300 | 썸네일 가로 픽셀 |
| `--out` | `runs/report` | 저장 위치 |

이미지는 base64 로 HTML 에 내장되므로 파일 하나만 옮기면 어디서든 열린다
(FNF 약 2MB, FD 약 1.5MB).

### PDF 로 뽑기

`--pdf` 를 붙이면 HTML 과 함께 PDF 도 만든다.

```bash
cd ~/Documents/fall-detection && ./venv_aihub/bin/python scripts/make_report.py --data "/Users/mymoon/Downloads/Sample 2/01.원천데이터/영상" --task fnf --pdf
```

저장 위치: `runs/report/fnf_report.pdf` (A4, 약 1.5MB, 12페이지)

처음 한 번만 변환 라이브러리를 설치한다.

```bash
uv pip install --python venv_aihub/bin/python weasyprint
```

**weasyprint 를 쓰는 이유.** Chrome headless(`--print-to-pdf`)도 가능하지만,
base64 이미지가 160장 들어간 2MB HTML 에서 렌더링이 5분을 넘겨도 끝나지 않았다.
weasyprint 는 같은 작업을 23초에 끝낸다. 스크립트는 weasyprint 를 우선 쓰고,
없으면 Chrome 으로 넘어간다.

인쇄용 스타일이 따로 들어 있어 PDF 에서는:

- 항상 라이트 테마로 고정된다 (다크 모드 화면이어도 인쇄물은 흰 배경)
- 카드가 페이지 경계에서 잘리지 않는다
- 가로 스크롤되던 프레임 5장이 한 줄에 균등 배치된다

### 수동으로 PDF 만들기

`--pdf` 가 실패하면 브라우저에서 직접 뽑아도 된다.

```bash
open ~/Documents/fall-detection/runs/report/fnf_report.html
```

⌘P → 대상을 **"PDF로 저장"** → 저장. 인쇄 스타일이 HTML 에 포함되어 있어
결과물은 동일하다.
