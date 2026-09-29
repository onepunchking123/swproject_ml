# 팀원용 실행 가이드

clone 부터 데모 → 평가 → 재학습까지. 모든 명령은 저장소 루트에서 실행한다.

## 0. 환경 (처음 한 번)

Python **3.11** 이 필요하다 (3.12 이상은 mediapipe 0.10 휠이 없다 — 데모만 돌리면 상관없다).

```bash
git clone https://github.com/onepunchking123/swproject_ml.git fall-detection && cd fall-detection
```

```bash
python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

`requirements.txt` 는 세 묶음이다 — 데모용 / 보고서 PDF 용(weasyprint) / AI Hub 베이스라인 재현용(mediapipe·scikit-learn).
데모만 볼 거면 아래 두 묶음은 주석 처리해도 된다. weasyprint 는 시스템 라이브러리가 필요하다:
macOS `brew install pango` · Ubuntu `sudo apt install libpango-1.0-0 libpangoft2-1.0-0`.

확인:

```bash
python -c "import torch, ultralytics, cv2; print(torch.__version__, 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))"
```

GPU 가 없어도 CPU 로 돈다 (데모 약 10 FPS).

## 1. 데이터 받기 (팀 Drive)

📁 [falldata](https://drive.google.com/drive/folders/1bgFi-Fg1-j8DAMG7fFPNnPsP4XqbxZph?usp=drive_link) — 팀 내부 공유. **외부 재배포 금지** (AI Hub 이용약관).

| 받을 것 | 놓을 곳 | 필요한 때 |
|---|---|---|
| `aihub_sample.zip` (895MB) | `data/` 에 풀어서 `data/aihub_sample/` | 데모 · AI Hub 평가 |
| `kps_trim.npz` · `manifest_full.csv` (20MB) | `runs/kps/` | 재학습 |

```bash
mkdir -p data runs/kps
unzip -q ~/Downloads/aihub_sample.zip -d data/
mv ~/Downloads/kps_trim.npz ~/Downloads/manifest_full.csv runs/kps/
```

확인: `ls data/aihub_sample` 에 `01.원천데이터` `02.라벨링데이터` 가 보이면 된다.

`yolo11n-pose.pt` 는 첫 실행 때 ultralytics 가 자동으로 받는다 (6MB).

## 2. 데모 — 영상 하나 실시간 판정

```bash
python scripts/realtime_demo.py --source 00003_H_A_FY_C1
```

창이 뜨고 골격·상태(normal/fall/fallen/no_person)·확률·FPS 가 오버레이된다. 정답 낙상 구간이
라벨에서 자동으로 붙어 화면에 표시된다. `q` 로 종료.

| 이름 | 장면 | 기대 결과 |
|---|---|---|
| `00003_H_A_FY_C1` | 전면낙상 (4.3s) | FALL ≈5.3s → 3초 뒤 **FALLEN_IMMOBILE** (빨간 배너) |
| `00015_H_A_SY_C1` | 측면낙상 (4.7s) | FALL ≈4.8s → FALLEN_IMMOBILE |
| `00151_H_A_BY_C1` | 후면낙상 (6.3s) | FALL ≈6.8s → FALLEN_IMMOBILE |
| `00047_H_A_N_C1` | 비낙상 (침대에 엎드림) | FALL 오탐 1회, CRITICAL 은 나지 않아야 함 |

`_C1` 을 `_C2`~`_C8` 로 바꾸면 같은 장면의 다른 카메라다.

자주 쓰는 옵션:

```bash
# 임의 영상 · 화면 없이 · 결과 영상과 이벤트 로그 저장
python scripts/realtime_demo.py --source path/to/clip.mp4 --no-display --save-video out.mp4 --events out.jsonl
```

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--t-fallen` | 3.0 | R1: 위험 상태 몇 초 지속되면 CRITICAL (데모용 값. [ISSUES.md](ISSUES.md) 2번) |
| `--thr` / `--thr-exit` | 0.5 / 0.35 | 위험 진입 / 해제 확률 (히스테리시스) |
| `--min-det` | 0.5 | 윈도우 내 사람 검출률 미만이면 `no_person` |
| `--device` | auto | `mps` · `cuda` · `cpu` |
| `--model` | `models/gru.pt` | `models/gru_pos.pt` 로 바꾸면 속도 채널 없는 이전 모델 |

`events.jsonl` 은 0.5초마다 한 줄(`state`, `prob`) + 이벤트 줄(`event`, `rule`). 규칙 변형을 시험할 때 YOLO 를 다시 돌리지 않고 이 파일만 재생하면 된다.

## 3. 데모 보고서 (PDF)

4영상을 저장 모드로 돌린 뒤 보고서를 만든다.

```bash
for s in 00003_H_A_FY_C1 00015_H_A_SY_C1 00151_H_A_BY_C1 00047_H_A_N_C1; do
  python scripts/realtime_demo.py --source $s --no-display --save-video runs/demo/${s%%_H*}_C1.mp4 --events runs/demo/${s%%_H*}_C1.jsonl
done
python scripts/make_demo_report.py --pdf
```

→ `runs/report/demo_report.pdf` — 영상마다 상태 띠 · 위험 확률 곡선(정답 구간 음영) · 이벤트 시점 스냅샷.

## 4. AI Hub 32영상 정량 평가

```bash
python scripts/aihub_pipeline_eval.py --sample data/aihub_sample --model models/gru.pt --out runs/aihub_sample
python scripts/make_pipeline_report.py --pred runs/aihub_sample/predictions.json --sample data/aihub_sample --pdf
```

첫 실행은 YOLO 추출에 영상당 ~15초(MPS). 키포인트는 `runs/aihub_kps/` 에 캐시되어 두 번째부터는 수 초.
→ `runs/report/pipeline_aihub.pdf` — 카메라별 판정, 낙상 Recall, 비낙상 FPR, 감지 지연.

## 5. 재학습 (영상 없이, 키포인트 캐시로)

```bash
# 배포 모델과 같은 설정 — 피험자 5-fold
python scripts/train_stage2.py --kps runs/kps/kps_trim.npz --manifest runs/kps/manifest_full.csv \
    --model gru --feat posvel --kfold 5 --out runs/mine
```

M3 MPS 에서 fold 당 20초. 결과는 `runs/mine/results_risk_kfold5_posvel.json` (fold 별 혼동행렬 포함).

| 옵션 | 뜻 |
|---|---|
| `--model gru\|lstm\|bilstm\|cnn1d\|stgcn\|rule` · `--all` | 모델 선택 |
| `--feat pos\|posvel` | 좌표 3채널 / 좌표+속도 5채널 |
| `--kfold 5` · `--loo-dataset` | 피험자 5-fold / 데이터셋 하나를 통째로 test |
| `--save-model DIR` | 가중치 저장 (`DIR/gru.pt`). 데모에 `--model DIR/gru.pt` 로 바로 쓸 수 있다 |

같은 실행을 다시 하면 끝난 조합은 건너뛴다 (결과 파일 증분 저장).

배포 모델을 바꾸려면 단일 split 로 학습해 저장하고 `models/gru.pt` 를 덮어쓴다:

```bash
python scripts/train_stage2.py --kps runs/kps/kps_trim.npz --manifest runs/kps/manifest_full.csv \
    --model gru --feat posvel --out runs/mine --save-model runs/mine/models && cp runs/mine/models/gru.pt models/gru.pt
```

## 6. 새 데이터로 키포인트 추출 (데이터 확장할 때만)

```bash
python scripts/extract_keypoints.py --manifest manifest.csv --root <영상루트> --out runs/kps_new --device mps
python scripts/pack_keypoints.py --kps runs/kps_new --out runs/kps_new.npz
```

manifest 형식은 `runs/kps/manifest_full.csv` 와 같다 (`clip,label,start,end,subject,cam,dataset`).
OmniFall zip 에서 manifest 를 만드는 것은 `scripts/build_manifest.py`, 데이터셋 하나씩 처리하는 것은 `scripts/expand_one.sh`.

## 7. 문제가 생기면

| 증상 | 원인 · 해결 |
|---|---|
| `영상을 찾을 수 없다` | `data/aihub_sample/` 이 없거나 이름이 다르다. `ls data/aihub_sample/01.원천데이터/영상` |
| 창이 안 뜬다 (Linux 서버) | `--no-display --save-video out.mp4` 로 저장해서 본다 |
| `numpy.dtype size changed` | numpy 가 2.x 로 올라갔다. `pip install "numpy==1.26.4"` |
| weasyprint import 오류 | pango 미설치. 위 0절. 또는 `--pdf` 빼고 HTML 만 만들어 브라우저에서 ⌘P → PDF |
| CPU 에서 너무 느리다 | `--resize 640` 으로 입력을 줄인다. 판정 정확도는 거의 같다 |
| MPS 에서 이상한 오류 | `--device cpu` 로 확인해 본다 |

AI Hub 공식 베이스라인(RandomForest) 재현 절차는 [baseline_run.md](baseline_run.md), 옛 상세 가이드는 [archive/how_to_run_aihub_baseline.md](archive/how_to_run_aihub_baseline.md).
