# 공개 낙상 데이터셋 조사 결과

_조사·검증: 2026-09-20. 모든 URL·MD5·크기는 실제 API 응답에서 확인했다._

## 결론

**OmniFall 통합 배포판을 주력 공개 데이터로 채택한다.** 8개 데이터셋(30.1GB)을
하나의 16클래스 시간 구간 라벨로 다시 붙인 것이라, 개별 데이터셋을 하나씩 받아
포맷을 맞추는 작업이 이미 끝나 있다. **라벨 스키마도 그대로 우리 프로젝트의 통합
manifest 형식으로 쓴다.** 나중에 AI Hub 데이터를 같은 컬럼으로 매핑해 넣으면
모든 모델이 하나의 로더로 돈다.

레지스트리는 [configs/datasets.yaml](../configs/datasets.yaml).

## 왜 공개 데이터셋이 지금 가능한가

AI Hub 는 해외 IP 다운로드를 차단하지만, Zenodo·Hugging Face·GitHub 는 그렇지
않다. **Colab 에서 직접 받아 Drive 로 옮길 수 있다.** 맥북 SSD 18GB 제약이
공개 데이터에 대해서는 사라진다.

```
Zenodo ──직접──> Colab /content (94GB 여유) ──> Google Drive (400GB)
                       ↓ zip 단위로: 다운로드 → MD5 → 복사 → 삭제
```

## OmniFall

- 논문: [arXiv 2505.19889](https://arxiv.org/abs/2505.19889) — "From Staged Through Synthetic to Wild"
- 어노테이션·split: [HF simplexsigil2/omnifall](https://huggingface.co/datasets/simplexsigil2/omnifall) (gated 아님, CC-BY-NC-4.0)
- 영상 클립 배포: [Zenodo 17170592](https://zenodo.org/records/17170592) (2025-09-21, 배포판 CC-BY-4.0)

### 라벨 체계 (16클래스)

| id | 라벨 | 의미 | 본 연구에서 |
|---|---|---|---|
| 0 | walk | 이동 | |
| **1** | **fall** | 넘어지는 동작. 관성 운동이 멎을 때 끝 | **낙상 이벤트** |
| **2** | **fallen** | 넘어진 뒤 바닥·매트리스에 있는 상태 | **미동 판정 대상 상태** |
| 3 | sit_down | 앉는 동작 | |
| 4 | sitting | 앉은 상태 | 정상 정지 |
| 5 | lie_down | 의도적으로 눕는 동작 | |
| **6** | **lying** | 의도적으로 누운 상태 | **fallen 과 반드시 구분** |
| 7 | stand_up | 일어남 (누움→앉음 포함) | 회복 신호 |
| 8 | standing | 서 있음 | |
| 9 | other | 기타 | |
| 10–15 | kneel/squat/crawl/jump | OOPS·synthetic 에만 등장 | 9 로 접어도 됨 |

**`fallen` 과 `lying` 의 구분이 이 데이터셋을 채택하는 가장 큰 이유다.** "미동 없음"
클래스는 이 위에 서야 한다 — 바닥에 넘어진 채 움직이지 않는 것(위험)과 침대에
누워 움직이지 않는 것(정상)은 다르게 판정되어야 한다. 공개 데이터셋 중 이 둘을
분리해 라벨링한 것은 거의 없다.

동적 라벨(fall, stand_up)은 동작이 시작되는 첫 프레임부터, 정적 라벨(fallen,
sitting)은 정지 상태에 도달한 첫 프레임부터 시작한다 (LABELS.md).

### 구성 요소

| dataset | 크기 | 피험자 | 시간 | 특징 | 원본 라이선스 |
|---|---|---|---|---|---|
| caucafall | 0.17GB | 10 | 16m | 낙상 5종 + ADL 5종 | 확인 필요 |
| edf | 0.99GB | 5 | 13m | 다중 카메라 | 확인 필요 |
| **OOPS** | 1.03GB | — | — | **실제 사고 영상** (in-the-wild) | 확인 필요 |
| **GMDCSA24** | 1.67GB | 4 | 21m | 81 낙상 + 79 ADL | **MIT** |
| mcfd | 2.80GB | 1 | 12m | 8 카메라 | 확인 필요 |
| up_fall | 3.18GB | 17 | 4h35m | staged 중 최대 | 확인 필요 |
| le2i | 9.07GB | 9 | 47m | 집·사무실·강의실 등 6환경 | CC-BY-NC-SA-3.0 |
| occu | 11.15GB | 5 | 14m | 가림(occlusion) 상황 | 확인 필요 |

Zenodo 배포판에 **없는** 것: cmdfall (384 연속 영상, 7 Kinect 뷰) 과 of-syn (합성
12,000편). cmdfall 은 원본을 별도 확보해야 하고, of-syn 은 HF 에서 직접 받을 수 있다.

### Zenodo 배포판의 레이아웃 — HF 공식과 다르다

HF 문서(STRUCTURE.md)는 연속 영상을 전제한다:

```
{root}/{dataset}/video/{path}.mp4        ← 라벨 CSV 의 (dataset, path) 로 해석
```

그러나 Zenodo zip 은 **라벨 구간대로 미리 잘라낸 클립**이다 (Cauca_fall.zip 을
받아 확인):

```
Cauca_fall/
├── caucafall.csv                      # omnifall_v1 스키마, 258 구간
├── mmaction2_annotations.txt          # "Subject.1\HopS1_9_0.000_9.100_1_1_caucafall.avi 9"
└── clips/Subject.{1..10}/
    ├── {Action}S{subj}_{label}_{start}_{end}_{subj}_{cam}_caucafall.avi              # 258개 (유지)
    └── Subject.{subj}_{action}_01_{start}_{end}_{label}_{cam}_{subj}_caucafall.avi     # 258개 (동일 내용 중복, 삭제)
```

파일명에 라벨·시작·끝·피험자·카메라가 전부 인코딩돼 있다.

| | 미리 잘린 클립 (Zenodo) | 연속 영상 (HF 공식) |
|---|---|---|
| Stage 2 시계열 분류 학습 | **바로 사용 가능** | 구간 잘라내기 필요 |
| 구간 경계 탐지 / 스트리밍 평가 | 불가 — 경계가 이미 주어짐 | 가능 |
| 미동 판정 (긴 정지 구간) | 클립 길이에 갇힘 | 앞뒤 맥락 활용 가능 |

Stage 2 학습에는 Zenodo 가 충분하다. 스트리밍 평가가 필요해지면 그때 원본 연속
영상을 확보한다.

**516 = 258 × 2 의 정체 (해결)**: 같은 클립이 두 가지 파일명 규칙으로 두 번 들어있다.
zip 의 CRC 로 대조한 결과 258쌍 전부 바이트 단위로 동일한 순수 중복이다.

```
FallBackwardsS1_1_1.930_3.770_1_1_caucafall.avi             # 짧은 형식: {Action}S{subj}_{label}_{start}_{end}_{subj}_{cam}
Subject.1_fall_backwards_01_1.930_3.770_1_1_1_caucafall.avi  # 긴 형식:  Subject.{subj}_{action}_01_{start}_{end}_{label}_{cam}_{subj}
```

`mmaction2_annotations.txt` 는 짧은 형식만 가리키므로 **짧은 형식을 남기고 긴 형식을
지운다.** 해제 직후 삭제하면 디스크가 50% 절감된다. 8개 zip 모두 같은 구조라면 30.1GB 의
실제 고유 콘텐츠는 약 15GB 다 — zip 마다 해제 시 검증한다.

### 알려진 함정 (KNOWN_PITFALLS.md)

- **mcfd / edf / cmdfall 에서 일부 구간의 `end` 가 영상 길이를 넘는다.** 다중 카메라
  동기화 오프셋을 적용한 결과다. 최대 초과: mcfd 1.19s, edf 7.36s, cmdfall 3.80s.
  로더에서 `end = min(end, duration)` 처리.
- config 이름 별칭: 맨 이름 `le2i` 는 `le2i-cs`(cross-subject) 의 별칭이다. 전체가
  아니다. cross-view 는 `le2i-cv` 를 명시.
- `path` 는 dataset 안에서만 유일하다. 캐시·조회 키는 `(dataset, path)` 쌍.

### split 구성

`of-sta-cs`(cross-subject) 가 기본이고, **`of-sta-cv`(cross-view) 가 본 연구의
핵심 주장인 카메라 위치 독립성을 검증하는 split 이다.** AI Hub 베이스라인이
카메라별 모델 8개인 것과 직접 대비된다. `of-sta-to-all-cs` 는 연출 데이터로 학습해
실제 사고(OOPS)까지 평가하는 일반화 지표.

## 2차 수집 — 현재 확보분의 빈 곳을 채우는 후보 (2026-09-20 조사)

OmniFall staged 8종의 약점 세 가지를 겨냥한다: 배우가 대부분 젊음, Stage 1 BBOX 없음,
미동 판정용 긴 정상 정지 구간 없음.

| 후보 | 채우는 빈 곳 | 크기 | 입수 경로 | 라이선스 |
|---|---|---|---|---|
| **OF-Syn** (OmniFall 합성 12,000편) | 노년(65+) 그룹, **카메라 고도·방위 라벨** | 9.72GB (AV1 tar) | **Drive 확보 완료** (`falldata/omnifall_syn/`, sha256 검증, 137MB/s) | CC-BY-NC-4.0 |
| **E-FPDS** | Stage 1 **쓰러진 사람 BBOX** 6,982장 (노년 413장) | ~2.3GB (train 재다운로드 중) | 브라우저 → rclone → `falldata/efpds/` (valid·test·FPDS_info 업로드 중) | 인용 필수 |
| **URFD** | 연속 시퀀스, RGB+depth+가속도 | 7.98GB (zip 170 + csv 140) | **Drive 확보 완료** (`falldata/urfd/` + manifest.json, 크기 검증) | CC-BY-NC-SA-4.0 |
| CMDFall | **연속 다중뷰 384편** — OmniFall 라벨은 있고 영상만 없음 | 대용량 | 저자 이메일 요청 | 연구용 |
| Toyota Smarthome | **60–80세 노인 ADL**, 긴 정지 구간 (미동 음성 샘플) | 대용량 | 라이선스 폼 | 연구용 |

OF-Syn 은 합성이라 **실측 성능 주장의 근거로는 쓰지 않는다.** 대신 카메라 위치를 통제한
ablation(같은 낙상을 eye/low/high/top × front/rear/left/right 로 본 결과)에 쓴다 — AI Hub
베이스라인의 카메라 의존성과 대비되는 본 연구의 핵심 주장을 검증하는 데 맞는 도구다.
AV1 코덱이라 OpenCV 가 못 읽을 수 있어 ffmpeg 로 h264 트랜스코딩이 필요하다.

### E-FPDS 라벨 형식 — "YOLO" 가 아니다

페이지는 YOLO 호환이라 하지만 실제 txt 는 **`class x_min x_max y_min y_max`** 픽셀 절대좌표다
(darknet BBox-Label-Tool 형식). 640×480 이미지의 `1 173 562 367 472` 를 `x1 y1 x2 y2` 로 읽으면
y=562 가 높이를 넘어 모순이고, `x_min x_max y_min y_max` 로 읽어야 x 173~562, y 367~472 의
넓고 낮은 박스(누운 사람)가 된다. 학습 전에 YOLO 정규화(`xc yc w h`, 0~1)로 변환한다.
이미지 해상도가 섞여 있으므로(640×480, 1920×1080 급) 변환 시 각 이미지 크기를 읽어야 한다.

split 구성(`FPDS_info/Info_splits.txt`): train 4,808장(split01·02·03·10·11) · valid 1,201장(split12·13) ·
test 973장(split04~08). split11 은 2,214장 전부 fallen, test 의 split05 는 non-fallen 이 다수 —
분포가 split 마다 크게 달라 **split 단위 평가에서 편향**이 생길 수 있다.

## 3차 조사 — 야간·노년·골격·정상 정지 축 (2026-09-20)

| 후보 | 채우는 빈 곳 | 규모 | 입수 | 라이선스 |
|---|---|---|---|---|
| **FallVision** | 침대·의자·서서 낙상(병실 시나리오), 58명 | raw 16.8GB + 키포인트 CSV 0.4GB (mask 32GB 제외) | **Drive 확보 완료** (`falldata/fallvision/`, 40/40 MD5, 79~91MB/s) | **CC0** — 서비스 학습 가능 |
| **TF-66** (Thermal Fall 66) | **야간·열화상**, 66명, 노인·병원 서브셋 | 562 낙상 + 250 비낙상, 140×60 @4fps | 저자 이메일 문의 (프리프린트에서 링크 삭제) | 비상업 |
| NTU RGB+D 120 | 3D 골격 대규모, `A43 falling` | 114,480 샘플 (골격 zip ~10GB) | ROSE Lab 등록 | 학술 |
| **ETRI-Activity3D** | **한국 노인 50명(64~88세)**, 아파트, Kinect 8대 | 112,620 샘플, RGB+D+골격 | ETRI 나눔 신청 (접속 거부로 미확인) | 학술 (낙상 클래스 포함 여부 확인 필요) |
| Toyota Smarthome **Untrimmed** | 노인 **평균 21분 연속 영상 536편** — 미동 음성 샘플 | 51 활동, RGB+D+골격 | 라이선스 폼 | 학술 |
| HOMAGE | 가정 다중 시점 27명 | 대용량 | 저장소 안내 | 학술 (우선순위 낮음) |

제외: IR-Fall 2024(비공개), figshare "SDU dataset"(SDUFall 아님), TST Fall v2(IEEE DataPort 구독),
FUKinect-Fall(depth 만·라이선스 미표기), MSR DailyActivity3D(소규모, 필요 시).

**FallVision 이 이번 조사의 최대 수확이다.** CC0 라 라이선스 제약이 전혀 없고, 침대·의자에서의
낙상이 있으며(치매환자 병실에서 가장 흔한 시나리오), Dataverse API 로 Colab 에서 바로 받는다.
TF-66 은 해상도 140×60 이라 YOLO-pose 를 못 쓰고 별도 열화상 분기 모델이 필요하다 — 야간
모니터링을 논문 범위에 넣을지에 따라 결정한다.

## 그 외

### URFD

[공식 페이지](https://fenix.ur.edu.pl/~mkepski/ds/uf.html). 낙상 30 (cam0+cam1) +
ADL 40 (cam0). zip 안이 **PNG 프레임 시퀀스**라 영상 파일이 아니다. RGB 외에
depth(PNG16)·가속도 CSV 가 있어 멀티모달 확장 여지. CC-BY-NC-SA-4.0, 인용 필수
(Kwolek & Kepski 2014).

### Roboflow (Stage 1 bbox)

사람 검출용 YOLO 포맷 bbox 데이터셋이 여럿 있다 (4천 장 규모). 프로젝트별 라이선스가
다르고 Roboflow API 키가 필요하다. 후보는 레지스트리에 기록.

## 라이선스 — 논문과 서비스를 갈라야 한다

| 용도 | 쓸 수 있는 데이터 |
|---|---|
| 논문 실험·비교·교차 평가 | 위 전부 |
| **서비스에 올리는 최종 모델** | **AI Hub 71641** (국내 활용 조건 약관 확인) + GMDCSA24 (MIT) |
| 서비스 학습에 **쓰면 안 되는 것** | le2i, urfd (CC-BY-NC) |

나머지(caucafall, edf, OOPS, mcfd, up_fall, occu)는 원본 라이선스를 각각 확인해야
한다. 확인 전까지는 논문 전용으로 취급한다.

## "미동 없음" 데이터의 공백

몇 분간의 정지를 라벨링한 공개 데이터셋은 없다. 전부 몇 초짜리 구간이다.

- **자세 판정은 학습** — `fallen` vs `lying` vs `sitting` 을 OmniFall 로 학습
- **미동 자체는 규칙** — 키포인트 변위 분산 + 지속 시간 (학습 데이터 불필요)
- **검증용 클립은 직접 촬영** — 소파에 누워 5분, 바닥에 엎드려 5분 등 수십 개.
  공개 데이터로 대체할 수 없는 부분이다.

## 하지 않을 것

YouTube 등에서 사람이 넘어지는 영상을 직접 수집하지 않는다. 저작권·초상권 문제가
있고, 실제 사고 영상은 OOPS-Fall 이 이미 연구용으로 큐레이션되어 있다.
