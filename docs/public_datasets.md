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
