# 추가 데이터셋 후보

_조사: 2026-09-23_

## 왜 더 필요한가

AI Hub 샘플 평가에서 FPR 0.750이 나왔다. 원인은 **`lying`(침대에 누움=정상) 데이터 부족**이다.

```
fallen  179개 (15.0%)   ← 바닥에 넘어짐 = 위험
lying    23개 ( 1.9%)   ← 침대에 누움 = 정상, 부족
```

모델이 "수평 자세 = 위험"으로 학습해, 침대에 엎드린 사람을 낙상으로 오판했다.
OmniFall 논문도 같은 현상을 보고한다 — *"zero-shot 모델은 fallen 과 lying 을 계속 혼동한다"*.

따라서 **정상적인 누운/앉은 자세가 풍부한 데이터**가 우선순위다.

## 즉시 사용 가능 (Drive 확보 완료)

| 데이터셋 | 크기 | 상태 | 비고 |
|---|---|---|---|
| OmniFall staged 4개 | 26GB | **추가 진행 중** | MCFD·UP_Fall·LE2I·OCCU |
| **OF-Synthetic** | 9GB | **추가 진행 중** | 10클래스 × 1,200개 균등. `lying` 1,200 |
| FallVision | 16GB | 미사용 | **침대/의자/서있기 낙상 구분** |
| URFD | 7.4GB | 미사용 | UR Fall Detection |
| EFPDS | 2.4GB | 미사용 | 보행자 검출 (낙상 무관) |

### OF-Synthetic 이 핵심이다

클래스별 정확히 1,200개씩 균등 배치돼 있다.

```
fall 1,200 · fallen 1,200 · lie_down 1,200 · lying 1,200 · other 1,200
sit_down 1,200 · sitting 1,200 · stand_up 1,200 · standing 1,200 · walking 1,200
```

`lying` 23 → 1,223개(53배)로 우리 문제를 직접 해결한다.

**주의: AV1 코덱이다.** OpenCV 가 디코딩하지 못해 13,601개 전부 실패했고 segfault 까지 났다.
ffmpeg(`libdav1d`)로 H.264 변환 후 써야 한다. 전체 변환은 오래 걸리므로
부족 클래스(`lying`·`lie_down`·`fallen`) 3,600개만 우선 변환한다.

### FallVision (다음 우선순위)

**낙상을 침대·의자·서있기 세 종류로 구분**한다. 침대 맥락을 학습시키기에 적합하다.
Drive 에 이미 있으므로 OF-Synthetic 다음으로 추가한다.

## 신규 후보 (미확보)

### ETRI-Activity3D — 한국 노인 일상활동

| 항목 | 값 |
|---|---|
| 규모 | 112,620 샘플 · 100명 · 55개 행동 |
| 대상 | **노인 일상활동** (식사·청소·독서 등) |
| 전체 용량 | **6.47TB** |
| **3D 관절만** | **20.83GB** (25 joints, CSV) |
| 출처 | ETRI (한국전자통신연구원) |
| 신청 | dhkim008@etri.re.kr 문의 |

**전체는 6.47TB 로 불가능하지만 관절 데이터 20.83GB 만 받으면 쓸 수 있다.**
본 파이프라인은 키포인트를 입력으로 쓰므로 영상이 필요 없다.

다만 관절 정의가 다르다 (Kinect 25 joints vs COCO 17 keypoints).
매핑 테이블을 만들어야 하며, 정확도 손실 가능성이 있다.

낙상 클래스가 있는지는 웹페이지에서 확인되지 않았다. 노인 일상활동 중심이므로
**`lying`·`sitting` 보강용**으로는 가치가 크지만 낙상 자체는 없을 수 있다.

### Toyota Smarthome

| 항목 | 값 |
|---|---|
| 규모 | 16,129 클립 · 31개 행동 · **60~80세 18명** |
| 환경 | 스마트홈 아파트, Kinect v1 × 7대 |
| 모달리티 | RGB + Depth + 3D Skeleton |
| 출처 | [INRIA 프로젝트 페이지](https://project.inria.fr/toyotasmarthome/) |

**실제 노인이 실제 집에서 생활하는 영상**이다. 연출이 아닌 자연스러운 일상이라
`lying`·`sitting` 의 실제 분포를 배우기에 좋다.

Untrimmed 버전도 있어 연속 영상 평가에 쓸 수 있다.

### NTU RGB+D

| 항목 | 값 |
|---|---|
| 규모 | 114,480 샘플 · 120 클래스 · 40명 |
| 특징 | **건강 관련 9개 클래스 포함** (낙상 포함 추정) |
| 모달리티 | RGB + IR + Depth + 3D Skeleton |

가장 크고 표준적인 행동인식 벤치마크다. 다만 피험자 연령이 10~35세라
노인 대상 연구에는 체형·움직임이 다를 수 있다.

### Fall Detection for Eldercare Robot (Zenodo, 2026-06)

최신 데이터셋이다. [zenodo.org/records/20966276](https://zenodo.org/records/20966276)
규모·구성 미확인. 파일 접근에 로그인이 필요할 수 있다.

## 권장 순서

```
1. OF-Synthetic 부족 클래스     ← 진행 중. 가장 직접적인 해결
2. FallVision (Drive 확보)      ← 침대 낙상 구분
3. Toyota Smarthome             ← 실제 노인 일상, lying 실제 분포
4. ETRI-Activity3D 관절만       ← 한국 노인, 20.83GB. 관절 매핑 필요
5. NTU RGB+D                    ← 규모는 크나 연령대가 안 맞음
```

1·2 는 이미 Drive 에 있어 즉시 가능하다. 3·4 는 신청 절차가 필요하다.

## 주의할 점

**데이터를 늘린다고 FPR 이 자동으로 개선되지는 않는다.** `lying` 이 늘어도
모델이 침대라는 맥락을 배우지 못하면 같은 오류가 반복된다.

데이터 보강과 함께 검토할 것:

- **위치 맥락** — 침대/소파 ROI 에서의 수평 자세는 감점
- **지속 시간** — `fallen` 이 일정 시간 지속될 때만 경보
- **자세 전이** — 서 있다가 갑자기 누운 것인지, 처음부터 누워 있었는지

---

## 2026-09-24 추가 조사 — 재학습용 후보 (파이프라인 변경 없이)

기준: **RGB 영상**이면 YOLO → 정규화 → GRU 를 그대로 태울 수 있다. 관절만 배포되는 데이터셋은
Kinect 25/20 관절 → COCO 17 매핑이 필요해 `extract_keypoints.py` 를 건너뛰는 별도 로더가 든다.

### 1순위 — CMDFALL (가장 크고, OmniFall 라벨이 이미 있다)

| 항목 | 값 |
|---|---|
| 규모 | **7시간 7분** · 50명(21~40세) · 7 Kinect 뷰 · 20동작 |
| 동작 | 낙상 8종 (서서·앉아서·**누워서** 넘어짐, 4방향) · 일상 12종 — **누움 4종(침대) · 앉음 4종(의자·침대)** |
| 형식 | RGB AVI 640×480 20Hz + 깊이 BIN + 가속도 |
| 용량 | 피험자당 ~7GB (RGB+깊이). RGB 만 요청하면 훨씬 작다 |
| 라벨 | **OmniFall `labels/cmdfall.csv` 로 16클래스 재라벨링 완료** → `build_manifest.py` 에 바로 연결 |
| 접근 | 연구 목적 무료. thanh-hai.tran@mica.edu.vn 메일 요청 |
| 출처 | [MICA 프로젝트 페이지](https://www.mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html) |

OmniFall 논문에서 staged 학습 데이터의 **절반 이상**을 차지하는 셋이다 (7h 7m / 전체 staged 약 12h).
현재 우리 학습 데이터(50분)의 **8.5배**. 침대 누움·침대에서 낙상이 모두 있어 `lying` vs `fallen` 문제를 직접 겨냥한다.
Zenodo 배포판에 빠져 있어 원저자에게 받아야 한다 — 메일 한 통이면 된다.

### 2순위 — Toyota Smarthome (실제 노인, 긴 정상 영상)

| 항목 | 값 |
|---|---|
| 규모 | trimmed 16,129클립 · **untrimmed 536영상 × 평균 21분** · 60~80세 18명 |
| 형식 | RGB 640×480 + 깊이 + 3D 골격 |
| 접근 | [프로젝트 페이지](https://project.inria.fr/toyotasmarthome/) 신청 폼 · toyotasmarthome@inria.fr |

낙상은 없다. 대신 **[ISSUES.md](ISSUES.md) 8번 "시간당 오경보" 측정에 필요한 긴 정상 영상**이 바로 이것이다.
untrimmed 536편을 파이프라인에 흘려 하루 오경보 횟수를 실측할 수 있다. 학습에는 `lying`·`sitting` 실제 분포 보강.

### 3순위 — NTU RGB+D 120 (규모, 낙상 클래스 A43)

| 항목 | 값 |
|---|---|
| 규모 | 114,480 샘플 · 120클래스 · 106명(10~57세) · 3 뷰 · 155 시점 |
| 관련 클래스 | A43 **falling down** · A8 sit down · A9 stand up · 그 외 일상 |
| 형식 | RGB 1920×1080 + 깊이 + IR + 25관절 골격 |
| 접근 | [ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition/) 계정 등록 → 릴리스 동의 → 승인 |

RGB 전체는 수백 GB 라 **A43 + 일상 몇 클래스만** 받는다. `lying` 이 명시 클래스로 없다는 게 약점.

### 관절만 배포 — 매핑 로더 필요 (파이프라인 변경 있음)

| 데이터셋 | 내용 | 왜 관심 | 접근 |
|---|---|---|---|
| **ETRI-Activity3D** | 한국 노인 50명(64~88세)+청년 50명 · 55동작 · 25관절 20.8GB | 유일한 **한국 노인** 실데이터. 낙상 클래스 유무 미확인 | ETRI 문의 |
| FUKinect-Fall | 21명(**19~72세**) · 걷기/굽히기/앉기/쪼그리기/**눕기**/낙상 · 1,008 깊이영상 + 20관절 7.5GB | 노인 포함, lying 있음. RGB 없음 | [GitHub](https://github.com/MuzafferAslan23/Fall-Detection-Dataset) SharePoint 링크 |

### 제외

| 데이터셋 | 이유 |
|---|---|
| TST v2 (9.3GB, 깊이+골격) | IEEE DataPort 유료 구독 필요 |
| falldataset.com (22,636장) | 정지 이미지 — 시계열 모델에 못 쓴다 |
| Zenodo 20966276 "Eldercare Robot" | 파일 비공개, 규모 미상 |
| Unidata 10,000영상 | 상업 판매 |
| Kaggle "Multiple Cameras Fall" | = MCFD, Drive 에 이미 있음 |

### 재학습 우선순위 (Drive 확보분 포함)

```
A. Drive 에 이미 있음 — 키포인트 추출만 하면 된다
   1. OmniFall staged 4종 (LE2I·MCFD·UP_Fall·OCCU)  26GB   → 환경 다양성 (ISSUES 6번)
   2. OF-Synthetic 부족 클래스 (lying·lie_down·fallen 3,600)  → lying 53배 (ISSUES 4번)  ※ AV1 변환
   3. FallVision 16GB                                  → 침대/의자 낙상 구분

B. 신청 필요 — 지금 메일 보내면 A 하는 동안 온다
   4. CMDFALL      메일 1통 · 7시간 · OmniFall 라벨 있음   → 학습량 8.5배, 침대 누움/낙상
   5. Toyota Smarthome untrimmed  신청 폼                → 시간당 오경보 측정 (학습 X, 평가용)
   6. NTU RGB+D A43 + 일상 일부  계정 승인
   7. ETRI-Activity3D 관절        ETRI 문의               → 한국 노인 (매핑 로더 필요)
```

A 를 끝내면 학습 데이터가 50분 → 약 3.5시간(+합성), CMDFALL 까지 오면 10시간을 넘는다.
**어디서 추출할지**가 병목이다 — Colab 은 세션 회수 8회로 실패했다. 로컬 M3 로 밤새 돌리는 것이
가장 확실하다 (YOLOv11n-pose 30 FPS 기준 10시간 영상 ≈ 10~12시간).
