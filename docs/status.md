# 현황 및 진행 방안

_최종 갱신: 2026-09-20_

## 한 줄 요약

- **OmniFall 30.06GB 는 Drive 에 확보 완료** (8/8 MD5 검증, 2026-09-20).
- AI Hub 는 **한국 IP에서만** 받을 수 있고 Colab(미국)은 막힌다. 맥북이 디스크 없이 스트리밍 중계한다.
- 이 회선은 총 3MB/s 라 VS 55GB 는 밤새 가능하지만, TS 436GB 는 **GCP 서울 VM** 에서 병렬로 돌려야 한다.

## 확인된 사실

### 데이터셋 구성 (dataSetSn=71641)

`aihubshell -mode l` 로 직접 조회한 실제 구조:

```
041.낙상사고 위험동작 영상-센서 쌍 데이터
└─ 3.개방데이터 / 1.데이터
   ├─ Training
   │  ├─ 01.원천데이터
   │  │  ├─ TS.z01 | 100 GB | filekey 531131  ┐
   │  │  ├─ TS.z02 | 100 GB | filekey 531132  │ 분할 zip 한 덩어리
   │  │  ├─ TS.z03 | 100 GB | filekey 531133  │ 전부 받아야 해제 가능
   │  │  ├─ TS.z04 | 100 GB | filekey 531134  │ = 436 GB
   │  │  └─ TS.zip |  36 GB | filekey 531135  ┘
   │  └─ 02.라벨링데이터
   │     └─ TL.zip | 126 MB | filekey 531136
   └─ Validation
      ├─ 01.원천데이터
      │  └─ VS.zip |  55 GB | filekey 531137   ← 단일 파일
      └─ 02.라벨링데이터
         └─ VL.zip |  16 MB | filekey 531138
```

**총 491GB.** 당초 300GB로 알았던 것보다 크다.

핵심 제약 두 가지:

1. **Training 원천데이터는 분할 다운로드가 불가능하다.** `TS.z01~z04 + TS.zip` 은
   하나의 분할 zip 아카이브라 5개를 모두 받아 합쳐야 풀린다. 파트별로 나눠
   처리하려던 계획은 성립하지 않는다.
2. **Validation 원천데이터(VS.zip, 55GB)는 단일 파일**이라 상대적으로 다루기 쉽다.

### 지역 차단 (핵심 장애물)

Colab VM(us-east4)에서 다운로드 시도 시:

```
Download failed with HTTP status 502.
AI 허브는 해외에서의 데이터 다운로드를 제한하고 있습니다.
```

- 파일 목록 조회(`-mode l`)는 해외에서도 된다 → API Key는 유효
- 다운로드(`-mode d`)만 한국 IP를 요구한다
- 승인 자체는 정상 반영됨 (에러 메시지가 "승인 필요" → "해외 제한" 으로 바뀐 것이 근거)
- **무료 Colab은 리전 선택이 불가능하다** ([colabtools#2875](https://github.com/googlecolab/colabtools/issues/2875), 미해결)

### 가용 자원

| 자원 | 상태 |
|---|---|
| 맥북 M3 16GB | SSD 여유 **3.5GB** (다른 세션 산출물·샘플·도커 이미지로 감소) — 스트리밍 외 불가 |
| Colab CLI | 인증 완료. CPU VM 디스크 94GB, RAM 14GB, 2코어 |
| Google Drive | **5TB** (`nnn290199@gmail.com`) — Colab CLI 주 계정으로 전환 완료. OmniFall 30.06GB 보관 중 |
| AI Hub API Key | 발급 완료, `~/.aihub_key` 에 저장 |

## AI Hub → Drive 스트리밍 (2026-09-20 실측으로 확정된 경로)

맥북(한국 IP)이 **디스크에 쓰지 않고** 중계한다: `scripts/aihub_stream.py`.

```
curl (AI Hub tar 스트림) → tarfile 스트림 파서 → 멤버(.partN)마다 rclone rcat → Drive
```

실측으로 확인된 제약:

| 항목 | 결과 | 의미 |
|---|---|---|
| 다운로드 응답 | `302 → stream.aihub.or.kr/shellStream.do?uuid=…`, tar 즉석 생성, `Content-Length`·`Accept-Ranges` 없음 | **Range·이어받기·병렬 분할 불가.** 한 filekey 는 한 스트림으로 처음부터 끝까지 |
| 연결당 속도 | 2.2MB/s (126MB 60초 평균) | VS.zip 55GB ≈ **7시간**, 중단 시 재시작 |
| 동시 2스트림 | 각 2.3 / 1.7MB/s | 제한은 연결당 — filekey 가 여럿(TS 5개)이면 병렬 유효 |
| 이 회선 원시 대역폭 | 국내 미러 단일 3.3MB/s, **6연결 합산 3.0MB/s** | **회선 총량이 ~3MB/s.** 병렬을 걸어도 총량은 같다 — 병렬은 빠른 회선(GCP 서울 VM·학교)에서만 유효 |
| 맥북 여유 디스크 | 3.5GB | 스트리밍이 필수 |

조각이 완료될 때마다 Drive 에 남으므로 도중에 끊겨도 업로드는 재사용된다 (다운로드는 다시).
Drive 는 5TB 라 TS 436GB 도 들어간다. 문제는 용량이 아니라 **이 회선(3MB/s)** 이다:

| 작업 | 이 회선 | GCP 서울 VM / 학교 회선 (5 filekey 병렬) |
|---|---|---|
| VS 55GB | 5~7시간 — 가능 (밤새) | 수십 분~1시간 |
| TS 436GB | 약 40시간 — 비현실적 | AI Hub 연결당 2.2MB/s 가정 시 약 11시간, 서버 상한이 더 높으면 단축 |

**TS 는 GCP 서울 VM(e2-small, 하루 ≈ $0.5, 크레딧 $300 내)에서 같은 스크립트를 돌린다.**
맥북이 잠들어도 상관없고, 한국 IP 이며, 회선이 기가비트급이다. GCP 계정·결제 설정이 선행 조건.

## 왜 Drive 400GB 만으로는 부족한가

Drive 에는 **다운로드 기능이 없다.** 누군가가 AI Hub 에서 받아서 올려야 하는데,
그 주체가 한국 IP 를 가져야 한다.

- Colab 에서 받기 → 미국 IP → 차단
- 맥북에서 받기 → 한국 IP 통과 → 그러나 SSD 18GB 로 436GB 를 거칠 수 없음

Drive 는 **최종 보관소**로서는 필요하지만, 다운로드 경로가 될 수는 없다.
전처리된 데이터·체크포인트·실험 결과를 넣을 곳으로 확보해 두는 것은 여전히 유효하다.

## 필요한 구성

```
[한국 IP + 대용량 디스크]   ← 이것이 없으면 이미지 작업 시작 불가
         ↓ 전처리로 축소 (프레임 샘플링 + 640px + JPEG q85)
    [Drive 400GB]          ← 학생 계정으로 확보
         ↓
   [Colab GPU 학습]
```

## 선택지

### A. GCP 서울 VM (`asia-northeast3`)

한국 IP 와 대용량 디스크를 **동시에** 해결한다.

- VM `e2-standard-4` (4코어/16GB): 시간당 약 $0.19
- 영구 디스크 500GB: 월 약 $20
- 전처리 20시간 기준 **약 $5**, 작업 후 VM 삭제하면 과금 중단
- 신규 가입 **$300 크레딧** 또는 [GCP for Education](https://cloud.google.com/edu) 학생 크레딧이면 사실상 무료
- Colab 에서 "Connect to a Custom GCE VM" 으로 이 VM 에 직접 붙을 수 있다

### B. 교내 GPU 서버 (창원대)

한국 IP + 저장공간 + GPU 를 한 번에 해결. **무료라면 최선.**
전산원 또는 지도교수 연구실에 문의. 확인에 시간이 걸리는 것이 단점.

### C. 외장 SSD

1TB 10만원대. 맥북이 한국 IP 이므로 이것만 있으면 다운로드가 된다.
가장 단순하지만 전처리를 맥북 CPU 로 해야 해서 느리다.

### D. AI Hub 안전학습장

AI Hub 자체 GPU 서버. 데이터 다운로드가 불필요하고 무료.
**단, 학습된 모델의 외부 반출 가능 여부를 반드시 확인해야 한다** —
서비스 탑재가 목표이므로 반출이 막히면 반쪽짜리가 된다.

## 지금 바로 할 수 있는 것

### 1. 라벨 데이터 확보 (142MB, 맥북에서 가능)

```bash
curl -sS -o ~/bin/aihubshell https://api.aihub.or.kr/api/aihubshell.do
chmod +x ~/bin/aihubshell
cd ~/Documents/fall-detection && mkdir -p data/labels && cd data/labels
~/bin/aihubshell -mode d -datasetkey 71641 -filekey 531136,531138 \
  -aihubapikey "$(cat ~/.aihub_key)"
```

이것으로 가능한 작업:

- **실제 JSON 스키마 확인** — 추측이 아닌 실제 구조에 맞춘 전처리 코드 작성
- 클래스 분포, 클립 구조, 키포인트 형식 파악
- **Stage 2 (시계열 낙상 분류) 전체** — LSTM/GRU/1D-CNN/ST-GCN 비교 실험.
  입력이 키포인트 좌표뿐이라 이미지가 필요 없고, 모델이 작아 맥북에서 학습된다
- 규칙 기반 베이스라인 (종횡비 + 몸통각도 + 하강속도) 구현·평가

### 2. 공개 데이터셋으로 Stage 1 프로토타입

Le2i / URFD 는 가입 없이 받을 수 있고 용량이 작다.
YOLOv11-pose + ByteTrack 파이프라인을 미리 완성해 두면,
AI Hub 이미지가 확보되는 즉시 학습만 돌리면 된다.

교차 데이터셋 평가(AI Hub 학습 → 공개셋 테스트)는 논문에서
일반화 성능 근거로 쓸 수 있다.

### 3. Drive 400GB 학생 계정 신청

전처리 결과물과 체크포인트 보관용. 어차피 필요하다.

## 권장 순서

1. **라벨 142MB 다운로드** → 스키마 확인 → 전처리 코드 작성
2. 병행: Drive 400GB 신청, 교내 GPU 서버 문의
3. Stage 2 비교 실험 (맥북에서 완결 가능)
4. 이미지 인프라 확보 후 Stage 1 학습
5. 두 단계 통합 → 미동 판정 로직 → 교차 평가

이미지 인프라를 기다리는 동안 1·3 이 진행되므로 일정 손실이 없다.

## 미해결 사항

- AI Hub 안전학습장의 모델 반출 정책 (선택지 D 채택 시 확인 필요)
- Training 436GB 를 실제로 받을지, Validation 55GB 만으로 Stage 1 을 학습할지
  — 후자는 데이터가 적지만 인프라 부담이 훨씬 작다. 라벨 분석 후 판단
