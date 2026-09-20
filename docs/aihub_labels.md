# AI Hub 라벨 실제 스키마 (VL.zip 분석)

_2026-09-20, `falldata/aihub/531138/VL.zip.part000` (16.5MB, 27,264 JSON) 을 직접 열어 확인._

## 결론 먼저

1. **키포인트는 없다.** 소개 페이지의 "영상: 키포인트(JSON)" 는 라벨 zip 에 들어있지 않다.
   공식 베이스라인이 MediaPipe 로 직접 추출한 이유다. **Stage 2(시계열) 학습에도 영상(VS.zip)이
   필요하다** — 라벨만으로 Stage 2 를 돌릴 수 있다는 앞선 판단은 AI Hub 에 대해서는 틀렸다.
2. 낙상 구간은 **프레임 단위**로 찍혀 있다 (`fall_start_frame`, `fall_end_frame`, 60fps · 600프레임).
   시계열 라벨로서는 정확하다.
3. 클립당 10장의 샘플 프레임에 **BBOX** 가 있다 — Stage 1 검출기 학습 데이터로 바로 쓸 수 있다.

## 구성

zip 안 최상위 그룹 세 개, 각각 `Y/{FY,BY,SY}/<clip>/` (낙상) 과 `N/N/<clip>/` (비낙상).
클립 ID 형식: `{장면5자리}_H_A_{클래스}_C{카메라}` (예: `00001_H_A_SY_C1`).
한글 폴더명은 CP949 로 인코딩되어 있어 `unzip` 에서는 `?` 로 보인다 — Python `zipfile` 로 열고
`name.encode("cp437").decode("cp949")` 로 복원한다.

| 그룹 | 클립당 파일 | 내용 |
|---|---|---|
| `영상/` | 1 | 장면 메타데이터 + 낙상 구간 프레임 |
| `센서/` | 1 | `영상/` 과 **동일 내용** (중복). 센서 시계열은 원천데이터 쪽 CSV 로 추정 |
| `이미지/` | 10 (`_I001` ~ `_I010`) | 샘플 프레임의 BBOX |

### Validation 규모

| 항목 | 값 |
|---|---|
| 장면 | 284 |
| 카메라 | C1~C8, 장면마다 8대 전부 |
| 클립 (장면×카메라) | 2,272 |
| 클래스 분포 (클립) | FY 776 · BY 576 · SY 352 · **N 568** |
| 해상도 · 길이 | 3840×2160 · 600 프레임 (60fps, 10초) |

낙상 1,704 vs 비낙상 568 — 비낙상이 25% 다. 같은 장면을 8 카메라로 찍었으므로
**split 은 장면(scene_id 앞 5자리) 단위**로 해야 카메라 간 누수가 없다.

## 스키마

### `영상/<clip>/<clip>.json` (= `센서/...`)

```json
{
 "metadata":   {"scene_id": "00001_H_A_SY_C1", "scene_format": "MP4", "scene_res": "3840 X 2160",
                "creator": "순천향대학교 산학협력단", "distributor": "NIA", "date": "2023-09-05"},
 "scene_info": {"scene_loc": "병원", "scene_pos": "병실", "scene_method": "none",
                "scene_IsFall": "낙상", "scene_cat_name": "측면낙상", "fall_type": "중심을 잃고 넘어짐",
                "scene_length": 600, "cam_num": 1},
 "actor_info": {"actor_id": "F", "actor_age": "adult1(청소년청년)", "actor_sex": "m"},
 "sensordata": {"fall_start_frame": 352, "fall_end_frame": 412},
 "scene_path": {"scene_path": "낙상/Y/SY/00001_H_A_SY_C1"}
}
```

비낙상(N) 은 `scene_IsFall: "비낙상"`, `scene_cat_name: "비낙상"`, `fall_type: "none"`,
`fall_start_frame = fall_end_frame = 0`.

### `이미지/<clip>/<clip>_I###.json`

```json
{
 "metadata": {"scene_id": "00001_H_A_SY_C1", "file_name": "00001_H_A_SY_C1_I001.JPG",
              "img_format": "JPG", "scene_res": "3840 x 2160", "date": "2023-09-05"},
 "bboxdata": {"bbox_location": "977.95, 823.38, 1731.22, 2116.57"},
 "img_path":  {"img_path": "낙상/Y/SY/00001_H_A_SY_C1"}
}
```

`bbox_location` 은 **문자열** `"x1, y1, x2, y2"` (4K 픽셀, 소수). YOLO 포맷으로 바꿀 때는
`(x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H` 로 정규화한다. 어느 프레임 번호의 JPG 인지는
JSON 에 없다 — 원천데이터의 JPG 파일과 `file_name` 으로 짝지어야 한다.

## omnifall_v1 스키마로의 매핑 (제안)

| 구간 (프레임) | omnifall label | 비고 |
|---|---|---|
| `[fall_start, fall_end]` | **1 fall** | 확정 |
| `(fall_end, 600]` | **2 fallen** | 넘어진 뒤 상태로 추정 — 영상으로 검증 필요 (일어나는 경우가 있을 수 있음) |
| `[0, fall_start)` | 미정 | walk/standing 구분 정보 없음. 학습에서 제외하거나 `9 other` |
| N 클립 전체 | 미정 | 비낙상 활동 종류 정보 없음 (`scene_method: none`) → `9 other` 또는 별도 `nonfall` |

`start/end` 는 프레임 ÷ 60 초. `subject` 는 `actor_id`, `cam` 은 `cam_num`, `dataset` 은 `aihub71641`.
낙상 방향(FY/BY/SY)은 omnifall 에 없으므로 보조 컬럼 `fall_dir` 로 보존한다.

## 다음에 필요한 것

- **VS.zip (55GB)** — 영상이 없으면 키포인트를 만들 수 없다. GCP 서울 VM 으로 전송
  ([gcp_seoul_relay.md](gcp_seoul_relay.md)).
- TL.zip 도 같은 스키마인지, Training 규모(장면 수·분포) 확인.
- `이미지/` BBOX 와 원천 JPG 의 프레임 번호 대응 규칙 확인 (VS 해제 후).
