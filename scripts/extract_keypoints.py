#!/usr/bin/env python3
"""OmniFall 클립 → YOLOv11-pose 키포인트 시퀀스 (.npy)

Stage 2 시계열 분류기의 입력을 만든다. 영상마다 사람을 검출하고 17개 COCO 키포인트를
프레임 단위로 뽑아 (T, 17, 3) 배열로 저장한다. 마지막 축은 (x, y, conf).

AI Hub 베이스라인과의 차이 — 이것이 본 연구의 핵심 주장이다:

    AI Hub : MediaPipe Holistic 1,662차원 (얼굴 468점이 84%) × 600프레임
             = 997,200차원을 flatten → RandomForest. 시간 순서가 소실된다.
    본 연구: YOLOv11-pose 17키포인트 × 2 = 34차원/프레임
             → 시퀀스 모델이 순서를 학습. 약 49배 작다.

좌표는 **프레임 크기로 정규화**하고 **엉덩이 중점 기준으로 평행이동**한다.
카메라 위치·거리·해상도가 달라도 같은 표현이 되게 하려는 것이다 —
AI Hub 베이스라인이 카메라별로 모델을 8개 둔 문제를 구조적으로 피한다.

사용법:
    python extract_keypoints.py --manifest /content/omnifall/manifest.csv \\
        --root /content/omnifall --out /content/keypoints --model yolo11n-pose.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# COCO 17 키포인트 인덱스
NOSE = 0
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12
L_ANK, R_ANK = 15, 16


def normalize(kps: np.ndarray, w: int, h: int) -> np.ndarray:
    """(T, 17, 3) 원본 픽셀 좌표 → 카메라 불변 표현.

    1. 프레임 크기로 나눠 해상도 의존 제거
    2. 엉덩이 중점을 원점으로 (카메라 위치·피사체 위치 무관)
    3. 어깨-엉덩이 거리로 스케일 정규화 (카메라 거리 무관)

    스케일 기준을 몸통 길이로 잡는 이유: 사람이 서 있든 누워 있든 몸통 길이는
    거의 일정하다. 바운딩 박스 높이로 나누면 누웠을 때 값이 폭증한다.
    """
    out = kps.copy().astype(np.float32)
    out[..., 0] /= max(w, 1)
    out[..., 1] /= max(h, 1)

    conf = out[..., 2:3]
    hip = (out[:, L_HIP, :2] + out[:, R_HIP, :2]) / 2      # (T, 2)
    sho = (out[:, L_SH, :2] + out[:, R_SH, :2]) / 2
    torso = np.linalg.norm(sho - hip, axis=-1, keepdims=True)  # (T, 1)
    torso = np.where(torso < 1e-3, 1.0, torso)                 # 검출 실패 프레임 보호

    out[..., :2] = (out[..., :2] - hip[:, None, :]) / torso[:, None, :]
    out[..., 2:3] = conf   # 신뢰도는 정규화하지 않는다
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="클립 경로의 기준 디렉토리")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, help="처리할 클립 수 (빠른 확인용)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--trim-to-dur", action="store_true",
                    help="앞 (end-start)×fps 프레임만 읽는다. OCCU 처럼 절단기가 -ss/-to 를 잘못 써서 "
                         "클립이 [start, start+end] 로 길게 잘린 배포판용 (edf 도 같은 증상)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics 가 필요하다: pip install ultralytics")
    import cv2

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]
    print(f"[*] 클립 {len(rows):,}개")

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    done = skipped = failed = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        clip = args.root / r["clip"]
        uid = f"{r['dataset']}__{Path(r['clip']).stem}"
        dst = args.out / f"{uid}.npy"
        if dst.exists():
            skipped += 1
            continue
        if not clip.exists():
            failed += 1
            continue

        cap = cv2.VideoCapture(str(clip))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        limit = None
        if args.trim_to_dur:
            dur = float(r["end"]) - float(r["start"])
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            limit = max(2, int(round(dur * fps)))
        frames = []
        while limit is None or len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            failed += 1
            continue

        # 배치 추론 — 프레임을 한 번에 넘기면 GPU 활용률이 크게 오른다
        seq = np.zeros((len(frames), 17, 3), dtype=np.float32)
        B = 32
        for s in range(0, len(frames), B):
            res = model(frames[s:s + B], imgsz=args.imgsz, conf=args.conf,
                        device=args.device, verbose=False)
            for j, rr in enumerate(res):
                if rr.keypoints is None or len(rr.keypoints.data) == 0:
                    continue   # 사람 미검출 → 0 으로 남긴다 (결측도 정보다)
                # 여러 명이면 가장 큰 박스 하나만 — OmniFall 은 단일 피험자 클립이다
                if rr.boxes is not None and len(rr.boxes) > 1:
                    areas = (rr.boxes.xywh[:, 2] * rr.boxes.xywh[:, 3]).cpu().numpy()
                    k = int(areas.argmax())
                else:
                    k = 0
                seq[s + j] = rr.keypoints.data[k].cpu().numpy()

        np.save(dst, normalize(seq, w, h))
        done += 1
        if i % 50 == 0 or i == len(rows):
            el = time.time() - t0
            print(f"  [{i}/{len(rows)}] 저장 {done} 건너뜀 {skipped} 실패 {failed}  "
                  f"{el / max(done, 1):.2f}초/클립  남은 시간 ~{(len(rows) - i) * el / max(done, 1) / 60:.0f}분",
                  flush=True)

    print(f"\n[*] 완료 — 저장 {done} · 건너뜀 {skipped} · 실패 {failed} · {(time.time() - t0) / 60:.1f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
