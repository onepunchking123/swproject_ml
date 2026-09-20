#!/usr/bin/env python3
"""AI Hub 공식 영상 모델(RandomForest) 베이스라인 추론.

원 소스(`video_mediapipe.ipynb`, `video_smote_train.ipynb`)의 파이프라인을 재현한다.
차이점은 하드코딩된 로컬 경로를 제거하고 CLI 로 만든 것뿐이며,
키포인트 추출 방식은 원본과 동일하다.

    영상 → MediaPipe Holistic → 1,662 차원/프레임
         → 600 프레임 → flatten(997,200) → RandomForest → 예측

원본 `extract_keypoints()` 와 동일한 구성:
    pose 33×4 + face 468×3 + 왼손 21×3 + 오른손 21×3 = 1,662

카메라 번호(C1~C8)는 파일명에서 판별해 해당 모델을 선택한다.

사용법:
    python aihub_infer.py --data <샘플루트> --models <모델루트> --task fnf
    python aihub_infer.py ... --task fd      # 낙상유형 3분류
    python aihub_infer.py ... --limit 4      # 일부만 빠르게 확인
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

# MediaPipe 로그 억제 (추론 진행 상황을 가리므로)
import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import joblib
import mediapipe as mp

SEQ_LEN = 600          # 모델이 요구하는 프레임 수
FEAT_DIM = 1662        # pose 132 + face 1404 + hands 126
POSE_N, FACE_N, HAND_N = 33, 468, 21

# 디렉토리명 → 라벨
LABEL_OF_DIR = {"FY": "FY", "BY": "BY", "SY": "SY", "N": "N"}

# FNF: 낙상유무 2분류. 학습 시 LabelEncoder 가 ['FALL','NonFall'] 를 정렬해
# FALL=0, NonFall=1 로 매핑했다 (video_smote_train.ipynb 의 FNF_SMOTE 참조).
FNF_CLASSES = ["FALL", "NonFall"]

# FD: 낙상유형 3분류. 라벨 리스트가 ['BY','FY','SY'] 로 정렬된다.
FD_CLASSES = ["BY", "FY", "SY"]


def extract_keypoints(results) -> np.ndarray:
    """원 소스의 extract_keypoints() 를 그대로 따른다.

    랜드마크가 없으면 0 으로 채운다 — 이 결측 패턴 자체가 학습에 반영되어 있다.
    """
    pose = (np.array([[r.x, r.y, r.z, r.visibility]
                      for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(POSE_N * 4))
    face = (np.array([[r.x, r.y, r.z]
                      for r in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(FACE_N * 3))
    lh = (np.array([[r.x, r.y, r.z]
                    for r in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(HAND_N * 3))
    rh = (np.array([[r.x, r.y, r.z]
                    for r in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(HAND_N * 3))
    return np.concatenate([pose, face, lh, rh])


def video_to_sequence(path: Path, holistic, max_frames: int = SEQ_LEN,
                      resize_to: int | None = 960) -> np.ndarray | None:
    """영상 한 편을 (600, 1662) 배열로 변환한다.

    4K 원본은 MediaPipe 에 그대로 넣으면 매우 느리므로 축소한다.
    좌표는 정규화(0~1) 되어 반환되므로 리사이즈가 결과에 영향을 주지 않는다.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_to and frame.shape[1] > resize_to:
            h = int(frame.shape[0] * resize_to / frame.shape[1])
            frame = cv2.resize(frame, (resize_to, h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        frames.append(extract_keypoints(holistic.process(img)))
    cap.release()

    if len(frames) < max_frames:
        return None
    return np.array(frames, dtype=np.float32)


def camera_of(name: str) -> int | None:
    m = re.search(r"_C(\d)\b", name) or re.search(r"_C(\d)\.", name)
    return int(m.group(1)) if m else None


def label_of(path: Path) -> str | None:
    for part in path.parts:
        if part in LABEL_OF_DIR:
            return LABEL_OF_DIR[part]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="샘플 영상 루트")
    ap.add_argument("--models", type=Path, required=True, help="AI학습모델파일/영상 경로")
    ap.add_argument("--task", choices=["fnf", "fd"], default="fnf",
                    help="fnf=낙상유무 2분류, fd=낙상유형 3분류")
    ap.add_argument("--limit", type=int, help="처리할 영상 수 제한 (빠른 확인용)")
    ap.add_argument("--resize", type=int, default=960,
                    help="MediaPipe 입력 가로 크기 (0=원본)")
    ap.add_argument("--cache", type=Path, default=Path("runs/keypoints"),
                    help="추출한 키포인트 캐시 디렉토리")
    ap.add_argument("--out", type=Path, default=Path("runs/aihub_baseline"))
    args = ap.parse_args()

    videos = sorted(args.data.rglob("*.mp4"))
    if args.task == "fd":
        videos = [v for v in videos if label_of(v) != "N"]   # 유형 분류는 낙상만
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        print("영상을 찾지 못했습니다:", args.data, file=sys.stderr)
        return 1

    model_dir = args.models / ("낙상분류" if args.task == "fnf" else "낙상유형")
    prefix = "FNF_RF_SMOTE_CAM_" if args.task == "fnf" else "RF_SMOTE_CAM_"
    classes = FNF_CLASSES if args.task == "fnf" else FD_CLASSES

    print(f"[*] task={args.task}  영상 {len(videos)}개  모델 {model_dir.name}")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    # --- 1) 키포인트 추출 ---------------------------------------------------
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.1, min_tracking_confidence=0.1)  # 원본과 동일

    seqs, metas = [], []
    t_extract = time.time()
    for i, v in enumerate(videos, 1):
        cache_f = args.cache / f"{v.stem}.npy"
        if cache_f.exists():
            seq = np.load(cache_f)
            status = "캐시"
        else:
            t0 = time.time()
            seq = video_to_sequence(v, holistic, resize_to=args.resize or None)
            if seq is None:
                print(f"  [{i}/{len(videos)}] {v.stem}: 600프레임 미만 — 건너뜀")
                continue
            np.save(cache_f, seq)
            status = f"{time.time()-t0:.0f}초"
        seqs.append(seq)
        metas.append({"file": v.stem, "cam": camera_of(v.stem), "true": label_of(v)})
        print(f"  [{i}/{len(videos)}] {v.stem}  cam=C{camera_of(v.stem)}  "
              f"true={label_of(v)}  ({status})")
    holistic.close()
    print(f"[*] 키포인트 추출 완료 ({time.time()-t_extract:.0f}초)\n")

    if not seqs:
        print("처리된 영상이 없습니다.", file=sys.stderr)
        return 1

    # --- 2) 카메라별 모델로 추론 -------------------------------------------
    results = []
    by_cam: dict[int, list[int]] = {}
    for idx, m in enumerate(metas):
        by_cam.setdefault(m["cam"], []).append(idx)

    for cam, idxs in sorted(by_cam.items()):
        mpath = model_dir / f"{prefix}{cam}.pkl"
        if not mpath.exists():
            print(f"[!] 모델 없음: {mpath.name}")
            continue
        model = joblib.load(mpath)
        X = np.stack([seqs[i].reshape(-1) for i in idxs])   # (n, 997200)
        if X.shape[1] != model.n_features_in_:
            print(f"[!] 차원 불일치 C{cam}: {X.shape[1]} vs {model.n_features_in_}")
            continue
        pred = model.predict(X)
        prob = model.predict_proba(X)
        for j, i in enumerate(idxs):
            results.append({
                **metas[i],
                "pred_idx": int(pred[j]),
                "pred": classes[int(pred[j])] if int(pred[j]) < len(classes) else str(pred[j]),
                "prob": [round(float(p), 4) for p in prob[j]],
            })
        del X, model
        print(f"[*] C{cam}: {len(idxs)}개 추론 완료")

    # --- 3) 결과 ------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"{'파일':<26}{'cam':>5}{'실제':>7}{'예측':>10}{'확률':>22}")
    print("-" * 74)
    for r in results:
        probs = " ".join(f"{p:.2f}" for p in r["prob"])
        print(f"{r['file']:<26}{'C'+str(r['cam']):>5}{r['true']:>7}"
              f"{r['pred']:>10}{probs:>22}")

    # 정확도 — FNF 는 낙상 3종을 FALL 로 묶는다
    if args.task == "fnf":
        def truth(r): return "NonFall" if r["true"] == "N" else "FALL"
    else:
        def truth(r): return r["true"]

    correct = sum(1 for r in results if r["pred"] == truth(r))
    print("-" * 74)
    print(f"정확도: {correct}/{len(results)} = {correct/len(results):.1%}")

    # confusion matrix
    labels = sorted({truth(r) for r in results} | {r["pred"] for r in results})
    print(f"\nConfusion matrix (행=실제, 열=예측)")
    print(f"{'':>10}" + "".join(f"{l:>10}" for l in labels))
    for a in labels:
        row = [sum(1 for r in results if truth(r) == a and r["pred"] == b) for b in labels]
        print(f"{a:>10}" + "".join(f"{c:>10}" for c in row))

    out_f = args.out / f"{args.task}_results.json"
    out_f.write_text(json.dumps({
        "task": args.task,
        "n": len(results),
        "accuracy": correct / len(results),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out_f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
