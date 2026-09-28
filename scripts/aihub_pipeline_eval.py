#!/usr/bin/env python3
"""AI Hub 샘플 영상 → 전체 파이프라인 → 판정 + 시각 보고서용 예측 저장.

    영상(600f) → YOLOv11-pose → 키포인트 → 슬라이딩 윈도우 → 시계열 모델 → 확률

학습 데이터(OmniFall)는 구간별로 잘린 짧은 클립이지만 AI Hub 샘플은 10초짜리
통짜 영상이다. 윈도우를 밀어가며 판정해 **프레임별 확률 시계열**을 만든다.

AI Hub 라벨에는 `fall_start_frame`/`fall_end_frame` 이 있으므로 감지 지연과
구간 IoU 까지 잴 수 있다. 베이스라인(RandomForest)은 영상 단위 판정이라
불가능한 평가다.

사용법:
    python aihub_pipeline_eval.py --sample <샘플루트> --model runs/models/gru.pt \
        --out runs/aihub_sample
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_core import build_model, featurize, normalize, pick_device, resample  # noqa: F401 (re-export)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/aihub_sample"))
    ap.add_argument("--pose", default="yolo11n-pose.pt")
    ap.add_argument("--window", type=int, default=120, help="윈도우 프레임 (2초@60fps)")
    ap.add_argument("--stride", type=int, default=30, help="윈도우 간격")
    ap.add_argument("--resize", type=int, default=960)
    ap.add_argument("--kps-cache", type=Path, default=Path("runs/aihub_kps"))
    args = ap.parse_args()

    import torch, cv2
    from ultralytics import YOLO
    dev = pick_device()

    ck = torch.load(args.model, map_location=dev, weights_only=False)
    names, seq_len, feat = ck["classes"], ck["seq_len"], ck.get("feat", "pos")
    model = build_model(ck["kind"], len(names), dev, ck.get("in_ch", 3))
    model.load_state_dict(ck["state_dict"]); model.eval()
    print(f"[*] 모델 {ck['kind']} · 클래스 {names} · seq {seq_len} · 입력 {feat} · {dev}")

    vids = sorted((args.sample / "01.원천데이터" / "영상").rglob("*.mp4"))
    labels = {}
    for j in (args.sample / "02.라벨링데이터" / "영상").rglob("*.json"):
        d = json.loads(j.read_text(encoding="utf-8"))
        labels[d["metadata"]["scene_id"]] = d
    print(f"[*] 영상 {len(vids)}개 · 라벨 {len(labels)}개")

    args.kps_cache.mkdir(parents=True, exist_ok=True)
    pose = YOLO(args.pose)
    results = []

    for i, v in enumerate(vids, 1):
        sid = v.stem
        cache = args.kps_cache / f"{sid}.npy"
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        if cache.exists():
            cap.release()
            seq = np.load(cache)
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok: break
                if args.resize and fr.shape[1] > args.resize:
                    hh = int(fr.shape[0]*args.resize/fr.shape[1])
                    fr = cv2.resize(fr, (args.resize, hh), interpolation=cv2.INTER_AREA)
                frames.append(fr)
            cap.release()
            raw = np.zeros((len(frames), 17, 3), np.float32)
            for s in range(0, len(frames), 32):
                for k, rr in enumerate(pose(frames[s:s+32], imgsz=640, conf=0.25,
                                            device=0 if dev=="cuda" else "cpu", verbose=False)):
                    if rr.keypoints is None or len(rr.keypoints.data) == 0: continue
                    idx = 0
                    if rr.boxes is not None and len(rr.boxes) > 1:
                        idx = int((rr.boxes.xywh[:,2]*rr.boxes.xywh[:,3]).cpu().numpy().argmax())
                    raw[s+k] = rr.keypoints.data[idx].cpu().numpy()
            seq = normalize(raw, w, h)
            np.save(cache, seq)

        # 슬라이딩 윈도우 추론 — 캐시는 정규화 좌표 3채널이므로 속도는 여기서 붙인다
        seq = featurize(seq, fps, feat)
        T = seq.shape[0]
        starts = list(range(0, max(1, T - args.window + 1), args.stride))
        wins = np.stack([resample(seq[s:s+args.window], seq_len) for s in starts])
        with torch.no_grad():
            prob = torch.softmax(model(torch.tensor(wins).to(dev)), 1).cpu().numpy()

        lab = labels.get(sid, {})
        si = lab.get("scene_info", {}); sd = lab.get("sensordata", {})
        gt_s, gt_e = sd.get("fall_start_frame", 0), sd.get("fall_end_frame", 0)
        cam = int(re.search(r"_C(\d)", sid).group(1)) if re.search(r"_C(\d)", sid) else 0

        fi = names.index("fall"); fli = names.index("fallen") if "fallen" in names else fi
        risk = prob[:, fi] + (prob[:, fli] if fli != fi else 0)
        results.append({
            "scene": sid, "cam": cam,
            "true_class": si.get("scene_cat_name", "?"),
            "is_fall": si.get("scene_IsFall") == "낙상",
            "gt_start": gt_s, "gt_end": gt_e,
            "starts": starts, "window": args.window,
            "prob": prob.round(4).tolist(), "risk": risk.round(4).tolist(),
            "risk_max": float(risk.max()),
            "peak_frame": int(starts[int(risk.argmax())] + args.window // 2),
        })
        print(f"  [{i}/{len(vids)}] {sid}  위험최대 {risk.max():.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions.json").write_text(
        json.dumps({"model": ck["kind"], "classes": names, "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out/'predictions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
