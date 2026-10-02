#!/usr/bin/env python3
"""URFD (UR Fall Detection) 연속 영상 → 슬라이딩 윈도우 평가. 학습에 쓰지 않은 외부 테스트셋.

AI Hub 샘플이 4장면(비낙상은 한 장면의 8카메라)뿐이라 모델 선택이 그 장면에 과적합될 위험이 있다.
URFD 는 낙상 30편 + 일상 40편(앉기·눕기·물건 줍기)의 **연속 영상**이고 프레임 라벨이 있다:
  -1 서있음/일상 · 0 넘어지는 중 · 1 누워 있음(넘어진 뒤)

입력은 cam0 RGB PNG 시퀀스 zip (fall-01-cam0-rgb.zip 안에 fall-01-cam0-rgb-001.png …), 30fps 640×480.
aihub_pipeline_eval.py 와 같은 구조로 predictions.json 을 남긴다 (make_pipeline_report 재사용 가능).

    python urfd_eval.py --root data/urfd --model models/gru.pt --out runs/urfd_v1
"""
from __future__ import annotations
import argparse, csv, io, json, re, sys, time, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_core import build_model, featurize, normalize, pick_device, resample

FPS = 30.0


def load_labels(root: Path) -> dict[str, np.ndarray]:
    lab = defaultdict(dict)
    for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        for row in csv.reader((root / f).open()):
            lab[row[0]][int(row[1])] = int(row[2])
    return {k: np.array([v.get(i, -1) for i in range(1, max(v) + 1)]) for k, v in lab.items()}


def read_frames(zp: Path, resize: int):
    import cv2
    with zipfile.ZipFile(zp) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".png"))
        for n in names:
            img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            if resize and img.shape[1] > resize:
                img = cv2.resize(img, (resize, int(img.shape[0] * resize / img.shape[1])), interpolation=cv2.INTER_AREA)
            yield img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/urfd"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pose", default="yolo11n-pose.pt")
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--stride-sec", type=float, default=0.5)
    ap.add_argument("--resize", type=int, default=640)
    ap.add_argument("--kps-cache", type=Path, default=Path("runs/urfd_kps"))
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    dev = pick_device()
    ck = torch.load(args.model, map_location=dev, weights_only=False)
    names, seq_len, feat = ck["classes"], ck["seq_len"], ck.get("feat", "pos")
    model = build_model(ck["kind"], len(names), dev, ck.get("in_ch", 3))
    model.load_state_dict(ck["state_dict"]); model.eval()
    fi, fli = names.index("fall"), names.index("fallen")
    print(f"[*] 모델 {ck['kind']} · {names} · 입력 {feat} · {dev}")

    labels = load_labels(args.root)
    zips = sorted(args.root.glob("*-cam0-rgb.zip"))
    print(f"[*] 시퀀스 {len(zips)}개 · 라벨 {len(labels)}개")
    args.kps_cache.mkdir(parents=True, exist_ok=True); args.out.mkdir(parents=True, exist_ok=True)
    pose = None
    win_n, stride_n = int(args.window_sec * FPS), int(args.stride_sec * FPS)
    results = []
    for i, zp in enumerate(zips, 1):
        sid = zp.name.replace("-cam0-rgb.zip", "")
        cache = args.kps_cache / f"{sid}.npy"
        if cache.exists():
            seq = np.load(cache)
        else:
            pose = pose or YOLO(args.pose)
            raw, w, h = [], None, None
            buf = []
            def flush():
                for rr in pose(buf, imgsz=640, conf=0.25, device=0 if dev == "cuda" else dev, verbose=False):
                    k = np.zeros((17, 3), np.float32)
                    if rr.keypoints is not None and len(rr.keypoints.data):
                        idx = 0
                        if rr.boxes is not None and len(rr.boxes) > 1:
                            idx = int((rr.boxes.xywh[:, 2] * rr.boxes.xywh[:, 3]).cpu().numpy().argmax())
                        k = rr.keypoints.data[idx].cpu().numpy()
                    raw.append(k)
                buf.clear()
            for img in read_frames(zp, args.resize):
                if w is None: h, w = img.shape[:2]
                buf.append(img)
                if len(buf) == 32: flush()
            if buf: flush()
            seq = normalize(np.stack(raw), w, h)
            np.save(cache, seq)
        lab = labels.get(sid)
        T = seq.shape[0]
        x = featurize(seq, FPS, feat)
        starts = list(range(0, max(1, T - win_n + 1), stride_n))
        wins = np.stack([resample(x[s:s + win_n], seq_len) for s in starts])
        with torch.no_grad():
            prob = torch.softmax(model(torch.tensor(wins).to(dev)), 1).cpu().numpy()
        risk = prob[:, fi] + prob[:, fli]
        is_fall = sid.startswith("fall")
        gt_s = int(np.argmax(lab == 0)) if is_fall and lab is not None and (lab == 0).any() else 0
        gt_e = int(len(lab) - np.argmax(lab[::-1] == 1)) if is_fall and lab is not None and (lab == 1).any() else 0
        results.append({"scene": sid, "cam": 0, "true_class": "fall" if is_fall else "adl", "is_fall": is_fall,
                        "gt_start": gt_s, "gt_end": gt_e, "starts": starts, "window": win_n,
                        "prob": prob.round(4).tolist(), "risk": risk.round(4).tolist(),
                        "risk_max": float(risk.max()), "peak_frame": int(starts[int(risk.argmax())] + win_n // 2),
                        "n_frames": T})
        print(f"  [{i}/{len(zips)}] {sid}  {T}f  위험최대 {risk.max():.3f}" + (f"  정답 낙상 {gt_s}~{gt_e}f" if is_fall else ""), flush=True)

    (args.out / "predictions.json").write_text(json.dumps({"model": ck["kind"], "classes": names, "fps": FPS, "results": results}), encoding="utf-8")
    # 요약
    fall = [r for r in results if r["is_fall"]]; adl = [r for r in results if not r["is_fall"]]
    rec = np.mean([r["risk_max"] > args.thr for r in fall])
    lat = []
    for r in fall:
        s = np.array(r["starts"]); rk = np.array(r["risk"]); c = s + r["window"] // 2
        h = np.where((rk > args.thr) & (c >= r["gt_start"]))[0]
        if len(h): lat.append((c[h[0]] - r["gt_start"]) / FPS)
    wr, runs = [], []
    for r in adl:
        rk = np.array(r["risk"]) > args.thr; wr.append(rk.mean()); b = c = 0
        for v in rk: c = c + 1 if v else 0; b = max(b, c)
        runs.append(b * args.stride_sec)
    summ = {"fall_recall": float(rec), "latency_median_s": float(np.median(lat)) if lat else None,
            "adl_risk_window_ratio": float(np.mean(wr)), "adl_r1_fired": int(sum(x >= 3 for x in runs)), "adl_n": len(adl),
            "adl_max_run_s": float(max(runs))}
    (args.out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\n낙상 {len(fall)}편 Recall {rec:.3f} · 지연 중앙값 {summ['latency_median_s']}s · "
          f"일상 {len(adl)}편 위험윈도우 {np.mean(wr):.3f} · R1(≥3s) {summ['adl_r1_fired']}/{len(adl)} · 최장 {max(runs):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
