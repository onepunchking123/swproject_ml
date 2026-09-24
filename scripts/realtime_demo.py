#!/usr/bin/env python3
"""실시간 낙상·미동 감지 프로토타입 — 영상 파일 입력.

    VideoCapture → resize → YOLOv11n-pose → 최대 박스 1인 → normalize()
                 → ring buffer(window_sec) → stride 마다 resample(64) → GRU → 상태
                 → 상태 머신(R1 낙상 후 방치 · R2 일반 미동) → 이벤트
                 → 오버레이(골격·상태·확률·FPS·알림) → 화면 / 저장 / events.jsonl

미동 감지는 모델이 아니라 **규칙**이 한다. 모델은 프레임 상태(normal/fall/fallen)만 내고,
시간 누적은 ImmobilityDetector 가 맡는다.

  R1  fallen 이 t_fallen 초 이상 지속           → CRITICAL  (넘어진 뒤 일어나지 못함)
  R2  움직임 < eps 가 t_still 초 지속 AND 몸통 수평 → WARNING   (누운 채 미동)

R2 의 "몸통 수평" 조건이 오탐을 막는다 — 앉아서 가만히 있는 것은 정상이다.

**사람 미검출 윈도우에서는 GRU 를 돌리지 않는다.** 전부 0 인 입력에 모델이
fallen 0.79 를 내는 것을 확인했다(pipeline_core 검증). 화면에서 사람이 사라지면
"넘어져 있다" 로 오판하므로 detection_rate < min_det 이면 상태를 no_person 으로 둔다.

사용법:
    python realtime_demo.py --source 00003_H_A_FY_C1          # data/ 아래에서 이름으로 찾음, 라벨 자동
    python realtime_demo.py --source path/to/clip.mp4 [--gt-json label.json] \
        [--save-video out.mp4] [--events out.jsonl] [--no-display]
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_core import (SKELETON, normalize, resample, torso_horizontality,
                           motion_energy, detection_rate, pick_device, load_checkpoint)

STATE_COLOR = {"normal": (80, 200, 80), "fall": (0, 100, 255), "fallen": (0, 0, 230),
               "no_person": (150, 150, 150)}


class ImmobilityDetector:
    """상태 시계열을 받아 R1·R2 이벤트를 낸다. 각 규칙은 발화 후 상태가 해제될 때까지 재발화하지 않는다."""

    def __init__(self, fps: float, t_fallen: float, t_still: float, eps: float, horiz_thr: float,
                 fall_cooldown: float = 2.0):
        self.fps = fps
        self.t_fallen, self.t_still, self.eps, self.horiz_thr = t_fallen, t_still, eps, horiz_thr
        self.fall_cooldown, self.last_fall_t = fall_cooldown, -1e9
        self.fallen_since: float | None = None
        self.still_since: float | None = None
        self.r1_fired = self.r2_fired = False
        self.prev_state = "no_person"

    def update(self, t: float, state: str, window: np.ndarray | None) -> list[tuple[str, str]]:
        """(event, rule) 목록을 돌려준다. window 는 최근 t_still 초 분량의 정규화 키포인트."""
        events = []

        # 낙상 순간 — 상승 에지
        if state == "fall" and self.prev_state != "fall" and t - self.last_fall_t >= self.fall_cooldown:
            events.append(("FALL", "model")); self.last_fall_t = t

        # R1 — 위험 상태(fall/fallen) 지속. 모델의 fall↔fallen 경계는 흐리지만 둘 다
        # "바닥에 있다" 는 뜻이므로, 정확한 라벨이 아니라 지속 시간이 관심사다.
        if state in ("fall", "fallen"):
            self.fallen_since = self.fallen_since or t
            if not self.r1_fired and t - self.fallen_since >= self.t_fallen:
                events.append(("FALLEN_IMMOBILE", "R1")); self.r1_fired = True
        else:
            self.fallen_since, self.r1_fired = None, False

        # R2 — 미동 + 수평. 사람이 보일 때만 판단한다
        if window is not None and state != "no_person":
            horiz = torso_horizontality(window[-1])
            moving = motion_energy(window)
            if moving < self.eps and horiz > self.horiz_thr:
                self.still_since = self.still_since or t
                if not self.r2_fired and t - self.still_since >= self.t_still:
                    events.append(("STILL_HORIZONTAL", "R2")); self.r2_fired = True
            else:
                self.still_since, self.r2_fired = None, False
        else:
            self.still_since, self.r2_fired = None, False

        self.prev_state = state
        return events


def largest_person(res) -> np.ndarray | None:
    """YOLO 결과에서 가장 큰 박스 한 명의 (17, 3) 키포인트. 없으면 None."""
    if res.keypoints is None or len(res.keypoints.data) == 0:
        return None
    k = 0
    if res.boxes is not None and len(res.boxes) > 1:
        wh = res.boxes.xywh[:, 2:4].cpu().numpy()
        k = int((wh[:, 0] * wh[:, 1]).argmax())
    return res.keypoints.data[k].cpu().numpy()


def draw(frame, kps_px, state, prob, classes, fps, banner, t, total_t, gt):
    """오버레이. kps_px 는 리사이즈된 프레임 픽셀 좌표 (17, 3)."""
    h, w = frame.shape[:2]
    col = STATE_COLOR.get(state, (200, 200, 200))
    if kps_px is not None:
        for i, j in SKELETON:
            if kps_px[i, 2] > 0.3 and kps_px[j, 2] > 0.3:
                cv2.line(frame, tuple(kps_px[i, :2].astype(int)), tuple(kps_px[j, :2].astype(int)), col, 2)
        for x, y, c in kps_px:
            if c > 0.3: cv2.circle(frame, (int(x), int(y)), 3, col, -1)

    # 상단 상태 패널
    cv2.rectangle(frame, (0, 0), (w, 64), (20, 20, 20), -1)
    cv2.putText(frame, state.upper(), (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
    x0 = 230
    for name, p in zip(classes, prob):
        cv2.putText(frame, f"{name} {p:.2f}", (x0, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.rectangle(frame, (x0, 32), (x0 + int(120 * p), 44), STATE_COLOR.get(name, (200, 200, 200)), -1)
        cv2.rectangle(frame, (x0, 32), (x0 + 120, 44), (90, 90, 90), 1)
        x0 += 150
    cv2.putText(frame, f"{fps:5.1f} FPS  t={t:5.1f}s", (w - 230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    # 하단 타임라인 — 정답 구간(있으면) + 현재 위치
    y = h - 22
    cv2.rectangle(frame, (12, y), (w - 12, y + 10), (60, 60, 60), -1)
    if gt and gt[1] > gt[0] and total_t > 0:
        a = 12 + int((w - 24) * gt[0] / total_t); b = 12 + int((w - 24) * gt[1] / total_t)
        cv2.rectangle(frame, (a, y), (b, y + 10), (60, 160, 230), -1)
    if total_t > 0:
        cx = 12 + int((w - 24) * min(t / total_t, 1.0))
        cv2.line(frame, (cx, y - 4), (cx, y + 14), (255, 255, 255), 2)

    # 알림 배너
    if banner:
        text, rule, color = banner
        cv2.rectangle(frame, (0, 72), (w, 118), color, -1)
        cv2.putText(frame, f"!!  {text}  [{rule}]", (16, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="영상 파일 경로, 또는 data/ 아래에서 찾을 파일명 (예: 00003_H_A_FY_C1)")
    ap.add_argument("--data-root", type=Path, default=Path("data"),
                    help="--source 가 경로가 아닐 때 검색할 루트")
    ap.add_argument("--model", type=Path, default=Path("runs/models/gru.pt"))
    ap.add_argument("--pose", default="yolo11n-pose.pt")
    ap.add_argument("--device", default="auto", help="auto|mps|cuda|cpu")
    ap.add_argument("--resize", type=int, default=960, help="가로 픽셀 (0=원본)")
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--stride-sec", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.5, help="위험 진입 임계 (fall+fallen 확률)")
    ap.add_argument("--thr-exit", type=float, default=0.35,
                    help="위험 해제 임계 — 진입보다 낮게 둬 경계에서 상태가 떨리는 것을 막는다(히스테리시스)")
    ap.add_argument("--fall-cooldown", type=float, default=2.0, help="FALL 이벤트 재발화 금지 초")
    ap.add_argument("--min-det", type=float, default=0.5, help="윈도우 내 최소 검출률 — 미만이면 no_person")
    ap.add_argument("--t-fallen", type=float, default=3.0, help="R1: fallen 지속 초")
    ap.add_argument("--t-still", type=float, default=5.0, help="R2: 미동 지속 초")
    ap.add_argument("--eps", type=float, default=0.02, help="R2: 움직임 임계 (몸통 길이 단위)")
    ap.add_argument("--horiz-thr", type=float, default=1.0, help="R2: 수평도 임계 (|x|/|y|)")
    ap.add_argument("--gt-json", type=Path, help="AI Hub 라벨 — 정답 낙상 구간 오버레이")
    ap.add_argument("--save-video", type=Path)
    ap.add_argument("--events", type=Path, help="events.jsonl 출력")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, help="디버그용 상한")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    dev = pick_device(args.device)
    model, classes, seq_len, kind = load_checkpoint(args.model, dev)
    pose = YOLO(args.pose)
    fi = classes.index("fall"); fli = classes.index("fallen")
    print(f"[*] {kind} · {classes} · seq {seq_len} · device {dev}")

    # --source 가 파일이 아니면 data/ 아래에서 <이름>.mp4 를 찾는다. 데모 영상을 프로젝트
    # 안에 두고 긴 경로 없이 이름만으로 돌리기 위해서다.
    if not args.source.exists():
        stem = args.source.stem if args.source.suffix == ".mp4" else args.source.name
        hits = sorted(args.data_root.rglob(f"{stem}.mp4"))
        if not hits:
            print(f"[!] 영상을 찾을 수 없다: {args.source}  ({args.data_root}/ 아래 검색)"); return 1
        args.source = hits[0]
        print(f"[*] 소스 → {args.source}")
    # 라벨을 안 줬으면 같은 이름의 .json 을 data/ 아래에서 찾는다 (AI Hub 샘플 구조)
    if args.gt_json is None:
        cand = sorted(args.data_root.rglob(f"{args.source.stem}.json"),
                      key=lambda q: ("영상" not in q.parts, str(q)))   # 영상/ 라벨 우선 (센서/ 것과 내용은 같다)
        if cand:
            args.gt_json = cand[0]
            print(f"[*] 라벨 → {args.gt_json}")
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        print(f"[!] 영상을 열 수 없다: {args.source}"); return 1
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.resize / W0 if args.resize and W0 > args.resize else 1.0
    W, H = int(W0 * scale), int(H0 * scale)
    win_n = max(int(args.window_sec * src_fps), seq_len // 2)
    stride_n = max(int(args.stride_sec * src_fps), 1)
    still_n = max(int(args.t_still * src_fps), 2)
    print(f"[*] {args.source.name} · {W0}x{H0}@{src_fps:.0f}fps → {W}x{H} · 윈도우 {win_n}f · stride {stride_n}f · 총 {n_total}f")

    gt = None
    if args.gt_json and args.gt_json.exists():
        sd = json.loads(args.gt_json.read_text(encoding="utf-8")).get("sensordata", {})
        gt = (sd.get("fall_start_frame", 0) / src_fps, sd.get("fall_end_frame", 0) / src_fps)
        print(f"[*] 정답 낙상 구간 {gt[0]:.1f}~{gt[1]:.1f}s")

    writer = None
    if args.save_video:
        args.save_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))
    ev_f = None
    if args.events:
        args.events.parent.mkdir(parents=True, exist_ok=True)
        ev_f = args.events.open("w", encoding="utf-8")

    buf: deque[np.ndarray] = deque(maxlen=max(win_n, still_n))
    det = ImmobilityDetector(src_fps, args.t_fallen, args.t_still, args.eps, args.horiz_thr, args.fall_cooldown)
    in_risk = False
    state, prob = "no_person", np.zeros(len(classes), np.float32)
    banner, banner_until = None, 0
    n = 0; t0 = time.time(); n_events = 0
    yolo_dev = 0 if dev == "cuda" else dev

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        t = n / src_fps

        res = pose(frame, imgsz=640, conf=0.25, device=yolo_dev, verbose=False)[0]
        kps_px = largest_person(res)
        raw = kps_px if kps_px is not None else np.zeros((17, 3), np.float32)
        buf.append(normalize(raw[None], W, H)[0])

        # stride 마다 판정
        if n % stride_n == 0 and len(buf) >= win_n // 2:
            win = np.stack(list(buf)[-win_n:])
            if detection_rate(win) < args.min_det:
                state, prob = "no_person", np.zeros(len(classes), np.float32)
            else:
                x = torch.tensor(resample(win, seq_len)[None]).to(dev)
                with torch.no_grad():
                    prob = torch.softmax(model(x), 1).cpu().numpy()[0]
                risk = prob[fi] + prob[fli]
                # 히스테리시스: 진입 thr, 해제 thr_exit. 경계 근처에서 normal↔fall 떨림을 막는다
                in_risk = risk > args.thr or (in_risk and risk > args.thr_exit)
                if in_risk:
                    state = "fallen" if prob[fli] >= prob[fi] else "fall"
                else:
                    state = "normal"
            if ev_f:  # 상태 행 — 보고서 타임라인 재구성용
                ev_f.write(json.dumps({"frame": n, "t_sec": round(t, 2), "state": state,
                                       "prob": prob.round(4).tolist(), "event": None, "rule": None}) + "\n")

        # 검출기는 매 프레임 — 지속 시간을 stride(0.5s) 단위로 양자화하면 임계 근처에서 놓친다
        # (실측: fallen 2.91s 로 3.0s 임계를 0.09s 차이로 못 넘김)
        if len(buf) >= win_n // 2:
            still_win = np.stack(list(buf)[-still_n:]) if len(buf) >= still_n else None
            for ev, rule in det.update(t, state, still_win):
                n_events += 1
                color = (0, 0, 200) if rule == "R1" else ((0, 120, 220) if rule == "R2" else (0, 90, 255))
                banner, banner_until = (ev, rule, color), n + int(2.0 * src_fps)
                print(f"  [{n:4d}f {t:5.1f}s] {ev:18s} rule={rule}  state={state}  prob={prob.round(2)}", flush=True)
                if ev_f:
                    ev_f.write(json.dumps({"frame": n, "t_sec": round(t, 2), "state": state,
                                           "prob": prob.round(4).tolist(), "event": ev, "rule": rule}) + "\n")

        if n >= banner_until:
            banner = None
        fps_now = (n + 1) / max(time.time() - t0, 1e-6)
        out = draw(frame, kps_px, state, prob, classes, fps_now, banner, t, n_total / src_fps, gt)
        if writer: writer.write(out)
        if not args.no_display:
            cv2.imshow("fall-detection prototype", out)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        n += 1

    el = time.time() - t0
    cap.release()
    if writer: writer.release()
    if ev_f: ev_f.close()
    if not args.no_display: cv2.destroyAllWindows()
    print(f"\n[*] {n}프레임 · {el:.1f}초 · 처리 {n/max(el,1e-6):.1f} FPS · 이벤트 {n_events}건")
    if args.save_video: print(f"    영상 → {args.save_video}")
    if args.events: print(f"    이벤트 → {args.events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
