#!/usr/bin/env python3
"""저장된 창 단위 예측(predictions.json)에 방치 감지 규칙(R1) 변형을 재생해 비교한다. 재학습·YOLO 재실행 없음.

입력은 aihub_pipeline_eval.py · urfd_eval.py 가 남긴 predictions.json (창마다 prob[normal, fall, fallen]).
상태 판정은 realtime_demo.py 와 같다 — 위험(fall+fallen) 확률 0.5 진입 / 0.35 해제 히스테리시스,
위험 중이면 fall·fallen 중 큰 쪽이 상태.

규칙 변형
  base      위험 상태가 T 초 **연속** → 발화 (현재 realtime_demo 동작)
  ratio     최근 T 초 창 중 위험 비율 ≥ r → 발화. 한 창 튀어도 타이머가 0 이 되지 않는다
  gate      FALL(넘어지는 동작) 이벤트 뒤 W 초 안에서만 base 판정. 처음부터 누워 있던 장면은 무시
  gate+ratio  둘 다

지표 (서비스 기준 세 가지를 같이 본다 — docs/results_data_expansion.md "평가 지표 교훈")
  낙상 순간    AI Hub·URFD 낙상 영상에서 FALL 이벤트가 정답 낙상 이후 1번이라도 나는가
  방치 감지    AI Hub 낙상 24편 중 R1 발화 (URFD 는 누운 구간이 3초 미만이라 제외)
  일상 오경보  AI Hub 비낙상 8 + URFD 일상 40 중 R1 발화

    python replay_rules.py --set v1=runs/vel/aihub_posvel,runs/urfd_v1 --set v4=runs/v4/aihub,runs/urfd_v4
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

THR, THR_EXIT = 0.5, 0.35


def states(prob: np.ndarray) -> list[str]:
    out, in_risk = [], False
    for p in prob:
        risk = p[1] + p[2]
        in_risk = risk > THR or (in_risk and risk > THR_EXIT)
        out.append(("fallen" if p[2] >= p[1] else "fall") if in_risk else "normal")
    return out


def replay(ts, st, rule, T=3.0, r=0.8, W=8.0, cooldown=2.0):
    """(FALL 시각 목록, R1 첫 발화 시각 or None)."""
    falls, last_fall, prev, since, r1 = [], -1e9, "normal", None, None
    risk = [s != "normal" for s in st]
    for i, (t, s) in enumerate(zip(ts, st)):
        if s == "fall" and prev != "fall" and t - last_fall >= cooldown:
            falls.append(t); last_fall = t
        prev = s
        if r1 is not None:
            continue
        armed = True
        if "gate" in rule:
            armed = any(0 <= t - f <= W for f in falls)
        if not armed:
            since = None
            continue
        if "ratio" in rule:
            lo = t - T
            idx = [j for j in range(i + 1) if ts[j] > lo - 1e-9]
            if ts[i] - ts[idx[0]] >= T - 0.5 - 1e-9 and np.mean([risk[j] for j in idx]) >= r:
                r1 = t
        else:
            if risk[i]:
                since = t if since is None else since
                if t - since >= T - 1e-9:
                    r1 = t
            else:
                since = None
    return falls, r1


def evaluate(pred: Path, fps_default: float, rule: str, **kw):
    d = json.loads(pred.read_text(encoding="utf-8"))
    fps = d.get("fps", fps_default)
    rows = []
    for r in d["results"]:
        prob = np.array(r["prob"])
        ts = [(s + r["window"] / 2) / fps for s in r["starts"]]
        falls, r1 = replay(ts, states(prob), rule, **kw)
        gt = r.get("gt_start", 0) / fps
        rows.append(dict(scene=r["scene"], is_fall=r["is_fall"], falls=falls, r1=r1, gt=gt,
                         caught=any(f >= gt - 0.5 for f in falls) if r["is_fall"] else None))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, help="이름=aihub_dir,urfd_dir")
    ap.add_argument("--rules", default="base,ratio,gate,gate+ratio")
    ap.add_argument("--t", type=float, default=3.0, help="R1 지속 초")
    ap.add_argument("--r", type=float, default=0.8, help="ratio 규칙의 위험 비율")
    ap.add_argument("--w", type=float, default=8.0, help="gate 규칙: FALL 뒤 몇 초까지 R1 을 허용")
    ap.add_argument("--json", type=Path, help="결과 저장")
    args = ap.parse_args()

    out = []
    print(f"R1 지속 {args.t}s · ratio {args.r} · gate 창 {args.w}s\n")
    print(f"{'모델':6s} {'규칙':11s} {'낙상순간 AI':>11s} {'URFD':>6s} {'방치감지 AI':>11s} {'지연(s)':>7s} {'오경보 AI':>9s} {'URFD':>6s}")
    for spec in args.set:
        name, dirs = spec.split("=", 1)
        ai_dir, ur_dir = dirs.split(",")
        for rule in args.rules.split(","):
            kw = dict(T=args.t, r=args.r, W=args.w)
            A = evaluate(Path(ai_dir) / "predictions.json", 60.0, rule, **kw)
            U = evaluate(Path(ur_dir) / "predictions.json", 30.0, rule, **kw)
            af, an = [x for x in A if x["is_fall"]], [x for x in A if not x["is_fall"]]
            uf, un = [x for x in U if x["is_fall"]], [x for x in U if not x["is_fall"]]
            lat = [x["r1"] - x["gt"] for x in af if x["r1"] is not None]
            m = dict(model=name, rule=rule,
                     fall_ai=sum(x["caught"] for x in af), fall_ai_n=len(af),
                     fall_ur=sum(x["caught"] for x in uf), fall_ur_n=len(uf),
                     r1_ai_fall=sum(x["r1"] is not None for x in af),
                     r1_latency=float(np.median(lat)) if lat else None,
                     r1_ai_adl=sum(x["r1"] is not None for x in an), ai_adl_n=len(an),
                     r1_ur_adl=sum(x["r1"] is not None for x in un), ur_adl_n=len(un),
                     ur_adl_fired=[x["scene"] for x in un if x["r1"] is not None])
            out.append(m)
            print(f"{name:6s} {rule:11s} {m['fall_ai']:>6d}/{m['fall_ai_n']:<4d} {m['fall_ur']:>3d}/{m['fall_ur_n']:<2d}"
                  f" {m['r1_ai_fall']:>6d}/{m['fall_ai_n']:<4d} {('%.1f' % m['r1_latency']) if lat else '-':>7s}"
                  f" {m['r1_ai_adl']:>5d}/{m['ai_adl_n']:<3d} {m['r1_ur_adl']:>3d}/{m['ur_adl_n']:<2d}")
        print()
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
