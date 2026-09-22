#!/usr/bin/env python3
"""자르기가 실패한 키포인트를 manifest 구간으로 다시 자른다.

OmniFall Zenodo 배포판의 edf 는 클립 92.5% 가 잘려 있지 않다 — manifest 에는
3초로 적혀 있으나 실제로는 시작 시각부터 영상 끝까지 들어있다 (최대 1,755프레임).
시작 시각이 0 인 클립만 정상이므로, 자르기에서 종료 시각이 적용되지 않은 것으로 보인다.

키포인트는 프레임 단위 배열이므로 영상을 다시 볼 필요 없이 슬라이스로 해결된다.

  잘린 구간 = arr[0 : (end-start) * fps]

fps 는 **정상 클립(start=0)에서 역산한 데이터셋 중앙값**을 쓴다.
클립마다 `프레임수/길이` 로 역산하면 오염된 클립에서 엉뚱한 값이 나온다.

사용법:
    python retrim_keypoints.py --kps <디렉토리> --manifest <csv> --dry-run
    python retrim_keypoints.py --kps <디렉토리> --manifest <csv> --out <디렉토리>
"""
from __future__ import annotations
import argparse, csv, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

TOL = 1.5   # 기대 프레임의 이 배수를 넘으면 자르기 실패로 본다


def estimate_fps(rows, kps: Path) -> dict[str, float]:
    """start==0 인 클립에서 fps 를 역산한다 (그 클립들은 자르기가 정상이다)."""
    acc = defaultdict(list)
    for r in rows:
        if float(r["start"]) != 0.0:
            continue
        dur = float(r["end"]) - float(r["start"])
        if dur <= 0.3:
            continue
        f = kps / f"{r['dataset']}__{Path(r['clip']).stem}.npy"
        if f.exists():
            acc[r["dataset"]].append(np.load(f, mmap_mode="r").shape[0] / dur)
    return {ds: float(np.median(v)) for ds, v in acc.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kps", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, help="출력 디렉토리 (미지정 시 --dry-run 만)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.out and not args.dry_run:
        ap.error("--out 또는 --dry-run 필요")

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    fps = estimate_fps(rows, args.kps)
    print("[*] fps 추정 (start=0 클립 기준)")
    for ds, v in sorted(fps.items()):
        print(f"    {ds:12s} {v:.1f}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    stat = defaultdict(lambda: [0, 0, 0])   # [전체, 재자름, 그대로]
    for r in rows:
        ds = r["dataset"]
        f = args.kps / f"{ds}__{Path(r['clip']).stem}.npy"
        if not f.exists():
            continue
        stat[ds][0] += 1
        a = np.load(f)
        dur = float(r["end"]) - float(r["start"])
        want = int(round(dur * fps.get(ds, 30.0)))

        if want >= 2 and a.shape[0] > want * TOL:
            # 시작은 맞고 끝만 안 잘린 형태이므로 앞에서부터 want 프레임을 취한다
            a = a[:want]
            stat[ds][1] += 1
        else:
            stat[ds][2] += 1
        if args.out:
            np.save(args.out / f.name, a)

    print(f"\n{'데이터셋':12s}{'전체':>8}{'재자름':>10}{'그대로':>10}")
    print("-" * 42)
    for ds in sorted(stat):
        t, c, k = stat[ds]
        print(f"{ds:12s}{t:8d}{c:10d}{k:10d}")
    tot = sum(v[0] for v in stat.values()); cut = sum(v[1] for v in stat.values())
    print(f"{'합계':12s}{tot:8d}{cut:10d}{tot-cut:10d}")
    if args.out:
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
