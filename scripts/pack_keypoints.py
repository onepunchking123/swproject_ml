#!/usr/bin/env python3
"""키포인트 .npy 1,194개 → 단일 .npz 묶음.

Google Drive FUSE 는 **작은 파일을 개별로 읽을 때 극도로 느리다**. 실측에서
`np.load` 가 D 상태(uninterruptible disk sleep)로 2분 넘게 멈췄고, 1,194개를
읽는 동안 GPU 사용률이 0% 였다.

파일 하나로 묶으면 순차 읽기 한 번으로 끝난다. 135MB 를 통째로 읽는 것이
4KB 파일 1,194번 읽는 것보다 훨씬 빠르다.

사용법:
    python pack_keypoints.py --kps <디렉토리> --out kps.npz          # 묶기
    python pack_keypoints.py --kps kps.npz --verify                  # 검증
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kps", type=Path, required=True, help=".npy 디렉토리 또는 .npz")
    ap.add_argument("--out", type=Path, help="출력 .npz")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        t0 = time.time()
        z = np.load(args.kps)
        keys = list(z.files)
        arrs = [z[k] for k in keys[:5]]
        print(f"{args.kps.name}: {len(keys):,}개 · 로드 {time.time()-t0:.1f}초")
        print(f"  예: {keys[0]}  shape {arrs[0].shape}  유한값 {np.isfinite(arrs[0]).all()}")
        return 0

    if not args.out:
        ap.error("--out 필요")
    fs = sorted(args.kps.glob("*.npy"))
    print(f"[*] {len(fs):,}개 묶는 중...")
    t0 = time.time()
    data = {}
    for i, f in enumerate(fs, 1):
        data[f.stem] = np.load(f)
        if i % 200 == 0:
            print(f"    {i}/{len(fs)}  {time.time()-t0:.0f}초", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **data)          # 압축하지 않는다 — 로드 속도 우선
    print(f"[*] 저장 {args.out}  {args.out.stat().st_size/1e6:.0f}MB  {time.time()-t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
