#!/usr/bin/env python3
"""헤더 fps 로 잘못 잘린 클립을 키포인트 단계에서 다시 자른다 (UP_Fall).

UP_Fall Zenodo 배포판은 컨테이너 헤더가 30fps 인데 실제 프레임은 약 18fps 다 (원본 UP-Fall 은 18fps 카메라).
클립 절단기가 헤더 fps 로 프레임 인덱스를 계산해 라벨 구간 [s, e] 초가 **원본 프레임 [30s, 30e]** 로
잘렸다 — 실제 시각으로는 1.67배 뒤쪽이다. 증거:

  · 같은 원본의 클립 프레임 수를 합치면 (마지막 end × 30) 의 60% 뿐이다
  · 빈 껍데기(5.7KB, 0프레임) 43개 중 41개가 start×30 ≥ 총 프레임 — 영상 끝을 지나서 자른 것
  · 0~0.58s 클립이 17프레임(=0.58×30), 그 다음 클립이 나머지 전부

모든 클립이 원본의 연속 구간이므로 되돌릴 수 있다: 각 클립을 원본 프레임 축 [30s, 30s+T) 에 놓고,
라벨 구간을 실제 fps 로 다시 자른다 — [round(fps·s), round(fps·e)). 영상 재추출이 필요 없다.
필요한 프레임이 추출된 범위 밖(틈)이면 coverage 가 떨어지고, --min-cover 미만은 버린다.

실제 fps 는 틈 없는 원본에서 총프레임/마지막end 의 중앙값으로 추정한다 (실측 17.94).

사용법:
    python recut_keypoints.py --kps runs/kps_upfall --manifest data/omnifall/manifest_upfall.csv \\
        --out runs/kps_upfall_fix --out-manifest data/omnifall/manifest_upfall_fix.csv
"""
from __future__ import annotations
import argparse, collections, csv, re
from pathlib import Path
import numpy as np

SRC = re.compile(r"_\d+_[\d.]+_[\d.]+_-?\d+_-?\d+_[A-Za-z][A-Za-z0-9_]*$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kps", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--header-fps", type=float, default=30.0, help="절단기가 쓴 (틀린) fps")
    ap.add_argument("--true-fps", type=float, help="실제 fps. 미지정 시 추정")
    ap.add_argument("--min-cover", type=float, default=0.9, help="재절단 구간 중 프레임이 있어야 하는 비율")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    ds = rows[0]["dataset"]
    src = collections.defaultdict(list)
    for r in rows:
        stem = Path(r["clip"]).stem
        f = args.kps / f"{ds}__{stem}.npy"
        arr = np.load(f) if f.exists() else None
        src[SRC.sub("", stem)].append((float(r["start"]), float(r["end"]), arr, r))

    # 실제 fps 추정 — 틈 없이 0부터 이어지는 원본만
    est = []
    for v in src.values():
        v.sort(key=lambda x: x[0])
        contiguous = v[0][0] == 0 and all(abs(b[0] - a[1]) < 1e-6 for a, b in zip(v, v[1:]))
        tot = sum(a.shape[0] for _, _, a, _ in v if a is not None)
        if contiguous and tot:
            est.append(tot / v[-1][1])
    fps = args.true_fps or float(np.median(est))
    print(f"[*] 원본 {len(src)}개 · 실제 fps {'지정' if args.true_fps else '추정'} {fps:.2f} "
          f"(틈 없는 원본 {len(est)}개 · 5~95% {np.percentile(est,5):.2f}~{np.percentile(est,95):.2f})")

    args.out.mkdir(parents=True, exist_ok=True)
    kept, dropped, cover_all = [], 0, []
    for key, v in src.items():
        # 원본 프레임 축에 클립을 놓는다
        n = int(np.ceil(max(e for _, e, _, _ in v) * args.header_fps)) + 2
        tl = np.full((n, 17, 3), np.nan, np.float32)
        for s, e, a, _ in v:
            if a is None:
                continue
            i0 = int(round(s * args.header_fps))
            tl[i0:i0 + a.shape[0]] = a[:max(0, n - i0)]
        for s, e, a, r in v:
            j0, j1 = int(round(s * fps)), int(round(e * fps))
            if j1 - j0 < 2:
                dropped += 1
                continue
            seg = tl[j0:j1]
            ok = ~np.isnan(seg[:, 0, 0])
            cover = ok.mean()
            cover_all.append(cover)
            if cover < args.min_cover:
                dropped += 1
                continue
            seg = np.nan_to_num(seg[ok], nan=0.0)        # 틈 프레임은 버린다 (≤10%)
            np.save(args.out / f"{ds}__{Path(r['clip']).stem}.npy", seg)
            kept.append(r)

    cover_all = np.array(cover_all)
    print(f"[*] 저장 {len(kept)} · 제외 {dropped} · coverage 중앙값 {np.median(cover_all):.3f} "
          f"· <0.9 {int((cover_all < 0.9).sum())}개")
    with args.out_manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(kept)
    print(f"→ {args.out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
