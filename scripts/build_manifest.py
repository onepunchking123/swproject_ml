#!/usr/bin/env python3
"""OmniFall staged + synthetic → 통합 manifest.

두 소스의 파일명 규칙이 다르다.

  staged    {접두}_{label}_{start}_{end}_{subject}_{cam}_{dataset}.avi
            → 뒤 6개 필드를 뒤에서 파싱. 같은 구간이 두 이름으로 중복 수록돼 있어
              짧은 쪽만 남긴다.

  synthetic {class}/{class}_{2글자}_{번호}.mp4
            → 디렉토리가 곧 클래스. subject/cam 은 없으므로 -1.
              합성 영상은 피험자 개념이 없으니 split 에서 별도 그룹으로 다룬다.

사용법:
    python build_manifest.py --staged <unz디렉토리> --syn <syn디렉토리> --out manifest.csv
"""
from __future__ import annotations
import argparse, csv, collections, re
from pathlib import Path

TAIL = re.compile(r"_(?P<label>\d+)_(?P<start>[\d.]+)_(?P<end>[\d.]+)_"
                  r"(?P<subject>-?\d+)_(?P<cam>-?\d+)_(?P<dataset>[A-Za-z][A-Za-z0-9_]*)\.avi$")

LABEL2ID = {"walk":0, "fall":1, "fallen":2, "sit_down":3, "sitting":4, "lie_down":5,
            "lying":6, "stand_up":7, "standing":8, "other":9, "kneel_down":10,
            "kneeling":11, "squat_down":12, "squatting":13, "crawl":14, "jump":15}


def staged_rows(root: Path) -> list[dict]:
    seen = {}
    for p in sorted(root.rglob("*.avi")):
        m = TAIL.search(p.name)
        if not m:
            continue
        d = m.groupdict()
        if d["dataset"] == "OOPS":       # Zenodo 배포판에 영상이 없다 (12KB 껍데기)
            continue
        key = (d["dataset"], d["subject"], d["cam"], d["start"], d["end"], d["label"])
        # 같은 구간이 짧은/긴 두 이름으로 들어 있는 중복만 걸러낸다 (CRC 동일 → 크기 동일).
        # UP_Fall 처럼 Activity/Trial 이 다른데 시간 구간만 같은 클립은 크기가 달라 둘 다 남는다.
        if key in seen and seen[key].stat().st_size != p.stat().st_size:
            key = key + (p.stem,)
        if key not in seen or len(p.name) < len(seen[key].name):
            seen[key] = p
    out = []
    for p in seen.values():
        d = TAIL.search(p.name).groupdict()
        out.append(dict(clip=str(p.relative_to(root)), label=int(d["label"]),
                        start=float(d["start"]), end=float(d["end"]),
                        subject=int(d["subject"]), cam=int(d["cam"]), dataset=d["dataset"]))
    return out


def syn_rows(root: Path) -> list[dict]:
    """합성은 클래스 디렉토리 구조다. 구간이 아니라 클립 전체가 하나의 행동이다."""
    out = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lab = LABEL2ID.get(cls_dir.name)
        if lab is None:
            print(f"[!] 미지의 클래스 무시: {cls_dir.name}")
            continue
        for i, v in enumerate(sorted(cls_dir.glob("*.mp4"))):
            # 전체 클립 = 하나의 행동. 논문 기준 5초 클립이라 end=5.0 으로 둬야 로더가 fps 를
            # 역산할 수 있다 (16fps — 다른 셋과 다르다). 피험자 개념이 없으므로 10개 가상 그룹에
            # 순환 배정해 k-fold 에서 합성 전체가 한 fold 로 몰리지 않게 한다.
            out.append(dict(clip=str(v.relative_to(root)), label=lab,
                            start=0.0, end=5.0,
                            subject=900 + i % 10, cam=-1, dataset="synthetic"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staged", type=Path)
    ap.add_argument("--syn", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    if args.staged and args.staged.exists():
        r = staged_rows(args.staged)
        print(f"[*] staged {len(r):,}행")
        rows += r
    if args.syn and args.syn.exists():
        r = syn_rows(args.syn)
        print(f"[*] synthetic {len(r):,}행")
        rows += r
    if not rows:
        print("[!] 행이 없다"); return 1

    rows.sort(key=lambda r: (r["dataset"], r["subject"], r["clip"]))
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    ID2 = {v: k for k, v in LABEL2ID.items()}
    print(f"\nmanifest {len(rows):,}행 → {args.out}")
    print("데이터셋:", dict(collections.Counter(r["dataset"] for r in rows)))
    print("\n라벨 분포:")
    for l, n in sorted(collections.Counter(r["label"] for r in rows).items()):
        print(f"  {l:2d} {ID2.get(l,'?'):10s} {n:6,}  {n/len(rows)*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
