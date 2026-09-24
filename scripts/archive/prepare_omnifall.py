#!/usr/bin/env python3
"""OmniFall Zenodo zip → 중복 제거된 클립 디렉토리 + 통합 manifest.

Drive 에 보관한 zip 을 학습 세션의 로컬 디스크로 풀 때 쓴다 (Drive FUSE 위에서 직접 풀지 않는다).

Zenodo 배포판은 같은 클립을 두 파일명 규칙으로 두 번 담고 있다 (Cauca 258쌍 CRC 동일 확인):

    짧은 형식: {Action}S{subj}_{label}_{start}_{end}_{subj}_{cam}_{dataset}.avi   ← 유지
    긴 형식:   Subject.{subj}_{action}_01_{start}_{end}_{label}_{cam}_{subj}_{dataset}.avi

긴 형식은 **추출하지 않는다** — 풀고 지우는 것보다 I/O 가 절반이다. 단, 짧은 형식 짝이
없거나 CRC 가 다른 긴 형식 멤버는 데이터 손실을 막기 위해 추출한다.

manifest 는 configs/datasets.yaml 의 omnifall_v1 스키마(path,label,start,end,subject,cam,dataset)에
`clip`(추출된 상대경로)·`source_zip` 을 더한 것이다. zip 안의 {dataset}.csv 와 행 수·라벨 분포를
대조해 불일치를 보고한다.

사용법:
    python prepare_omnifall.py --zips /content/drive/MyDrive/falldata/omnifall --out /content/omnifall --staging /content/zips
    python prepare_omnifall.py --zips <dir> --out <dir> --only Cauca_fall.zip
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import shutil
import sys
import zipfile
from pathlib import Path

SHORT = re.compile(r"^(?P<action>.+?)S(?P<subject>\d+)_(?P<label>\d+)_(?P<start>[\d.]+)_(?P<end>[\d.]+)_(?P=subject)_(?P<cam>\d+)_(?P<dataset>[A-Za-z0-9]+)\.(?P<ext>avi|mp4|mkv)$")
LONG = re.compile(r"^Subject\.(?P<subject>\d+)_(?P<action>.+?)_\d+_(?P<start>[\d.]+)_(?P<end>[\d.]+)_(?P<label>\d+)_(?P<cam>\d+)_(?P=subject)_(?P<dataset>[A-Za-z0-9]+)\.(?P<ext>avi|mp4|mkv)$")
VIDEO_EXT = (".avi", ".mp4", ".mkv")
MANIFEST_COLS = ["dataset", "clip", "path", "label", "start", "end", "subject", "cam", "source_zip"]


def key_of(m: re.Match) -> tuple:
    return (m["subject"], m["start"], m["end"], m["label"], m["cam"])


def plan_members(z: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], dict]:
    """추출할 멤버를 고른다. 짧은 형식은 전부, 긴 형식은 짝이 없거나 내용이 다를 때만."""
    shorts, longs, others = {}, {}, []
    for info in z.infolist():
        if info.is_dir() or not info.filename.lower().endswith(VIDEO_EXT):
            continue
        base = info.filename.rsplit("/", 1)[-1]
        if m := SHORT.match(base):
            shorts[key_of(m)] = info
        elif m := LONG.match(base):
            longs[key_of(m)] = info
        else:
            others.append(info)

    extract = list(shorts.values())
    dup = orphan = mismatch = 0
    for k, info in longs.items():
        twin = shorts.get(k)
        if twin is None:
            orphan += 1; extract.append(info)
        elif twin.CRC != info.CRC or twin.file_size != info.file_size:
            mismatch += 1; extract.append(info)
        else:
            dup += 1
    extract.extend(others)
    return extract, dict(short=len(shorts), long=len(longs), dup_skipped=dup,
                         orphan_long=orphan, crc_mismatch=mismatch, unpatterned=len(others))


def manifest_rows(extracted: list[zipfile.ZipInfo], zip_name: str) -> list[dict]:
    rows = []
    for info in extracted:
        base = info.filename.rsplit("/", 1)[-1]
        m = SHORT.match(base) or LONG.match(base)
        if not m:
            continue
        rows.append(dict(
            dataset=m["dataset"], clip=info.filename, path=Path(info.filename).with_suffix("").as_posix(),
            label=int(m["label"]), start=float(m["start"]), end=float(m["end"]),
            subject=int(m["subject"]), cam=int(m["cam"]), source_zip=zip_name,
        ))
    return rows


def read_inner_csv(z: zipfile.ZipFile) -> list[dict]:
    names = [n for n in z.namelist() if n.lower().endswith(".csv") and n.count("/") == 1]
    if not names:
        return []
    with z.open(names[0]) as f:
        return list(csv.DictReader(line.decode("utf-8") for line in f))


def compare_with_csv(rows: list[dict], inner: list[dict]) -> list[str]:
    notes = []
    if not inner:
        return ["zip 안에 라벨 CSV 가 없다"]
    if len(rows) != len(inner):
        notes.append(f"행 수 불일치: manifest {len(rows)} vs CSV {len(inner)}")
    dist_m = collections.Counter(r["label"] for r in rows)
    dist_c = collections.Counter(int(r["label"]) for r in inner)
    if dist_m != dist_c:
        notes.append(f"라벨 분포 불일치: manifest {dict(sorted(dist_m.items()))} vs CSV {dict(sorted(dist_c.items()))}")
    return notes


def stage_zip(src: Path, staging: Path | None) -> Path:
    """Drive(FUSE) 위의 zip 은 로컬로 먼저 복사한다."""
    if staging is None:
        return src
    dst = staging / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if dst.stat().st_size != src.stat().st_size:
        sys.exit(f"복사 크기 불일치: {src.name}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zips", type=Path, required=True, help="zip 이 있는 디렉토리 (Drive 경로 가능)")
    ap.add_argument("--out", type=Path, required=True, help="클립을 풀 로컬 루트")
    ap.add_argument("--only", nargs="*", default=[], help="처리할 zip 이름 (기본: 전부, 작은 것부터)")
    ap.add_argument("--staging", type=Path, help="zip 을 먼저 복사할 로컬 디렉토리 (Drive 에서 읽을 때 권장)")
    ap.add_argument("--keep-staged", action="store_true", help="처리 후 복사한 zip 을 지우지 않는다")
    args = ap.parse_args()

    zips = sorted(args.zips.glob("*.zip"), key=lambda p: p.stat().st_size)
    if args.only:
        zips = [p for p in zips if p.name in set(args.only)]
    if not zips:
        sys.exit("처리할 zip 이 없다")
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    problems = 0
    for src in zips:
        print(f"▶ {src.name}  {src.stat().st_size / 1e9:.2f}GB", flush=True)
        local = stage_zip(src, args.staging)
        with zipfile.ZipFile(local) as z:
            extract, st = plan_members(z)
            print(f"  짧은 {st['short']} / 긴 {st['long']} → 중복 제외 {st['dup_skipped']}"
                  f" · 짝없음 {st['orphan_long']} · CRC상이 {st['crc_mismatch']} · 패턴밖 {st['unpatterned']}", flush=True)
            for i, info in enumerate(extract, 1):
                z.extract(info, args.out)
                if i % 500 == 0:
                    print(f"  {i}/{len(extract)}", flush=True)
            for info in z.infolist():                       # 라벨 CSV·txt 는 원본 대조용으로 보존
                if info.filename.lower().endswith((".csv", ".txt")):
                    z.extract(info, args.out)
            rows = manifest_rows(extract, src.name)
            notes = compare_with_csv(rows, read_inner_csv(z))
        all_rows.extend(rows)
        top = args.out / Path(extract[0].filename).parts[0] if extract else None
        size = sum(f.stat().st_size for f in top.rglob("*") if f.is_file()) if top else 0
        print(f"  추출 {len(extract)}개 · {size / 1e9:.2f}GB · manifest {len(rows)}행"
              + (f" · ⚠ {'; '.join(notes)}" if notes else " · CSV 대조 일치"), flush=True)
        problems += bool(notes)
        if args.staging and not args.keep_staged and local != src:
            local.unlink()

    mpath = args.out / "manifest.csv"
    with mpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS); w.writeheader(); w.writerows(all_rows)
    print(f"\nmanifest: {mpath}  ({len(all_rows)}행)")
    print("라벨 분포:", dict(sorted(collections.Counter(r["label"] for r in all_rows).items())))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
