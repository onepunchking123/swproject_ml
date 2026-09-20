#!/usr/bin/env python3
"""공개 데이터셋(OF-Syn, URFD)을 Colab VM 을 거쳐 Google Drive 로 옮긴다.

fetch_omnifall.py 와 같은 방식이다 — Drive 가 마운트된 세션에서:

    colab upload -s omnifall configs/datasets.yaml /content/datasets.yaml
    colab exec   -s omnifall --timeout 21600 -f scripts/fetch_public.py

다른 점은 검증 방식이다. Zenodo 는 MD5 를 줬지만,
  - OF-Syn (Hugging Face) 은 LFS oid = **sha256** 을 준다 (x-linked-etag)
  - URFD 는 체크섬이 없다 → Content-Length 와 크기 대조만 한다
두 서버 모두 Range 를 지원하므로 aria2c 로 받는다 (URFD 는 파일 110개를 동시 4개씩).

Drive 레이아웃:
    falldata/omnifall_syn/omnifall-synthetic_av1.tar (+ .sha256 사이드카)
    falldata/urfd/{fall,adl}-NN-camK-{rgb,d}.zip (+ manifest.json: 파일별 크기)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

REGISTRY = Path("/content/datasets.yaml")
DRIVE = Path("/content/drive/MyDrive")
LOCAL_TMP = Path("/content/dl")
DO = {"omnifall_syn": True, "urfd": True}
URFD_DEPTH = True            # cam0 depth zip 도 받는다 (2.3GB)
ARIA_CONN = 16               # 큰 파일 하나에 쓰는 연결 수
ARIA_JOBS = 4                # URFD 처럼 파일이 많을 때 동시 파일 수


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def gb(n: float) -> str:
    return f"{n / 1e9:.2f}GB"


def head_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "fall-detection/fetch"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except Exception:
        return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def aria(urls: list[tuple[str, str]], dest: Path, conn: int, jobs: int) -> bool:
    """(url, 파일명) 목록을 aria2c 입력 파일로 만들어 한 번에 받는다. 출력은 실시간으로 흘린다."""
    dest.mkdir(parents=True, exist_ok=True)
    lst = dest / "_aria_input.txt"
    lst.write_text("".join(f"{u}\n  out={n}\n" for u, n in urls))
    cmd = ["aria2c", "-i", str(lst), "-d", str(dest), "-j", str(jobs), "-x", str(conn), "-s", str(conn),
           "-k", "8M", "-c", "--file-allocation=none", "--auto-file-renaming=false", "--allow-overwrite=true",
           "--console-log-level=warn", "--summary-interval=20", "--max-tries=5", "--retry-wait=10"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        s = line.strip()
        if "DL:" in s or "ERR" in s or "error" in s.lower():
            log("  " + s[:120])
    rc = proc.wait()
    lst.unlink(missing_ok=True)
    return rc == 0


def to_drive(local: Path, remote_dir: Path) -> bool:
    remote_dir.mkdir(parents=True, exist_ok=True)
    target = remote_dir / local.name
    if target.exists() and target.stat().st_size == local.stat().st_size:
        return True
    shutil.copyfile(local, target)
    return target.stat().st_size == local.stat().st_size


def do_omnifall_syn(reg: dict) -> str:
    src = reg["sources"]["omnifall_syn"]
    url, size, sha = src["url"], int(src["size_bytes"]), src["sha256"]
    name = Path(src["file"]).name
    remote_dir = DRIVE / src["drive_dest"]
    log(f"▶ OF-Syn {name} {gb(size)}")
    side = remote_dir / f"{name}.sha256"
    if (remote_dir / name).exists() and side.exists() and side.read_text().strip() == sha \
            and (remote_dir / name).stat().st_size == size:
        log("  이미 Drive 에 있음 — 건너뜀"); return "skip"
    local = LOCAL_TMP / name
    t0 = time.time()
    if not aria([(url, name)], LOCAL_TMP, ARIA_CONN, 1):
        log("  aria2c 실패"); return "dl_fail"
    log(f"  다운로드 {gb(local.stat().st_size)} · 평균 {local.stat().st_size / max(time.time() - t0, 1e-6) / 1e6:.1f}MB/s")
    if local.stat().st_size != size:
        log(f"  크기 불일치 {local.stat().st_size} ≠ {size}"); return "size_fail"
    log("  sha256 검증 중"); got = sha256_of(local)
    if got != sha:
        log(f"  sha256 불일치! {got}"); local.unlink(); return "hash_fail"
    log("  Drive 복사 중")
    if not to_drive(local, remote_dir):
        return "copy_fail"
    side.write_text(sha + "\n"); local.unlink()
    log(f"  완료 → {remote_dir / name}"); return "ok"


def urfd_files(reg: dict) -> list[tuple[str, str]]:
    src = reg["sources"]["urfd"]
    pat = src["url_pattern"]
    out = []
    for nn in range(1, 31):
        for cam in ("cam0", "cam1"):
            out.append((pat.format(type="fall", nn=nn, cam=cam, modality="rgb"), f"fall-{nn:02d}-{cam}-rgb.zip"))
        if URFD_DEPTH:
            out.append((pat.format(type="fall", nn=nn, cam="cam0", modality="d"), f"fall-{nn:02d}-cam0-d.zip"))
    for nn in range(1, 41):
        out.append((pat.format(type="adl", nn=nn, cam="cam0", modality="rgb"), f"adl-{nn:02d}-cam0-rgb.zip"))
        if URFD_DEPTH:
            out.append((pat.format(type="adl", nn=nn, cam="cam0", modality="d"), f"adl-{nn:02d}-cam0-d.zip"))
    # 가속도·동기화 CSV 는 작으니 같이
    for nn in range(1, 31):
        base = pat.rsplit("/", 1)[0]
        out.append((f"{base}/fall-{nn:02d}-acc.csv", f"fall-{nn:02d}-acc.csv"))
        out.append((f"{base}/fall-{nn:02d}-data.csv", f"fall-{nn:02d}-data.csv"))
    for nn in range(1, 41):
        base = pat.rsplit("/", 1)[0]
        out.append((f"{base}/adl-{nn:02d}-acc.csv", f"adl-{nn:02d}-acc.csv"))
        out.append((f"{base}/adl-{nn:02d}-data.csv", f"adl-{nn:02d}-data.csv"))
    return out


def do_urfd(reg: dict) -> str:
    src = reg["sources"]["urfd"]
    remote_dir = DRIVE / src["drive_dest"]
    files = urfd_files(reg)
    log(f"▶ URFD {len(files)}개 파일 (zip {sum(1 for _, n in files if n.endswith('.zip'))} + csv)")
    manifest_path = remote_dir / "manifest.json"
    have = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    todo = [(u, n) for u, n in files if not ((remote_dir / n).exists() and have.get(n) == (remote_dir / n).stat().st_size)]
    log(f"  이미 Drive 에 {len(files) - len(todo)}개, 받을 것 {len(todo)}개")
    if not todo:
        return "skip"
    local = LOCAL_TMP / "urfd"
    t0 = time.time()
    ok = aria(todo, local, 4, ARIA_JOBS)
    got = {n: (local / n).stat().st_size for _, n in todo if (local / n).exists()}
    log(f"  다운로드 {len(got)}/{len(todo)} · {gb(sum(got.values()))} · {sum(got.values()) / max(time.time() - t0, 1e-6) / 1e6:.1f}MB/s")
    # 크기 검증: 서버 Content-Length 와 대조 (체크섬이 없다)
    bad = []
    for u, n in todo:
        if n not in got:
            bad.append(n); continue
        exp = head_size(u)
        if exp is not None and exp != got[n]:
            bad.append(n); log(f"  크기 불일치 {n}: {got[n]} ≠ {exp}")
    log("  Drive 복사 중")
    for n, sz in got.items():
        if n in bad:
            continue
        if to_drive(local / n, remote_dir):
            have[n] = sz; (local / n).unlink()
    manifest_path.write_text(json.dumps(have, indent=1, sort_keys=True))
    log(f"  완료 {len(have)}/{len(files)} → {remote_dir}" + (f" · 실패 {len(bad)}: {bad[:5]}" if bad else ""))
    return "ok" if ok and not bad else "partial"


def main() -> int:
    if not (DRIVE / "falldata").is_dir():
        sys.exit("Drive 가 마운트되어 있지 않다 (colab drivemount 필요)")
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    log(f"/content 여유 {gb(shutil.disk_usage('/content').free)}")
    results = {}
    if DO["omnifall_syn"]:
        results["omnifall_syn"] = do_omnifall_syn(reg)
    if DO["urfd"]:
        results["urfd"] = do_urfd(reg)
    print("\n" + "=" * 50)
    for k, v in results.items():
        print(f"  {v:10s} {k}")
    return 0 if all(v in ("ok", "skip") for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
