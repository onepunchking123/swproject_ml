#!/usr/bin/env python3
"""OmniFall Zenodo 배포판을 Colab VM 을 거쳐 Google Drive 로 옮긴다.

실행 위치는 Colab VM 이다. Drive 가 마운트된 세션에서:

    colab upload -s omnifall configs/datasets.yaml /content/datasets.yaml
    colab exec   -s omnifall -f scripts/fetch_omnifall.py

`colab run` 은 새 VM 을 만들어 Drive 마운트가 없으므로 쓰지 않는다.
`colab exec` 는 argv 를 전달하지 않으므로 설정은 아래 상수로 둔다.

zip 하나씩 처리한다:
    다운로드(중단 시 Range 로 재개) → MD5 검증 → Drive 복사 → 크기 대조
    → `<이름>.md5` 사이드카 기록 → 로컬 삭제

Drive 에 같은 크기의 파일과 사이드카가 이미 있으면 건너뛰므로 재실행이 안전하다.
해제와 중복 제거는 여기서 하지 않는다 — Drive FUSE 위에서 해제하지 않고,
학습 세션에서 Drive→/content 로 zip 을 복사한 뒤 한다 (configs/datasets.yaml transfer 참고).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

# ── 설정 ────────────────────────────────────────────────────────────────
REGISTRY = Path("/content/datasets.yaml")
DRIVE_ROOT = Path("/content/drive/MyDrive/falldata/omnifall")
LOCAL_TMP = Path("/content/dl")

ONLY: list[str] = []      # 예: ["Cauca_fall.zip"]. 비어 있으면 전부 — 작은 것부터
MAX_GB: float | None = None   # 이번 실행의 누적 용량 상한. 예: 7 → 작은 5개까지만
RETRIES = 3
CHUNK = 8 << 20           # 8MB
# ────────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def gb(n: int | float) -> str:
    return f"{n / 1e9:.2f}GB"


def load_files() -> list[dict]:
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    files = reg["sources"]["omnifall_zenodo"]["files"]
    files = sorted(files, key=lambda f: f["size_gb"])
    if ONLY:
        files = [f for f in files if f["name"] in ONLY]
        missing = set(ONLY) - {f["name"] for f in files}
        if missing:
            sys.exit(f"레지스트리에 없는 이름: {sorted(missing)}")
    return files


def check_drive() -> None:
    if not Path("/content/drive/MyDrive").is_dir():
        sys.exit("Drive 가 마운트되어 있지 않다. 사용자가 `colab drivemount -s <세션>` 을 먼저 실행해야 한다.")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    local_free = shutil.disk_usage("/content").free
    log(f"/content 여유 {gb(local_free)} · Drive 대상 {DRIVE_ROOT}")
    # FUSE 위의 disk_usage 는 신뢰할 수 없어 참고만 한다
    try:
        log(f"Drive 여유(참고) {gb(shutil.disk_usage(DRIVE_ROOT).free)}")
    except OSError:
        pass


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path, expected: int) -> None:
    """이어받기를 지원하는 다운로드. 서버가 Range 를 무시하면 처음부터 받는다."""
    for attempt in range(1, RETRIES + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if have >= expected:
            return
        req = urllib.request.Request(url, headers={"User-Agent": "fall-detection/fetch"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    log("  서버가 Range 를 지원하지 않아 처음부터 받는다")
                    have = 0
                mode = "ab" if have else "wb"
                t0, last, done = time.time(), time.time(), have
                with dst.open(mode) as f:
                    while chunk := r.read(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 15:
                            speed = (done - have) / max(time.time() - t0, 1e-6)
                            log(f"  {gb(done)}/{gb(expected)}  {speed / 1e6:.0f}MB/s")
                            last = time.time()
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log(f"  다운로드 실패 ({attempt}/{RETRIES}): {e}")
            time.sleep(10 * attempt)
    sys.exit(f"다운로드 포기: {dst.name}")


def already_on_drive(name: str, expected: int, md5: str) -> bool:
    target, sidecar = DRIVE_ROOT / name, DRIVE_ROOT / f"{name}.md5"
    if not (target.exists() and sidecar.exists()):
        return False
    if target.stat().st_size != expected:
        log(f"  Drive 에 있으나 크기 불일치 → 다시 받는다 ({gb(target.stat().st_size)} ≠ {gb(expected)})")
        return False
    if sidecar.read_text().strip() != md5:
        log("  Drive 사이드카 MD5 불일치 → 다시 받는다")
        return False
    return True


def transfer(f: dict) -> str:
    name, md5, url = f["name"], f["md5"], f["url"]
    expected = int(f["size_bytes"])  # Zenodo API 의 정확한 바이트 수. size_gb 는 표시용 근사값이라 비교에 쓰지 않는다
    log(f"▶ {name}  {gb(expected)}")

    if already_on_drive(name, expected, md5):
        log("  이미 Drive 에 있음 — 건너뜀")
        return "skip"

    local = LOCAL_TMP / name
    download(url, local, expected)
    actual = local.stat().st_size

    log(f"  MD5 검증 중 ({gb(actual)})")
    got = md5_of(local)
    if got != md5:
        local.unlink(missing_ok=True)
        log(f"  MD5 불일치! 기대 {md5} / 실제 {got} — 로컬 삭제")
        return "md5_fail"

    log("  Drive 로 복사 중")
    target = DRIVE_ROOT / name
    shutil.copyfile(local, target)
    if target.stat().st_size != actual:
        log(f"  Drive 복사 크기 불일치 ({gb(target.stat().st_size)} ≠ {gb(actual)}) — 로컬 보존")
        return "copy_fail"
    (DRIVE_ROOT / f"{name}.md5").write_text(got + "\n")
    local.unlink()
    log(f"  완료 → {target}")
    return "ok"


def main() -> int:
    check_drive()
    files = load_files()
    plan_total = sum(f["size_gb"] for f in files)
    log(f"대상 {len(files)}개, 합계 {plan_total:.2f}GB" + (f", 이번 실행 상한 {MAX_GB}GB" if MAX_GB else ""))

    results, done_gb = {}, 0.0
    for f in files:
        if MAX_GB is not None and done_gb + f["size_gb"] > MAX_GB:
            log(f"■ {f['name']} 은 상한 초과로 이번 실행에서 제외")
            results[f["name"]] = "deferred"
            continue
        results[f["name"]] = transfer(f)
        if results[f["name"]] in ("ok", "skip"):
            done_gb += f["size_gb"]

    print("\n" + "=" * 60)
    for name, r in results.items():
        print(f"  {r:9s} {name}")
    print("=" * 60)
    log(f"Drive 확보 {done_gb:.2f}GB / 대상 {plan_total:.2f}GB")
    return 0 if all(r in ("ok", "skip", "deferred") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
