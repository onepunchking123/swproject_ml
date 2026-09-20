#!/usr/bin/env python3
"""AI Hub 파일을 맥북 디스크를 거치지 않고 Google Drive 로 스트리밍한다.

AI Hub 는 해외 IP 다운로드를 차단하므로 Colab 이 직접 받을 수 없고, 한국 IP 인 맥북이
중계해야 한다. 맥북 SSD 여유가 수 GB 뿐이라 내려받아 올리는 방식은 불가능하다.

aihubshell 이 실제로 보내는 요청은 하나다:

    GET https://api.aihub.or.kr/down/0.6/{datasetkey}.do?fileSn={filekey}   (헤더 apikey)
      → 302 https://stream.aihub.or.kr/shellStream.do?uuid=...  → 200 application/x-tar

응답은 **요청마다 서버가 즉석 생성하는 tar 스트림**이다 (Content-Length 없음, Range 불가,
2026-09-20 실측). 따라서 이어받기·병렬 분할은 불가능하고, 단일 연결(실측 2.2MB/s)로 처음부터
끝까지 한 번에 흘려야 한다. 대신 tar 안의 `<파일>.partN` 조각을 **멤버 단위로 잘라 각각
rclone 에 넘기므로**, 도중에 끊겨도 이미 올라간 조각은 남는다 (다운로드는 다시 해야 한다).

    curl (단일 스트림) → tarfile 스트림 파서 → 멤버마다 rclone rcat gdrive:<dest>/<파일>.partNNN
                                                 └ Drive 에 같은 크기로 이미 있으면 바이트만 버리고 건너뜀

마지막에 parts.json 을 남긴다 — Colab 에서 번호순으로 cat 하면 원본 zip 이 된다.

사용법 (맥북, 한국 IP. 긴 작업은 caffeinate 로 잠들지 않게):
    python scripts/aihub_stream.py --filekey 531138 --dry-run                 # 구조만 (업로드 없음)
    python scripts/aihub_stream.py --filekey 531136 531138                    # 라벨 142MB
    python scripts/aihub_stream.py --filekey 531136 531138 --buffer-dir runs/buf        # tar 전체 버퍼
    caffeinate -i python scripts/aihub_stream.py --filekey 531137 --member-buffer runs/buf  # VS.zip 55GB

버퍼 없이 직접 파이프로 넘기면 실패한다 (2026-09-20 실측). 업로드가 다운로드보다
느리면 파이프가 막히고 AI Hub 가 연결을 끊는다 — `curl (18) transfer closed`.
실측 속도차가 원인이다: 다운로드 4.5MB/s vs 업로드 0.4MB/s.

  --buffer-dir     tar 전체를 받은 뒤 업로드. tar 크기만큼 디스크가 필요하다.
  --member-buffer  조각(.partN) 하나씩 받아 올리고 즉시 삭제. 조각 하나(보통 1GB)만
                   있으면 되므로 디스크가 작을 때 쓴다. 55GB 를 19GB 디스크로 처리 가능.

API 키는 ~/.aihub_key 에서 읽는다. 출력에 키가 찍히지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

BASE = "https://api.aihub.or.kr/down/0.6/{dataset}.do?fileSn={filekey}"
PART_RE = re.compile(r"^(?P<base>.+)\.part(?P<n>\d+)$")
CHUNK = 8 << 20


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def api_key() -> str:
    p = Path.home() / ".aihub_key"
    if not p.exists():
        sys.exit("~/.aihub_key 가 없다")
    return p.read_text().strip()


def rclone_size(remote: str) -> int | None:
    r = subprocess.run(["rclone", "lsjson", remote], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    items = json.loads(r.stdout or "[]")
    return items[0]["Size"] if items else None


def remote_name(dest: str, leaf: str) -> tuple[str, str, int]:
    """tar 멤버 이름 → (원격 경로, 원본 파일명, 조각 번호). 조각 번호는 3자리로 정규화해 정렬을 보장한다."""
    m = PART_RE.match(leaf)
    if m:
        return f"{dest}/{m['base']}.part{int(m['n']):03d}", m["base"], int(m["n"])
    return f"{dest}/{leaf}", leaf, 0


def pump(src, size: int, sink) -> int:
    """src 에서 정확히 size 바이트를 읽어 sink 로 보낸다 (sink 가 None 이면 버린다). 진행을 로그한다."""
    done, t0, last = 0, time.time(), time.time()
    while done < size:
        buf = src.read(min(CHUNK, size - done))
        if not buf:
            raise EOFError(f"스트림이 {done:,}/{size:,} 바이트에서 끊겼다")
        if sink is not None:
            sink.write(buf)
        done += len(buf)
        if time.time() - last > 30:
            log(f"    {done / 1e9:.2f}/{size / 1e9:.2f}GB  {done / max(time.time() - t0, 1e-6) / 1e6:.1f}MB/s")
            last = time.time()
    return done


def stream_filekey(dataset: str, fk: str, dest: str, key: str, dry_run: bool,
                   buffer_dir: Path | None = None,
                   member_buffer: Path | None = None) -> bool:
    url = BASE.format(dataset=dataset, filekey=fk)
    dest_fk = f"{dest}/{fk}"
    log(f"▶ filekey {fk} → {dest_fk}" + ("  (dry-run: 업로드 없음)" if dry_run else ""))
    parts, ok, t_start = [], True, time.time()
    curl, tar_path = None, None
    if buffer_dir is not None:
        # 업로드가 다운로드보다 느린 회선에서는 파이프가 막혀 AI Hub 서버가 연결을 끊는다 (실측).
        # tar 를 디스크에 먼저 받아 두면 서버 속도로 받고, 업로드는 따로 진행된다. 디스크가 tar 크기만큼 필요하다.
        buffer_dir.mkdir(parents=True, exist_ok=True)
        tar_path = buffer_dir / f"{fk}.tar"
        for attempt in range(1, 4):
            r = subprocess.run(["curl", "-sSL", "-H", f"apikey:{key}", "-o", str(tar_path), url],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
            log(f"  버퍼 다운로드 실패 ({attempt}/3): {r.stderr.strip()[:160]}"); time.sleep(10 * attempt)
        else:
            return False
        log(f"  버퍼 {tar_path.stat().st_size / 1e6:.0f}MB 확보 ({(time.time() - t_start) / 60:.1f}분) → 업로드 시작")
        src = tar_path.open("rb")
    else:
        curl = subprocess.Popen(["curl", "-sSL", "-H", f"apikey:{key}", url],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert curl.stdout is not None
        src = curl.stdout
    try:
        with tarfile.open(fileobj=src, mode="r|") as tar:
            for ti in tar:
                if not ti.isreg():
                    continue
                leaf = ti.name.rsplit("/", 1)[-1]
                remote, base, idx = remote_name(dest_fk, leaf)
                f = tar.extractfile(ti)
                assert f is not None
                entry = dict(name=ti.name, base=base, index=idx, size=ti.size, remote=remote)
                if dry_run:
                    log(f"  {leaf}  {ti.size:,} B"); pump(f, ti.size, None)
                elif rclone_size(remote) == ti.size:
                    log(f"  건너뜀 (Drive 에 있음) {leaf}  {ti.size:,} B"); pump(f, ti.size, None)
                elif member_buffer is not None:
                    # 조각 하나를 디스크에 받아 업로드하고 즉시 지운다. 버퍼가 tar 전체가 아니라
                    # 조각 하나 크기(보통 1GB)만 필요하므로, 디스크가 작아도 55GB 를 처리할 수 있다.
                    # 파이프가 막혀 서버가 연결을 끊는 문제도 같이 해결된다.
                    t0 = time.time()
                    member_buffer.mkdir(parents=True, exist_ok=True)
                    tmp = member_buffer / f"{leaf}.tmp"
                    try:
                        with tmp.open("wb") as out:
                            pump(f, ti.size, out)
                        dl = time.time() - t0
                        t1 = time.time()
                        r = subprocess.run(["rclone", "copyto", str(tmp), remote],
                                           capture_output=True, text=True)
                        got = rclone_size(remote)
                        if r.returncode == 0 and got == ti.size:
                            log(f"  완료 {leaf}  {ti.size / 1e6:.0f}MB  "
                                f"↓{ti.size / max(dl, 1e-6) / 1e6:.1f} ↑{ti.size / max(time.time() - t1, 1e-6) / 1e6:.1f}MB/s")
                        else:
                            ok = False
                            log(f"  실패 {leaf}: rclone={r.returncode} size={got} {r.stderr.strip()[:160]}")
                            subprocess.run(["rclone", "deletefile", remote], capture_output=True)
                    finally:
                        tmp.unlink(missing_ok=True)
                else:
                    t0 = time.time()
                    rc = subprocess.Popen(["rclone", "rcat", "--size", str(ti.size), remote],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
                    assert rc.stdin is not None
                    try:
                        pump(f, ti.size, rc.stdin)
                    except Exception:
                        rc.kill(); rc.wait()
                        raise
                    # communicate() 가 stdin 을 닫아 EOF 를 보내고 stderr 를 회수한다 (먼저 닫으면 flush 오류)
                    _, err = rc.communicate()
                    got = rclone_size(remote)
                    if rc.returncode == 0 and got == ti.size:
                        log(f"  완료 {leaf}  {ti.size / 1e6:.0f}MB  {ti.size / max(time.time() - t0, 1e-6) / 1e6:.1f}MB/s")
                    else:
                        ok = False
                        log(f"  실패 {leaf}: rclone={rc.returncode} size={got} {err.decode(errors='replace').strip()[:160]}")
                        subprocess.run(["rclone", "deletefile", remote], capture_output=True)
                parts.append(entry)
    except (EOFError, tarfile.ReadError, OSError) as e:
        ok = False
        log(f"  스트림 오류: {e}")
    finally:
        if curl is not None:
            curl.wait()
            if curl.returncode not in (0, None):
                ok = False
                log(f"  curl 종료 코드 {curl.returncode}: {curl.stderr.read().decode(errors='replace').strip()[:200] if curl.stderr else ''}")
        if tar_path is not None:
            src.close(); tar_path.unlink(missing_ok=True)

    parts.sort(key=lambda p: (p["base"], p["index"]))
    total = sum(p["size"] for p in parts)
    log(f"  멤버 {len(parts)}개 · 합계 {total / 1e9:.3f}GB · {(time.time() - t_start) / 60:.1f}분 · {'성공' if ok else '실패 있음'}")
    if not dry_run and parts:
        manifest = dict(filekey=fk, dataset=dataset, total_bytes=total, complete=ok, parts=parts)
        subprocess.run(["rclone", "rcat", f"{dest_fk}/parts.json"],
                       input=json.dumps(manifest, ensure_ascii=False, indent=1).encode(), capture_output=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="71641")
    ap.add_argument("--filekey", nargs="+", required=True)
    ap.add_argument("--dest", default="gdrive:falldata/aihub", help="rclone 원격 경로")
    ap.add_argument("--dry-run", action="store_true", help="스트림을 읽어 구조만 출력하고 업로드하지 않는다")
    ap.add_argument("--buffer-dir", type=Path, help="tar 전체를 이 디렉토리에 먼저 받은 뒤 업로드 (tar 크기만큼 디스크 필요)")
    ap.add_argument("--member-buffer", type=Path,
                    help="조각(.partN) 하나씩 디스크에 받아 업로드하고 즉시 삭제 "
                         "(조각 하나 크기만 필요 — 보통 1GB. 디스크가 작을 때 권장)")
    args = ap.parse_args()
    if args.buffer_dir and args.member_buffer:
        sys.exit("--buffer-dir 와 --member-buffer 는 함께 쓸 수 없다")
    key = api_key()
    ok = True
    for fk in args.filekey:
        ok &= stream_filekey(args.dataset, fk, args.dest, key, args.dry_run,
                             args.buffer_dir, args.member_buffer)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
