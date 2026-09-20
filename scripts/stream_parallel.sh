#!/usr/bin/env bash
# AI Hub → Drive 병렬 스트리밍
#
# aihub_stream.py 는 filekey 를 순차 처리한다. AI Hub 는 연결당 속도를 제한하므로
# (실측 2.2MB/s), filekey 가 여럿이면 프로세스를 나눠 동시에 받는 편이 빠르다.
# 동시 2스트림 실측: 각 2.3 / 1.7MB/s — 합산 4.0MB/s 로 단일 대비 1.8배.
#
# 단, 한 filekey 는 쪼갤 수 없다 (Range·이어받기 불가). VS.zip 처럼 단일 파일은
# 병렬화해도 이득이 없고, TS 5개 파트처럼 filekey 가 여럿일 때만 유효하다.
#
# 이 회선의 원시 대역폭이 3.3MB/s 로 측정됐으므로 동시 실행은 2~3개가 상한이다.
# 그 이상은 서로 대역폭을 나눠 가질 뿐 총량이 늘지 않는다.
#
# 사용법:
#   bash scripts/stream_parallel.sh 531136 531138          # 라벨 2개 동시
#   bash scripts/stream_parallel.sh 531131 531132 531133   # TS 파트 3개 동시
#
# 로그는 runs/stream/<filekey>.log 에 쌓인다. 진행 상황:
#   tail -f runs/stream/531136.log

set -uo pipefail
cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
    echo "사용법: bash scripts/stream_parallel.sh <filekey> [filekey...]"
    echo
    echo "  라벨    531136 (TL 126MB)  531138 (VL 16MB)"
    echo "  검증    531137 (VS 55GB)"
    echo "  학습    531131~531135 (TS 436GB, 분할 zip 한 덩어리)"
    exit 1
fi

# rclone 인증 확인 — 없으면 몇 시간 받고 나서 업로드가 전부 실패한다
if ! rclone lsd gdrive: >/dev/null 2>&1; then
    echo "[!] rclone gdrive 인증이 없습니다. 먼저 실행하세요:"
    echo
    echo "    rclone config reconnect gdrive:"
    echo
    echo "  브라우저에서 Colab 과 같은 계정으로 로그인해야 합니다."
    exit 1
fi

mkdir -p runs/stream
PY=./venv_aihub/bin/python
[ -x "$PY" ] || PY=python3

pids=()
for fk in "$@"; do
    log="runs/stream/${fk}.log"
    echo "[*] filekey $fk 시작 → $log"
    # caffeinate: 긴 작업 중 맥북이 잠들면 연결이 끊긴다
    caffeinate -i "$PY" scripts/aihub_stream.py --filekey "$fk" > "$log" 2>&1 &
    pids+=($!)
    sleep 2   # 동시 시작 시 서버가 거절하는 경우가 있어 간격을 둔다
done

echo
echo "[*] ${#pids[@]}개 스트림 실행 중 (PID: ${pids[*]})"
echo "[*] 진행 확인:  tail -f runs/stream/*.log"
echo "[*] 중단:       kill ${pids[*]}"
echo

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[✓] filekey $(echo "$@" | cut -d' ' -f$((i+1))) 완료"
    else
        echo "[✗] filekey $(echo "$@" | cut -d' ' -f$((i+1))) 실패 — 로그 확인"
        fail=1
    fi
done

echo
echo "=== Drive 결과 ==="
rclone ls gdrive:falldata/aihub 2>/dev/null | tail -20
exit $fail
