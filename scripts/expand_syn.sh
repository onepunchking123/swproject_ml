#!/bin/zsh
# OF-Synthetic (AV1 tar) 에서 부족 클래스만 꺼내 H.264 로 변환하고 키포인트를 뽑는다 — 로컬(M3).
#   bash scripts/expand_syn.sh lying lie_down fallen
# OpenCV 는 AV1 을 못 읽으므로 ffmpeg(libdav1d) 로 변환한다. 변환본은 320 가로로 줄인다 —
# YOLO 입력이 640 이라 1280×720 원본은 낭비이고, 디코드·추출이 3배 빨라진다.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./venv_aihub/bin/python
TAR=data/syn/omnifall-synthetic_av1.tar
RAW=data/syn/raw
H264=data/syn/h264
[ -f "$TAR" ] || { echo "없음: $TAR"; exit 1; }
[ $# -ge 1 ] || { echo "클래스명 필요"; exit 1; }

echo "[*] tar 멤버 접두 확인"
PREFIX=$(tar -tf "$TAR" | head -1 | cut -d/ -f1)
echo "    접두: $PREFIX"
mkdir -p "$RAW" "$H264"
for CLS in "$@"; do
  echo "[*] $CLS 해제"
  tar -xf "$TAR" -C "$RAW" "$PREFIX/$CLS" 2>/dev/null || tar -xf "$TAR" -C "$RAW" "./$CLS" 2>/dev/null || tar -xf "$TAR" -C "$RAW" "$CLS"
  SRC=$(find "$RAW" -type d -name "$CLS" | head -1)
  N=$(ls "$SRC"/*.mp4 | wc -l | tr -d ' ')
  mkdir -p "$H264/$CLS"
  echo "[*] $CLS 변환 $N개 (8 병렬)"
  ls "$SRC"/*.mp4 | xargs -P 8 -I{} sh -c 'o="'"$H264/$CLS"'/$(basename "{}")"; [ -f "$o" ] || ffmpeg -y -v error -i "{}" -vf scale=320:-2 -c:v libx264 -preset veryfast -crf 26 -an "$o"'
  echo "    변환 완료 $(ls "$H264/$CLS"/*.mp4 | wc -l | tr -d ' ')개 · $(du -sh "$H264/$CLS" | cut -f1)"
  rm -rf "$SRC"
done

echo "[*] manifest"
$PY scripts/build_manifest.py --syn "$H264" --out data/syn/manifest_syn.csv

echo "[*] 추출 (caffeinate)"
mkdir -p runs/kps_syn
caffeinate -i $PY scripts/extract_keypoints.py --manifest data/syn/manifest_syn.csv --root "$H264" \
    --out runs/kps_syn --device mps 2>&1 | grep -v "NMS time limit"
echo "DONE SYN"
