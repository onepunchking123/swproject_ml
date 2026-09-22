#!/usr/bin/env bash
# 데이터셋 하나만 처리한다 — Colab 세션 수명이 짧아 26GB 를 한 번에 못 돌린다.
#
# 실측: 7번의 세션 회수 중 3번이 대량 해제 도중이었다. 세션당 하나씩 끊어
# 처리하고 결과를 Drive 에 즉시 저장하면, 죽어도 그 데이터셋만 다시 하면 된다.
#
# 사용법 (Colab VM 안에서):
#   bash expand_one.sh OCCU        # staged zip 하나
#   bash expand_one.sh syn lying   # 합성 클래스 하나
set -uo pipefail
D=/content/drive/MyDrive/falldata
W=/content/work
mkdir -p "$W" "$D/kps2"

MODE=${1:?"사용법: expand_one.sh <데이터셋名|syn> [클래스]"}

if [ "$MODE" = "syn" ]; then
  CLS=${2:?"합성 클래스명 필요"}
  echo "[*] 합성 $CLS — AV1→H264 변환 후 추출"
  mkdir -p /content/syn_raw /content/syn_h264/$CLS
  tar -xf "$D/omnifall_syn/omnifall-synthetic_av1.tar" -C /content/syn_raw \
      --wildcards "./$CLS/*" 2>/dev/null
  n=0
  for f in /content/syn_raw/$CLS/*.mp4; do
    [ -f "$f" ] || continue
    o=/content/syn_h264/$CLS/$(basename "$f")
    [ -f "$o" ] || ffmpeg -y -v error -i "$f" -c:v libx264 -preset ultrafast -crf 28 -an "$o" 2>/dev/null
    n=$((n+1))
  done
  echo "[*] 변환 $n개"
  ROOT=/content/syn_h264
  python3 "$W/build_manifest.py" --syn "$ROOT" --out /content/mf_$CLS.csv
  cp /content/mf_$CLS.csv "$D/mf_syn_$CLS.csv"
else
  echo "[*] staged $MODE 해제"
  mkdir -p "$W/unz/$MODE"
  cp "$D/omnifall/$MODE.zip" /content/_t.zip
  unzip -qq -o /content/_t.zip -d "$W/unz/$MODE"
  rm -f /content/_t.zip
  echo "[*] 클립 $(find "$W/unz/$MODE" -name '*.avi'|wc -l)개"
  ROOT=$W/unz
  python3 "$W/build_manifest.py" --staged "$ROOT" --out /content/mf_$MODE.csv
  cp /content/mf_$MODE.csv "$D/mf_$MODE.csv"
fi

pip install -q ultralytics 2>&1 | tail -1
# 키포인트는 Drive 에 직접 쓴다 — 세션이 죽어도 남는다
python3 "$W/extract_keypoints.py" --manifest /content/mf_*.csv --root "$ROOT" \
    --out "$D/kps2" --model yolo11n-pose.pt --device 0
echo "[*] 완료 — Drive kps2: $(ls "$D/kps2"|wc -l)개"
