#!/bin/zsh
# 로컬(M3)에서 OmniFall staged 데이터셋 하나를 학습 가능한 키포인트로 만든다.
#   zip(Drive 에서 받아둔 것) → 해제(중복 긴 이름 제외) → zip 삭제 → manifest → YOLO 추출 → 영상 삭제
# 디스크가 좁아(≈16GB) 한 번에 하나씩, 단계마다 큰 파일을 지운다. 추출은 클립별 .npy 라 중단 후 재실행하면 이어간다.
#
#   bash scripts/expand_local.sh LE2I          # data/omnifall/LE2I.zip 이 있어야 한다
#
# 끝난 뒤 fps 역산 검사는 사람이 본다 (UP_Fall 처럼 헤더 fps 로 잘못 잘린 경우 recut_keypoints.py).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?"데이터셋 zip 이름 (LE2I · MCFD · OCCU)"}
PY=./venv_aihub/bin/python
ZIP=data/omnifall/$NAME.zip
UNZ=data/omnifall/unz_$NAME
[ -f "$ZIP" ] || { echo "없음: $ZIP"; exit 1; }

echo "[*] $NAME 목록 확인"
unzip -Z1 "$ZIP" | grep '\.avi$' > /tmp/_all_$NAME.txt
# 긴 형식(Subject.N_..._dataset.avi) 중복은 풀지 않는다 — 짧은 형식만 (CRC 동일 확인됨, datasets.yaml)
grep -v '/Subject\.[0-9]*_[a-z_]*_[0-9]*_' /tmp/_all_$NAME.txt > /tmp/_short_$NAME.txt || cp /tmp/_all_$NAME.txt /tmp/_short_$NAME.txt
echo "    avi $(wc -l < /tmp/_all_$NAME.txt | tr -d ' ')개 → 해제 $(wc -l < /tmp/_short_$NAME.txt | tr -d ' ')개"

echo "[*] 해제"
mkdir -p "$UNZ"
unzip -qq -o "$ZIP" -d "$UNZ" "$NAME/*.csv" "$NAME/*.txt" "$NAME/*.py" 2>/dev/null || true   # 상위 라벨·메모만 (OCCU 는 depth 원본 .bin 수십만 개가 들어 있다)
xargs -n 200 unzip -qq -o "$ZIP" -d "$UNZ" < /tmp/_short_$NAME.txt 2>/dev/null || true   # BSD xargs: -a 없음, stdin 으로
N_AVI=$(find "$UNZ" -name '*.avi' | wc -l | tr -d ' ')
echo "    $N_AVI개 · $(du -sh "$UNZ" | cut -f1) · 여유 $(df -h / | tail -1 | awk '{print $4}')"
# 해제가 기대치의 90% 미만이면 zip 을 지우지 않는다 — 한 번 지우면 9GB 를 다시 받아야 한다
EXPECT=$(wc -l < /tmp/_short_$NAME.txt | tr -d ' ')
if [ "$N_AVI" -lt $(( EXPECT * 9 / 10 )) ]; then echo "[!] 해제 부족 ($N_AVI/$EXPECT) — zip 보존, 중단"; exit 1; fi
rm -f "$ZIP"

echo "[*] manifest"
$PY scripts/build_manifest.py --staged "$UNZ" --out data/omnifall/manifest_$NAME.csv

echo "[*] 추출 (caffeinate)"
mkdir -p runs/kps_$NAME
caffeinate -i $PY scripts/extract_keypoints.py --manifest data/omnifall/manifest_$NAME.csv --root "$UNZ" \
    --out runs/kps_$NAME --device mps 2>&1 | grep -v "NMS time limit"

echo "[*] fps 역산 (T / (end-start))"
$PY - "$NAME" <<'PYEOF'
import csv, sys, numpy as np
from pathlib import Path
name=sys.argv[1]; rows=list(csv.DictReader(open(f"data/omnifall/manifest_{name}.csv")))
fps=[]; miss=0
for r in rows:
    f=Path(f"runs/kps_{name}")/f"{r['dataset']}__{Path(r['clip']).stem}.npy"
    if not f.exists(): miss+=1; continue
    d=float(r["end"])-float(r["start"])
    if d>0.3: fps.append(np.load(f,mmap_mode="r").shape[0]/d)
fps=np.array(fps)
print(f"    클립 {len(rows)} · 누락 {miss} · fps 중앙값 {np.median(fps):.1f} · 5~95% {np.percentile(fps,5):.1f}~{np.percentile(fps,95):.1f}")
print("    → 5~95% 폭이 넓으면 자르기 결함. recut_keypoints.py 또는 retrim_keypoints.py 검토")
PYEOF
echo "[*] 영상 삭제 여부는 fps 검사 뒤 사람이 결정: rm -rf $UNZ"
echo "DONE $NAME"
