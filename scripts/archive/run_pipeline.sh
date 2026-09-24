#!/usr/bin/env bash
# Colab 세션에서 전체 파이프라인을 한 번에 돌린다.
#
# Colab 세션은 예고 없이 회수된다 (실측: GPU 세션이 40분 만에 404).
# /content 는 휘발성이므로 **결과는 반드시 Drive 에 남긴다**.
#
#   압축해제(/content, 재생성 가능)  →  키포인트(.npy, Drive 저장)  →  학습
#                                        └ 수십 MB 뿐이라 Drive 에 둬도 부담 없다
#                                        └ 세션이 죽어도 여기서 이어서 시작
#
# 사용법 (Colab VM 안에서):
#   bash run_pipeline.sh

set -uo pipefail

DRIVE=/content/drive/MyDrive/falldata
WORK=/content/work
KPS=$DRIVE/kps                 # ← Drive. 세션이 죽어도 살아남는다
MANIFEST=$DRIVE/manifest.csv

if [ ! -d "$DRIVE" ]; then
    echo "[!] Drive 가 마운트되지 않았습니다. 먼저 실행하세요:"
    echo "    colab drivemount -s <세션명>"
    exit 1
fi

mkdir -p "$WORK/unz" "$KPS"
cd "$WORK"

# ── 1. 압축 해제 ────────────────────────────────────────────────
# 기본은 작은 3개(약 2.8GB)만 쓴다. Colab 세션은 예고 없이 회수되므로
# 재시작 비용을 낮추는 편이 낫다 — 이 3개로도 1,194 클립·19명이 나온다.
# 전부 쓰려면:  DATASETS="Cauca_fall EDF GMDCSA24 MCFD UP_Fall LE2I OCCU" bash run_pipeline.sh
#
# OOPS 는 항상 제외한다 — Zenodo 배포판에 영상이 없다 (저작권, 12KB 껍데기뿐).
# cut.py 로 원본을 따로 구해야 하므로 지금은 쓸 수 없다.
DATASETS=${DATASETS:-"Cauca_fall EDF GMDCSA24"}
for z in $DATASETS; do
    [ -d "unz/$z" ] && continue
    [ -f "$DRIVE/omnifall/$z.zip" ] || { echo "[!] $z.zip 없음"; continue; }
    echo "[*] $z 해제"
    mkdir -p "unz/$z"
    cp "$DRIVE/omnifall/$z.zip" /content/_t.zip
    unzip -qq -o /content/_t.zip -d "unz/$z"
    rm -f /content/_t.zip
done
echo "[*] 해제 완료: $(du -sh unz | cut -f1)"

# ── 2. manifest ─────────────────────────────────────────────────
python3 - <<'PY'
import csv, re, collections
from pathlib import Path
root = Path("/content/work/unz")
# 파일명 접두부는 데이터셋마다 다르다. 뒤 6개 필드만 공통이므로 뒤에서 파싱한다.
# dataset 은 영문자로 시작해야 한다 — \w+ 는 숫자도 먹어 subject 를 삼킨다.
TAIL = re.compile(r"_(?P<label>\d+)_(?P<start>[\d.]+)_(?P<end>[\d.]+)_"
                  r"(?P<subject>-?\d+)_(?P<cam>-?\d+)_(?P<dataset>[A-Za-z][A-Za-z0-9]*)\.avi$")
seen = {}
for p in sorted(root.rglob("*.avi")):
    m = TAIL.search(p.name)
    if not m: continue
    d = m.groupdict()
    if d["dataset"] == "OOPS": continue          # 영상 없음
    key = (d["dataset"], d["subject"], d["cam"], d["start"], d["end"], d["label"])
    # 같은 구간이 두 파일명 규칙으로 중복 수록돼 있다. 짧은 쪽만 남긴다.
    if key not in seen or len(p.name) < len(seen[key].name):
        seen[key] = p
rows = []
for p in seen.values():
    d = TAIL.search(p.name).groupdict()
    rows.append(dict(clip=str(p.relative_to(root)), label=int(d["label"]),
                     start=float(d["start"]), end=float(d["end"]),
                     subject=int(d["subject"]), cam=int(d["cam"]), dataset=d["dataset"]))
rows.sort(key=lambda r: (r["dataset"], r["subject"], r["clip"]))
out = Path("/content/drive/MyDrive/falldata/manifest.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
LBL = {0:"walk",1:"fall",2:"fallen",3:"sit_down",4:"sitting",5:"lie_down",
       6:"lying",7:"stand_up",8:"standing",9:"other"}
print(f"manifest {len(rows):,}행 → {out}")
print("데이터셋:", dict(collections.Counter(r["dataset"] for r in rows)))
for l, n in sorted(collections.Counter(r["label"] for r in rows).items()):
    print(f"  {l} {LBL[l]:10s} {n:5,} {n/len(rows)*100:5.1f}%")
PY

# ── 3. 키포인트 추출 (Drive 에 저장 — 이미 있으면 건너뛴다) ──────
# CPU 세션에서 돌린다. GPU 세션은 회수가 잦고(실측 40분), 추출은 0.8초/클립이라
# CPU 로도 감당된다. GPU 는 학습할 때만 쓴다.
pip install -q ultralytics 2>&1 | tail -1
DEV=$(python3 -c "import torch;print(0 if torch.cuda.is_available() else 'cpu')")
echo "[*] 추출 device=$DEV"
python3 extract_keypoints.py --manifest "$MANIFEST" --root "$WORK/unz" \
    --out "$KPS" --model yolo11n-pose.pt --device "$DEV" 2>&1 | tail -20

echo "[*] 키포인트: $(ls "$KPS" | wc -l)개 → $KPS"
echo "[*] 이제 학습:  python3 train_stage2.py --kps $KPS --manifest $MANIFEST --all"
