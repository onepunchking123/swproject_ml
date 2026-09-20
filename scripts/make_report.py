#!/usr/bin/env python3
"""추론 결과를 영상 썸네일과 함께 HTML 보고서로 정리한다.

영상별로 시간 순 프레임을 뽑아 판정 결과와 나란히 보여준다.
낙상은 시간에 걸쳐 일어나므로 한 장이 아니라 여러 시점을 봐야
모델이 왜 그렇게 판정했는지 가늠할 수 있다.

사용법:
    python make_report.py --data <영상루트> --task fnf
    python make_report.py --data <영상루트> --task fd --frames 6
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import json
from pathlib import Path

import cv2

FNF_LABEL = {"FALL": "낙상", "NonFall": "비낙상"}
TYPE_LABEL = {"FY": "전면낙상", "BY": "후면낙상", "SY": "측면낙상", "N": "비낙상"}


CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def html_to_pdf(html_file: Path, pdf_file: Path) -> bool:
    """HTML → PDF 변환.

    weasyprint 를 우선 쓴다. 순수 Python 이라 빠르고 안정적이다.
    Chrome headless 는 base64 이미지가 수백 장 들어간 큰 HTML 에서
    렌더링이 끝나지 않는 경우가 있어 보조 경로로만 둔다.
    """
    try:
        from weasyprint import HTML
        HTML(filename=str(html_file)).write_pdf(str(pdf_file))
        return pdf_file.exists() and pdf_file.stat().st_size > 1000
    except ImportError:
        pass
    except Exception as e:
        print(f"[!] weasyprint 실패: {type(e).__name__}: {e}")

    # 보조 경로 — Chrome headless
    import shutil
    import subprocess
    import tempfile

    browser = next((p for p in CHROME_PATHS if Path(p).exists()), None) \
        or shutil.which("chromium") or shutil.which("google-chrome")
    if not browser:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([
                browser, "--headless", "--disable-gpu", "--no-sandbox",
                f"--user-data-dir={tmp}", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_file}", html_file.resolve().as_uri(),
            ], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("[!] Chrome 변환 시간 초과")
        return False
    return pdf_file.exists() and pdf_file.stat().st_size > 1000


def truth_of(r: dict, task: str) -> str:
    if task == "fnf":
        return "NonFall" if r["true"] == "N" else "FALL"
    return r["true"]


def thumb_b64(video: Path, n_frames: int, width: int) -> list[tuple[int, str]]:
    """영상에서 균등 간격으로 프레임을 뽑아 base64 JPEG 로 반환한다."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    idxs = [round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]
    out = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h = int(frame.shape[0] * width / frame.shape[1])
        frame = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            out.append((idx, base64.b64encode(buf).decode()))
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="영상 루트")
    ap.add_argument("--task", choices=["fnf", "fd"], default="fnf")
    ap.add_argument("--results", type=Path, default=Path("runs/aihub_baseline"))
    ap.add_argument("--out", type=Path, default=Path("runs/report"))
    ap.add_argument("--frames", type=int, default=5, help="영상당 추출 프레임 수")
    ap.add_argument("--width", type=int, default=300, help="썸네일 가로 픽셀")
    ap.add_argument("--pdf", action="store_true", help="HTML 과 함께 PDF 도 생성")
    args = ap.parse_args()

    res_f = args.results / f"{args.task}_results.json"
    if not res_f.exists():
        print(f"결과 파일이 없습니다: {res_f}")
        print("먼저 aihub_infer.py 를 실행하세요.")
        return 1

    data = json.loads(res_f.read_text(encoding="utf-8"))
    results = data["results"]
    classes = (["FALL", "NonFall"] if args.task == "fnf" else ["BY", "FY", "SY"])

    videos = {v.stem: v for v in args.data.rglob("*.mp4")}
    print(f"[*] {len(results)}개 결과, 영상 {len(videos)}개 매칭")

    # 장면 ID 로 묶는다 (00003_H_A_FY_C1 → 00003)
    scenes: dict[str, list[dict]] = collections.defaultdict(list)
    for r in results:
        scenes[r["file"].split("_")[0]].append(r)

    # 썸네일 추출
    thumbs: dict[str, list[tuple[int, str]]] = {}
    for i, r in enumerate(results, 1):
        v = videos.get(r["file"])
        if v:
            thumbs[r["file"]] = thumb_b64(v, args.frames, args.width)
        print(f"  [{i}/{len(results)}] {r['file']}", end="\r")
    print()

    # 집계
    correct = sum(1 for r in results if r["pred"] == truth_of(r, args.task))
    by_cam = collections.defaultdict(lambda: [0, 0])
    for r in results:
        by_cam[r["cam"]][1] += 1
        if r["pred"] == truth_of(r, args.task):
            by_cam[r["cam"]][0] += 1

    cm = collections.Counter((truth_of(r, args.task), r["pred"]) for r in results)
    labels = sorted({truth_of(r, args.task) for r in results} | {r["pred"] for r in results})

    # ---- HTML ----
    E = html.escape
    task_name = "낙상유무 탐지 (FNF)" if args.task == "fnf" else "낙상유형 분류 (FD)"
    P = []
    P.append(f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>AI Hub 베이스라인 — {E(task_name)}</title>
<style>
:root {{
  --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#e2e2e2; --card:#fafafa;
  --ok:#0a7d33; --ok-bg:#e8f6ec; --bad:#c62828; --bad-bg:#fdeaea; --accent:#1a4f8a;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#16181c; --fg:#e8e8e8; --muted:#9aa0a6; --line:#2e3238; --card:#1e2126;
  --ok:#6ee7a0; --ok-bg:#12331f; --bad:#ff8a8a; --bad-bg:#3a1d1d; --accent:#7fb3f0;
}} }}
:root[data-theme="dark"] {{
  --bg:#16181c; --fg:#e8e8e8; --muted:#9aa0a6; --line:#2e3238; --card:#1e2126;
  --ok:#6ee7a0; --ok-bg:#12331f; --bad:#ff8a8a; --bad-bg:#3a1d1d; --accent:#7fb3f0;
}}
* {{ box-sizing:border-box }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Segoe UI",sans-serif; }}
.wrap {{ max-width:1180px; margin:0 auto }}
h1 {{ font-size:24px; margin:0 0 4px }}
h2 {{ font-size:19px; margin:36px 0 12px; padding-bottom:6px; border-bottom:2px solid var(--line) }}
h3 {{ font-size:16px; margin:24px 0 10px; color:var(--accent) }}
.sub {{ color:var(--muted); margin-bottom:24px; font-size:14px }}
.stats {{ display:flex; flex-wrap:wrap; gap:10px; margin:16px 0 }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:12px 18px; min-width:120px; flex:1 }}
.stat .v {{ font-size:26px; font-weight:700; line-height:1.2 }}
.stat .l {{ font-size:12px; color:var(--muted); margin-top:2px }}
table {{ border-collapse:collapse; margin:12px 0; font-size:14px }}
th,td {{ border:1px solid var(--line); padding:7px 12px; text-align:center }}
th {{ background:var(--card); font-weight:600 }}
td.diag {{ background:var(--ok-bg); font-weight:700 }}
td.off {{ background:var(--bad-bg) }}
.card {{ border:1px solid var(--line); border-radius:12px; margin:14px 0;
  overflow:hidden; background:var(--card) }}
.card.wrong {{ border-color:var(--bad); border-width:2px }}
.hd {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px;
  padding:11px 15px; border-bottom:1px solid var(--line) }}
.fn {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; font-weight:600 }}
.badge {{ font-size:12px; padding:3px 10px; border-radius:99px; font-weight:600; white-space:nowrap }}
.b-ok {{ background:var(--ok-bg); color:var(--ok) }}
.b-bad {{ background:var(--bad-bg); color:var(--bad) }}
.b-n {{ background:var(--line); color:var(--fg) }}
.sp {{ flex:1 }}
.frames {{ display:flex; gap:0; overflow-x:auto; background:#000 }}
.frames figure {{ margin:0; flex:0 0 auto; position:relative }}
.frames img {{ display:block; height:auto }}
.frames figcaption {{ position:absolute; left:0; bottom:0; background:rgba(0,0,0,.7);
  color:#fff; font-size:10px; padding:2px 6px; font-family:ui-monospace,monospace }}
.prob {{ padding:9px 15px; font-size:13px; display:flex; flex-wrap:wrap;
  gap:14px; align-items:center; border-top:1px solid var(--line) }}
.bar {{ display:inline-block; height:7px; border-radius:4px; background:var(--accent);
  vertical-align:middle; margin:0 5px }}
.note {{ background:var(--card); border-left:3px solid var(--accent);
  padding:11px 15px; margin:14px 0; border-radius:0 8px 8px 0; font-size:14px }}
@media (max-width:640px) {{ body {{ padding:16px 12px }} .stat {{ min-width:100px }} }}

/* 인쇄 / PDF — 항상 라이트 테마로 고정하고 카드가 페이지에서 잘리지 않게 한다 */
@media print {{
  :root {{
    --bg:#fff; --fg:#111; --muted:#555; --line:#ccc; --card:#f7f7f7;
    --ok:#0a7d33; --ok-bg:#e8f6ec; --bad:#c62828; --bad-bg:#fdeaea; --accent:#1a4f8a;
  }}
  body {{ padding:0; font-size:11pt; background:#fff; color:#111 }}
  .wrap {{ max-width:none }}
  h1 {{ font-size:18pt }}
  h2 {{ font-size:14pt; margin-top:18pt; page-break-after:avoid }}
  h3 {{ font-size:12pt; page-break-after:avoid }}
  .card, .stat, table, .note {{ page-break-inside:avoid; break-inside:avoid }}
  /* 가로 스크롤은 인쇄에서 잘리므로 균등 분할해 한 줄에 모두 넣는다 */
  .frames {{ overflow:visible; display:flex }}
  .frames figure {{ flex:1 1 0; min-width:0 }}
  .frames img {{ width:100%; height:auto }}
  .stats {{ gap:6px }}
  a[href]:after {{ content:"" }}
}}
@page {{ size:A4; margin:12mm 10mm }}
</style></head><body><div class="wrap">""")

    P.append(f"<h1>AI Hub 공식 모델 베이스라인</h1>")
    P.append(f'<div class="sub">{E(task_name)} · RandomForest (카메라 위치별 8개 모델) · '
             f'MediaPipe Holistic 997,200차원 입력</div>')

    P.append('<div class="stats">')
    P.append(f'<div class="stat"><div class="v">{data["accuracy"]:.1%}</div>'
             f'<div class="l">정확도 ({correct}/{len(results)})</div></div>')
    if args.task == "fnf":
        TP = cm[("FALL", "FALL")]; FN = cm[("FALL", "NonFall")]
        TN = cm[("NonFall", "NonFall")]; FP = cm[("NonFall", "FALL")]
        rec = TP / (TP + FN) if TP + FN else 0
        fpr = FP / (FP + TN) if FP + TN else 0
        prec = TP / (TP + FP) if TP + FP else 0
        P.append(f'<div class="stat"><div class="v">{rec:.1%}</div><div class="l">Recall (낙상 감지율)</div></div>')
        P.append(f'<div class="stat"><div class="v">{prec:.1%}</div><div class="l">Precision</div></div>')
        P.append(f'<div class="stat"><div class="v">{fpr:.1%}</div><div class="l">FPR (오경보율)</div></div>')
        P.append(f'<div class="stat"><div class="v">{FN}</div><div class="l">놓친 낙상</div></div>')
    P.append("</div>")

    # confusion matrix
    P.append("<h2>Confusion Matrix</h2><table><tr><th>실제 \\ 예측</th>")
    for l in labels:
        P.append(f"<th>{E(FNF_LABEL.get(l, TYPE_LABEL.get(l, l)))}</th>")
    P.append("</tr>")
    for a in labels:
        P.append(f"<tr><th>{E(FNF_LABEL.get(a, TYPE_LABEL.get(a, a)))}</th>")
        for b in labels:
            c = cm[(a, b)]
            P.append(f'<td class="{"diag" if a==b else ("off" if c else "")}">{c}</td>')
        P.append("</tr>")
    P.append("</table>")

    # 카메라별
    P.append("<h2>카메라별 정확도</h2><table><tr><th>카메라</th>")
    for c in sorted(by_cam):
        P.append(f"<th>C{c}</th>")
    P.append("</tr><tr><th>정확도</th>")
    for c in sorted(by_cam):
        ok, n = by_cam[c]
        P.append(f'<td class="{"diag" if ok==n else "off"}">{ok}/{n}</td>')
    P.append("</tr></table>")

    worst = min(by_cam.items(), key=lambda kv: kv[1][0] / kv[1][1])
    if worst[1][0] < worst[1][1]:
        P.append(f'<div class="note"><b>C{worst[0]}</b> 의 정확도가 '
                 f'{worst[1][0]}/{worst[1][1]} 으로 가장 낮다. 오분류가 특정 카메라에 '
                 f'몰린다는 것은, 카메라 위치별로 모델을 나눈 구조에서 일부 각도는 '
                 f'낙상이 잘 판별되지 않음을 뜻한다. 실환경에서는 설치 각도를 '
                 f'통제할 수 없으므로 구조적 한계가 된다.</div>')

    # 장면별 상세
    P.append("<h2>영상별 판정 결과</h2>")
    P.append('<div class="sub">같은 장면을 카메라 8대로 촬영한 것이다. '
             '동일 사건에 대해 카메라마다 판정이 갈리는지 확인할 수 있다.</div>')

    for scene in sorted(scenes):
        rows = sorted(scenes[scene], key=lambda r: r["cam"])
        true_lab = TYPE_LABEL.get(rows[0]["true"], rows[0]["true"])
        n_ok = sum(1 for r in rows if r["pred"] == truth_of(r, args.task))
        P.append(f'<h3>장면 {E(scene)} — {E(true_lab)} '
                 f'<span style="color:var(--muted);font-weight:400">'
                 f'({n_ok}/{len(rows)} 정답)</span></h3>')

        for r in rows:
            ok = r["pred"] == truth_of(r, args.task)
            pl = FNF_LABEL.get(r["pred"], TYPE_LABEL.get(r["pred"], r["pred"]))
            tl = FNF_LABEL.get(truth_of(r, args.task),
                               TYPE_LABEL.get(truth_of(r, args.task), truth_of(r, args.task)))
            P.append(f'<div class="card{"" if ok else " wrong"}">')
            P.append(f'<div class="hd"><span class="fn">{E(r["file"])}</span>')
            P.append(f'<span class="badge b-n">C{r["cam"]}</span>')
            P.append(f'<span class="badge b-n">실제 {E(tl)}</span>')
            P.append(f'<span class="badge {"b-ok" if ok else "b-bad"}">'
                     f'예측 {E(pl)} {"✓" if ok else "✗"}</span><span class="sp"></span></div>')

            fr = thumbs.get(r["file"], [])
            if fr:
                P.append('<div class="frames">')
                for idx, b64 in fr:
                    P.append(f'<figure><img src="data:image/jpeg;base64,{b64}" '
                             f'width="{args.width}" alt="frame {idx}">'
                             f'<figcaption>{idx}f / {idx/60:.1f}s</figcaption></figure>')
                P.append("</div>")

            P.append('<div class="prob">')
            for cls, p in zip(classes, r["prob"]):
                cl = FNF_LABEL.get(cls, TYPE_LABEL.get(cls, cls))
                P.append(f'<span>{E(cl)} <b>{p:.2f}</b>'
                         f'<span class="bar" style="width:{max(2,round(p*70))}px"></span></span>')
            P.append("</div></div>")

    P.append('<h2>한계</h2><div class="note">'
             '이 결과는 <b>4개 장면</b>(낙상 3 + 비낙상 1)을 카메라 8대로 촬영한 것이다. '
             f'{len(results)}건은 독립 샘플이 아니므로 통계적 신뢰구간이 매우 넓다. '
             '본 평가는 Validation 원천데이터(55GB) 확보 후 다시 수행해야 한다. '
             '현 단계의 목적은 파이프라인 검증과 재현 절차 확정이다.</div>')
    P.append("</div></body></html>")

    args.out.mkdir(parents=True, exist_ok=True)
    out_f = args.out / f"{args.task}_report.html"
    out_f.write_text("".join(P), encoding="utf-8")
    size = out_f.stat().st_size / 1e6
    print(f"[*] 보고서 저장: {out_f}  ({size:.1f} MB)")

    if args.pdf:
        pdf_f = out_f.with_suffix(".pdf")
        if html_to_pdf(out_f, pdf_f):
            print(f"[*] PDF 저장: {pdf_f}  ({pdf_f.stat().st_size/1e6:.1f} MB)")
        else:
            print("[!] PDF 변환 실패 — 브라우저에서 HTML 을 열고 "
                  "⌘P → 'PDF로 저장' 을 사용하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
