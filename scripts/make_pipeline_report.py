#!/usr/bin/env python3
"""파이프라인 판정 결과 → 프레임 썸네일 + 확률 시계열 HTML/PDF 보고서.

영상 카드 하나에 시간순 프레임을 가로로 붙이고 그 아래에 확률을 한 줄로 쓴다.
정답 낙상 구간은 확률 그래프에 음영으로 표시해 **감지 시점**을 눈으로 볼 수 있다.

같은 장면을 카메라 8대로 찍었으므로 장면별로 묶어 **카메라별 판정 편차**를
바로 비교할 수 있게 배치한다. 이것이 본 연구의 핵심 주장(카메라 독립성)에
대한 직접 증거다.

사용법:
    python make_pipeline_report.py --pred runs/aihub_sample/predictions.json \
        --sample ~/Downloads/Sample --out runs/report [--pdf]
"""
from __future__ import annotations
import argparse, base64, collections, html, json
from pathlib import Path
import numpy as np
import cv2

CHROME = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]


def thumbs(video: Path, n: int, width: int) -> list[tuple[int, str]]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 600
    out = []
    for i in range(n):
        idx = round(i * (total - 1) / (n - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok: continue
        h = int(fr.shape[0] * width / fr.shape[1])
        fr = cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok: out.append((idx, base64.b64encode(buf).decode()))
    cap.release()
    return out


def spark(risk, starts, window, total, gt_s, gt_e, w=560, h=54):
    """위험 확률 시계열 SVG. 정답 구간을 음영으로 표시한다."""
    if not risk: return ""
    pts = []
    for r, s in zip(risk, starts):
        x = (s + window/2) / max(total, 1) * w
        pts.append(f"{x:.1f},{h - r*(h-8) - 4:.1f}")
    shade = ""
    if gt_e > gt_s:
        x1, x2 = gt_s/max(total,1)*w, gt_e/max(total,1)*w
        shade = (f'<rect x="{x1:.1f}" y="0" width="{max(x2-x1,2):.1f}" height="{h}" '
                 f'fill="var(--gt)" opacity="0.28"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'preserveAspectRatio="none" role="img" aria-label="위험 확률 시계열">'
            f'{shade}<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="var(--accent)" stroke-width="2"/>'
            f'<line x1="0" y1="{h-4-0.5*(h-8):.1f}" x2="{w}" y2="{h-4-0.5*(h-8):.1f}" '
            f'stroke="var(--line)" stroke-dasharray="3 3" stroke-width="1"/></svg>')


def to_pdf(src: Path, dst: Path) -> bool:
    try:
        from weasyprint import HTML
        HTML(filename=str(src)).write_pdf(str(dst))
        return dst.exists() and dst.stat().st_size > 1000
    except Exception as e:
        print(f"[!] PDF 실패: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/report"))
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--width", type=int, default=300)
    ap.add_argument("--thr", type=float, default=0.5, help="위험 판정 임계값")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    d = json.loads(args.pred.read_text(encoding="utf-8"))
    res, names, kind = d["results"], d["classes"], d["model"]
    vids = {v.stem: v for v in (args.sample/"01.원천데이터"/"영상").rglob("*.mp4")}

    # 지표
    TP = sum(1 for r in res if r["is_fall"] and r["risk_max"] > args.thr)
    FN = sum(1 for r in res if r["is_fall"] and r["risk_max"] <= args.thr)
    FP = sum(1 for r in res if not r["is_fall"] and r["risk_max"] > args.thr)
    TN = sum(1 for r in res if not r["is_fall"] and r["risk_max"] <= args.thr)
    rec = TP/max(TP+FN,1); fpr = FP/max(FP+TN,1)
    prec = TP/max(TP+FP,1)

    # 감지 지연 (정답 구간 있는 것만)
    lat = []
    for r in res:
        if not r["is_fall"] or r["gt_end"] <= r["gt_start"]: continue
        over = [s + r["window"]//2 for s, v in zip(r["starts"], r["risk"]) if v > args.thr]
        if over: lat.append(min(over) - r["gt_start"])

    # 장면별 카메라 편차
    scenes = collections.defaultdict(list)
    for r in res: scenes[r["scene"].rsplit("_C",1)[0]].append(r)
    dev = {k: float(np.std([x["risk_max"] for x in v])) for k, v in scenes.items()}

    print(f"[*] 썸네일 추출 {len(res)}개")
    th = {}
    for i, r in enumerate(res, 1):
        v = vids.get(r["scene"])
        if v: th[r["scene"]] = thumbs(v, args.frames, args.width)
        print(f"  [{i}/{len(res)}] {r['scene']}", end="\r")
    print()

    E = html.escape
    P = [f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>파이프라인 판정 — AI Hub 샘플</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--card:#fafafa;
--ok:#0a7d33;--ok-bg:#e8f6ec;--bad:#c62828;--bad-bg:#fdeaea;--accent:#1a4f8a;--gt:#e8a33d;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16181c;--fg:#e8e8e8;
--muted:#9aa0a6;--line:#2e3238;--card:#1e2126;--ok:#6ee7a0;--ok-bg:#12331f;--bad:#ff8a8a;
--bad-bg:#3a1d1d;--accent:#7fb3f0;--gt:#d99a2b;}}}}
:root[data-theme="dark"]{{--bg:#16181c;--fg:#e8e8e8;--muted:#9aa0a6;--line:#2e3238;--card:#1e2126;
--ok:#6ee7a0;--ok-bg:#12331f;--bad:#ff8a8a;--bad-bg:#3a1d1d;--accent:#7fb3f0;--gt:#d99a2b;}}
*{{box-sizing:border-box}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Segoe UI",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 4px}}
h2{{font-size:19px;margin:34px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--line)}}
h3{{font-size:16px;margin:22px 0 10px;color:var(--accent)}}
.sub{{color:var(--muted);margin-bottom:22px;font-size:14px}}
.stats{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px 18px;min-width:118px;flex:1}}
.stat .v{{font-size:26px;font-weight:700;line-height:1.2}}
.stat .l{{font-size:12px;color:var(--muted);margin-top:2px}}
table{{border-collapse:collapse;margin:12px 0;font-size:14px}}
th,td{{border:1px solid var(--line);padding:7px 12px;text-align:center}}
th{{background:var(--card);font-weight:600}}
td.diag{{background:var(--ok-bg);font-weight:700}} td.off{{background:var(--bad-bg)}}
.card{{border:1px solid var(--line);border-radius:12px;margin:14px 0;overflow:hidden;background:var(--card)}}
.card.wrong{{border-color:var(--bad);border-width:2px}}
.hd{{display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:11px 15px;border-bottom:1px solid var(--line)}}
.fn{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;font-weight:600}}
.badge{{font-size:12px;padding:3px 10px;border-radius:99px;font-weight:600;white-space:nowrap}}
.b-ok{{background:var(--ok-bg);color:var(--ok)}} .b-bad{{background:var(--bad-bg);color:var(--bad)}}
.b-n{{background:var(--line);color:var(--fg)}}
.frames{{display:flex;gap:0;overflow-x:auto;background:#000}}
.frames figure{{margin:0;flex:0 0 auto;position:relative}}
.frames img{{display:block;height:auto}}
.frames figcaption{{position:absolute;left:0;bottom:0;background:rgba(0,0,0,.7);color:#fff;
font-size:10px;padding:2px 6px;font-family:ui-monospace,monospace}}
.prob{{padding:9px 15px;font-size:13px;display:flex;flex-wrap:wrap;gap:14px;
align-items:center;border-top:1px solid var(--line)}}
.bar{{display:inline-block;height:7px;border-radius:4px;background:var(--accent);
vertical-align:middle;margin:0 5px}}
.spark{{padding:4px 15px 10px}}
.note{{background:var(--card);border-left:3px solid var(--accent);padding:11px 15px;
margin:14px 0;border-radius:0 8px 8px 0;font-size:14px}}
@media print{{:root{{--bg:#fff;--fg:#111;--muted:#555;--line:#ccc;--card:#f7f7f7;
--ok:#0a7d33;--ok-bg:#e8f6ec;--bad:#c62828;--bad-bg:#fdeaea;--accent:#1a4f8a}}
body{{padding:0;font-size:11pt}} .wrap{{max-width:none}}
h2,h3{{page-break-after:avoid}} .card,.stat,table,.note{{page-break-inside:avoid}}
.frames{{overflow:visible;display:flex}} .frames figure{{flex:1 1 0;min-width:0}}
.frames img{{width:100%;height:auto}}}}
@page{{size:A4;margin:12mm 10mm}}
</style></head><body><div class="wrap">"""]

    P.append("<h1>파이프라인 판정 결과 — AI Hub 샘플</h1>")
    P.append(f'<div class="sub">YOLOv11-pose → 키포인트(카메라 불변 정규화) → '
             f'슬라이딩 윈도우 → {E(kind)} · 영상 {len(res)}개</div>')
    P.append('<div class="stats">')
    for v, l in [(f"{rec:.1%}","Recall (낙상 감지)"), (f"{fpr:.1%}","FPR (오경보)"),
                 (f"{prec:.1%}","Precision"), (str(FN),"놓친 낙상"),
                 (f"{np.median(lat):.0f}f" if lat else "—","감지 지연 중앙값")]:
        P.append(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>')
    P.append("</div>")

    P.append("<h2>영상 단위 판정</h2><table><tr><th>실제 \\ 예측</th><th>낙상</th><th>비낙상</th></tr>")
    P.append(f'<tr><th>낙상</th><td class="diag">{TP}</td><td class="off">{FN}</td></tr>')
    P.append(f'<tr><th>비낙상</th><td class="off">{FP}</td><td class="diag">{TN}</td></tr></table>')
    if lat:
        P.append(f'<div class="note">감지 지연: 중앙값 <b>{np.median(lat):.0f}프레임</b> '
                 f'({np.median(lat)/60:.2f}초) · 범위 {min(lat)}~{max(lat)}. '
                 f'음수는 정답 구간 시작 전에 감지한 것이다.</div>')

    P.append("<h2>카메라별 판정 편차</h2>")
    P.append('<div class="sub">같은 장면을 8각도로 촬영했다. 표준편차가 작을수록 '
             '카메라 위치에 강건하다.</div><table><tr><th>장면</th><th>클래스</th>'
             '<th>위험확률 평균</th><th>표준편차</th></tr>')
    for sc, v in sorted(scenes.items()):
        vals = [x["risk_max"] for x in v]
        P.append(f'<tr><td>{E(sc)}</td><td>{E(v[0]["true_class"])}</td>'
                 f'<td>{np.mean(vals):.3f}</td><td><b>{np.std(vals):.3f}</b></td></tr>')
    P.append("</table>")
    P.append('<div class="note">AI Hub 공식 베이스라인(RandomForest, 카메라별 모델 8개)은 '
             '장면 00151 에서 표준편차 <b>0.30</b> 이었다 — C5 에서 0.98, C2 에서 0.23 으로 '
             '같은 낙상에 정반대 판정을 냈다. 본 파이프라인은 좌표를 엉덩이 중점·몸통 길이로 '
             '정규화해 카메라 불변 표현을 쓴다.</div>')

    P.append("<h2>영상별 판정</h2>")
    for sc in sorted(scenes):
        rows = sorted(scenes[sc], key=lambda r: r["cam"])
        nok = sum(1 for r in rows if (r["risk_max"] > args.thr) == r["is_fall"])
        P.append(f'<h3>장면 {E(sc)} — {E(rows[0]["true_class"])} '
                 f'<span style="color:var(--muted);font-weight:400">({nok}/{len(rows)} 정답)</span></h3>')
        for r in rows:
            pred = r["risk_max"] > args.thr
            ok = pred == r["is_fall"]
            P.append(f'<div class="card{"" if ok else " wrong"}">')
            P.append(f'<div class="hd"><span class="fn">{E(r["scene"])}</span>'
                     f'<span class="badge b-n">C{r["cam"]}</span>'
                     f'<span class="badge b-n">실제 {E(r["true_class"])}</span>'
                     f'<span class="badge {"b-ok" if ok else "b-bad"}">'
                     f'예측 {"낙상" if pred else "비낙상"} {"✓" if ok else "✗"}</span></div>')
            fr = th.get(r["scene"], [])
            if fr:
                P.append('<div class="frames">')
                for idx, b64 in fr:
                    P.append(f'<figure><img src="data:image/jpeg;base64,{b64}" '
                             f'width="{args.width}" alt="frame {idx}">'
                             f'<figcaption>{idx}f / {idx/60:.1f}s</figcaption></figure>')
                P.append("</div>")
            total = max(r["starts"]) + r["window"] if r["starts"] else 600
            P.append(f'<div class="spark">'
                     f'{spark(r["risk"], r["starts"], r["window"], total, r["gt_start"], r["gt_end"])}</div>')
            P.append('<div class="prob">')
            P.append(f'<span>위험 최대 <b>{r["risk_max"]:.2f}</b>'
                     f'<span class="bar" style="width:{max(2,round(r["risk_max"]*70))}px"></span></span>')
            P.append(f'<span style="color:var(--muted)">피크 {r["peak_frame"]}f</span>')
            if r["gt_end"] > r["gt_start"]:
                P.append(f'<span style="color:var(--muted)">정답 {r["gt_start"]}~{r["gt_end"]}f</span>')
            P.append("</div></div>")

    P.append('<h2>한계</h2><div class="note">장면 4개(낙상 3 + 비낙상 1)를 카메라 8대로 '
             f'촬영한 {len(res)}영상이다. 독립 표본이 아니므로 <b>정량 성능의 신뢰구간이 매우 넓다</b>. '
             '이 결과는 카메라 강건성의 정성적 증거로 사용하고, 정량 성능은 OmniFall 5-fold '
             '결과로 보고한다.</div>')
    P.append("</div></body></html>")

    args.out.mkdir(parents=True, exist_ok=True)
    f = args.out / "pipeline_aihub.html"
    f.write_text("".join(P), encoding="utf-8")
    print(f"[*] HTML {f} ({f.stat().st_size/1e6:.1f}MB)")
    if args.pdf:
        p = f.with_suffix(".pdf")
        if to_pdf(f, p): print(f"[*] PDF {p} ({p.stat().st_size/1e6:.1f}MB)")
    print(f"\nRecall {rec:.3f} · FPR {fpr:.3f} · 놓친 낙상 {FN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
