#!/usr/bin/env python3
"""실시간 데모 결과(events.jsonl) → 상태 타임라인 + 알림 시점 스냅샷 보고서.

realtime_demo.py 가 남긴 events.jsonl 과 저장 영상을 읽어, 영상마다
  · 시간축 상태 띠 (normal/fall/fallen/no_person 색)
  · 위험 확률 스파크라인 + 정답 낙상 구간 음영
  · 이벤트 시점 프레임 스냅샷 (FALL · FALLEN_IMMOBILE · STILL_HORIZONTAL)
을 카드로 만든다. AI Hub 보고서(make_pipeline_report.py)와 같은 형태다.

사용법:
    python make_demo_report.py --demo runs/demo --labels <Sample>/02.라벨링데이터/영상 --pdf
"""
from __future__ import annotations
import argparse, base64, html, json
from pathlib import Path
import cv2, numpy as np

COL = {"normal": "#4fc06a", "fall": "#ff7a1a", "fallen": "#e02020", "no_person": "#9a9a9a"}
EV_LABEL = {"FALL": "낙상 감지", "FALLEN_IMMOBILE": "낙상 후 방치 (R1)", "STILL_HORIZONTAL": "누운 채 미동 (R2)"}
EV_LEVEL = {"FALL": "warn", "FALLEN_IMMOBILE": "crit", "STILL_HORIZONTAL": "warn"}


def snap(video: Path, frame: int, width: int) -> str | None:
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, fr = cap.read(); cap.release()
    if not ok: return None
    h = int(fr.shape[0] * width / fr.shape[1])
    fr = cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 74])
    return base64.b64encode(buf).decode() if ok else None


def state_strip(states, total, w=560, h=14):
    """상태 시계열을 색 띠로. 각 stride 구간을 그 시점 상태 색으로 칠한다."""
    parts = []
    for i, r in enumerate(states):
        x0 = r["frame"] / total * w
        x1 = (states[i + 1]["frame"] if i + 1 < len(states) else total) / total * w
        parts.append(f'<rect x="{x0:.1f}" y="0" width="{max(x1-x0,1):.1f}" height="{h}" fill="{COL.get(r["state"],"#ccc")}"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none" '
            f'role="img" aria-label="상태 타임라인">{"".join(parts)}</svg>')


def spark(states, total, gt, w=560, h=50):
    pts = " ".join(f"{r['frame']/total*w:.1f},{h-4-(r['prob'][1]+r['prob'][2])*(h-8):.1f}" for r in states)
    shade = ""
    if gt and gt[1] > gt[0]:
        shade = (f'<rect x="{gt[0]/total*w:.1f}" y="0" width="{max((gt[1]-gt[0])/total*w,2):.1f}" '
                 f'height="{h}" fill="var(--gt)" opacity=".3"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none" role="img" '
            f'aria-label="위험 확률">{shade}<line x1="0" y1="{h/2:.0f}" x2="{w}" y2="{h/2:.0f}" '
            f'stroke="var(--line)" stroke-dasharray="3 3"/><polyline points="{pts}" fill="none" '
            f'stroke="var(--accent)" stroke-width="2"/></svg>')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", type=Path, default=Path("runs/demo"), help="events.jsonl 과 mp4 가 있는 폴더")
    ap.add_argument("--labels", type=Path, default=Path("data/aihub_sample/02.라벨링데이터/영상"),
                    help="AI Hub 라벨 루트 (정답 구간 음영용)")
    ap.add_argument("--out", type=Path, default=Path("runs/report"))
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    labels = {}
    if args.labels and args.labels.exists():
        for j in args.labels.rglob("*.json"):
            d = json.loads(j.read_text(encoding="utf-8"))
            sid = d["metadata"]["scene_id"]; sd = d.get("sensordata", {})
            labels[sid.split("_H_A_")[0] + "_" + sid.rsplit("_", 1)[1]] = (
                sd.get("fall_start_frame", 0), sd.get("fall_end_frame", 0),
                d["scene_info"].get("scene_cat_name", "?"), d["scene_info"].get("scene_IsFall") == "낙상")

    cards, summary = [], []
    for ev_f in sorted(args.demo.glob("*.jsonl")):
        rows = [json.loads(l) for l in ev_f.open(encoding="utf-8")]
        states = [r for r in rows if r["event"] is None]
        events = [r for r in rows if r["event"]]
        if not states: continue
        video = ev_f.with_suffix(".mp4")
        total = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT)) if video.exists() else states[-1]["frame"] + 1
        key = ev_f.stem                                   # 예: 00003_C1
        gt_s, gt_e, cls, is_fall = labels.get(key, (0, 0, "?", None))
        crit = [e for e in events if EV_LEVEL[e["event"]] == "crit"]
        falls = [e for e in events if e["event"] == "FALL"]
        first_fall = falls[0]["frame"] if falls else None
        latency = (first_fall - gt_s) if (first_fall is not None and gt_e > gt_s) else None
        summary.append(dict(key=key, cls=cls, is_fall=is_fall, n_fall=len(falls), n_crit=len(crit),
                            latency=latency, first_fall=first_fall))

        E = html.escape
        snaps = ""
        for e in events:
            b64 = snap(video, e["frame"], args.width) if video.exists() else None
            lvl = EV_LEVEL[e["event"]]
            snaps += (f'<figure class="ev {lvl}">'
                      + (f'<img src="data:image/jpeg;base64,{b64}" width="{args.width}" alt="{E(e["event"])} at {e["frame"]}f">' if b64 else "")
                      + f'<figcaption><b>{E(EV_LABEL[e["event"]])}</b> · {e["frame"]}f / {e["t_sec"]}s · '
                        f'상태 {E(e["state"])} · 위험 {e["prob"][1]+e["prob"][2]:.2f}</figcaption></figure>')
        verdict = ""
        if is_fall is not None:
            ok = (len(falls) > 0) == is_fall
            verdict = f'<span class="badge {"b-ok" if ok else "b-bad"}">{"정답" if ok else "오답"}</span>'
        cards.append(f'''
<div class="card">
 <div class="hd"><span class="fn">{E(key)}</span><span class="badge b-n">{E(str(cls))}</span>
   {verdict}<span class="badge b-n">FALL {len(falls)} · CRITICAL {len(crit)}</span>
   {f'<span class="badge b-n">감지 지연 {latency:+d}f ({latency/60:+.2f}s)</span>' if latency is not None else ""}</div>
 <div class="strip">{state_strip(states, total)}</div>
 <div class="spark">{spark(states, total, (gt_s, gt_e))}</div>
 <div class="snaps">{snaps or '<div class="none">이벤트 없음</div>'}</div>
</div>''')

    n_fall_vid = sum(1 for s in summary if s["is_fall"]); n_nonfall = sum(1 for s in summary if s["is_fall"] is False)
    det = sum(1 for s in summary if s["is_fall"] and s["n_fall"] > 0)
    fp = sum(1 for s in summary if s["is_fall"] is False and s["n_fall"] > 0)
    crit_fp = sum(1 for s in summary if s["is_fall"] is False and s["n_crit"] > 0)
    lats = [s["latency"] for s in summary if s["latency"] is not None]

    css = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--card:#fafafa;--ok:#0a7d33;--ok-bg:#e8f6ec;
--bad:#c62828;--bad-bg:#fdeaea;--accent:#1a4f8a;--gt:#e8a33d;--warn:#b26a00;--crit:#c62828}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181c;--fg:#e8e8e8;--muted:#9aa0a6;
--line:#2e3238;--card:#1e2126;--ok:#6ee7a0;--ok-bg:#12331f;--bad:#ff8a8a;--bad-bg:#3a1d1d;--accent:#7fb3f0;--gt:#d99a2b}}
:root[data-theme="dark"]{--bg:#16181c;--fg:#e8e8e8;--muted:#9aa0a6;--line:#2e3238;--card:#1e2126;--ok:#6ee7a0;
--ok-bg:#12331f;--bad:#ff8a8a;--bad-bg:#3a1d1d;--accent:#7fb3f0;--gt:#d99a2b}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto}h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 12px;
padding-bottom:6px;border-bottom:2px solid var(--line)}.sub{color:var(--muted);font-size:14px;margin-bottom:20px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}.stat{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px 18px;min-width:118px;flex:1}.stat .v{font-size:25px;font-weight:700;line-height:1.2}
.stat .l{font-size:12px;color:var(--muted)}.card{border:1px solid var(--line);border-radius:12px;margin:14px 0;
overflow:hidden;background:var(--card)}.hd{display:flex;flex-wrap:wrap;align-items:center;gap:9px;padding:10px 14px;
border-bottom:1px solid var(--line)}.fn{font-family:ui-monospace,Menlo,monospace;font-size:13px;font-weight:600}
.badge{font-size:12px;padding:3px 10px;border-radius:99px;font-weight:600}.b-ok{background:var(--ok-bg);color:var(--ok)}
.b-bad{background:var(--bad-bg);color:var(--bad)}.b-n{background:var(--line)}.strip{padding:10px 14px 0}
.spark{padding:4px 14px 8px}.snaps{display:flex;flex-wrap:wrap;gap:12px;padding:10px 14px 14px;border-top:1px solid var(--line)}
.ev{margin:0;border:2px solid var(--warn);border-radius:8px;overflow:hidden;background:#000}
.ev.crit{border-color:var(--crit)}.ev img{display:block}.ev figcaption{background:var(--card);color:var(--fg);
font-size:12px;padding:6px 8px}.none{color:var(--muted);font-size:13px}
.legend span{display:inline-block;margin-right:14px;font-size:13px}.legend i{display:inline-block;width:12px;height:12px;
border-radius:3px;vertical-align:middle;margin-right:5px}
.note{background:var(--card);border-left:3px solid var(--accent);padding:11px 15px;margin:14px 0;border-radius:0 8px 8px 0;font-size:14px}
@media print{:root{--bg:#fff;--fg:#111;--muted:#555;--line:#ccc;--card:#f7f7f7;--accent:#1a4f8a}
body{padding:0;font-size:11pt}.wrap{max-width:none}.card,.stat,.note{page-break-inside:avoid}h2{page-break-after:avoid}
.snaps{flex-wrap:nowrap;overflow:visible}.ev{flex:1 1 0;min-width:0}.ev img{width:100%;height:auto}}
@page{size:A4;margin:12mm 10mm}"""
    E = html.escape
    P = [f'<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>실시간 데모 결과</title><style>{css}</style></head><body><div class="wrap">',
         '<h1>실시간 프로토타입 — 낙상·미동 감지</h1>',
         f'<div class="sub">YOLOv11-pose → 카메라 불변 키포인트 → GRU → 규칙 기반 미동 감지 · 영상 {len(summary)}개 · AI Hub 샘플</div>',
         '<div class="stats">']
    for v, l in [(f"{det}/{n_fall_vid}", "낙상 영상 감지"), (f"{fp}/{n_nonfall}", "비낙상 FALL 오탐"),
                 (f"{crit_fp}/{n_nonfall}", "비낙상 CRITICAL 오탐"),
                 (f"{np.median(lats):+.0f}f" if lats else "—", "감지 지연 중앙값")]:
        P.append(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>')
    P.append('</div>')
    P.append('<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{s}</span>' for s, c in COL.items())
             + '<span><i style="background:var(--gt);opacity:.5"></i>정답 낙상 구간</span></div>')
    P.append('<div class="note"><b>미동 감지 규칙.</b> R1 — 위험 상태(fall/fallen)가 3초 이상 지속되면 '
             '<b>낙상 후 방치</b>(CRITICAL). R2 — 움직임이 거의 없고 몸통이 수평인 상태가 5초 지속되면 '
             '<b>누운 채 미동</b>(WARNING). 모델은 프레임 상태만 내고 시간 누적은 규칙이 한다. '
             '사람이 검출되지 않은 윈도우에서는 판정하지 않는다(no_person).</div>')
    P.append("<h2>영상별 결과</h2>" + "".join(cards))
    P.append('<h2>한계</h2><div class="note">AI Hub 샘플은 장면 4개뿐이며 비낙상은 1장면이다. '
             '비낙상 영상의 FALL 오탐은 학습 데이터의 <code>lying</code> 부족(23개, 1.9%)에서 오는 알려진 한계다 — '
             '피험자가 침대에 엎드리는 장면을 낙상으로 본다. 다만 위험 상태가 2초 만에 풀려 <b>CRITICAL(R1)은 '
             '발화하지 않았다</b>. 지속 시간 조건이 순간 오판을 걸러낸 사례다.</div>')
    P.append("</div></body></html>")

    args.out.mkdir(parents=True, exist_ok=True)
    f = args.out / "demo_report.html"; f.write_text("".join(P), encoding="utf-8")
    print(f"[*] HTML {f} ({f.stat().st_size/1e6:.1f}MB)")
    if args.pdf:
        try:
            from weasyprint import HTML
            p = f.with_suffix(".pdf"); HTML(filename=str(f)).write_pdf(str(p))
            print(f"[*] PDF {p} ({p.stat().st_size/1e6:.1f}MB)")
        except Exception as e:
            print(f"[!] PDF 실패: {e}")
    for s in summary:
        print(f"  {s['key']:12s} {str(s['cls']):8s} FALL {s['n_fall']} CRIT {s['n_crit']} "
              + (f"지연 {s['latency']:+d}f" if s["latency"] is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
