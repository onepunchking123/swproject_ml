#!/usr/bin/env python3
"""OmniFall test 클립 → 판정 결과 시각 보고서.

AI Hub 샘플 보고서와 목적이 다르다.

  AI Hub  600프레임 통짜 영상 · 같은 장면 8각도
          → 슬라이딩 윈도우 · 감지 지연 · 카메라 편차
  OmniFall 구간별로 잘린 짧은 클립 · 각기 다른 장면
          → 클립 하나 = 판정 하나 · **어떤 장면에서 틀렸나**

특히 `fallen`(바닥에 넘어짐=위험) ↔ `lying`(침대에 누움=정상) 혼동 사례를
따로 모은다. FPR 문제의 실체를 눈으로 확인하기 위해서다.

사용법:
    python make_testset_report.py --kps kps.npz --manifest manifest.csv \
        --model models/gru.pt --root <영상루트> --out runs/report
"""
from __future__ import annotations
import argparse, base64, collections, csv, html, json
from pathlib import Path
import numpy as np
import cv2

LBL = {0:"walk",1:"fall",2:"fallen",3:"sit_down",4:"sitting",5:"lie_down",
       6:"lying",7:"stand_up",8:"standing",9:"other"}
RISK = {"normal":[0,3,4,5,6,7,8,9], "fall":[1], "fallen":[2]}


def thumbs(video: Path, n: int, width: int):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for i in range(n):
        idx = round(i * (total - 1) / max(n - 1, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok: continue
        h = int(fr.shape[0] * width / fr.shape[1])
        fr = cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok: out.append((idx, base64.b64encode(buf).decode()))
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kps", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="클립 경로 기준 디렉토리")
    ap.add_argument("--out", type=Path, default=Path("runs/report"))
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--width", type=int, default=260)
    ap.add_argument("--max-cards", type=int, default=90, help="카드 최대 수 (오답 우선)")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_stage2 import load, TASKS
    from pipeline_core import build_model, pick_device

    dev = pick_device()
    ck = torch.load(args.model, map_location=dev, weights_only=False)
    names, seq_len, feat = ck["classes"], ck["seq_len"], ck.get("feat", "pos")
    model = build_model(ck["kind"], len(names), dev, ck.get("in_ch", 3))
    model.load_state_dict(ck["state_dict"]); model.eval()

    X, y, groups, cams, dsets, _ = load(args.kps, args.manifest, seq_len, "risk", feat)
    # 피험자 단위 test split — train_stage2 와 같은 seed
    rng = np.random.default_rng(0)
    uniq = np.array(sorted(set(groups))); rng.shuffle(uniq)
    n_te = max(1, int(len(uniq) * 0.15))
    te = np.isin(groups, uniq[:n_te])
    print(f"[*] test {te.sum()}개 (피험자 {n_te}명)")

    with torch.no_grad():
        prob = torch.softmax(model(torch.tensor(X[te]).to(dev)), 1).cpu().numpy()
    pred = prob.argmax(1)
    yt = y[te]

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    uid2row = {f"{r['dataset']}__{Path(r['clip']).stem}": r for r in rows}
    # load() 와 같은 순서로 uid 를 재구성
    uids = []
    pack = np.load(args.kps) if args.kps.suffix == ".npz" else None
    for r in rows:
        uid = f"{r['dataset']}__{Path(r['clip']).stem}"
        if pack is not None and uid not in pack: continue
        # load() 는 2프레임 미만 클립을 건너뛴다 — 같은 필터를 적용해 순서를 맞춘다
        if pack is not None and pack[uid].shape[0] < 2: continue
        uids.append(uid)
    uids = np.array(uids)[te]

    items = []
    for i, uid in enumerate(uids):
        r = uid2row[uid]
        items.append(dict(uid=uid, ds=r["dataset"], clip=r["clip"],
                          true=names[yt[i]], pred=names[pred[i]],
                          orig=LBL.get(int(r["label"]), "?"),
                          prob=prob[i].round(3).tolist(), ok=bool(yt[i]==pred[i])))

    # 오답 우선 정렬 — fallen↔lying 혼동을 맨 앞에
    def rank(it):
        if not it["ok"] and it["orig"] in ("lying","lie_down") and it["pred"]!="normal": return 0
        if not it["ok"] and it["true"]=="fallen": return 1
        if not it["ok"]: return 2
        return 3
    items.sort(key=rank)
    show = items[:args.max_cards]

    print(f"[*] 썸네일 {len(show)}개 추출")
    th = {}
    for i, it in enumerate(show, 1):
        v = args.root / it["clip"]
        if v.exists(): th[it["uid"]] = thumbs(v, args.frames, args.width)
        print(f"  [{i}/{len(show)}]", end="\r")
    print()

    acc = float((yt == pred).mean())
    cm = np.zeros((len(names), len(names)), int)
    for t, p in zip(yt, pred): cm[t, p] += 1
    ri = [names.index("fall"), names.index("fallen")]
    ni = [i for i in range(len(names)) if i not in ri]
    TP = cm[np.ix_(ri,ri)].sum(); FN = cm[np.ix_(ri,ni)].sum()
    FP = cm[np.ix_(ni,ri)].sum(); TN = cm[np.ix_(ni,ni)].sum()
    rec, fpr = TP/max(TP+FN,1), FP/max(FP+TN,1)

    # lying/lie_down 원본 라벨이 위험으로 오판된 수
    lie_fp = sum(1 for it in items if it["orig"] in ("lying","lie_down") and it["pred"]!="normal")
    lie_all = sum(1 for it in items if it["orig"] in ("lying","lie_down"))

    E = html.escape
    P = [f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>테스트셋 판정 — OmniFall</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--card:#fafafa;
--ok:#0a7d33;--ok-bg:#e8f6ec;--bad:#c62828;--bad-bg:#fdeaea;--accent:#1a4f8a;--warn:#b26a00;--warn-bg:#fff4e0}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#16181c;--fg:#e8e8e8;
--muted:#9aa0a6;--line:#2e3238;--card:#1e2126;--ok:#6ee7a0;--ok-bg:#12331f;--bad:#ff8a8a;
--bad-bg:#3a1d1d;--accent:#7fb3f0;--warn:#e0a955;--warn-bg:#3a2e18}}}}
:root[data-theme="dark"]{{--bg:#16181c;--fg:#e8e8e8;--muted:#9aa0a6;--line:#2e3238;--card:#1e2126;
--ok:#6ee7a0;--ok-bg:#12331f;--bad:#ff8a8a;--bad-bg:#3a1d1d;--accent:#7fb3f0;--warn:#e0a955;--warn-bg:#3a2e18}}
*{{box-sizing:border-box}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Segoe UI",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 4px}}
h2{{font-size:19px;margin:32px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--line)}}
.sub{{color:var(--muted);margin-bottom:20px;font-size:14px}}
.stats{{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 18px;min-width:112px;flex:1}}
.stat .v{{font-size:25px;font-weight:700;line-height:1.2}}
.stat .l{{font-size:12px;color:var(--muted);margin-top:2px}}
table{{border-collapse:collapse;margin:12px 0;font-size:14px}}
th,td{{border:1px solid var(--line);padding:7px 12px;text-align:center}}
th{{background:var(--card);font-weight:600}}
td.diag{{background:var(--ok-bg);font-weight:700}} td.off{{background:var(--bad-bg)}}
.card{{border:1px solid var(--line);border-radius:12px;margin:12px 0;overflow:hidden;background:var(--card)}}
.card.wrong{{border-color:var(--bad);border-width:2px}}
.card.lie{{border-color:var(--warn);border-width:2px}}
.hd{{display:flex;flex-wrap:wrap;align-items:center;gap:9px;padding:10px 14px;border-bottom:1px solid var(--line)}}
.fn{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;font-weight:600}}
.badge{{font-size:12px;padding:3px 10px;border-radius:99px;font-weight:600;white-space:nowrap}}
.b-ok{{background:var(--ok-bg);color:var(--ok)}} .b-bad{{background:var(--bad-bg);color:var(--bad)}}
.b-n{{background:var(--line);color:var(--fg)}} .b-w{{background:var(--warn-bg);color:var(--warn)}}
.frames{{display:flex;gap:0;overflow-x:auto;background:#000}}
.frames figure{{margin:0;flex:0 0 auto;position:relative}}
.frames img{{display:block;height:auto}}
.frames figcaption{{position:absolute;left:0;bottom:0;background:rgba(0,0,0,.7);color:#fff;
font-size:10px;padding:2px 6px;font-family:ui-monospace,monospace}}
.prob{{padding:8px 14px;font-size:13px;display:flex;flex-wrap:wrap;gap:14px;align-items:center;border-top:1px solid var(--line)}}
.bar{{display:inline-block;height:7px;border-radius:4px;background:var(--accent);vertical-align:middle;margin:0 5px}}
.note{{background:var(--card);border-left:3px solid var(--accent);padding:11px 15px;margin:14px 0;border-radius:0 8px 8px 0;font-size:14px}}
.note.warn{{border-left-color:var(--warn)}}
@media print{{:root{{--bg:#fff;--fg:#111;--muted:#555;--line:#ccc;--card:#f7f7f7;
--ok:#0a7d33;--ok-bg:#e8f6ec;--bad:#c62828;--bad-bg:#fdeaea;--accent:#1a4f8a;--warn:#b26a00;--warn-bg:#fff4e0}}
body{{padding:0;font-size:11pt}} .wrap{{max-width:none}} h2{{page-break-after:avoid}}
.card,.stat,table,.note{{page-break-inside:avoid}}
.frames{{overflow:visible;display:flex}} .frames figure{{flex:1 1 0;min-width:0}}
.frames img{{width:100%;height:auto}}}}
@page{{size:A4;margin:12mm 10mm}}
</style></head><body><div class="wrap">"""]

    P.append("<h1>테스트셋 판정 결과 — OmniFall</h1>")
    P.append(f'<div class="sub">YOLOv11-pose → 키포인트 → {E(ck["kind"])} · '
             f'test {int(te.sum())}클립 (피험자 단위 분할)</div>')
    P.append('<div class="stats">')
    for v, l in [(f"{acc:.1%}","Accuracy"), (f"{rec:.1%}","Recall (위험)"),
                 (f"{fpr:.1%}","FPR"), (str(int(FN)),"놓친 위험"),
                 (f"{lie_fp}/{lie_all}","lying 오판")]:
        P.append(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>')
    P.append("</div>")

    P.append("<h2>Confusion Matrix</h2><table><tr><th>실제 \\ 예측</th>")
    for n in names: P.append(f"<th>{E(n)}</th>")
    P.append("</tr>")
    for i, n in enumerate(names):
        P.append(f"<tr><th>{E(n)}</th>")
        for j in range(len(names)):
            P.append(f'<td class="{"diag" if i==j else ("off" if cm[i,j] else "")}">{cm[i,j]}</td>')
        P.append("</tr>")
    P.append("</table>")

    if lie_all:
        P.append(f'<div class="note warn"><b>lying/lie_down 오판 {lie_fp}/{lie_all}건.</b> '
                 f'침대·소파에 누운 정상 자세를 위험으로 본 사례다. AI Hub 샘플에서 '
                 f'FPR 0.750 을 만든 것과 같은 원인이며, 아래 노란 테두리 카드에서 확인할 수 있다.</div>')

    P.append("<h2>판정 사례</h2>")
    P.append('<div class="sub">오답을 앞에 배치했다. 노란 테두리는 '
             'lying/lie_down 을 위험으로 오판한 사례, 빨간 테두리는 그 밖의 오답이다.</div>')
    for it in show:
        lie = (not it["ok"]) and it["orig"] in ("lying","lie_down")
        cls = " lie" if lie else ("" if it["ok"] else " wrong")
        P.append(f'<div class="card{cls}">')
        P.append(f'<div class="hd"><span class="fn">{E(it["uid"][:56])}</span>'
                 f'<span class="badge b-n">{E(it["ds"])}</span>'
                 f'<span class="badge b-n">원본 {E(it["orig"])}</span>'
                 f'<span class="badge b-n">실제 {E(it["true"])}</span>'
                 f'<span class="badge {"b-ok" if it["ok"] else ("b-w" if lie else "b-bad")}">'
                 f'예측 {E(it["pred"])} {"✓" if it["ok"] else "✗"}</span></div>')
        fr = th.get(it["uid"], [])
        if fr:
            P.append('<div class="frames">')
            for idx, b64 in fr:
                P.append(f'<figure><img src="data:image/jpeg;base64,{b64}" '
                         f'width="{args.width}" alt="frame {idx}">'
                         f'<figcaption>{idx}f</figcaption></figure>')
            P.append("</div>")
        P.append('<div class="prob">')
        for nm, pv in zip(names, it["prob"]):
            P.append(f'<span>{E(nm)} <b>{pv:.2f}</b>'
                     f'<span class="bar" style="width:{max(2,round(pv*70))}px"></span></span>')
        P.append("</div></div>")

    P.append(f'<div class="note">전체 test {int(te.sum())}개 중 {len(show)}개를 표시했다 '
             f'(오답 우선). 전체 수치는 상단 지표와 confusion matrix 를 참고한다.</div>')
    P.append("</div></body></html>")

    args.out.mkdir(parents=True, exist_ok=True)
    f = args.out / "testset_omnifall.html"
    f.write_text("".join(P), encoding="utf-8")
    print(f"[*] HTML {f} ({f.stat().st_size/1e6:.1f}MB)")
    if args.pdf:
        try:
            from weasyprint import HTML
            p = f.with_suffix(".pdf")
            HTML(filename=str(f)).write_pdf(str(p))
            print(f"[*] PDF {p} ({p.stat().st_size/1e6:.1f}MB)")
        except Exception as e:
            print(f"[!] PDF 실패: {e}")
    print(f"\nAcc {acc:.3f} · Recall {rec:.3f} · FPR {fpr:.3f} · lying 오판 {lie_fp}/{lie_all}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
