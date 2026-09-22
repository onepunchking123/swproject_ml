#!/usr/bin/env python3
"""Stage 2 — 키포인트 시퀀스로 낙상/위험상태를 분류한다.

입력은 extract_keypoints.py 가 만든 (T, 17, 3) 배열이다. 클립 길이가 제각각이므로
고정 길이 윈도우로 리샘플해 배치를 만든다.

여러 모델을 같은 데이터·같은 split 으로 학습해 비교표를 만드는 것이 목적이다.
`--model` 로 고르고, `--all` 이면 전부 순차 학습한다.

    rule     규칙 기반 (학습 없음) — 몸통 각도 + 하강 속도. 필수 베이스라인
    cnn1d    1D-CNN
    lstm     LSTM
    gru      GRU
    bilstm   BiLSTM + Attention
    stgcn    ST-GCN (골격 그래프)

split 은 **피험자 단위**다. 같은 사람의 다른 클립이 train/test 에 갈리면
체형·습관이 누수되어 성능이 부풀려진다.

사용법:
    python train_stage2.py --kps kps --manifest manifest.csv --all
    python train_stage2.py --kps kps --manifest manifest.csv --model lstm --epochs 60
    python train_stage2.py ... --task binary      # fall vs 그 외
    python train_stage2.py ... --holdout-cam 7 8  # 카메라 일반화 실험
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

LBL = {0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting",
       5: "lie_down", 6: "lying", 7: "stand_up", 8: "standing", 9: "other"}

# 본 연구의 관심 대상. fallen 과 lying 을 반드시 구분해야 한다 —
# 바닥에 넘어진 채 미동(위험) vs 침대에 누워 미동(정상).
TASKS = {
    "binary":   {"fall": [1], "other": [0, 2, 3, 4, 5, 6, 7, 8, 9]},
    "risk":     {"normal": [0, 3, 4, 5, 6, 7, 8, 9], "fall": [1], "fallen": [2]},
    "full":     None,   # 10클래스 그대로
}


def load(kps_dir: Path, manifest: Path, seq_len: int, task: str):
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    mapping = TASKS[task]
    if mapping:
        to_new = {old: i for i, (_, olds) in enumerate(mapping.items()) for old in olds}
        names = list(mapping)
    else:
        to_new = {i: i for i in range(10)}
        names = [LBL[i] for i in range(10)]

    X, y, groups, cams, dsets = [], [], [], [], []
    missing = 0
    for r in rows:
        uid = f"{r['dataset']}__{Path(r['clip']).stem}"
        f = kps_dir / f"{uid}.npy"
        if not f.exists():
            missing += 1
            continue
        a = np.load(f)                       # (T, 17, 3)
        if a.shape[0] < 2:
            continue
        # 고정 길이로 리샘플 — 낙상은 1~2초로 짧고 fallen 은 길다. 길이 자체가
        # 단서가 되면 안 되므로 모든 클립을 같은 길이로 만든다.
        idx = np.linspace(0, a.shape[0] - 1, seq_len)
        lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
        w = (idx - lo)[:, None, None]
        seq = a[lo] * (1 - w) + a[hi] * w     # 선형 보간
        X.append(seq.astype(np.float32))
        y.append(to_new[int(r["label"])])
        groups.append(f"{r['dataset']}_{r['subject']}")
        cams.append(int(r["cam"]))
        dsets.append(r["dataset"])
    if missing:
        print(f"[!] 키포인트 없음 {missing}개 (추출 중이면 정상)")
    return (np.stack(X), np.array(y), np.array(groups),
            np.array(cams), np.array(dsets), names)


def split_by_subject(groups, y, seed=0, val=0.15, test=0.15):
    """피험자 단위 분할. 클립 단위보다 엄격하다."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    n = len(uniq)
    n_te, n_va = max(1, int(n * test)), max(1, int(n * val))
    te, va, tr = set(uniq[:n_te]), set(uniq[n_te:n_te + n_va]), set(uniq[n_te + n_va:])
    m = lambda s: np.array([g in s for g in groups])
    return m(tr), m(va), m(te)


def metrics(y_true, y_pred, names, fall_idx):
    """Recall 우선. 낙상을 놓치면 사람이 다치고, 오탐은 확인하면 된다."""
    K = len(names)
    cm = np.zeros((K, K), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    out = {"acc": float((y_true == y_pred).mean()), "cm": cm.tolist()}
    tp = cm[fall_idx, fall_idx]
    fn = cm[fall_idx].sum() - tp
    fp = cm[:, fall_idx].sum() - tp
    tn = cm.sum() - tp - fn - fp
    out["recall"] = float(tp / max(tp + fn, 1))
    out["precision"] = float(tp / max(tp + fp, 1))
    out["fpr"] = float(fp / max(fp + tn, 1))
    out["f1"] = float(2 * out["precision"] * out["recall"] /
                      max(out["precision"] + out["recall"], 1e-9))
    return out


# ── 규칙 기반 베이스라인 (학습 없음) ──────────────────────────────────
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def rule_predict(X, fall_idx, other_idx):
    """몸통이 수직→수평으로 급변하고 하강 속도가 크면 낙상.

    정규화 좌표에서 어깨-엉덩이 벡터를 보면 된다. 서 있으면 y 성분이 크고,
    누우면 x 성분이 커진다.
    """
    sho = X[:, :, [L_SH, R_SH], :2].mean(2)      # (N, T, 2)
    hip = X[:, :, [L_HIP, R_HIP], :2].mean(2)
    v = sho - hip
    # |y|/|x| 가 작을수록 수평. 각 프레임의 "수평도"
    horiz = np.abs(v[..., 0]) / (np.abs(v[..., 1]) + 1e-6)
    start = horiz[:, :len(horiz[0]) // 3].mean(1)    # 앞 1/3
    end = horiz[:, -len(horiz[0]) // 3:].mean(1)     # 뒤 1/3
    drop = hip[:, -1, 1] - hip[:, 0, 1]              # 엉덩이 y 변화 (아래로 +)
    is_fall = ((end - start) > 0.8) & (drop > 0.1)
    return np.where(is_fall, fall_idx, other_idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kps", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--task", choices=list(TASKS), default="risk")
    ap.add_argument("--model", default="lstm",
                    choices=["rule", "cnn1d", "lstm", "gru", "bilstm", "stgcn"])
    ap.add_argument("--all", action="store_true", help="모든 모델 순차 학습")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-cam", type=int, nargs="*",
                    help="이 카메라를 test 로 (카메라 일반화 실험)")
    ap.add_argument("--kfold", type=int, metavar="K",
                    help="피험자 K-fold 교차검증 (STEP 2)")
    ap.add_argument("--loo-dataset", action="store_true",
                    help="데이터셋 leave-one-out (STEP 3)")
    ap.add_argument("--out", type=Path, default=Path("runs/stage2"))
    args = ap.parse_args()

    X, y, groups, cams, dsets, names = load(args.kps, args.manifest, args.seq_len, args.task)
    fall_idx = names.index("fall")
    other_idx = 0 if names[0] != "fall" else 1
    print(f"[*] 클립 {len(X):,}개 · 클래스 {names} · 시퀀스 {args.seq_len}프레임")
    print(f"    분포 {dict(Counter(names[i] for i in y))}")

    if args.holdout_cam:
        te = np.isin(cams, args.holdout_cam)
        rest = ~te
        tr_m, va_m, _ = split_by_subject(groups[rest], y[rest], args.seed, val=0.18, test=0.0)
        tr, va = np.zeros_like(te), np.zeros_like(te)
        tr[np.where(rest)[0]] = tr_m
        va[np.where(rest)[0]] = va_m
        print(f"    카메라 홀드아웃: test=C{args.holdout_cam}")
    else:
        tr, va, te = split_by_subject(groups, y, args.seed)
    print(f"    train {tr.sum()} · val {va.sum()} · test {te.sum()}  (피험자 단위 분할)")

    args.out.mkdir(parents=True, exist_ok=True)
    todo = ["rule", "cnn1d", "lstm", "gru", "bilstm", "stgcn"] if args.all else [args.model]

    # 실행할 (split 이름, tr, va, te) 목록을 만든다.
    # Colab 세션이 예고 없이 회수되므로 각 실행마다 결과를 즉시 파일에 쓴다.
    if args.kfold:
        runs = kfold_splits(groups, args.kfold, args.seed)
        tag = f"{args.task}_kfold{args.kfold}"
    elif args.loo_dataset:
        runs = loo_dataset_splits(dsets, groups, args.seed)
        tag = f"{args.task}_loodataset"
    else:
        runs = [("single", tr, va, te)]
        tag = f"{args.task}" + (f"_camholdout{''.join(map(str,args.holdout_cam))}" if args.holdout_cam else "")

    f = args.out / f"results_{tag}.json"
    # 이미 끝난 조합은 건너뛴다 — 세션이 죽어도 이어서 돌릴 수 있다
    done = {}
    if f.exists():
        try:
            done = json.loads(f.read_text(encoding="utf-8")).get("results", {})
            if done:
                print(f"[*] 기존 결과 {len(done)}건 발견 — 건너뛴다")
        except Exception:
            pass
    results = dict(done)

    for split_name, tr_i, va_i, te_i in runs:
        for name in todo:
            key = name if split_name == "single" else f"{name}@{split_name}"
            if key in results:
                continue
            t0 = time.time()
            if name == "rule":
                pred = rule_predict(X[te_i], fall_idx, other_idx)
                m = metrics(y[te_i], pred, names, fall_idx)
            else:
                m = train_nn(name, X, y, tr_i, va_i, te_i, names, fall_idx, args)
            m["seconds"] = round(time.time() - t0, 1)
            m["split"] = split_name
            m["n_test"] = int(te_i.sum())
            results[key] = m
            print(f"[{key}] acc {m['acc']:.3f} · recall {m['recall']:.3f} · "
                  f"prec {m['precision']:.3f} · f1 {m['f1']:.3f} · fpr {m['fpr']:.3f} "
                  f"({m['seconds']}초)", flush=True)
            # 증분 저장 — 여기서 죽어도 여기까지는 남는다
            f.write_text(json.dumps({"task": args.task, "classes": names, "n": len(X),
                                     "results": results}, indent=2), encoding="utf-8")

    summarize(results, todo, names)
    print(f"\n저장: {f}")
    return 0


def kfold_splits(groups, k, seed):
    """피험자를 k 조로 나눈다. 각 조가 한 번씩 test 가 되고, 나머지 중 일부가 val."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    out = []
    for i, fold in enumerate(folds):
        te = np.isin(groups, fold)
        rest = np.array(sorted(set(uniq) - set(fold)))
        n_va = max(1, len(rest) // 6)
        va = np.isin(groups, rest[:n_va])
        tr = np.isin(groups, rest[n_va:])
        out.append((f"fold{i}", tr, va, te))
    return out


def loo_dataset_splits(dsets, groups, seed):
    """데이터셋 하나를 통째로 test 로. 환경이 바뀌어도 되는지 본다."""
    rng = np.random.default_rng(seed)
    out = []
    for ds in sorted(set(dsets)):
        te = dsets == ds
        rest_g = np.array(sorted(set(groups[~te])))
        rng.shuffle(rest_g)
        n_va = max(1, len(rest_g) // 6)
        va = np.isin(groups, rest_g[:n_va]) & ~te
        tr = np.isin(groups, rest_g[n_va:]) & ~te
        out.append((ds, tr, va, te))
    return out


def summarize(results, todo, names):
    """모델별 요약. 여러 split 이면 평균 ± 표준편차."""
    print(f"\n{'모델':12s}{'Acc':>8}{'Recall':>9}{'Prec':>8}{'F1':>8}{'FPR':>8}")
    print("-" * 53)
    for name in todo:
        ms = [m for k, m in results.items() if k == name or k.startswith(f"{name}@")]
        if not ms:
            continue
        if len(ms) == 1:
            m = ms[0]
            print(f"{name:12s}{m['acc']:8.3f}{m['recall']:9.3f}{m['precision']:8.3f}"
                  f"{m['f1']:8.3f}{m['fpr']:8.3f}")
        else:
            a = {k: np.array([m[k] for m in ms]) for k in ("acc", "recall", "precision", "f1", "fpr")}
            print(f"{name:12s}{a['acc'].mean():8.3f}{a['recall'].mean():9.3f}"
                  f"{a['precision'].mean():8.3f}{a['f1'].mean():8.3f}{a['fpr'].mean():8.3f}")
            print(f"{'  ±std':12s}{a['acc'].std():8.3f}{a['recall'].std():9.3f}"
                  f"{a['precision'].std():8.3f}{a['f1'].std():8.3f}{a['fpr'].std():8.3f}")
            for m in ms:
                print(f"    {m['split']:10s} recall {m['recall']:.3f} · fpr {m['fpr']:.3f} "
                      f"· n={m['n_test']}")


def train_nn(kind, X, y, tr, va, te, names, fall_idx, args):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    K, T = len(names), X.shape[1]

    class CNN1D(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(
                nn.Conv1d(51, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, K))
        def forward(s, x): return s.net(x.flatten(2).transpose(1, 2))

    class RNN(nn.Module):
        def __init__(s, cell, bi=False, attn=False):
            super().__init__()
            s.rnn = cell(51, 128, 2, batch_first=True, bidirectional=bi, dropout=0.3)
            s.attn, d = attn, 128 * (2 if bi else 1)
            if attn:
                s.a = nn.Linear(d, 1)
            s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, K))
        def forward(s, x):
            o, _ = s.rnn(x.flatten(2))
            if s.attn:
                w = torch.softmax(s.a(o), 1)
                return s.fc((o * w).sum(1))
            return s.fc(o[:, -1])

    class STGCN(nn.Module):
        """골격 구조를 인접행렬로 명시한다. 관절 간 연결이 학습에 반영된다.

        공간(그래프) 합성곱과 시간 합성곱을 번갈아 쌓는다. 한 층만 쓰고 시간축을
        평균내면 시간 정보가 사라져 한 클래스로 붕괴한다 (합성 데이터에서 실측).
        각 블록 뒤에 BatchNorm 을 둬 학습을 안정시킨다.
        """
        E = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),
             (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

        def __init__(s):
            super().__init__()
            A = torch.eye(17)
            for i, j in s.E:
                A[i, j] = A[j, i] = 1
            A = A / A.sum(1, keepdim=True)
            s.register_buffer("A", A)
            s.gc1, s.bn1 = nn.Linear(3, 64), nn.BatchNorm2d(64)
            s.tc1 = nn.Conv2d(64, 64, (5, 1), padding=(2, 0))
            s.gc2, s.bn2 = nn.Linear(64, 128), nn.BatchNorm2d(128)
            s.tc2 = nn.Conv2d(128, 128, (5, 1), stride=(2, 1), padding=(2, 0))
            s.bn3 = nn.BatchNorm2d(128)
            s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, K))

        def _blk(s, x, gc, bn, tc):
            # x: (B,T,V,C) → 그래프 합성곱 → (B,C,T,V) 로 바꿔 시간 합성곱
            h = torch.relu(gc(torch.einsum("ij,btjc->btic", s.A, x)))
            h = h.permute(0, 3, 1, 2)            # (B,C,T,V)
            h = torch.relu(bn(tc(h)))
            return h.permute(0, 2, 3, 1)         # 다시 (B,T,V,C)

        def forward(s, x):                       # (B,T,17,3)
            h = s._blk(x, s.gc1, s.bn1, s.tc1)
            h = s._blk(h, s.gc2, s.bn2, s.tc2)
            h = s.bn3(h.permute(0, 3, 1, 2))     # (B,C,T,V)
            return s.fc(h.mean((2, 3)))          # 시간·관절 평균

    m = {"cnn1d": CNN1D,
         "lstm": lambda: RNN(nn.LSTM),
         "gru": lambda: RNN(nn.GRU),
         "bilstm": lambda: RNN(nn.LSTM, bi=True, attn=True),
         "stgcn": STGCN}[kind]().to(dev)

    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    # 클래스 불균형 보정 — fall 이 34%, lying 이 1% 인 상태로 그냥 학습하면
    # 소수 클래스를 통째로 무시하는 해로 수렴한다.
    cnt = np.bincount(y[tr], minlength=K).astype(np.float32)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    idx_tr = np.where(tr)[0]
    best, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        m.train()
        np.random.shuffle(idx_tr)
        for s in range(0, len(idx_tr), args.batch):
            b = idx_tr[s:s + args.batch]
            xb, yb = Xt[b].to(dev), yt[b].to(dev)
            opt.zero_grad()
            loss = lossf(m(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sched.step()

        m.eval()
        with torch.no_grad():
            pv = m(Xt[va].to(dev)).argmax(1).cpu().numpy()
        # 낙상 감지가 목적이므로 정확도가 아닌 Recall 로 모델을 고른다
        r = metrics(y[va], pv, names, fall_idx)["recall"]
        if r > best:
            best, best_state, patience = r, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 15:
                break

    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        pt = m(Xt[te].to(dev)).argmax(1).cpu().numpy()
    out = metrics(y[te], pt, names, fall_idx)
    out["params"] = sum(p.numel() for p in m.parameters())
    out["epochs_run"] = ep + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
