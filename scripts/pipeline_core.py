#!/usr/bin/env python3
"""파이프라인 공통 모듈 — 정규화 · 리샘플 · 모델 빌드 · 자세 헬퍼.

`aihub_pipeline_eval.py` 와 `extract_keypoints.py` 에 같은 `normalize()` 가 복사돼 있었다.
프로토타입이 세 번째 복사본을 만들면 나중에 셋이 갈라진다. 여기로 단일화한다.

(`extract_keypoints.py` 는 Colab 에 단독 업로드하는 스크립트라 자기 복사본을 유지한다.
 바꿀 때는 이 파일과 함께 바꿔야 한다.)

좌표 규약 — 모든 함수는 COCO 17 키포인트, 마지막 축 (x, y, conf) 를 가정한다.
정규화 뒤에는 엉덩이 중점이 원점, 몸통(어깨~엉덩이) 길이가 1 이다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# COCO 17 인덱스
NOSE = 0
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12
L_ANK, R_ANK = 15, 16

# 골격 연결 (오버레이·ST-GCN 공용)
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16)]


def normalize(kps: np.ndarray, w: int, h: int) -> np.ndarray:
    """(T, 17, 3) 원본 픽셀 좌표 → 카메라 불변 표현.

    1. 프레임 크기로 나눠 해상도 의존 제거
    2. 엉덩이 중점을 원점으로 (카메라 위치·피사체 위치 무관)
    3. 몸통 길이로 나눠 카메라 거리 무관

    스케일 기준을 몸통으로 잡는 이유: 서 있든 누워 있든 몸통 길이는 거의 일정하다.
    바운딩 박스 높이로 나누면 누웠을 때 값이 폭증한다.
    검출 실패 프레임(전부 0)은 몸통 길이가 0 이므로 1 로 보호해 NaN 을 막는다.
    """
    out = kps.copy().astype(np.float32)
    out[..., 0] /= max(w, 1)
    out[..., 1] /= max(h, 1)
    conf = out[..., 2:3]
    hip = (out[:, L_HIP, :2] + out[:, R_HIP, :2]) / 2
    sho = (out[:, L_SH, :2] + out[:, R_SH, :2]) / 2
    torso = np.linalg.norm(sho - hip, axis=-1, keepdims=True)
    torso = np.where(torso < 1e-3, 1.0, torso)
    out[..., :2] = (out[..., :2] - hip[:, None, :]) / torso[:, None, :]
    out[..., 2:3] = conf
    return out


def resample(seq: np.ndarray, n: int) -> np.ndarray:
    """(T, 17, 3) → (n, 17, 3) 선형 보간. 클립 길이가 단서가 되지 않게 고정 길이로 맞춘다."""
    if seq.shape[0] == 1:
        return np.repeat(seq, n, axis=0).astype(np.float32)
    idx = np.linspace(0, seq.shape[0] - 1, n)
    lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    w = (idx - lo)[:, None, None]
    return (seq[lo] * (1 - w) + seq[hi] * w).astype(np.float32)


# ── 자세 헬퍼 (미동 규칙용) ────────────────────────────────────────
def torso_horizontality(frame: np.ndarray) -> float:
    """정규화된 (17, 3) 한 프레임의 몸통 수평도.

    어깨-엉덩이 벡터의 |x|/|y|. 서 있으면 ≈0, 누우면 크다.
    정규화 좌표에서 계산하므로 카메라에 무관하다.
    """
    v = frame[[L_SH, R_SH], :2].mean(0) - frame[[L_HIP, R_HIP], :2].mean(0)
    return float(abs(v[0]) / (abs(v[1]) + 1e-6))


def motion_energy(window: np.ndarray) -> float:
    """(T, 17, 3) 윈도우 안의 움직임 크기 — 프레임 간 관절 변위의 평균.

    정규화 좌표라 몸통 길이 단위다. 검출 실패 프레임(conf 0)은 제외한다.
    """
    ok = window[..., 2] > 0.05                       # (T, 17)
    xy = window[..., :2]
    d = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)    # (T-1, 17)
    valid = ok[1:] & ok[:-1]
    return float(d[valid].mean()) if valid.any() else 0.0


def detection_rate(window: np.ndarray) -> float:
    """윈도우 안에서 사람이 검출된 프레임 비율."""
    return float((np.abs(window[..., :2]).sum(axis=(1, 2)) > 0).mean())


# ── 모델 ─────────────────────────────────────────────────────────
def build_model(kind: str, K: int, dev: str):
    """train_stage2.py 와 동일한 구조. 체크포인트 로드용."""
    import torch
    import torch.nn as nn

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
            if attn: s.a = nn.Linear(d, 1)
            s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(d, K))
        def forward(s, x):
            o, _ = s.rnn(x.flatten(2))
            if s.attn:
                w = torch.softmax(s.a(o), 1); return s.fc((o * w).sum(1))
            return s.fc(o[:, -1])

    class STGCN(nn.Module):
        def __init__(s):
            super().__init__()
            A = torch.eye(17)
            for i, j in SKELETON: A[i, j] = A[j, i] = 1
            s.register_buffer("A", A / A.sum(1, keepdim=True))
            s.gc1, s.bn1 = nn.Linear(3, 64), nn.BatchNorm2d(64)
            s.tc1 = nn.Conv2d(64, 64, (5, 1), padding=(2, 0))
            s.gc2, s.bn2 = nn.Linear(64, 128), nn.BatchNorm2d(128)
            s.tc2 = nn.Conv2d(128, 128, (5, 1), stride=(2, 1), padding=(2, 0))
            s.bn3 = nn.BatchNorm2d(128)
            s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, K))
        def _blk(s, x, gc, bn, tc):
            h = torch.relu(gc(torch.einsum("ij,btjc->btic", s.A, x)))
            h = h.permute(0, 3, 1, 2); h = torch.relu(bn(tc(h)))
            return h.permute(0, 2, 3, 1)
        def forward(s, x):
            h = s._blk(x, s.gc1, s.bn1, s.tc1)
            h = s._blk(h, s.gc2, s.bn2, s.tc2)
            return s.fc(s.bn3(h.permute(0, 3, 1, 2)).mean((2, 3)))

    return {"cnn1d": CNN1D, "lstm": lambda: RNN(nn.LSTM), "gru": lambda: RNN(nn.GRU),
            "bilstm": lambda: RNN(nn.LSTM, bi=True, attn=True), "stgcn": STGCN}[kind]().to(dev)


def pick_device(pref: str = "auto") -> str:
    """auto → mps(M시리즈) > cuda > cpu."""
    import torch
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_checkpoint(path: Path, dev: str):
    """train_stage2.py --save-model 이 남긴 .pt 를 (model, classes, seq_len) 로 돌려준다."""
    import torch
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = build_model(ck["kind"], len(ck["classes"]), dev)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["classes"], ck["seq_len"], ck["kind"]
