"""
Temperature scaling — calibration sau training.

Khi đã có validation set và logits, fit T tối thiểu hoá NLL rồi lưu vào
`saved_model/temperature.json`. Inference sẽ tự nạp T (mặc định 1.0).

Chỉ áp dụng cho softmax đa lớp; multi-label sigmoid có thể dùng Platt scaling
riêng cho từng nhãn (nâng cao, chưa làm ở đây).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

DEFAULT_PATH = "saved_model/temperature.json"


@dataclass
class Temperature:
    value: float = 1.0
    fitted: bool = False

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "Temperature":
        if not os.path.isfile(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(value=float(data.get("temperature", 1.0)), fitted=True)
        except Exception:
            return cls()

    def save(self, path: str = DEFAULT_PATH) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"temperature": float(self.value), "fitted": True}, f)

    def apply_logits(self, logits: np.ndarray) -> np.ndarray:
        """Chia logits cho T rồi softmax."""
        scaled = np.asarray(logits, dtype=np.float64) / max(float(self.value), 1e-3)
        e = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-8)

    def apply_probs(self, probs: np.ndarray) -> np.ndarray:
        """
        Khi chỉ có probabilities (không có logits) — convert ngược thành logits gần đúng
        rồi áp T. Dùng khi mô hình export đã bake softmax.
        """
        p = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0 - 1e-9)
        return self.apply_logits(np.log(p))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: tuple[float, float, float] = (0.5, 5.0, 0.05),
) -> Temperature:
    """
    Fit T qua grid search (đơn giản, đủ cho demo).
    Args:
      logits: (N, C) raw logits trên validation set
      labels: (N,) int class indices
    """
    best_t, best_nll = 1.0, float("inf")
    lo, hi, step = grid
    t = lo
    while t <= hi + 1e-9:
        scaled = logits / t
        e = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = e / (e.sum(axis=1, keepdims=True) + 1e-12)
        nll = -np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean()
        if nll < best_nll:
            best_nll, best_t = nll, t
        t += step
    return Temperature(value=float(best_t), fitted=True)
