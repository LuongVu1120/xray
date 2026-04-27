"""
MC-Dropout uncertainty estimation.

Chạy model nhiều lần với dropout BẬT (training=True) → trung bình prediction +
predictive entropy + standard deviation. Yêu cầu model có Dropout layers
(train.py mặc định có Dropout 0.4 và 0.3).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .predict import _normalize_predictions, detect_inference_mode, labels_for_mode


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) cho phân phối xác suất."""
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())


def _multilabel_entropy(p: np.ndarray) -> float:
    """Average bernoulli entropy cho multi-label (mỗi nhãn độc lập)."""
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(h.mean())


def predict_with_mc_dropout(
    model: tf.keras.Model,
    img_array: np.ndarray,
    n_samples: int = 10,
) -> dict:
    """
    Returns:
      {
        "mean": list[float],          # mean probabilities
        "std": list[float],            # std per class
        "entropy": float,              # uncertainty score (lớn = mơ hồ)
        "labels": list[str],
        "mode": str,
        "n_samples": int,
      }
    """
    samples = []
    last_size = 1
    for _ in range(max(1, n_samples)):
        raw = model(img_array, training=True).numpy()[0]
        last_size = int(np.asarray(raw).size)
        _, probs, _ = _normalize_predictions(raw)
        samples.append(probs)

    arr = np.stack(samples, axis=0)  # (n_samples, n_classes)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    mode = detect_inference_mode(last_size)
    if mode == "multilabel_14":
        ent = _multilabel_entropy(mean)
    else:
        ent = _entropy(mean)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "entropy": float(ent),
        "labels": labels_for_mode(mode, last_size),
        "mode": mode,
        "n_samples": int(n_samples),
    }
