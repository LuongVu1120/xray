import io
import random
from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

LABELS = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to model-ready numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def _softmax_2d(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-8)


def run_inference(model: tf.keras.Model, img_array: np.ndarray) -> tuple[int, list[float]]:
    """
    Run model inference and return (predicted_class_idx, all_scores).
    """
    predictions = model.predict(img_array, verbose=0)[0]
    if len(predictions) == 1:
        p = float(predictions[0])
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return int(p > 0.5), [1.0 - p, p]
    pred_idx = int(np.argmax(predictions))
    return pred_idx, predictions.tolist()


def run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    in_details: list[dict[str, Any]],
    out_details: list[dict[str, Any]],
    img_array: np.ndarray,
) -> tuple[int, list[float]]:
    """
    TFLite inference. Supports float and typical int8 quantized I/O; output (N,) logits or prob.
    """
    inp, out_d = in_details[0], out_details[0]
    x = np.asarray(img_array, dtype=np.float32, order="C")

    if np.dtype(inp["dtype"]) == np.float32:
        in_t = x
    else:
        sc, z = inp.get("quantization", (0.0, 0)) or (0, 0)
        if not sc or abs(float(sc)) < 1e-30:
            in_t = x.astype(inp["dtype"])
        else:
            in_t = np.round(x / float(sc) + float(z or 0)).clip(-128, 127).astype(inp["dtype"])

    interpreter.set_tensor(inp["index"], in_t)
    interpreter.invoke()
    raw = np.asarray(interpreter.get_tensor(out_d["index"]))

    if np.dtype(out_d["dtype"]) == np.float32:
        out = raw.astype(np.float64).ravel()
    else:
        sc, z = out_d.get("quantization", (0.0, 0)) or (0, 0)
        if sc and abs(float(sc)) > 1e-30:
            out = (raw.astype(np.float32).ravel() - float(z or 0)) * float(sc)
        else:
            out = raw.astype(np.float64).ravel()

    n = int(out.shape[0])
    if n == 1:
        logit = float(out[0])
        p = float(1.0 / (1.0 + np.exp(-logit)))
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return int(p > 0.5), [1.0 - p, p]
    probs = out.astype(np.float64)
    if (probs < 0).any() or (probs > 1.01).any() or not np.isclose(probs.sum(), 1.0, atol=0.1):
        probs = _softmax_2d(probs)
    pred_idx = int(np.argmax(probs))
    return pred_idx, [float(p) for p in probs.tolist()]


def run_demo_inference() -> tuple[int, list[float]]:
    """
    Return realistic demo predictions when no model is loaded (DEMO_MODE=true).
    """
    scenarios: list[Tuple[int, list[float]]] = [
        (0, [0.88, 0.12]),
        (1, [0.12, 0.88]),
    ]
    idx, base = random.choice(scenarios)
    noise = [random.uniform(-0.03, 0.03) for _ in range(2)]
    scores = [max(0.01, min(0.99, base[i] + noise[i])) for i in range(2)]
    t = sum(scores)
    scores = [s / t for s in scores]
    return idx, scores


def get_recommendation(label: str, confidence: float) -> str:
    u = (label or "").upper()
    if u == "NORMAL":
        return "Ảnh X-quang không phát hiện dấu hiệu bất thường rõ ràng. Nên khám định kỳ theo lịch."
    if u == "PNEUMONIA":
        if confidence > 0.80:
            return "Phát hiện dấu hiệu nghi ngờ viêm phổi với độ tin cậy cao. Cần tư vấn bác sĩ chuyên khoa ngay."
        return "Phát hiện một số dấu hiệu nghi ngờ. Nên thực hiện thêm xét nghiệm và tham khảo ý kiến bác sĩ."
    return "Phát hiện bất thường chưa xác định rõ loại bệnh. Cần thực hiện thêm kiểm tra chuyên sâu."
