import io
import random
from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

# Thứ tự khớp `flow_from_directory` (alphabetical): Normal, Other, Pneumonia.
LABELS_THREE = ["Normal", "Other", "Pneumonia"]
LABELS_TWO = ["Normal", "Pneumonia"]

# 14 nhãn NIH ChestX-ray14 (multi-label, sigmoid). Chuẩn theo paper CheXNet.
LABELS_FOURTEEN = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

LABELS = LABELS_THREE
IMG_SIZE = 224
MULTILABEL_THRESHOLD = 0.5


class InferenceMode:
    BINARY = "binary"
    SOFTMAX_3 = "softmax_3"
    MULTILABEL_14 = "multilabel_14"
    SOFTMAX_N = "softmax_n"


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to model-ready numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def _softmax_2d(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-8)


def detect_inference_mode(num_outputs: int) -> str:
    if num_outputs == 1:
        return InferenceMode.BINARY
    if num_outputs == 2:
        return InferenceMode.BINARY
    if num_outputs == 3:
        return InferenceMode.SOFTMAX_3
    if num_outputs == 14:
        return InferenceMode.MULTILABEL_14
    return InferenceMode.SOFTMAX_N


def labels_for_mode(mode: str, num_outputs: int) -> list[str]:
    if mode == InferenceMode.BINARY:
        return LABELS_TWO
    if mode == InferenceMode.SOFTMAX_3:
        return LABELS_THREE
    if mode == InferenceMode.MULTILABEL_14:
        return LABELS_FOURTEEN
    return [f"Class_{i}" for i in range(num_outputs)]


def _normalize_predictions(raw: np.ndarray) -> tuple[str, np.ndarray, list[str]]:
    """
    Trả (mode, scores [0..1] dạng numpy, labels).

    - 1 output: sigmoid binary → mở rộng thành [1-p, p].
    - 2/3 output: softmax đa lớp.
    - 14 output: sigmoid multi-label, mỗi nhãn độc lập.
    - n khác: nếu nhìn giống logit thì softmax.
    """
    arr = np.asarray(raw, dtype=np.float64).ravel()
    n = int(arr.size)
    mode = detect_inference_mode(n)

    if n == 1:
        p = float(arr[0])
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return InferenceMode.BINARY, np.array([1.0 - p, p]), LABELS_TWO

    if mode == InferenceMode.MULTILABEL_14:
        if (arr < 0).any() or (arr > 1.01).any():
            arr = 1.0 / (1.0 + np.exp(-arr))
        return mode, np.clip(arr, 0.0, 1.0), LABELS_FOURTEEN

    # softmax-like
    if (arr < 0).any() or (arr > 1.01).any() or not np.isclose(arr.sum(), 1.0, atol=0.1):
        arr = _softmax_2d(arr)
    labels = labels_for_mode(mode, n)
    return mode, arr, labels


def run_inference(
    model: tf.keras.Model, img_array: np.ndarray
) -> tuple[int, list[float], list[str], str]:
    """
    Returns (predicted_class_idx, score list, labels, mode).
    Dùng `model(x, training=False)` thay `model.predict` để giảm overhead với batch=1.
    """
    raw = model(img_array, training=False).numpy()[0]
    mode, scores, labels_used = _normalize_predictions(raw)
    pred_idx = int(np.argmax(scores))
    return pred_idx, scores.tolist(), labels_used, mode


def run_tta_inference(
    model: tf.keras.Model, img_array: np.ndarray, n_crops: int = 0
) -> tuple[int, list[float], list[str], str]:
    """
    Test-Time Augmentation: trung bình prediction của ảnh gốc + horizontal flip
    (+ tùy chọn vài center-zoom crops). Tăng nhẹ accuracy mà không cần train thêm.
    """
    views = [img_array, img_array[:, :, ::-1, :].copy()]
    if n_crops > 0:
        h, w = img_array.shape[1], img_array.shape[2]
        for k in range(n_crops):
            margin = int(min(h, w) * (0.04 + 0.02 * k))
            crop = img_array[:, margin : h - margin, margin : w - margin, :]
            resized = tf.image.resize(crop, (h, w)).numpy()
            views.append(resized)

    accum: np.ndarray | None = None
    last_raw_size = 1
    for v in views:
        raw = model(v, training=False).numpy()[0]
        last_raw_size = int(np.asarray(raw).size)
        _, probs, _ = _normalize_predictions(raw)
        accum = probs if accum is None else accum + probs

    assert accum is not None
    accum = accum / float(len(views))
    mode = detect_inference_mode(last_raw_size)
    labels_used = labels_for_mode(mode, last_raw_size)
    pred_idx = int(np.argmax(accum))
    return pred_idx, accum.tolist(), labels_used, mode


def run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    in_details: list[dict[str, Any]],
    out_details: list[dict[str, Any]],
    img_array: np.ndarray,
) -> tuple[int, list[float], list[str], str]:
    """TFLite inference. Hỗ trợ float và int8 quantized I/O."""
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

    mode, scores, labels_used = _normalize_predictions(out)
    pred_idx = int(np.argmax(scores))
    return pred_idx, scores.tolist(), labels_used, mode


def run_demo_inference(target_mode: str = InferenceMode.SOFTMAX_3) -> tuple[int, list[float], list[str], str]:
    """Mô phỏng kết quả khi không có model thật (DEMO_MODE=true)."""
    if target_mode == InferenceMode.MULTILABEL_14:
        scores = [0.05] * 14
        focus = random.randint(0, 13)
        scores[focus] = round(random.uniform(0.65, 0.92), 2)
        # cộng thêm một bệnh thường đi kèm
        co = random.choice([i for i in range(14) if i != focus])
        scores[co] = round(random.uniform(0.30, 0.55), 2)
        return focus, scores, LABELS_FOURTEEN, InferenceMode.MULTILABEL_14

    scenarios: list[Tuple[int, list[float]]] = [
        (0, [0.75, 0.15, 0.10]),
        (1, [0.12, 0.70, 0.18]),
        (2, [0.08, 0.12, 0.80]),
    ]
    idx, base = random.choice(scenarios)
    noise = [random.uniform(-0.03, 0.03) for _ in range(3)]
    scores = [max(0.01, min(0.99, base[i] + noise[i])) for i in range(3)]
    t = sum(scores)
    scores = [s / t for s in scores]
    return idx, scores, LABELS_THREE, InferenceMode.SOFTMAX_3


def get_recommendation(label: str, confidence: float) -> str:
    u = (label or "").strip().upper()
    if u == "NORMAL":
        return "Ảnh X-quang không phát hiện dấu hiệu bất thường rõ ràng. Nên khám định kỳ theo lịch."
    if u == "PNEUMONIA":
        if confidence > 0.80:
            return "Phát hiện dấu hiệu nghi ngờ viêm phổi với độ tin cậy cao. Cần tư vấn bác sĩ chuyên khoa ngay."
        return "Phát hiện một số dấu hiệu nghi ngờ. Nên thực hiện thêm xét nghiệm và tham khảo ý kiến bác sĩ."
    if u == "OTHER":
        if confidence > 0.75:
            return (
                "Phát hiện bất thường (nhóm khác, không phải viêm phổi điển hình). "
                "Cần bác sĩ X-quang hoặc chuyên khoa đánh giá thêm."
            )
        return "Có dấu hiệu bất thường chưa phân loại rõ. Nên tái khám hoặc chụp bổ sung nếu bác sĩ chỉ định."
    return "Phát hiện bất thường chưa xác định rõ loại bệnh. Cần thực hiện thêm kiểm tra chuyên sâu."


def multilabel_findings(scores: list[float], labels: list[str], threshold: float = MULTILABEL_THRESHOLD) -> list[dict]:
    """Lọc các nhãn vượt ngưỡng cho chế độ multi-label."""
    out = []
    for label, score in zip(labels, scores):
        if score >= threshold:
            out.append({"label": label, "score": float(score)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out
