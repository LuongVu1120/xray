"""
Tool registry cho agent. Mỗi tool là 1 hàm thuần (sync/async) +
metadata để có thể nâng cấp lên function-calling LLM sau này.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import cv2
import numpy as np

from model.gradcam import (
    demo_gradcam_class_index,
    generate_demo_gradcam,
    generate_gradcam,
    generate_gradcam_pp,
)
from model.predict import (
    InferenceMode,
    multilabel_findings,
    preprocess_image,
    run_inference,
    run_tflite_inference,
    run_tta_inference,
)
from model.uncertainty import predict_with_mc_dropout

from .knowledge import lookup_pathology_info
from .pubmed import search_pubmed

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict = field(default_factory=dict)


REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    REGISTRY[spec.name] = spec


def _encode_jpeg_b64(bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}"


# ── Tool: classify ──────────────────────────────────────────────────────────


def tool_classify(
    *,
    img_bytes: bytes,
    keras_model=None,
    tflite_interpreter=None,
    tflite_in=None,
    tflite_out=None,
    use_tta: bool = True,
) -> dict:
    """Chạy inference (Keras → TTA, TFLite → 1 lượt). Trả scores + labels + mode."""
    arr = preprocess_image(img_bytes)
    if keras_model is not None:
        if use_tta:
            pred_idx, scores, labels, mode = run_tta_inference(keras_model, arr)
        else:
            pred_idx, scores, labels, mode = run_inference(keras_model, arr)
    elif tflite_interpreter is not None:
        pred_idx, scores, labels, mode = run_tflite_inference(
            tflite_interpreter, tflite_in, tflite_out, arr
        )
    else:
        from model.predict import run_demo_inference

        pred_idx, scores, labels, mode = run_demo_inference()

    findings: list[dict] = []
    if mode == InferenceMode.MULTILABEL_14:
        findings = multilabel_findings(scores, labels)

    return {
        "img_array": arr,
        "mode": mode,
        "labels": labels,
        "scores": scores,
        "top_idx": pred_idx,
        "top_label": labels[pred_idx] if pred_idx < len(labels) else labels[0],
        "top_score": float(scores[pred_idx]) if pred_idx < len(scores) else 0.0,
        "findings": findings,
    }


register(
    ToolSpec(
        name="classify_xray",
        description="Phân loại ảnh X-quang (binary / softmax3 / multi-label NIH-14).",
        handler=tool_classify,
        parameters={"img_bytes": "bytes", "use_tta": "bool"},
    )
)


# ── Tool: uncertainty ───────────────────────────────────────────────────────


def tool_uncertainty(*, keras_model, img_array: np.ndarray, n_samples: int = 8) -> dict:
    if keras_model is None:
        return {"available": False, "reason": "no Keras model loaded"}
    try:
        out = predict_with_mc_dropout(keras_model, img_array, n_samples=n_samples)
        out["available"] = True
        return out
    except Exception as e:
        logger.warning("MC-Dropout failed: %s", e)
        return {"available": False, "reason": str(e)}


register(
    ToolSpec(
        name="uncertainty_mc_dropout",
        description="Ước lượng bất định bằng MC-Dropout (cần model Keras).",
        handler=tool_uncertainty,
        parameters={"img_array": "np.ndarray", "n_samples": "int"},
    )
)


# ── Tool: heatmap ───────────────────────────────────────────────────────────


def tool_heatmap(
    *,
    img_array: np.ndarray,
    class_idx: int,
    num_classes: int,
    keras_model=None,
    use_pp: bool = True,
) -> dict:
    """Sinh Grad-CAM (++ nếu có model Keras), demo overlay nếu không."""
    try:
        if keras_model is not None:
            if use_pp:
                bgr, severity = generate_gradcam_pp(keras_model, img_array, class_idx)
                method = "grad-cam++"
            else:
                bgr, severity = generate_gradcam(keras_model, img_array, class_idx)
                method = "grad-cam"
        else:
            cam_idx = demo_gradcam_class_index(class_idx, num_classes)
            bgr, severity = generate_demo_gradcam(img_array, cam_idx)
            method = "demo-overlay"
    except Exception as e:
        logger.warning("Grad-CAM failed (%s); falling back to demo overlay", e)
        cam_idx = demo_gradcam_class_index(class_idx, num_classes)
        bgr, severity = generate_demo_gradcam(img_array, cam_idx)
        method = "demo-overlay"

    return {
        "heatmap_b64": _encode_jpeg_b64(bgr),
        "severity": float(severity),
        "method": method,
    }


register(
    ToolSpec(
        name="heatmap_gradcam",
        description="Tạo heatmap Grad-CAM++ cho class chỉ định.",
        handler=tool_heatmap,
        parameters={"img_array": "np.ndarray", "class_idx": "int"},
    )
)


# ── Tool: lookup ────────────────────────────────────────────────────────────


register(
    ToolSpec(
        name="lookup_pathology_info",
        description="Tra cứu mô tả ngắn + bước tiếp theo cho 1 nhãn bệnh.",
        handler=lookup_pathology_info,
        parameters={"label": "str"},
    )
)


# ── Tool: pubmed ────────────────────────────────────────────────────────────


async def tool_pubmed(*, query: str, max_results: int = 3) -> list[dict]:
    articles = await search_pubmed(query, max_results=max_results)
    return [a.model_dump() for a in articles]


register(
    ToolSpec(
        name="search_pubmed",
        description="Tra cứu bài báo PubMed liên quan (NCBI E-utilities, free).",
        handler=tool_pubmed,
        parameters={"query": "str", "max_results": "int"},
    )
)


def list_tools() -> list[dict]:
    return [
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in REGISTRY.values()
    ]
