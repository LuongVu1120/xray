"""
Orchestrator: pipeline xác định + bước cuối tổng hợp báo cáo (LLM nếu có).

Lý do không dùng tool-calling LLM:
- Không phụ thuộc API key/cost,
- Bước nào cũng cần chạy → deterministic chuỗi rẻ và predictable hơn,
- LLM tận dụng đúng việc nó giỏi: tổng hợp ngôn ngữ.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from .knowledge import lookup_pathology_info
from .llm import build_report_messages, fallback_report, llm_available, stream_chat
from .schemas import AgentEvent, PatientContext
from .tools import tool_classify, tool_heatmap, tool_pubmed, tool_uncertainty

logger = logging.getLogger(__name__)


def _event(step: str, status: str, data: Optional[dict] = None, message: Optional[str] = None) -> AgentEvent:
    return AgentEvent(step=step, status=status, data=data, message=message)


async def run_diagnostic_pipeline(
    *,
    img_bytes: bytes,
    patient: PatientContext,
    keras_model: Any = None,
    tflite_interpreter: Any = None,
    tflite_in: Any = None,
    tflite_out: Any = None,
    use_tta: bool = True,
    use_uncertainty: bool = True,
    use_pubmed: bool = True,
    use_llm: bool = True,
) -> AsyncIterator[AgentEvent]:
    """Generator phát ra AgentEvent. Caller dùng để stream qua SSE."""

    # Bước 1: classify
    yield _event("classify", "started", message="Phân loại ảnh X-quang…")
    classify = await asyncio.to_thread(
        tool_classify,
        img_bytes=img_bytes,
        keras_model=keras_model,
        tflite_interpreter=tflite_interpreter,
        tflite_in=tflite_in,
        tflite_out=tflite_out,
        use_tta=use_tta,
    )
    img_array = classify.pop("img_array")
    yield _event(
        "classify",
        "done",
        data={
            "mode": classify["mode"],
            "labels": classify["labels"],
            "scores": classify["scores"],
            "top_label": classify["top_label"],
            "top_score": classify["top_score"],
            "findings": classify["findings"],
        },
    )

    # Bước 2: uncertainty (chỉ với Keras model)
    uncertainty: Optional[dict] = None
    if use_uncertainty and keras_model is not None:
        yield _event("uncertainty", "started", message="Ước lượng bất định MC-Dropout…")
        uncertainty = await asyncio.to_thread(
            tool_uncertainty, keras_model=keras_model, img_array=img_array, n_samples=8
        )
        yield _event("uncertainty", "done", data=uncertainty)

    # Bước 3: heatmap
    yield _event("heatmap", "started", message="Sinh Grad-CAM++…")
    heatmap = await asyncio.to_thread(
        tool_heatmap,
        img_array=img_array,
        class_idx=classify["top_idx"],
        num_classes=len(classify["labels"]),
        keras_model=keras_model,
        use_pp=True,
    )
    yield _event(
        "heatmap",
        "done",
        data={
            "heatmap": heatmap["heatmap_b64"],
            "severity": heatmap["severity"],
            "method": heatmap["method"],
        },
    )

    # Bước 4: tra cứu kiến thức
    yield _event("knowledge", "started", message="Tra cứu thông tin bệnh…")
    info = lookup_pathology_info(classify["top_label"])
    yield _event("knowledge", "done", data=info)

    # Bước 5: PubMed
    articles: list[dict] = []
    if use_pubmed:
        yield _event("pubmed", "started", message="Tìm bài báo PubMed liên quan…")
        query_terms = [classify["top_label"]]
        if patient.symptoms:
            query_terms.append(patient.symptoms[:80])
        if patient.age:
            query_terms.append(f"{patient.age}yo")
        articles = await tool_pubmed(query=" ".join(query_terms), max_results=3)
        yield _event("pubmed", "done", data={"articles": articles})

    # Bước 6: tổng hợp báo cáo
    payload = {
        "top_label": classify["top_label"],
        "top_confidence_pct": round(classify["top_score"] * 100, 1),
        "all_scores": {
            label: round(float(s) * 100, 1)
            for label, s in zip(classify["labels"], classify["scores"])
        },
        "multilabel_findings": classify["findings"],
        "uncertainty": uncertainty,
        "pathology_info": info,
        "pubmed": articles,
        "patient": patient.model_dump(exclude_none=True),
        "heatmap_severity_pct": round(heatmap["severity"] * 100, 1),
        "heatmap_method": heatmap["method"],
    }

    yield _event("report", "started", message="Soạn báo cáo y khoa…")

    if use_llm and llm_available():
        try:
            buffer = []
            async for delta in stream_chat(build_report_messages(payload)):
                buffer.append(delta)
                yield _event("report", "delta", data={"text": delta})
            full_text = "".join(buffer).strip() or fallback_report(payload)
            yield _event(
                "report",
                "done",
                data={"text": full_text, "source": "llm", "payload": payload},
            )
            return
        except Exception as e:
            logger.warning("LLM streaming failed (%s); fallback template", e)
            yield _event("report", "delta", data={"text": ""}, message=f"LLM lỗi, dùng template: {e}")

    text = fallback_report(payload)
    yield _event("report", "done", data={"text": text, "source": "template", "payload": payload})
