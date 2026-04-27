"""
LLM client tương thích OpenAI (Groq / OpenAI / OpenRouter).
- Stream tokens qua SSE.
- Fallback mode (template) khi không có API key.

Env vars:
    LLM_API_KEY            (alias: GROQ_API_KEY hoặc OPENAI_API_KEY)
    LLM_BASE_URL           (default: https://api.groq.com/openai/v1)
    LLM_MODEL              (default: llama-3.3-70b-versatile)
    LLM_TIMEOUT_S          (default: 60)
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


def _get_api_key() -> str | None:
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or None
    )


def llm_available() -> bool:
    return bool(_get_api_key())


def llm_config() -> dict:
    return {
        "base_url": os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        "model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        "timeout": float(os.getenv("LLM_TIMEOUT_S", "60")),
    }


async def stream_chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> AsyncIterator[str]:
    """
    Stream từng token text từ LLM tương thích OpenAI.
    Yield string deltas; raise nếu không có key/model.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("LLM API key not configured")
    cfg = llm_config()
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=cfg["timeout"]) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise RuntimeError(f"LLM HTTP {resp.status_code}: {body[:300]!r}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content


def build_report_messages(payload: dict) -> list[dict]:
    """
    Tạo messages cho LLM compose báo cáo phong cách radiologist.
    payload chứa: top finding, all_scores, multi-label findings, uncertainty,
    pathology info, patient context, pubmed articles.
    """
    system = (
        "Bạn là trợ lý AI hỗ trợ bác sĩ X-quang. "
        "Soạn báo cáo bằng tiếng Việt, ngắn gọn, có cấu trúc 'Findings', "
        "'Impression' và 'Recommendations'. Không bịa số liệu. "
        "Luôn nhắc kết quả AI chỉ tham khảo, không thay thế bác sĩ."
    )
    user = (
        "Dữ liệu phân tích AI (JSON):\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n\nHãy soạn báo cáo theo cấu trúc:\n"
        "## Findings\n"
        "- (mô tả phát hiện chính, dùng số % độ tin cậy đã cho)\n"
        "## Impression\n"
        "- (kết luận ngắn gọn 1–3 dòng)\n"
        "## Recommendations\n"
        "- (đề xuất bước tiếp theo cụ thể, có ưu tiên)\n\n"
        "Cuối báo cáo thêm dòng cảnh báo AI."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def fallback_report(payload: dict) -> str:
    """Báo cáo template không cần LLM — dùng khi thiếu API key."""
    top = payload.get("top_label", "Unknown")
    conf = payload.get("top_confidence_pct", 0)
    findings = payload.get("multilabel_findings") or []
    info = payload.get("pathology_info") or {}
    unc = payload.get("uncertainty") or {}
    next_steps = info.get("next_steps") or []
    common = info.get("common_findings") or []
    articles = payload.get("pubmed") or []
    patient = payload.get("patient") or {}

    lines: list[str] = []
    lines.append("## Findings")
    if findings:
        for f in findings[:6]:
            lines.append(f"- {f['label']}: {f['score'] * 100:.1f}%")
    else:
        lines.append(f"- Phát hiện chính: **{top}** ({conf}%)")
    if common:
        lines.append(f"- Dấu hiệu thường gặp: {', '.join(common)}")
    if unc.get("entropy") is not None:
        lines.append(f"- Độ bất định (entropy MC-Dropout): {unc['entropy']:.3f}")

    lines.append("\n## Impression")
    desc = info.get("description") or "Cần đánh giá thêm bởi bác sĩ chuyên khoa."
    lines.append(f"- {desc}")
    if patient.get("age") or patient.get("sex"):
        lines.append(f"- Bối cảnh BN: {patient.get('age', '?')} tuổi, {patient.get('sex', 'N/A')}.")

    lines.append("\n## Recommendations")
    if next_steps:
        for s in next_steps:
            lines.append(f"- {s}")
    else:
        lines.append("- Hội chẩn bác sĩ X-quang.")
    if articles:
        lines.append("\n### Tài liệu tham khảo (PubMed)")
        for a in articles[:3]:
            lines.append(f"- [{a['title']}]({a['url']})")

    lines.append("\n> ⚠️ Báo cáo do AI sinh ra, chỉ mang tính tham khảo. Không thay thế chẩn đoán của bác sĩ.")
    return "\n".join(lines)
