"""
Smoke test agent pipeline trên ảnh giả lập (no model loaded).
Chạy: python scripts/smoke_agent.py
"""
from __future__ import annotations

import asyncio
import io
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from agent.orchestrator import run_diagnostic_pipeline  # noqa: E402
from agent.schemas import PatientContext  # noqa: E402


def _make_fake_xray() -> bytes:
    img = (np.random.RandomState(0).rand(512, 512, 3) * 200 + 30).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def main() -> None:
    img = _make_fake_xray()
    patient = PatientContext(age=65, sex="male", symptoms="ho kéo dài 3 tuần, sốt nhẹ")

    async for evt in run_diagnostic_pipeline(
        img_bytes=img,
        patient=patient,
        keras_model=None,
        tflite_interpreter=None,
        use_tta=False,
        use_uncertainty=False,
        use_pubmed=True,
        use_llm=False,
    ):
        d = evt.data or {}
        if evt.step == "report" and evt.status == "done":
            print(f"\n=== {evt.step} ({evt.status}) ===")
            print(d.get("text", "")[:500], "...\n")
        else:
            preview = {k: v for k, v in d.items() if k != "heatmap"}
            print(f"[{evt.step}] {evt.status} :: {preview if preview else (evt.message or '')}")


if __name__ == "__main__":
    asyncio.run(main())
