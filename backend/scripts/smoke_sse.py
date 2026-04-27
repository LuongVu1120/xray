"""Gọi /agent/diagnose qua HTTP SSE để xác minh streaming."""
from __future__ import annotations

import io
import os
import sys

import httpx
import numpy as np
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = os.getenv("AGENT_URL", "http://localhost:8000/agent/diagnose")


def _make_fake_xray() -> bytes:
    arr = (np.random.RandomState(7).rand(512, 512, 3) * 200 + 30).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main() -> None:
    files = {"file": ("fake.jpg", _make_fake_xray(), "image/jpeg")}
    data = {
        "patient": '{"age": 60, "sex": "male", "symptoms": "ho keo dai"}',
        "use_llm": "false",
        "use_uncertainty": "true",
        "use_pubmed": "false",
        "use_tta": "true",
    }
    n_events = 0
    with httpx.Client(timeout=120) as client:
        with client.stream("POST", URL, files=files, data=data) as resp:
            print("HTTP", resp.status_code)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                n_events += 1
                preview = payload[:160]
                print(preview + ("…" if len(payload) > 160 else ""))
    print(f"\nTotal events: {n_events}")


if __name__ == "__main__":
    main()
