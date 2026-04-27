"""
Export Keras .h5 model to TFLite (chạy local hoặc Colab, cùng thư mục gốc = backend).

Ví dụ từ thư mục backend:
  python scripts/export_tflite.py --input saved_model/xray_model.h5 --output saved_model/xray_model.tflite

Nếu cần INT8: thêm representative dataset vào TFLiteConverter;
model float thường tương thích đơn giản hơn khi triển khai lần đầu.
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running as `python scripts/export_tflite.py` from repo root
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import tensorflow as tf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="saved_model/xray_model.h5", help="Keras .h5 or .keras")
    p.add_argument("--output", default="saved_model/xray_model.tflite", help="Output .tflite path")
    p.add_argument("--default-opt", action="store_true", help="tf.lite.Optimize.DEFAULT (smaller, may be int8)")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Not found: {args.input}")

    model = tf.keras.models.load_model(args.input)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.default_opt:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = conv.convert()
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(tflite_model)
    print("Wrote", args.output, "size", os.path.getsize(args.output) // 1024, "KB")


if __name__ == "__main__":
    main()
