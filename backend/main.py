"""
AI X-Ray Diagnosis System — FastAPI Backend
===========================================
Start server:
    uvicorn main:app --reload --port 8000

Environment variables (.env):
    MODEL_PATH=saved_model/xray_model.h5   # or .tflite
    MODEL_TYPE=auto                        # auto | keras | tflite
    DEMO_MODE=true
    ALLOWED_ORIGINS=http://localhost:3000,https://your-app.vercel.app
    GDRIVE_FILE_ID=                        # optional: download model if missing
    WEB_CONCURRENCY=1                      # hint: use 1 worker on Railway (set in host)

Deploy (Railway / Render): set start command to use a single worker, e.g.
    uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
and increase healthcheck timeout (60s+) if loading DenseNet on cold start.

UptimeRobot: ping GET /ping every ~14 minutes to reduce cold starts on free tiers.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import cv2
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from model.gradcam import generate_gradcam, generate_demo_gradcam
from model.predict import (
    LABELS,
    get_recommendation,
    preprocess_image,
    run_demo_inference,
    run_inference,
    run_tflite_inference,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Render free ~512MB RAM: TensorFlow + DenseNet dễ OOM → 502 ở proxy. Bật trên host: TF_LOW_MEMORY=true
if os.getenv("TF_LOW_MEMORY", "").lower() in ("1", "true", "yes"):
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    logger.info("TF_LOW_MEMORY: giảm song song TF để tiết kiệm RAM")


def _parse_allowed_origins(raw: str) -> list[str]:
    """
    Hỗ trợ nhiều origin phân tách bằng dấu phẩy; bỏ BOM/utf-8-sig nếu file .env từ Windows.
    """
    out: list[str] = []
    for o in raw.split(","):
        o = o.strip().lstrip("\ufeff").strip()
        if o:
            out.append(o)
    return out


MODEL_PATH = os.getenv("MODEL_PATH", "saved_model/xray_model.h5")
MODEL_TYPE = os.getenv("MODEL_TYPE", "auto").strip().lower()
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID", "").strip()
_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = _parse_allowed_origins(_origins)
# Bổ sung: localhost/127.0.0.1 mọi cổng; mọi host *.vercel.app (prod & preview)
_CORS_ORIGIN_RE = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://[a-zA-Z0-9.-]+\.vercel\.app$"

logger.info(
    "CORS allow_origins=%s + regex (local dev, *.vercel.app)",
    ALLOWED_ORIGINS,
)
MAX_FILE_MB = 10

model: Optional[tf.keras.Model] = None
tflite_interpreter: Optional[tf.lite.Interpreter] = None
tflite_in_details: list[dict[str, Any]] = []
tflite_out_details: list[dict[str, Any]] = []
use_tflite: bool = False


def _model_dir() -> str:
    d = os.path.dirname(MODEL_PATH)
    return d if d else "."


def ensure_model_from_gdrive() -> None:
    if not GDRIVE_FILE_ID or os.path.exists(MODEL_PATH):
        return
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError("GDRIVE_FILE_ID is set; install gdown: pip install gdown") from e
    os.makedirs(_model_dir(), exist_ok=True)
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    logger.info("Downloading model from Google Drive to %s ...", MODEL_PATH)
    gdown.download(url, MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        raise RuntimeError("Model download from Google Drive failed or file too small.")


def _resolve_mode(path: str) -> str:
    if MODEL_TYPE == "tflite":
        return "tflite"
    if MODEL_TYPE in ("keras", "h5", "hdf5"):
        return "keras"
    p = (path or "").lower()
    if p.endswith(".tflite"):
        return "tflite"
    if p.endswith(".h5") or p.endswith(".keras"):
        return "keras"
    return "keras"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tflite_interpreter, tflite_in_details, tflite_out_details, use_tflite

    if not DEMO_MODE:
        try:
            ensure_model_from_gdrive()
        except Exception as e:
            logger.error("Model download: %s", e)
            if not os.path.exists(MODEL_PATH):
                logger.warning("Model file missing after download; predictions will use demo path until fixed")

    if not DEMO_MODE and os.path.exists(MODEL_PATH):
        mode = _resolve_mode(MODEL_PATH)
        if mode == "tflite":
            logger.info("Loading TFLite model from %s ...", MODEL_PATH)
            tflite_interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            tflite_interpreter.allocate_tensors()
            tflite_in_details = tflite_interpreter.get_input_details()
            tflite_out_details = tflite_interpreter.get_output_details()
            use_tflite = True
            logger.info("✅ TFLite model ready (Grad-CAM uses saliency-style overlay, not full Grad-CAM)")
        else:
            logger.info("Loading Keras model from %s ...", MODEL_PATH)
            model = tf.keras.models.load_model(MODEL_PATH)
            use_tflite = False
            logger.info("✅ Keras model loaded successfully")
    else:
        if DEMO_MODE:
            logger.warning("⚠️  DEMO_MODE=true — using simulated predictions")
        else:
            logger.warning("⚠️  Model not found at %s — no predictions until file exists", MODEL_PATH)
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="AI X-Ray Diagnosis API",
    description="Phân tích ảnh X-quang phổi bằng Deep Learning (DenseNet121 + Grad-CAM)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=_CORS_ORIGIN_RE,
    # False: axios mặc định không gửi cookie; True đôi khi gây lỗi CORS nếu client/host không đồng bộ
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image(contents: bytes, filename: str):
    """Validate file size and format."""
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File quá lớn. Tối đa {MAX_FILE_MB}MB")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"jpg", "jpeg", "png", "bmp", "tiff", "webp"}:
        raise HTTPException(status_code=415, detail="Định dạng không hỗ trợ. Dùng JPG, PNG, BMP, TIFF")
    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception:
        raise HTTPException(status_code=422, detail="File ảnh bị lỗi hoặc không hợp lệ")


@app.get("/")
async def root():
    return {
        "service": "AI X-Ray Diagnosis API",
        "version": "1.0.0",
        "demo_mode": DEMO_MODE,
        "model_loaded": (model is not None) or (tflite_interpreter is not None),
        "tflite": use_tflite,
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "ping": "GET /ping",
            "docs": "GET /docs",
        },
    }


@app.get("/ping")
async def ping():
    return {"status": "alive"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "demo_mode": DEMO_MODE,
        "model_loaded": (model is not None) or (tflite_interpreter is not None),
        "tflite": use_tflite,
        "labels": LABELS,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Nhận ảnh X-quang, trả về chẩn đoán AI kèm Grad-CAM heatmap.

    Returns:
        diagnosis, confidence, all_scores, heatmap, severity, recommendation,
        demo_mode; if TFLite, heatmap is saliency-style (not full Grad-CAM)
    """
    contents = await file.read()
    validate_image(contents, file.filename or "upload")

    logger.info("Processing: %s (%dKB)", file.filename, len(contents) // 1024)

    try:
        img_array = preprocess_image(contents)

        if tflite_interpreter is not None:
            pred_idx, scores = run_tflite_inference(
                tflite_interpreter,
                tflite_in_details,
                tflite_out_details,
                img_array,
            )
            # True Grad-CAM cần graph Keras; TFLite dùng vùng kích hoạt tương tự demo
            cam_idx = 0 if pred_idx == 0 else 2
            heatmap_img, severity = generate_demo_gradcam(img_array, cam_idx)
            is_demo = False
        elif model is not None:
            pred_idx, scores = run_inference(model, img_array)
            heatmap_img, severity = generate_gradcam(model, img_array, pred_idx)
            is_demo = False
        else:
            pred_idx, scores = run_demo_inference()
            heatmap_img, severity = generate_demo_gradcam(img_array, 0 if pred_idx == 0 else 2)
            is_demo = True

        diagnosis = LABELS[pred_idx] if pred_idx < len(LABELS) else LABELS[0]
        confidence = round(scores[pred_idx] * 100, 1) if pred_idx < len(scores) else 0.0
        severity_pct = round(severity * 100, 1)

        _, buffer = cv2.imencode(".jpg", heatmap_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        heatmap_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

        all_scores: dict = {}
        for i, label in enumerate(LABELS):
            v = float(scores[i]) * 100 if i < len(scores) else 0.0
            all_scores[label] = round(v, 1)

        body: dict = {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "all_scores": all_scores,
            "heatmap": heatmap_b64,
            "severity": severity_pct,
            "recommendation": get_recommendation(diagnosis, confidence / 100),
            "demo_mode": is_demo,
        }
        if use_tflite and tflite_interpreter is not None:
            body["tflite"] = True
            body["heatmap_note"] = "Saliency-style overlay; train Keras + .h5 for full Grad-CAM"
        return JSONResponse(body)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")
