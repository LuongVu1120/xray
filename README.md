# 🏥 AI X-Ray Diagnosis System

> Hệ thống chẩn đoán ảnh X-quang phổi bằng Deep Learning — DenseNet121 + Grad-CAM

[![Made by](https://img.shields.io/badge/Made%20by-Vu%20Dai%20Luong-blue)](https://github.com/vudailuong)
![Tech](https://img.shields.io/badge/Stack-Python%20%7C%20TensorFlow%20%7C%20FastAPI%20%7C%20Next.js-informational)
![Cost](https://img.shields.io/badge/Cost-0%20VND-success)

---

## ✨ Tính năng

- **Phân loại 3 nhóm**: Normal / Pneumonia / Other
- **Grad-CAM Heatmap**: Highlight chính xác vùng bất thường trên phổi
- **Confidence Score**: Điểm tin cậy cho từng nhãn chẩn đoán
- **Demo Mode**: Chạy được ngay không cần model (dùng để test UI)
- **Responsive**: Giao diện đẹp trên cả mobile và desktop

---

## 🛠 Tech Stack

| Tầng | Công nghệ |
|------|-----------|
| AI/ML | Python, TensorFlow, DenseNet121, Grad-CAM |
| Backend | FastAPI, OpenCV, Pillow |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Deploy | Vercel (frontend) + Railway.app (backend) |
| Dataset | NIH Chest X-Ray (112,000 ảnh) |

---

## 🚀 Chạy dự án

### Yêu cầu
- Python 3.10+
- Node.js 18+

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy file env
cp .env.example .env
# Mặc định DEMO_MODE=true — không cần model để chạy thử

uvicorn main:app --reload --port 8000
```

API sẽ chạy tại: http://localhost:8000
Docs: http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend
npm install

# Copy file env
cp .env.local.example .env.local

npm run dev
```

App sẽ chạy tại: http://localhost:3000

---

## 🤖 Train Model (cần GPU)

Khuyên dùng **Google Colab** (GPU T4 miễn phí):

1. Mở: https://colab.research.google.com
2. Upload file `backend/model/train.py`
3. Tải dataset NIH từ Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data
4. Chạy script → download `xray_model.h5`
5. Đặt file vào `backend/saved_model/xray_model.h5`
6. Đổi `DEMO_MODE=false` trong `.env`

---

## 📁 Cấu trúc thư mục

```
xray-ai/
├── backend/
│   ├── main.py                 ← FastAPI server
│   ├── requirements.txt
│   ├── model/
│   │   ├── train.py            ← Training script
│   │   ├── predict.py          ← Inference logic
│   │   └── gradcam.py          ← Grad-CAM implementation
│   └── saved_model/            ← Đặt xray_model.h5 ở đây
├── frontend/
│   ├── src/
│   │   ├── app/                ← Next.js App Router
│   │   ├── components/         ← UI components
│   │   ├── hooks/              ← Custom hooks
│   │   └── lib/                ← API client
│   └── package.json
└── README.md
```

---

## 📊 Kết quả mô hình

| Chỉ số | Giá trị |
|--------|---------|
| Dataset | 1,500 ảnh (500/nhóm) |
| Accuracy (val) | ~82–87% |
| Thời gian inference | < 3 giây |
| Kiến trúc | DenseNet121 + Transfer Learning |

---

## 🔗 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Thông tin service |
| GET | `/health` | Kiểm tra trạng thái |
| POST | `/predict` | Phân tích ảnh X-quang |

### Ví dụ gọi API

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@chest_xray.jpg"
```

Response:
```json
{
  "diagnosis": "Pneumonia",
  "confidence": 87.3,
  "all_scores": { "Normal": 6.1, "Other": 6.6, "Pneumonia": 87.3 },
  "heatmap": "data:image/jpeg;base64,...",
  "severity": 74.2,
  "recommendation": "Phát hiện dấu hiệu nghi ngờ viêm phổi...",
  "demo_mode": false
}
```

---

## 📝 License

MIT License — Vu Dai Luong © 2026
