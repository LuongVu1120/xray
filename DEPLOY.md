# Hướng dẫn deploy xray-ai (Frontend + Backend)

Cấu trúc: **Next.js** (Vercel) + **FastAPI + TensorFlow** (Railway, khuyên dùng). Model lớn nên tải từ **Google Drive** bằng `GDRIVE_FILE_ID` lúc container khởi động (đã tích hợp sẵn trong `main.py`).

## A. Deploy backend (Railway)

1. Đăng ký [Railway](https://railway.app), tạo project **New Project → Deploy from GitHub** (chọn repo `xray-ai`).
2. Tạo service mới, chọn **Root Directory** = `backend` (quận thư mục chứa `Dockerfile`).
3. Railway sẽ build image từ `backend/Dockerfile` (Nixpacks cũng được nhưng Docker ổn định hơn với TensorFlow). Nếu không tự nhận Docker: **Settings → Build → Builder** = Dockerfile.
4. Tab **Variables**, thêm (không commit secret vào git):

| Biến | Gợi ý |
|------|--------|
| `DEMO_MODE` | `false` (nếu có model thật) |
| `MODEL_PATH` | `saved_model/xray_model.h5` |
| `GDRIVE_FILE_ID` | ID file trên Google Drive (link bất kỳ với) |
| `ALLOWED_ORIGINS` | `https://tên-frontend.vercel.app` — nếu dùng domain riêng, thêm cả domain đó. Subdomain `*.vercel.app` đã được phép thêm bằng regex trong code. |
| (tuỳ chọn) `MODEL_TYPE` | `auto` hoặc `tflite` nếu dùng TFLite |

5. **Healthcheck** (Settings → Health): path `/health` hoặc `/ping`, **Timeout** 120–300 giây lần đầu (load TensorFlow + model).
6. Sau khi build xong, mở tab **Settings → Networking → Generate domain** để có URL dạng `https://xxxxx.up.railway.app`. Đó là `NEXT_PUBLIC_API_URL` cho frontend.

**Lưu ý:** Gói miễn phí RAM 512MB có thể thiếu với Keras lớn; cân nhắc TFLite (`scripts/export_tflite.py`) hoặc gói trả phí. Build Docker có thể 10–20 phút vì cài TensorFlow.

---

## B. Deploy frontend (Vercel)

1. Đăng kỳ [Vercel](https://vercel.com), **Add New… → Project**, import cùng repo GitHub.
2. **Root Directory** = `frontend` (rất quan trọng).
3. **Environment Variables** (Production + Preview tùy nhu cầu):

| Biến | Giá trị |
|------|---------|
| `NEXT_PUBLIC_API_URL` | URL backend Railway, ví dụ `https://xxxxx.up.railway.app` (không có dấu `/` cuối) |

4. **Deploy**. Domain Vercel dạng `https://tên.dự-án.vercel.app` đã tương thích CORS với backend (regex `*.vercel.app` + `ALLOWED_ORIGINS` bạn cấu hình).
5. Nếu bạn mua **custom domain**, thêm origin đó vào `ALLOWED_ORIGINS` trên Railway.

---

## C. Uptime (tuỳ chọn)

- [UptimeRobot](https://uptimerobot.com) ping `GET https://<railway-url>/ping` mỗi 14 phút để giảm cold start trên gói free.

---

## D. Kiểm tra nhanh

- Backend: `https://<railway>/docs` mở Swagger.
- Backend: `https://<railway>/health` trả `model_loaded: true` sau khi tải xong model.
- Frontend: mở site Vercel → upload ảnh thử; tab Network phải gọi `POST` tới `NEXT_PUBLIC_API_URL/predict` và status 200.

---

## Sự cố thường gặp

- **CORS:** Đảm bảo `ALLOWED_ORIGINS` có URL production (và custom domain nếu có). Local đã dùng regex; Vercel `*.vercel.app` đã bổ sung trong code.
- **Hết bộ nhớ (OOM) khi build/run:** dùng TFLite, hoặc tăng RAM trên nền tảng host.
- **Model không tìm thấy:** kiểm tra `GDRIVE_FILE_ID`, file Drive phải **Anyone with the link (Viewer)**.
