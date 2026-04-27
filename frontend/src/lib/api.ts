import axios, { isAxiosError } from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
// Render free (cold start + tải model) thường > 30s — tăng timeout; ghi đè: NEXT_PUBLIC_API_TIMEOUT_MS
const API_TIMEOUT_MS = Number(process.env.NEXT_PUBLIC_API_TIMEOUT_MS) || 180_000;

export interface PredictResponse {
  diagnosis: "Normal" | "Pneumonia" | "Other";
  confidence: number;
  all_scores: Record<string, number>;
  heatmap: string;
  severity: number;
  recommendation: string;
  demo_mode: boolean;
}

export interface HealthResponse {
  status: string;
  demo_mode: boolean;
  model_loaded: boolean;
  labels: string[];
}

export const apiClient = axios.create({
  baseURL: API_URL,
  timeout: API_TIMEOUT_MS,
});

function formatPredictError(err: unknown): string {
  if (isAxiosError(err) && err.response) {
    const d = err.response.data as { detail?: string | { msg?: string } } | string | undefined;
    if (err.response.status >= 500) {
      if (typeof d === "string") return d;
      if (d && typeof d === "object" && "detail" in d) {
        const det = d.detail;
        if (typeof det === "string") return `Lỗi máy chủ (${err.response.status}): ${det}`;
        if (Array.isArray(det)) {
          return `Lỗi máy chủ (${err.response.status}): ${JSON.stringify(det)}`;
        }
      }
      return `Lỗi máy chủ (${err.response.status})`;
    }
    if (err.response.data && typeof err.response.data === "object" && "detail" in err.response.data) {
      const det = (err.response.data as { detail?: string }).detail;
      if (typeof det === "string") return det;
    }
  }
  if (isAxiosError(err) && (err.code === "ECONNABORTED" || /timeout/i.test(String(err.message)))) {
    return (
      "Hết thời gian chờ API (timeout). Free tier Render có thể “ngủ” 50s+ lần đầu; " +
      "thử bấm lại sau 1–2 phút hoặc dùng UptimeRobot ping /ping. Nếu vẫn lỗi, tăng NEXT_PUBLIC_API_TIMEOUT_MS trên Vercel."
    );
  }
  if (isAxiosError(err) && err.message === "Network Error") {
    const isProdApi =
      typeof API_URL === "string" &&
      !API_URL.includes("localhost") &&
      !API_URL.includes("127.0.0.1");
    if (isProdApi) {
      return (
        "Không kết nối được tới API sau khi chờ (thường gặp trên Render free). " +
        "Nguyên nhân hay gặp: (1) Backend trả 502 khi hết RAM lúc chạy TensorFlow — mở Render → Logs để xem OOM/killed; " +
        "thử bật DEMO_MODE=true, dùng model TFLite, hoặc gói có nhiều RAM hơn. " +
        "(2) Thiếu CORS / sai NEXT_PUBLIC_API_URL — kiểm tra biến trên Vercel trùng URL Render. " +
        "(3) Server đang cold start — thử lại sau hoặc UptimeRobot ping /ping."
      );
    }
    return (
      "Network Error (thường do CORS): backend cần ALLOWED_ORIGINS khớp URL frontend. " +
      "Hoặc API không chạy / sai NEXT_PUBLIC_API_URL."
    );
  }
  if (err instanceof Error) return err.message;
  return "Có lỗi xảy ra. Vui lòng thử lại.";
}

export async function predictXray(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  try {
    const { data } = await apiClient.post<PredictResponse>("/predict", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  } catch (e) {
    throw new Error(formatPredictError(e));
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>("/health");
  return data;
}
