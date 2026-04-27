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
    return (
      "Network Error (thường do CORS): backend phải có ALLOWED_ORIGINS gồm URL frontend " +
      "(ví dụ http://localhost:3000) — không dùng hai dòng trùng key trong .env. " +
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
