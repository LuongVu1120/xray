"use client";
import { PredictResponse } from "@/lib/api";
import { ScoreBar } from "./ScoreBar";
import { AlertCircle, CheckCircle, Activity, Info } from "lucide-react";
import clsx from "clsx";

interface Props { result: PredictResponse; }

const diagnosisConfig = {
  Normal:    { icon: CheckCircle, color: "text-emerald-400", ring: "ring-emerald-500/30", bg: "bg-emerald-500/5" },
  Pneumonia: { icon: AlertCircle, color: "text-red-400",     ring: "ring-red-500/30",     bg: "bg-red-500/5"     },
  Other:     { icon: Activity,    color: "text-amber-400",   ring: "ring-amber-500/30",   bg: "bg-amber-500/5"   },
};

export function ResultCard({ result }: Props) {
  const cfg = diagnosisConfig[result.diagnosis] ?? diagnosisConfig["Other"];
  const Icon = cfg.icon;

  return (
    <div className="space-y-5 animate-[fadeUp_0.4s_ease_forwards]">

      {/* Demo notice */}
      {result.demo_mode && (
        <div className="flex items-center gap-2 bg-amber-500/10 border border-amber-500/20 rounded-xl px-4 py-3">
          <Info className="w-4 h-4 text-amber-400 shrink-0" />
          <p className="text-xs text-amber-300">
            Đang chạy ở chế độ Demo — kết quả mô phỏng. Train model thật để có kết quả chính xác.
          </p>
        </div>
      )}

      {/* Main diagnosis */}
      <div className={clsx("card p-6 ring-1", cfg.ring, cfg.bg)}>
        <div className="flex items-start gap-4">
          <div className={clsx("p-3 rounded-xl bg-slate-800", cfg.bg)}>
            <Icon className={clsx("w-7 h-7", cfg.color)} />
          </div>
          <div className="flex-1">
            <p className="label mb-1">Kết quả chẩn đoán</p>
            <h2 className={clsx("text-3xl font-bold tracking-tight", cfg.color)}>
              {result.diagnosis}
            </h2>
            <p className="text-slate-400 text-sm mt-1">
              Độ tin cậy:{" "}
              <span className={clsx("font-semibold font-mono", cfg.color)}>
                {result.confidence}%
              </span>
            </p>
          </div>
        </div>

        {/* Severity bar */}
        {result.diagnosis !== "Normal" && (
          <div className="mt-5 pt-5 border-t border-slate-800">
            <div className="flex justify-between mb-2">
              <span className="label">Mức độ bất thường</span>
              <span className="text-xs font-mono text-slate-400">{result.severity}%</span>
            </div>
            <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={clsx(
                  "h-full rounded-full transition-all duration-1000",
                  result.severity > 70 ? "bg-red-500" :
                  result.severity > 40 ? "bg-amber-500" : "bg-emerald-500"
                )}
                style={{ width: `${result.severity}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Score breakdown */}
      <div className="card p-6 space-y-4">
        <p className="label">Phân tích chi tiết</p>
        {Object.entries(result.all_scores)
          .sort(([, a], [, b]) => b - a)
          .map(([label, score]) => (
            <ScoreBar
              key={label}
              label={label}
              score={score}
              isTop={label === result.diagnosis}
            />
          ))}
      </div>

      {/* Recommendation */}
      <div className="card p-5 flex items-start gap-3">
        <Info className="w-5 h-5 text-brand-400 shrink-0 mt-0.5" />
        <div>
          <p className="label mb-1.5">Khuyến nghị</p>
          <p className="text-slate-300 text-sm leading-relaxed">{result.recommendation}</p>
          <p className="text-slate-600 text-xs mt-3 font-mono">
            ⚠️ Kết quả AI chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ
          </p>
        </div>
      </div>
    </div>
  );
}
