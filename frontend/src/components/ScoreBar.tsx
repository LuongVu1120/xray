"use client";
import clsx from "clsx";

interface Props {
  label: string;
  score: number;
  isTop?: boolean;
}

const colors: Record<string, string> = {
  Normal:    "bg-emerald-500",
  Pneumonia: "bg-red-500",
  Other:     "bg-amber-500",
};

const textColors: Record<string, string> = {
  Normal:    "text-emerald-400",
  Pneumonia: "text-red-400",
  Other:     "text-amber-400",
};

export function ScoreBar({ label, score, isTop }: Props) {
  const bar   = colors[label]     ?? "bg-brand-500";
  const text  = textColors[label] ?? "text-brand-400";

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <span className={clsx("text-sm font-medium", isTop ? text : "text-slate-400")}>
          {label}
          {isTop && <span className="ml-2 text-xs opacity-60">← chẩn đoán</span>}
        </span>
        <span className={clsx("text-sm font-mono font-semibold", isTop ? text : "text-slate-400")}>
          {score.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all duration-700 ease-out", bar, !isTop && "opacity-40")}
          style={{ width: `${score}%` }}
        />
      </div>
    </div>
  );
}
