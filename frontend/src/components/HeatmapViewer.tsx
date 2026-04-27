"use client";
import { useState } from "react";
import clsx from "clsx";

interface Props {
  original: string;   // base64 or object URL
  heatmap: string;    // base64 data URI
}

export function HeatmapViewer({ original, heatmap }: Props) {
  const [view, setView] = useState<"split" | "original" | "heatmap">("split");

  return (
    <div className="card overflow-hidden">
      {/* Tab switcher */}
      <div className="flex border-b border-slate-800">
        {(["split", "original", "heatmap"] as const).map((v) => (
          <button
            key={v}
            onClick={() => setView(v)}
            className={clsx(
              "flex-1 py-2.5 text-xs font-medium transition-colors capitalize",
              view === v
                ? "text-brand-400 border-b-2 border-brand-500 bg-brand-500/5"
                : "text-slate-500 hover:text-slate-300"
            )}
          >
            {v === "split" ? "So sánh" : v === "original" ? "Ảnh gốc" : "Grad-CAM"}
          </button>
        ))}
      </div>

      <div className="p-4">
        {view === "split" ? (
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <p className="label text-center">Ảnh gốc</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={original} alt="Original X-Ray" className="w-full rounded-xl object-contain bg-black aspect-square" />
            </div>
            <div className="space-y-2">
              <p className="label text-center">Grad-CAM</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={heatmap} alt="Grad-CAM Heatmap" className="w-full rounded-xl object-contain bg-black aspect-square" />
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="label text-center">{view === "original" ? "Ảnh X-quang gốc" : "Grad-CAM — vùng đỏ = bất thường"}</p>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={view === "original" ? original : heatmap}
              alt={view}
              className="w-full max-h-96 rounded-xl object-contain bg-black"
            />
            {view === "heatmap" && (
              <div className="flex items-center justify-center gap-3 pt-2">
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <span className="w-4 h-2 rounded bg-blue-600" /> Bình thường
                </div>
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <span className="w-4 h-2 rounded bg-green-500" /> Nhẹ
                </div>
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <span className="w-4 h-2 rounded bg-yellow-500" /> Trung bình
                </div>
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <span className="w-4 h-2 rounded bg-red-600" /> Cao
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
