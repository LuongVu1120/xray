"use client";
import { useState, useRef } from "react";
import { usePredict } from "@/hooks/usePredict";
import { UploadZone } from "@/components/UploadZone";
import { ResultCard } from "@/components/ResultCard";
import { HeatmapViewer } from "@/components/HeatmapViewer";
import { Loader2, Stethoscope, Github, Brain, Zap, Shield } from "lucide-react";
import clsx from "clsx";

export default function Home() {
  const [file, setFile]         = useState<File | null>(null);
  const [preview, setPreview]   = useState<string | null>(null);
  const resultRef               = useRef<HTMLDivElement>(null);

  const { mutate, data, isPending, error, reset } = usePredict();

  function handleFile(f: File) {
    setFile(f);
    reset();
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }

  function handleAnalyze() {
    if (!file) return;
    mutate(file, {
      onSuccess: () => {
        setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
      }
    });
  }

  return (
    <div className="min-h-screen grid-bg">
      {/* Header */}
      <header className="border-b border-slate-800/60 backdrop-blur-sm sticky top-0 z-50 bg-slate-950/80">
        <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-brand-600/20 border border-brand-500/30 flex items-center justify-center">
              <Stethoscope className="w-4 h-4 text-brand-400" />
            </div>
            <div>
              <span className="font-semibold text-white text-sm">AI X-Ray</span>
              <span className="text-slate-500 text-sm"> Diagnosis</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="hidden sm:flex items-center gap-1.5 text-xs text-slate-500">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              DenseNet121 + Grad-CAM
            </span>
            <a
              href="https://github.com/vudailuong"
              target="_blank"
              rel="noreferrer"
              className="text-slate-500 hover:text-slate-300 transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-10 space-y-10">

        {/* Hero */}
        <section className="text-center space-y-4 pt-4">
          <div className="inline-flex items-center gap-2 bg-brand-600/10 border border-brand-500/20 rounded-full px-4 py-1.5 text-xs text-brand-400 font-medium">
            <Brain className="w-3.5 h-3.5" />
            Deep Learning · Computer Vision · Explainable AI
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-white">
            Chẩn đoán{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-cyan-400">
              X-Quang Phổi
            </span>
            <br />bằng Trí Tuệ Nhân Tạo
          </h1>
          <p className="text-slate-400 max-w-xl mx-auto text-base leading-relaxed">
            Upload ảnh X-quang — AI phân loại{" "}
            <span className="text-emerald-400">Normal</span>,{" "}
            <span className="text-red-400">Pneumonia</span>,{" "}
            <span className="text-amber-400">Other</span>{" "}
            và highlight chính xác vùng bất thường bằng Grad-CAM.
          </p>

          {/* Stats */}
          <div className="flex justify-center gap-8 pt-2">
            {[
              { icon: Zap,    label: "Phân tích",    value: "< 5 giây" },
              { icon: Brain,  label: "Dataset",      value: "112K ảnh" },
              { icon: Shield, label: "Độ chính xác", value: "~85%" },
            ].map(({ icon: Icon, label, value }) => (
              <div key={label} className="text-center">
                <div className="flex items-center gap-1.5 justify-center mb-1">
                  <Icon className="w-3.5 h-3.5 text-brand-400" />
                  <span className="text-xs text-slate-500">{label}</span>
                </div>
                <p className="text-base font-semibold text-white font-mono">{value}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Upload + Analyze */}
        <section className="card p-6 space-y-5">
          <div>
            <p className="label mb-3">Tải ảnh lên</p>
            <UploadZone onFile={handleFile} disabled={isPending} />
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-xl px-4 py-3">
              ⚠️ {(error as Error).message || "Có lỗi xảy ra. Vui lòng thử lại."}
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={!file || isPending}
            className={clsx(
              "btn-primary w-full flex items-center justify-center gap-2 text-base",
              isPending && "animate-pulse"
            )}
          >
            {isPending ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Đang phân tích ảnh...
              </>
            ) : (
              <>
                <Stethoscope className="w-5 h-5" />
                Phân tích X-Quang
              </>
            )}
          </button>
        </section>

        {/* Results */}
        {data && (
          <section ref={resultRef} className="space-y-5 animate-[fadeUp_0.4s_ease_forwards]">
            <div className="flex items-center gap-2">
              <div className="h-px flex-1 bg-slate-800" />
              <span className="label px-3">Kết quả phân tích</span>
              <div className="h-px flex-1 bg-slate-800" />
            </div>

            <div className="grid lg:grid-cols-2 gap-5">
              {/* Left: images */}
              <div className="space-y-5">
                {preview && (
                  <HeatmapViewer original={preview} heatmap={data.heatmap} />
                )}
              </div>

              {/* Right: diagnosis */}
              <div>
                <ResultCard result={data} />
              </div>
            </div>

            <button
              onClick={() => { setFile(null); setPreview(null); reset(); }}
              className="text-sm text-slate-500 hover:text-slate-300 transition-colors flex items-center gap-1.5 mx-auto"
            >
              ↩ Phân tích ảnh khác
            </button>
          </section>
        )}

        {/* About */}
        <section className="card p-6 space-y-4">
          <p className="label">Về dự án</p>
          <div className="grid sm:grid-cols-3 gap-4">
            {[
              { title: "DenseNet121", desc: "Kiến trúc CNN được dùng phổ biến nhất trong AI y tế, pretrained trên ImageNet" },
              { title: "Grad-CAM", desc: "Explainable AI — giúp bác sĩ hiểu vùng nào AI đang 'nhìn' khi chẩn đoán" },
              { title: "NIH Dataset", desc: "112,000 ảnh X-quang thật từ National Institutes of Health — Mỹ" },
            ].map(({ title, desc }) => (
              <div key={title} className="bg-slate-800/50 rounded-xl p-4 space-y-2">
                <h3 className="font-semibold text-white text-sm font-mono">{title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-600 text-center pt-2 font-mono">
            Built by Vu Dai Luong · Portfolio Project 2026 · Python + TensorFlow + FastAPI + Next.js
          </p>
        </section>

      </main>
    </div>
  );
}
