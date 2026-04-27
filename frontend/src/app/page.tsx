"use client";
import { useState, useRef } from "react";
import { usePredict } from "@/hooks/usePredict";
import { useAgentDiagnose } from "@/hooks/useAgentDiagnose";
import { UploadZone } from "@/components/UploadZone";
import { ResultCard } from "@/components/ResultCard";
import { HeatmapViewer } from "@/components/HeatmapViewer";
import { AgentReport } from "@/components/AgentReport";
import { PatientForm } from "@/components/PatientForm";
import {
  Loader2,
  Stethoscope,
  Github,
  Brain,
  Zap,
  Shield,
  Bot,
  Activity,
} from "lucide-react";
import clsx from "clsx";
import type { PatientContext } from "@/lib/api";

type Tab = "classifier" | "agent";

export default function Home() {
  const [tab, setTab] = useState<Tab>("agent");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [patient, setPatient] = useState<PatientContext>({});
  const resultRef = useRef<HTMLDivElement>(null);

  const predict = usePredict();
  const agent = useAgentDiagnose();

  const isPending = tab === "classifier" ? predict.isPending : agent.isRunning;
  const error = tab === "classifier" ? (predict.error as Error | null)?.message : agent.error;

  function handleFile(f: File) {
    setFile(f);
    predict.reset();
    agent.reset();
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }

  function handleAnalyze() {
    if (!file) return;
    if (tab === "classifier") {
      predict.mutate(file, {
        onSuccess: () => {
          setTimeout(
            () => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
            100
          );
        },
      });
    } else {
      agent.run(file, {
        patient: Object.keys(patient).length ? patient : undefined,
        use_tta: true,
        use_uncertainty: true,
        use_pubmed: true,
        use_llm: true,
      });
      setTimeout(
        () => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }),
        300
      );
    }
  }

  const showClassicResult = tab === "classifier" && predict.data;
  const showAgentResult =
    tab === "agent" &&
    (agent.isRunning || agent.events.length > 0 || agent.reportText);

  return (
    <div className="min-h-screen grid-bg">
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
              CheXNet · Grad-CAM++ · LLM Agent
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
        <section className="text-center space-y-4 pt-4">
          <div className="inline-flex items-center gap-2 bg-brand-600/10 border border-brand-500/20 rounded-full px-4 py-1.5 text-xs text-brand-400 font-medium">
            <Brain className="w-3.5 h-3.5" />
            Agentic AI · Computer Vision · Multi-label · LLM Reasoning
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-white">
            Chẩn đoán{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-cyan-400">
              X-Quang Phổi
            </span>
            <br />
            bằng AI Agent
          </h1>
          <p className="text-slate-400 max-w-2xl mx-auto text-base leading-relaxed">
            Pipeline kết hợp <span className="text-brand-300">CNN multi-label</span>,{" "}
            <span className="text-amber-300">MC-Dropout uncertainty</span>,{" "}
            <span className="text-emerald-300">Grad-CAM++</span> và{" "}
            <span className="text-cyan-300">LLM</span> tổng hợp báo cáo có cấu trúc bác sĩ X-quang.
          </p>

          <div className="flex justify-center gap-8 pt-2">
            {[
              { icon: Zap, label: "Pipeline", value: "5 bước" },
              { icon: Brain, label: "Nhãn NIH", value: "14 bệnh" },
              { icon: Shield, label: "Uncertainty", value: "MC-Dropout" },
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

        <section className="card p-6 space-y-5">
          <div className="flex bg-slate-800/40 rounded-xl p-1 border border-slate-700/50 w-full sm:w-fit">
            <button
              onClick={() => setTab("agent")}
              className={clsx(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all flex-1 sm:flex-none justify-center",
                tab === "agent"
                  ? "bg-brand-600/20 text-brand-300 border border-brand-500/30"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <Bot className="w-4 h-4" />
              Agent (mới)
            </button>
            <button
              onClick={() => setTab("classifier")}
              className={clsx(
                "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all flex-1 sm:flex-none justify-center",
                tab === "classifier"
                  ? "bg-brand-600/20 text-brand-300 border border-brand-500/30"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <Activity className="w-4 h-4" />
              Classifier
            </button>
          </div>

          <div>
            <p className="label mb-3">Tải ảnh X-quang lên</p>
            <UploadZone onFile={handleFile} disabled={isPending} />
          </div>

          {tab === "agent" && (
            <div>
              <p className="label mb-3">Bối cảnh bệnh nhân (tuỳ chọn)</p>
              <PatientForm value={patient} onChange={setPatient} disabled={isPending} />
            </div>
          )}

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-xl px-4 py-3">
              ⚠️ {error}
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
                Đang phân tích…
              </>
            ) : tab === "agent" ? (
              <>
                <Bot className="w-5 h-5" />
                Chạy AI Agent
              </>
            ) : (
              <>
                <Stethoscope className="w-5 h-5" />
                Phân tích X-Quang
              </>
            )}
          </button>
        </section>

        {(showClassicResult || showAgentResult) && (
          <section ref={resultRef} className="space-y-5 animate-[fadeUp_0.4s_ease_forwards]">
            <div className="flex items-center gap-2">
              <div className="h-px flex-1 bg-slate-800" />
              <span className="label px-3">Kết quả phân tích</span>
              <div className="h-px flex-1 bg-slate-800" />
            </div>

            {showAgentResult && (
              <div className="grid lg:grid-cols-2 gap-5">
                <div className="space-y-5">
                  {preview && (
                    <HeatmapViewer
                      original={preview}
                      heatmap={agent.heatmap || preview}
                    />
                  )}
                </div>
                <div>
                  <AgentReport state={agent} onCancel={agent.cancel} />
                </div>
              </div>
            )}

            {showClassicResult && predict.data && (
              <div className="grid lg:grid-cols-2 gap-5">
                <div className="space-y-5">
                  {preview && (
                    <HeatmapViewer original={preview} heatmap={predict.data.heatmap} />
                  )}
                </div>
                <div>
                  <ResultCard result={predict.data} />
                </div>
              </div>
            )}

            <button
              onClick={() => {
                setFile(null);
                setPreview(null);
                predict.reset();
                agent.reset();
              }}
              className="text-sm text-slate-500 hover:text-slate-300 transition-colors flex items-center gap-1.5 mx-auto"
            >
              ↩ Phân tích ảnh khác
            </button>
          </section>
        )}

        <section className="card p-6 space-y-4">
          <p className="label">Về dự án</p>
          <div className="grid sm:grid-cols-3 gap-4">
            {[
              {
                title: "CheXNet 14-label",
                desc: "DenseNet121 đa nhãn (14 bệnh NIH) + TTA + temperature scaling",
              },
              {
                title: "Grad-CAM++",
                desc: "Định vị bất thường chính xác hơn Grad-CAM gốc, hỗ trợ DenseNet lồng",
              },
              {
                title: "Diagnostic Agent",
                desc: "Tool registry: classify · MC-Dropout · Grad-CAM · PubMed · LLM compose",
              },
            ].map(({ title, desc }) => (
              <div key={title} className="bg-slate-800/50 rounded-xl p-4 space-y-2">
                <h3 className="font-semibold text-white text-sm font-mono">{title}</h3>
                <p className="text-xs text-slate-400 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-600 text-center pt-2 font-mono">
            Built by Vu Dai Luong · Portfolio Project 2026 · CV + LLM Agent
          </p>
        </section>
      </main>
    </div>
  );
}
