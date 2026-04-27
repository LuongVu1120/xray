"use client";
import { useMemo } from "react";
import { Bot, Loader2, Sparkles, BookOpen, Activity, AlertCircle, ExternalLink } from "lucide-react";
import clsx from "clsx";
import type { AgentState } from "@/hooks/useAgentDiagnose";

interface Props {
  state: AgentState;
  onCancel: () => void;
}

const STEP_META: Record<
  string,
  { label: string; icon: typeof Bot }
> = {
  classify: { label: "Phân loại CNN", icon: Activity },
  uncertainty: { label: "Bất định MC-Dropout", icon: Sparkles },
  heatmap: { label: "Grad-CAM++", icon: Activity },
  knowledge: { label: "Tra cứu bệnh học", icon: BookOpen },
  pubmed: { label: "PubMed", icon: BookOpen },
  report: { label: "Soạn báo cáo", icon: Bot },
};

const STEP_ORDER = ["classify", "uncertainty", "heatmap", "knowledge", "pubmed", "report"];

function StepStatus({ step, state }: { step: string; state: AgentState }) {
  const events = state.events.filter((e) => e.step === step);
  const last = events[events.length - 1];
  const { label, icon: Icon } = STEP_META[step] ?? { label: step, icon: Bot };

  let badge: "pending" | "running" | "done" | "error" = "pending";
  if (last?.status === "done") badge = "done";
  else if (last?.status === "error") badge = "error";
  else if (last?.status === "started" || last?.status === "delta") badge = "running";

  return (
    <div className="flex items-center gap-2 text-xs">
      <span
        className={clsx(
          "w-7 h-7 rounded-lg flex items-center justify-center border",
          badge === "done" && "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
          badge === "running" && "bg-brand-500/10 border-brand-500/30 text-brand-400",
          badge === "pending" && "bg-slate-800 border-slate-700 text-slate-500",
          badge === "error" && "bg-red-500/10 border-red-500/30 text-red-400"
        )}
      >
        {badge === "running" ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Icon className="w-3.5 h-3.5" />}
      </span>
      <span
        className={clsx(
          "font-medium",
          badge === "done" && "text-slate-200",
          badge === "running" && "text-brand-300",
          badge === "pending" && "text-slate-500",
          badge === "error" && "text-red-300"
        )}
      >
        {label}
      </span>
    </div>
  );
}

function MultilabelChips({ findings }: { findings?: Array<{ label: string; score: number }> }) {
  if (!findings || findings.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {findings.map((f) => (
        <span
          key={f.label}
          className="text-[11px] font-mono px-2 py-1 rounded-md bg-red-500/10 text-red-300 border border-red-500/20"
        >
          {f.label} · {(f.score * 100).toFixed(0)}%
        </span>
      ))}
    </div>
  );
}

export function AgentReport({ state, onCancel }: Props) {
  const topLine = useMemo(() => {
    const c = state.classify;
    if (!c) return null;
    return `${c.top_label} · ${(((c.top_score as number) ?? 0) * 100).toFixed(1)}%`;
  }, [state.classify]);

  const entropy = state.uncertainty?.available
    ? (state.uncertainty.entropy as number).toFixed(3)
    : null;

  return (
    <div className="card p-6 space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500/20 to-cyan-500/20 border border-brand-500/30 flex items-center justify-center">
            <Bot className="w-4.5 h-4.5 text-brand-400" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white">AI Diagnostic Agent</p>
            <p className="text-xs text-slate-500 font-mono">
              CNN + MC-Dropout + Grad-CAM++ + PubMed + LLM
            </p>
          </div>
        </div>
        {state.isRunning && (
          <button
            onClick={onCancel}
            className="text-xs text-slate-500 hover:text-red-400 transition-colors"
          >
            Huỷ
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {STEP_ORDER.map((s) => (
          <StepStatus key={s} step={s} state={state} />
        ))}
      </div>

      {state.error && (
        <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 rounded-xl px-3 py-2">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <p className="text-xs text-red-300">{state.error}</p>
        </div>
      )}

      {(topLine || entropy) && (
        <div className="flex flex-wrap gap-2">
          {topLine && (
            <span className="text-xs px-2.5 py-1 rounded-lg bg-brand-500/10 text-brand-300 border border-brand-500/20 font-mono">
              Top: {topLine}
            </span>
          )}
          {entropy && (
            <span className="text-xs px-2.5 py-1 rounded-lg bg-amber-500/10 text-amber-300 border border-amber-500/20 font-mono">
              Entropy: {entropy}
            </span>
          )}
          {state.classify?.mode && (
            <span className="text-xs px-2.5 py-1 rounded-lg bg-slate-800 text-slate-400 border border-slate-700 font-mono">
              Mode: {state.classify.mode as string}
            </span>
          )}
        </div>
      )}

      <MultilabelChips findings={state.classify?.findings as any} />

      {state.heatmap && (
        <div>
          <p className="label mb-2">Grad-CAM++</p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={state.heatmap}
            alt="Agent heatmap"
            className="w-full max-h-72 rounded-xl object-contain bg-black"
          />
        </div>
      )}

      {(state.reportText || state.isRunning) && (
        <div>
          <p className="label mb-2">Báo cáo y khoa</p>
          <pre className="whitespace-pre-wrap break-words text-sm text-slate-200 leading-relaxed bg-slate-900/60 border border-slate-800 rounded-xl p-4 font-sans">
            {state.reportText || (
              <span className="text-slate-500 italic">Đang soạn…</span>
            )}
          </pre>
        </div>
      )}

      {state.pubmed?.articles && (state.pubmed.articles as any[]).length > 0 && (
        <div>
          <p className="label mb-2">PubMed gợi ý</p>
          <ul className="space-y-1.5">
            {(state.pubmed.articles as any[]).map((a: any) => (
              <li key={a.pmid} className="text-xs">
                <a
                  href={a.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-brand-400 hover:text-brand-300 inline-flex items-center gap-1"
                >
                  <ExternalLink className="w-3 h-3" />
                  {a.title}
                </a>
                {a.journal && (
                  <span className="text-slate-500"> — {a.journal}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
