"use client";
import { useCallback, useRef, useState } from "react";
import {
  streamAgentDiagnose,
  type AgentDiagnoseOptions,
  type AgentEvent,
} from "@/lib/api";

export interface AgentState {
  events: AgentEvent[];
  reportText: string;
  classify?: AgentEvent["data"];
  uncertainty?: AgentEvent["data"];
  heatmap?: string;
  knowledge?: AgentEvent["data"];
  pubmed?: AgentEvent["data"];
  reportPayload?: AgentEvent["data"];
  isRunning: boolean;
  error?: string;
  currentStep?: string;
}

const initial: AgentState = { events: [], reportText: "", isRunning: false };

export function useAgentDiagnose() {
  const [state, setState] = useState<AgentState>(initial);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async (file: File, opts: AgentDiagnoseOptions = {}) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setState({ ...initial, isRunning: true });

    try {
      for await (const evt of streamAgentDiagnose(file, { ...opts, signal: ctrl.signal })) {
        setState((prev) => {
          const next: AgentState = {
            ...prev,
            events: [...prev.events, evt],
            currentStep: evt.step,
          };
          if (evt.status === "error" || evt.step === "fatal") {
            next.error = evt.message || "Agent gặp lỗi";
            return next;
          }
          if (evt.status === "done" && evt.data) {
            switch (evt.step) {
              case "classify":
                next.classify = evt.data;
                break;
              case "uncertainty":
                next.uncertainty = evt.data;
                break;
              case "heatmap":
                next.heatmap = evt.data.heatmap as string | undefined;
                break;
              case "knowledge":
                next.knowledge = evt.data;
                break;
              case "pubmed":
                next.pubmed = evt.data;
                break;
              case "report":
                next.reportText = (evt.data.text as string) ?? prev.reportText;
                next.reportPayload = evt.data.payload as Record<string, any> | undefined;
                break;
            }
          }
          if (evt.status === "delta" && evt.step === "report") {
            next.reportText = prev.reportText + (evt.data?.text ?? "");
          }
          return next;
        });
      }
    } catch (e: any) {
      if (e?.name !== "AbortError") {
        setState((prev) => ({ ...prev, error: e?.message || "Lỗi không xác định" }));
      }
    } finally {
      setState((prev) => ({ ...prev, isRunning: false }));
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(initial);
  }, []);

  return { ...state, run, cancel, reset };
}
