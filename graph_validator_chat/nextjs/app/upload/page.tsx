"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { UploadCloud } from "lucide-react";
import { Fraunces, Space_Grotesk } from "next/font/google";
import { cn } from "@/lib/utils";

import { Button } from "@/components/ui/button";
import SideNav from "@/components/SideNav";
import { savePipelinePayload } from "@/lib/pipeline-bootstrap";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const fraunces = Fraunces({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  style: ["normal", "italic"],
});

const PIPELINE_PROGRESS_KEY = "pipeline_progress";
const GRAPH_CACHE_KEYS = [
  "graph_html",
  "graph_timestamp",
  "graph_triples_count",
  "graph_html_chaotic",
  "graph_timestamp_chaotic",
  "graph_triples_count_chaotic",
  "graph_html_tree",
  "graph_timestamp_tree",
  "graph_triples_count_tree",
  "graph_layout",
];
const PIPELINE_STAGE_ORDER = [
  { id: "ingest", label: "Input ingest" },
  { id: "sentence_split", label: "Sentence splitting" },
  { id: "informative_filter", label: "Informative filtering" },
  { id: "ner", label: "NER" },
  { id: "coref", label: "COREF / entity mapping" },
  { id: "triple_generation", label: "Triple creation" },
  { id: "merge_relations", label: "Graph cleaning (merge)" },
  { id: "simplify_relations", label: "Graph cleaning (simplify)" },
  { id: "graph_build", label: "Graph build" },
  { id: "validator_init", label: "Validator warm-up" },
  { id: "complete", label: "Complete" },
];

interface PipelineProgress {
  stage: string;
  message: string;
  progress: number;
}

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [patentId, setPatentId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<PipelineProgress | null>(null);
  const [progressHistory, setProgressHistory] = useState<PipelineProgress[]>([]);
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const canSubmit = Boolean(file) || patentId.trim().length > 0;
  const currentStageIndex = progress
    ? PIPELINE_STAGE_ORDER.findIndex((stage) => stage.id === progress.stage)
    : -1;

  const recordProgress = (update: PipelineProgress) => {
    setProgress(update);
    setProgressHistory((prev) => {
      const existingIndex = prev.findIndex((item) => item.stage === update.stage);
      if (existingIndex >= 0) {
        const next = [...prev];
        next[existingIndex] = update;
        return next;
      }
      return [...prev, update];
    });
    if (typeof window !== "undefined") {
      localStorage.setItem(PIPELINE_PROGRESS_KEY, JSON.stringify(update));
    }
  };

  const stopProgressPolling = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  const checkProgress = async () => {
    try {
      const response = await fetch("/api/pipeline/progress");
      if (!response.ok) return;
      const data = await response.json();
      if (data?.progress?.stage && data.progress.stage !== "idle") {
        recordProgress(data.progress);
        if (data.progress.stage === "complete") {
          stopProgressPolling();
          setSubmitting(false);
          router.push("/analyze");
        }
        if (data.progress.stage === "error") {
          stopProgressPolling();
          setSubmitting(false);
        }
      }
    } catch {
      // Ignore transient progress errors.
    }
  };

  const startProgressPolling = () => {
    if (progressIntervalRef.current) return;
    progressIntervalRef.current = setInterval(checkProgress, 1000);
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    const verifySession = async () => {
      try {
        const response = await fetch("/api/session");
        if (response.ok) {
          const data = await response.json();
          const sessionId = data?.session_id;
          const storedSession = localStorage.getItem("server_session_id");
          if (sessionId && storedSession && storedSession !== sessionId) {
            localStorage.removeItem(PIPELINE_PROGRESS_KEY);
            localStorage.removeItem("pipeline_result");
            localStorage.removeItem("pipeline_payload");
            GRAPH_CACHE_KEYS.forEach((key) => localStorage.removeItem(key));
            localStorage.removeItem("chat_messages");
            localStorage.removeItem("chat_triples");
            localStorage.removeItem("chat_changes");
            localStorage.removeItem("chat_stats");
          }
          if (sessionId && storedSession !== sessionId) {
            localStorage.setItem("server_session_id", sessionId);
          }
        }
      } catch {
        // Ignore session check errors.
      }
    };

    verifySession().finally(() => {
      const stored = localStorage.getItem(PIPELINE_PROGRESS_KEY);
      if (stored) {
        try {
          const parsed = JSON.parse(stored) as PipelineProgress;
            if (parsed?.stage && parsed.stage !== "idle") {
              setProgress(parsed);
              setProgressHistory([parsed]);
              if (parsed.stage === "complete") {
                router.push("/analyze");
                return;
              }
            }
        } catch {
          // Ignore invalid stored progress.
        }
      }
      try {
        const existingResult = localStorage.getItem("pipeline_result");
        if (existingResult) {
          router.push("/analyze");
          return;
        }
      } catch {
        // Ignore localStorage read errors.
      }
      checkProgress();
    });
    return () => {
      stopProgressPolling();
    };
  }, []);

  const handleSubmit = async () => {
    if (!canSubmit || submitting) {
      return;
    }

    try {
      setSubmitting(true);
      let payload: Record<string, string> = {};

      if (file) {
        const isTxt = file.name.toLowerCase().endsWith(".txt");
        if (!isTxt) {
          setError("Please upload a .txt file for now.");
          setSubmitting(false);
          return;
        }
        const text = await file.text();
        if (!text.trim()) {
          setError("The selected text file is empty.");
          setSubmitting(false);
          return;
        }
        payload = { text, filename: file.name };
      } else {
        const trimmed = patentId.trim();
        if (!trimmed) {
          setError("Please enter a patent ID.");
          setSubmitting(false);
          return;
        }
        payload = { patent_id: trimmed };
      }

      setError(null);
      setProgressHistory([]);
      recordProgress({
        stage: "starting",
        message: "Starting pipeline",
        progress: 1,
      });
      startProgressPolling();
      savePipelinePayload(payload);

      const response = await fetch("/api/pipeline/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.error) {
        throw new Error(data.error || "Failed to run the pipeline.");
      }

      if (typeof window !== "undefined") {
        GRAPH_CACHE_KEYS.forEach((key) => localStorage.removeItem(key));
        localStorage.setItem("pipeline_result", JSON.stringify(data));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run the pipeline.");
      recordProgress({
        stage: "error",
        message: err instanceof Error ? err.message : "Failed to run the pipeline.",
        progress: 0,
      });
      stopProgressPolling();
      setSubmitting(false);
    } finally {
      // submitting is reset when we see complete/error progress
    }
  };

  return (
    <div className={cn(spaceGrotesk.className, "min-h-dvh bg-background text-foreground flex")}>
      <SideNav current="upload" />
      <main className="flex-1">
        <div className="mx-auto flex min-h-dvh w-full max-w-2xl flex-col justify-center px-6 py-16">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl border bg-background shadow-sm">
              <UploadCloud className="h-5 w-5 text-amber-500" />
            </div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              Claim Studio
            </p>
          </div>

          <h1
            className={cn(
              fraunces.className,
              "mt-6 text-balance text-4xl font-semibold tracking-tight sm:text-5xl"
            )}
          >
            Upload or reference a patent
          </h1>

          <p className="mt-3 text-pretty text-base text-muted-foreground">
            Add a source file or enter a patent ID to start validation.
          </p>

          <div className="mt-10 space-y-6 rounded-3xl border bg-background/80 p-6 shadow-sm">
            <div className="space-y-2">
              <label className="text-sm font-semibold" htmlFor="upload-file">
                Upload file
              </label>
              <input
                id="upload-file"
                type="file"
                accept=".txt"
                className="w-full rounded-2xl border px-4 py-3 text-sm"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </div>

            <div className="flex items-center justify-center text-xs font-semibold uppercase tracking-[0.3em] text-muted-foreground">
              OR
            </div>

            <div className="space-y-2">
              <label className="text-sm font-semibold" htmlFor="patent-id">
                Patent ID
              </label>
              <input
                id="patent-id"
                type="text"
                placeholder="US-XXXXXXXX"
                value={patentId}
                onChange={(e) => setPatentId(e.target.value)}
                className="w-full rounded-2xl border px-4 py-3 text-sm"
              />
            </div>

            {error && (
              <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}

            {progress && progress.stage !== "idle" && (
              <div className="rounded-2xl border bg-background/70 p-4">
                <div className="flex items-center justify-between text-sm font-semibold">
                  <span>Pipeline progress</span>
                  <span>{Math.round(progress.progress)}%</span>
                </div>
                <div className="mt-2 h-2 w-full rounded-full bg-muted">
                  <div
                    className="h-2 rounded-full bg-amber-500 transition-all"
                    style={{
                      width: `${Math.min(100, Math.max(0, progress.progress))}%`,
                    }}
                  />
                </div>
                <p className="mt-2 text-xs text-muted-foreground">{progress.message}</p>
                <div className="mt-4 grid gap-2 text-xs">
                  {PIPELINE_STAGE_ORDER.map((stage, index) => {
                    const isActive = currentStageIndex === index;
                    const isDone = currentStageIndex > index;
                    const dotClass = isDone
                      ? "bg-emerald-500"
                      : isActive
                      ? "bg-amber-500"
                      : "bg-muted-foreground/30";
                    const textClass = isDone
                      ? "text-emerald-600"
                      : isActive
                      ? "text-foreground"
                      : "text-muted-foreground";

                    return (
                      <div key={stage.id} className="flex items-center gap-2">
                        <span className={`h-2 w-2 rounded-full ${dotClass}`} />
                        <span className={textClass}>{stage.label}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <Button className="w-full rounded-full" disabled={!canSubmit || submitting} onClick={handleSubmit}>
              {submitting ? "Processing..." : "Enter"}
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}
