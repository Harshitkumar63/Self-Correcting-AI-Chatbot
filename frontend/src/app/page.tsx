"use client";

import { useState, useCallback } from "react";
import { Separator } from "@/components/ui/separator";
import ChatInterface from "@/components/ChatInterface";
import EvaluationMonitor from "@/components/EvaluationMonitor";
import FineTuningPanel from "@/components/FineTuningPanel";

export default function Dashboard() {
  // Shared refresh counter — incremented when a new chat message is sent
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleNewMessage = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  return (
    <div className="flex min-h-screen flex-col">
      {/* ── Header ─────────────────────────────────────── */}
      <header className="glass sticky top-0 z-50 border-b border-border/30">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="gradient-primary flex h-10 w-10 items-center justify-center rounded-xl text-lg shadow-lg shadow-primary/20">
              🧠
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                Self-Improving LLM Pipeline
              </h1>
              <p className="text-xs text-muted-foreground">
                Automated Evaluation & LoRA Fine-Tuning Feedback Loop
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 rounded-full bg-secondary/50 px-4 py-1.5">
              <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs text-muted-foreground">
                Pipeline Active
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Content ───────────────────────────────── */}
      <main className="mx-auto w-full max-w-[1600px] flex-1 p-6">
        <div className="grid h-[calc(100vh-120px)] grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Left Panel — Chat */}
          <div className="lg:col-span-5 flex flex-col">
            <ChatInterface onNewMessage={handleNewMessage} />
          </div>

          {/* Vertical Separator (desktop only) */}
          <div className="hidden lg:flex lg:col-span-0 items-center">
            <Separator orientation="vertical" className="bg-border/20 h-full" />
          </div>

          {/* Right Panel — Monitor + Analytics */}
          <div className="lg:col-span-6 flex flex-col gap-6 overflow-hidden">
            <div className="flex-1 min-h-0">
              <EvaluationMonitor refreshTrigger={refreshTrigger} />
            </div>
            <div className="flex-shrink-0">
              <FineTuningPanel refreshTrigger={refreshTrigger} />
            </div>
          </div>
        </div>
      </main>

      {/* ── Footer ─────────────────────────────────────── */}
      <footer className="border-t border-border/20 py-4">
        <div className="mx-auto max-w-[1600px] px-6 flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            Qwen2.5-0.5B-Instruct • all-MiniLM-L6-v2 • PEFT/LoRA
          </p>
          <p className="text-xs text-muted-foreground">
            Threshold: 0.70 similarity • 50 samples to trigger training
          </p>
        </div>
      </footer>
    </div>
  );
}
