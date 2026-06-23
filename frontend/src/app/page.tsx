"use client";

import { useState, useCallback } from "react";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ChatInterface from "@/components/ChatInterface";
import EvaluationMonitor from "@/components/EvaluationMonitor";
import FineTuningPanel from "@/components/FineTuningPanel";
import AdminCurationQueue from "@/components/AdminCurationQueue";

export default function Dashboard() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [activeTab, setActiveTab] = useState("pipeline");

  const handleNewMessage = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  return (
    <div className="flex min-h-screen flex-col bg-background">
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
                Hybrid Evaluation • Teacher Correction • Human-in-the-Loop Curation
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
      <main className="mx-auto w-full max-w-[1600px] flex-1 px-6 py-4">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          {/* Tab Buttons */}
          <TabsList className="mb-4 bg-secondary/40 border border-border/30 p-1">
            <TabsTrigger
              value="pipeline"
              id="tab-pipeline"
              className="data-[state=active]:gradient-primary data-[state=active]:text-primary-foreground px-4 py-1.5 text-xs font-medium"
            >
              🔄 Pipeline Dashboard
            </TabsTrigger>
            <TabsTrigger
              value="curation"
              id="tab-curation"
              className="data-[state=active]:gradient-primary data-[state=active]:text-primary-foreground px-4 py-1.5 text-xs font-medium"
            >
              🎛️ Admin Curation Queue
            </TabsTrigger>
          </TabsList>

          {/* ── Pipeline Dashboard Tab ──────────────────── */}
          <TabsContent value="pipeline" className="mt-0">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              {/* Left Panel — Chat */}
              <div className="lg:col-span-5" style={{ height: "calc(100vh - 180px)" }}>
                <ChatInterface onNewMessage={handleNewMessage} />
              </div>

              {/* Vertical Separator (desktop only) */}
              <div className="hidden lg:flex lg:col-span-1 items-center justify-center">
                <Separator orientation="vertical" className="bg-border/20 h-full" />
              </div>

              {/* Right Panel — Monitor + Fine-Tuning stacked vertically */}
              <div className="lg:col-span-6 flex flex-col gap-6" style={{ height: "calc(100vh - 180px)" }}>
                <div style={{ flex: "1 1 55%", minHeight: 0 }}>
                  <EvaluationMonitor refreshTrigger={refreshTrigger} />
                </div>
                <div style={{ flex: "0 0 auto" }}>
                  <FineTuningPanel refreshTrigger={refreshTrigger} />
                </div>
              </div>
            </div>
          </TabsContent>

          {/* ── Admin Curation Queue Tab ────────────────── */}
          <TabsContent value="curation" className="mt-0">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              {/* Left Panel — Chat */}
              <div className="lg:col-span-5" style={{ height: "calc(100vh - 180px)" }}>
                <ChatInterface onNewMessage={handleNewMessage} />
              </div>

              {/* Vertical Separator */}
              <div className="hidden lg:flex lg:col-span-1 items-center justify-center">
                <Separator orientation="vertical" className="bg-border/20 h-full" />
              </div>

              {/* Right Panel — Curation Queue */}
              <div className="lg:col-span-6" style={{ height: "calc(100vh - 180px)" }}>
                <AdminCurationQueue refreshTrigger={refreshTrigger} />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* ── Footer ─────────────────────────────────────── */}
      <footer className="border-t border-border/20 py-3">
        <div className="mx-auto max-w-[1600px] px-6 flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            Qwen2.5-0.5B-Instruct • all-MiniLM-L6-v2 • PEFT/LoRA • Hybrid Evaluator
          </p>
          <p className="text-xs text-muted-foreground">
            Threshold: 0.70 hybrid score • 50 samples to trigger training
          </p>
        </div>
      </footer>
    </div>
  );
}
