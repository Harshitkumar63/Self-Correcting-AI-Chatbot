"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ChatInterface from "@/components/ChatInterface";
import ConversationSidebar from "@/components/ConversationSidebar";
import EvaluationMonitor from "@/components/EvaluationMonitor";
import FineTuningPanel from "@/components/FineTuningPanel";
import AdminCurationQueue from "@/components/AdminCurationQueue";
import ScoreCharts from "@/components/ScoreCharts";
import PerformanceDashboard from "@/components/PerformanceDashboard";
import GroundTruthManager from "@/components/GroundTruthManager";
import ThemeToggle from "@/components/ThemeToggle";
import { isAuthenticated, isAdmin, getStoredUser, logout } from "@/lib/auth";

export default function Dashboard() {
  const router = useRouter();
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [activeTab, setActiveTab] = useState("pipeline");
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);
  const [username, setUsername] = useState<string>("");
  const [authChecked, setAuthChecked] = useState(false);

  // Auth guard
  useEffect(() => {
    if (!isAuthenticated()) {
      router.push("/login");
      return;
    }
    const user = getStoredUser();
    setUserRole(user?.role || null);
    setUsername(user?.username || "");
    setAuthChecked(true);
  }, [router]);

  const handleNewMessage = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  const handleConversationCreated = useCallback((id: number) => {
    setConversationId(id);
  }, []);

  if (!authChecked) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* ── Header ─────────────────────────────────────── */}
      <header className="glass sticky top-0 z-50 border-b border-border/30">
        <div className="mx-auto flex max-w-[1800px] items-center justify-between px-6 py-4">
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
            <ThemeToggle />

            <div className="flex items-center gap-2 rounded-full bg-secondary/50 px-4 py-1.5">
              <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs text-muted-foreground">
                Pipeline Active
              </span>
            </div>

            {/* User info + logout */}
            <div className="flex items-center gap-2 rounded-full bg-secondary/50 px-4 py-1.5">
              <span className="text-xs text-muted-foreground">
                👤 {username}
              </span>
              <span className="text-[10px] text-primary/70 font-medium uppercase">
                {userRole}
              </span>
            </div>
            <button
              id="logout-btn"
              onClick={logout}
              className="rounded-lg border border-border/30 bg-secondary/40 px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-secondary/70 transition-all"
            >
              Logout
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Content ───────────────────────────────── */}
      <main className="mx-auto w-full max-w-[1800px] flex-1 px-6 py-4">
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
              value="analytics"
              id="tab-analytics"
              className="data-[state=active]:gradient-primary data-[state=active]:text-primary-foreground px-4 py-1.5 text-xs font-medium"
            >
              📈 Analytics
            </TabsTrigger>
            <TabsTrigger
              value="performance"
              id="tab-performance"
              className="data-[state=active]:gradient-primary data-[state=active]:text-primary-foreground px-4 py-1.5 text-xs font-medium"
            >
              🏋️ Performance
            </TabsTrigger>
            {isAdmin() && (
              <TabsTrigger
                value="curation"
                id="tab-curation"
                className="data-[state=active]:gradient-primary data-[state=active]:text-primary-foreground px-4 py-1.5 text-xs font-medium"
              >
                🎛️ Admin Curation Queue
              </TabsTrigger>
            )}
            {isAdmin() && (
              <TabsTrigger
                value="knowledge"
                id="tab-knowledge"
                className="data-[state=active]:gradient-primary data-[state=active]:text-primary-foreground px-4 py-1.5 text-xs font-medium"
              >
                📚 Knowledge Base
              </TabsTrigger>
            )}
          </TabsList>

          {/* ── Pipeline Dashboard Tab ──────────────────── */}
          <TabsContent value="pipeline" className="mt-0">
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-14">
              {/* Left — Conversation Sidebar */}
              <div className="lg:col-span-2" style={{ height: "calc(100vh - 180px)" }}>
                <ConversationSidebar
                  activeId={conversationId}
                  onSelect={setConversationId}
                  onNew={setConversationId}
                />
              </div>

              {/* Middle — Chat */}
              <div className="lg:col-span-5" style={{ height: "calc(100vh - 180px)" }}>
                <ChatInterface
                  conversationId={conversationId}
                  onNewMessage={handleNewMessage}
                  onConversationCreated={handleConversationCreated}
                />
              </div>

              {/* Right Panel — Monitor + Fine-Tuning */}
              <div className="lg:col-span-7 flex flex-col gap-4" style={{ height: "calc(100vh - 180px)" }}>
                <div style={{ flex: "1 1 55%", minHeight: 0 }}>
                  <EvaluationMonitor refreshTrigger={refreshTrigger} />
                </div>
                <div style={{ flex: "0 0 auto" }}>
                  <FineTuningPanel refreshTrigger={refreshTrigger} />
                </div>
              </div>
            </div>
          </TabsContent>

          {/* ── Analytics Tab ───────────────────────────── */}
          <TabsContent value="analytics" className="mt-0">
            <ScoreCharts refreshTrigger={refreshTrigger} />
          </TabsContent>

          {/* ── Performance Dashboard Tab ────────────────── */}
          <TabsContent value="performance" className="mt-0">
            <PerformanceDashboard refreshTrigger={refreshTrigger} />
          </TabsContent>

          {/* ── Admin Curation Queue Tab ────────────────── */}
          <TabsContent value="curation" className="mt-0">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
              <div className="lg:col-span-5" style={{ height: "calc(100vh - 180px)" }}>
                <ChatInterface
                  conversationId={conversationId}
                  onNewMessage={handleNewMessage}
                  onConversationCreated={handleConversationCreated}
                />
              </div>

              <div className="hidden lg:flex lg:col-span-1 items-center justify-center">
                <Separator orientation="vertical" className="bg-border/20 h-full" />
              </div>

              <div className="lg:col-span-6" style={{ height: "calc(100vh - 180px)" }}>
                <AdminCurationQueue refreshTrigger={refreshTrigger} />
              </div>
            </div>
          </TabsContent>

          {/* ── Knowledge Base Tab ──────────────────────── */}
          <TabsContent value="knowledge" className="mt-0">
            <div style={{ height: "calc(100vh - 180px)" }}>
              <GroundTruthManager />
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* ── Footer ─────────────────────────────────────── */}
      <footer className="border-t border-border/20 py-3">
        <div className="mx-auto max-w-[1800px] px-6 flex items-center justify-between">
          <p className="text-xs text-muted-foreground">
            Qwen2.5-0.5B-Instruct • all-MiniLM-L6-v2 • PEFT/LoRA • Hybrid Evaluator
          </p>
          <p className="text-xs text-muted-foreground">
            v3.0.0 • API v1 • JWT Auth • Threshold: 0.70
          </p>
        </div>
      </footer>
    </div>
  );
}
