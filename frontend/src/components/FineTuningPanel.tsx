"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  fetchStats,
  triggerTraining,
  fetchTuningStatus,
  StatsResponse,
} from "@/lib/api";

// ── Props ──────────────────────────────────────────────────

interface FineTuningPanelProps {
  refreshTrigger?: number;
}

// ── Component ──────────────────────────────────────────────

export default function FineTuningPanel({
  refreshTrigger,
}: FineTuningPanelProps) {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<string>("idle");
  const [trainingMessage, setTrainingMessage] = useState<string>("");
  const [isTriggering, setIsTriggering] = useState(false);

  const loadStats = useCallback(async () => {
    try {
      const data = await fetchStats();
      setStats(data);
    } catch {
      // Backend may not be running
    }
  }, []);

  const loadTrainingStatus = useCallback(async () => {
    try {
      const data = await fetchTuningStatus();
      setTrainingStatus(data.training_status);
      setTrainingMessage(data.message);
    } catch {
      // ignore
    }
  }, []);

  // Load on mount + refresh trigger
  useEffect(() => {
    loadStats();
    loadTrainingStatus();
  }, [loadStats, loadTrainingStatus, refreshTrigger]);

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadStats();
      if (trainingStatus === "preparing" || trainingStatus === "training") {
        loadTrainingStatus();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [loadStats, loadTrainingStatus, trainingStatus]);

  const handleTriggerTraining = async () => {
    setIsTriggering(true);
    try {
      const result = await triggerTraining();
      setTrainingStatus(result.training_status);
      setTrainingMessage(result.message);
    } catch (err) {
      setTrainingMessage(
        `Error: ${err instanceof Error ? err.message : "Failed to trigger training"}`
      );
    } finally {
      setIsTriggering(false);
    }
  };

  const queueSize = stats?.training_queue_size ?? 0;
  const threshold = stats?.training_queue_threshold ?? 50;
  const progress = stats?.training_queue_progress ?? 0;

  const statusConfig: Record<
    string,
    { label: string; color: string; bgColor: string }
  > = {
    idle: {
      label: "Idle",
      color: "text-muted-foreground",
      bgColor: "bg-muted/50",
    },
    preparing: {
      label: "Preparing",
      color: "text-amber-400",
      bgColor: "bg-amber-500/15",
    },
    training: {
      label: "Training",
      color: "text-blue-400",
      bgColor: "bg-blue-500/15",
    },
    completed: {
      label: "Completed",
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/15",
    },
    failed: {
      label: "Failed",
      color: "text-red-400",
      bgColor: "bg-red-500/15",
    },
  };

  const currentStatus = statusConfig[trainingStatus] || statusConfig.idle;

  return (
    <Card className="gradient-card border-border/50">
      <CardHeader className="border-b border-border/30 pb-4">
        <div className="flex items-center gap-3">
          <div className="gradient-danger flex h-9 w-9 items-center justify-center rounded-lg text-sm">
            🔧
          </div>
          <div>
            <CardTitle className="text-lg">Fine-Tuning Analytics</CardTitle>
            <p className="text-xs text-muted-foreground">
              LoRA training pipeline status
            </p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6 pt-6">
        {/* ── Training Queue ───────────────────────────── */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-foreground/80">
              Training Queue
            </span>
            <span className="text-xs tabular-nums text-muted-foreground">
              {queueSize}/{threshold} samples
            </span>
          </div>
          <Progress
            value={progress}
            className="h-3 bg-secondary/60"
          />
          <p className="text-xs text-muted-foreground">
            {threshold - queueSize > 0
              ? `${threshold - queueSize} more flagged responses needed to auto-trigger`
              : "🔔 Threshold reached — ready to fine-tune!"}
          </p>
        </div>

        <Separator className="bg-border/30" />

        {/* ── Stats Grid ───────────────────────────────── */}
        {stats && (
          <div className="grid grid-cols-2 gap-3">
            <StatCard
              label="Total Queries"
              value={stats.total_queries.toString()}
              icon="📨"
            />
            <StatCard
              label="Avg Score"
              value={`${(stats.average_score * 100).toFixed(1)}%`}
              icon="📈"
              valueColor={
                stats.average_score >= 0.7
                  ? "text-emerald-400"
                  : "text-amber-400"
              }
            />
            <StatCard
              label="Passed"
              value={stats.passed_count.toString()}
              icon="✅"
              valueColor="text-emerald-400"
            />
            <StatCard
              label="Flagged"
              value={stats.flagged_count.toString()}
              icon="⚠️"
              valueColor="text-red-400"
            />
          </div>
        )}

        <Separator className="bg-border/30" />

        {/* ── Training Status ──────────────────────────── */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-foreground/80">
              Training Status
            </span>
            <Badge
              className={`text-xs ${currentStatus.color} ${currentStatus.bgColor} border-0`}
            >
              {trainingStatus === "training" && (
                <span className="mr-1 inline-block h-2 w-2 rounded-full bg-blue-400 animate-pulse" />
              )}
              {currentStatus.label}
            </Badge>
          </div>

          {trainingMessage && (
            <p className="text-xs text-muted-foreground bg-secondary/30 rounded-lg p-3 leading-relaxed">
              {trainingMessage}
            </p>
          )}

          <Button
            id="trigger-training-btn"
            onClick={handleTriggerTraining}
            disabled={
              isTriggering ||
              queueSize === 0 ||
              trainingStatus === "training" ||
              trainingStatus === "preparing"
            }
            className="w-full gradient-primary hover:opacity-90 transition-opacity"
          >
            {isTriggering ? (
              <span className="flex items-center gap-2">
                <span className="animate-spin">⏳</span> Triggering...
              </span>
            ) : trainingStatus === "training" ? (
              <span className="flex items-center gap-2">
                <span className="animate-spin">🔄</span> Training in
                Progress...
              </span>
            ) : (
              `🚀 Trigger Fine-Tuning (${queueSize} samples)`
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// ── Stat Card ──────────────────────────────────────────────

function StatCard({
  label,
  value,
  icon,
  valueColor = "text-foreground",
}: {
  label: string;
  value: string;
  icon: string;
  valueColor?: string;
}) {
  return (
    <div className="rounded-xl bg-secondary/30 border border-border/20 p-3 transition-colors hover:bg-secondary/50">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-sm">{icon}</span>
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p className={`text-xl font-bold tabular-nums ${valueColor}`}>{value}</p>
    </div>
  );
}
