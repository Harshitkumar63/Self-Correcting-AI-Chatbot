"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
} from "recharts";
import {
  fetchTrainingHistory,
  fetchScoreTimeline,
  fetchScoreDistribution,
  TrainingRun,
  TimelineEntry,
  DistributionEntry,
} from "@/lib/api";

// ── Props ──────────────────────────────────────────────────

interface PerformanceDashboardProps {
  refreshTrigger?: number;
}

// ── Component ──────────────────────────────────────────────

export default function PerformanceDashboard({
  refreshTrigger,
}: PerformanceDashboardProps) {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [distribution, setDistribution] = useState<DistributionEntry[]>([]);
  const [avgScore, setAvgScore] = useState(0);
  const [totalEvals, setTotalEvals] = useState(0);
  const [selectedRun, setSelectedRun] = useState<TrainingRun | null>(null);
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [historyData, timelineData, distData] = await Promise.all([
        fetchTrainingHistory(),
        fetchScoreTimeline(30, "day"),
        fetchScoreDistribution(),
      ]);

      setRuns(historyData.runs);
      setTimeline(timelineData.timeline);
      setDistribution(distData.distribution);
      setAvgScore(distData.avg_score);
      setTotalEvals(distData.total);

      // Auto-select the latest run
      if (historyData.runs.length > 0 && !selectedRun) {
        setSelectedRun(historyData.runs[0]);
      }
    } catch {
      // API might not be available yet
    } finally {
      setLoading(false);
    }
  }, [selectedRun]);

  useEffect(() => {
    loadData();
  }, [loadData, refreshTrigger]);

  // Format loss history for the chart
  const lossChartData = selectedRun?.loss_history?.map((entry) => ({
    step: entry.step,
    loss: entry.loss,
    lr: entry.learning_rate * 10000, // Scale for visibility
    epoch: entry.epoch,
  })) || [];

  // Format timeline for percentages
  const timelineChartData = timeline.map((entry) => ({
    date: entry.date.slice(5), // "MM-DD"
    "Hybrid Score": Math.round(entry.avg_hybrid * 100),
    "Cosine Score": Math.round(entry.avg_cosine * 100),
    "Pass Rate": Math.round(entry.pass_rate * 100),
    queries: entry.count,
  }));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* ── Summary Stats Row ───────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          icon="🏋️"
          label="Training Runs"
          value={runs.length.toString()}
          sub="Total completed"
          color="text-blue-400"
        />
        <SummaryCard
          icon="📊"
          label="Avg Hybrid Score"
          value={`${(avgScore * 100).toFixed(1)}%`}
          sub={`${totalEvals} evaluations`}
          color={avgScore >= 0.7 ? "text-emerald-400" : "text-amber-400"}
        />
        <SummaryCard
          icon="📉"
          label="Best Final Loss"
          value={
            runs.length > 0
              ? Math.min(
                  ...runs
                    .filter((r) => r.final_loss !== null)
                    .map((r) => r.final_loss!)
                ).toFixed(4)
              : "N/A"
          }
          sub="Lowest training loss"
          color="text-emerald-400"
        />
        <SummaryCard
          icon="📈"
          label="Latest Run"
          value={
            runs.length > 0
              ? `${runs[0].samples_trained} samples`
              : "No runs yet"
          }
          sub={
            runs.length > 0
              ? new Date(runs[0].completed_at).toLocaleDateString()
              : ""
          }
          color="text-purple-400"
        />
      </div>

      {/* ── Score Timeline Chart ─────────────────────────────── */}
      <Card className="gradient-card border-border/50">
        <CardHeader className="border-b border-border/30 px-5 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg text-sm"
                 style={{ background: "linear-gradient(135deg, oklch(0.55 0.20 280), oklch(0.50 0.22 260))" }}>
              📈
            </div>
            <div>
              <CardTitle className="text-base">Model Performance Over Time</CardTitle>
              <p className="text-[11px] text-muted-foreground">
                Daily average scores — last 30 days
              </p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="pt-4 px-2 pb-2">
          {timelineChartData.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="mb-3 text-3xl opacity-50">📋</div>
              <p className="text-sm text-muted-foreground">
                No evaluation data yet. Start chatting to see performance trends.
              </p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={timelineChartData}>
                <defs>
                  <linearGradient id="gradHybrid" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="oklch(0.60 0.20 280)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="oklch(0.60 0.20 280)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradCosine" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="oklch(0.65 0.20 170)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="oklch(0.65 0.20 170)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: "#888" }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#888" }} unit="%" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(20,20,30,0.95)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "12px",
                    fontSize: "12px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "11px" }} />
                <Area
                  type="monotone"
                  dataKey="Hybrid Score"
                  stroke="oklch(0.60 0.20 280)"
                  strokeWidth={2}
                  fill="url(#gradHybrid)"
                />
                <Area
                  type="monotone"
                  dataKey="Cosine Score"
                  stroke="oklch(0.65 0.20 170)"
                  strokeWidth={2}
                  fill="url(#gradCosine)"
                />
                <Line
                  type="monotone"
                  dataKey="Pass Rate"
                  stroke="oklch(0.70 0.18 140)"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      {/* ── Two-Column: Distribution + Training Loss ──────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Score Distribution */}
        <Card className="gradient-card border-border/50">
          <CardHeader className="border-b border-border/30 px-5 py-3">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg text-sm bg-amber-500/20">
                📊
              </div>
              <div>
                <CardTitle className="text-base">Score Distribution</CardTitle>
                <p className="text-[11px] text-muted-foreground">
                  Hybrid score histogram across all evaluations
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-4 px-2 pb-2">
            {distribution.length === 0 || totalEvals === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="mb-3 text-3xl opacity-50">📊</div>
                <p className="text-sm text-muted-foreground">No data yet.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={distribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="range" tick={{ fontSize: 9, fill: "#888" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#888" }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(20,20,30,0.95)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: "12px",
                      fontSize: "12px",
                    }}
                  />
                  <Bar
                    dataKey="count"
                    fill="oklch(0.60 0.20 280)"
                    radius={[4, 4, 0, 0]}
                    name="Evaluations"
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Training Loss Curve */}
        <Card className="gradient-card border-border/50">
          <CardHeader className="border-b border-border/30 px-5 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg text-sm bg-red-500/20">
                  📉
                </div>
                <div>
                  <CardTitle className="text-base">Training Loss Curve</CardTitle>
                  <p className="text-[11px] text-muted-foreground">
                    {selectedRun
                      ? `Run ${selectedRun.run_id} — ${selectedRun.samples_trained} samples`
                      : "Select a training run"}
                  </p>
                </div>
              </div>
              {runs.length > 1 && (
                <select
                  className="rounded-lg border border-border/30 bg-secondary/40 px-2 py-1 text-xs text-muted-foreground"
                  value={selectedRun?.run_id || ""}
                  onChange={(e) => {
                    const run = runs.find((r) => r.run_id === e.target.value);
                    if (run) setSelectedRun(run);
                  }}
                >
                  {runs.map((run) => (
                    <option key={run.run_id} value={run.run_id}>
                      {new Date(run.completed_at).toLocaleDateString()} — {run.samples_trained} samples
                    </option>
                  ))}
                </select>
              )}
            </div>
          </CardHeader>
          <CardContent className="pt-4 px-2 pb-2">
            {lossChartData.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="mb-3 text-3xl opacity-50">🏋️</div>
                <p className="text-sm text-muted-foreground">
                  No training runs yet. Trigger fine-tuning to see loss curves.
                </p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={lossChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="step" tick={{ fontSize: 10, fill: "#888" }} label={{ value: "Step", position: "insideBottom", offset: -5, fontSize: 10, fill: "#888" }} />
                  <YAxis tick={{ fontSize: 10, fill: "#888" }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(20,20,30,0.95)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: "12px",
                      fontSize: "12px",
                    }}
                    formatter={(value, name) => [
                      name === "lr" ? (Number(value) / 10000).toExponential(2) : Number(value).toFixed(4),
                      name === "lr" ? "Learning Rate" : "Loss",
                    ]}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px" }} />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="oklch(0.65 0.25 25)"
                    strokeWidth={2}
                    dot={{ r: 2 }}
                    name="Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Training Runs Table ──────────────────────────────── */}
      {runs.length > 0 && (
        <Card className="gradient-card border-border/50">
          <CardHeader className="border-b border-border/30 px-5 py-3">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg text-sm bg-blue-500/20">
                📋
              </div>
              <div>
                <CardTitle className="text-base">Training Run History</CardTitle>
                <p className="text-[11px] text-muted-foreground">
                  All {runs.length} completed fine-tuning runs
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border/30 text-muted-foreground">
                    <th className="py-2 px-3 text-left font-medium">Date</th>
                    <th className="py-2 px-3 text-center font-medium">Samples</th>
                    <th className="py-2 px-3 text-center font-medium">Epochs</th>
                    <th className="py-2 px-3 text-center font-medium">Final Loss</th>
                    <th className="py-2 px-3 text-center font-medium">Runtime</th>
                    <th className="py-2 px-3 text-center font-medium">LoRA Config</th>
                    <th className="py-2 px-3 text-center font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr
                      key={run.run_id}
                      className={`border-b border-border/20 hover:bg-secondary/30 transition-colors cursor-pointer ${
                        selectedRun?.run_id === run.run_id ? "bg-secondary/40" : ""
                      }`}
                      onClick={() => setSelectedRun(run)}
                    >
                      <td className="py-2.5 px-3 text-foreground/80">
                        {new Date(run.completed_at).toLocaleDateString()}{" "}
                        <span className="text-muted-foreground/60">
                          {new Date(run.completed_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-center font-medium tabular-nums">
                        {run.samples_trained}
                      </td>
                      <td className="py-2.5 px-3 text-center tabular-nums">
                        {run.epochs}
                      </td>
                      <td className="py-2.5 px-3 text-center">
                        <span
                          className={`font-bold tabular-nums ${
                            run.final_loss !== null && run.final_loss < 1
                              ? "text-emerald-400"
                              : "text-amber-400"
                          }`}
                        >
                          {run.final_loss !== null ? run.final_loss.toFixed(4) : "—"}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-center tabular-nums text-muted-foreground">
                        {run.train_runtime
                          ? `${Math.round(run.train_runtime)}s`
                          : "—"}
                      </td>
                      <td className="py-2.5 px-3 text-center">
                        <Badge className="text-[10px] bg-secondary/50 border-border/30" variant="outline">
                          r={run.lora_r} α={run.lora_alpha}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3 text-center">
                        <Badge className="text-[10px] bg-emerald-500/15 text-emerald-400 border-emerald-500/25" variant="outline">
                          ✓ Complete
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ── Summary Card ──────────────────────────────────────────

function SummaryCard({
  icon,
  label,
  value,
  sub,
  color = "text-foreground",
}: {
  icon: string;
  label: string;
  value: string;
  sub: string;
  color?: string;
}) {
  return (
    <Card className="gradient-card border-border/50">
      <CardContent className="p-4">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-lg">{icon}</span>
          <span className="text-[11px] text-muted-foreground font-medium">{label}</span>
        </div>
        <p className={`text-2xl font-bold tabular-nums ${color}`}>{value}</p>
        <p className="text-[10px] text-muted-foreground mt-0.5">{sub}</p>
      </CardContent>
    </Card>
  );
}
