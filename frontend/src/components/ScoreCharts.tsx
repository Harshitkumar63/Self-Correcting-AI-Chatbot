"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchChartData, ChartDataResponse } from "@/lib/api";

interface ScoreChartsProps {
  refreshTrigger?: number;
}

export default function ScoreCharts({ refreshTrigger }: ScoreChartsProps) {
  const [data, setData] = useState<ChartDataResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    try {
      const chartData = await fetchChartData();
      setData(chartData);
    } catch {
      // Backend may not be running
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData, refreshTrigger]);

  useEffect(() => {
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, [loadData]);

  if (loading || !data) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i} className="gradient-card border-border/50">
            <CardContent className="p-6">
              <div className="h-40 animate-shimmer rounded-lg" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const maxCount = Math.max(...data.score_distribution.map((d) => d.count), 1);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Score Distribution */}
      <Card className="gradient-card border-border/50">
        <CardHeader className="pb-2 px-5 pt-4">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            📊 Score Distribution
          </CardTitle>
        </CardHeader>
        <CardContent className="px-5 pb-5">
          <div className="space-y-1.5">
            {data.score_distribution.map((bucket) => (
              <div key={bucket.range} className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground w-14 text-right tabular-nums">
                  {bucket.range}
                </span>
                <div className="flex-1 h-5 bg-secondary/40 rounded-md overflow-hidden">
                  <div
                    className="h-full rounded-md transition-all duration-500"
                    style={{
                      width: `${(bucket.count / maxCount) * 100}%`,
                      background: getBarColor(bucket.range),
                    }}
                  />
                </div>
                <span className="text-[10px] text-muted-foreground w-6 tabular-nums">
                  {bucket.count}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Pass/Fail Breakdown */}
      <Card className="gradient-card border-border/50">
        <CardHeader className="pb-2 px-5 pt-4">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            🎯 Pass/Fail Breakdown
          </CardTitle>
        </CardHeader>
        <CardContent className="px-5 pb-5">
          <div className="flex items-center justify-center py-4">
            <div className="relative w-40 h-40">
              <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                {/* Background circle */}
                <circle
                  cx="50" cy="50" r="40"
                  fill="none"
                  stroke="oklch(0.22 0.01 260)"
                  strokeWidth="12"
                />
                {/* Pass arc */}
                <circle
                  cx="50" cy="50" r="40"
                  fill="none"
                  stroke="oklch(0.55 0.20 160)"
                  strokeWidth="12"
                  strokeDasharray={`${data.status_breakdown.pass_rate * 2.51} 251.3`}
                  strokeLinecap="round"
                  className="transition-all duration-1000"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-2xl font-bold text-emerald-400 tabular-nums">
                  {data.status_breakdown.pass_rate}%
                </span>
                <span className="text-[10px] text-muted-foreground">Pass Rate</span>
              </div>
            </div>
          </div>
          <div className="flex justify-center gap-6 mt-2">
            <div className="text-center">
              <span className="text-lg font-bold text-emerald-400 tabular-nums">
                {data.status_breakdown.passed}
              </span>
              <p className="text-[10px] text-muted-foreground">Passed</p>
            </div>
            <div className="text-center">
              <span className="text-lg font-bold text-red-400 tabular-nums">
                {data.status_breakdown.flagged}
              </span>
              <p className="text-[10px] text-muted-foreground">Flagged</p>
            </div>
            <div className="text-center">
              <span className="text-lg font-bold text-foreground/80 tabular-nums">
                {data.status_breakdown.total}
              </span>
              <p className="text-[10px] text-muted-foreground">Total</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Score Trend */}
      <Card className="gradient-card border-border/50 md:col-span-2">
        <CardHeader className="pb-2 px-5 pt-4">
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            📈 Score Trend (last {data.trend_data.length} queries)
          </CardTitle>
        </CardHeader>
        <CardContent className="px-5 pb-5">
          {data.trend_data.length === 0 ? (
            <div className="flex items-center justify-center py-8 text-sm text-muted-foreground">
              No data yet. Start chatting to see trends.
            </div>
          ) : (
            <div className="relative h-40">
              <TrendChart data={data.trend_data} />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ── Trend Chart (pure CSS/SVG) ────────────────────────────

function TrendChart({
  data,
}: {
  data: Array<{ hybrid_score: number; cosine_score: number; status: string }>;
}) {
  if (data.length === 0) return null;

  const width = 800;
  const height = 140;
  const padding = 10;

  const xScale = (i: number) =>
    padding + (i / (data.length - 1 || 1)) * (width - padding * 2);
  const yScale = (score: number) =>
    height - padding - score * (height - padding * 2);

  // Hybrid score line
  const hybridPath = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(i)} ${yScale(d.hybrid_score)}`)
    .join(" ");

  // Cosine score line
  const cosinePath = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${xScale(i)} ${yScale(d.cosine_score)}`)
    .join(" ");

  // Threshold line
  const thresholdY = yScale(0.7);

  return (
    <div className="w-full h-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full" preserveAspectRatio="none">
        {/* Grid lines */}
        {[0.25, 0.5, 0.75, 1.0].map((v) => (
          <line
            key={v}
            x1={padding}
            y1={yScale(v)}
            x2={width - padding}
            y2={yScale(v)}
            stroke="oklch(0.25 0.01 260)"
            strokeWidth="0.5"
          />
        ))}

        {/* Threshold line */}
        <line
          x1={padding}
          y1={thresholdY}
          x2={width - padding}
          y2={thresholdY}
          stroke="oklch(0.60 0.22 25)"
          strokeWidth="1"
          strokeDasharray="6 4"
          opacity="0.5"
        />

        {/* Cosine score line */}
        <path
          d={cosinePath}
          fill="none"
          stroke="oklch(0.55 0.15 200)"
          strokeWidth="1.5"
          opacity="0.5"
        />

        {/* Hybrid score line */}
        <path
          d={hybridPath}
          fill="none"
          stroke="oklch(0.65 0.25 270)"
          strokeWidth="2"
        />

        {/* Data points */}
        {data.map((d, i) => (
          <circle
            key={i}
            cx={xScale(i)}
            cy={yScale(d.hybrid_score)}
            r="3"
            fill={d.status === "Passed" ? "oklch(0.55 0.20 160)" : "oklch(0.60 0.22 25)"}
            className="transition-all"
          />
        ))}
      </svg>

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 mt-2">
        <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <span className="h-2 w-2 rounded-full" style={{ background: "oklch(0.65 0.25 270)" }} />
          Hybrid Score
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <span className="h-2 w-2 rounded-full" style={{ background: "oklch(0.55 0.15 200)" }} />
          Cosine Score
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
          <span className="h-0.5 w-3 rounded" style={{ background: "oklch(0.60 0.22 25)" }} />
          Threshold (0.70)
        </span>
      </div>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────

function getBarColor(range: string): string {
  const start = parseInt(range);
  if (start >= 70) return "oklch(0.55 0.20 160)";
  if (start >= 50) return "oklch(0.70 0.18 50)";
  return "oklch(0.60 0.22 25)";
}
