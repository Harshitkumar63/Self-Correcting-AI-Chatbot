"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { fetchLogs, LogEntry } from "@/lib/api";

// ── Props ──────────────────────────────────────────────────

interface EvaluationMonitorProps {
  refreshTrigger?: number; // increment to force refresh
}

// ── Component ──────────────────────────────────────────────

export default function EvaluationMonitor({
  refreshTrigger,
}: EvaluationMonitorProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(false);

  const loadLogs = useCallback(async (pageNum: number) => {
    setLoading(true);
    try {
      const data = await fetchLogs(pageNum, 10);
      setLogs(data.logs);
      setTotal(data.total);
      setPage(data.page);
      setTotalPages(data.total_pages);
    } catch {
      // Backend may not be running yet
    } finally {
      setLoading(false);
    }
  }, []);

  // Load on mount and when refreshTrigger changes
  useEffect(() => {
    loadLogs(1);
  }, [loadLogs, refreshTrigger]);

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => loadLogs(page), 5000);
    return () => clearInterval(interval);
  }, [loadLogs, page]);

  const truncate = (text: string, maxLen: number = 60) =>
    text.length > maxLen ? text.slice(0, maxLen) + "…" : text;

  return (
    <Card className="gradient-card border-border/50 flex flex-col overflow-hidden">
      <CardHeader className="border-b border-border/30 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="gradient-success flex h-9 w-9 items-center justify-center rounded-lg text-sm">
              📊
            </div>
            <div>
              <CardTitle className="text-lg">Evaluation Monitor</CardTitle>
              <p className="text-xs text-muted-foreground">
                Real-time quality metrics — {total} interactions logged
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-muted-foreground">Live</span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 p-0 overflow-hidden">
        <ScrollArea className="h-[300px]">
          {logs.length === 0 && !loading ? (
            <div className="flex flex-col items-center justify-center py-16 text-center animate-fade-in">
              <div className="mb-3 text-4xl opacity-50">📋</div>
              <p className="text-sm text-muted-foreground">
                No evaluations yet. Start chatting to see results here.
              </p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="border-border/30 hover:bg-transparent">
                  <TableHead className="text-xs font-semibold text-muted-foreground w-[30%]">
                    User Query
                  </TableHead>
                  <TableHead className="text-xs font-semibold text-muted-foreground w-[35%]">
                    LLM Output
                  </TableHead>
                  <TableHead className="text-xs font-semibold text-muted-foreground text-center w-[15%]">
                    Score
                  </TableHead>
                  <TableHead className="text-xs font-semibold text-muted-foreground text-center w-[20%]">
                    Status
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {loading && logs.length === 0
                  ? Array.from({ length: 3 }).map((_, i) => (
                      <TableRow key={`skeleton-${i}`} className="border-border/20">
                        <TableCell colSpan={4}>
                          <div className="h-4 rounded animate-shimmer" />
                        </TableCell>
                      </TableRow>
                    ))
                  : logs.map((log, idx) => (
                      <TableRow
                        key={log.id}
                        className="border-border/20 hover:bg-secondary/30 transition-colors animate-fade-in"
                        style={{ animationDelay: `${idx * 30}ms` }}
                      >
                        <TableCell className="text-xs text-foreground/80 py-3">
                          {truncate(log.user_query)}
                        </TableCell>
                        <TableCell className="text-xs text-foreground/70 py-3">
                          {truncate(log.llm_response, 80)}
                        </TableCell>
                        <TableCell className="text-center py-3">
                          <ScoreIndicator score={log.similarity_score} />
                        </TableCell>
                        <TableCell className="text-center py-3">
                          <Badge
                            className={`text-xs font-medium ${
                              log.evaluation_status === "Passed"
                                ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/25"
                                : "bg-red-500/15 text-red-400 border-red-500/25"
                            }`}
                            variant="outline"
                          >
                            {log.evaluation_status === "Passed" ? "✓ Passed" : "⚠ Flagged"}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
              </TableBody>
            </Table>
          )}
        </ScrollArea>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-border/30 px-4 py-3">
            <span className="text-xs text-muted-foreground">
              Page {page} of {totalPages}
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => loadLogs(page - 1)}
                disabled={page <= 1}
                className="text-xs h-7"
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => loadLogs(page + 1)}
                disabled={page >= totalPages}
                className="text-xs h-7"
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Score Indicator ────────────────────────────────────────

function ScoreIndicator({ score }: { score: number }) {
  const pct = (score * 100).toFixed(1);
  const color =
    score >= 0.7
      ? "text-emerald-400"
      : score >= 0.5
        ? "text-amber-400"
        : "text-red-400";

  return (
    <div className="flex flex-col items-center gap-0.5">
      <span className={`text-sm font-bold tabular-nums ${color}`}>
        {pct}%
      </span>
      <div className="h-1 w-12 rounded-full bg-secondary overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            score >= 0.7
              ? "bg-emerald-400"
              : score >= 0.5
                ? "bg-amber-400"
                : "bg-red-400"
          }`}
          style={{ width: `${score * 100}%` }}
        />
      </div>
    </div>
  );
}
