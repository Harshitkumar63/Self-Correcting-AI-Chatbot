"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  fetchCurationQueue,
  approveCurationItem,
  rejectCurationItem,
  editCurationItem,
  CurationItem,
} from "@/lib/api";

// ── Props ──────────────────────────────────────────────────

interface AdminCurationQueueProps {
  refreshTrigger?: number;
}

// ── Component ──────────────────────────────────────────────

export default function AdminCurationQueue({
  refreshTrigger,
}: AdminCurationQueueProps) {
  const [items, setItems] = useState<CurationItem[]>([]);
  const [stats, setStats] = useState({ total: 0, pending: 0, approved: 0, rejected: 0 });
  const [loading, setLoading] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const loadQueue = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchCurationQueue();
      setItems(data.items);
      setStats(data.stats);
    } catch {
      // Backend may not be running
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadQueue();
  }, [loadQueue, refreshTrigger]);

  // Auto-refresh every 8 seconds
  useEffect(() => {
    const interval = setInterval(loadQueue, 8000);
    return () => clearInterval(interval);
  }, [loadQueue]);

  const handleApprove = async (itemId: string) => {
    setActionLoading(itemId);
    try {
      await approveCurationItem(itemId);
      await loadQueue();
    } catch (err) {
      console.error("Approve failed:", err);
    } finally {
      setActionLoading(null);
    }
  };

  const handleReject = async (itemId: string) => {
    setActionLoading(itemId);
    try {
      await rejectCurationItem(itemId);
      await loadQueue();
    } catch (err) {
      console.error("Reject failed:", err);
    } finally {
      setActionLoading(null);
    }
  };

  const handleEditApprove = async (itemId: string) => {
    if (!editText.trim()) return;
    setActionLoading(itemId);
    try {
      await editCurationItem(itemId, editText.trim());
      setEditingId(null);
      setEditText("");
      await loadQueue();
    } catch (err) {
      console.error("Edit-approve failed:", err);
    } finally {
      setActionLoading(null);
    }
  };

  const handleApproveAll = async () => {
    setActionLoading("batch");
    try {
      for (const item of items) {
        await approveCurationItem(item.id);
      }
      await loadQueue();
    } catch (err) {
      console.error("Batch approve failed:", err);
    } finally {
      setActionLoading(null);
    }
  };

  const startEditing = (item: CurationItem) => {
    setEditingId(item.id);
    setEditText(item.teacher_correction);
    setExpandedId(item.id);
  };

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
    if (editingId === id) {
      setEditingId(null);
    }
  };

  return (
    <Card className="gradient-card border-border/50 flex h-full flex-col overflow-hidden">
      <CardHeader className="border-b border-border/30 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg text-sm"
              style={{ background: "linear-gradient(135deg, oklch(0.60 0.22 45), oklch(0.55 0.20 30))" }}>
              🎛️
            </div>
            <div>
              <CardTitle className="text-lg">Admin Curation Queue</CardTitle>
              <p className="text-xs text-muted-foreground">
                Review, edit, and approve flagged responses for training
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {stats.pending > 0 && (
              <Badge className="bg-amber-500/15 text-amber-400 border-amber-500/25 text-xs" variant="outline">
                {stats.pending} pending
              </Badge>
            )}
          </div>
        </div>

        {/* Stats bar */}
        <div className="mt-3 grid grid-cols-4 gap-2">
          <MiniStat label="Total" value={stats.total} icon="📦" />
          <MiniStat label="Pending" value={stats.pending} icon="⏳" color="text-amber-400" />
          <MiniStat label="Approved" value={stats.approved} icon="✅" color="text-emerald-400" />
          <MiniStat label="Rejected" value={stats.rejected} icon="❌" color="text-red-400" />
        </div>

        {items.length > 1 && (
          <div className="mt-3 flex justify-end">
            <Button
              id="approve-all-btn"
              variant="outline"
              size="sm"
              onClick={handleApproveAll}
              disabled={actionLoading === "batch" || items.length === 0}
              className="text-xs border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/10"
            >
              {actionLoading === "batch" ? "⏳ Approving..." : `✅ Approve All (${items.length})`}
            </Button>
          </div>
        )}
      </CardHeader>

      <CardContent className="flex-1 p-0 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-auto">
          {items.length === 0 && !loading ? (
            <div className="flex flex-col items-center justify-center py-16 text-center animate-fade-in">
              <div className="mb-3 text-4xl opacity-50">🎉</div>
              <h3 className="text-sm font-medium text-foreground/80">
                Queue is clear!
              </h3>
              <p className="text-xs text-muted-foreground mt-1 max-w-xs">
                No pending items to review. Flagged responses will appear here
                for admin curation.
              </p>
            </div>
          ) : loading && items.length === 0 ? (
            <div className="space-y-3 p-4">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="h-20 rounded-xl animate-shimmer" />
              ))}
            </div>
          ) : (
            <div className="space-y-3 p-4">
              {items.map((item, idx) => (
                <div
                  key={item.id}
                  className="rounded-xl border border-border/30 bg-secondary/20 overflow-hidden transition-all duration-300 hover:border-border/50 animate-fade-in"
                  style={{ animationDelay: `${idx * 50}ms` }}
                >
                  {/* Collapsed Header */}
                  <button
                    className="w-full text-left px-4 py-3 flex items-center justify-between gap-3"
                    onClick={() => toggleExpand(item.id)}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-foreground/90 truncate font-medium">
                        {item.query}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <HybridScoreBadge score={item.hybrid_score} />
                        {item.hallucination_detected && (
                          <Badge className="text-[10px] bg-red-500/15 text-red-400 border-red-500/25 px-1.5 py-0" variant="outline">
                            ⚠ Hallucination
                          </Badge>
                        )}
                        <CompletenessStars rating={item.completeness} />
                        <span className="text-[10px] text-muted-foreground ml-auto">
                          {new Date(item.created_at).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                    <span className="text-muted-foreground text-xs flex-shrink-0">
                      {expandedId === item.id ? "▲" : "▼"}
                    </span>
                  </button>

                  {/* Expanded Detail */}
                  {expandedId === item.id && (
                    <div className="px-4 pb-4 space-y-3 animate-fade-in border-t border-border/20">
                      {/* Bad Response */}
                      <div className="mt-3">
                        <p className="text-[10px] font-semibold text-red-400 uppercase tracking-wider mb-1">
                          ❌ Bad LLM Response
                        </p>
                        <div className="rounded-lg bg-red-500/5 border border-red-500/15 p-3">
                          <p className="text-xs text-red-300/80 leading-relaxed whitespace-pre-wrap">
                            {item.bad_response}
                          </p>
                        </div>
                      </div>

                      {/* Teacher Correction */}
                      <div>
                        <p className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider mb-1">
                          ✅ Teacher Correction
                        </p>
                        {editingId === item.id ? (
                          <textarea
                            value={editText}
                            onChange={(e) => setEditText(e.target.value)}
                            className="w-full rounded-lg bg-emerald-500/5 border border-emerald-500/25 p-3 text-xs text-emerald-200 leading-relaxed min-h-[100px] resize-y focus:outline-none focus:ring-2 focus:ring-emerald-500/30 font-sans"
                            placeholder="Edit the correction..."
                          />
                        ) : (
                          <div className="rounded-lg bg-emerald-500/5 border border-emerald-500/15 p-3">
                            <p className="text-xs text-emerald-300/80 leading-relaxed whitespace-pre-wrap">
                              {item.teacher_correction}
                            </p>
                          </div>
                        )}
                      </div>

                      {/* Score Details */}
                      <div className="grid grid-cols-3 gap-2">
                        <ScoreDetail label="Cosine" value={item.eval_score} />
                        <ScoreDetail label="LLM Judge" value={item.llm_judge_score} />
                        <ScoreDetail label="Hybrid" value={item.hybrid_score} />
                      </div>

                      <Separator className="bg-border/20" />

                      {/* Action Buttons */}
                      <div className="flex items-center gap-2">
                        {editingId === item.id ? (
                          <>
                            <Button
                              id={`save-edit-${item.id}`}
                              size="sm"
                              onClick={() => handleEditApprove(item.id)}
                              disabled={actionLoading === item.id || !editText.trim()}
                              className="text-xs gradient-success hover:opacity-90"
                            >
                              {actionLoading === item.id ? "⏳" : "💾"} Save & Approve
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => { setEditingId(null); setEditText(""); }}
                              className="text-xs"
                            >
                              Cancel
                            </Button>
                          </>
                        ) : (
                          <>
                            <Button
                              id={`approve-${item.id}`}
                              size="sm"
                              onClick={() => handleApprove(item.id)}
                              disabled={actionLoading === item.id}
                              className="text-xs gradient-success hover:opacity-90"
                            >
                              {actionLoading === item.id ? "⏳" : "✅"} Approve
                            </Button>
                            <Button
                              id={`edit-${item.id}`}
                              variant="outline"
                              size="sm"
                              onClick={() => startEditing(item)}
                              className="text-xs border-blue-500/30 text-blue-400 hover:bg-blue-500/10"
                            >
                              ✏️ Edit
                            </Button>
                            <Button
                              id={`reject-${item.id}`}
                              variant="outline"
                              size="sm"
                              onClick={() => handleReject(item.id)}
                              disabled={actionLoading === item.id}
                              className="text-xs border-red-500/30 text-red-400 hover:bg-red-500/10"
                            >
                              {actionLoading === item.id ? "⏳" : "❌"} Reject
                            </Button>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ── Sub-components ─────────────────────────────────────────

function MiniStat({
  label,
  value,
  icon,
  color = "text-foreground",
}: {
  label: string;
  value: number;
  icon: string;
  color?: string;
}) {
  return (
    <div className="rounded-lg bg-secondary/30 border border-border/20 px-3 py-2 text-center">
      <span className="text-xs">{icon}</span>
      <p className={`text-lg font-bold tabular-nums ${color}`}>{value}</p>
      <p className="text-[10px] text-muted-foreground">{label}</p>
    </div>
  );
}

function HybridScoreBadge({ score }: { score: number }) {
  const pct = (score * 100).toFixed(0);
  const colorClass =
    score >= 0.7
      ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/25"
      : score >= 0.5
        ? "bg-amber-500/15 text-amber-400 border-amber-500/25"
        : "bg-red-500/15 text-red-400 border-red-500/25";

  return (
    <Badge className={`text-[10px] px-1.5 py-0 ${colorClass}`} variant="outline">
      {pct}%
    </Badge>
  );
}

function CompletenessStars({ rating }: { rating: number }) {
  return (
    <span className="text-[10px] text-amber-400/80" title={`Completeness: ${rating}/5`}>
      {"★".repeat(rating)}
      {"☆".repeat(5 - rating)}
    </span>
  );
}

function ScoreDetail({ label, value }: { label: string; value: number }) {
  const pct = (value * 100).toFixed(1);
  const color =
    value >= 0.7
      ? "text-emerald-400"
      : value >= 0.5
        ? "text-amber-400"
        : "text-red-400";

  return (
    <div className="rounded-lg bg-secondary/30 border border-border/20 px-3 py-2 text-center">
      <p className="text-[10px] text-muted-foreground mb-0.5">{label}</p>
      <p className={`text-sm font-bold tabular-nums ${color}`}>{pct}%</p>
    </div>
  );
}
