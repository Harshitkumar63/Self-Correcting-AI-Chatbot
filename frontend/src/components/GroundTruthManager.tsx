"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  fetchGroundTruth,
  addGroundTruth,
  updateGroundTruth,
  deleteGroundTruth,
  GroundTruthEntry,
} from "@/lib/api";

export default function GroundTruthManager() {
  const [entries, setEntries] = useState<GroundTruthEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [editQuery, setEditQuery] = useState("");
  const [editAnswer, setEditAnswer] = useState("");
  const [showAdd, setShowAdd] = useState(false);
  const [newQuery, setNewQuery] = useState("");
  const [newAnswer, setNewAnswer] = useState("");

  const loadEntries = useCallback(async () => {
    try {
      const data = await fetchGroundTruth();
      setEntries(data.entries);
    } catch {
      // Backend may not be running
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadEntries();
  }, [loadEntries]);

  const handleAdd = async () => {
    if (!newQuery.trim() || !newAnswer.trim()) return;
    try {
      await addGroundTruth({ query: newQuery.trim(), answer: newAnswer.trim() });
      setNewQuery("");
      setNewAnswer("");
      setShowAdd(false);
      await loadEntries();
    } catch {
      // Error adding
    }
  };

  const handleEdit = (idx: number) => {
    setEditingIdx(idx);
    setEditQuery(entries[idx].query);
    setEditAnswer(entries[idx].answer);
  };

  const handleSave = async () => {
    if (editingIdx === null || !editQuery.trim() || !editAnswer.trim()) return;
    try {
      await updateGroundTruth(editingIdx, {
        query: editQuery.trim(),
        answer: editAnswer.trim(),
      });
      setEditingIdx(null);
      await loadEntries();
    } catch {
      // Error updating
    }
  };

  const handleDelete = async (idx: number) => {
    try {
      await deleteGroundTruth(idx);
      await loadEntries();
    } catch {
      // Error deleting
    }
  };

  const filtered = entries.filter(
    (e) =>
      e.query.toLowerCase().includes(searchQuery.toLowerCase()) ||
      e.answer.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Card className="gradient-card border-border/50 flex flex-col h-full overflow-hidden">
      <CardHeader className="border-b border-border/30 px-5 py-3 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="gradient-success flex h-8 w-8 items-center justify-center rounded-lg text-sm">
              📚
            </div>
            <div>
              <CardTitle className="text-base">Knowledge Base</CardTitle>
              <p className="text-[11px] text-muted-foreground">
                {entries.length} ground truth Q&A pairs
              </p>
            </div>
          </div>
          <Button
            id="add-gt-btn"
            onClick={() => setShowAdd(!showAdd)}
            className="text-xs h-8 gradient-primary hover:opacity-90"
          >
            {showAdd ? "Cancel" : "➕ Add Entry"}
          </Button>
        </div>

        {/* Search */}
        <div className="mt-3">
          <Input
            id="gt-search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search Q&A pairs..."
            className="text-xs h-8 border-border/40 bg-secondary/40"
          />
        </div>

        {/* Add Form */}
        {showAdd && (
          <div className="mt-3 space-y-2 animate-fade-in rounded-lg bg-secondary/30 p-3 border border-border/20">
            <Input
              id="gt-new-query"
              value={newQuery}
              onChange={(e) => setNewQuery(e.target.value)}
              placeholder="Question..."
              className="text-xs h-8 border-border/40 bg-secondary/40"
            />
            <textarea
              id="gt-new-answer"
              value={newAnswer}
              onChange={(e) => setNewAnswer(e.target.value)}
              placeholder="Answer..."
              className="w-full rounded-md border border-border/40 bg-secondary/40 px-3 py-2 text-xs text-foreground placeholder:text-muted-foreground/50 resize-none h-20"
            />
            <Button
              onClick={handleAdd}
              disabled={!newQuery.trim() || !newAnswer.trim()}
              className="w-full text-xs h-8 gradient-primary hover:opacity-90"
            >
              Save Entry
            </Button>
          </div>
        )}
      </CardHeader>

      <CardContent className="flex-1 p-0 overflow-y-auto">
        {loading ? (
          <div className="p-4 space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 animate-shimmer rounded-lg" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="mb-3 text-3xl opacity-50">📚</div>
            <p className="text-sm text-muted-foreground">
              {searchQuery ? "No matching entries" : "No ground truth entries yet"}
            </p>
          </div>
        ) : (
          <div className="divide-y divide-border/20">
            {filtered.map((entry, idx) => {
              const realIdx = entries.indexOf(entry);
              const isEditing = editingIdx === realIdx;

              return (
                <div
                  key={realIdx}
                  className="px-4 py-3 hover:bg-secondary/20 transition-colors animate-fade-in"
                  style={{ animationDelay: `${idx * 20}ms` }}
                >
                  {isEditing ? (
                    <div className="space-y-2">
                      <Input
                        value={editQuery}
                        onChange={(e) => setEditQuery(e.target.value)}
                        className="text-xs h-8 border-border/40 bg-secondary/40"
                      />
                      <textarea
                        value={editAnswer}
                        onChange={(e) => setEditAnswer(e.target.value)}
                        className="w-full rounded-md border border-border/40 bg-secondary/40 px-3 py-2 text-xs text-foreground resize-none h-16"
                      />
                      <div className="flex gap-2">
                        <Button
                          onClick={handleSave}
                          className="text-xs h-7 gradient-primary hover:opacity-90"
                        >
                          Save
                        </Button>
                        <Button
                          onClick={() => setEditingIdx(null)}
                          variant="outline"
                          className="text-xs h-7"
                        >
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-foreground/90">
                            Q: {entry.query}
                          </p>
                          <p className="text-[11px] text-muted-foreground mt-1 line-clamp-2">
                            A: {entry.answer}
                          </p>
                        </div>
                        <div className="flex items-center gap-1 flex-shrink-0">
                          <Badge className="text-[9px] bg-secondary/50 text-muted-foreground border-0">
                            #{realIdx}
                          </Badge>
                          <button
                            onClick={() => handleEdit(realIdx)}
                            className="text-xs text-primary/70 hover:text-primary p-1"
                            title="Edit"
                          >
                            ✏️
                          </button>
                          <button
                            onClick={() => handleDelete(realIdx)}
                            className="text-xs text-red-400/70 hover:text-red-400 p-1"
                            title="Delete"
                          >
                            🗑️
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
