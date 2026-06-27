"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  fetchConversations,
  createConversation,
  deleteConversation,
  ConversationResponse,
} from "@/lib/api";

interface ConversationSidebarProps {
  activeId: number | null;
  onSelect: (id: number) => void;
  onNew: (id: number) => void;
}

export default function ConversationSidebar({
  activeId,
  onSelect,
  onNew,
}: ConversationSidebarProps) {
  const [conversations, setConversations] = useState<ConversationResponse[]>([]);
  const [loading, setLoading] = useState(false);

  const loadConversations = useCallback(async () => {
    try {
      const data = await fetchConversations();
      setConversations(data.conversations);
    } catch {
      // Backend may not be running
    }
  }, []);

  useEffect(() => {
    loadConversations();
    const interval = setInterval(loadConversations, 10000);
    return () => clearInterval(interval);
  }, [loadConversations]);

  const handleNew = async () => {
    setLoading(true);
    try {
      const conv = await createConversation();
      setConversations((prev) => [conv, ...prev]);
      onNew(conv.id);
    } catch {
      // Error creating conversation
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    try {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) {
        onSelect(conversations[0]?.id || 0);
      }
    } catch {
      // Error
    }
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return "";
    const d = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <div className="flex h-full flex-col gradient-card border border-border/50 rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="border-b border-border/30 px-4 py-3 flex-shrink-0">
        <Button
          id="new-conversation-btn"
          onClick={handleNew}
          disabled={loading}
          className="w-full gradient-primary hover:opacity-90 transition-opacity text-xs h-9"
        >
          {loading ? "Creating..." : "✨ New Conversation"}
        </Button>
      </div>

      {/* Conversation List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {conversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <div className="mb-2 text-2xl opacity-50">💬</div>
            <p className="text-xs text-muted-foreground">
              No conversations yet
            </p>
          </div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => onSelect(conv.id)}
              className={`group relative flex cursor-pointer items-center gap-3 rounded-xl px-3 py-2.5 transition-all ${
                activeId === conv.id
                  ? "bg-primary/15 border border-primary/30"
                  : "hover:bg-secondary/40 border border-transparent"
              }`}
            >
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-foreground/90 truncate">
                  {conv.title}
                </p>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[10px] text-muted-foreground">
                    {conv.message_count} msgs
                  </span>
                  <span className="text-[10px] text-muted-foreground/60">
                    {formatDate(conv.updated_at)}
                  </span>
                </div>
              </div>
              {/* Delete button */}
              <button
                onClick={(e) => handleDelete(e, conv.id)}
                className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground/40 hover:text-red-400 text-xs p-1"
                title="Delete conversation"
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
