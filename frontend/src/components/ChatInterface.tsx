"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

import { sendMessage, fetchConversation, ChatResponse } from "@/lib/api";

// ── Types ──────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  score?: number;
  hybridScore?: number;
  llmJudgeScore?: number;
  status?: "Passed" | "Flagged";
  groundTruth?: string;
  teacherCorrection?: string;
  hallucinationDetected?: boolean;
  completeness?: number;
  timestamp: Date;
}

interface ChatInterfaceProps {
  conversationId?: number | null;
  onNewMessage?: () => void;
  onConversationCreated?: (id: number) => void;
}

// ── Component ──────────────────────────────────────────────

export default function ChatInterface({
  conversationId,
  onNewMessage,
  onConversationCreated,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load conversation history when conversationId changes
  useEffect(() => {
    if (!conversationId) {
      setMessages([]);
      return;
    }

    const loadHistory = async () => {
      try {
        const data = await fetchConversation(conversationId);
        const loaded: Message[] = [];
        for (const msg of data.messages) {
          loaded.push({
            id: `user-${msg.id}`,
            role: "user",
            content: msg.user_query,
            timestamp: new Date(msg.created_at || Date.now()),
          });
          loaded.push({
            id: `assistant-${msg.id}`,
            role: "assistant",
            content: msg.llm_response,
            score: msg.similarity_score,
            hybridScore: msg.hybrid_score ?? undefined,
            llmJudgeScore: msg.llm_judge_score ?? undefined,
            status: msg.evaluation_status as "Passed" | "Flagged",
            hallucinationDetected: msg.hallucination_detected ?? undefined,
            completeness: msg.completeness ?? undefined,
            teacherCorrection: msg.teacher_correction ?? undefined,
            timestamp: new Date(msg.created_at || Date.now()),
          });
        }
        setMessages(loaded);
      } catch {
        // Conversation might not exist yet
        setMessages([]);
      }
    };

    loadHistory();
  }, [conversationId]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = useCallback(async () => {
    const query = input.trim();
    if (!query || isLoading) return;

    // Add user message
    const userMsg: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: query,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const data: ChatResponse = await sendMessage(query, conversationId ?? undefined);

      // If a new conversation was created, notify parent
      if (!conversationId && data.conversation_id) {
        onConversationCreated?.(data.conversation_id);
      }

      const assistantMsg: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.response,
        score: data.similarity_score,
        hybridScore: data.hybrid_score,
        llmJudgeScore: data.llm_judge_score,
        status: data.evaluation_status,
        groundTruth: data.ground_truth,
        teacherCorrection: data.teacher_correction ?? undefined,
        hallucinationDetected: data.hallucination_detected,
        completeness: data.completeness,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
      onNewMessage?.();
    } catch (err) {
      const errorMsg: Message = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : "Failed to get response. Is the backend running?"}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [input, isLoading, conversationId, onNewMessage, onConversationCreated]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Card className="gradient-card border-border/50 flex h-full flex-col overflow-hidden">
      <CardHeader className="border-b border-border/30 pb-4">
        <div className="flex items-center gap-3">
          <div className="gradient-primary flex h-9 w-9 items-center justify-center rounded-lg text-sm">
            💬
          </div>
          <div>
            <CardTitle className="text-lg">Chat Interface</CardTitle>
            <p className="text-xs text-muted-foreground">
              Interact with the LLM — responses are auto-evaluated
            </p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col p-0 min-h-0">
        {/* Messages area — native scroll */}
        <div className="flex-1 min-h-0 overflow-y-auto p-4" ref={scrollRef}>
          <div className="space-y-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 text-center animate-fade-in">
                <div className="mb-4 text-5xl">🧠</div>
                <h3 className="text-lg font-semibold text-foreground/80">
                  Self-Improving Pipeline
                </h3>
                <p className="mt-1 max-w-sm text-sm text-muted-foreground">
                  Send a message to get an LLM response. Each answer is
                  evaluated using hybrid scoring (cosine + LLM judge).
                </p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={msg.id}
                className={`flex animate-fade-in ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
                style={{ animationDelay: `${idx * 50}ms` }}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                    msg.role === "user"
                      ? "gradient-primary text-primary-foreground"
                      : "bg-secondary/60 text-secondary-foreground"
                  }`}
                >
                  <p className="text-sm leading-relaxed whitespace-pre-wrap">
                    {msg.content}
                  </p>

                  {/* Evaluation badges for assistant messages */}
                  {msg.role === "assistant" && msg.hybridScore !== undefined && (
                    <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-white/10 pt-2">
                      {/* Status badge */}
                      <Badge
                        className={`text-xs font-medium ${
                          msg.status === "Passed"
                            ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                            : "bg-red-500/20 text-red-300 border-red-500/30"
                        }`}
                        variant="outline"
                      >
                        {msg.status === "Passed" ? "✓" : "⚠"} {msg.status}
                      </Badge>

                      {/* Hybrid score */}
                      <span className="text-xs text-muted-foreground">
                        Hybrid: <span className="font-semibold tabular-nums">
                          {((msg.hybridScore ?? 0) * 100).toFixed(1)}%
                        </span>
                      </span>

                      {/* Cosine score */}
                      <span className="text-[10px] text-muted-foreground/70">
                        cos:{((msg.score ?? 0) * 100).toFixed(0)}%
                      </span>

                      {/* Hallucination indicator */}
                      {msg.hallucinationDetected && (
                        <Badge className="text-[10px] bg-red-500/15 text-red-400 border-red-500/25 px-1.5 py-0" variant="outline">
                          ⚠ Hallucination
                        </Badge>
                      )}

                      {/* Completeness stars */}
                      {msg.completeness !== undefined && (
                        <span className="text-[10px] text-amber-400/80" title={`Completeness: ${msg.completeness}/5`}>
                          {"★".repeat(msg.completeness)}{"☆".repeat(5 - msg.completeness)}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Show teacher correction for flagged responses */}
                  {msg.role === "assistant" && msg.status === "Flagged" && msg.teacherCorrection && (
                    <div className="mt-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-3">
                      <p className="text-xs font-semibold text-emerald-400 mb-1">
                        🍎 Teacher Correction:
                      </p>
                      <p className="text-xs text-emerald-300/80 leading-relaxed">
                        {msg.teacherCorrection}
                      </p>
                    </div>
                  )}

                  {/* Fallback: show ground truth if no teacher correction */}
                  {msg.role === "assistant" && msg.status === "Flagged" && !msg.teacherCorrection && msg.groundTruth && (
                    <div className="mt-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-3">
                      <p className="text-xs font-semibold text-emerald-400 mb-1">
                        ✅ Expected Answer:
                      </p>
                      <p className="text-xs text-emerald-300/80 leading-relaxed">
                        {msg.groundTruth}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {isLoading && (
              <div className="flex justify-start animate-fade-in">
                <div className="rounded-2xl bg-secondary/60 px-5 py-4">
                  <div className="flex items-center gap-1.5">
                    <span className="typing-dot h-2 w-2 rounded-full bg-primary" />
                    <span className="typing-dot h-2 w-2 rounded-full bg-primary" />
                    <span className="typing-dot h-2 w-2 rounded-full bg-primary" />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Input area */}
        <div className="border-t border-border/30 p-4">
          <div className="flex items-center gap-3">
            <Input
              ref={inputRef}
              id="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask something... (e.g. 'What is machine learning?')"
              className="flex-1 border-border/40 bg-secondary/40 transition-all focus:border-primary/60 focus:ring-2 focus:ring-primary/20"
              disabled={isLoading}
              autoFocus
            />
            <Button
              id="chat-send-btn"
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="gradient-primary hover:opacity-90 transition-opacity px-6"
            >
              {isLoading ? (
                <span className="animate-spin">⏳</span>
              ) : (
                "Send"
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
