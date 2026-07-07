/**
 * API Client — Communicates with the FastAPI backend.
 *
 * All endpoints target the API_BASE from environment variable.
 * Includes JWT authentication headers and v1 API prefix.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const API_V1 = `${API_BASE}/api/v1`;

// ── Auth Helpers ───────────────────────────────────────────

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("auth_token");
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

// ── Auth Types ─────────────────────────────────────────────

export interface TokenResponse {
  access_token: string;
  token_type: string;
  role: string;
  username: string;
}

export interface UserResponse {
  id: number;
  username: string;
  role: string;
  created_at: string | null;
}

// ── Chat Types ─────────────────────────────────────────────

export interface ChatResponse {
  response: string;
  similarity_score: number;
  evaluation_status: "Passed" | "Flagged";
  ground_truth: string;
  matched_query: string;
  feedback_status: {
    curation_item_id?: string;
    queue_size?: number;
    sample_count: number;
    threshold: number;
    should_trigger: boolean;
  } | null;
  conversation_id: number | null;
  hybrid_score: number;
  llm_judge_score: number;
  hallucination_detected: boolean;
  completeness: number;
  teacher_correction: string | null;
}

// ── Conversation Types ─────────────────────────────────────

export interface ConversationResponse {
  id: number;
  title: string;
  user_id: number | null;
  created_at: string | null;
  updated_at: string | null;
  message_count: number;
}

export interface ConversationListResponse {
  conversations: ConversationResponse[];
  total: number;
}

export interface ConversationDetailResponse {
  id: number;
  title: string;
  messages: Array<{
    id: number;
    user_query: string;
    llm_response: string;
    similarity_score: number;
    evaluation_status: string;
    hybrid_score: number | null;
    llm_judge_score: number | null;
    hallucination_detected: boolean | null;
    completeness: number | null;
    teacher_correction: string | null;
    created_at: string | null;
  }>;
  created_at: string | null;
}

// ── Log Types ──────────────────────────────────────────────

export interface LogEntry {
  id: number;
  user_query: string;
  llm_response: string;
  similarity_score: number;
  evaluation_status: string;
  ground_truth_used: string | null;
  matched_query: string | null;
  hybrid_score: number | null;
  llm_judge_score: number | null;
  hallucination_detected: boolean | null;
  completeness: number | null;
  created_at: string | null;
}

export interface LogsResponse {
  logs: LogEntry[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface StatsResponse {
  total_queries: number;
  average_score: number;
  flagged_count: number;
  passed_count: number;
  training_queue_size: number;
  training_queue_threshold: number;
  training_queue_progress: number;
}

export interface ChartDataResponse {
  score_distribution: Array<{ range: string; count: number }>;
  trend_data: Array<{
    id: number;
    cosine_score: number;
    hybrid_score: number;
    status: string;
    created_at: string | null;
  }>;
  status_breakdown: {
    passed: number;
    flagged: number;
    total: number;
    pass_rate: number;
  };
}

// ── Tuning Types ───────────────────────────────────────────

export interface TuningResponse {
  message: string;
  training_status: string;
  samples_count: number;
  threshold: number;
  training_triggered: boolean;
  details?: Record<string, unknown>;
}

// ── Curation Types ─────────────────────────────────────────

export interface CurationItem {
  id: string;
  query: string;
  bad_response: string;
  teacher_correction: string;
  eval_score: number;
  hybrid_score: number;
  hallucination_detected: boolean;
  completeness: number;
  llm_judge_score: number;
  status: "pending" | "approved" | "rejected";
  created_at: string;
  reviewed_at?: string;
}

export interface CurationQueueResponse {
  items: CurationItem[];
  stats: {
    total: number;
    pending: number;
    approved: number;
    rejected: number;
  };
}

export interface CurationActionResponse {
  success: boolean;
  message: string;
  item: CurationItem | null;
}

export interface CurationStatsResponse {
  total: number;
  pending: number;
  approved: number;
  rejected: number;
}

// ── Ground Truth Types ─────────────────────────────────────

export interface GroundTruthEntry {
  query: string;
  answer: string;
}

export interface GroundTruthListResponse {
  entries: GroundTruthEntry[];
  total: number;
}

// ── Auth API ───────────────────────────────────────────────

export async function loginUser(
  username: string,
  password: string
): Promise<TokenResponse> {
  const res = await fetch(`${API_V1}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Login failed: ${res.status}`);
  }

  return res.json();
}

export async function fetchCurrentUser(): Promise<UserResponse> {
  const res = await fetch(`${API_V1}/auth/me`, { headers: authHeaders() });

  if (!res.ok) {
    throw new Error(`Auth error: ${res.status}`);
  }

  return res.json();
}

// ── Chat API ───────────────────────────────────────────────

export async function sendMessage(
  query: string,
  conversationId?: number
): Promise<ChatResponse> {
  const res = await fetch(`${API_V1}/chat`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({
      query,
      conversation_id: conversationId || null,
    }),
  });

  if (!res.ok) {
    throw new Error(`Chat API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function sendMessageStream(
  query: string,
  conversationId: number | undefined,
  onToken: (token: string) => void,
  onEval: (data: Record<string, unknown>) => void,
  onError?: (error: string) => void,
): Promise<void> {
  const res = await fetch(`${API_V1}/chat/stream`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({
      query,
      conversation_id: conversationId || null,
    }),
  });

  if (!res.ok) {
    throw new Error(`Stream error: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === "token") {
            onToken(data.content);
          } else if (data.type === "eval") {
            onEval(data);
          } else if (data.type === "error") {
            onError?.(data.message);
          }
        } catch {
          // Skip malformed JSON
        }
      }
    }
  }
}

// ── Conversation API ───────────────────────────────────────

export async function createConversation(
  title?: string
): Promise<ConversationResponse> {
  const res = await fetch(`${API_V1}/conversations`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ title: title || null }),
  });

  if (!res.ok) {
    throw new Error(`Create conversation error: ${res.status}`);
  }

  return res.json();
}

export async function fetchConversations(): Promise<ConversationListResponse> {
  const res = await fetch(`${API_V1}/conversations`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Conversations error: ${res.status}`);
  }

  return res.json();
}

export async function fetchConversation(
  id: number
): Promise<ConversationDetailResponse> {
  const res = await fetch(`${API_V1}/conversations/${id}`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Conversation error: ${res.status}`);
  }

  return res.json();
}

export async function deleteConversation(id: number): Promise<void> {
  const res = await fetch(`${API_V1}/conversations/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Delete conversation error: ${res.status}`);
  }
}

// ── Logs API ───────────────────────────────────────────────

export async function fetchLogs(
  page: number = 1,
  perPage: number = 20
): Promise<LogsResponse> {
  const res = await fetch(
    `${API_V1}/logs?page=${page}&per_page=${perPage}`,
    { headers: authHeaders() }
  );

  if (!res.ok) {
    throw new Error(`Logs API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_V1}/logs/stats`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Stats API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchChartData(): Promise<ChartDataResponse> {
  const res = await fetch(`${API_V1}/logs/chart-data`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Chart data error: ${res.status}`);
  }

  return res.json();
}

export async function exportLogs(format: "json" | "csv" = "json"): Promise<Blob> {
  const res = await fetch(`${API_V1}/logs/export?format=${format}`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Export error: ${res.status}`);
  }

  return res.blob();
}

// ── Tuning API ─────────────────────────────────────────────

export async function triggerTraining(): Promise<TuningResponse> {
  const res = await fetch(`${API_V1}/trigger-tuning`, {
    method: "POST",
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Tuning API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchTuningStatus(): Promise<TuningResponse> {
  const res = await fetch(`${API_V1}/tuning-status`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Tuning status error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

// ── Curation API ───────────────────────────────────────────

export async function fetchCurationQueue(): Promise<CurationQueueResponse> {
  const res = await fetch(`${API_V1}/curation/queue`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Curation queue error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchAllCurationItems(): Promise<CurationQueueResponse> {
  const res = await fetch(`${API_V1}/curation/all`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Curation all error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function approveCurationItem(
  itemId: string,
  editedCorrection?: string
): Promise<CurationActionResponse> {
  const res = await fetch(`${API_V1}/curation/approve/${itemId}`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({
      edited_correction: editedCorrection || null,
    }),
  });

  if (!res.ok) {
    throw new Error(`Approve error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function rejectCurationItem(
  itemId: string
): Promise<CurationActionResponse> {
  const res = await fetch(`${API_V1}/curation/reject/${itemId}`, {
    method: "POST",
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Reject error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function editCurationItem(
  itemId: string,
  editedCorrection: string
): Promise<CurationActionResponse> {
  const res = await fetch(`${API_V1}/curation/edit/${itemId}`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({ edited_correction: editedCorrection }),
  });

  if (!res.ok) {
    throw new Error(`Edit error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchCurationStats(): Promise<CurationStatsResponse> {
  const res = await fetch(`${API_V1}/curation/stats`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Curation stats error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

// ── Ground Truth API ───────────────────────────────────────

export async function fetchGroundTruth(): Promise<GroundTruthListResponse> {
  const res = await fetch(`${API_V1}/ground-truth`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Ground truth error: ${res.status}`);
  }

  return res.json();
}

export async function addGroundTruth(
  entry: GroundTruthEntry
): Promise<GroundTruthEntry> {
  const res = await fetch(`${API_V1}/ground-truth`, {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(entry),
  });

  if (!res.ok) {
    throw new Error(`Add ground truth error: ${res.status}`);
  }

  return res.json();
}

export async function updateGroundTruth(
  index: number,
  entry: GroundTruthEntry
): Promise<GroundTruthEntry> {
  const res = await fetch(`${API_V1}/ground-truth/${index}`, {
    method: "PUT",
    headers: authHeaders(),
    body: JSON.stringify(entry),
  });

  if (!res.ok) {
    throw new Error(`Update ground truth error: ${res.status}`);
  }

  return res.json();
}

export async function deleteGroundTruth(index: number): Promise<void> {
  const res = await fetch(`${API_V1}/ground-truth/${index}`, {
    method: "DELETE",
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Delete ground truth error: ${res.status}`);
  }
}

// ── Analytics Types ────────────────────────────────────────

export interface LossEntry {
  step: number;
  epoch: number;
  loss: number;
  learning_rate: number;
}

export interface TrainingRun {
  run_id: string;
  started_at: string;
  completed_at: string;
  samples_trained: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  lora_r: number;
  lora_alpha: number;
  final_loss: number | null;
  train_runtime: number | null;
  loss_history: LossEntry[];
  adapter_path: string;
}

export interface TrainingHistoryResponse {
  runs: TrainingRun[];
  total_runs: number;
}

export interface TimelineEntry {
  date: string;
  avg_cosine: number;
  avg_hybrid: number;
  pass_rate: number;
  count: number;
}

export interface ScoreTimelineResponse {
  timeline: TimelineEntry[];
  bucket: string;
}

export interface DistributionEntry {
  range: string;
  count: number;
}

export interface ScoreDistributionResponse {
  distribution: DistributionEntry[];
  total: number;
  avg_score: number;
}

// ── Analytics API ──────────────────────────────────────────

export async function fetchTrainingHistory(): Promise<TrainingHistoryResponse> {
  const res = await fetch(`${API_V1}/analytics/training-history`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Training history error: ${res.status}`);
  }

  return res.json();
}

export async function fetchScoreTimeline(
  days: number = 30,
  bucket: string = "day"
): Promise<ScoreTimelineResponse> {
  const res = await fetch(
    `${API_V1}/analytics/score-timeline?days=${days}&bucket=${bucket}`,
    { headers: authHeaders() }
  );

  if (!res.ok) {
    throw new Error(`Score timeline error: ${res.status}`);
  }

  return res.json();
}

export async function fetchScoreDistribution(): Promise<ScoreDistributionResponse> {
  const res = await fetch(`${API_V1}/analytics/score-distribution`, {
    headers: authHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Score distribution error: ${res.status}`);
  }

  return res.json();
}
