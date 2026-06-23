/**
 * API Client — Communicates with the FastAPI backend.
 *
 * All endpoints target http://localhost:8000 (the FastAPI server).
 */

const API_BASE = "http://localhost:8000";

// ── Types ──────────────────────────────────────────────────

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
  // Hybrid evaluator fields
  hybrid_score: number;
  llm_judge_score: number;
  hallucination_detected: boolean;
  completeness: number;
  teacher_correction: string | null;
}

export interface LogEntry {
  id: number;
  user_query: string;
  llm_response: string;
  similarity_score: number;
  evaluation_status: string;
  ground_truth_used: string | null;
  matched_query: string | null;
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

// ── API Functions ──────────────────────────────────────────

export async function sendMessage(query: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    throw new Error(`Chat API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchLogs(
  page: number = 1,
  perPage: number = 20
): Promise<LogsResponse> {
  const res = await fetch(
    `${API_BASE}/api/logs?page=${page}&per_page=${perPage}`
  );

  if (!res.ok) {
    throw new Error(`Logs API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE}/api/logs/stats`);

  if (!res.ok) {
    throw new Error(`Stats API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function triggerTraining(): Promise<TuningResponse> {
  const res = await fetch(`${API_BASE}/api/trigger-tuning`, {
    method: "POST",
  });

  if (!res.ok) {
    throw new Error(`Tuning API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchTuningStatus(): Promise<TuningResponse> {
  const res = await fetch(`${API_BASE}/api/tuning-status`);

  if (!res.ok) {
    throw new Error(`Tuning status error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

// ── Curation API Functions ─────────────────────────────────

export async function fetchCurationQueue(): Promise<CurationQueueResponse> {
  const res = await fetch(`${API_BASE}/api/curation/queue`);

  if (!res.ok) {
    throw new Error(`Curation queue error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchAllCurationItems(): Promise<CurationQueueResponse> {
  const res = await fetch(`${API_BASE}/api/curation/all`);

  if (!res.ok) {
    throw new Error(`Curation all error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function approveCurationItem(
  itemId: string,
  editedCorrection?: string
): Promise<CurationActionResponse> {
  const res = await fetch(`${API_BASE}/api/curation/approve/${itemId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
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
  const res = await fetch(`${API_BASE}/api/curation/reject/${itemId}`, {
    method: "POST",
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
  const res = await fetch(`${API_BASE}/api/curation/edit/${itemId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ edited_correction: editedCorrection }),
  });

  if (!res.ok) {
    throw new Error(`Edit error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export async function fetchCurationStats(): Promise<CurationStatsResponse> {
  const res = await fetch(`${API_BASE}/api/curation/stats`);

  if (!res.ok) {
    throw new Error(`Curation stats error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}
