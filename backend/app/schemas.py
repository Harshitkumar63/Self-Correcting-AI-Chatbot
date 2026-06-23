"""
Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Chat ────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str = Field(
        ..., min_length=1, max_length=2000, description="The user's query"
    )


class ChatResponse(BaseModel):
    """Response returned after processing a chat query."""

    response: str
    similarity_score: float
    evaluation_status: str  # "Passed" | "Flagged"
    ground_truth: str
    matched_query: str
    feedback_status: Optional[dict] = None  # feedback collection info

    # ── New hybrid evaluator fields ────────────────────────
    hybrid_score: float = 0.0
    llm_judge_score: float = 0.0
    hallucination_detected: bool = False
    completeness: int = 3
    teacher_correction: Optional[str] = None


# ── Logs ────────────────────────────────────────────────────


class LogEntry(BaseModel):
    """Single interaction log entry."""

    id: int
    user_query: str
    llm_response: str
    similarity_score: float
    evaluation_status: str
    ground_truth_used: Optional[str] = None
    matched_query: Optional[str] = None
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


class LogsResponse(BaseModel):
    """Paginated logs response."""

    logs: list[LogEntry]
    total: int
    page: int
    per_page: int
    total_pages: int


class StatsResponse(BaseModel):
    """Summary statistics for the dashboard."""

    total_queries: int
    average_score: float
    flagged_count: int
    passed_count: int
    training_queue_size: int
    training_queue_threshold: int
    training_queue_progress: float  # percentage


# ── Tuning ──────────────────────────────────────────────────


class TuningResponse(BaseModel):
    """Response from the training trigger endpoint."""

    message: str
    training_status: str  # "idle" | "preparing" | "training" | "completed" | "failed"
    samples_count: int
    threshold: int
    training_triggered: bool
    details: Optional[dict] = None


# ── Curation Queue ──────────────────────────────────────────


class CurationItem(BaseModel):
    """A single item in the curation queue."""

    id: str
    query: str
    bad_response: str
    teacher_correction: str
    eval_score: float
    hybrid_score: float
    hallucination_detected: bool
    completeness: int
    llm_judge_score: float
    status: str  # "pending" | "approved" | "rejected"
    created_at: str
    reviewed_at: Optional[str] = None


class CurationQueueResponse(BaseModel):
    """Response containing the curation queue items and stats."""

    items: list[CurationItem]
    stats: dict


class CurationActionRequest(BaseModel):
    """Request body for approve/edit curation actions."""

    edited_correction: Optional[str] = Field(
        None, description="Edited correction text (optional, for edit-and-approve)"
    )


class CurationActionResponse(BaseModel):
    """Response from a curation action."""

    success: bool
    message: str
    item: Optional[CurationItem] = None


class CurationStatsResponse(BaseModel):
    """Curation queue statistics."""

    total: int
    pending: int
    approved: int
    rejected: int
