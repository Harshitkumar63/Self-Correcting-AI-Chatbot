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
