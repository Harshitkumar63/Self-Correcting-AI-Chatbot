"""
Pydantic schemas for request/response validation.

Includes input sanitization validators to strip HTML/script tags.
"""

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Sanitization Helpers ────────────────────────────────────

def _sanitize_text(text: str) -> str:
    """Strip HTML tags and script content from text input."""
    # Remove script tags and their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove all HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Auth ────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    """Login credentials."""
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=200)


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class UserCreate(BaseModel):
    """Request to create a new user (admin only)."""
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=200)
    role: str = Field(default="user", pattern="^(admin|user)$")


class UserResponse(BaseModel):
    """User profile response."""
    id: int
    username: str
    role: str
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


# ── Chat ────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str = Field(
        ..., min_length=1, max_length=2000, description="The user's query"
    )
    conversation_id: Optional[int] = Field(
        None, description="Optional conversation ID for context"
    )

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Strip HTML/script tags from user input."""
        return _sanitize_text(v)


class ChatResponse(BaseModel):
    """Response returned after processing a chat query."""

    response: str
    similarity_score: float
    evaluation_status: str  # "Passed" | "Flagged"
    ground_truth: str
    matched_query: str
    feedback_status: Optional[dict] = None
    conversation_id: Optional[int] = None

    # Hybrid evaluator fields
    hybrid_score: float = 0.0
    llm_judge_score: float = 0.0
    hallucination_detected: bool = False
    completeness: int = 3
    teacher_correction: Optional[str] = None


# ── Conversations ───────────────────────────────────────────


class ConversationCreate(BaseModel):
    """Request to create a new conversation."""
    title: Optional[str] = Field(None, max_length=200)


class ConversationResponse(BaseModel):
    """Single conversation with message count."""
    id: int
    title: str
    user_id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message_count: int = 0

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """List of conversations."""
    conversations: list[ConversationResponse]
    total: int


class ConversationDetailResponse(BaseModel):
    """Conversation with full message history."""
    id: int
    title: str
    messages: list[dict]
    created_at: Optional[str] = None


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
    hybrid_score: Optional[float] = None
    llm_judge_score: Optional[float] = None
    hallucination_detected: Optional[bool] = None
    completeness: Optional[int] = None
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
    training_queue_progress: float


class ChartDataResponse(BaseModel):
    """Data for analytics charts."""
    score_distribution: list[dict]
    trend_data: list[dict]
    status_breakdown: dict


# ── Tuning ──────────────────────────────────────────────────


class TuningResponse(BaseModel):
    """Response from the training trigger endpoint."""

    message: str
    training_status: str
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

    @field_validator('edited_correction')
    @classmethod
    def sanitize_correction(cls, v: Optional[str]) -> Optional[str]:
        """Strip HTML/script tags from edited corrections."""
        if v is not None:
            return _sanitize_text(v)
        return v


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


# ── Ground Truth Management ─────────────────────────────────


class GroundTruthEntry(BaseModel):
    """A single ground truth Q&A pair."""
    query: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=5000)


class GroundTruthListResponse(BaseModel):
    """List of all ground truth entries."""
    entries: list[GroundTruthEntry]
    total: int
