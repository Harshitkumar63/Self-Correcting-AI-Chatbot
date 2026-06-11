"""
Logs Router — GET /api/logs, GET /api/logs/stats

Provides paginated interaction logs and dashboard statistics.
"""

import logging
import math

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import InteractionLog
from app.schemas import LogEntry, LogsResponse, StatsResponse
from app.services.ml_service import MLService, get_ml_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["logs"])


@router.get("/logs", response_model=LogsResponse)
async def get_logs(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch paginated interaction logs, newest first.

    Returns log entries with query, response, score, and status
    for display in the Evaluation Monitor.
    """
    # Count total
    count_result = await db.execute(
        select(func.count()).select_from(InteractionLog)
    )
    total = count_result.scalar() or 0

    # Fetch page
    offset = (page - 1) * per_page
    result = await db.execute(
        select(InteractionLog)
        .order_by(InteractionLog.created_at.desc())
        .offset(offset)
        .limit(per_page)
    )
    logs = result.scalars().all()

    total_pages = math.ceil(total / per_page) if total > 0 else 1

    return LogsResponse(
        logs=[
            LogEntry(
                id=log.id,
                user_query=log.user_query,
                llm_response=log.llm_response,
                similarity_score=log.similarity_score,
                evaluation_status=log.evaluation_status,
                ground_truth_used=log.ground_truth_used,
                matched_query=log.matched_query,
                created_at=log.created_at.isoformat() if log.created_at else None,
            )
            for log in logs
        ],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
    )


@router.get("/logs/stats", response_model=StatsResponse)
async def get_stats(
    db: AsyncSession = Depends(get_db),
    ml: MLService = Depends(get_ml_service),
):
    """
    Get summary statistics for the dashboard.

    Returns total queries, average score, flagged/passed counts,
    and training queue status.
    """
    # Total queries
    count_result = await db.execute(
        select(func.count()).select_from(InteractionLog)
    )
    total = count_result.scalar() or 0

    # Average score
    avg_result = await db.execute(
        select(func.avg(InteractionLog.similarity_score))
    )
    avg_score = avg_result.scalar() or 0.0

    # Flagged count
    flagged_result = await db.execute(
        select(func.count())
        .select_from(InteractionLog)
        .where(InteractionLog.evaluation_status == "Flagged")
    )
    flagged = flagged_result.scalar() or 0

    # Passed count
    passed_result = await db.execute(
        select(func.count())
        .select_from(InteractionLog)
        .where(InteractionLog.evaluation_status == "Passed")
    )
    passed = passed_result.scalar() or 0

    # Training queue status
    feedback_status = ml.get_feedback_status()

    return StatsResponse(
        total_queries=total,
        average_score=round(float(avg_score), 4),
        flagged_count=flagged,
        passed_count=passed,
        training_queue_size=feedback_status["sample_count"],
        training_queue_threshold=feedback_status["threshold"],
        training_queue_progress=feedback_status["progress_pct"],
    )
