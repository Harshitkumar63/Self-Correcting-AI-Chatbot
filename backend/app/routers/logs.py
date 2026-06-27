"""
Logs Router — GET /api/v1/logs, GET /api/v1/logs/stats, GET /api/v1/logs/export, GET /api/v1/logs/chart-data

Provides paginated interaction logs, dashboard statistics, export, and chart data.
"""

import csv
import io
import json
import logging
import math

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import InteractionLog
from app.schemas import ChartDataResponse, LogEntry, LogsResponse, StatsResponse
from app.services.ml_service import MLService, get_ml_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["logs"])


@router.get("/logs", response_model=LogsResponse)
async def get_logs(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch paginated interaction logs, newest first.
    """
    count_result = await db.execute(
        select(func.count()).select_from(InteractionLog)
    )
    total = count_result.scalar() or 0

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
                hybrid_score=log.hybrid_score,
                llm_judge_score=log.llm_judge_score,
                hallucination_detected=log.hallucination_detected,
                completeness=log.completeness,
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
    """
    count_result = await db.execute(
        select(func.count()).select_from(InteractionLog)
    )
    total = count_result.scalar() or 0

    avg_result = await db.execute(
        select(func.avg(InteractionLog.similarity_score))
    )
    avg_score = avg_result.scalar() or 0.0

    flagged_result = await db.execute(
        select(func.count())
        .select_from(InteractionLog)
        .where(InteractionLog.evaluation_status == "Flagged")
    )
    flagged = flagged_result.scalar() or 0

    passed_result = await db.execute(
        select(func.count())
        .select_from(InteractionLog)
        .where(InteractionLog.evaluation_status == "Passed")
    )
    passed = passed_result.scalar() or 0

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


@router.get("/logs/chart-data", response_model=ChartDataResponse)
async def get_chart_data(
    db: AsyncSession = Depends(get_db),
):
    """
    Get data formatted for analytics charts.
    """
    # Score distribution (buckets of 10%)
    result = await db.execute(
        select(InteractionLog.similarity_score, InteractionLog.hybrid_score)
        .order_by(InteractionLog.created_at.desc())
        .limit(200)
    )
    rows = result.all()

    # Build score distribution histogram
    buckets = {f"{i*10}-{i*10+10}%": 0 for i in range(10)}
    for row in rows:
        score = row.hybrid_score if row.hybrid_score else row.similarity_score
        bucket_idx = min(int(score * 10), 9)
        bucket_key = f"{bucket_idx*10}-{bucket_idx*10+10}%"
        buckets[bucket_key] += 1

    score_distribution = [
        {"range": k, "count": v} for k, v in buckets.items()
    ]

    # Trend data (last 50 queries in chronological order)
    trend_result = await db.execute(
        select(
            InteractionLog.id,
            InteractionLog.similarity_score,
            InteractionLog.hybrid_score,
            InteractionLog.evaluation_status,
            InteractionLog.created_at,
        )
        .order_by(InteractionLog.created_at.desc())
        .limit(50)
    )
    trend_rows = trend_result.all()

    trend_data = [
        {
            "id": row.id,
            "cosine_score": round(row.similarity_score, 4),
            "hybrid_score": round(row.hybrid_score, 4) if row.hybrid_score else round(row.similarity_score, 4),
            "status": row.evaluation_status,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in reversed(list(trend_rows))
    ]

    # Status breakdown
    total = len(rows)
    passed = sum(1 for r in rows if r.similarity_score >= 0.7)
    flagged = total - passed

    status_breakdown = {
        "passed": passed,
        "flagged": flagged,
        "total": total,
        "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
    }

    return ChartDataResponse(
        score_distribution=score_distribution,
        trend_data=trend_data,
        status_breakdown=status_breakdown,
    )


@router.get("/logs/export")
async def export_logs(
    format: str = Query("json", description="Export format: json or csv"),
    db: AsyncSession = Depends(get_db),
):
    """
    Export all interaction logs as JSON or CSV.
    """
    result = await db.execute(
        select(InteractionLog).order_by(InteractionLog.created_at.desc())
    )
    logs = result.scalars().all()

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "user_query", "llm_response", "similarity_score",
            "hybrid_score", "evaluation_status", "hallucination_detected",
            "completeness", "ground_truth_used", "created_at",
        ])
        for log in logs:
            writer.writerow([
                log.id, log.user_query, log.llm_response, log.similarity_score,
                log.hybrid_score, log.evaluation_status, log.hallucination_detected,
                log.completeness, log.ground_truth_used,
                log.created_at.isoformat() if log.created_at else "",
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=interaction_logs.csv"},
        )
    else:
        data = [log.to_dict() for log in logs]
        return StreamingResponse(
            iter([json.dumps(data, indent=2)]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=interaction_logs.json"},
        )
