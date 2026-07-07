"""
Analytics Router — GET /api/v1/analytics/training-history,
                  GET /api/v1/analytics/score-timeline

Provides historical training metrics and score timeline data
for the Performance Dashboard.
"""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import InteractionLog
from ml_pipeline.trainer import get_all_training_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/training-history")
async def training_history():
    """
    Return all historical training run metrics.

    Each run includes: run_id, timestamps, samples_trained,
    final_loss, loss_history (per-step), hyperparameters.
    """
    metrics = get_all_training_metrics()
    return {
        "runs": metrics,
        "total_runs": len(metrics),
    }


@router.get("/score-timeline")
async def score_timeline(
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    bucket: str = Query("day", description="Bucket size: hour, day, week"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return average evaluation scores bucketed over time.

    Used to render the "Model Performance Over Time" line chart.
    Buckets: hour (for last 24h), day (for last 30d), week (for last 90d+).
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Fetch all logs after cutoff
    result = await db.execute(
        select(
            InteractionLog.created_at,
            InteractionLog.similarity_score,
            InteractionLog.hybrid_score,
            InteractionLog.evaluation_status,
        )
        .where(InteractionLog.created_at >= cutoff)
        .order_by(InteractionLog.created_at.asc())
    )
    rows = result.all()

    if not rows:
        return {"timeline": [], "bucket": bucket}

    # Group into buckets
    buckets: dict[str, list] = {}

    for row in rows:
        created = row.created_at
        if created is None:
            continue

        # Determine bucket key
        if bucket == "hour":
            key = created.strftime("%Y-%m-%d %H:00")
        elif bucket == "week":
            # Start of ISO week
            start_of_week = created - timedelta(days=created.weekday())
            key = start_of_week.strftime("%Y-%m-%d")
        else:  # day
            key = created.strftime("%Y-%m-%d")

        if key not in buckets:
            buckets[key] = []

        buckets[key].append({
            "cosine": row.similarity_score or 0,
            "hybrid": row.hybrid_score or 0,
            "passed": row.evaluation_status == "Passed",
        })

    # Aggregate per bucket
    timeline = []
    for key in sorted(buckets.keys()):
        entries = buckets[key]
        avg_cosine = sum(e["cosine"] for e in entries) / len(entries)
        avg_hybrid = sum(e["hybrid"] for e in entries) / len(entries)
        pass_rate = sum(1 for e in entries if e["passed"]) / len(entries)

        timeline.append({
            "date": key,
            "avg_cosine": round(avg_cosine, 4),
            "avg_hybrid": round(avg_hybrid, 4),
            "pass_rate": round(pass_rate, 4),
            "count": len(entries),
        })

    return {"timeline": timeline, "bucket": bucket}


@router.get("/score-distribution")
async def score_distribution(
    db: AsyncSession = Depends(get_db),
):
    """
    Return the distribution of hybrid scores in histogram buckets (0-10%, 10-20%, ..., 90-100%).

    Used to render a quality distribution bar chart.
    """
    result = await db.execute(
        select(InteractionLog.hybrid_score)
        .where(InteractionLog.hybrid_score.isnot(None))
    )
    scores = [row[0] for row in result.all() if row[0] is not None]

    # Create 10 buckets
    bucket_labels = [f"{i*10}-{(i+1)*10}%" for i in range(10)]
    bucket_counts = [0] * 10

    for score in scores:
        idx = min(int(score * 10), 9)  # 1.0 goes to bucket 9
        bucket_counts[idx] += 1

    distribution = [
        {"range": label, "count": count}
        for label, count in zip(bucket_labels, bucket_counts)
    ]

    return {
        "distribution": distribution,
        "total": len(scores),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
    }
