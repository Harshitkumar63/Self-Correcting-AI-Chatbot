"""
Tuning Router — POST /api/trigger-tuning, GET /api/tuning-status

Handles manual and automatic triggering of LoRA fine-tuning,
and reports training status.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends

from app.schemas import TuningResponse
from app.services.ml_service import MLService, get_ml_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tuning"])


@router.post("/trigger-tuning", response_model=TuningResponse)
async def trigger_tuning(
    background_tasks: BackgroundTasks,
    ml: MLService = Depends(get_ml_service),
):
    """
    Manually trigger a LoRA fine-tuning run.

    The training runs as a background task so the API responds
    immediately. Check /api/tuning-status for progress.
    """
    feedback_status = ml.get_feedback_status()
    sample_count = feedback_status["sample_count"]
    threshold = feedback_status["threshold"]

    if sample_count == 0:
        return TuningResponse(
            message="No training samples available. Send more queries to collect flagged responses.",
            training_status="idle",
            samples_count=sample_count,
            threshold=threshold,
            training_triggered=False,
        )

    # Run training in background
    background_tasks.add_task(ml.trigger_training)

    return TuningResponse(
        message=f"Fine-tuning triggered with {sample_count} samples. Training started in background.",
        training_status="preparing",
        samples_count=sample_count,
        threshold=threshold,
        training_triggered=True,
    )


@router.get("/tuning-status", response_model=TuningResponse)
async def tuning_status(
    ml: MLService = Depends(get_ml_service),
):
    """
    Check the current status of the fine-tuning process.
    """
    training = ml.get_training_status()
    feedback = ml.get_feedback_status()

    return TuningResponse(
        message=training.get("message", "No training in progress."),
        training_status=training.get("state", "idle"),
        samples_count=feedback["sample_count"],
        threshold=feedback["threshold"],
        training_triggered=training.get("state", "idle") in ("preparing", "training"),
        details=training,
    )
