"""
Chat Router — POST /api/chat

Receives user queries, processes them through the ML pipeline
(inference → hybrid evaluation → teacher correction → curation),
saves the log, and returns results.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import InteractionLog
from app.schemas import ChatRequest, ChatResponse
from app.services.ml_service import MLService, get_ml_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    ml: MLService = Depends(get_ml_service),
):
    """
    Process a user query through the full self-improving pipeline.

    1. Forward to inference engine for LLM response.
    2. Hybrid evaluation (cosine similarity + LLM-as-a-Judge).
    3. If flagged, generate teacher correction and add to curation queue.
    4. Save interaction log to database.
    5. Return response with all evaluation metrics.
    """
    # Run the ML pipeline
    result = await ml.process_query(request.query)

    # Save to database
    log_entry = InteractionLog(
        user_query=request.query,
        llm_response=result["response"],
        similarity_score=result["similarity_score"],
        evaluation_status=result["evaluation_status"],
        ground_truth_used=result["ground_truth"],
        matched_query=result["matched_query"],
    )
    db.add(log_entry)
    await db.commit()

    logger.info(
        "Chat processed: cosine=%.4f hybrid=%.4f status=%s hallucination=%s",
        result["similarity_score"],
        result["hybrid_score"],
        result["evaluation_status"],
        result["hallucination_detected"],
    )

    return ChatResponse(
        response=result["response"],
        similarity_score=result["similarity_score"],
        evaluation_status=result["evaluation_status"],
        ground_truth=result["ground_truth"],
        matched_query=result["matched_query"],
        feedback_status=result.get("feedback_status"),
        # New hybrid evaluator fields
        hybrid_score=result["hybrid_score"],
        llm_judge_score=result["llm_judge_score"],
        hallucination_detected=result["hallucination_detected"],
        completeness=result["completeness"],
        teacher_correction=result.get("teacher_correction"),
    )
