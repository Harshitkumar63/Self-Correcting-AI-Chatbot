"""
Chat Router — POST /api/v1/chat, POST /api/v1/chat/stream

Receives user queries, processes them through the ML pipeline
(inference → hybrid evaluation → teacher correction → curation),
saves the log, and returns results. Includes SSE streaming support.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Conversation, InteractionLog
from app.schemas import ChatRequest, ChatResponse
from app.services.ml_service import MLService, get_ml_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


async def _get_conversation_history(
    conversation_id: Optional[int], db: AsyncSession
) -> list[dict]:
    """Load recent messages from a conversation for multi-turn context."""
    if conversation_id is None:
        return []

    result = await db.execute(
        select(InteractionLog)
        .where(InteractionLog.conversation_id == conversation_id)
        .order_by(InteractionLog.created_at.desc())
        .limit(10)  # Last 10 messages for context
    )
    logs = result.scalars().all()

    # Build conversation history in chronological order
    history = []
    for log in reversed(list(logs)):
        history.append({"role": "user", "content": log.user_query})
        history.append({"role": "assistant", "content": log.llm_response})

    return history


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    ml: MLService = Depends(get_ml_service),
):
    """
    Process a user query through the full self-improving pipeline.

    1. Auto-create conversation if needed.
    2. Load conversation history for multi-turn context.
    3. Forward to inference engine for LLM response.
    4. Hybrid evaluation (cosine similarity + LLM-as-a-Judge).
    5. If flagged, generate teacher correction and add to curation queue.
    6. Save interaction log to database.
    7. Return response with all evaluation metrics.
    """
    # Handle conversation
    conversation_id = request.conversation_id
    if conversation_id is None:
        # Auto-create a new conversation
        title = request.query[:50] + ("..." if len(request.query) > 50 else "")
        conversation = Conversation(title=title)
        db.add(conversation)
        await db.flush()
        conversation_id = conversation.id

    # Get conversation history for multi-turn context
    history = await _get_conversation_history(conversation_id, db)

    # Run the ML pipeline with history
    result = await ml.process_query(request.query, history=history)

    # Save to database with all hybrid evaluation fields
    log_entry = InteractionLog(
        conversation_id=conversation_id,
        user_query=request.query,
        llm_response=result["response"],
        similarity_score=result["similarity_score"],
        evaluation_status=result["evaluation_status"],
        ground_truth_used=result["ground_truth"],
        matched_query=result["matched_query"],
        hybrid_score=result["hybrid_score"],
        llm_judge_score=result["llm_judge_score"],
        hallucination_detected=result["hallucination_detected"],
        completeness=result["completeness"],
        teacher_correction=result.get("teacher_correction"),
    )
    db.add(log_entry)
    await db.commit()

    logger.info(
        "Chat processed: cosine=%.4f hybrid=%.4f status=%s hallucination=%s conv=%d",
        result["similarity_score"],
        result["hybrid_score"],
        result["evaluation_status"],
        result["hallucination_detected"],
        conversation_id,
    )

    return ChatResponse(
        response=result["response"],
        similarity_score=result["similarity_score"],
        evaluation_status=result["evaluation_status"],
        ground_truth=result["ground_truth"],
        matched_query=result["matched_query"],
        feedback_status=result.get("feedback_status"),
        conversation_id=conversation_id,
        hybrid_score=result["hybrid_score"],
        llm_judge_score=result["llm_judge_score"],
        hallucination_detected=result["hallucination_detected"],
        completeness=result["completeness"],
        teacher_correction=result.get("teacher_correction"),
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    ml: MLService = Depends(get_ml_service),
):
    """
    SSE streaming endpoint — streams tokens then sends final evaluation.

    Events:
      - data: {"type": "token", "content": "..."} — streaming tokens
      - data: {"type": "eval", ...} — final evaluation result
      - data: {"type": "done"} — stream complete
    """
    # Handle conversation
    conversation_id = request.conversation_id
    if conversation_id is None:
        title = request.query[:50] + ("..." if len(request.query) > 50 else "")
        conversation = Conversation(title=title)
        db.add(conversation)
        await db.flush()
        conversation_id = conversation.id

    history = await _get_conversation_history(conversation_id, db)

    async def event_generator():
        try:
            # Stream tokens from inference
            full_response = ""
            for token in ml.generate_stream(request.query, history=history):
                full_response += token
                event_data = json.dumps({"type": "token", "content": token})
                yield f"data: {event_data}\n\n"

            # Run evaluation on the complete response
            eval_result = ml.evaluate_response(request.query, full_response)

            # Save to database
            log_entry = InteractionLog(
                conversation_id=conversation_id,
                user_query=request.query,
                llm_response=full_response,
                similarity_score=eval_result.score,
                evaluation_status=eval_result.status,
                ground_truth_used=eval_result.ground_truth_answer,
                matched_query=eval_result.matched_query,
                hybrid_score=eval_result.hybrid_score,
                llm_judge_score=eval_result.llm_judge_score,
                hallucination_detected=eval_result.hallucination_detected,
                completeness=eval_result.completeness,
            )
            db.add(log_entry)
            await db.commit()

            # Send evaluation result
            eval_data = json.dumps({
                "type": "eval",
                "similarity_score": eval_result.score,
                "evaluation_status": eval_result.status,
                "hybrid_score": eval_result.hybrid_score,
                "llm_judge_score": eval_result.llm_judge_score,
                "hallucination_detected": eval_result.hallucination_detected,
                "completeness": eval_result.completeness,
                "conversation_id": conversation_id,
            })
            yield f"data: {eval_data}\n\n"

            # Done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            error_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
