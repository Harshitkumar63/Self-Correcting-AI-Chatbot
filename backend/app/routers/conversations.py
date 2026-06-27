"""
Conversations Router — CRUD endpoints for chat sessions.

Provides endpoints to create, list, view, and delete conversations.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Conversation, InteractionLog
from app.schemas import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    body: ConversationCreate = ConversationCreate(),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new conversation session.
    """
    title = body.title or "New Conversation"
    conversation = Conversation(title=title)
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    logger.info("Created conversation: id=%d title='%s'", conversation.id, title)

    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        user_id=conversation.user_id,
        created_at=conversation.created_at.isoformat() if conversation.created_at else None,
        updated_at=conversation.updated_at.isoformat() if conversation.updated_at else None,
        message_count=0,
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    db: AsyncSession = Depends(get_db),
):
    """
    List all conversations, newest first.
    """
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()

    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c.id,
                title=c.title,
                user_id=c.user_id,
                created_at=c.created_at.isoformat() if c.created_at else None,
                updated_at=c.updated_at.isoformat() if c.updated_at else None,
                message_count=len(c.messages) if c.messages else 0,
            )
            for c in conversations
        ],
        total=len(conversations),
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a conversation with its full message history.
    """
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat() if conversation.created_at else None,
        messages=[msg.to_dict() for msg in conversation.messages],
    )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a conversation and all its messages.
    """
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conversation)
    await db.commit()

    logger.info("Deleted conversation: id=%d", conversation_id)

    return {"success": True, "message": f"Conversation {conversation_id} deleted"}
