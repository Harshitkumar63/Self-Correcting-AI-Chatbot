"""
SQLAlchemy ORM Models for the self-improving LLM pipeline.

Includes:
- User: Authentication and role management
- Conversation: Chat session grouping
- InteractionLog: Individual query/response records with evaluation metrics
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
)
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    """
    User accounts for authentication and role-based access control.

    Roles:
        - 'admin': Full access including curation queue and training triggers.
        - 'user': Chat and log viewing only.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="user")  # "admin" | "user"
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class Conversation(Base):
    """
    Groups related chat messages into a conversation session.

    Each conversation belongs to a user and contains multiple interaction logs.
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False, default="New Conversation")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "InteractionLog", back_populates="conversation",
        cascade="all, delete-orphan", order_by="InteractionLog.created_at",
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}')>"

    def to_dict(self) -> dict:
        """Serialize to dictionary for API responses."""
        return {
            "id": self.id,
            "title": self.title,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": len(self.messages) if self.messages else 0,
        }


class InteractionLog(Base):
    """
    Stores every user ↔ LLM interaction along with evaluation metrics.

    Each row represents one chat exchange: the user's query, the model's
    response, and the full evaluation result including hybrid scoring.
    """

    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    user_query = Column(Text, nullable=False)
    llm_response = Column(Text, nullable=False)
    similarity_score = Column(Float, nullable=False, default=0.0)
    evaluation_status = Column(
        String(20), nullable=False, default="Pending"
    )  # "Passed" | "Flagged" | "Pending"
    ground_truth_used = Column(Text, nullable=True)
    matched_query = Column(Text, nullable=True)

    # Hybrid evaluation fields
    hybrid_score = Column(Float, nullable=True, default=0.0)
    llm_judge_score = Column(Float, nullable=True, default=0.0)
    hallucination_detected = Column(Boolean, nullable=True, default=False)
    completeness = Column(Integer, nullable=True, default=3)
    teacher_correction = Column(Text, nullable=True)

    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return (
            f"<InteractionLog(id={self.id}, score={self.similarity_score}, "
            f"status='{self.evaluation_status}')>"
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for API responses."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_query": self.user_query,
            "llm_response": self.llm_response,
            "similarity_score": self.similarity_score,
            "evaluation_status": self.evaluation_status,
            "ground_truth_used": self.ground_truth_used,
            "matched_query": self.matched_query,
            "hybrid_score": self.hybrid_score,
            "llm_judge_score": self.llm_judge_score,
            "hallucination_detected": self.hallucination_detected,
            "completeness": self.completeness,
            "teacher_correction": self.teacher_correction,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
