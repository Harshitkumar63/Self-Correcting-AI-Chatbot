"""
SQLAlchemy ORM Models for the interaction log database.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from app.database import Base


class InteractionLog(Base):
    """
    Stores every user ↔ LLM interaction along with evaluation metrics.

    Each row represents one chat exchange: the user's query, the model's
    response, and the evaluation result (similarity score + pass/fail status).
    """

    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_query = Column(Text, nullable=False)
    llm_response = Column(Text, nullable=False)
    similarity_score = Column(Float, nullable=False, default=0.0)
    evaluation_status = Column(
        String(20), nullable=False, default="Pending"
    )  # "Passed" | "Flagged" | "Pending"
    ground_truth_used = Column(Text, nullable=True)
    matched_query = Column(Text, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=datetime.utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<InteractionLog(id={self.id}, score={self.similarity_score}, "
            f"status='{self.evaluation_status}')>"
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for API responses."""
        return {
            "id": self.id,
            "user_query": self.user_query,
            "llm_response": self.llm_response,
            "similarity_score": self.similarity_score,
            "evaluation_status": self.evaluation_status,
            "ground_truth_used": self.ground_truth_used,
            "matched_query": self.matched_query,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
