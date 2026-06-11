"""
ML Service — Bridge between FastAPI backend and the ML pipeline.

Provides a singleton interface that lazily initializes the ML modules
and orchestrates the query → inference → evaluation → feedback flow.
"""

import logging
from typing import Optional

from ml_pipeline.inference import InferenceEngine
from ml_pipeline.evaluation import EvaluationEngine, EvalResult
from ml_pipeline.feedback import FeedbackCollector
from ml_pipeline.trainer import run_training, get_training_status
from ml_pipeline.config import config

logger = logging.getLogger(__name__)


class MLService:
    """
    Orchestrates the full ML pipeline for each user query:
    1. Generate response via InferenceEngine
    2. Evaluate response via EvaluationEngine
    3. Collect feedback if evaluation fails

    Uses lazy initialization — ML models are loaded on first use.
    """

    _instance: Optional["MLService"] = None

    def __init__(self):
        self._inference = InferenceEngine()
        self._evaluation = EvaluationEngine()
        self._feedback = FeedbackCollector()

    @classmethod
    def get_instance(cls) -> "MLService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def process_query(self, query: str) -> dict:
        """
        Full pipeline: generate → evaluate → feedback.

        Args:
            query: The user's input query.

        Returns:
            Dict with response, score, status, ground truth, and feedback info.
        """
        # Step 1: Generate LLM response
        logger.info("Processing query: %.60s...", query)
        response = self._inference.generate(query)

        # Step 2: Evaluate response
        eval_result: EvalResult = self._evaluation.evaluate(query, response)

        # Step 3: Collect feedback if flagged
        feedback_status = None
        if not eval_result.passed:
            feedback_status = self._feedback.collect(
                query=query,
                bad_response=response,
                correct_response=eval_result.ground_truth_answer,
                score=eval_result.score,
            )

        return {
            "response": response,
            "similarity_score": eval_result.score,
            "evaluation_status": eval_result.status,
            "ground_truth": eval_result.ground_truth_answer,
            "matched_query": eval_result.matched_query,
            "feedback_status": feedback_status,
        }

    def get_feedback_status(self) -> dict:
        """Get current feedback collection status."""
        return self._feedback.get_status()

    def trigger_training(self) -> dict:
        """Trigger a LoRA fine-tuning run."""
        return run_training(self._feedback)

    def get_training_status(self) -> dict:
        """Get current training status."""
        return get_training_status()


def get_ml_service() -> MLService:
    """FastAPI dependency for the ML service."""
    return MLService.get_instance()
