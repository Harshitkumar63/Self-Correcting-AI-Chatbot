"""
ML Service — Bridge between FastAPI backend and the ML pipeline.

Provides a singleton interface that lazily initializes the ML modules
and orchestrates the full self-improving pipeline:
  query → inference → hybrid evaluation → teacher correction → curation queue
"""

import logging
from typing import Optional

from ml_pipeline.inference import InferenceEngine
from ml_pipeline.evaluation import EvaluationEngine, EvalResult
from ml_pipeline.feedback import FeedbackCollector, CurationQueue
from ml_pipeline.teacher import TeacherCorrector
from ml_pipeline.trainer import run_training, get_training_status
from ml_pipeline.config import config

logger = logging.getLogger(__name__)


class MLService:
    """
    Orchestrates the full self-improving ML pipeline for each user query:
    1. Generate response via InferenceEngine
    2. Evaluate response via Hybrid Evaluator (cosine + LLM judge)
    3. If flagged: generate teacher correction
    4. Route to curation queue for admin review

    Uses lazy initialization — ML models are loaded on first use.
    """

    _instance: Optional["MLService"] = None

    def __init__(self):
        self._inference = InferenceEngine()
        self._evaluation = EvaluationEngine()
        self._feedback = FeedbackCollector()
        self._curation = CurationQueue()
        self._teacher = TeacherCorrector()

        # Share the inference engine to avoid loading the model multiple times
        self._teacher.set_inference_engine(self._inference)
        self._evaluation.get_llm_judge().set_inference_engine(self._inference)

    @classmethod
    def get_instance(cls) -> "MLService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def process_query(self, query: str) -> dict:
        """
        Full pipeline: generate → hybrid evaluate → teacher correct → queue.

        Args:
            query: The user's input query.

        Returns:
            Dict with response, all evaluation metrics, and teacher correction.
        """
        # Step 1: Generate LLM response
        logger.info("Processing query: %.60s...", query)
        response = self._inference.generate(query)

        # Step 2: Hybrid evaluation (cosine + LLM judge)
        eval_result: EvalResult = self._evaluation.evaluate(query, response)

        # Step 3: Teacher correction + curation queue (if flagged)
        feedback_status = None
        teacher_correction = None

        if not eval_result.passed:
            # Generate teacher-verified correction
            teacher_correction = self._teacher.generate_correction(
                query=query,
                bad_response=response,
                ground_truth_answer=eval_result.ground_truth_answer,
                gt_match_similarity=eval_result.gt_match_similarity,
            )

            # Route to curation queue (NOT directly to training dataset)
            queue_result = self._curation.add_to_queue(
                query=query,
                bad_response=response,
                teacher_correction=teacher_correction,
                eval_score=eval_result.score,
                hybrid_score=eval_result.hybrid_score,
                hallucination_detected=eval_result.hallucination_detected,
                completeness=eval_result.completeness,
                llm_judge_score=eval_result.llm_judge_score,
            )

            feedback_status = {
                "curation_item_id": queue_result["item_id"],
                "queue_size": queue_result["queue_size"],
                "sample_count": self._feedback.get_sample_count(),
                "threshold": config.fine_tune_trigger_count,
                "should_trigger": self._feedback.should_trigger_training(),
            }

        return {
            "response": response,
            "similarity_score": eval_result.score,
            "evaluation_status": eval_result.status,
            "ground_truth": eval_result.ground_truth_answer,
            "matched_query": eval_result.matched_query,
            "feedback_status": feedback_status,
            # New hybrid evaluator fields
            "hybrid_score": eval_result.hybrid_score,
            "llm_judge_score": eval_result.llm_judge_score,
            "hallucination_detected": eval_result.hallucination_detected,
            "completeness": eval_result.completeness,
            "teacher_correction": teacher_correction,
        }

    # ── Feedback / Training ─────────────────────────────────

    def get_feedback_status(self) -> dict:
        """Get current feedback collection status."""
        return self._feedback.get_status()

    def trigger_training(self) -> dict:
        """Trigger a LoRA fine-tuning run."""
        return run_training(self._feedback)

    def get_training_status(self) -> dict:
        """Get current training status."""
        return get_training_status()

    # ── Curation Queue ──────────────────────────────────────

    def get_curation_queue(self) -> dict:
        """Get all pending curation items + stats."""
        items = self._curation.get_pending_items()
        stats = self._curation.get_stats()
        return {"items": items, "stats": stats}

    def get_curation_all(self) -> dict:
        """Get all curation items (all statuses) + stats."""
        items = self._curation.get_all_items()
        stats = self._curation.get_stats()
        return {"items": items, "stats": stats}

    def approve_curation_item(self, item_id: str, edited_correction: Optional[str] = None) -> Optional[dict]:
        """Approve a curation item and add to training dataset."""
        return self._curation.approve_item(item_id, edited_correction)

    def reject_curation_item(self, item_id: str) -> Optional[dict]:
        """Reject a curation item."""
        return self._curation.reject_item(item_id)

    def get_curation_stats(self) -> dict:
        """Get curation queue statistics."""
        return self._curation.get_stats()


def get_ml_service() -> MLService:
    """FastAPI dependency for the ML service."""
    return MLService.get_instance()
