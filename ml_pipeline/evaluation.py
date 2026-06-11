"""
Evaluation Module — Scores LLM responses using semantic similarity.

Loads the sentence-transformers/all-MiniLM-L6-v2 model and compares
the LLM output against ground-truth answers using cosine similarity.
A score ≥ 0.70 is considered "Passed"; below is "Flagged".
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util

from ml_pipeline.config import config

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a single LLM response."""

    score: float               # Cosine similarity [0, 1]
    passed: bool               # True if score ≥ threshold
    status: str                # "Passed" or "Flagged"
    ground_truth_answer: str   # The reference answer used
    matched_query: str         # The ground-truth query matched


class EvaluationEngine:
    """
    Evaluates LLM responses by computing cosine similarity between
    the model output and the closest ground-truth reference answer.
    """

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._ground_truth: list[dict] = []
        self._gt_query_embeddings: Optional[np.ndarray] = None

    # ── Private helpers ─────────────────────────────────────────

    def _load(self) -> None:
        """Load the sentence-transformer model and ground truth data."""
        if self._model is not None:
            return

        logger.info("Loading evaluation model: %s", config.eval_model_id)
        self._model = SentenceTransformer(config.eval_model_id)

        self._load_ground_truth()
        logger.info(
            "Evaluation engine ready with %d ground truth pairs.",
            len(self._ground_truth),
        )

    def _load_ground_truth(self) -> None:
        """Load ground truth Q&A pairs and pre-compute query embeddings."""
        gt_path = config.ground_truth_path
        if not gt_path.exists():
            logger.warning(
                "Ground truth file not found at %s — evaluation will "
                "return default scores.",
                gt_path,
            )
            self._ground_truth = []
            self._gt_query_embeddings = None
            return

        with open(gt_path, "r", encoding="utf-8") as f:
            self._ground_truth = json.load(f)

        # Pre-compute embeddings for all ground-truth queries
        queries = [item["query"] for item in self._ground_truth]
        self._gt_query_embeddings = self._model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        )

    def _find_closest_ground_truth(self, query: str) -> dict:
        """
        Find the ground-truth entry whose query is most similar
        to the incoming user query.
        """
        if not self._ground_truth or self._gt_query_embeddings is None:
            return {"query": query, "answer": ""}

        query_embedding = self._model.encode(
            query, convert_to_tensor=True, show_progress_bar=False
        )

        # Cosine similarity between user query and all GT queries
        similarities = util.cos_sim(query_embedding, self._gt_query_embeddings)
        best_idx = int(similarities.argmax())

        return self._ground_truth[best_idx]

    # ── Public API ──────────────────────────────────────────────

    def evaluate(self, query: str, response: str) -> EvalResult:
        """
        Evaluate an LLM response against the ground truth.

        1. Find the closest matching ground-truth query.
        2. Compute cosine similarity between the LLM response
           and the ground-truth answer.
        3. Return the score and pass/fail status.

        Args:
            query: The original user query.
            response: The LLM-generated answer.

        Returns:
            EvalResult with score, status, and matched ground truth.
        """
        self._load()

        gt_entry = self._find_closest_ground_truth(query)
        gt_answer = gt_entry.get("answer", "")

        if not gt_answer:
            # No ground truth available — return neutral score
            return EvalResult(
                score=0.5,
                passed=False,
                status="Flagged",
                ground_truth_answer="(no ground truth available)",
                matched_query=query,
            )

        # Encode both the LLM response and the ground truth answer
        embeddings = self._model.encode(
            [response, gt_answer],
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        score = max(0.0, min(1.0, score))  # clamp to [0, 1]

        passed = score >= config.similarity_threshold
        status = "Passed" if passed else "Flagged"

        logger.info(
            "Evaluation: score=%.4f status=%s query=%.50s...",
            score,
            status,
            query,
        )

        return EvalResult(
            score=round(score, 4),
            passed=passed,
            status=status,
            ground_truth_answer=gt_answer,
            matched_query=gt_entry.get("query", query),
        )

    def reload_ground_truth(self) -> None:
        """Re-load ground truth from disk (e.g. after updates)."""
        if self._model is not None:
            self._load_ground_truth()
            logger.info("Ground truth reloaded.")

    def is_loaded(self) -> bool:
        """Check if the evaluation model is loaded."""
        return self._model is not None
