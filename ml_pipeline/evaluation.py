"""
Evaluation Module — Hybrid Evaluator System.

Combines two evaluation signals:
  1. Semantic similarity (sentence-transformers cosine similarity)
  2. LLM-as-a-Judge (prompted Qwen model returning structured JSON)

The hybrid score is a weighted blend of both signals. A score below
the threshold is flagged for teacher correction and curation.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util

from ml_pipeline.config import config

logger = logging.getLogger(__name__)


# ── LLM Judge Prompt Template ──────────────────────────────

LLM_JUDGE_SYSTEM_PROMPT = """\
You are an expert AI evaluator. Your task is to judge the quality of an AI assistant's response.

Evaluate the response against the reference answer and return ONLY a JSON object with these exact fields:
- "score": float between 0.0 and 1.0 (how correct and relevant the response is)
- "hallucination_detected": boolean (true if the response contains fabricated or incorrect facts)
- "completeness": integer from 1 to 5 (how complete the response is: 1=empty/irrelevant, 5=comprehensive)

Return ONLY the JSON object, no other text."""

LLM_JUDGE_USER_TEMPLATE = """\
Question: {query}

AI Response: {response}

Reference Answer: {ground_truth}

Evaluate the AI Response and return the JSON verdict."""


@dataclass
class EvalResult:
    """Result of evaluating a single LLM response."""

    # ── Original fields (preserved) ────────────────────────
    score: float               # Cosine similarity [0, 1]
    passed: bool               # True if hybrid_score ≥ threshold
    status: str                # "Passed" or "Flagged"
    ground_truth_answer: str   # The reference answer used
    matched_query: str         # The ground-truth query matched

    # ── New hybrid evaluator fields ────────────────────────
    hybrid_score: float = 0.0          # Weighted blend of cosine + LLM judge
    llm_judge_score: float = 0.0       # LLM-as-a-Judge score [0, 1]
    hallucination_detected: bool = False
    completeness: int = 3              # 1–5 rating
    gt_match_similarity: float = 0.0   # How well the query matched ground truth


class LLMJudge:
    """
    Uses a prompted LLM to evaluate response quality.

    Returns a structured JSON verdict with score, hallucination detection,
    and completeness rating. Reuses the inference engine to avoid
    loading a separate model.
    """

    def __init__(self):
        self._inference_engine = None

    def _get_inference_engine(self):
        """Lazy-load the inference engine."""
        if self._inference_engine is None:
            from ml_pipeline.inference import InferenceEngine
            self._inference_engine = InferenceEngine()
        return self._inference_engine

    def set_inference_engine(self, engine):
        """Allow sharing the inference engine from MLService."""
        self._inference_engine = engine

    def judge(
        self,
        query: str,
        response: str,
        ground_truth: str,
    ) -> dict:
        """
        Run LLM-as-a-Judge evaluation.

        Args:
            query: The original user query.
            response: The LLM-generated answer to evaluate.
            ground_truth: The reference answer.

        Returns:
            Dict with 'score', 'hallucination_detected', 'completeness'.
        """
        defaults = {
            "score": 0.5,
            "hallucination_detected": False,
            "completeness": 3,
        }

        if not ground_truth or ground_truth == "(no ground truth available)":
            return defaults

        try:
            engine = self._get_inference_engine()

            user_prompt = LLM_JUDGE_USER_TEMPLATE.format(
                query=query,
                response=response,
                ground_truth=ground_truth,
            )

            # Use low temperature for deterministic judging
            raw_output = engine.generate(
                prompt=user_prompt,
                system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
                max_new_tokens=150,
                temperature=config.llm_judge_temperature,
            )

            return self._parse_judge_output(raw_output, defaults)

        except Exception as e:
            logger.warning("LLM Judge failed: %s — using defaults", str(e))
            return defaults

    def _parse_judge_output(self, raw: str, defaults: dict) -> dict:
        """
        Parse the JSON output from the LLM judge, with robust fallbacks.
        """
        try:
            # Try to extract JSON from the raw output
            json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(raw.strip())

            # Validate and clamp values
            score = float(parsed.get("score", defaults["score"]))
            score = max(0.0, min(1.0, score))

            hallucination = bool(parsed.get("hallucination_detected", defaults["hallucination_detected"]))

            completeness = int(parsed.get("completeness", defaults["completeness"]))
            completeness = max(1, min(5, completeness))

            return {
                "score": score,
                "hallucination_detected": hallucination,
                "completeness": completeness,
            }

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Failed to parse LLM judge output: %s — raw: %.100s", str(e), raw)
            return defaults


class EvaluationEngine:
    """
    Hybrid evaluator that combines cosine similarity with LLM-as-a-Judge.

    The hybrid score is computed as:
        hybrid = cosine_weight * cosine_score + llm_judge_weight * judge_score

    Pass/fail is determined using the hybrid score against the threshold.
    """

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._ground_truth: list[dict] = []
        self._gt_query_embeddings: Optional[np.ndarray] = None
        self._llm_judge = LLMJudge()

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

    def _find_closest_ground_truth(self, query: str) -> tuple[dict, float]:
        """
        Find the ground-truth entry whose query is most similar
        to the incoming user query.

        Returns:
            Tuple of (gt_entry_dict, similarity_score).
        """
        if not self._ground_truth or self._gt_query_embeddings is None:
            return {"query": query, "answer": ""}, 0.0

        query_embedding = self._model.encode(
            query, convert_to_tensor=True, show_progress_bar=False
        )

        # Cosine similarity between user query and all GT queries
        similarities = util.cos_sim(query_embedding, self._gt_query_embeddings)
        best_idx = int(similarities.argmax())
        best_sim = float(similarities[0][best_idx].item())

        return self._ground_truth[best_idx], best_sim

    def get_llm_judge(self) -> LLMJudge:
        """Expose the LLM judge for external engine sharing."""
        return self._llm_judge

    # ── Public API ──────────────────────────────────────────────

    def evaluate(self, query: str, response: str) -> EvalResult:
        """
        Evaluate an LLM response using the hybrid evaluator.

        1. Find the closest matching ground-truth query.
        2. Compute cosine similarity (response vs. ground-truth answer).
        3. Run LLM-as-a-Judge for structured quality assessment.
        4. Compute weighted hybrid score.
        5. Return enriched EvalResult.

        Args:
            query: The original user query.
            response: The LLM-generated answer.

        Returns:
            EvalResult with all original and new evaluation fields.
        """
        self._load()

        gt_entry, gt_match_sim = self._find_closest_ground_truth(query)
        gt_answer = gt_entry.get("answer", "")

        if not gt_answer:
            # No ground truth available — return neutral score
            return EvalResult(
                score=0.5,
                passed=False,
                status="Flagged",
                ground_truth_answer="(no ground truth available)",
                matched_query=query,
                hybrid_score=0.5,
                llm_judge_score=0.5,
                hallucination_detected=False,
                completeness=3,
                gt_match_similarity=0.0,
            )

        # ── Step 1: Cosine similarity (original logic) ──────────
        embeddings = self._model.encode(
            [response, gt_answer],
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        cosine_score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        cosine_score = max(0.0, min(1.0, cosine_score))  # clamp to [0, 1]

        # ── Step 2: LLM-as-a-Judge ─────────────────────────────
        judge_result = self._llm_judge.judge(
            query=query,
            response=response,
            ground_truth=gt_answer,
        )

        llm_judge_score = judge_result["score"]
        hallucination = judge_result["hallucination_detected"]
        completeness = judge_result["completeness"]

        # ── Step 3: Compute hybrid score ────────────────────────
        hybrid_score = (
            config.cosine_weight * cosine_score
            + config.llm_judge_weight * llm_judge_score
        )
        hybrid_score = max(0.0, min(1.0, hybrid_score))

        # If hallucination detected, penalize the hybrid score
        if hallucination:
            hybrid_score = hybrid_score * 0.7

        passed = hybrid_score >= config.similarity_threshold
        status = "Passed" if passed else "Flagged"

        logger.info(
            "Hybrid Evaluation: cosine=%.4f judge=%.4f hybrid=%.4f "
            "hallucination=%s completeness=%d status=%s query=%.50s...",
            cosine_score,
            llm_judge_score,
            hybrid_score,
            hallucination,
            completeness,
            status,
            query,
        )

        return EvalResult(
            score=round(cosine_score, 4),
            passed=passed,
            status=status,
            ground_truth_answer=gt_answer,
            matched_query=gt_entry.get("query", query),
            hybrid_score=round(hybrid_score, 4),
            llm_judge_score=round(llm_judge_score, 4),
            hallucination_detected=hallucination,
            completeness=completeness,
            gt_match_similarity=round(gt_match_sim, 4),
        )

    def reload_ground_truth(self) -> None:
        """Re-load ground truth from disk (e.g. after updates)."""
        if self._model is not None:
            self._load_ground_truth()
            logger.info("Ground truth reloaded.")

    def is_loaded(self) -> bool:
        """Check if the evaluation model is loaded."""
        return self._model is not None
