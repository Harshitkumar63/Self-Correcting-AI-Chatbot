"""
Teacher-Correction Module — Generates clean, verified corrections.

When an LLM response is flagged as low-quality, the Teacher generates
a "Corrected Ground-Truth" answer using a specialized prompt template.
The bad response is NEVER stored for training — only the teacher's
verified correction paired with the original user prompt.
"""

import logging
from typing import Optional

from ml_pipeline.config import config

logger = logging.getLogger(__name__)


# ── Teacher Prompt Templates ───────────────────────────────

TEACHER_SYSTEM_PROMPT = """\
You are an expert teacher AI. Your job is to produce a clean, accurate, \
and comprehensive answer to a user's question.

Rules:
- Be factually accurate and concise.
- Use clear, educational language.
- Do NOT reference or critique the previous bad answer.
- Return ONLY the corrected answer — no meta-commentary, no preamble.
- If a reference answer is provided, use it as the authoritative source \
  but feel free to rephrase or enhance clarity."""

TEACHER_WITH_REFERENCE = """\
A student asked: {query}

An AI assistant gave a bad answer that was flagged for low quality.

Reference answer (authoritative source):
{ground_truth}

Please provide a clean, accurate, and complete answer to the student's question."""

TEACHER_WITHOUT_REFERENCE = """\
A student asked: {query}

An AI assistant gave a bad answer that was flagged for low quality: \
"{bad_response}"

There is no reference answer available. Using your knowledge, provide a \
clean, accurate, and complete answer to the student's question."""


class TeacherCorrector:
    """
    Generates teacher-verified corrections for flagged LLM responses.

    When ground truth is available and relevant (cosine sim above threshold),
    the teacher uses it as a reference to produce a clean correction.
    When no good ground truth match exists, the teacher generates a
    correction from model knowledge alone.
    """

    def __init__(self):
        self._inference_engine = None

    def set_inference_engine(self, engine) -> None:
        """Share the inference engine from MLService to avoid duplicate loading."""
        self._inference_engine = engine

    def _get_inference_engine(self):
        """Lazy-load the inference engine if not shared."""
        if self._inference_engine is None:
            from ml_pipeline.inference import InferenceEngine
            self._inference_engine = InferenceEngine()
        return self._inference_engine

    def generate_correction(
        self,
        query: str,
        bad_response: str,
        ground_truth_answer: str,
        gt_match_similarity: float = 0.0,
    ) -> str:
        """
        Generate a teacher-verified correction for a flagged response.

        If a relevant ground-truth answer exists (similarity above
        threshold), it is used as the reference. Otherwise, the
        teacher generates a correction from model knowledge.

        Args:
            query: The original user query.
            bad_response: The flagged LLM response (for context).
            ground_truth_answer: The closest ground-truth answer found.
            gt_match_similarity: How well the query matched the GT entry.

        Returns:
            A clean, teacher-verified correction string.
        """
        try:
            engine = self._get_inference_engine()

            # Decide whether ground truth is relevant enough to use
            has_good_reference = (
                ground_truth_answer
                and ground_truth_answer != "(no ground truth available)"
                and gt_match_similarity >= config.gt_match_threshold
            )

            if has_good_reference:
                user_prompt = TEACHER_WITH_REFERENCE.format(
                    query=query,
                    ground_truth=ground_truth_answer,
                )
            else:
                user_prompt = TEACHER_WITHOUT_REFERENCE.format(
                    query=query,
                    bad_response=bad_response[:300],  # Truncate for prompt space
                )

            correction = engine.generate(
                prompt=user_prompt,
                system_prompt=TEACHER_SYSTEM_PROMPT,
                max_new_tokens=300,
                temperature=0.3,  # Slightly creative but still factual
            )

            if not correction.strip():
                logger.warning("Teacher generated empty correction, using ground truth fallback.")
                return ground_truth_answer if has_good_reference else bad_response

            logger.info(
                "Teacher correction generated (%d chars) for query: %.50s...",
                len(correction),
                query,
            )
            return correction.strip()

        except Exception as e:
            logger.exception("Teacher correction failed: %s", str(e))
            # Fallback: use ground truth if available, otherwise return bad response
            if ground_truth_answer and ground_truth_answer != "(no ground truth available)":
                return ground_truth_answer
            return bad_response
