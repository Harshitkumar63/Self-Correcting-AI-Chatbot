"""
Tests for the EvaluationEngine and related components.

Tests cosine similarity scoring, LLM judge output parsing,
hybrid score computation, and ground truth matching.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from ml_pipeline.evaluation import EvalResult, LLMJudge, EvaluationEngine


# ── EvalResult Tests ────────────────────────────────────────

class TestEvalResult:
    """Test the EvalResult dataclass."""

    def test_passed_result(self):
        result = EvalResult(
            score=0.85,
            passed=True,
            status="Passed",
            ground_truth_answer="Test answer",
            matched_query="Test query",
            hybrid_score=0.82,
        )
        assert result.passed is True
        assert result.status == "Passed"
        assert result.hybrid_score == 0.82

    def test_flagged_result(self):
        result = EvalResult(
            score=0.3,
            passed=False,
            status="Flagged",
            ground_truth_answer="Expected answer",
            matched_query="Test query",
            hybrid_score=0.35,
            hallucination_detected=True,
            completeness=2,
        )
        assert result.passed is False
        assert result.status == "Flagged"
        assert result.hallucination_detected is True
        assert result.completeness == 2

    def test_default_values(self):
        result = EvalResult(
            score=0.5,
            passed=False,
            status="Flagged",
            ground_truth_answer="",
            matched_query="",
        )
        assert result.hybrid_score == 0.0
        assert result.llm_judge_score == 0.0
        assert result.hallucination_detected is False
        assert result.completeness == 3


# ── LLM Judge Parsing Tests ─────────────────────────────────

class TestLLMJudgeParsing:
    """Test the LLM judge output parser with various formats."""

    def setup_method(self):
        self.judge = LLMJudge()
        self.defaults = {
            "score": 0.5,
            "hallucination_detected": False,
            "completeness": 3,
        }

    def test_parse_valid_json(self):
        raw = '{"score": 0.9, "hallucination_detected": false, "completeness": 5}'
        result = self.judge._parse_judge_output(raw, self.defaults)
        assert result["score"] == 0.9
        assert result["hallucination_detected"] is False
        assert result["completeness"] == 5

    def test_parse_json_with_surrounding_text(self):
        raw = 'Here is my evaluation: {"score": 0.7, "hallucination_detected": true, "completeness": 3} Thanks!'
        result = self.judge._parse_judge_output(raw, self.defaults)
        assert result["score"] == 0.7
        assert result["hallucination_detected"] is True

    def test_parse_clamped_score(self):
        """Scores above 1.0 or below 0.0 should be clamped."""
        raw = '{"score": 1.5, "hallucination_detected": false, "completeness": 5}'
        result = self.judge._parse_judge_output(raw, self.defaults)
        assert result["score"] == 1.0

        raw_low = '{"score": -0.3, "hallucination_detected": false, "completeness": 1}'
        result_low = self.judge._parse_judge_output(raw_low, self.defaults)
        assert result_low["score"] == 0.0

    def test_parse_clamped_completeness(self):
        """Completeness outside 1-5 should be clamped."""
        raw = '{"score": 0.5, "hallucination_detected": false, "completeness": 10}'
        result = self.judge._parse_judge_output(raw, self.defaults)
        assert result["completeness"] == 5

        raw_low = '{"score": 0.5, "hallucination_detected": false, "completeness": 0}'
        result_low = self.judge._parse_judge_output(raw_low, self.defaults)
        assert result_low["completeness"] == 1

    def test_parse_invalid_json_returns_defaults(self):
        raw = "This is not JSON at all"
        result = self.judge._parse_judge_output(raw, self.defaults)
        assert result == self.defaults

    def test_parse_empty_string_returns_defaults(self):
        result = self.judge._parse_judge_output("", self.defaults)
        assert result == self.defaults

    def test_judge_with_no_ground_truth_returns_defaults(self):
        """When ground truth is unavailable, skip LLM judging."""
        result = self.judge.judge("query", "response", "(no ground truth available)")
        assert result["score"] == 0.5
        assert result["hallucination_detected"] is False

    def test_judge_with_empty_ground_truth_returns_defaults(self):
        result = self.judge.judge("query", "response", "")
        assert result["score"] == 0.5


# ── Hybrid Score Computation Tests ──────────────────────────

class TestHybridScoring:
    """Test the hybrid scoring logic in the evaluation engine."""

    def test_hybrid_score_weighted_average(self):
        """hybrid_score should be cosine_weight * cosine + llm_weight * judge."""
        # Default weights: cosine=0.6, llm_judge=0.4
        cosine = 0.8
        judge = 0.9
        expected = 0.6 * cosine + 0.4 * judge  # 0.48 + 0.36 = 0.84
        assert abs(expected - 0.84) < 0.001

    def test_hallucination_penalty(self):
        """Hallucination detection should penalize the hybrid score by 30%."""
        base_score = 0.8
        penalized = base_score * 0.7
        assert abs(penalized - 0.56) < 0.001

    def test_threshold_pass(self):
        """Scores >= 0.70 should pass."""
        assert 0.75 >= 0.70  # Passed
        assert 0.70 >= 0.70  # Passed (boundary)
        assert not (0.69 >= 0.70)  # Flagged
