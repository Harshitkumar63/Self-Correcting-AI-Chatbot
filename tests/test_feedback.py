"""
Tests for the FeedbackCollector and CurationQueue.

Tests dataset writing, sample counting, curation queue operations
(add, approve, reject), archive functionality, and stats.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from ml_pipeline.feedback import FeedbackCollector, CurationQueue


# ── FeedbackCollector Tests ─────────────────────────────────

class TestFeedbackCollector:
    """Tests for the FeedbackCollector class."""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create a FeedbackCollector with temp paths."""
        dataset_path = tmp_path / "data" / "fine_tune_dataset.jsonl"
        with patch("ml_pipeline.feedback.config") as mock_config:
            mock_config.fine_tune_dataset_path = dataset_path
            mock_config.fine_tune_trigger_count = 3
            mock_config.archive_dir = tmp_path / "archive"
            collector = FeedbackCollector()
            collector._dataset_path = dataset_path
            collector._threshold = 3
            yield collector

    def test_empty_dataset(self, collector):
        """Fresh collector should have 0 samples."""
        assert collector.get_sample_count() == 0

    def test_collect_single_sample(self, collector):
        """Collecting one sample should increment count to 1."""
        result = collector.collect(
            query="What is AI?",
            bad_response="It's magic.",
            correct_response="AI is a branch of computer science.",
            score=0.3,
        )
        assert result["sample_count"] == 1
        assert result["should_trigger"] is False

    def test_collect_multiple_samples(self, collector):
        """Collecting enough samples should trigger training."""
        for i in range(3):
            result = collector.collect(
                query=f"Question {i}",
                bad_response=f"Bad answer {i}",
                correct_response=f"Good answer {i}",
                score=0.3,
            )
        assert result["sample_count"] == 3
        assert result["should_trigger"] is True

    def test_should_trigger_training(self, collector):
        """should_trigger should be True when count >= threshold."""
        assert collector.should_trigger_training() is False

        for i in range(3):
            collector.collect(f"Q{i}", f"Bad{i}", f"Good{i}", 0.2)

        assert collector.should_trigger_training() is True

    def test_get_samples(self, collector):
        """get_samples should return all collected entries."""
        collector.collect("Q1", "Bad1", "Good1", 0.3)
        collector.collect("Q2", "Bad2", "Good2", 0.4)

        samples = collector.get_samples()
        assert len(samples) == 2
        assert samples[0]["instruction"] == "Q1"
        assert samples[0]["output"] == "Good1"
        assert samples[1]["instruction"] == "Q2"

    def test_archive_and_clear(self, collector, tmp_path):
        """Archive should copy dataset and clear the original."""
        with patch("ml_pipeline.feedback.config") as mock_config:
            mock_config.archive_dir = tmp_path / "archive"

            collector.collect("Q1", "Bad1", "Good1", 0.3)
            assert collector.get_sample_count() == 1

            archive_path = collector.archive_and_clear()
            assert archive_path is not None
            assert archive_path.exists()
            assert collector.get_sample_count() == 0

    def test_archive_empty_dataset(self, collector, tmp_path):
        """Archiving an empty dataset should return None."""
        with patch("ml_pipeline.feedback.config") as mock_config:
            mock_config.archive_dir = tmp_path / "archive"
            assert collector.archive_and_clear() is None

    def test_get_status(self, collector):
        """Status should reflect current state."""
        status = collector.get_status()
        assert status["sample_count"] == 0
        assert status["threshold"] == 3
        assert status["progress_pct"] == 0.0
        assert status["should_trigger"] is False


# ── CurationQueue Tests ─────────────────────────────────────

class TestCurationQueue:
    """Tests for the CurationQueue class."""

    @pytest.fixture
    def queue(self, tmp_path):
        """Create a CurationQueue with temp paths."""
        queue_path = tmp_path / "data" / "curation_queue.jsonl"
        dataset_path = tmp_path / "data" / "fine_tune_dataset.jsonl"
        with patch("ml_pipeline.feedback.config") as mock_config:
            mock_config.curation_queue_path = queue_path
            mock_config.fine_tune_dataset_path = dataset_path
            q = CurationQueue()
            q._queue_path = queue_path
            q._dataset_path = dataset_path
            yield q

    def test_add_to_queue(self, queue):
        """Adding an item should return a valid ID and queue size."""
        result = queue.add_to_queue(
            query="What is Python?",
            bad_response="Python is a snake.",
            teacher_correction="Python is a programming language.",
            eval_score=0.3,
            hybrid_score=0.35,
            hallucination_detected=True,
            completeness=2,
            llm_judge_score=0.4,
        )
        assert "item_id" in result
        assert result["queue_size"] == 1

    def test_get_pending_items(self, queue):
        """Should return only pending items."""
        queue.add_to_queue("Q1", "Bad1", "Good1", 0.3, 0.35, False, 3, 0.4)
        queue.add_to_queue("Q2", "Bad2", "Good2", 0.4, 0.45, True, 2, 0.5)

        pending = queue.get_pending_items()
        assert len(pending) == 2
        assert all(item["status"] == "pending" for item in pending)

    def test_approve_item(self, queue):
        """Approving should change status and add to dataset."""
        result = queue.add_to_queue("Q1", "Bad1", "Good1", 0.3, 0.35, False, 3, 0.4)
        item_id = result["item_id"]

        approved = queue.approve_item(item_id)
        assert approved is not None
        assert approved["status"] == "approved"
        assert approved["reviewed_at"] is not None

        # Check dataset was appended
        assert queue._dataset_path.exists()
        with open(queue._dataset_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["instruction"] == "Q1"
        assert entry["output"] == "Good1"

    def test_approve_with_edited_correction(self, queue):
        """Approving with an edit should use the edited text."""
        result = queue.add_to_queue("Q1", "Bad1", "Good1", 0.3, 0.35, False, 3, 0.4)
        item_id = result["item_id"]

        approved = queue.approve_item(item_id, edited_correction="Better answer")
        assert approved["teacher_correction"] == "Better answer"

        with open(queue._dataset_path) as f:
            entry = json.loads(f.readline())
        assert entry["output"] == "Better answer"

    def test_reject_item(self, queue):
        """Rejecting should change status and NOT add to dataset."""
        result = queue.add_to_queue("Q1", "Bad1", "Good1", 0.3, 0.35, False, 3, 0.4)
        item_id = result["item_id"]

        rejected = queue.reject_item(item_id)
        assert rejected is not None
        assert rejected["status"] == "rejected"

        # Dataset should not exist or be empty
        if queue._dataset_path.exists():
            assert queue._dataset_path.read_text().strip() == ""

    def test_approve_nonexistent_item(self, queue):
        """Approving a non-existent ID should return None."""
        assert queue.approve_item("nonexistent") is None

    def test_reject_nonexistent_item(self, queue):
        """Rejecting a non-existent ID should return None."""
        assert queue.reject_item("nonexistent") is None

    def test_get_stats(self, queue):
        """Stats should reflect queue state."""
        queue.add_to_queue("Q1", "Bad1", "Good1", 0.3, 0.35, False, 3, 0.4)
        r2 = queue.add_to_queue("Q2", "Bad2", "Good2", 0.4, 0.45, True, 2, 0.5)
        queue.approve_item(r2["item_id"])

        stats = queue.get_stats()
        assert stats["total"] == 2
        assert stats["pending"] == 1
        assert stats["approved"] == 1
        assert stats["rejected"] == 0

    def test_get_all_items(self, queue):
        """get_all_items should return items in all statuses."""
        r1 = queue.add_to_queue("Q1", "Bad1", "Good1", 0.3, 0.35, False, 3, 0.4)
        r2 = queue.add_to_queue("Q2", "Bad2", "Good2", 0.4, 0.45, True, 2, 0.5)

        queue.approve_item(r1["item_id"])
        queue.reject_item(r2["item_id"])

        all_items = queue.get_all_items()
        assert len(all_items) == 2
        statuses = {item["status"] for item in all_items}
        assert statuses == {"approved", "rejected"}
