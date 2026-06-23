"""
Feedback Module — Curation Queue + Training Dataset Management.

When an LLM output is flagged, the teacher-corrected sample is routed
to a curation queue for admin review. Only approved items enter the
fine-tuning dataset. Bad responses are NEVER stored for training.
"""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ml_pipeline.config import config

logger = logging.getLogger(__name__)


class CurationQueue:
    """
    Manages the human-in-the-loop curation queue.

    Flagged responses + teacher corrections are stored in a JSONL queue
    file. Admins can approve (with optional edits), or reject items.
    Only approved items are appended to the fine-tuning dataset.
    """

    def __init__(self):
        self._queue_path: Path = config.curation_queue_path
        self._dataset_path: Path = config.fine_tune_dataset_path

    def _ensure_file(self) -> None:
        """Create the queue file if it doesn't exist."""
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._queue_path.exists():
            self._queue_path.touch()

    def add_to_queue(
        self,
        query: str,
        bad_response: str,
        teacher_correction: str,
        eval_score: float,
        hybrid_score: float,
        hallucination_detected: bool,
        completeness: int,
        llm_judge_score: float,
    ) -> dict:
        """
        Add a flagged item to the curation queue.

        Args:
            query: Original user query.
            bad_response: The flagged LLM response.
            teacher_correction: The teacher-generated correction.
            eval_score: Cosine similarity score.
            hybrid_score: Weighted hybrid score.
            hallucination_detected: Whether hallucination was detected.
            completeness: Completeness rating (1-5).
            llm_judge_score: LLM judge score.

        Returns:
            Dict with item ID and queue stats.
        """
        self._ensure_file()

        item_id = str(uuid.uuid4())[:8]

        entry = {
            "id": item_id,
            "query": query,
            "bad_response": bad_response,
            "teacher_correction": teacher_correction,
            "eval_score": eval_score,
            "hybrid_score": hybrid_score,
            "hallucination_detected": hallucination_detected,
            "completeness": completeness,
            "llm_judge_score": llm_judge_score,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
        }

        with open(self._queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            "Added to curation queue: id=%s query=%.50s...",
            item_id,
            query,
        )

        return {
            "item_id": item_id,
            "queue_size": self.get_pending_count(),
        }

    def get_all_items(self) -> list[dict]:
        """Read all items from the curation queue."""
        self._ensure_file()
        items = []
        with open(self._queue_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def get_pending_items(self) -> list[dict]:
        """Get all items with status 'pending'."""
        return [item for item in self.get_all_items() if item.get("status") == "pending"]

    def get_pending_count(self) -> int:
        """Count pending items in the queue."""
        return len(self.get_pending_items())

    def _update_item_status(self, item_id: str, new_status: str, edited_correction: Optional[str] = None) -> Optional[dict]:
        """
        Update an item's status in the queue file.

        Returns the updated item, or None if not found.
        """
        self._ensure_file()
        items = self.get_all_items()
        updated_item = None

        for item in items:
            if item.get("id") == item_id:
                item["status"] = new_status
                item["reviewed_at"] = datetime.utcnow().isoformat()
                if edited_correction is not None:
                    item["teacher_correction"] = edited_correction
                updated_item = item
                break

        if updated_item is None:
            return None

        # Rewrite the entire file with updated items
        with open(self._queue_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return updated_item

    def approve_item(self, item_id: str, edited_correction: Optional[str] = None) -> Optional[dict]:
        """
        Approve a curation item and append it to the training dataset.

        Args:
            item_id: The item's unique ID.
            edited_correction: Optionally override the teacher correction.

        Returns:
            The approved item dict, or None if not found.
        """
        updated = self._update_item_status(item_id, "approved", edited_correction)
        if updated is None:
            return None

        # Append to fine-tuning dataset
        correction = edited_correction or updated["teacher_correction"]
        self._append_to_dataset(
            query=updated["query"],
            bad_response=updated["bad_response"],
            correct_response=correction,
            score=updated.get("hybrid_score", updated.get("eval_score", 0.0)),
        )

        logger.info("Curation item approved: id=%s", item_id)
        return updated

    def reject_item(self, item_id: str) -> Optional[dict]:
        """
        Reject a curation item.

        Returns the rejected item dict, or None if not found.
        """
        updated = self._update_item_status(item_id, "rejected")
        if updated:
            logger.info("Curation item rejected: id=%s", item_id)
        return updated

    def _append_to_dataset(
        self,
        query: str,
        bad_response: str,
        correct_response: str,
        score: float,
    ) -> None:
        """Append an approved item to the fine-tuning dataset."""
        self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._dataset_path.exists():
            self._dataset_path.touch()

        entry = {
            "instruction": query,
            "input": "",
            "output": correct_response,
            "metadata": {
                "bad_response": bad_response,
                "score": score,
                "collected_at": datetime.utcnow().isoformat(),
                "source": "teacher_corrected",
            },
        }

        with open(self._dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_stats(self) -> dict:
        """Return curation queue statistics."""
        items = self.get_all_items()
        pending = sum(1 for i in items if i.get("status") == "pending")
        approved = sum(1 for i in items if i.get("status") == "approved")
        rejected = sum(1 for i in items if i.get("status") == "rejected")
        return {
            "total": len(items),
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
        }


class FeedbackCollector:
    """
    Manages the feedback loop by:
    - Collecting bad responses and their corrections
    - Writing them to a JSONL training file
    - Tracking whether the auto-training threshold has been reached
    """

    def __init__(self):
        self._dataset_path: Path = config.fine_tune_dataset_path
        self._threshold: int = config.fine_tune_trigger_count

    # ── Private helpers ─────────────────────────────────────────

    def _ensure_file(self) -> None:
        """Create the dataset file if it doesn't exist."""
        self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._dataset_path.exists():
            self._dataset_path.touch()

    # ── Public API ──────────────────────────────────────────────

    def collect(
        self,
        query: str,
        bad_response: str,
        correct_response: str,
        score: float,
    ) -> dict:
        """
        Append a flagged interaction to the fine-tuning dataset.

        Args:
            query: Original user query.
            bad_response: The LLM response that was flagged.
            correct_response: The ground-truth or teacher-corrected answer.
            score: The similarity score that triggered flagging.

        Returns:
            Dict with sample count and whether training should trigger.
        """
        self._ensure_file()

        entry = {
            "instruction": query,
            "input": "",
            "output": correct_response,
            "metadata": {
                "bad_response": bad_response,
                "score": score,
                "collected_at": datetime.utcnow().isoformat(),
            },
        }

        with open(self._dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        count = self.get_sample_count()
        logger.info(
            "Feedback collected: %d/%d samples (score=%.4f, query=%.50s...)",
            count,
            self._threshold,
            score,
            query,
        )

        return {
            "sample_count": count,
            "threshold": self._threshold,
            "should_trigger": self.should_trigger_training(),
        }

    def get_sample_count(self) -> int:
        """Count the number of samples in the fine-tuning dataset."""
        self._ensure_file()
        with open(self._dataset_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def should_trigger_training(self) -> bool:
        """Check if the dataset has enough samples to trigger training."""
        return self.get_sample_count() >= self._threshold

    def get_samples(self) -> list[dict]:
        """Read all samples from the dataset file."""
        self._ensure_file()
        samples = []
        with open(self._dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def archive_and_clear(self) -> Optional[Path]:
        """
        Archive the current dataset (for record-keeping) and clear it
        so the next training cycle starts fresh.

        Returns:
            Path to the archived file, or None if nothing to archive.
        """
        if self.get_sample_count() == 0:
            return None

        archive_dir = config.archive_dir
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"dataset_{timestamp}.jsonl"

        shutil.copy2(self._dataset_path, archive_path)

        # Clear the active dataset
        with open(self._dataset_path, "w", encoding="utf-8") as f:
            f.write("")

        logger.info(
            "Dataset archived to %s and cleared.", archive_path
        )
        return archive_path

    def get_status(self) -> dict:
        """Return current feedback collection status."""
        count = self.get_sample_count()
        return {
            "sample_count": count,
            "threshold": self._threshold,
            "progress_pct": round((count / self._threshold) * 100, 1),
            "should_trigger": count >= self._threshold,
        }
