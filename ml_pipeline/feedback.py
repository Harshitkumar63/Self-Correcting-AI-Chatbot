"""
Feedback Module — Collects flagged (low-quality) responses for retraining.

When the evaluation score falls below the threshold, this module appends
the prompt and correct answer to a JSONL file that serves as the
fine-tuning dataset.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from ml_pipeline.config import config

logger = logging.getLogger(__name__)


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
