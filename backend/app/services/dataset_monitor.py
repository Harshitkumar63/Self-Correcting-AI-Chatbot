"""
Dataset Monitor — Background task that watches the fine-tuning dataset.

Periodically checks the size of the approved training dataset. When
the sample count reaches the configured threshold, auto-triggers
a LoRA fine-tuning run.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from ml_pipeline.config import config
from ml_pipeline.feedback import FeedbackCollector
from ml_pipeline.trainer import run_training, get_training_status, TrainingStatus

logger = logging.getLogger(__name__)


class DatasetMonitor:
    """
    Async background monitor that checks the fine-tune dataset size
    at regular intervals and auto-triggers training when the
    threshold is reached.
    """

    def __init__(self):
        self._feedback = FeedbackCollector()
        self._interval: int = config.dataset_monitor_interval
        self._threshold: int = config.fine_tune_trigger_count
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # Status tracking
        self._last_check: Optional[str] = None
        self._last_count: int = 0
        self._auto_trains_triggered: int = 0

    async def start(self) -> None:
        """Start the background monitoring loop."""
        if self._running:
            logger.warning("Dataset monitor is already running.")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "🔄 Dataset monitor started (interval=%ds, threshold=%d samples).",
            self._interval,
            self._threshold,
        )

    async def stop(self) -> None:
        """Stop the background monitoring loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Dataset monitor stopped.")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop — runs until stopped."""
        while self._running:
            try:
                await self._check_dataset()
            except Exception as e:
                logger.error("Dataset monitor check failed: %s", str(e))

            await asyncio.sleep(self._interval)

    async def _check_dataset(self) -> None:
        """
        Check the dataset size and trigger training if threshold is met.
        """
        count = self._feedback.get_sample_count()
        self._last_check = datetime.utcnow().isoformat()
        self._last_count = count

        # Check if training is already active
        training_state = get_training_status().get("state", "idle")
        if training_state in ("preparing", "training"):
            logger.debug(
                "Dataset monitor: training already active (%s), skipping.",
                training_state,
            )
            return

        if count >= self._threshold:
            logger.info(
                "🔔 Dataset monitor: threshold reached! (%d/%d samples) "
                "— auto-triggering LoRA training.",
                count,
                self._threshold,
            )
            self._auto_trains_triggered += 1

            # Run training in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, run_training, self._feedback)

            logger.info("✅ Auto-triggered training complete.")
        else:
            logger.debug(
                "Dataset monitor: %d/%d samples (need %d more).",
                count,
                self._threshold,
                self._threshold - count,
            )

    def get_status(self) -> dict:
        """Return the current monitor status."""
        return {
            "running": self._running,
            "interval_seconds": self._interval,
            "threshold": self._threshold,
            "last_check": self._last_check,
            "last_count": self._last_count,
            "auto_trains_triggered": self._auto_trains_triggered,
            "samples_until_trigger": max(0, self._threshold - self._last_count),
        }


# Singleton instance
_monitor: Optional[DatasetMonitor] = None


def get_dataset_monitor() -> DatasetMonitor:
    """Get or create the singleton DatasetMonitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = DatasetMonitor()
    return _monitor
