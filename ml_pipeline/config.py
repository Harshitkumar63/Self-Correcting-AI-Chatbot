"""
Central configuration for the ML pipeline.

All model IDs, paths, thresholds, and hyperparameters are defined here
so they can be tuned from a single location.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


# Base directory of the ml_pipeline package
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class PipelineConfig:
    """Master configuration for the self-improving pipeline."""

    # ── Inference ──────────────────────────────────────────────
    inference_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"  # "cpu", "cuda", or "auto"

    # ── Evaluation ─────────────────────────────────────────────
    eval_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.70  # below this → flagged

    # ── Feedback ───────────────────────────────────────────────
    fine_tune_dataset_path: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "fine_tune_dataset.jsonl"
    )
    fine_tune_trigger_count: int = 50  # samples before auto-retrain

    # ── Ground Truth ───────────────────────────────────────────
    ground_truth_path: Path = field(
        default_factory=lambda: BASE_DIR / "ground_truth.json"
    )

    # ── LoRA Training ──────────────────────────────────────────
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    training_epochs: int = 3
    training_batch_size: int = 4
    learning_rate: float = 2e-4
    checkpoint_dir: Path = field(
        default_factory=lambda: BASE_DIR / "checkpoints"
    )
    archive_dir: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "archive"
    )

    def __post_init__(self):
        """Ensure required directories exist."""
        self.fine_tune_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)


# Singleton config instance
config = PipelineConfig()
