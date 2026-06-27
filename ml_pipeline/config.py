"""
Central configuration for the ML pipeline.

All model IDs, paths, thresholds, and hyperparameters are defined here
so they can be tuned from a single location. Values can be overridden
via environment variables.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


# Base directory of the ml_pipeline package
BASE_DIR = Path(__file__).resolve().parent


def _env(key: str, default, cast=str):
    """Read an environment variable and cast to the desired type."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        if cast is bool:
            return val.lower() in ("true", "1", "yes")
        return cast(val)
    except (ValueError, TypeError):
        return default


@dataclass
class PipelineConfig:
    """Master configuration for the self-improving pipeline."""

    # ── Inference ──────────────────────────────────────────────
    inference_model_id: str = field(
        default_factory=lambda: _env("INFERENCE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    )
    max_new_tokens: int = field(
        default_factory=lambda: _env("MAX_NEW_TOKENS", 256, int)
    )
    temperature: float = field(
        default_factory=lambda: _env("TEMPERATURE", 0.7, float)
    )
    top_p: float = field(
        default_factory=lambda: _env("TOP_P", 0.9, float)
    )
    device: str = field(
        default_factory=lambda: _env("DEVICE", "auto")
    )

    # ── Evaluation ─────────────────────────────────────────────
    eval_model_id: str = field(
        default_factory=lambda: _env("EVAL_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    )
    similarity_threshold: float = field(
        default_factory=lambda: _env("SIMILARITY_THRESHOLD", 0.70, float)
    )

    # ── Feedback ───────────────────────────────────────────────
    fine_tune_dataset_path: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "fine_tune_dataset.jsonl"
    )
    fine_tune_trigger_count: int = field(
        default_factory=lambda: _env("FINE_TUNE_TRIGGER_COUNT", 50, int)
    )

    # ── Ground Truth ───────────────────────────────────────────
    ground_truth_path: Path = field(
        default_factory=lambda: BASE_DIR / "ground_truth.json"
    )

    # ── Hybrid Evaluator ───────────────────────────────────────
    cosine_weight: float = field(
        default_factory=lambda: _env("COSINE_WEIGHT", 0.6, float)
    )
    llm_judge_weight: float = field(
        default_factory=lambda: _env("LLM_JUDGE_WEIGHT", 0.4, float)
    )
    llm_judge_temperature: float = 0.1

    # ── Teacher Correction ─────────────────────────────────────
    teacher_model_id: str = field(
        default_factory=lambda: _env("INFERENCE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    )
    gt_match_threshold: float = 0.50

    # ── Curation Queue ─────────────────────────────────────────
    curation_queue_path: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "curation_queue.jsonl"
    )
    auto_approve_threshold: float = 0.85

    # ── Dataset Monitor ────────────────────────────────────────
    dataset_monitor_interval: int = 30

    # ── LoRA Training ──────────────────────────────────────────
    lora_r: int = field(
        default_factory=lambda: _env("LORA_R", 8, int)
    )
    lora_alpha: int = field(
        default_factory=lambda: _env("LORA_ALPHA", 32, int)
    )
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    training_epochs: int = field(
        default_factory=lambda: _env("TRAINING_EPOCHS", 3, int)
    )
    training_batch_size: int = field(
        default_factory=lambda: _env("TRAINING_BATCH_SIZE", 4, int)
    )
    learning_rate: float = field(
        default_factory=lambda: _env("LEARNING_RATE", 2e-4, float)
    )
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
