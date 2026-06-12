"""
Training Module — PEFT/LoRA fine-tuning using the collected feedback dataset.

Loads the base Qwen2.5-0.5B model, applies LoRA adapters, and trains
on the feedback JSONL file using HuggingFace's SFTTrainer.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ml_pipeline.config import config
from ml_pipeline.feedback import FeedbackCollector

logger = logging.getLogger(__name__)


class TrainingStatus:
    """Tracks the state of a training run."""

    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

    def __init__(self):
        self.state: str = self.IDLE
        self.message: str = ""
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "message": self.message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# Global training status
training_status = TrainingStatus()


def run_training(feedback_collector: Optional[FeedbackCollector] = None) -> dict:
    """
    Execute a LoRA fine-tuning run using the feedback dataset.

    Steps:
    1. Load the feedback JSONL dataset.
    2. Load the base model and apply LoRA config.
    3. Train using SFTTrainer.
    4. Save LoRA adapters to the checkpoint directory.
    5. Archive the used dataset.

    Args:
        feedback_collector: Optional FeedbackCollector instance.
            If None, creates a new one.

    Returns:
        Dict with training results and status.
    """
    global training_status

    if feedback_collector is None:
        feedback_collector = FeedbackCollector()

    training_status.state = TrainingStatus.PREPARING
    training_status.started_at = datetime.utcnow().isoformat()
    training_status.message = "Preparing training data..."

    try:
        # ── Step 1: Load dataset ────────────────────────────────
        dataset_path = config.fine_tune_dataset_path
        if not dataset_path.exists() or feedback_collector.get_sample_count() == 0:
            training_status.state = TrainingStatus.FAILED
            training_status.message = "No training data available."
            training_status.error = "Dataset is empty."
            return training_status.to_dict()

        sample_count = feedback_collector.get_sample_count()
        logger.info("Starting training with %d samples.", sample_count)

        # Import heavy dependencies only when training is triggered
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer, SFTConfig

        # ── Step 2: Load dataset from JSONL ─────────────────────
        training_status.message = "Loading dataset..."
        dataset = load_dataset(
            "json",
            data_files=str(dataset_path),
            split="train",
        )

        # Format dataset for SFT — combine instruction and output
        def format_sample(example):
            text = (
                f"<|im_start|>system\n"
                f"You are a helpful, accurate, and concise assistant.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{example['instruction']}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"{example['output']}<|im_end|>"
            )
            return {"text": text}

        dataset = dataset.map(format_sample)

        # ── Step 3: Load model + tokenizer ──────────────────────
        training_status.message = "Loading base model..."
        training_status.state = TrainingStatus.TRAINING

        tokenizer = AutoTokenizer.from_pretrained(
            config.inference_model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            config.inference_model_id,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )
        if device == "cpu":
            model = model.to("cpu")

        # ── Step 4: Apply LoRA ──────────────────────────────────
        training_status.message = "Applying LoRA adapters..."

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # ── Step 5: Configure trainer ───────────────────────────
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = config.checkpoint_dir / f"lora_{timestamp}"

        training_args = SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=config.training_batch_size,
            num_train_epochs=config.training_epochs,
            learning_rate=config.learning_rate,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            fp16=(device == "cuda"),
            report_to="none",  # disable wandb etc.
            max_length=512,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

        # ── Step 6: Train ───────────────────────────────────────
        training_status.message = f"Training on {sample_count} samples..."
        logger.info("Starting LoRA training...")

        trainer.train()

        # ── Step 7: Save adapter ────────────────────────────────
        training_status.message = "Saving LoRA adapter..."
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        logger.info("LoRA adapter saved to %s", output_dir)

        # ── Step 8: Archive dataset ─────────────────────────────
        archive_path = feedback_collector.archive_and_clear()

        # ── Done ────────────────────────────────────────────────
        training_status.state = TrainingStatus.COMPLETED
        training_status.completed_at = datetime.utcnow().isoformat()
        training_status.message = (
            f"Training complete! {sample_count} samples processed. "
            f"Adapter saved to {output_dir.name}"
        )

        return {
            **training_status.to_dict(),
            "samples_trained": sample_count,
            "adapter_path": str(output_dir),
            "archive_path": str(archive_path) if archive_path else None,
        }

    except Exception as e:
        logger.exception("Training failed: %s", str(e))
        training_status.state = TrainingStatus.FAILED
        training_status.completed_at = datetime.utcnow().isoformat()
        training_status.error = str(e)
        training_status.message = f"Training failed: {str(e)}"
        return training_status.to_dict()


def get_training_status() -> dict:
    """Return the current training status."""
    return training_status.to_dict()
