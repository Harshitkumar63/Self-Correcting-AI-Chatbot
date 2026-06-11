"""
Inference Module — Generates LLM responses using Qwen2.5-0.5B-Instruct.

Uses HuggingFace Transformers with lazy model loading to avoid
loading the model until the first inference call.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ml_pipeline.config import config

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Wrapper around the Qwen2.5-0.5B-Instruct model for text generation.
    
    The model and tokenizer are loaded lazily on the first call to
    `generate()` so that the import of this module doesn't block startup.
    """

    def __init__(self):
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device: Optional[str] = None

    # ── Private helpers ─────────────────────────────────────────

    def _resolve_device(self) -> str:
        """Pick the best available device."""
        if config.device != "auto":
            return config.device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> None:
        """Download (if needed) and load the model + tokenizer."""
        if self._model is not None:
            return

        logger.info("Loading inference model: %s", config.inference_model_id)
        self._device = self._resolve_device()

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.inference_model_id,
            trust_remote_code=True,
        )

        # Load in float16 on GPU, float32 on CPU
        dtype = torch.float16 if self._device == "cuda" else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            config.inference_model_id,
            torch_dtype=dtype,
            device_map=self._device if self._device == "cuda" else None,
            trust_remote_code=True,
        )

        if self._device == "cpu":
            self._model = self._model.to("cpu")

        self._model.eval()
        logger.info("Model loaded on device: %s", self._device)

    # ── Public API ──────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate a response for the given user prompt.

        Args:
            prompt: The user's query text.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.

        Returns:
            The model's generated text response.
        """
        self._load_model()

        _max_tokens = max_new_tokens or config.max_new_tokens
        _temperature = temperature or config.temperature
        _top_p = top_p or config.top_p

        # Format as ChatML (Qwen2.5-Instruct expected format)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, accurate, and concise assistant. "
                    "Answer the user's question directly and factually."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(
            self._model.device
        )

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=_max_tokens,
                temperature=_temperature,
                top_p=_top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the input)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        logger.info(
            "Generated response (%d tokens) for query: %.60s...",
            len(generated_ids),
            prompt,
        )
        return response

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return self._model is not None

    def unload(self) -> None:
        """Free model from memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Inference model unloaded.")
