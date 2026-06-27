"""
Inference Module — Generates LLM responses using Qwen2.5-0.5B-Instruct.

Supports:
- Single-turn and multi-turn conversation generation
- Token-by-token streaming for SSE endpoints
- Lazy model loading to avoid blocking startup
"""

import logging
from typing import Generator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

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

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Build the messages list for the chat template."""
        _system = system_prompt or (
            "You are a helpful, accurate, and concise assistant. "
            "Answer the user's question directly and factually."
        )

        messages = [{"role": "system", "content": _system}]

        # Add conversation history for multi-turn context
        if history:
            # Limit to last 10 messages to prevent context overflow
            for msg in history[-10:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        messages.append({"role": "user", "content": prompt})
        return messages

    # ── Public API ──────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate a response for the given user prompt.

        Args:
            prompt: The user's query text.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.
            system_prompt: Override the default system prompt.
            history: Conversation history for multi-turn context.
                     List of {"role": "user/assistant", "content": "..."}.

        Returns:
            The model's generated text response.
        """
        self._load_model()

        _max_tokens = max_new_tokens or config.max_new_tokens
        _temperature = temperature or config.temperature
        _top_p = top_p or config.top_p

        messages = self._build_messages(prompt, system_prompt, history)

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

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens one-by-one using TextIteratorStreamer.

        Args:
            prompt: The user's query text.
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.
            system_prompt: Override the default system prompt.
            history: Conversation history for multi-turn context.

        Yields:
            String tokens as they are generated.
        """
        self._load_model()

        _max_tokens = max_new_tokens or config.max_new_tokens
        _temperature = temperature or config.temperature
        _top_p = top_p or config.top_p

        messages = self._build_messages(prompt, system_prompt, history)

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt").to(
            self._model.device
        )

        # Create a streamer that yields tokens as they're generated
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Run generation in a separate thread
        generation_kwargs = {
            **inputs,
            "max_new_tokens": _max_tokens,
            "temperature": _temperature,
            "top_p": _top_p,
            "do_sample": True,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they arrive
        for token_text in streamer:
            if token_text:
                yield token_text

        thread.join()

        logger.info("Streamed response for query: %.60s...", prompt)

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
