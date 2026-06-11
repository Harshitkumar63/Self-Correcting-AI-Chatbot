"""
Self-Improving LLM Pipeline — ML Pipeline Module

Core ML components:
- InferenceEngine: Generates responses using Qwen2.5-0.5B-Instruct
- EvaluationEngine: Scores responses via semantic similarity
- FeedbackCollector: Stores flagged responses for retraining
- run_training: PEFT/LoRA fine-tuning script
"""

from ml_pipeline.config import PipelineConfig
