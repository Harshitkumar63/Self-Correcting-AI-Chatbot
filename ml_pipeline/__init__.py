"""
Self-Improving LLM Pipeline — ML Pipeline Module

Core ML components:
- InferenceEngine: Generates responses using Qwen2.5-0.5B-Instruct
- EvaluationEngine: Hybrid evaluator (cosine + LLM-as-a-Judge)
- LLMJudge: Structured quality assessment via prompted LLM
- FeedbackCollector: Stores approved responses for retraining
- CurationQueue: Human-in-the-loop review queue
- TeacherCorrector: Generates clean, verified corrections
- run_training: PEFT/LoRA fine-tuning script
"""

from ml_pipeline.config import PipelineConfig
