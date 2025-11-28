"""TaskEval Mini: Task-oriented LLM evaluation toolkit.

This package provides:
- A small synthetic dataset of support-style requests.
- Utilities to load tasks.
- Simple model runner interfaces.
- Basic metrics for structured JSON outputs.
- Evaluation scripts to compare model behaviour.
"""
__all__ = [
    "config",
    "data_loader",
    "model_runner",
    "metrics",
    "evaluator",
    "prompts",
]
