"""
Standard Evaluation and Benchmarking Suite for AI-DNA.
Provides task loaders (MMLU, GSM8K, ARC, IFEval), batched inference engine,
and model card metadata generators.
"""

from .tasks import (
    FALLBACK_QUESTIONS,
    MMLU_SUBJECTS_SAMPLE,
    TASK_LOADERS,
    TASK_SCORERS,
    check_mc,
    check_gsm8k,
    check_ifeval,
    load_mmlu,
    load_gsm8k,
    load_arc,
    load_ifeval,
)
from .engine import (
    auto_detect_resources,
    auto_batch,
    BatchedInferenceEngine,
    evaluate_hf_model,
)
from .card_updater import (
    format_benchmark_markdown_table,
    update_readme_model_index,
)

__all__ = [
    "FALLBACK_QUESTIONS",
    "MMLU_SUBJECTS_SAMPLE",
    "TASK_LOADERS",
    "TASK_SCORERS",
    "check_mc",
    "check_gsm8k",
    "check_ifeval",
    "load_mmlu",
    "load_gsm8k",
    "load_arc",
    "load_ifeval",
    "auto_detect_resources",
    "auto_batch",
    "BatchedInferenceEngine",
    "evaluate_hf_model",
    "format_benchmark_markdown_table",
    "update_readme_model_index",
]
