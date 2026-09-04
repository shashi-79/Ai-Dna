"""
Training and Optimization Package for AI-DNA.
Includes FastClock gradient descent, MultiModal streaming trainer, losses, metrics,
and failproof stateful checkpoint managers.
"""

from .losses import JointLoss
from .metrics import EvaluationMetrics, FalsificationChecker, FalsificationResult
from .fast_clock import FastClockTrainer
from .checkpoint_manager import FailproofCheckpointManager
from .dataset_manager import StreamDatasetManager
from .trainer import MultiModalStreamingTrainer

__all__ = [
    "JointLoss",
    "EvaluationMetrics",
    "FalsificationChecker",
    "FalsificationResult",
    "FastClockTrainer",
    "FailproofCheckpointManager",
    "StreamDatasetManager",
    "MultiModalStreamingTrainer",
]
