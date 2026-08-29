"""
Failproof Multi-Modal Streaming Training Suite.
Provides zero-disk streaming dataset managers, stateful atomic checkpointing,
and resilient multi-modal training loops.
"""

from .dataset_manager import StreamDatasetManager
from .checkpoint_manager import FailproofCheckpointManager
from .trainer import MultiModalStreamingTrainer

__all__ = [
    "StreamDatasetManager",
    "FailproofCheckpointManager",
    "MultiModalStreamingTrainer",
]
