"""
Innovation Tracker for persistent structural innovation IDs.
Ensures identical historical origin tracking across mutations and multi-parent evolution.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional
import threading


@dataclass
class InnovationRecord:
    innovation_id: int
    node_type: str
    source_id: Optional[int]
    target_id: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class InnovationTracker:
    """
    Thread-safe global and local tracker for structural innovation IDs.
    Assigns persistent IDs to structural nodes and genes.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, start_id: int = 1):
        self.current_id = start_id
        self._history: Dict[Tuple[str, Optional[int], Optional[int]], int] = {}
        self._records: Dict[int, InnovationRecord] = {}

    @classmethod
    def get_global_tracker(cls) -> "InnovationTracker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_global_tracker(cls, start_id: int = 1):
        with cls._lock:
            cls._instance = cls(start_id)
            return cls._instance

    def get_innovation_id(
        self,
        node_type: str,
        source_id: Optional[int] = None,
        target_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        key = (node_type, source_id, target_id)
        if key in self._history:
            return self._history[key]

        assigned_id = self.current_id
        self.current_id += 1
        self._history[key] = assigned_id
        self._records[assigned_id] = InnovationRecord(
            innovation_id=assigned_id,
            node_type=node_type,
            source_id=source_id,
            target_id=target_id,
            metadata=metadata or {},
        )
        return assigned_id

    def get_record(self, innovation_id: int) -> Optional[InnovationRecord]:
        return self._records.get(innovation_id)

    def total_innovations(self) -> int:
        return self.current_id - 1
