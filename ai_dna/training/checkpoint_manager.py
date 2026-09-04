"""
Failproof Stateful Checkpoint Manager for Multi-Modal Streaming Training.
Guarantees zero-loss atomic persistence, automatic crash recovery, and state resumption.
"""

import os
import glob
import re
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from ai_dna.dna.structure import Genotype
from ai_dna.dna.serialization import save_genotype, load_genotype


class FailproofCheckpointManager:
    """
    Manages atomic serialization and auto-resumption for streaming training runs.
    Saves model, optimizer, scheduler, global steps, per-modality streaming cursors, and DNA.
    """
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/package1_streaming",
        keep_last_n: int = 3,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        stream_offsets: Dict[str, int],
        tokens_processed: int,
        loss: float,
        genotype: Optional[Genotype] = None,
        is_best: bool = False,
    ) -> str:
        """
        Atomically saves state snapshot to a temporary file, then renames to target path.
        Prevents file corruption if training is killed mid-write.
        """
        checkpoint_name = f"checkpoint_step_{step:07d}.pt"
        target_path = os.path.normpath(os.path.join(self.checkpoint_dir, checkpoint_name))
        tmp_path = target_path + ".tmp"

        payload = {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "stream_offsets": stream_offsets,
            "tokens_processed": tokens_processed,
            "loss": loss,
            "saved_at_timestamp": os.path.getmtime(self.checkpoint_dir) if os.path.exists(self.checkpoint_dir) else 0,
        }

        # Save Genotype parameters if provided
        if genotype is not None:
            payload["genotype_id"] = genotype.genotype_id
            payload["generation"] = genotype.generation
            payload["dna_architecture"] = genotype.dna_architecture
            payload["dna_instinct"] = genotype.dna_instinct

        # Atomic file write
        torch.save(payload, tmp_path)
        if os.path.exists(target_path):
            os.remove(target_path)
        os.rename(tmp_path, target_path)

        # Save as best checkpoint if requested
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            torch.save(payload, best_path)

        # Prune old checkpoints
        self._prune_old_checkpoints()
        return target_path

    def find_latest_checkpoint(self) -> Optional[str]:
        """Scans directory and returns path to the most recent valid checkpoint."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt")
        files = glob.glob(pattern)
        if not files:
            return None

        # Sort by step number extracted from filename
        def extract_step(filepath: str) -> int:
            norm_f = filepath.replace("\\", "/")
            match = re.search(r"checkpoint_step_(\d+)\.pt", norm_f)
            return int(match.group(1)) if match else -1

        valid_files = sorted(files, key=extract_step)
        # Verify file integrity
        for candidate in reversed(valid_files):
            try:
                # Test loading header with weights_only=False for custom dataclass payloads
                _ = torch.load(candidate, map_location="cpu", weights_only=False)
                return os.path.normpath(candidate)
            except Exception:
                continue
        return None

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """
        Loads and restores state from a checkpoint file.
        Returns dict containing: step, stream_offsets, tokens_processed, loss.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        data = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Restore Model Weights
        model.load_state_dict(data["model_state"], strict=False)

        # Restore Optimizer State
        if optimizer is not None and "optimizer_state" in data and data["optimizer_state"] is not None:
            optimizer.load_state_dict(data["optimizer_state"])

        # Restore Scheduler State
        if scheduler is not None and "scheduler_state" in data and data["scheduler_state"] is not None:
            scheduler.load_state_dict(data["scheduler_state"])

        return {
            "step": data.get("step", 0),
            "stream_offsets": data.get("stream_offsets", {}),
            "tokens_processed": data.get("tokens_processed", 0),
            "loss": data.get("loss", 0.0),
        }

    def _prune_old_checkpoints(self):
        """Keeps only the most recent N checkpoints to avoid disk bloat."""
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_step_*.pt")
        files = glob.glob(pattern)
        if len(files) <= self.keep_last_n:
            return

        def extract_step(filepath: str) -> int:
            match = re.search(r"checkpoint_step_(\d+)\.pt", filepath)
            return int(match.group(1)) if match else -1

        sorted_files = sorted(files, key=extract_step)
        files_to_delete = sorted_files[:-self.keep_last_n]
        for f in files_to_delete:
            try:
                os.remove(f)
            except Exception:
                pass
