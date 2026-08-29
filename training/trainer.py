"""
Failproof Multi-Modal Streaming Trainer for AI-DNA Package 1.
Features interleaved multi-modal dispatching, joint gradient backpropagation,
atomic periodic checkpointing, and graceful interrupt handling (Ctrl+C).
"""

import os
import sys
import time
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from ai_dna.dna.structure import Genotype
from .dataset_manager import StreamDatasetManager
from .checkpoint_manager import FailproofCheckpointManager


class MultiModalStreamingTrainer:
    """
    Unified failproof multi-modal trainer for streaming foundation datasets.
    """
    def __init__(
        self,
        model: nn.Module,
        genotype: Genotype,
        dataset_manager: StreamDatasetManager,
        checkpoint_manager: FailproofCheckpointManager,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_steps: int = 120000,
        max_duration_hours: Optional[float] = None,
        save_every_steps: int = 250,
        log_every_steps: int = 25,
        grad_clip: float = 1.0,
        device: torch.device = torch.device("cpu"),
        auto_resume: bool = True,
    ):
        self.model = model.to(device)
        self.genotype = genotype
        self.dataset_manager = dataset_manager
        self.checkpoint_manager = checkpoint_manager
        self.max_steps = max_steps
        self.max_duration_hours = max_duration_hours
        self.save_every_steps = save_every_steps
        self.log_every_steps = log_every_steps
        self.grad_clip = grad_clip
        self.device = device
        self.auto_resume = auto_resume

        # Trainable parameters (Fast Clock LoRA + Projections + Heads)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = True
            trainable = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_steps, eta_min=1e-5)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        self.current_step = 0
        self.tokens_processed = 0
        self.best_loss = float("inf")
        self.stop_requested = False

        # Register interruption signal traps (SIGINT / SIGTERM)
        self._register_signals()

        # Check for existing checkpoint to resume
        if self.auto_resume:
            self._try_resume_from_checkpoint()

    def _register_signals(self):
        """Catches Ctrl+C or kill signals to save state before exiting."""
        def handler(sig, frame):
            print("\n" + "=" * 70)
            print("[INTERRUPT DETECTED]: Gracefully pausing training...")
            print("   Completing current step & saving atomic state...")
            print("=" * 70)
            self.stop_requested = True

        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)

    def _try_resume_from_checkpoint(self):
        """Discovers and loads the most recent valid checkpoint."""
        latest_cp = self.checkpoint_manager.find_latest_checkpoint()
        if latest_cp:
            print(f"\n[AUTO-RESUME]: Found valid checkpoint at: {latest_cp}")
            state = self.checkpoint_manager.load_checkpoint(
                latest_cp,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
            self.current_step = state["step"]
            self.tokens_processed = state["tokens_processed"]
            loaded_offsets = state["stream_offsets"]
            
            # Update dataset streaming cursors
            if loaded_offsets:
                self.dataset_manager.offsets.update(loaded_offsets)
                self.dataset_manager._init_streams()
            
            print(f"   [+] Resumed at Step: {self.current_step} | Total Tokens: {self.tokens_processed:,}")
            print(f"   [+] Stream Cursors: {loaded_offsets}\n")
        else:
            print("\n[INIT]: No prior checkpoint found. Starting fresh from Step 0.\n")

    def train_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """Executes forward pass, loss calculation, and backpropagation for one batch."""
        self.model.train()
        self.optimizer.zero_grad()

        modality = batch["modality"]
        loss_type = batch["loss_type"]
        aux_loss_val = torch.tensor(0.0, device=self.device)

        if loss_type == "autoregressive":
            inp = batch["input"]
            target = batch["target"]
            # Forward pass through unified MoE
            h, aux_loss, _, _ = self.model(inp, modality=modality, is_causal=True)
            logits = self.model.ar_head(h)
            
            # Cross entropy loss on next-token targets
            vocab_size = logits.size(-1)
            target_flat = target.view(-1) % vocab_size
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), target_flat)
            total_loss = ce_loss + 0.01 * aux_loss
            aux_loss_val = aux_loss

        elif loss_type == "vqa_captioning":
            images = batch["input"]
            captions = batch["prompt"]
            # Pass image patches + caption tokens through multimodal stream
            h, aux_loss, _, _ = self.model.forward_multimodal(text_inputs=captions, vision_inputs=images, is_causal=True)
            logits = self.model.ar_head(h)
            
            # Target the caption slice at the end of the sequence
            cap_len = captions.size(1)
            cap_logits = logits[:, -cap_len:, :]
            vocab_size = cap_logits.size(-1)
            target_flat = captions.view(-1) % vocab_size
            ce_loss = F.cross_entropy(cap_logits.reshape(-1, vocab_size), target_flat)
            total_loss = ce_loss + 0.01 * aux_loss
            aux_loss_val = aux_loss

        elif loss_type == "audio_spectrogram":
            inp = batch["input"]
            target = batch["target"]
            h, aux_loss, _, _ = self.model(inp, modality="audio")
            pred_spec = self.model.audio_head(h)
            spec_loss = F.mse_loss(pred_spec, target)
            total_loss = spec_loss + 0.01 * aux_loss
            aux_loss_val = aux_loss

        elif loss_type == "video_action":
            inp = batch["input"]
            target = batch["target"]
            h, aux_loss, _, _ = self.model(inp, modality="video")
            logits = self.model.cls_head(h)
            cls_loss = F.cross_entropy(logits, target)
            total_loss = cls_loss + 0.01 * aux_loss
            aux_loss_val = aux_loss

        elif loss_type == "diffusion_denoise":
            noisy_x = batch["noisy_input"]
            target_noise = batch["target_noise"]
            timesteps = batch["timesteps"]
            prompts = batch["prompt"]
            
            h_prompt, aux_loss, _, _ = self.model(prompts, modality="text")
            pred_noise = self.model.diff_head(noisy_x, timesteps, h_prompt[:, :noisy_x.size(1), :])
            diff_loss = F.mse_loss(pred_noise, target_noise)
            total_loss = diff_loss + 0.01 * aux_loss
            aux_loss_val = aux_loss

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Backward pass & Gradient clipping
        total_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        return total_loss.detach(), aux_loss_val.item()

    def train_loop(self):
        """Main failproof multi-modal training execution loop."""
        duration_msg = f" | Max Duration: {self.max_duration_hours:.2f} Hours" if self.max_duration_hours else ""
        print("=" * 70)
        print("[START] FAILPROOF STREAMING TRAINING (PACKAGE 1)")
        print(f"   Target Steps: {self.max_steps} | Save Interval: Every {self.save_every_steps} steps{duration_msg}")
        print(f"   Hardware Device: {self.device}")
        print("=" * 70 + "\n")

        start_time = time.time()
        running_loss = 0.0
        step_window_start = time.time()

        while self.current_step < self.max_steps:
            if self.stop_requested:
                self._save_emergency_checkpoint(running_loss)
                break

            # Check max duration limit
            elapsed_hours = (time.time() - start_time) / 3600.0
            if self.max_duration_hours is not None and elapsed_hours >= self.max_duration_hours:
                print(f"\n[TIME LIMIT REACHED]: Completed target training duration of {self.max_duration_hours:.2f} hours.")
                break

            self.current_step += 1
            batch = self.dataset_manager.get_interleaved_batch()
            
            # Estimate tokens in batch
            batch_tokens = batch["input"].numel() if "input" in batch else (batch.get("noisy_input", torch.empty(0)).numel())
            self.tokens_processed += batch_tokens

            step_loss, aux_l = self.train_step(batch)
            running_loss += step_loss.item()

            # Periodic Progress Logging
            if self.current_step % self.log_every_steps == 0:
                elapsed_window = time.time() - step_window_start
                steps_per_sec = self.log_every_steps / max(1e-4, elapsed_window)
                tokens_per_sec = (batch_tokens * self.log_every_steps) / max(1e-4, elapsed_window)
                avg_loss = running_loss / self.log_every_steps
                lr_curr = self.optimizer.param_groups[0]["lr"]

                # GPU VRAM Tracking
                vram_info = ""
                if self.device.type == "cuda":
                    vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                    vram_max = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    vram_info = f" | GPU VRAM: {vram_mb:.0f}MB / {vram_max:.0f}MB"

                # Estimate time remaining
                remaining_steps = self.max_steps - self.current_step
                eta_sec = remaining_steps / max(1e-4, steps_per_sec)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

                print(
                    f"[{self.current_step:06d}/{self.max_steps:06d}] "
                    f"Modality: {batch['modality'].upper():<9} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_curr:.2e} | "
                    f"Throughput: {tokens_per_sec:,.0f} tok/s ({steps_per_sec:.1f} it/s){vram_info} | "
                    f"ETA: {eta_str}"
                )
                running_loss = 0.0
                step_window_start = time.time()

            # Periodic Atomic Checkpointing
            if self.current_step % self.save_every_steps == 0:
                is_best = step_loss.item() < self.best_loss
                if is_best:
                    self.best_loss = step_loss.item()

                saved_path = self.checkpoint_manager.save_checkpoint(
                    step=self.current_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    stream_offsets=self.dataset_manager.get_state(),
                    tokens_processed=self.tokens_processed,
                    loss=step_loss.item(),
                    genotype=self.genotype,
                    is_best=is_best,
                )
                print(f"   [CHECKPOINT SAVED]: {saved_path} (Tokens: {self.tokens_processed:,})")

        # Final completion save
        if self.current_step >= self.max_steps:
            final_path = self.checkpoint_manager.save_checkpoint(
                step=self.current_step,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                stream_offsets=self.dataset_manager.get_state(),
                tokens_processed=self.tokens_processed,
                loss=running_loss / max(1, self.log_every_steps),
                genotype=self.genotype,
                is_best=True,
            )
            total_duration = time.time() - start_time
            print("\n" + "=" * 70)
            print(f"[COMPLETE] TRAINING COMPLETE: {self.current_step} Steps in {total_duration:.1f}s")
            print(f"   Final Model State Saved: {final_path}")
            print("=" * 70)

    def _save_emergency_checkpoint(self, current_loss: float):
        """Saves an atomic checkpoint when interruption is requested."""
        path = self.checkpoint_manager.save_checkpoint(
            step=self.current_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            stream_offsets=self.dataset_manager.get_state(),
            tokens_processed=self.tokens_processed,
            loss=current_loss,
            genotype=self.genotype,
        )
        print(f"\n[FAILPROOF RECOVERY]: Emergency checkpoint safely preserved at:\n   -> {path}")
        print(f"   To resume from this exact token offset, simply re-run the training script!\n")
        sys.exit(0)
