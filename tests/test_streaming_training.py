"""
Unit tests for the Failproof Multi-Modal Streaming Training Suite (`training/`).
Tests atomic checkpointing, stateful stream resumption, and crash recovery.
"""

import os
import shutil
import unittest
import torch
from ai_dna.dna.structure import Genotype
from ai_dna.growth.engine import GrowthEngine
from training.dataset_manager import StreamDatasetManager
from training.checkpoint_manager import FailproofCheckpointManager
from training.trainer import MultiModalStreamingTrainer


class TestStreamingTraining(unittest.TestCase):

    def setUp(self):
        self.test_dir = "checkpoints/test_streaming_tmp"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_stream_dataset_manager(self):
        # 1. Test dataset manager batch generation
        dm = StreamDatasetManager(batch_size=2, seq_len=16, d_model=32, use_mock_fallback=True)
        batch = dm.get_interleaved_batch()
        self.assertIn("modality", batch)
        self.assertIn("loss_type", batch)

        # 2. Test state extraction
        state = dm.get_state()
        self.assertIn("text", state)
        self.assertIn("math", state)
        self.assertIn("diffusion", state)

    def test_checkpoint_manager_atomic_save_load(self):
        cm = FailproofCheckpointManager(checkpoint_dir=self.test_dir, keep_last_n=2)
        growth_engine = GrowthEngine()
        genotype = Genotype()
        model = growth_engine.grow_phenotype_model(genotype)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        stream_offsets = {"text": 100, "math": 50, "vision": 25}
        saved_path = cm.save_checkpoint(
            step=10,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            stream_offsets=stream_offsets,
            tokens_processed=5000,
            loss=1.234,
            genotype=genotype,
        )
        self.assertTrue(os.path.exists(saved_path))
        self.assertFalse(saved_path.endswith(".tmp"))

        # Find latest
        latest = cm.find_latest_checkpoint()
        self.assertEqual(latest, saved_path)

        # Load back into fresh model
        model_new = growth_engine.grow_phenotype_model(genotype)
        optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=1e-3)
        restored_data = cm.load_checkpoint(latest, model=model_new, optimizer=optimizer_new)

        self.assertEqual(restored_data["step"], 10)
        self.assertEqual(restored_data["tokens_processed"], 5000)
        self.assertEqual(restored_data["stream_offsets"]["text"], 100)

    def test_mid_run_pause_and_resume_simulation(self):
        # 1. Run first phase for 5 steps
        cm = FailproofCheckpointManager(checkpoint_dir=self.test_dir, keep_last_n=2)
        dm1 = StreamDatasetManager(batch_size=2, seq_len=16, d_model=32, use_mock_fallback=True)
        growth_engine = GrowthEngine()
        genotype = Genotype()
        model1 = growth_engine.grow_phenotype_model(genotype)

        trainer1 = MultiModalStreamingTrainer(
            model=model1,
            genotype=genotype,
            dataset_manager=dm1,
            checkpoint_manager=cm,
            max_steps=5,
            save_every_steps=5,
            log_every_steps=1,
            auto_resume=False,
        )
        trainer1.train_loop()
        self.assertEqual(trainer1.current_step, 5)

        # 2. Simulate interruption and resume with a brand new Trainer instance
        dm2 = StreamDatasetManager(batch_size=2, seq_len=16, d_model=32, use_mock_fallback=True)
        model2 = growth_engine.grow_phenotype_model(genotype)

        trainer2 = MultiModalStreamingTrainer(
            model=model2,
            genotype=genotype,
            dataset_manager=dm2,
            checkpoint_manager=cm,
            max_steps=10,
            save_every_steps=5,
            log_every_steps=1,
            auto_resume=True,  # Auto-detects Step 5 checkpoint
        )
        self.assertEqual(trainer2.current_step, 5)
        self.assertGreater(trainer2.tokens_processed, 0)

        # 3. Continue training to Step 10
        trainer2.train_loop()
        self.assertEqual(trainer2.current_step, 10)


if __name__ == "__main__":
    unittest.main()
