"""
Unit tests for Recurrent Depth & Looped Transformer execution.
"""

import os
import unittest
import torch
from ai_dna.models.recurrent_causal_lm import (
    RecurrentQwenForCausalLM,
    RecurrentRMSNorm,
    RecurrentRoPE,
)


class TestRecurrentDepth(unittest.TestCase):

    def test_recurrent_norm_and_rope(self):
        norm = RecurrentRMSNorm(dim=64)
        x = torch.randn(2, 4, 64)
        y = norm(x)
        self.assertEqual(y.shape, (2, 4, 64))
        self.assertFalse(torch.isnan(y).any())

        cos, sin = RecurrentRoPE.precompute_cos_sin(seq_len=16, dim=32, device=torch.device("cpu"))
        self.assertEqual(cos.shape, (1, 1, 16, 32))
        q = torch.randn(2, 4, 16, 32)
        q_rot = RecurrentRoPE.apply_rotary_emb(q, cos, sin)
        self.assertEqual(q_rot.shape, q.shape)

    def test_recurrent_qwen_forward_toy(self):
        config = {
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "recurrent_depth": 4,
            "recurrent_strategy": "step_lora",
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "intermediate_size": 128,
            "rope_theta": 10000.0,
        }
        model = RecurrentQwenForCausalLM(config)
        dev = torch.device("cpu")

        # Initialize toy weights
        model.embed_tokens = torch.randn(100, 64)
        model.lm_head = torch.randn(100, 64)
        model.final_norm.weight.data.fill_(1.0)
        model.input_layernorm.weight.data.fill_(1.0)
        model.post_attention_layernorm.weight.data.fill_(1.0)
        model.step_embeddings = torch.randn(4, 64)

        # Base layer weights
        model.base_weights = {
            "self_attn.q_proj.weight": torch.randn(64, 64),
            "self_attn.k_proj.weight": torch.randn(32, 64),
            "self_attn.v_proj.weight": torch.randn(32, 64),
            "self_attn.o_proj.weight": torch.randn(64, 64),
            "mlp.gate_proj.weight": torch.randn(128, 64),
            "mlp.up_proj.weight": torch.randn(128, 64),
            "mlp.down_proj.weight": torch.randn(64, 128),
        }

        # Step LoRA adapters (rank 4)
        for t in range(4):
            model.step_adapters[f"{t}.self_attn.q_proj.weight.lora_A"] = torch.randn(64, 4) * 0.01
            model.step_adapters[f"{t}.self_attn.q_proj.weight.lora_B"] = torch.randn(4, 64) * 0.01
            model.step_adapters[f"{t}.mlp.down_proj.weight.lora_A"] = torch.randn(64, 4) * 0.01
            model.step_adapters[f"{t}.mlp.down_proj.weight.lora_B"] = torch.randn(4, 128) * 0.01

        model.cos_cached, model.sin_cached = RecurrentRoPE.precompute_cos_sin(
            seq_len=64, dim=16, device=dev
        )

        # Test forward pass
        input_ids = torch.tensor([[1, 5, 10, 20], [2, 4, 8, 16]])
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 4, 100))
        self.assertFalse(torch.isnan(logits).any())

        # Test generation
        out_ids = model.generate(input_ids, max_new_tokens=5)
        self.assertEqual(out_ids.shape, (2, 9))

    def test_two_stage_layer_first_fusion_in_memory(self):
        import tempfile
        import shutil
        from safetensors.torch import save_file as safetensors_save_file
        from ai_dna.evolution.fusion import fuse_feedforward_layers_in_memory, DonorSpec

        # Primary with 4 layers (64x64 / 64x128)
        prim_weights = {
            "model.embed_tokens.weight": torch.randn(100, 64),
            "lm_head.weight": torch.randn(100, 64),
            "model.norm.weight": torch.ones(64),
        }
        for l in range(4):
            prim_weights[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(64, 64)
            prim_weights[f"model.layers.{l}.mlp.down_proj.weight"] = torch.randn(64, 128)

        # Donor with 4 layers
        donor_weights = {}
        for l in range(4):
            donor_weights[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(64, 64)
            donor_weights[f"model.layers.{l}.mlp.down_proj.weight"] = torch.randn(64, 128)

        temp_dir = os.path.join(os.path.dirname(__file__), "tmp_test_donor")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            donor_dir = os.path.join(temp_dir, "donor_code")
            os.makedirs(donor_dir, exist_ok=True)
            safetensors_save_file(donor_weights, os.path.join(donor_dir, "model.safetensors"))

            donor_spec = DonorSpec(path=donor_dir, weight=0.05, specialization="code")
            fused_weights, loaded_donors, depth_meta = fuse_feedforward_layers_in_memory(
                prim_weights=prim_weights,
                donor_specs=[donor_spec],
                rank=8,
                outlier_threshold=6.0,
            )

            # Invariant 1: Layer 0 MUST be 100% frozen primary (shallow band protection)
            self.assertTrue(torch.equal(
                fused_weights["model.layers.0.mlp.down_proj.weight"],
                prim_weights["model.layers.0.mlp.down_proj.weight"]
            ))
            # Invariant 2: Deep layer (Layer 3) must absorb donor information
            self.assertFalse(torch.equal(
                fused_weights["model.layers.3.mlp.down_proj.weight"],
                prim_weights["model.layers.3.mlp.down_proj.weight"]
            ))
            # Invariant 3: Vocabulary and norm unchanged
            self.assertTrue(torch.equal(
                fused_weights["model.embed_tokens.weight"],
                prim_weights["model.embed_tokens.weight"]
            ))
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
