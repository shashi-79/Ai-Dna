"""
Unit tests for YaRN RoPE and GRPO Reasoning Engine.
"""

import torch
import torch.nn as nn
from ai_dna.dna.structure import Genotype
from ai_dna.growth.engine import GrowthEngine
from ai_dna.models.rope import RoPE, RoPE2D, RoPE3D
from ai_dna.reasoning.verifier import ReasoningVerifier
from ai_dna.reasoning.grpo import GRPOTrainer


def test_yarn_rope_scaling():
    rope = RoPE(dim=64, base=500000.0, max_position_embeddings=2048, scaling_factor=2.0)
    q = torch.randn(2, 4, 512, 64)
    k = torch.randn(2, 4, 512, 64)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.isnan(q_rot).any()
    assert not torch.isnan(k_rot).any()


def test_yarn_rope_long_context():
    rope = RoPE(dim=64, base=500000.0, max_position_embeddings=2048, scaling_factor=4.0)
    # Long context sequence (4096 tokens > 2048)
    q_long = torch.randn(1, 2, 4096, 64)
    k_long = torch.randn(1, 2, 4096, 64)
    q_rot, k_rot = rope(q_long, k_long)
    assert q_rot.shape == (1, 2, 4096, 64)
    assert not torch.isinf(q_rot).any()


def test_reasoning_verifier():
    verifier = ReasoningVerifier()
    
    # Valid thought tags and correct answer
    text_good = "<thought> let's compute step by step: 15 + 28 is 43 </thought> 43"
    score_dict = verifier.compute_composite_reward(text_good, ground_truth_answer="43", token_length=15)
    assert score_dict["reward_accuracy"] == 1.0
    assert score_dict["reward_format"] == 1.0
    assert score_dict["reward_total"] > 1.0

    # Bad format without thought tags
    text_bad = "43"
    score_dict_bad = verifier.compute_composite_reward(text_bad, ground_truth_answer="43", token_length=2)
    assert score_dict_bad["reward_format"] == 0.0


def test_grpo_trainer_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genotype = Genotype.create_default(genotype_id="test_grpo_dna")
    growth_engine = GrowthEngine(device=device)
    model = growth_engine.grow_phenotype_model(genotype).to(device)
    
    verifier = ReasoningVerifier()
    trainer = GRPOTrainer(model=model, verifier=verifier, group_size=2, device=device)
    
    prompt = torch.tensor([[10, 65, 11, 75, 12]], dtype=torch.long, device=device)
    metrics = trainer.step_grpo_update(prompt, ground_truth_answers=["140", "140"], max_gen_len=4)
    
    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "mean_reward" in metrics
    assert not math_is_nan(metrics["total_loss"])


def math_is_nan(val):
    import math
    return math.isnan(val)
