"""
End-to-End Integration Test for AI DNA Bidirectional Lifecycle:
D_0 -> G -> W_0 -> FastClock -> W_0* -> SlowClock -> D_1 -> G -> W_1 -> Inference.
"""

import torch
from ai_dna.dna.structure import Genotype
from ai_dna.growth.engine import GrowthEngine
from ai_dna.models.phenotype import PhenotypeNeuralNetwork
from ai_dna.training.fast_clock import FastClockTrainer
from ai_dna.encoding.slow_clock import SlowClockEncoder
from ai_dna.inference.pipeline import InferencePipeline


def test_complete_bidirectional_lifecycle():
    # 1. Genotype D_0
    d_0 = Genotype.create_default(genotype_id="cycle_root")
    d_0.dna_architecture.vocab_size = 20
    d_0.dna_architecture.d_model = 16
    d_0.dna_architecture.num_layers = 1
    d_0.dna_architecture.num_experts = 2
    d_0.dna_architecture.d_expert_hidden = 32

    # 2. Growth G(D_0) -> W_0
    growth_engine = GrowthEngine()
    w_0 = PhenotypeNeuralNetwork(d_0)
    w0_weights = growth_engine.grow_phenotype_weights(d_0)
    w0_state = w_0.state_dict()
    for k, v in w0_weights.items():
        if k in w0_state and w0_state[k].shape == v.shape:
            w0_state[k] = v
    w_0.load_state_dict(w0_state)

    # 3. Fast Clock Training
    trainer = FastClockTrainer(w_0, learning_rate=1e-2)
    x = torch.randint(0, 20, (4, 8))
    y = torch.randint(0, 10, (4,))
    loss, _ = trainer.train_step_classification(x, y, modality="text")
    assert loss >= 0.0

    # 4. Slow Clock Encoding -> D_1
    slow_clock = SlowClockEncoder(rank_ratio=0.5, encoder_steps=15)
    d_1, summary = slow_clock.step(d_0, w_0.state_dict())
    assert d_1.generation == 1

    # 5. Growth G(D_1) -> W_1
    w_1 = PhenotypeNeuralNetwork(d_1)
    w1_weights = growth_engine.grow_phenotype_weights(d_1)
    w1_state = w_1.state_dict()
    for k, v in w1_weights.items():
        if k in w1_state and w1_state[k].shape == v.shape:
            w1_state[k] = v
    w_1.load_state_dict(w1_state)

    # 6. Inference on regenerated model
    pipeline = InferencePipeline(genotype=d_1)
    prompt = torch.tensor([[1, 2, 3]])
    res = pipeline.generate(prompt, modality="text", mode="autoregressive", max_new_tokens=3)
    assert res["mode"] == "autoregressive"
    assert res["output"].shape == (1, 6)
