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


def test_geca_lifecycle():
    # 1. Genotype D_0
    d_0 = Genotype.create_default(genotype_id="geca_root")
    d_0.dna_architecture.vocab_size = 20
    d_0.dna_architecture.d_model = 16
    d_0.dna_architecture.num_layers = 1
    d_0.dna_architecture.num_experts = 2
    d_0.dna_architecture.d_expert_hidden = 32

    # 2. Growth G(D_0) -> W_0
    growth_engine = GrowthEngine()
    w_0 = growth_engine.grow_phenotype_model(d_0)

    # 3. Slow Clock Encoding -> D_1 (extracts GECA anchors)
    slow_clock = SlowClockEncoder(rank_ratio=0.5, encoder_steps=5)
    d_1, summary = slow_clock.step(d_0, w_0.state_dict(), phenotype_model=w_0)
    
    assert "text" in d_1.calibration_anchors
    assert d_1.calibration_anchors["text"].shape == (8, 16 + 20) # K=8 anchors, d_model=16, vocab_size=20

    # 4. Serialize & Deserialize Genotype
    from ai_dna.dna.serialization import genotype_to_dict, dict_to_genotype
    serialized = genotype_to_dict(d_1)
    deserialized = dict_to_genotype(serialized)
    
    assert "text" in deserialized.calibration_anchors
    assert deserialized.calibration_anchors["text"].shape == (8, 16 + 20)

    # 5. Re-grow G(D_1) -> triggers auto-calibration
    w_1 = growth_engine.grow_phenotype_model(deserialized)
    assert w_1 is not None


def test_net2net_equivalence():
    # 1. Setup base Genotype and grow original model
    d_0 = Genotype.create_default(genotype_id="net2net_root")
    d_0.dna_architecture.vocab_size = 20
    d_0.dna_architecture.d_model = 16
    d_0.dna_architecture.num_layers = 1
    d_0.dna_architecture.num_experts = 2
    d_0.dna_architecture.d_expert_hidden = 32

    growth_engine = GrowthEngine()
    
    # Initialize the CPPN parameters inside genotype
    _ = growth_engine.instantiate_cppn(d_0)
    w_original = growth_engine.grow_phenotype_weights(d_0)

    # 2. Perform Net2Net expansion on genotype
    slow_clock = SlowClockEncoder(rank_ratio=0.5, encoder_steps=5)
    slow_clock._expand_cppn_genotype(d_0, delta_dim=16)

    assert d_0.dna_instinct.cppn_hidden_dim == 48

    # 3. Grow model from expanded genotype
    w_expanded = growth_engine.grow_phenotype_weights(d_0)

    # 4. Assert mathematical identity before any optimization runs
    for k in w_original.keys():
        assert k in w_expanded
        # Check maximum absolute difference is zero (within numerical precision float32)
        diff = torch.max(torch.abs(w_original[k] - w_expanded[k])).item()
        assert diff < 1e-6, f"Net2Net broke identity on weight key: {k} (diff = {diff})"


