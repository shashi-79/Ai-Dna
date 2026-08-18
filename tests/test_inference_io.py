"""
Tests for Inference Input/Output Engine and Dynamic Execution Pipeline.
"""

import torch
from ai_dna.dna.structure import Genotype
from ai_dna.models.phenotype import PhenotypeNeuralNetwork
from ai_dna.inference.pipeline import InferencePipeline
from ai_dna.inference.sparse_executor import SparseHardwareExecutor


def test_inference_pipeline_autoregressive():
    genotype = Genotype.create_default(genotype_id="inf_test")
    genotype.dna_architecture.vocab_size = 50
    genotype.dna_architecture.d_model = 32
    genotype.dna_architecture.num_layers = 2
    genotype.dna_architecture.num_experts = 2

    pipeline = InferencePipeline(genotype=genotype)
    prompt = torch.tensor([[1, 5, 10, 15]])
    res = pipeline.generate(prompt, modality="text", mode="autoregressive", max_new_tokens=5)

    assert res["mode"] == "autoregressive"
    assert res["output"].shape == (1, 9)  # 4 prompt tokens + 5 new tokens


def test_inference_pipeline_classification():
    genotype = Genotype.create_default(genotype_id="inf_cls_test")
    genotype.dna_architecture.vocab_size = 50
    genotype.dna_architecture.d_model = 32
    genotype.dna_architecture.num_layers = 2

    pipeline = InferencePipeline(genotype=genotype)
    inputs = torch.randint(0, 50, (2, 8))
    res = pipeline.generate(inputs, modality="text", mode="classify")

    assert res["mode"] == "classify"
    assert res["logits"].shape == (2, 10)
    assert res["predictions"].shape == (2,)


def test_sparse_hardware_executor():
    executor = SparseHardwareExecutor(num_experts=2)
    x = torch.randn(2, 4, 16)
    gate_probs = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
                               [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]])
    hard_mask = (gate_probs > 0.5).float()

    expert_modules = torch.nn.ModuleList([
        torch.nn.Linear(16, 16),
        torch.nn.Linear(16, 16)
    ])

    out = executor.execute_sparse_moe(x, gate_probs, hard_mask, expert_modules)
    assert out.shape == (2, 4, 16)


def test_triton_sparse_moe_fallback():
    from ai_dna.inference.triton_kernels import TritonSparseMoEExecutor, is_triton_available
    a = torch.randn(8, 16)
    b = torch.randn(16, 32)
    # Should work seamlessly with triton (on GPU) or fallback to PyTorch
    c = TritonSparseMoEExecutor.triton_gemm(a, b)
    assert c.shape == (8, 32)
    assert isinstance(is_triton_available(), bool)


def test_bpe_train_from_blank():
    from ai_dna.encoding.tokenizers import TextBPETokenizer
    tokenizer = TextBPETokenizer(vocab_size=256 + 4)
    # Starts with base 260 tokens (4 special + 256 bytes)
    assert tokenizer.vocab_size == 260
    
    corpus = ["hello how are you", "hello system learns fast"]
    tokenizer.train(corpus, target_vocab_size=270)
    
    assert tokenizer.vocab_size == 270
    assert len(tokenizer.merges) == 10


def test_bpe_encode_decode_roundtrip():
    from ai_dna.encoding.tokenizers import TextBPETokenizer
    tokenizer = TextBPETokenizer(vocab_size=260)
    corpus = ["ai dna grows neural phenotype dynamically from genotype guide"]
    tokenizer.train(corpus, target_vocab_size=280)
    
    text = "phenotype guided by dna genotype"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == text


def test_bpe_evolve_across_generations():
    from ai_dna.encoding.tokenizers import TextBPETokenizer
    tokenizer = TextBPETokenizer(vocab_size=260)
    tokenizer.train(["generation zero starts blank"], target_vocab_size=270)
    assert tokenizer.vocab_size == 270
    
    # Evolve to next generation with new dataset
    added = tokenizer.evolve(["generation one evolves from previous merges"], target_vocab_size=280)
    assert tokenizer.vocab_size == 280
    assert added == 10
    
    # Verify old token IDs are preserved and decodable
    encoded_old = tokenizer.encode("generation zero")
    decoded_old = tokenizer.decode(encoded_old, skip_special_tokens=True)
    assert decoded_old == "generation zero"

