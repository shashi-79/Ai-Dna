# Omni-Modal AI DNA Architecture

Implementation of the **Omni-Modal AI DNA Architecture: Genotypic Instinct Encoding, Phenotypic Transferability, and Multi-Generational AI Evolution** ([`idea.md`](idea.md)).

---

## 🔬 Core Architectural Principles

1. **Tripartite Knowledge Separation**:
   $$\text{AI Capability} = D_{\text{instinct}} + W_{\text{parametric}} + K_{\text{external}}$$
   - **$D_{\text{instinct}}$ (Genotype)**: Compact developmental blueprint & topological routing rules.
   - **$W_{\text{parametric}}$ (Phenotype)**: Dynamic weights learned during Fast Clock gradient descent.
   - **$K_{\text{external}}$ (Retrieval/Archive)**: Large, dynamic factual memory maintained hierarchically.

2. **Bidirectional Evolutionary Lifecycle**:
   $$D_t \xrightarrow{\text{Growth } G} W_t \xrightarrow{\text{Fast Clock}} W_t^* \xrightarrow{\text{Slow Clock } E} D_{t+1}$$

3. **Dynamic Sparse Routing & Straight-Through Estimator**:
   $$z_{b,s,e} = \sum_{k=1}^r A_{b,s,e,k} B_{b,s,e,k}, \qquad M_{\text{gate}} = M_{\text{hard}} + \operatorname{sg}(P_{\text{gate}} - M_{\text{hard}})$$

4. **Multi-Parent Fusion with Innovation Tracking**:
   $$D_c = F(D_1, D_2, \ldots, D_n)$$
   Aligns shared structural nodes via historical Innovation IDs and blends specialized disjoint nodes.

---

## 📦 Codebase Structure

```
ai_dna/
├── dna/               # Genotype definition, InnovationTracker, Serialization
├── growth/            # CPPN coordinate network, Substrate coordinates, GrowthEngine
├── routing/           # Low-Rank Sparse Router, Straight-Through Estimator, Load Balancing
├── memory/            # Chunked Working Memory, Compressed Archive, Latent Retrieval
├── models/            # Multi-Modal Encoders, Dynamic Output Decoders, Phenotype Backbone
├── inference/         # Omni-Modal Intake Engine, Output Decoders, Sparse Executor, Pipeline
├── encoding/          # SVD Instinct Filter, Inverse CPPN Optimizer, EWC Consolidation, Slow Clock
├── evolution/         # Mutation operator, Compatibility checker, Multi-Parent Fusion, Fitness
├── training/          # Fast Clock gradient trainer, Joint loss, Standardized metrics
├── experiments/       # Phases 1 to 6 Validation Experiment suites
└── tests/             # Comprehensive pytest suite
```

---

## 🚀 Quickstart & Usage

### 1. Running the Validation Benchmark Suites

Execute validation experiments using the unified CLI:

```bash
# Run all benchmark experiments (Phases 1 to 6) in quick validation mode
python run_benchmarks.py --quick

# Run specific experiment (e.g., Exp 1: SVD Instinct-Filter Hypothesis)
python run_benchmarks.py --experiment exp1

# Run Transferability Curve evaluation (Exp 2)
python run_benchmarks.py --experiment exp2

# Run Multi-Parent Fusion evaluation (Exp 6)
python run_benchmarks.py --experiment exp6
```

### 2. Python API Example

```python
import torch
from ai_dna import (
    Genotype,
    GrowthEngine,
    InferencePipeline,
    FastClockTrainer,
    SlowClockEncoder,
)

# 1. Instantiate root Genotype D_0
d_0 = Genotype.create_default(genotype_id="D_0")

# 2. Grow Phenotype Neural Network
pipeline = InferencePipeline(genotype=d_0)

# 3. Dynamic Multi-Modal Inference
# Autoregressive generation
prompt = torch.tensor([[1, 24, 52, 10]])
result_ar = pipeline.generate(prompt, modality="text", mode="autoregressive", max_new_tokens=20)
print("Generated Tokens:", result_ar["output"])

# Multi-class classification
res_cls = pipeline.generate(prompt, modality="text", mode="classify")
print("Class Predictions:", res_cls["predictions"])
```

### 3. Running Automated Tests

```bash
python -m pytest tests/ -v
```
