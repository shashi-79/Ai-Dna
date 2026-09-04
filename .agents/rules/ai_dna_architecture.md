# AI-DNA Architecture (CL-DNA & Pure AI-DNA Engine Invariants)

When working on the AI-DNA project, STRICTLY adhere to the Cumulative Layered DNA (CL-DNA) architecture and pure AI-DNA dependent engine invariants.

## Core Architectural Invariants:
1. **STRICTLY NO ORGAN ARCHITECTURE (PURE AI-DNA DEPENDENT ARCHITECTURE ONLY)**:
   - **STRICTLY PROHIBITED**: Never use, write, import, suggest, or introduce "organ architecture" or non-AIDNA modular shells (such as `CoreSmolLM2Organ`, `CoreCLIPOrgan`, `CoreWhisperOrgan`, `CoreDiffusionOrgan`, `CoreAudioGenOrgan`, `CoreVideoGenOrgan`).
   - All perception, reasoning, and generative modalities MUST be strictly driven by core AI-DNA framework components:
     - `Genotype` (`.aidna` container)
     - `GrowthEngine` ($W = G(D)$ Phenotype Regrowth)
     - `PhenotypeNeuralNetwork` (MLA Attention + Top-K MoE Backbone)
     - `SlowClockEncoder` (Slow Clock Genotypic Encoding & EWC Retention)
     - `AIDNAFastClock` (Fast Clock Sensory Activation Dynamics)
     - `ReasoningVerifier` (CoT `<thought>` Trace & PRM Verification Rewards)

2. **NO SVD MATRIX FACTORIZATION FOR MODEL WEIGHT STORAGE**:
   - NEVER use SVD factor matrix decomposition when serializing model weights into `.aidna` genotype files.
   - SVD matrix truncation degrades model weights; store exact raw parameter tensors (`raw.<key>`) to preserve 100% reconstruction fidelity without quality loss.

3. **STRICT HARDWARE CUDA GPU ENFORCEMENT**:
   - All engines, converters, growth pipelines, and benchmark tools MUST strictly run on GPU CUDA (`cuda:0`).

4. **ZERO STUBS OR SYNTHETIC DUMMY FALLBACKS**:
   - NEVER mask errors or substitute model execution with synthetic harmonic equations, dummy stubs, or fake fallbacks.
   - Always execute true neural trajectories through the `PhenotypeNeuralNetwork`.

5. **LoRA + CPPN Hybrid**:
   - Phenotype models learn new tasks via Low-Rank Adaptation (LoRA).
   - Only the extracted LoRA adapter weights are encoded back into the genotype via the Inverse CPPN Encoder, NEVER the entire dense model weights.

6. **Cumulative Growth Lifecycle**:
   - The Genotype accumulates structural knowledge via layers.
   - A subsequent generation (e.g. Gen 1) DNA consists of the Base Gen 0 CPPN parameters merged with the newly encoded Gen 0 LoRA adapter CPPN parameters.
   - Base model checkpoints are never saved/loaded directly for generational transitions. The base model is always physically "grown" from the Base Gen 0 DNA, and the adapters are grown from the Adapter DNA layers.

## Engineering & Implementation Invariants:
7. **Explicit Attention Causality**:
   - Attention mechanisms (MLA, FlashAttention, SDPA) must ALWAYS receive and respect an explicit `is_causal` flag.
   - NEVER use `is_causal = (mask is None)` heuristics.

8. **Strict Genotype Round-Trip Serialization**:
   - All constitutional hyperparameters in `Genotype` (`DNAArchitecture.coord_dim` defaulting to 32, `kv_latent_dim`, `rope_theta`, `lora_rank`, `DNARouting.top_k_experts`, `DNARouting.routing_noise_std`, `DNAMemory.kv_quant_bits`, `page_size`, `max_pages`) must be explicitly serialized in `genotype_to_dict` and reconstructed in `dict_to_genotype`.

9. **Empirical Reality Check Protocol**:
   - Validate all generational evolutions against the 4-way baseline comparison ($W_R$, $W^*$, LoRA Transfer, Random LoRA) and track Sample Efficiency ($S_E$), True Compression Ratio ($C_R$), and Behavioral KL Divergence ($D_{KL}$).
