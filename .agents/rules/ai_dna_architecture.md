# AI-DNA Architecture (CL-DNA Paradigm)

When working on the AI-DNA project, STRICTLY adhere to the Cumulative Layered DNA (CL-DNA) architecture. 

## Architectural Constraints:
1. **NO Hypernetworks or SVD**: SVD filters and Hypernetworks are deprecated. Do not use them or suggest them for encoding structural information.
2. **LoRA + CPPN Hybrid**: 
   - Phenotype models learn new tasks exclusively via Low-Rank Adaptation (LoRA).
   - Only the extracted LoRA adapter weights are encoded back into the genotype via the Inverse CPPN Encoder, NEVER the entire dense model weights.
3. **Cumulative Growth Lifecycle**:
   - The Genotype accumulates structural knowledge via layers. 
   - A subsequent generation (e.g. Gen 1) DNA consists of the Base Gen 0 CPPN parameters merged with the newly encoded Gen 0 LoRA adapter CPPN parameters.
   - Base model checkpoints are never saved/loaded directly for generational transitions. The base model is always physically "grown" from the Base Gen 0 DNA, and the adapters are grown from the Adapter DNA layers.
4. **Capacity Management**: Rely on Elastic Weight Consolidation (EWC) and optimization within the CPPN rather than endlessly expanding the CPPN parameter capacity (`hidden_dim`) across generations.

## Engineering & Implementation Invariants:
5. **Explicit Attention Causality**:
   - Attention mechanisms (MLA, FlashAttention, SDPA) must ALWAYS receive and respect an explicit `is_causal` flag.
   - NEVER use `is_causal = (mask is None)` heuristics, as bidirectional vision patches, audio spectrograms, and classification sequences do not supply an explicit attention mask but MUST NOT be causally masked.
6. **Strict Genotype Round-Trip Serialization**:
   - All constitutional hyperparameters in `Genotype` (`DNAArchitecture.coord_dim` defaulting to 32, `kv_latent_dim`, `rope_theta`, `lora_rank`, `DNARouting.top_k_experts`, `DNARouting.routing_noise_std`, `DNAMemory.kv_quant_bits`, `page_size`, `max_pages`) must be explicitly serialized in `genotype_to_dict` and reconstructed in `dict_to_genotype`.
7. **Offline-Safe Dataset Ingestion**:
   - Data pipelines must support instant offline fallback (e.g., via `AI_DNA_OFFLINE=1` or instant network checks) to prevent test runners and benchmarks from blocking on remote HTTP retries.
8. **Empirical Reality Check Protocol**:
   - Validate all generational evolutions against the 4-way baseline comparison ($W_R$, $W^*$, LoRA Transfer, Random LoRA) and track Sample Efficiency ($S_E$), True Compression Ratio ($C_R$), and Behavioral KL Divergence ($D_{KL}$).
