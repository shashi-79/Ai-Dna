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
