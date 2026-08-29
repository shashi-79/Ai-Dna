"""
Unit tests for the 9 newly implemented Core Enhancements:
1. RFF / SIREN Coordinate Embedding & Manifold Isomorphism
2. CPPN with RFF support
3. Adaptive SVD Rank & GPM (Gradient Projection Memory)
4. MultiHeadLatentAttention QK-Norm
5. SparseMoELayer with DeepSeek-V3 Shared Base Expert
6. DNAInstinct Epigenetic Methylation Masks
7. MAP-Elites Quality-Diversity Archive
8. Process-Supervised Step-Level GRPO & Verifier
"""

import unittest
import torch
import torch.nn as nn
from ai_dna.growth.coordinates import SubstrateCoordinateGenerator
from ai_dna.growth.cppn import CPPNNetwork
from ai_dna.encoding.ewc import adaptive_svd_rank, GPMConsolidator
from ai_dna.models.mla import MultiHeadLatentAttention
from ai_dna.models.phenotype import SparseMoELayer, GenerativeSparseRouter
from ai_dna.dna.structure import DNAInstinct, Genotype
from ai_dna.evolution.fitness import MapElitesArchive
from ai_dna.reasoning.verifier import ReasoningVerifier
from ai_dna.reasoning.grpo import GRPOTrainer
from ai_dna.growth.engine import GrowthEngine


class TestCoreEnhancements(unittest.TestCase):

    def test_rff_and_manifold_isomorphism(self):
        # 1. Test RFF embedding
        coords = torch.randn(4, 8, 32)
        rff = SubstrateCoordinateGenerator.apply_rff_embedding(coords, num_fourier_feats=16, sigma=1.0)
        self.assertEqual(rff.shape, (4, 8, 32))  # 16 cos + 16 sin = 32

        # 2. Test Manifold Isomorphism
        order = SubstrateCoordinateGenerator.compute_manifold_isomorphism_order(64)
        self.assertEqual(order.shape, (64,))
        self.assertTrue(torch.all(order >= -1.0) and torch.all(order <= 1.0))

        # 3. Test CPPN with RFF
        cppn_rff = CPPNNetwork(in_features=32, hidden_dim=32, use_rff=True, rff_features=16)
        out = cppn_rff(coords)
        self.assertEqual(out.shape, (4, 8, 1))

    def test_gpm_and_adaptive_svd(self):
        # 1. Test Adaptive SVD Rank
        u = torch.randn(64, 3)
        v = torch.randn(3, 64)
        w_lowrank = u @ v + 0.001 * torch.randn(64, 64)
        k_opt = adaptive_svd_rank(w_lowrank, energy_threshold=0.95)
        self.assertLessEqual(k_opt, 5)

        # 2. Test GPM Null-Space Projection
        gpm = GPMConsolidator(energy_threshold=0.95)
        acts = torch.randn(32, 64)
        gpm.update_activation_basis("layer_1", acts)
        delta_w = torch.randn(64, 64)
        delta_safe = gpm.project_gradient_or_delta("layer_1", delta_w)
        self.assertEqual(delta_safe.shape, (64, 64))

    def test_qk_norm_and_shared_moe(self):
        # 1. Test MLA with QK-Norm
        mla = MultiHeadLatentAttention(d_model=64, num_heads=4, d_kv_latent=16)
        self.assertTrue(hasattr(mla, "q_norm"))
        self.assertTrue(hasattr(mla, "k_norm"))
        x = torch.randn(2, 8, 64)
        out = mla(x)
        self.assertEqual(out.shape, (2, 8, 64))

        # 2. Test SparseMoELayer with Shared Base Expert
        from ai_dna.dna.structure import DNARouting
        dna_routing = DNARouting(top_k_experts=2)
        router = GenerativeSparseRouter(d_model=64, num_experts=4, dna_routing=dna_routing)
        moe = SparseMoELayer(d_model=64, num_experts=4, d_expert_hidden=128, router=router, use_shared_expert=True)
        self.assertIsNotNone(moe.shared_expert)
        h_out, aux_loss = moe(x)
        self.assertEqual(h_out.shape, (2, 8, 64))

    def test_epigenetics_and_map_elites(self):
        # 1. Test Epigenetic Mask dataclass
        mask = {"cppn.backbone.0.weight": torch.ones(32, 32)}
        instinct = DNAInstinct(epigenetic_mask=mask)
        self.assertIn("cppn.backbone.0.weight", instinct.epigenetic_mask)

        # 2. Test MAP-Elites 2D QD Archive
        archive = MapElitesArchive(dim_x_bins=5, dim_y_bins=5, x_range=(0.0, 1.0), y_range=(1.0, 10.0))
        g1 = Genotype(genotype_id="elite_1")
        g2 = Genotype(genotype_id="elite_2")

        added1 = archive.add_or_replace(g1, fitness=0.85, behavior_x=0.7, behavior_y=4.5)
        added2 = archive.add_or_replace(g2, fitness=0.92, behavior_x=0.2, behavior_y=8.0)
        self.assertTrue(added1)
        self.assertTrue(added2)
        self.assertEqual(len(archive.get_elites()), 2)
        self.assertGreater(archive.coverage(), 0.0)

    def test_step_level_grpo(self):
        # 1. Test Step-Level Reasoning Verifier
        verifier = ReasoningVerifier()
        sample_cot = "<thought> Step 1: calculate 5 + 3 = 8 </thought> <thought> Step 2: 8 * 2 = 16 </thought> 16"
        step_rewards = verifier.compute_step_level_rewards(sample_cot, ground_truth_answer="16")
        self.assertEqual(len(step_rewards), 2)
        self.assertGreater(step_rewards[-1], 0.5)

        # 2. Test GRPOTrainer Step Advantages
        growth_engine = GrowthEngine()
        genotype = Genotype()
        model = growth_engine.grow_phenotype_model(genotype)
        trainer = GRPOTrainer(model=model, verifier=verifier, group_size=2)
        candidate_tokens = torch.randint(1, 100, (4, 12))
        adv, metrics = trainer.compute_step_level_advantages(candidate_tokens, prompt_len=4, ground_truth_answers=["16", "16"])
        self.assertEqual(adv.shape, (4,))
        self.assertIn("mean_step_reward", metrics)


if __name__ == "__main__":
    unittest.main()
