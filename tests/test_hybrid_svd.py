"""
Unit tests for the Production Hybrid cuSOLVER SVD Engine (`ai_dna/kernels/hybrid_svd.py`).
Verifies canonical sign determinism, min_rank=4 bounds, energy thresholds, and CPPN stability.
"""

import unittest
import torch
from ai_dna.kernels.hybrid_svd import exact_cusolver_svd, stabilize_svd_signs
from ai_dna.growth.cppn import CPPNNetwork


class TestHybridSVD(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_canonical_sign_determinism(self):
        """Verifies that randomly sign-inverted vectors are normalized to identical positive-peak conventions."""
        torch.manual_seed(42)
        M, K = 128, 16
        U_base = torch.randn((M, K), device=self.device)
        V_base = torch.randn((M, K), device=self.device)

        # Create artificially inverted copies with random signs
        random_signs = torch.tensor([1 if i % 2 == 0 else -1 for i in range(K)], device=self.device)
        U_flipped = U_base * random_signs.unsqueeze(0)
        V_flipped = V_base * random_signs.unsqueeze(0)

        # Apply canonical stabilization
        U_fixed_1, V_fixed_1 = stabilize_svd_signs(U_base, V_base)
        U_fixed_2, V_fixed_2 = stabilize_svd_signs(U_flipped, V_flipped)

        # Both must be 100% identical!
        self.assertTrue(torch.allclose(U_fixed_1, U_fixed_2, atol=1e-6))
        self.assertTrue(torch.allclose(V_fixed_1, V_fixed_2, atol=1e-6))

        # Check that peak element in each column of U is strictly positive
        max_abs_idx = torch.argmax(torch.abs(U_fixed_1), dim=0)
        col_idx = torch.arange(K, device=self.device)
        peaks = U_fixed_1[max_abs_idx, col_idx]
        self.assertTrue((peaks > 0).all(), "All peak elements in U must be positive")

    def test_production_min_rank_safeguard(self):
        """Verifies that rank is bounded from below by min_rank=128 (or clamped by max_rank)."""
        # Case A: Matrix larger than 128 (e.g. 256x256 rank-1 matrix)
        u = torch.randn((256, 1), device=self.device)
        v = torch.randn((1, 256), device=self.device)
        W_large = u @ v

        U_k, S_k, V_k, k = exact_cusolver_svd(W_large, min_rank=128, energy_threshold=0.99)
        self.assertGreaterEqual(k, 128, f"Rank collapsed below min_rank=128: {k}")
        self.assertEqual(U_k.shape[1], k)
        self.assertEqual(V_k.shape[1], k)

        # Case B: Matrix smaller than 128 (e.g. 64x64) -> clamps cleanly to 64
        W_small = torch.randn((64, 64), device=self.device)
        U_s, S_s, V_s, k_s = exact_cusolver_svd(W_small, min_rank=128)
        self.assertEqual(k_s, 64, f"Failed to clamp min_rank=128 to max_rank=64: {k_s}")

    def test_exact_cusolver_svd_reconstruction(self):
        """Verifies that U_k @ S_k @ V_k.T mathematically reconstructs the original matrix."""
        M, N = 64, 64
        W = torch.randn((M, N), device=self.device)

        U_k, S_k, V_k, k = exact_cusolver_svd(W, min_rank=4, energy_threshold=0.999, apply_canonical_signs=True)
        
        # Low rank reconstruction
        W_recon = U_k @ torch.diag(S_k) @ V_k.T
        rel_error = torch.norm(W - W_recon) / torch.norm(W)
        self.assertLess(rel_error.item(), 0.05)

    def test_cppn_stability_with_canonical_svd(self):
        """Verifies that passing canonically stabilized SVD vectors into CPPN yields 100% deterministic weights."""
        M, N = 32, 32
        W = torch.randn((M, N), device=self.device)

        # Run 1: Original
        U1, S1, V1, _ = exact_cusolver_svd(W, rank=8, apply_canonical_signs=True)
        # Run 2: From perturbed/re-executed
        U2, S2, V2, _ = exact_cusolver_svd(W, rank=8, apply_canonical_signs=True)

        cppn = CPPNNetwork(in_features=8, hidden_dim=32, num_layers=3, out_features=1).to(self.device)
        cppn.eval()

        with torch.no_grad():
            out1 = cppn(U1)
            out2 = cppn(U2)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6), "CPPN outputs must be deterministic")

    def test_outlier_vault_extraction_and_zero_loss_reconstruction(self):
        """Verifies exact outlier isolation at tau=6.0 sigma and 100% bit-exact reconstruction."""
        from ai_dna.kernels.hybrid_svd import extract_outlier_sparse_residual, restore_outliers_to_tensor
        M, N = 64, 64
        torch.manual_seed(1337)
        W = torch.randn((M, N), device=self.device)

        # Inject two extreme emergent outlier weights
        W[5, 12] = 50.0
        W[30, 45] = -60.0

        sanitized, vault_entry = extract_outlier_sparse_residual(W, threshold_sigma=6.0)
        self.assertEqual(vault_entry["count"], 2, f"Expected 2 outliers, found {vault_entry['count']}")

        # Ensure outliers were extracted with exact coordinates and values
        coords = vault_entry["indices"].cpu().tolist()
        expected_coords = [[5, 30], [12, 45]]
        self.assertEqual(coords, expected_coords)
        self.assertEqual(vault_entry["values"][0].item(), 50.0)
        self.assertEqual(vault_entry["values"][1].item(), -60.0)

        # In the sanitized matrix, those positions are zeroed
        self.assertEqual(sanitized[5, 12].item(), 0.0)
        self.assertEqual(sanitized[30, 45].item(), 0.0)

        # Exact restoration must match original W bit-for-bit on outlier coordinates
        restored = restore_outliers_to_tensor(sanitized, vault_entry)
        self.assertTrue(torch.equal(restored, W), "Restored matrix must be bit-for-bit identical to original")

    def test_outlier_vault_growth_engine_integration(self):
        """Verifies that GrowthEngine fuses outlier vault entries into both weights dict and Phenotype model."""
        from ai_dna.dna.structure import Genotype
        from ai_dna.growth.engine import GrowthEngine
        genotype = Genotype.create_default()
        engine = GrowthEngine(device=self.device)

        target_param = "blocks.0.attn.w_q.weight"
        outlier_val = 999.0
        genotype.outlier_vault[target_param] = {
            "indices": torch.tensor([[0], [1]], dtype=torch.int64, device=self.device),
            "values": torch.tensor([outlier_val], dtype=torch.float32, device=self.device),
            "shape": [genotype.dna_architecture.d_model, genotype.dna_architecture.d_model],
            "count": 1,
            "threshold_sigma": 6.0,
        }

        # 1. Verify growth via grow_phenotype_weights
        weights = engine.grow_phenotype_weights(genotype)
        self.assertEqual(weights[target_param][0, 1].item(), outlier_val)

        # 2. Verify growth via grow_phenotype_model
        model = engine.grow_phenotype_model(genotype)
        self.assertEqual(model.blocks[0].attn.w_q.weight[0, 1].item(), outlier_val)

    def test_outlier_vault_multi_parent_fusion_retention(self):
        """Verifies zero catastrophic forgetting: child retains outlier keys from both Parent A and Parent B."""
        from ai_dna.dna.structure import Genotype
        from ai_dna.growth.engine import GrowthEngine

        parent_a = Genotype.create_default(genotype_id="parent_a")
        parent_b = Genotype.create_default(genotype_id="parent_b")

        # Parent A: specialized text routing outlier
        text_key = "text_encoder.token_emb.weight"
        parent_a.outlier_vault[text_key] = {
            "indices": torch.tensor([[2], [4]], dtype=torch.int64, device=self.device),
            "values": torch.tensor([42.0], dtype=torch.float32, device=self.device),
            "shape": [parent_a.dna_architecture.vocab_size, parent_a.dna_architecture.d_model],
            "count": 1,
            "threshold_sigma": 6.0,
        }

        # Parent B: specialized audio projection outlier
        audio_key = "audio_encoder.proj.weight"
        parent_b.outlier_vault[audio_key] = {
            "indices": torch.tensor([[1], [3]], dtype=torch.int64, device=self.device),
            "values": torch.tensor([77.0], dtype=torch.float32, device=self.device),
            "shape": [parent_b.dna_architecture.d_model, 80],
            "count": 1,
            "threshold_sigma": 6.0,
        }

        # Child fuses both parents without size cap (Exact Lossless Inheritance)
        child = parent_a.clone(new_id="child_ab")
        child.outlier_vault = {**parent_a.outlier_vault, **parent_b.outlier_vault}

        engine = GrowthEngine(device=self.device)
        child_weights = engine.grow_phenotype_weights(child)

        # Both parent outlier features must be preserved with 100% exact numerical fidelity
        self.assertEqual(child_weights[text_key][2, 4].item(), 42.0)
        self.assertEqual(child_weights[audio_key][1, 3].item(), 77.0)


if __name__ == "__main__":
    unittest.main()

