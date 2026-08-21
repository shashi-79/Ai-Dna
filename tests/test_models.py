"""
Tests for Models, Encoders, RoPE, MLA, and Phenotype Neural Network.
"""

import torch
from ai_dna.models.rope import RoPE, RoPE2D, RoPE3D
from ai_dna.models.mla import MultiHeadLatentAttention
from ai_dna.models.modules import (
    TextEncoder,
    VisionEncoder,
    AudioEncoder,
    VideoEncoder,
    ContrastiveAlignmentHead,
)
from ai_dna.models.phenotype import PhenotypeNeuralNetwork
from ai_dna.dna.structure import Genotype


def test_rope_1d_2d_3d():
    # 1D RoPE
    rope1d = RoPE(dim=16)
    q = torch.randn(2, 4, 32, 16)
    k = torch.randn(2, 4, 32, 16)
    q_rot, k_rot = rope1d(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

    # 2D RoPE
    rope2d = RoPE2D(dim=16)
    q_vis = torch.randn(2, 4, 16, 16) # 4x4 patches = 16
    k_vis = torch.randn(2, 4, 16, 16)
    q_rot2, k_rot2 = rope2d(q_vis, k_vis, h=4, w=4)
    assert q_rot2.shape == q_vis.shape

    # 3D RoPE
    rope3d = RoPE3D(dim=18)
    q_vid = torch.randn(2, 4, 32, 18) # 2x4x4 tubes = 32
    k_vid = torch.randn(2, 4, 32, 18)
    q_rot3, k_rot3 = rope3d(q_vid, k_vid, t=2, h=4, w=4)
    assert q_rot3.shape == q_vid.shape


def test_multi_head_latent_attention():
    mla = MultiHeadLatentAttention(d_model=64, num_heads=4, d_kv_latent=16)
    x = torch.randn(2, 20, 64)
    out = mla(x)
    assert out.shape == (2, 20, 64)


def test_contrastive_intake_encoders():
    # Text
    text_enc = TextEncoder(vocab_size=100, d_model=64)
    tokens = torch.randint(0, 100, (2, 10))
    h_text = text_enc(tokens)
    assert h_text.shape == (2, 10, 64)

    # Vision (CLIP-style patch projection)
    vis_enc = VisionEncoder(in_channels=3, d_model=64, patch_size=4)
    img = torch.randn(2, 3, 16, 16) # 4x4 = 16 patches + 1 [CLS] = 17
    h_vis = vis_enc(img)
    assert h_vis.shape == (2, 17, 64)

    # Audio
    aud_enc = AudioEncoder(in_dim=80, d_model=64)
    audio = torch.randn(2, 25, 80)
    h_aud = aud_enc(audio)
    assert h_aud.shape == (2, 25, 64)

    # Video
    vid_enc = VideoEncoder(in_channels=3, d_model=64, temporal_patch_size=2, spatial_patch_size=4)
    video = torch.randn(2, 3, 4, 16, 16) # (4//2) * (16//4) * (16//4) = 2 * 4 * 4 = 32 tubes + 1 [CLS] = 33
    h_vid = vid_enc(video)
    assert h_vid.shape == (2, 33, 64)

    # Contrastive Alignment Head
    con_head = ContrastiveAlignmentHead(d_model=64, embed_dim=32)
    z_text = con_head(h_text)
    z_vis = con_head(h_vis)
    assert z_text.shape == (2, 32)
    assert z_vis.shape == (2, 32)
    loss = con_head.compute_loss(z_text, z_vis)
    assert loss.item() >= 0.0


def test_phenotype_neural_network_forward_all_modalities():
    genotype = Genotype.create_default()
    phenotype = PhenotypeNeuralNetwork(genotype)

    # Forward Text
    tokens = torch.randint(0, 100, (2, 16))
    h_text, aux_loss, archive, metrics = phenotype(tokens, modality="text")
    assert h_text.shape == (2, 16, genotype.dna_architecture.d_model)
    assert aux_loss.item() >= 0.0

    # Forward Vision
    img = torch.randn(2, 3, 16, 16)
    h_vis, _, _, _ = phenotype(img, modality="vision")
    assert h_vis.shape[0] == 2
    assert h_vis.shape[2] == genotype.dna_architecture.d_model

    # Forward Audio
    aud = torch.randn(2, 20, 80)
    h_aud, _, _, _ = phenotype(aud, modality="audio")
    assert h_aud.shape == (2, 20, genotype.dna_architecture.d_model)

    # Forward Video
    vid = torch.randn(2, 3, 4, 16, 16)
    h_vid, _, _, _ = phenotype(vid, modality="video")
    assert h_vid.shape[0] == 2
    assert h_vid.shape[2] == genotype.dna_architecture.d_model

    # Forward Unified Multimodal Token Stream (Text + Vision + Audio)
    h_uni, aux_uni, _, _ = phenotype.forward_multimodal(
        text_inputs=tokens,
        vision_inputs=img,
        audio_inputs=aud,
    )
    # Expected sequence length = 16 (text) + 17 (vision) + 20 (audio) = 53
    assert h_uni.shape == (2, 53, genotype.dna_architecture.d_model)
    assert aux_uni.item() >= 0.0
