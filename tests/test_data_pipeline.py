"""
Unit Tests for Omni-Modal Hugging Face Dataset Pipeline (data.py).
Tests all modalities separately: Text, Vision, Audio, Video, Code, Bio, Tabular, and Multi-Modal.
"""

import os
import torch
from data import (
    DataType,
    CustomTextTokenizer,
    CustomBioTokenizer,
    CustomAudioProcessor,
    CustomVisionProcessor,
    CustomVideoProcessor,
    get_text_dataset,
    get_vision_dataset,
    get_audio_dataset,
    get_video_dataset,
    get_code_dataset,
    get_bio_dataset,
    get_tabular_dataset,
    get_multimodal_dataset,
    get_dataset,
    get_dataloader,
)


def test_text_tokenizer():
    tokenizer = CustomTextTokenizer(vocab_size=128, mode="word")
    encoded = tokenizer.encode("ai dna learns and evolves dynamically")
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dim() == 1
    assert len(encoded) > 0

    decoded = tokenizer.decode(encoded)
    assert isinstance(decoded, str)
    assert "ai" in decoded


def test_bio_tokenizer():
    dna_tok = CustomBioTokenizer(bio_type="dna")
    seq = "ATGCGTAA"
    encoded = dna_tok.encode(seq)
    assert len(encoded) == len(seq)
    decoded = dna_tok.decode(encoded)
    assert decoded == seq


def test_audio_processor():
    proc = CustomAudioProcessor(sample_rate=16000, n_mels=80)
    waveform = torch.randn(16000)
    spec = proc.waveform_to_spectrogram(waveform, target_seq_len=16)
    assert spec.shape == (16, 80)


def test_vision_processor():
    proc = CustomVisionProcessor(img_size=(16, 16), in_channels=3)
    dummy_img = torch.rand(3, 32, 32)
    processed = proc.process_image(dummy_img)
    assert processed.shape == (3, 16, 16)


def test_video_processor():
    proc = CustomVideoProcessor(num_frames=8, img_size=(32, 32), in_channels=3)
    frames = [torch.rand(3, 32, 32) for _ in range(12)]
    video_tensor = proc.process_frames(frames)
    assert video_tensor.shape == (3, 8, 32, 32)


def test_text_dataset_loading():
    ds = get_text_dataset(dataset_name="wikitext", seq_len=16, max_samples=20)
    assert len(ds) >= 10
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)


def test_vision_dataset_loading():
    ds = get_vision_dataset(dataset_name="cifar10", img_size=(16, 16), max_samples=20)
    assert len(ds) >= 10
    img, label = ds[0]
    assert img.shape == (3, 16, 16)
    assert isinstance(label.item(), int)


def test_audio_dataset_loading():
    ds = get_audio_dataset(dataset_name="speech_commands", seq_len=16, n_mels=80, max_samples=20)
    assert len(ds) >= 10
    spec, label = ds[0]
    assert spec.shape == (16, 80)
    assert isinstance(label.item(), int)


def test_video_dataset_loading():
    ds = get_video_dataset(dataset_name="pierreroucoux/moving-mnist", num_frames=8, img_size=(32, 32), max_samples=10)
    assert len(ds) >= 5
    video, label = ds[0]
    assert video.shape == (3, 8, 32, 32)


def test_code_dataset_loading():
    ds = get_code_dataset(dataset_name="bigcode/the-stack-smol", seq_len=32, max_samples=20)
    assert len(ds) >= 10
    x, y = ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)


def test_bio_dataset_loading():
    ds = get_bio_dataset(dataset_name="dnapromoter", seq_len=32, bio_type="dna", max_samples=20)
    assert len(ds) >= 10
    x, y = ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)


def test_tabular_dataset_loading():
    ds = get_tabular_dataset(dataset_name="california_housing", num_features=16, max_samples=20)
    assert len(ds) >= 10
    feats, target = ds[0]
    assert feats.shape == (16,)


def test_multimodal_dataset_loading():
    ds = get_multimodal_dataset(dataset_name="nlphuji/flickr30k", text_seq_len=16, img_size=(16, 16), max_samples=10)
    assert len(ds) >= 5
    sample = ds[0]
    assert "text" in sample
    assert "vision" in sample
    assert "audio" in sample
    assert sample["text"].shape == (16,)
    assert sample["vision"].shape == (3, 16, 16)
    assert sample["audio"].shape == (16, 80)


def test_unified_dataloader_routing():
    loader = get_dataloader(data_type="text", batch_size=8, max_samples=16)
    for batch in loader:
        x, y = batch
        assert x.shape[0] == 8 or x.shape[0] == 16
        assert x.shape[1] == 32
        break

    loader_vis = get_dataloader(data_type="vision", batch_size=4, max_samples=8)
    for batch in loader_vis:
        img, lbl = batch
        assert img.shape == (4, 3, 16, 16)
        break
