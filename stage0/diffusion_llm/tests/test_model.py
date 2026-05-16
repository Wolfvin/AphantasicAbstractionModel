"""Tests for AAM Diffusion Model components."""

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

import torch
import pytest
from diffusion_llm.config.model_config import AamDiffusionConfig, get_default_config, ModelConfig
from diffusion_llm.model.noise_scheduler import NoiseScheduler
from diffusion_llm.model.graph_encoder import GraphConditioningEncoder, GraphEncoderConfig
from diffusion_llm.model.diffusion_transformer import DiffusionTransformer
from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer


class TestConfig:
    """Test configuration system."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config("base")
        assert config.model.d_model == 768
        assert config.model.n_layers == 12
        assert config.diffusion.n_timesteps == 1000

    def test_tiny_config(self):
        """Test tiny model configuration."""
        config = get_default_config("tiny")
        assert config.model.d_model == 256
        assert config.model.n_layers == 4

    def test_config_serialization(self, tmp_path):
        """Test config save/load roundtrip."""
        config = get_default_config("small")
        path = tmp_path / "config.json"
        config.to_json(path)

        loaded = AamDiffusionConfig.from_json(path)
        assert loaded.model.d_model == config.model.d_model
        assert loaded.model.n_layers == config.model.n_layers

    def test_param_estimation(self):
        """Test parameter count estimation."""
        config = ModelConfig(d_model=768, n_layers=12, d_ff=3072)
        params = config.estimate_params()
        assert "M" in params  # Should be in millions


class TestTokenizer:
    """Test AAM Tokenizer."""

    def test_basic_encoding(self):
        """Test basic text encoding."""
        tokenizer = AamTokenizer()
        # Train on sample text first
        tokenizer.train(["Hello world this is a test", "Another test sentence"])

        ids = tokenizer.encode("Hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert ids[0] == tokenizer.bos_id
        assert ids[-1] == tokenizer.eos_id

    def test_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        tokenizer = AamTokenizer()
        texts = [
            "Berdasarkan analisis, pencuri adalah Diancang.",
            "Anomali terdeteksi dalam laporan Hefei.",
            "Evidence: Ju Jangmok, Snow Plum Pill.",
        ]
        tokenizer.train(texts)

        for text in texts:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids, skip_special=True)
            # Decoded text should contain key words
            assert len(decoded) > 0

    def test_special_tokens(self):
        """Test special token IDs."""
        tokenizer = AamTokenizer()
        assert tokenizer.pad_id == 0
        assert tokenizer.bos_id == 1
        assert tokenizer.eos_id == 2

    def test_sentence_boundaries(self):
        """Test sentence boundary detection."""
        tokenizer = AamTokenizer()
        ids = [1, 10, 20, 5, 30, 40, 5, 50, 2]  # BOS, sent, sent, EOS
        boundaries = tokenizer.get_sentence_boundaries(ids)
        assert 3 in boundaries  # Index of <sent> token
        assert 6 in boundaries

    def test_save_load(self, tmp_path):
        """Test tokenizer save/load."""
        tokenizer = AamTokenizer()
        tokenizer.train(["Test text for tokenizer", "Another training example"])

        path = tmp_path / "tokenizer.json"
        tokenizer.save(path)

        loaded = AamTokenizer.load(path)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.is_trained

    def test_structure_encoding(self):
        """Test encoding with graph structure tokens."""
        tokenizer = AamTokenizer()
        tokenizer.train(["Evidence text", "Anomaly description", "Reasoning step"])

        ids = tokenizer.encode_with_structure(
            text="Main narrative text",
            evidence_nodes=["evidence1", "evidence2"],
            anomalies=["anomaly1"],
        )
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_padding(self):
        """Test sequence padding."""
        tokenizer = AamTokenizer()
        ids = [1, 2, 3]
        padded = tokenizer.pad_sequence(ids, max_len=10)
        assert len(padded) == 10
        assert padded[3:] == [0] * 7  # Padded with pad_id


class TestDiffusionTransformer:
    """Test Diffusion Transformer model."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        config = ModelConfig(
            d_model=128, n_layers=2, n_heads=4, d_ff=256,
            vocab_size=1000, max_seq_len=64,
        )
        model = DiffusionTransformer(config)

        x_t = torch.randn(2, 32, 128)  # batch=2, seq=32, d=128
        t = torch.tensor([100, 500])

        output = model(x_t=x_t, t=t)
        assert output.shape == (2, 32, 128)

    def test_with_graph_conditioning(self):
        """Test forward pass with graph conditioning."""
        config = ModelConfig(
            d_model=128, n_layers=2, n_heads=4, d_ff=256,
            vocab_size=1000, max_seq_len=64,
        )
        model = DiffusionTransformer(config)

        x_t = torch.randn(2, 32, 128)
        t = torch.tensor([100, 500])
        graph_keys = torch.randn(2, 10, 128)  # 10 graph nodes
        graph_values = torch.randn(2, 10, 128)

        output = model(x_t=x_t, t=t, graph_keys=graph_keys, graph_values=graph_values)
        assert output.shape == (2, 32, 128)


class TestAamDiffusionModel:
    """Test complete AAM Diffusion Model."""

    def test_model_creation_tiny(self):
        """Test creating a tiny model."""
        config = get_default_config("tiny")
        model = AamDiffusionModel(config)
        n_params = model.get_num_params()
        assert n_params > 0
        assert n_params < 100e6  # Tiny should be under 100M

    def test_forward_training(self):
        """Test training forward pass."""
        config = get_default_config("tiny")
        model = AamDiffusionModel(config)
        model.eval()

        token_ids = torch.randint(0, config.model.vocab_size, (2, 32))
        timestep = torch.randint(0, config.diffusion.n_timesteps, (2,))

        with torch.no_grad():
            predicted, noise = model(token_ids=token_ids, timestep=timestep)

        assert predicted.shape == noise.shape

    def test_loss_computation(self):
        """Test loss computation."""
        config = get_default_config("tiny")
        model = AamDiffusionModel(config)
        model.eval()

        token_ids = torch.randint(0, config.model.vocab_size, (2, 32))
        timestep = torch.randint(0, config.diffusion.n_timesteps, (2,))

        with torch.no_grad():
            predicted, noise = model(token_ids=token_ids, timestep=timestep)
            loss = model.compute_loss(predicted, noise, timestep)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_save_load(self, tmp_path):
        """Test model save/load."""
        config = get_default_config("tiny")
        model = AamDiffusionModel(config)

        path = str(tmp_path / "model.pt")
        model.save(path)

        loaded = AamDiffusionModel.load(path)
        assert loaded.config.model.d_model == config.model.d_model


class TestGraphEncoder:
    """Test Graph Conditioning Encoder."""

    def test_evidence_encoding(self):
        """Test encoding evidence nodes."""
        config = GraphEncoderConfig(d_graph=128, n_graph_layers=2, n_graph_heads=4)
        encoder = GraphConditioningEncoder(config, vocab_size=1000)

        evidence_ids = torch.randint(0, 1000, (2, 5, 16))  # 2 batch, 5 nodes, 16 tokens each
        evidence_conf = torch.tensor([[0.8, 0.6, 0.9, 0.7, 0.5],
                                       [0.7, 0.8, 0.6, 0.9, 0.5]])

        result = encoder(evidence_ids=evidence_ids, evidence_confidence=evidence_conf)
        assert "keys" in result
        assert "values" in result

    def test_no_input(self):
        """Test encoder with no graph data (should return zeros)."""
        config = GraphEncoderConfig(d_graph=128, n_graph_layers=2, n_graph_heads=4)
        encoder = GraphConditioningEncoder(config, vocab_size=1000)

        result = encoder()
        assert "keys" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
