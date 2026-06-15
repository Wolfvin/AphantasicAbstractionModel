"""Tests for AGNN Adapter — portability layer for model auto-detection.

These tests verify that SelfAdapter correctly:
  1. Detects model architecture from config (Qwen, Llama, Gemma, etc.)
  2. Builds accurate ModelProfile from model.config
  3. Raises UnsupportedModelError for unrecognized/incomplete configs
  4. Computes recommended_hook_layer as num_layers // 2
  5. Projects vectors between arbitrary dimensions
  6. Adapts AGNNGraph embedding_dim to match model's hidden_size
  7. Works entirely with mock configs (no real model required)
"""

import sys
import os

# Ensure src/ is on sys.path (same pattern as other AGNN tests)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import numpy as np

from agnn.adapter import ModelProfile, SelfAdapter, UnsupportedModelError
from agnn.graph import AGNNGraph


# ──────────────────────────────────────────────────────
#  Mock Config Objects
# ──────────────────────────────────────────────────────

class MockConfig:
    """Minimal mock of a HuggingFace PretrainedConfig.

    Only includes the fields that SelfAdapter reads:
      - hidden_size
      - num_hidden_layers
      - model_type
      - _name_or_path
    """
    def __init__(self, hidden_size, num_hidden_layers, model_type=None,
                 name_or_path=None, **kwargs):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.model_type = model_type
        self._name_or_path = name_or_path
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockModel:
    """Minimal mock of a HuggingFace model.

    Provides .config and a class name that can be used for
    architecture detection fallback.
    """
    def __init__(self, config, class_name=None):
        self.config = config
        # Override class name for detection testing
        if class_name:
            self.__class__.__name__ = class_name


# ──────────────────────────────────────────────────────
#  Fixtures — Pre-built Mock Configs
# ──────────────────────────────────────────────────────

@pytest.fixture
def qwen3_config():
    """Mock config for Qwen3-0.6B.

    Real values: hidden_size=1024, num_hidden_layers=28, model_type="qwen2"
    """
    return MockConfig(
        hidden_size=1024,
        num_hidden_layers=28,
        model_type="qwen2",
        name_or_path="Qwen/Qwen3-0.6B",
    )


@pytest.fixture
def llama_config():
    """Mock config for Llama3.2-1B.

    Real values: hidden_size=2048, num_hidden_layers=32, model_type="llama"
    """
    return MockConfig(
        hidden_size=2048,
        num_hidden_layers=32,
        model_type="llama",
        name_or_path="meta-llama/Llama-3.2-1B",
    )


@pytest.fixture
def gemma_config():
    """Mock config for Gemma2-2B.

    Real values: hidden_size=2048, num_hidden_layers=18, model_type="gemma2"
    """
    return MockConfig(
        hidden_size=2048,
        num_hidden_layers=18,
        model_type="gemma2",
        name_or_path="google/gemma-2-2b",
    )


@pytest.fixture
def mistral_config():
    """Mock config for Mistral-7B.

    Real values: hidden_size=4096, num_hidden_layers=32, model_type="mistral"
    """
    return MockConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        model_type="mistral",
        name_or_path="mistralai/Mistral-7B-v0.1",
    )


@pytest.fixture
def unknown_config():
    """Config with no model_type — should fall back to "unknown" architecture."""
    return MockConfig(
        hidden_size=512,
        num_hidden_layers=12,
        model_type=None,
        name_or_path="custom/tiny-model",
    )


# ──────────────────────────────────────────────────────
#  Tests — ModelProfile
# ──────────────────────────────────────────────────────

class TestModelProfile:
    """Tests for ModelProfile dataclass validation."""

    def test_valid_profile(self):
        """A well-formed profile should be created without errors."""
        profile = ModelProfile(
            model_id="test-model",
            hidden_size=1024,
            num_layers=28,
            recommended_hook_layer=14,
            architecture="qwen",
        )
        assert profile.model_id == "test-model"
        assert profile.hidden_size == 1024
        assert profile.num_layers == 28
        assert profile.recommended_hook_layer == 14
        assert profile.architecture == "qwen"

    def test_frozen_profile(self):
        """ModelProfile should be immutable (frozen=True)."""
        profile = ModelProfile(
            model_id="test", hidden_size=1024, num_layers=28,
            recommended_hook_layer=14, architecture="qwen",
        )
        with pytest.raises(AttributeError):
            profile.hidden_size = 2048

    def test_invalid_hidden_size_zero(self):
        """hidden_size must be positive."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            ModelProfile(
                model_id="test", hidden_size=0, num_layers=12,
                recommended_hook_layer=6, architecture="unknown",
            )

    def test_invalid_hidden_size_negative(self):
        """hidden_size must be positive (negative is invalid)."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            ModelProfile(
                model_id="test", hidden_size=-1, num_layers=12,
                recommended_hook_layer=6, architecture="unknown",
            )

    def test_invalid_num_layers_zero(self):
        """num_layers must be positive."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            ModelProfile(
                model_id="test", hidden_size=1024, num_layers=0,
                recommended_hook_layer=0, architecture="unknown",
            )

    def test_invalid_hook_layer_out_of_range(self):
        """recommended_hook_layer must be less than num_layers."""
        with pytest.raises(ValueError, match="recommended_hook_layer"):
            ModelProfile(
                model_id="test", hidden_size=1024, num_layers=12,
                recommended_hook_layer=12, architecture="unknown",
            )


# ──────────────────────────────────────────────────────
#  Tests — SelfAdapter.from_config (Architecture Detection)
# ──────────────────────────────────────────────────────

class TestArchitectureDetection:
    """Tests for model family detection from config and model class name."""

    def test_qwen3_detected_as_qwen(self, qwen3_config):
        """Qwen3 config (model_type="qwen2") should detect as "qwen" architecture."""
        adapter = SelfAdapter.from_config(qwen3_config)
        assert adapter.profile.architecture == "qwen"

    def test_llama_detected_as_llama(self, llama_config):
        """Llama config (model_type="llama") should detect as "llama" architecture."""
        adapter = SelfAdapter.from_config(llama_config)
        assert adapter.profile.architecture == "llama"

    def test_gemma_detected_as_gemma(self, gemma_config):
        """Gemma2 config (model_type="gemma2") should detect as "gemma" architecture."""
        adapter = SelfAdapter.from_config(gemma_config)
        assert adapter.profile.architecture == "gemma"

    def test_mistral_detected_as_mistral(self, mistral_config):
        """Mistral config (model_type="mistral") should detect as "mistral" architecture."""
        adapter = SelfAdapter.from_config(mistral_config)
        assert adapter.profile.architecture == "mistral"

    def test_unknown_model_type_falls_back(self, unknown_config):
        """Config with no model_type should fall back to "unknown" architecture."""
        adapter = SelfAdapter.from_config(unknown_config)
        assert adapter.profile.architecture == "unknown"

    def test_class_name_fallback_qwen(self):
        """When model_type is missing, class name should detect architecture."""
        config = MockConfig(hidden_size=1024, num_hidden_layers=28, model_type=None)
        model = MockModel(config, class_name="Qwen2ForCausalLM")
        adapter = SelfAdapter.from_config(config, model=model)
        assert adapter.profile.architecture == "qwen"

    def test_class_name_fallback_llama(self):
        """When model_type is missing, class name should detect architecture."""
        config = MockConfig(hidden_size=2048, num_hidden_layers=32, model_type=None)
        model = MockModel(config, class_name="LlamaForCausalLM")
        adapter = SelfAdapter.from_config(config, model=model)
        assert adapter.profile.architecture == "llama"

    def test_class_name_fallback_mistral(self):
        """When model_type is missing, class name should detect architecture."""
        config = MockConfig(hidden_size=4096, num_hidden_layers=32, model_type=None)
        model = MockModel(config, class_name="MistralForCausalLM")
        adapter = SelfAdapter.from_config(config, model=model)
        assert adapter.profile.architecture == "mistral"

    def test_class_name_fallback_gemma(self):
        """When model_type is missing, class name should detect architecture."""
        config = MockConfig(hidden_size=2048, num_hidden_layers=18, model_type=None)
        model = MockModel(config, class_name="Gemma2ForCausalLM")
        adapter = SelfAdapter.from_config(config, model=model)
        assert adapter.profile.architecture == "gemma"


# ──────────────────────────────────────────────────────
#  Tests — SelfAdapter.from_model
# ──────────────────────────────────────────────────────

class TestFromModel:
    """Tests for SelfAdapter.from_model factory method."""

    def test_from_model_qwen3(self, qwen3_config):
        """from_model should auto-detect profile from a model with config."""
        model = MockModel(qwen3_config)
        adapter = SelfAdapter.from_model(model)
        assert adapter.profile.hidden_size == 1024
        assert adapter.profile.num_layers == 28
        assert adapter.profile.architecture == "qwen"

    def test_from_model_none_raises(self):
        """from_model(None) should raise UnsupportedModelError."""
        with pytest.raises(UnsupportedModelError, match="None model"):
            SelfAdapter.from_model(None)

    def test_from_model_no_config_raises(self):
        """from_model with an object that has no .config should raise."""
        class NoConfigModel:
            pass

        with pytest.raises(UnsupportedModelError, match="no .config"):
            SelfAdapter.from_model(NoConfigModel())


# ──────────────────────────────────────────────────────
#  Tests — UnsupportedModelError
# ──────────────────────────────────────────────────────

class TestUnsupportedModelError:
    """Tests for UnsupportedModelError raising on bad configs."""

    def test_missing_hidden_size(self):
        """Config without hidden_size should raise UnsupportedModelError."""
        config = MockConfig(hidden_size=1024, num_hidden_layers=28)
        del config.hidden_size  # Remove it after creation
        with pytest.raises(UnsupportedModelError, match="hidden_size"):
            SelfAdapter.from_config(config)

    def test_missing_num_hidden_layers(self):
        """Config without num_hidden_layers should raise UnsupportedModelError."""
        config = MockConfig(hidden_size=1024, num_hidden_layers=28)
        del config.num_hidden_layers  # Remove it after creation
        with pytest.raises(UnsupportedModelError, match="num_hidden_layers"):
            SelfAdapter.from_config(config)

    def test_unrecognized_model_type_with_valid_dims(self, unknown_config):
        """Unknown model_type but valid dims should NOT raise — returns "unknown" arch."""
        adapter = SelfAdapter.from_config(unknown_config)
        assert adapter.profile.architecture == "unknown"
        assert adapter.profile.hidden_size == 512
        assert adapter.profile.num_layers == 12


# ──────────────────────────────────────────────────────
#  Tests — recommended_hook_layer
# ──────────────────────────────────────────────────────

class TestRecommendedHookLayer:
    """Tests that recommended_hook_layer = num_layers // 2."""

    def test_qwen3_hook_layer(self, qwen3_config):
        """Qwen3: 28 layers → hook at layer 14."""
        adapter = SelfAdapter.from_config(qwen3_config)
        assert adapter.profile.recommended_hook_layer == 14  # 28 // 2

    def test_llama_hook_layer(self, llama_config):
        """Llama: 32 layers → hook at layer 16."""
        adapter = SelfAdapter.from_config(llama_config)
        assert adapter.profile.recommended_hook_layer == 16  # 32 // 2

    def test_gemma_hook_layer(self, gemma_config):
        """Gemma2: 18 layers → hook at layer 9."""
        adapter = SelfAdapter.from_config(gemma_config)
        assert adapter.profile.recommended_hook_layer == 9  # 18 // 2

    def test_mistral_hook_layer(self, mistral_config):
        """Mistral: 32 layers → hook at layer 16."""
        adapter = SelfAdapter.from_config(mistral_config)
        assert adapter.profile.recommended_hook_layer == 16  # 32 // 2

    def test_odd_num_layers(self):
        """Odd number of layers: 13 layers → hook at layer 6 (floor division)."""
        config = MockConfig(hidden_size=768, num_hidden_layers=13, model_type="llama")
        adapter = SelfAdapter.from_config(config)
        assert adapter.profile.recommended_hook_layer == 6  # 13 // 2


# ──────────────────────────────────────────────────────
#  Tests — Profile Fields
# ──────────────────────────────────────────────────────

class TestProfileFields:
    """Tests that profile fields are populated correctly from config."""

    def test_qwen3_profile_complete(self, qwen3_config):
        """Verify all fields of a Qwen3 profile match expected values."""
        adapter = SelfAdapter.from_config(qwen3_config)
        p = adapter.profile

        assert p.model_id == "Qwen/Qwen3-0.6B"
        assert p.hidden_size == 1024
        assert p.num_layers == 28
        assert p.recommended_hook_layer == 14
        assert p.architecture == "qwen"

    def test_llama_profile_complete(self, llama_config):
        """Verify all fields of a Llama profile match expected values."""
        adapter = SelfAdapter.from_config(llama_config)
        p = adapter.profile

        assert p.model_id == "meta-llama/Llama-3.2-1B"
        assert p.hidden_size == 2048
        assert p.num_layers == 32
        assert p.recommended_hook_layer == 16
        assert p.architecture == "llama"

    def test_gemma_profile_complete(self, gemma_config):
        """Verify all fields of a Gemma2 profile match expected values."""
        adapter = SelfAdapter.from_config(gemma_config)
        p = adapter.profile

        assert p.model_id == "google/gemma-2-2b"
        assert p.hidden_size == 2048
        assert p.num_layers == 18
        assert p.recommended_hook_layer == 9
        assert p.architecture == "gemma"

    def test_model_id_fallback_when_no_name_or_path(self):
        """Config without _name_or_path should use "unknown" as model_id."""
        config = MockConfig(hidden_size=512, num_hidden_layers=12, model_type="llama")
        # _name_or_path defaults to None in MockConfig, which means "unknown"
        adapter = SelfAdapter.from_config(config)
        assert adapter.profile.model_id == "unknown"


# ──────────────────────────────────────────────────────
#  Tests — project_to_hidden
# ──────────────────────────────────────────────────────

class TestProjectToHidden:
    """Tests for vector projection between arbitrary dimensions."""

    def test_same_dimension_no_op(self, qwen3_config):
        """Vector of same size as hidden_size should pass through unchanged."""
        adapter = SelfAdapter.from_config(qwen3_config)
        vector = np.random.randn(1024).astype(np.float32)
        result = adapter.project_to_hidden(vector)
        np.testing.assert_array_equal(result, vector)

    def test_pad_smaller_vector(self, qwen3_config):
        """Smaller vector should be zero-padded to hidden_size."""
        adapter = SelfAdapter.from_config(qwen3_config)
        vector = np.ones(64, dtype=np.float32)
        result = adapter.project_to_hidden(vector)

        assert result.shape == (1024,)
        np.testing.assert_array_equal(result[:64], vector)
        np.testing.assert_array_equal(result[64:], 0.0)

    def test_truncate_larger_vector(self, qwen3_config):
        """Larger vector should be truncated to hidden_size."""
        adapter = SelfAdapter.from_config(qwen3_config)
        vector = np.random.randn(2048).astype(np.float32)
        result = adapter.project_to_hidden(vector)

        assert result.shape == (1024,)
        np.testing.assert_array_equal(result, vector[:1024])

    def test_1d_to_4096(self, mistral_config):
        """Project a tiny 1-dim vector up to Mistral's 4096."""
        adapter = SelfAdapter.from_config(mistral_config)
        vector = np.array([0.5], dtype=np.float32)
        result = adapter.project_to_hidden(vector)

        assert result.shape == (4096,)
        assert result[0] == pytest.approx(0.5)
        np.testing.assert_array_equal(result[1:], 0.0)


# ──────────────────────────────────────────────────────
#  Tests — adapt_graph
# ──────────────────────────────────────────────────────

class TestAdaptGraph:
    """Tests that adapt_graph modifies AGNNGraph's embedding_dim."""

    def test_adapt_graph_changes_embedding_dim(self, qwen3_config):
        """adapt_graph should set graph._embedding_dim to model's hidden_size."""
        adapter = SelfAdapter.from_config(qwen3_config)
        graph = AGNNGraph(embedding_dim=64)  # Default 64-dim

        assert graph._embedding_dim == 64
        adapter.adapt_graph(graph)
        assert graph._embedding_dim == 1024  # Qwen3's hidden_size

    def test_adapt_graph_no_change_when_matching(self, qwen3_config):
        """adapt_graph should be a no-op if dims already match."""
        adapter = SelfAdapter.from_config(qwen3_config)
        graph = AGNNGraph(embedding_dim=1024)  # Already matches

        adapter.adapt_graph(graph)
        assert graph._embedding_dim == 1024

    def test_adapt_graph_llama(self, llama_config):
        """Llama adapter should set graph dim to 2048."""
        adapter = SelfAdapter.from_config(llama_config)
        graph = AGNNGraph()

        adapter.adapt_graph(graph)
        assert graph._embedding_dim == 2048

    def test_adapt_graph_gemma(self, gemma_config):
        """Gemma adapter should set graph dim to 2048."""
        adapter = SelfAdapter.from_config(gemma_config)
        graph = AGNNGraph()

        adapter.adapt_graph(graph)
        assert graph._embedding_dim == 2048

    def test_adapt_graph_mistral(self, mistral_config):
        """Mistral adapter should set graph dim to 4096."""
        adapter = SelfAdapter.from_config(mistral_config)
        graph = AGNNGraph()

        adapter.adapt_graph(graph)
        assert graph._embedding_dim == 4096

    def test_adapt_graph_new_nodes_use_new_dim(self, qwen3_config):
        """After adapt_graph, new nodes should get embeddings of the new dim."""
        adapter = SelfAdapter.from_config(qwen3_config)
        graph = AGNNGraph(embedding_dim=64)
        adapter.adapt_graph(graph)

        from agnn.graph import AGNNNode, NodeType
        node = AGNNNode(id="test", label="test", node_type=NodeType.ENTITY)
        graph.add_node(node)

        # add_node initializes zero embeddings with random vectors of _embedding_dim
        assert node.embedding.shape == (1024,)


# ──────────────────────────────────────────────────────
#  Tests — Concrete Example (Qwen3)
# ──────────────────────────────────────────────────────

class TestConcreteExample:
    """Concrete example: adapter initialized from mock Qwen3 config.

    This test demonstrates the exact output when SelfAdapter is created
    from a Qwen3-0.6B config, as required by the PR description.
    """

    def test_qwen3_concrete_output(self, qwen3_config):
        """Full concrete example of Qwen3 adapter initialization and profile output.

        Expected output:
            ModelProfile(
                model_id='Qwen/Qwen3-0.6B',
                hidden_size=1024,
                num_layers=28,
                recommended_hook_layer=14,
                architecture='qwen'
            )
        """
        adapter = SelfAdapter.from_config(qwen3_config)
        p = adapter.profile

        # All assertions matching the concrete example
        assert p.model_id == "Qwen/Qwen3-0.6B"
        assert p.hidden_size == 1024
        assert p.num_layers == 28
        assert p.recommended_hook_layer == 14
        assert p.architecture == "qwen"

        # Verify adapter property access
        assert adapter.profile is p

        # Verify repr
        repr_str = repr(adapter)
        assert "Qwen/Qwen3-0.6B" in repr_str
        assert "1024" in repr_str
        assert "28" in repr_str
        assert "qwen" in repr_str

        # Verify projection works
        small_vec = np.ones(64, dtype=np.float32)
        projected = adapter.project_to_hidden(small_vec)
        assert projected.shape == (1024,)

        # Verify graph adaptation works
        graph = AGNNGraph(embedding_dim=64)
        adapter.adapt_graph(graph)
        assert graph._embedding_dim == 1024
