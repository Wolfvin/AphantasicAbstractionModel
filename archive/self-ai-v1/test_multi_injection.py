#!/usr/bin/env python3
# @WHO:   self-ai/tests/test_multi_injection.py
# @WHAT:  Unit tests for multi-experience injection (v40: _compute_combined_vector)
# @PART:  self-ai/tests
# @ENTRY: python -m pytest self-ai/tests/test_multi_injection.py -v

"""Unit tests for multi-experience injection.

Tests that _compute_combined_vector correctly:
  1. Combines multiple nodes via weighted average (weights = node.confidence)
  2. Falls back to equal weights when all confidences are 0
  3. Returns None for empty node list
  4. Matches single-node injection (regression check)
  5. Skips nodes without embeddings
  6. Skips non-injectable nodes

Uses mock Qwen3 model — no real model loading required.
"""

import os
import sys

# ─── PATH SETUP ───
# self-ai/src must be on sys.path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')
SRC_DIR = os.path.abspath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest
import torch
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from unconscious.injector import UnconsciousInjector


# ═══════════════════════════════════════════════════════════════════
#  Test fixtures — mock Qwen3 model and UnderstandingNode helpers
# ═══════════════════════════════════════════════════════════════════

def _make_mock_model(hidden_size=1024, num_layers=28):
    """Create a mock Qwen3 model for testing without loading real weights.

    Mimics the interface UnconsciousInjector expects:
      - model.config.hidden_size
      - model.config.num_hidden_layers
      - model.model.layers[i].register_forward_hook()
      - model.parameters() → iterator of tensors (for device detection)
    """
    model = MagicMock()
    model.config.hidden_size = hidden_size
    model.config.num_hidden_layers = num_layers

    # Create mock layers with register_forward_hook
    mock_layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        mock_handle = MagicMock()
        mock_handle.remove = MagicMock()
        layer.register_forward_hook = MagicMock(return_value=mock_handle)
        mock_layers.append(layer)

    model.model.layers = mock_layers

    # model.parameters() for device detection in injector
    dummy_param = torch.randn(1)
    model.parameters = MagicMock(return_value=iter([dummy_param]))

    return model


def _make_node(node_id, confidence, embedding=None, concept="test concept",
               lifecycle=None, epistemic=None):
    """Create a mock UnderstandingNode with controlled attributes.

    Uses SimpleNamespace for flexibility — we only need the attributes
    that UnconsciousInjector reads (id, confidence, condition_embedding,
    lifecycle, epistemic, accuracy, etc.).
    """
    node = SimpleNamespace(
        id=node_id,
        name=f"Test node {node_id}",
        concept=concept,
        abstraction=f"Test abstraction for {node_id}",
        condition_embedding=embedding,
        confidence=confidence,
        accuracy=0.8,
        source='test',
        lifecycle=lifecycle,  # None = backward compatible (injectable)
        epistemic=epistemic,  # None = backward compatible (injectable)
        members=[],
        last_used_at=None,
        seed_scores=None,
        times_applied=1,
        times_correct=1,
        times_failed=0,
    )
    return node


# ═══════════════════════════════════════════════════════════════════
#  Test: _compute_combined_vector — weighted average correctness
# ═══════════════════════════════════════════════════════════════════

class TestComputeCombinedVector:
    """Tests for _compute_combined_vector method."""

    def test_two_nodes_weighted_average(self):
        """Test that 2 nodes with different confidences produce correct weighted average.

        With identity projection (default init), projected vectors equal
        input vectors. The weighted average should be:
            combined = (conf1 * vec1 + conf2 * vec2) / (conf1 + conf2)
        then normalized to unit length.
        """
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        # Create two nodes with known uniform embeddings and different confidences
        emb1 = [1.0] * 1024
        emb2 = [2.0] * 1024

        node1 = _make_node("n1", confidence=0.3, embedding=emb1)
        node2 = _make_node("n2", confidence=0.7, embedding=emb2)

        nodes = [(node1, 0.9), (node2, 0.8)]  # (node, score) tuples

        result = injector._compute_combined_vector(nodes)

        assert result is not None, "Should return a vector for 2 valid nodes"
        assert result.shape == (1024,), f"Expected shape (1024,), got {result.shape}"

        # Verify the weighted average manually
        vec1 = torch.tensor(emb1, dtype=torch.float32)
        vec2 = torch.tensor(emb2, dtype=torch.float32)

        # With identity projection, projected vectors == input vectors
        # Weighted average: (0.3 * [1,...] + 0.7 * [2,...]) / (0.3 + 0.7)
        # = (0.3 * [1,...] + 0.7 * [2,...]) / 1.0
        # = [0.3 + 1.4, ...] = [1.7, 1.7, ...]
        expected_raw = 0.3 * vec1 + 0.7 * vec2
        expected = expected_raw / expected_raw.norm()

        assert torch.allclose(result, expected, atol=1e-5), \
            "Weighted average does not match expected values"

        # Result should be normalized (unit norm)
        assert abs(result.norm().item() - 1.0) < 1e-6, \
            "Result should be normalized to unit length"

    def test_single_node_regression(self):
        """Test that 1 node produces the same result as single injection before.

        This is a regression check — with a single node, the combined vector
        should simply be the projected (then normalized) embedding of that node.
        With identity projection, projected == input.
        """
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        emb = [1.0, 2.0, 3.0] + [0.0] * 1021  # 1024-dim with some variation
        node1 = _make_node("n1", confidence=0.8, embedding=emb)

        nodes = [(node1, 0.9)]

        result = injector._compute_combined_vector(nodes)

        assert result is not None, "Should return a vector for 1 valid node"
        assert result.shape == (1024,), f"Expected shape (1024,), got {result.shape}"

        # Single node with confidence 0.8 → weight = 0.8
        # After weighted average (just the vector itself), normalized
        expected = torch.tensor(emb, dtype=torch.float32)
        expected = expected / expected.norm()

        assert torch.allclose(result, expected, atol=1e-5), \
            "Single node result should match projected+normalized embedding"

    def test_zero_nodes_no_injection(self):
        """Test that 0 nodes returns None (no injection)."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        result = injector._compute_combined_vector([])

        assert result is None, "Empty nodes list should return None"

    def test_all_zero_confidence_equal_weights(self):
        """Test that if all confidences sum to 0, equal weights are used as fallback.

        We use nodes with confidence=0.3 (pass injectable threshold >= 0.2)
        but override the internal weight calculation to simulate zero total weight.
        Since _compute_combined_vector uses node.confidence as weights directly,
        we test the fallback by creating a scenario where the code exercises
        the equal-weights path.

        To truly test the zero-weight fallback, we patch the weights to sum
        to zero after governance filtering but before combination.
        """
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        # Create nodes with injectable confidence (>= 0.2) but we'll
        # verify the equal-weight fallback by creating nodes with the
        # same embedding but very small confidence values that would
        # still be injectable. The key test: if weights sum to 0, use equal.
        #
        # Since _is_injectable requires confidence >= 0.2, we can't use
        # confidence=0 directly. Instead, we test by patching the
        # _is_injectable to allow confidence=0, then verify the fallback.
        emb1 = [1.0] * 1024
        emb2 = [3.0] * 1024

        node1 = _make_node("n1", confidence=0.3, embedding=emb1)
        node2 = _make_node("n2", confidence=0.5, embedding=emb2)

        # Patch _is_injectable to always return True, and patch confidence
        # to be 0 for the weight calculation inside _compute_combined_vector
        injector._is_injectable = lambda node: True

        # Temporarily set confidence to 0 to trigger the fallback
        node1_zero = _make_node("n1", confidence=0.0, embedding=emb1)
        node2_zero = _make_node("n2", confidence=0.0, embedding=emb2)

        nodes = [(node1_zero, 0.9), (node2_zero, 0.8)]

        result = injector._compute_combined_vector(nodes)

        assert result is not None, "Should return a vector even with zero confidences"
        assert result.shape == (1024,), f"Expected shape (1024,), got {result.shape}"

        # With equal weights: 0.5 * [1,...] + 0.5 * [3,...] = [2,...]
        vec1 = torch.tensor(emb1, dtype=torch.float32)
        vec2 = torch.tensor(emb2, dtype=torch.float32)
        expected_raw = 0.5 * vec1 + 0.5 * vec2
        expected = expected_raw / expected_raw.norm()

        assert torch.allclose(result, expected, atol=1e-5), \
            "Equal weight fallback should produce correct average"

    def test_node_without_embedding_skipped(self):
        """Test that nodes without condition_embedding are skipped."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        # Node without embedding — should be skipped
        node1 = _make_node("n1", confidence=0.8, embedding=None)
        # Node with embedding — should be used
        emb2 = [2.0] * 1024
        node2 = _make_node("n2", confidence=0.6, embedding=emb2)

        nodes = [(node1, 0.9), (node2, 0.8)]

        result = injector._compute_combined_vector(nodes)

        # Only node2 should contribute
        assert result is not None, "Should return a vector from the valid node"
        expected = torch.tensor(emb2, dtype=torch.float32)
        expected = expected / expected.norm()
        assert torch.allclose(result, expected, atol=1e-5), \
            "Only node with embedding should contribute"

    def test_node_with_empty_embedding_skipped(self):
        """Test that nodes with empty list embedding are skipped."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        node1 = _make_node("n1", confidence=0.8, embedding=[])
        emb2 = [2.0] * 1024
        node2 = _make_node("n2", confidence=0.6, embedding=emb2)

        nodes = [(node1, 0.9), (node2, 0.8)]

        result = injector._compute_combined_vector(nodes)

        assert result is not None
        expected = torch.tensor(emb2, dtype=torch.float32)
        expected = expected / expected.norm()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_non_injectable_node_skipped(self):
        """Test that non-injectable nodes (low confidence, deprecated) are skipped."""
        from governance.states import LifecycleState

        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        # Deprecated node — should be skipped by governance filter
        emb1 = [1.0] * 1024
        node1 = _make_node("n1", confidence=0.8, embedding=emb1,
                          lifecycle=LifecycleState.DEPRECATED)

        # Valid node — should be used
        emb2 = [2.0] * 1024
        node2 = _make_node("n2", confidence=0.6, embedding=emb2)

        nodes = [(node1, 0.9), (node2, 0.8)]

        result = injector._compute_combined_vector(nodes)

        assert result is not None
        expected = torch.tensor(emb2, dtype=torch.float32)
        expected = expected / expected.norm()
        assert torch.allclose(result, expected, atol=1e-5), \
            "Deprecated node should be skipped"

    def test_all_nodes_non_injectable_returns_none(self):
        """Test that if all nodes are non-injectable, None is returned."""
        from governance.states import LifecycleState

        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        emb = [1.0] * 1024
        node1 = _make_node("n1", confidence=0.8, embedding=emb,
                          lifecycle=LifecycleState.DEPRECATED)
        node2 = _make_node("n2", confidence=0.05, embedding=emb)  # below 0.2 threshold

        nodes = [(node1, 0.9), (node2, 0.8)]

        result = injector._compute_combined_vector(nodes)

        assert result is None, "All non-injectable nodes should return None"


# ═══════════════════════════════════════════════════════════════════
#  Test: active() interface — no breaking change
# ═══════════════════════════════════════════════════════════════════

class TestActiveInterface:
    """Tests that active() interface is unchanged."""

    def test_active_returns_self(self):
        """active() should return self for context manager protocol."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        nodes = [(_make_node("n1", 0.8, [1.0] * 1024), 0.9)]

        result = injector.active(nodes)

        assert result is injector, "active() should return self"

    def test_context_manager_protocol(self):
        """Test that injector works as context manager with active()."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        nodes = [(_make_node("n1", 0.8, [1.0] * 1024), 0.9)]

        with injector.active(nodes):
            # __enter__ should have computed the experience vector
            assert injector._experience_vector is not None, \
                "Experience vector should be set inside context"
            assert injector.is_active(), "Injector should be active inside context"

        # __exit__ should have cleaned up
        assert injector._experience_vector is None, \
            "Experience vector should be None after context exit"
        assert not injector.is_active(), "Injector should not be active after context exit"

    def test_active_with_empty_nodes_no_crash(self):
        """active() with empty nodes should not crash."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        with injector.active([]):
            pass  # Should not crash

        assert not injector.is_active()

    def test_active_disabled_no_injection(self):
        """active() with disabled injector should not inject."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=False)

        nodes = [(_make_node("n1", 0.8, [1.0] * 1024), 0.9)]

        with injector.active(nodes):
            assert injector._experience_vector is None, \
                "Disabled injector should not set experience vector"

    def test_injection_log_updated(self):
        """Test that injection log is properly updated after active()."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        node1 = _make_node("n1", 0.8, [1.0] * 1024)
        nodes = [(node1, 0.9)]

        with injector.active(nodes):
            log = injector.get_injection_log()
            assert log['active'] is True, "Log should show active injection"
            assert 'n1' in log['nodes'], "Log should contain injected node ID"
            assert log['strength'] == 0.3, "Log should record injection strength"
            assert log['layer'] == 14, "Log should record hook layer"


# ═══════════════════════════════════════════════════════════════════
#  Test: _get_node_embedding — cached vs on-the-fly
# ═══════════════════════════════════════════════════════════════════

class TestGetNodeEmbedding:
    """Tests for _get_node_embedding — cached embedding vs bge-m3 encoding."""

    def test_cached_list_embedding(self):
        """Node with condition_embedding as list should use cached value."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        emb = [1.0, 2.0, 3.0] + [0.0] * 1021
        node = _make_node("n1", 0.8, embedding=emb)

        result = injector._get_node_embedding(node)

        assert result is not None
        expected = torch.tensor(emb, dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_cached_tensor_embedding(self):
        """Node with condition_embedding as tensor should use cached value."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        emb = torch.randn(1024)
        node = _make_node("n1", 0.8, embedding=emb)

        result = injector._get_node_embedding(node)

        assert result is not None
        assert torch.allclose(result, emb.float().flatten())

    def test_no_embedding_no_model(self):
        """Node without embedding when bge-m3 unavailable should return None."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)
        # Mark embedding model as already attempted (so no lazy load)
        injector._embedding_model_loaded = True
        injector._embedding_model = None

        node = _make_node("n1", 0.8, embedding=None)

        result = injector._get_node_embedding(node)

        assert result is None, "Should return None when no cached embedding and no model"

    def test_empty_embedding_returns_none(self):
        """Node with empty list embedding should return None (not crash)."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)
        injector._embedding_model_loaded = True
        injector._embedding_model = None

        node = _make_node("n1", 0.8, embedding=[])

        result = injector._get_node_embedding(node)

        assert result is None


# ═══════════════════════════════════════════════════════════════════
#  Test: Integration — full flow with mock
# ═══════════════════════════════════════════════════════════════════

class TestIntegrationFlow:
    """Integration tests for the full multi-injection flow."""

    def test_hook_fn_adds_vector_to_hidden_state(self):
        """Test that _hook_fn correctly adds the experience vector to hidden states.

        This verifies the hook modifies hidden states at the last token position.
        """
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True, injection_strength=0.5)

        # Set up a known experience vector
        exp_vec = torch.ones(1024)
        injector._experience_vector = exp_vec
        injector._active = True

        # Create mock hidden states: (batch=1, seq_len=3, hidden_size=1024)
        hidden = torch.zeros(1, 3, 1024)
        output = hidden  # bare tensor (Qwen3 format)

        result = injector._hook_fn(None, None, output)

        # Last token position should have injection added
        # injection = exp_vec * 0.5 (injection_strength)
        expected_injection = exp_vec * 0.5
        assert torch.allclose(result[0, -1, :], expected_injection), \
            "Last token should have experience vector added"

        # Other positions should be unchanged
        assert torch.allclose(result[0, 0, :], torch.zeros(1024)), \
            "Non-last positions should be unchanged"

    def test_hook_fn_tuple_output(self):
        """Test that _hook_fn handles tuple output (older HF models)."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True, injection_strength=1.0)

        exp_vec = torch.ones(1024)
        injector._experience_vector = exp_vec
        injector._active = True

        hidden = torch.zeros(1, 3, 1024)
        extra = MagicMock()  # present_key_value or similar
        output = (hidden, extra)

        result = injector._hook_fn(None, None, output)

        assert isinstance(result, tuple), "Should return tuple when input is tuple"
        assert len(result) == 2, "Should preserve tuple length"
        # First element is modified hidden states
        assert torch.allclose(result[0][0, -1, :], exp_vec), \
            "Last token should have vector added"
        # Second element should be the same extra object
        assert result[1] is extra, "Extra tuple elements should be preserved"

    def test_hook_fn_inactive_no_modification(self):
        """Test that _hook_fn does nothing when injector is inactive."""
        model = _make_mock_model()
        injector = UnconsciousInjector(model, enabled=True)

        injector._experience_vector = None
        injector._active = False

        hidden = torch.randn(1, 3, 1024)
        output = hidden.clone()

        result = injector._hook_fn(None, None, output)

        assert torch.equal(result, output), "Should not modify when inactive"

    def test_combined_vector_dimension_matches_hidden_size(self):
        """Test that combined vector dimension matches QWEN3_HIDDEN_SIZE."""
        # Test with non-standard hidden size to verify no hardcoding
        model = _make_mock_model(hidden_size=768, num_layers=12)
        injector = UnconsciousInjector(model, enabled=True)

        # Embedding is still 1024-dim from bge-m3
        emb = [1.0] * 1024
        node1 = _make_node("n1", 0.8, embedding=emb)

        nodes = [(node1, 0.9)]
        result = injector._compute_combined_vector(nodes)

        assert result is not None
        assert result.shape == (768,), \
            f"Combined vector should match hidden_size=768, got {result.shape}"


# ═══════════════════════════════════════════════════════════════════
#  Import check — ensure UnconsciousInjector can be imported
# ═══════════════════════════════════════════════════════════════════

class TestImport:
    """Verify the module can be imported without errors."""

    def test_import_unconscious_injector(self):
        """Test that UnconsciousInjector can be imported from unconscious.injector."""
        from unconscious.injector import UnconsciousInjector
        assert UnconsciousInjector is not None

    def test_import_from_package(self):
        """Test that UnconsciousInjector can be imported from unconscious package."""
        from unconscious import UnconsciousInjector
        assert UnconsciousInjector is not None


if __name__ == '__main__':
    # Run with: python test_multi_injection.py
    pytest.main([__file__, '-v'])
