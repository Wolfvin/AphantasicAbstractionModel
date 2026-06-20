"""
Phase 1 (Aphantasic Node Representation) — unit tests.

Covers the four Phase 1 components:
  A. Episome fields (amodal_definition, causal_anchors, definition_dirty)
  B. DefinitionExtractor (lazy generation, cache, invalidation, /no_think)
  C. CausalAnchorBuilder (SPO reuse, relation pre-computation, no double-bump)
  D. AphantasicChainFormatter (3-layer KONSEP/DEFINISI/RELASI output)
  E. AGNNCore integration (lazy populate on articulate, reinforce invalidate)

Design notes:
  - All LLM-touching tests use mock tokenizer/model fixtures (same
    pattern as test_qwen3_integration.py) so the suite runs without a
    real Qwen3-0.6B checkpoint.
  - The Episome-field tests assert backward compatibility: existing
    callers that construct Episome(id, text, confidence) without the
    new fields must keep working (defaults: amodal_definition="",
    causal_anchors=(), definition_dirty=False).
  - The CausalAnchorBuilder tests assert the no-double-bump contract:
    when ``relation`` is passed, the builder must NOT call
    ``classifier.classify()`` a second time. This is the regression
    guard for the bug found during Phase 1 development (the
    frequency-table persistence test was failing because the builder
    was bumping "causes" twice per learn() call).

Run:
    python -m pytest AGNN/tests/test_aphantasic_node_representation.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Path setup — same convention as the other AGNN test modules.
_AGNP_ROOT = Path(__file__).resolve().parent.parent
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))

# Load AGNN/core.py by path to avoid the self-ai/src/core/ name collision.
import importlib.util as _ilu  # noqa: E402

_core_path = _AGNP_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_phase1_module", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_phase1_module"] = agnn_core_module
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore

from engrams.episodic_engram import Episome  # noqa: E402
from neocortex.aphantasic_chain_formatter import AphantasicChainFormatter  # noqa: E402
from neocortex.causal_anchor_builder import CausalAnchorBuilder  # noqa: E402
from neocortex.definition_extractor import DefinitionExtractor  # noqa: E402
from neocortex.semantic_role_classifier import (  # noqa: E402
    RelationType,
    SemanticRoleClassifier,
)


# ======================================================================
# Component A — Episome fields (backward compatibility + defaults)
# ======================================================================


def test_episome_new_fields_default_empty():
    """Episome constructed without Phase 1 fields must use safe defaults.

    This is the backward-compatibility guard: every existing caller
    that constructs Episome(id, text, confidence) — including
    Subiculum.relay_output() before Phase 1 — must keep working
    without passing the new kwargs.
    """
    epi = Episome(id=1, text="api menyebabkan panas", confidence=0.6)
    assert epi.amodal_definition == ""
    assert epi.causal_anchors == ()
    assert epi.definition_dirty is False


def test_episome_new_fields_accept_values():
    """Episome accepts the new Phase 1 fields when supplied."""
    epi = Episome(
        id=2,
        text="api menyebabkan panas",
        confidence=0.6,
        amodal_definition="fenomena pembakaran",
        causal_anchors=(("CAUSAL", "panas"),),
        definition_dirty=True,
    )
    assert epi.amodal_definition == "fenomena pembakaran"
    assert epi.causal_anchors == (("CAUSAL", "panas"),)
    assert epi.definition_dirty is True


def test_episome_causal_anchors_is_tuple_of_tuples():
    """causal_anchors must be a tuple of tuples (immutable, hashable).

    The docstring promises this so the dataclass can be used in
    sets / dict keys in future work. We assert the type explicitly
    because a list-of-tuples would silently break that contract.
    """
    epi = Episome(
        id=3,
        text="x",
        confidence=0.5,
        causal_anchors=(("CAUSAL", "y"), ("FUNCTIONAL", "z")),
    )
    assert isinstance(epi.causal_anchors, tuple)
    assert all(isinstance(a, tuple) for a in epi.causal_anchors)


# ======================================================================
# Component B — DefinitionExtractor
# ======================================================================


class _FakeTokenizer:
    """Minimal tokenizer mock for DefinitionExtractor tests.

    Returns a canned chat-formatted string + a canned decode output.
    Mirrors the _FakeTokenizer in test_qwen3_integration.py but
    trimmed to just the methods DefinitionExtractor calls.
    """

    def __init__(self, decode_output: str = "fenomena pembakaran"):
        self._decode_output = decode_output
        self.apply_chat_template_calls = []
        self.call_calls = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        self.apply_chat_template_calls.append(messages)
        # Concatenate system + user content so the resulting "chat
        # text" is non-empty (the real Qwen3 chat template wraps each
        # message in <|im_start|>role … <|im_end|> tokens).
        return " ".join(m.get("content", "") for m in messages)

    def __call__(self, text, return_tensors="pt"):
        self.call_calls.append(text)
        # Return a fake inputs dict whose input_ids has a known shape.
        class _FakeTensor:
            def __init__(self, n):
                self.shape = (1, n)
            def __getitem__(self, k):
                return self
        return {"input_ids": _FakeFakeTensor()}

    def decode(self, token_ids, skip_special_tokens=True):
        return self._decode_output


class _FakeFakeTensor:
    """Stand-in for a tensor with .shape[1] used by DefinitionExtractor."""
    shape = (1, 5)  # 5 prompt tokens

    def __getitem__(self, idx):
        # Slice [input_len:] — return something indexable but its
        # contents don't matter because decode() ignores the arg.
        return self


class _FakeModel:
    """Minimal model mock: returns a 1-element list whose [0] is a
    tensor of length input_len + len(generated)."""

    def __init__(self, n_generated=3):
        self._n_generated = n_generated
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        input_ids = kwargs.get("input_ids")
        input_len = 5  # matches _FakeFakeTensor.shape[1]
        total = input_len + self._n_generated

        class _GenTensor:
            shape = (total,)

            def __getitem__(self, idx):
                return self

        return [_GenTensor()]


def _install_fake_torch(monkeypatch):
    """Install a fake torch module with a working no_grad() context."""
    import types

    class _NoOpContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: _NoOpContext()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_definition_extractor_returns_empty_for_empty_text(monkeypatch):
    """Empty/whitespace text short-circuits to empty string."""
    _install_fake_torch(monkeypatch)
    extractor = DefinitionExtractor()
    result = extractor.extract(
        text="   ",
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
    )
    assert result == ""


def test_definition_extractor_returns_empty_when_model_none(monkeypatch):
    """When model is None, return empty string (no LLM call)."""
    _install_fake_torch(monkeypatch)
    extractor = DefinitionExtractor()
    result = extractor.extract(
        text="api menyebabkan panas",
        model=None,
        tokenizer=_FakeTokenizer(),
    )
    assert result == ""


def test_definition_extractor_returns_empty_when_tokenizer_none(monkeypatch):
    """When tokenizer is None, return empty string."""
    _install_fake_torch(monkeypatch)
    extractor = DefinitionExtractor()
    result = extractor.extract(
        text="api menyebabkan panas",
        model=_FakeModel(),
        tokenizer=None,
    )
    assert result == ""


def test_definition_extractor_generates_and_caches(monkeypatch):
    """First call generates; second call hits the cache (no second LLM call)."""
    _install_fake_torch(monkeypatch)
    tokenizer = _FakeTokenizer(decode_output="fenomena pembakaran")
    model = _FakeModel()
    extractor = DefinitionExtractor()

    first = extractor.extract("api menyebabkan panas", model, tokenizer)
    assert first == "fenomena pembakaran"
    assert len(model.generate_calls) == 1

    # Second call with the same text must hit the cache.
    second = extractor.extract("api menyebabkan panas", model, tokenizer)
    assert second == "fenomena pembakaran"
    assert len(model.generate_calls) == 1, (
        "second call must hit the cache — model.generate must NOT be "
        "called again"
    )
    assert extractor.cache_size == 1


def test_definition_extractor_force_refresh_bypasses_cache(monkeypatch):
    """force_refresh=True re-generates even on a cache hit."""
    _install_fake_torch(monkeypatch)
    tokenizer = _FakeTokenizer(decode_output="updated definition")
    model = _FakeModel()
    extractor = DefinitionExtractor()

    extractor.extract("api", model, tokenizer)
    assert len(model.generate_calls) == 1

    extractor.extract("api", model, tokenizer, force_refresh=True)
    assert len(model.generate_calls) == 2, (
        "force_refresh=True must trigger a new LLM call"
    )


def test_definition_extractor_invalidate_drops_cache_entry(monkeypatch):
    """invalidate(text) drops the cache entry for that text."""
    _install_fake_torch(monkeypatch)
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    extractor = DefinitionExtractor()

    extractor.extract("api menyebabkan panas", model, tokenizer)
    assert extractor.cache_size == 1

    extractor.invalidate("api menyebabkan panas")
    assert extractor.cache_size == 0

    # Next extract re-generates.
    extractor.extract("api menyebabkan panas", model, tokenizer)
    assert len(model.generate_calls) == 2


def test_definition_extractor_caches_by_text_hash_across_calls(monkeypatch):
    """Two different Episomes with the same text share one cache entry."""
    _install_fake_torch(monkeypatch)
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    extractor = DefinitionExtractor()

    # First Episome's text.
    extractor.extract("api menyebabkan panas", model, tokenizer)
    # Second Episome with the same text (e.g. user re-learned after penalize).
    extractor.extract("api menyebabkan panas", model, tokenizer)
    assert extractor.cache_size == 1, (
        "two extracts with the same text must share one cache entry"
    )
    assert len(model.generate_calls) == 1


def test_definition_extractor_enforces_word_limit(monkeypatch):
    """Generated definitions are truncated to max_words."""
    _install_fake_torch(monkeypatch)
    # A 20-word decode output.
    long_output = " ".join(["word"] * 20)
    tokenizer = _FakeTokenizer(decode_output=long_output)
    model = _FakeModel()
    extractor = DefinitionExtractor(max_words=15)

    result = extractor.extract("api", model, tokenizer)
    assert len(result.split()) == 15, (
        f"definition must be truncated to 15 words, got {len(result.split())}"
    )


def test_definition_extractor_system_message_uses_no_think(monkeypatch):
    """The system message sent to Qwen3 must start with /no_think.

    This is the Phase 1 contract: definition extraction is NOT a
    reasoning task, so Long-CoT (thinking mode) is overkill and adds
    noise. We assert the /no_think prefix appears in the system
    message captured by the fake tokenizer.
    """
    _install_fake_torch(monkeypatch)
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    extractor = DefinitionExtractor()

    extractor.extract("api menyebabkan panas", model, tokenizer)
    assert tokenizer.apply_chat_template_calls, (
        "apply_chat_template must be called"
    )
    messages = tokenizer.apply_chat_template_calls[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("/no_think"), (
        "system message must start with /no_think to disable Long-CoT"
    )


# ======================================================================
# Component C — CausalAnchorBuilder
# ======================================================================


def test_causal_anchor_builder_causal_correction():
    """'api menyebabkan panas' (CAUSAL) → (('CAUSAL', 'panas'),)."""
    classifier = SemanticRoleClassifier()
    builder = CausalAnchorBuilder(classifier)
    anchors = builder.build("api menyebabkan panas", relation=RelationType.CAUSAL)
    assert anchors == (("CAUSAL", "panas"),)


def test_causal_anchor_builder_functional_correction():
    """'api membutuhkan oksigen' (FUNCTIONAL) → (('FUNCTIONAL', 'oksigen'),)."""
    classifier = SemanticRoleClassifier()
    builder = CausalAnchorBuilder(classifier)
    anchors = builder.build(
        "api membutuhkan oksigen", relation=RelationType.FUNCTIONAL
    )
    assert anchors == (("FUNCTIONAL", "oksigen"),)


def test_causal_anchor_builder_differential_correction():
    """'api tidak menyebabkan dingin' (DIFFERENTIAL) →
    (('DIFFERENTIAL', 'dingin'),)."""
    classifier = SemanticRoleClassifier()
    builder = CausalAnchorBuilder(classifier)
    anchors = builder.build(
        "api tidak menyebabkan dingin", relation=RelationType.DIFFERENTIAL
    )
    assert anchors == (("DIFFERENTIAL", "dingin"),)


def test_causal_anchor_builder_categorical_returns_empty():
    """CATEGORICAL corrections yield no anchors (identity, not cause-effect)."""
    classifier = SemanticRoleClassifier()
    builder = CausalAnchorBuilder(classifier)
    anchors = builder.build(
        "api adalah phenomenon", relation=RelationType.CATEGORICAL
    )
    assert anchors == ()


def test_causal_anchor_builder_single_token_returns_empty():
    """Single-token corrections ('api') yield no SPO object → no anchors."""
    classifier = SemanticRoleClassifier()
    builder = CausalAnchorBuilder(classifier)
    anchors = builder.build("api", relation=RelationType.CATEGORICAL)
    assert anchors == ()


def test_causal_anchor_builder_empty_correction_returns_empty():
    """Empty/whitespace correction returns empty tuple."""
    classifier = SemanticRoleClassifier()
    builder = CausalAnchorBuilder(classifier)
    assert builder.build("", relation=RelationType.CAUSAL) == ()
    assert builder.build("   ", relation=RelationType.CAUSAL) == ()


def test_causal_anchor_builder_no_double_bump_when_relation_supplied():
    """When ``relation`` is supplied, builder must NOT call classify() again.

    This is the regression guard for the frequency-table double-bump
    bug found during Phase 1 development. The builder should reuse the
    caller-supplied relation, not call classifier.classify() a second
    time (which would bump the frequency table twice per learn()).
    """
    classifier = SemanticRoleClassifier()
    builder = CausalAnchorBuilder(classifier)

    # Count classify calls.
    original_classify = classifier.classify
    call_count = {"n": 0}

    def counting_classify(text):
        call_count["n"] += 1
        return original_classify(text)

    classifier.classify = counting_classify

    # build() with a pre-computed relation must NOT call classify().
    builder.build("api menyebabkan panas", relation=RelationType.CAUSAL)
    assert call_count["n"] == 0, (
        "build(relation=...) must NOT call classifier.classify() — "
        "that would double-bump the frequency table"
    )


def test_causal_anchor_builder_relation_is_required():
    """``build()`` requires a pre-computed ``relation`` argument.

    Replaces ``test_causal_anchor_builder_falls_back_to_classify_without_relation``
    (dead-code-audit §3.4): the ``relation is None`` fallback branch
    in ``CausalAnchorBuilder.build()`` was production-dead —
    ``TrisynapticCircuit.encode()`` always passes a pre-computed
    relation. The branch has been removed and ``relation`` is now a
    required parameter.

    Contract pinned by this test:
      * Calling ``build()`` without ``relation`` raises ``TypeError``
        (the Python signature enforces it; no ``None`` default).
      * Calling ``build()`` with ``relation`` never calls
        ``classifier.classify()`` — the supplied relation is used
        directly. (This is the same no-double-bump property pinned by
        ``test_causal_anchor_builder_no_double_bump_when_relation_supplied``,
        re-stated here from the "no fallback" angle to make the
        post-removal contract explicit.)
    """
    classifier = SemanticRoleClassifier()
    builder = CausalAnchorBuilder(classifier)

    import inspect as _inspect

    sig = _inspect.signature(builder.build)
    params = sig.parameters
    assert "relation" in params, "build() must accept a `relation` parameter"
    assert (
        params["relation"].default is _inspect.Parameter.empty
    ), "build(relation=...) must be required — no default, no None fallback"

    # Calling build() without relation raises TypeError (no fallback
    # to classify()).
    with pytest.raises(TypeError):
        builder.build("api menyebabkan panas")

    # And build() with relation does NOT call classify().
    call_count = {"n": 0}
    original_classify = classifier.classify

    def counting_classify(text):
        call_count["n"] += 1
        return original_classify(text)

    classifier.classify = counting_classify

    builder.build("api menyebabkan panas", relation=RelationType.CAUSAL)
    assert call_count["n"] == 0, (
        "build(relation=...) must NOT call classifier.classify() — "
        "the relation is supplied by the caller; no fallback path."
    )


# ======================================================================
# Component D — AphantasicChainFormatter
# ======================================================================


def test_formatter_empty_episomes_returns_empty():
    """No episomes → empty string."""
    formatter = AphantasicChainFormatter()
    assert formatter.format([]) == ""


def test_formatter_surface_only_when_no_layers_2_3():
    """When amodal_definition + causal_anchors are empty, emit KONSEP only.

    This is the backward-compat case: pre-Phase-1 Episomes (or Phase 1
    Episomes that haven't been articulated yet) produce a KONSEP-only
    block.
    """
    formatter = AphantasicChainFormatter()
    epi = Episome(id=1, text="api", confidence=0.6)
    out = formatter.format([epi])
    assert out == "KONSEP: api"


def test_formatter_full_3_layer_block():
    """All three layers present → KONSEP + DEFINISI + RELASI block."""
    formatter = AphantasicChainFormatter()
    epi = Episome(
        id=1,
        text="api menyebabkan panas",
        confidence=0.6,
        amodal_definition="fenomena pembakaran yang menghasilkan panas",
        causal_anchors=(("CAUSAL", "panas"), ("FUNCTIONAL", "oksigen")),
    )
    out = formatter.format([epi])
    assert "KONSEP: api menyebabkan panas" in out
    assert "DEFINISI: fenomena pembakaran yang menghasilkan panas" in out
    assert "RELASI:" in out
    assert "(CAUSAL) → panas" in out
    assert "(FUNCTIONAL) → oksigen" in out


def test_formatter_multiple_episomes_separated_by_blank_line():
    """Multiple episomes → multiple blocks separated by blank lines."""
    formatter = AphantasicChainFormatter()
    e1 = Episome(id=1, text="api", confidence=0.6, amodal_definition="fenomena pembakaran")
    e2 = Episome(id=2, text="panas", confidence=0.6, amodal_definition="bentuk energi")
    out = formatter.format([e1, e2])
    blocks = out.split("\n\n")
    assert len(blocks) == 2
    assert "KONSEP: api" in blocks[0]
    assert "KONSEP: panas" in blocks[1]


def test_formatter_anchors_sorted_by_priority():
    """Anchors sorted CAUSAL > FUNCTIONAL > DIFFERENTIAL."""
    formatter = AphantasicChainFormatter()
    epi = Episome(
        id=1,
        text="x",
        confidence=0.6,
        amodal_definition="d",
        # Out of order on purpose.
        causal_anchors=(
            ("DIFFERENTIAL", "z"),
            ("CAUSAL", "a"),
            ("FUNCTIONAL", "b"),
        ),
    )
    out = formatter.format([epi])
    # The CAUSAL line must appear before FUNCTIONAL, which appears
    # before DIFFERENTIAL.
    causal_idx = out.index("(CAUSAL)")
    functional_idx = out.index("(FUNCTIONAL)")
    differential_idx = out.index("(DIFFERENTIAL)")
    assert causal_idx < functional_idx < differential_idx


def test_formatter_truncates_long_surface_text():
    """Surface text longer than max_surface_chars is truncated with ellipsis."""
    formatter = AphantasicChainFormatter(max_surface_chars=20)
    long_text = "a" * 100
    epi = Episome(id=1, text=long_text, confidence=0.6)
    out = formatter.format([epi])
    assert "…" in out
    # The visible portion (before the ellipsis) must be ≤ 20 chars.
    konspe_line = out.split("\n")[0]
    visible = konspe_line[len("KONSEP: "):-1]  # strip prefix + ellipsis
    assert len(visible) <= 20


# ======================================================================
# Component E — AGNNCore integration
# ======================================================================


def test_agnn_core_phase1_helpers_initialised():
    """AGNNCore constructs the Phase 1 helpers on init."""
    core = AGNNCore(model_path=None)
    assert core._definition_extractor is not None
    assert core._chain_formatter is not None
    assert isinstance(core._definition_extractor, DefinitionExtractor)
    assert isinstance(core._chain_formatter, AphantasicChainFormatter)


def test_agnn_core_reinforce_delta_tracker_initialised():
    """AGNNCore starts with an empty reinforce delta tracker."""
    core = AGNNCore(model_path=None)
    assert core._reinforce_deltas == {}


def test_agnn_core_definition_invalidate_threshold_is_03():
    """The invalidate threshold matches the user-confirmed value (0.3)."""
    assert AGNNCore._DEFINITION_INVALIDATE_THRESHOLD == 0.3


def test_agnn_core_articulate_aphantasic_exists():
    """The new ``_articulate_aphantasic`` method exists and is callable."""
    core = AGNNCore(model_path=None)
    assert callable(getattr(core, "_articulate_aphantasic", None))


def test_agnn_core_articulate_aphantasic_empty_episomes_returns_empty():
    """Empty episomes list short-circuits to empty string."""
    core = AGNNCore(model_path=None)
    result = core._articulate_aphantasic("question?", [])
    assert result == ""


def test_agnn_core_reinforce_marks_dirty_at_threshold(monkeypatch):
    """Three reinforces (Δ=0.3) cross the threshold → dirty flag set + cache invalidated.

    This is the user-confirmed cache-invalidation contract:
      - Two reinforces (Δ=0.2) do NOT cross the threshold → no invalidation.
      - Three reinforces (Δ=0.3) cross the threshold → definition_dirty=True,
        cache invalidated, delta resets to 0.
    """
    _install_fake_torch(monkeypatch)
    core = AGNNCore(model_path=None)
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    result = core.learn("q?", "wrong", "api menyebabkan panas")
    node_id = result["node_id"]
    if node_id is None:
        pytest.skip("learn() returned null node_id")

    # Two reinforces — should NOT cross the 0.3 threshold.
    core.reinforce(node_id)
    core.reinforce(node_id)
    epi = core._find_episome(node_id)
    assert epi.definition_dirty is False, (
        "two reinforces (Δ=0.2) must NOT invalidate the definition"
    )
    assert core._reinforce_deltas[node_id] == pytest.approx(0.2)

    # Third reinforce — crosses the threshold.
    core.reinforce(node_id)
    assert epi.definition_dirty is True, (
        "three reinforces (Δ=0.3) must set definition_dirty=True"
    )
    # Delta resets to 0 after invalidation.
    assert core._reinforce_deltas[node_id] == pytest.approx(0.0)


def test_agnn_core_reinforce_unknown_id_is_safe_noop():
    """reinforce(unknown_id) must not crash and must not touch the delta tracker."""
    core = AGNNCore(model_path=None)
    # Should not raise.
    core.reinforce(99999)
    assert core._reinforce_deltas == {}


def test_agnn_core_populate_definition_uses_force_refresh_when_dirty(monkeypatch):
    """When episome.definition_dirty=True, _populate_definition forces refresh."""
    _install_fake_torch(monkeypatch)
    core = AGNNCore(model_path=None)

    # Manually craft an episome with dirty=True + a pre-existing defn.
    epi = Episome(
        id=1,
        text="api menyebabkan panas",
        confidence=0.6,
        amodal_definition="old definition",
        definition_dirty=True,
    )

    # Mock the extractor to track force_refresh.
    refresh_calls = []

    def mock_extract(text, model, tokenizer, force_refresh=False):
        refresh_calls.append(force_refresh)
        return "new definition"

    core._definition_extractor.extract = mock_extract
    core._populate_definition(epi)

    assert refresh_calls == [True], (
        "force_refresh must be True when episome.definition_dirty is True"
    )
    assert epi.amodal_definition == "new definition"
    assert epi.definition_dirty is False, (
        "dirty flag must be reset after a populate attempt"
    )


def test_agnn_core_populate_definition_skips_when_cache_hit():
    """When amodal_definition is non-empty + not dirty, skip the LLM call."""
    core = AGNNCore(model_path=None)
    epi = Episome(
        id=1,
        text="api",
        confidence=0.6,
        amodal_definition="existing definition",
        definition_dirty=False,
    )

    call_count = {"n": 0}

    def mock_extract(*args, **kwargs):
        call_count["n"] += 1
        return "should not be called"

    core._definition_extractor.extract = mock_extract
    core._populate_definition(epi)
    assert call_count["n"] == 0, (
        "extract must NOT be called when amodal_definition is non-empty "
        "and definition_dirty is False"
    )
