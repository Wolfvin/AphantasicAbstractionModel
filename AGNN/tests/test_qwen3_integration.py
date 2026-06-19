"""
Tests for Qwen3-0.6B integration into ``AGNNCore`` (feat/agnn-qwen3-integration).

Covers the 5 Definition-of-Done requirements from the task brief:

    1. ``process()`` with no model available returns a valid dict, no crash.
    2. ``_load_model()`` with a mocked ``AutoModelForCausalLM`` is called
       exactly once (lazy load — not re-attempted on subsequent calls).
    3. ``_generate()`` with a mocked tokenizer/model returns a string.
    4. The prompt template includes the chain text.
    5. Chains longer than 800 chars are truncated in the prompt.

All tests use mocking — no real Qwen3-0.6B weights are needed. The
fake ``transformers`` module is injected into ``sys.modules`` per-test
and cleaned up afterwards so it never leaks into other test files.

Run:
    cd <repo-root>
    python -m pytest AGNN/tests/test_qwen3_integration.py -v
"""

import importlib.util as _ilu
import os
import sys
import types
from pathlib import Path

import pytest

# ----------------------------------------------------------------------
# Make AGNN package importable when running from repo root.
# Tests are invoked as: python -m pytest AGNN/tests/test_qwen3_integration.py -v
# So we add the AGNN/ directory (parent of tests/) to sys.path.
# ----------------------------------------------------------------------

_AGNP_ROOT = Path(__file__).resolve().parent.parent
_SELF_AI_SRC = _AGNP_ROOT.parent / "self-ai" / "src"

# Insert self-ai/src FIRST (lower priority) so that the AGNN package
# (inserted next) wins on name collisions. This matters because both
# trees expose a "core" name: AGNN/core.py vs self-ai/src/core/.
if _SELF_AI_SRC.exists() and str(_SELF_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_SELF_AI_SRC))
if str(_AGNP_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGNP_ROOT))


# ----------------------------------------------------------------------
# Load AGNN/core.py directly by path (avoids collision with
# self-ai/src/core/ package). Same pattern as test_core_wired.py.
# ----------------------------------------------------------------------

_core_path = _AGNP_ROOT / "core.py"
_spec = _ilu.spec_from_file_location("agnn_core_module_qwen3", _core_path)
agnn_core_module = _ilu.module_from_spec(_spec)
sys.modules["agnn_core_module_qwen3"] = agnn_core_module  # register before exec
_spec.loader.exec_module(agnn_core_module)
AGNNCore = agnn_core_module.AGNNCore


# ----------------------------------------------------------------------
# Fake transformers module helpers
# ----------------------------------------------------------------------


class _FakeTensor:
    """Minimal fake tensor with ``.shape[1]`` and slicing support.

    Mimics just enough of ``torch.Tensor`` for the
    ``outputs[0][input_len:]`` slice in ``_generate()`` to work.
    """

    def __init__(self, data, dim1=None):
        # data can be a list or another _FakeTensor
        if isinstance(data, _FakeTensor):
            self._data = list(data._data)
            self._shape = list(data._shape)
        else:
            self._data = list(data)
            # Treat as 2-D: [batch, seq_len]. dim1 overrides seq_len.
            self._shape = [1, dim1 if dim1 is not None else len(self._data)]

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, key):
        # outputs[0] -> return a 1-D-ish fake tensor over the data
        if isinstance(key, int):
            return _FakeTensor(self._data, dim1=len(self._data))
        # slicing like [start:] -> slice the underlying data
        if isinstance(key, slice):
            return _FakeTensor(self._data[key], dim1=len(self._data[key]))
        raise TypeError(f"unsupported key type: {type(key)}")


class _FakeInputs:
    """Fake tokenizer output: dict-like with ``input_ids`` tensor."""

    def __init__(self, prompt: str):
        # Tokenize by splitting on whitespace — good enough for tests.
        tokens = prompt.split()
        self["input_ids"] = _FakeTensor(tokens, dim1=len(tokens))
        self["prompt"] = prompt

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if not hasattr(self, "_data"):
            self._data = {}
        self._data[key] = value
        # Also set as attribute for .input_ids access pattern.
        setattr(self, key.replace("-", "_"), value)

    def __init_subclass__(cls, **kw):
        pass

    def get(self, key, default=None):
        return getattr(self, key.replace("-", "_"), default)


def _make_fake_inputs(prompt: str):
    """Build a fake tokenizer-returned dict with ``input_ids``."""
    tokens = prompt.split()
    fake_tensor = _FakeTensor(tokens, dim1=len(tokens))
    return {"input_ids": fake_tensor}


class _FakeTokenizer:
    """Fake HF tokenizer for ``_generate()`` tests.

    Supports the three methods ``_generate`` exercises on a real Qwen3
    tokenizer: ``__call__`` (tokenize), ``decode`` (detokenize), and
    ``apply_chat_template`` (wrap a messages list in chat format).
    The chat-template stub returns the first message's ``content``
    verbatim so tests that capture the prompt still see the original
    formatted string (e.g. ``[Knowledge Graph Context]...Q: ...\\nA:``).
    """

    def __init__(self, decode_output: str = "generated answer"):
        self._decode_output = decode_output
        self.calls = []  # audit log

    def __call__(self, prompt, return_tensors="pt"):
        self.calls.append(("call", prompt))
        return _make_fake_inputs(prompt)

    def decode(self, token_ids, skip_special_tokens=True):
        self.calls.append(("decode", token_ids))
        return self._decode_output

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        self.calls.append(("apply_chat_template", messages))
        # Return the user message content as the "chat-formatted" text.
        # This keeps tests focused on _generate's contract (decoded
        # output, max_new_tokens forwarding, fallback-on-error) rather
        # than on Qwen3-specific chat tokenization.
        if not messages:
            return ""
        return str(messages[0].get("content", ""))


class _FakeModel:
    """Fake HF model for ``_generate()`` tests."""

    def __init__(self, generated_tokens=None):
        # generated_tokens: list of token strings to "produce".
        # If None, defaults to a single-token output.
        self._generated = generated_tokens or ["gen"]
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        # outputs[0] should be sliceable; wrap in _FakeTensor so
        # outputs[0][input_len:] works.
        # We return a 1-element list whose [0] is a fake tensor over
        # the prompt tokens + generated tokens (so slicing past
        # input_len yields the generated part).
        input_ids = kwargs.get("input_ids")
        # input_ids might be a _FakeTensor or a dict's value.
        input_len = 1
        if hasattr(input_ids, "shape"):
            input_len = input_ids.shape[1]
        elif isinstance(input_ids, dict):
            input_len = input_ids["input_ids"].shape[1]
        # Total = input_len + len(generated). Slice [input_len:] yields
        # the generated portion.
        total = input_len + len(self._generated)
        return [_FakeTensor(["tok"] * total, dim1=total)]


def _install_fake_transformers(monkeypatch, tokenizer_cls=_FakeTokenizer,
                               model_cls=_FakeModel):
    """Inject a fake ``transformers`` module into ``sys.modules``.

    Returns a dict with ``tokenizer_cls`` and ``model_cls`` for the
    test to inspect call counts. The fake module exposes
    ``AutoTokenizer`` and ``AutoModelForCausalLM`` classes whose
    ``from_pretrained`` return instances of the given fake classes.

    Also injects a fake ``torch`` module (needed because ``_generate``
    imports ``torch`` for ``torch.no_grad()``).

    Cleanup is automatic via ``monkeypatch`` (it restores
    ``sys.modules`` on test teardown).
    """
    fake_module = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path, *args, **kwargs):
            return tokenizer_cls()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, *args, **kwargs):
            return model_cls()

    fake_module.AutoTokenizer = _AutoTokenizer
    fake_module.AutoModelForCausalLM = _AutoModelForCausalLM

    # Also install a fake torch module (imported inside _generate).
    fake_torch = types.ModuleType("torch")

    class _NoOpContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_torch.no_grad = lambda: _NoOpContext()
    fake_torch.Tensor = _FakeTensor

    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    return {"tokenizer_cls": tokenizer_cls, "model_cls": model_cls}


def _install_fake_torch_only(monkeypatch):
    """Inject only a fake ``torch`` module (for _generate tests that
    manually set ``core._tokenizer`` and ``core.model``).

    ``_generate()`` does ``import torch`` inside the function body, so
    without a fake torch the import fails and the except clause
    swallows the error — returning the prompt instead of the decoded
    output. This helper prevents that.
    """
    fake_torch = types.ModuleType("torch")

    class _NoOpContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_torch.no_grad = lambda: _NoOpContext()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def brain_no_model(monkeypatch):
    """AGNNCore with no model path AND QWEN_PATH unset.

    This is the "no model available" scenario. ``_articulate()`` should
    return the graceful fallback (chain snippet), and ``process()``
    should return a valid dict without crashing.
    """
    monkeypatch.delenv("QWEN_PATH", raising=False)
    core = AGNNCore(model_path=None)
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")
    return core


@pytest.fixture(autouse=True)
def _clear_qwen_path(monkeypatch):
    """Auto-clear QWEN_PATH for every test so they're isolated.

    Tests that need QWEN_PATH set will set it explicitly.
    """
    monkeypatch.delenv("QWEN_PATH", raising=False)


# ======================================================================
# Requirement 1: process() with no model returns valid dict, no crash
# ======================================================================


def test_process_without_model_returns_valid_dict(brain_no_model: AGNNCore):
    """``process()`` must return a valid dict even when no model is loaded.

    Setup: AGNNCore with ``model_path=None`` and ``QWEN_PATH`` unset.
    After learning one fact, ``process()`` must:
        - return a dict (not raise)
        - contain ``answer``, ``chain``, ``chain_confidence`` keys
        - ``answer`` must be a non-empty string (the graceful fallback
          includes the chain snippet)
    """
    brain_no_model.learn("q?", "wrong", "Socrates is a human")
    result = brain_no_model.process("human")

    assert isinstance(result, dict), "process() must return a dict"
    assert "answer" in result
    assert "chain" in result
    assert "chain_confidence" in result
    assert isinstance(result["answer"], str)
    assert result["answer"], "answer must be non-empty (graceful fallback)"


def test_process_without_model_does_not_crash_on_multiple_calls(brain_no_model):
    """Multiple ``process()`` calls with no model must not crash."""
    brain_no_model.learn("q?", "wrong", "Socrates is a human")
    for _ in range(5):
        result = brain_no_model.process("human")
        assert isinstance(result, dict)
        assert "answer" in result


def test_process_without_model_answer_contains_chain(brain_no_model):
    """The graceful-fallback answer must include the chain text."""
    brain_no_model.learn("q?", "wrong", "Socrates is a human")
    result = brain_no_model.process("human")
    # The fallback string is "[Graph context: {chain}] (model not loaded)"
    # so the answer should mention "Graph context" and "model not loaded".
    assert "Graph context" in result["answer"] or "model not loaded" in result["answer"], (
        "fallback answer should indicate model is not loaded"
    )


# ======================================================================
# Requirement 2: _load_model() is called exactly once (lazy)
# ======================================================================


def test_load_model_called_only_once(monkeypatch):
    """``_load_model()`` must be called at most once even across
    multiple ``_articulate()`` calls.

    Setup: mock ``transformers`` with ``AutoModelForCausalLM`` that
    returns a fake model. Set ``model_path`` so loading is attempted.
    Call ``_articulate()`` three times. Assert
    ``AutoModelForCausalLM.from_pretrained`` was called exactly once.
    """
    call_count = {"model": 0, "tokenizer": 0}

    class _CountingTokenizer(_FakeTokenizer):
        def __init__(self):
            super().__init__()
            call_count["tokenizer"] += 1

    class _CountingModel(_FakeModel):
        def __init__(self):
            super().__init__()
            call_count["model"] += 1

    _install_fake_transformers(
        monkeypatch,
        tokenizer_cls=_CountingTokenizer,
        model_cls=_CountingModel,
    )

    core = AGNNCore(model_path="/fake/qwen/path")
    # Ensure the EngramComplex is available (needed for learn/process).
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    # Call _articulate three times.
    core._articulate("question 1", "chain 1")
    core._articulate("question 2", "chain 2")
    core._articulate("question 3", "chain 3")

    assert call_count["model"] == 1, (
        f"AutoModelForCausalLM.from_pretrained must be called exactly once "
        f"(lazy load), got {call_count['model']} calls"
    )
    assert call_count["tokenizer"] == 1, (
        f"AutoTokenizer.from_pretrained must be called exactly once, "
        f"got {call_count['tokenizer']} calls"
    )


def test_load_model_not_called_when_no_path(monkeypatch):
    """``_load_model()`` must NOT be called when no model path is available."""
    monkeypatch.delenv("QWEN_PATH", raising=False)
    core = AGNNCore(model_path=None)

    load_called = {"n": 0}
    original_load = core._load_model

    def counting_load():
        load_called["n"] += 1
        return original_load()

    core._load_model = counting_load

    core._articulate("q", "chain")
    assert load_called["n"] == 0, (
        "_load_model() must not be called when no path is available"
    )


def test_load_model_skips_after_failed_attempt(monkeypatch):
    """After a failed load attempt, ``_load_model`` is not re-called.

    This prevents repeated expensive ``from_pretrained`` calls when the
    model is genuinely unavailable (e.g. bad path).
    """
    # Install fake transformers that RAISE on from_pretrained.
    fake_module = types.ModuleType("transformers")

    class _FailingTokenizer:
        @staticmethod
        def from_pretrained(path, *args, **kwargs):
            raise RuntimeError("fake load failure")

    class _FailingModel:
        @staticmethod
        def from_pretrained(path, *args, **kwargs):
            raise RuntimeError("fake load failure")

    fake_module.AutoTokenizer = _FailingTokenizer
    fake_module.AutoModelForCausalLM = _FailingModel
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    call_count = {"n": 0}
    original = AGNNCore._load_model

    def counting_load(self):
        call_count["n"] += 1
        return original(self)

    monkeypatch.setattr(AGNNCore, "_load_model", counting_load)

    core = AGNNCore(model_path="/fake/path")
    core._articulate("q1", "chain1")
    core._articulate("q2", "chain2")
    core._articulate("q3", "chain3")

    assert call_count["n"] == 1, (
        f"_load_model must be called at most once even on failure, "
        f"got {call_count['n']} calls"
    )
    assert core.model is None, "model must remain None after failed load"


# ======================================================================
# Requirement 3: _generate() with mock tokenizer/model returns string
# ======================================================================


def test_generate_returns_string_with_mock_model(monkeypatch):
    """``_generate()`` with mocked tokenizer + model must return a string."""
    _install_fake_torch_only(monkeypatch)
    core = AGNNCore(model_path=None)
    # Manually inject mock tokenizer + model (skip lazy load).
    core._tokenizer = _FakeTokenizer(decode_output="hello world")
    core.model = _FakeModel(generated_tokens=["a", "b", "c"])

    result = core._generate("test prompt")
    assert isinstance(result, str), (
        f"_generate() must return a string, got {type(result).__name__}"
    )


def test_generate_returns_expected_decoded_string(monkeypatch):
    """``_generate()`` must return the tokenizer's decoded output."""
    _install_fake_torch_only(monkeypatch)
    expected = "the generated answer text"
    core = AGNNCore(model_path=None)
    core._tokenizer = _FakeTokenizer(decode_output=expected)
    core.model = _FakeModel()

    result = core._generate("any prompt")
    assert result == expected, (
        f"_generate() must return the decoded output {expected!r}, "
        f"got {result!r}"
    )


def test_generate_respects_max_new_tokens(monkeypatch):
    """``_generate(prompt, max_new_tokens=N)`` must pass N to model.generate."""
    _install_fake_torch_only(monkeypatch)
    core = AGNNCore(model_path=None)
    core._tokenizer = _FakeTokenizer(decode_output="out")
    fake_model = _FakeModel()
    core.model = fake_model

    core._generate("prompt", max_new_tokens=128)
    assert fake_model.generate_calls, "model.generate must be called"
    last_call = fake_model.generate_calls[-1]
    assert last_call.get("max_new_tokens") == 128, (
        f"max_new_tokens=128 must be forwarded to model.generate, "
        f"got {last_call.get('max_new_tokens')}"
    )


def test_generate_falls_back_to_prompt_on_exception(monkeypatch):
    """``_generate()`` must return the prompt itself on any exception."""
    _install_fake_torch_only(monkeypatch)
    core = AGNNCore(model_path=None)
    # A tokenizer that raises on __call__.
    class _BoomTokenizer:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("boom")

        def decode(self, *args, **kwargs):
            return ""

    core._tokenizer = _BoomTokenizer()
    core.model = _FakeModel()

    result = core._generate("fallback prompt")
    assert result == "fallback prompt", (
        f"_generate() must fall back to returning the prompt on error, "
        f"got {result!r}"
    )


# ======================================================================
# Requirement 4: prompt template — chain appears in the prompt
# ======================================================================


def test_prompt_template_includes_chain(monkeypatch):
    """The chain text must appear in the prompt passed to ``_generate()``."""
    _install_fake_transformers(monkeypatch)

    core = AGNNCore(model_path="/fake/qwen")
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    captured_prompts = []
    original_generate = core._generate

    def capturing_generate(prompt, max_new_tokens=256):
        captured_prompts.append(prompt)
        return "mock answer"

    core._generate = capturing_generate

    chain_text = "Socrates is a human -> every human is mortal"
    core._articulate("Is Socrates mortal?", chain_text)

    assert len(captured_prompts) == 1, "generate must be called exactly once"
    prompt = captured_prompts[0]
    assert chain_text in prompt, (
        f"chain text must appear in the prompt, got prompt:\n{prompt}"
    )


def test_prompt_template_has_correct_structure(monkeypatch):
    """The prompt must follow the spec template:
    ``[Knowledge Graph Context]\\n{chain}\\nQ: {question}\\nA:``.
    """
    _install_fake_transformers(monkeypatch)

    core = AGNNCore(model_path="/fake/qwen")
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    captured = []
    core._generate = lambda prompt, max_new_tokens=256: captured.append(prompt) or ""

    core._articulate("what is human?", "mortal creature")

    assert len(captured) == 1
    prompt = captured[0]
    assert prompt.startswith("[Knowledge Graph Context]\n"), (
        f"prompt must start with '[Knowledge Graph Context]\\n', got:\n{prompt!r}"
    )
    assert "Q: what is human?\nA:" in prompt, (
        f"prompt must contain 'Q: {{question}}\\nA:', got:\n{prompt!r}"
    )
    assert "mortal creature" in prompt, (
        f"chain must appear between header and Q:, got:\n{prompt!r}"
    )


# ======================================================================
# Requirement 5: chain > 800 chars is truncated
# ======================================================================


def test_chain_truncated_when_over_800_chars(monkeypatch):
    """Chains longer than 800 chars must be truncated to 800 chars in the prompt."""
    _install_fake_transformers(monkeypatch)

    core = AGNNCore(model_path="/fake/qwen")
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    captured = []
    core._generate = lambda prompt, max_new_tokens=256: captured.append(prompt) or ""

    # Build a chain that's 1000 chars long.
    long_chain = "A" * 1000
    core._articulate("question?", long_chain)

    assert len(captured) == 1
    prompt = captured[0]
    # The chain portion (between header and "Q:") must be exactly 800 chars.
    # Prompt structure: "[Knowledge Graph Context]\n{chain}\nQ: ...\nA:"
    prefix = "[Knowledge Graph Context]\n"
    assert prompt.startswith(prefix)
    rest = prompt[len(prefix):]
    # Find the chain portion: everything up to the "\nQ: " marker.
    q_marker = "\nQ: "
    q_idx = rest.find(q_marker)
    assert q_idx != -1, "prompt must contain '\\nQ: ' marker"
    chain_in_prompt = rest[:q_idx]
    assert len(chain_in_prompt) == 800, (
        f"chain must be truncated to exactly 800 chars, got {len(chain_in_prompt)}"
    )
    assert chain_in_prompt == "A" * 800, (
        "truncated chain must be the first 800 chars of the original"
    )


def test_chain_not_truncated_when_under_800_chars(monkeypatch):
    """Chains ≤ 800 chars must NOT be truncated."""
    _install_fake_transformers(monkeypatch)

    core = AGNNCore(model_path="/fake/qwen")
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    captured = []
    core._generate = lambda prompt, max_new_tokens=256: captured.append(prompt) or ""

    short_chain = "B" * 500  # under the 800-char limit
    core._articulate("q?", short_chain)

    assert len(captured) == 1
    prompt = captured[0]
    assert short_chain in prompt, (
        "chains ≤ 800 chars must appear in full in the prompt"
    )


def test_chain_exactly_800_chars_not_truncated(monkeypatch):
    """A chain of exactly 800 chars must NOT be truncated (boundary case)."""
    _install_fake_transformers(monkeypatch)

    core = AGNNCore(model_path="/fake/qwen")
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    captured = []
    core._generate = lambda prompt, max_new_tokens=256: captured.append(prompt) or ""

    boundary_chain = "C" * 800
    core._articulate("q?", boundary_chain)

    assert len(captured) == 1
    prompt = captured[0]
    prefix = "[Knowledge Graph Context]\n"
    rest = prompt[len(prefix):]
    q_idx = rest.find("\nQ: ")
    chain_in_prompt = rest[:q_idx]
    assert len(chain_in_prompt) == 800, (
        f"chain of exactly 800 chars must not be truncated, "
        f"got {len(chain_in_prompt)}"
    )


# ======================================================================
# Bonus: QWEN_PATH env var resolution
# ======================================================================


def test_qwen_path_env_var_triggers_load(monkeypatch):
    """When ``model_path=None`` but ``QWEN_PATH`` is set, loading is attempted."""
    monkeypatch.setenv("QWEN_PATH", "/fake/qwen/from/env")
    _install_fake_transformers(monkeypatch)

    core = AGNNCore(model_path=None)
    if core.graph is None:
        pytest.skip("EngramComplex (self-ai/src/agnn) not available")

    # _articulate should trigger _load_model which reads QWEN_PATH.
    core._articulate("q", "chain")

    assert core.model is not None, (
        "model must be loaded when QWEN_PATH env var is set"
    )
    assert core._tokenizer is not None


def test_explicit_model_path_takes_precedence_over_qwen_path(monkeypatch):
    """``model_path`` constructor arg takes precedence over ``QWEN_PATH``."""
    monkeypatch.setenv("QWEN_PATH", "/from/env/var")
    captured_paths = []

    fake_module = types.ModuleType("transformers")

    class _PathCapturingTokenizer:
        @staticmethod
        def from_pretrained(path, *args, **kwargs):
            captured_paths.append(("tokenizer", path))
            return _FakeTokenizer()

    class _PathCapturingModel:
        @staticmethod
        def from_pretrained(path, *args, **kwargs):
            captured_paths.append(("model", path))
            return _FakeModel()

    fake_module.AutoTokenizer = _PathCapturingTokenizer
    fake_module.AutoModelForCausalLM = _PathCapturingModel
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    core = AGNNCore(model_path="/explicit/path")
    core._articulate("q", "chain")

    # Both tokenizer and model should have been loaded with the explicit path.
    assert any(p == "/explicit/path" for _, p in captured_paths), (
        f"explicit model_path must be used, got paths: {captured_paths}"
    )
