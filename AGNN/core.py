"""
AGNN - Aphantic Graph Neural Network
Rebuilding human brain with code. Every name = real neuroanatomical term.

Architecture:
- Hippocampus (fast encoding, episomic memory)
- Neocortex (slow learning, semantic memory)
- Limbic system (confidence modulation)
- Spiking dynamics (neural replay)
- Deductive reasoning (BA 44)

Vision: Model kecil yang semakin pintar hari ke hari, infinite expand without retraining.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Bootstrap sys.path BEFORE any sibling-package import.
# ----------------------------------------------------------------------
#
# ``core.py`` may be loaded as either ``core`` (when AGNN/ is on
# sys.path, the convention used by every other module in AGNN/) or
# ``AGNN.core`` (when the repo root is on sys.path, the convention a
# user follows when they write ``from AGNN.core import AGNNCore``).
# Either way, sibling-package imports like ``from neocortex... import
# ...`` must work — and they only work if AGNN/ itself is on sys.path
# at module-load time.
#
# This bootstrap MUST run BEFORE the try/except blocks below that
# import ``neocortex.bootstrap_classifier`` and
# ``neocortex.aphantasic_chain_formatter``. Pre-fix, those imports
# ran first and silently swallowed ``ModuleNotFoundError`` when AGNN/
# was not yet on sys.path — the cluster learner then quietly
# disabled itself and AGNNCore fell back to the legacy
# SemanticRoleClassifier with zero warning (issue #94).
_AGNP_ROOT = os.path.dirname(os.path.abspath(__file__))
if _AGNP_ROOT not in sys.path:
    sys.path.insert(0, _AGNP_ROOT)


# Phase 1 (Aphantasic Node Representation): import the two new helper
# modules at module-load time so ``AGNNCore.__init__`` can construct
# them eagerly. We wrap the imports in try/except so a missing
# dependency (e.g. a future rename) doesn't break AGNNCore construction
# — Phase 1 falls back to the pre-Phase-1 surface-form-only chain if
# either helper is unavailable. The fallback is graceful: the
# articulate prompt loses its DEFINISI/RELASI sections but the
# KONSEP + Q/A structure (and all 197 existing tests) keep working.
try:
    from neocortex.aphantasic_chain_formatter import AphantasicChainFormatter
    from neocortex.definition_extractor import DefinitionExtractor
    _PHASE1_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    # The sys.path bootstrap above should make this import succeed in
    # any supported load path. If it still fails, warn loudly so the
    # user sees the silent-degradation cause instead of debugging a
    # half-functional AGNNCore (issue #94).
    warnings.warn(
        f"AGNNCore Phase 1 helpers unavailable "
        f"({type(exc).__name__}: {exc}). Falling back to the "
        f"pre-Phase-1 surface-form-only chain. This may indicate a "
        f"broken install.",
        RuntimeWarning,
        stacklevel=2,
    )
    AphantasicChainFormatter = None  # type: ignore[assignment,misc]
    DefinitionExtractor = None  # type: ignore[assignment,misc]
    _PHASE1_AVAILABLE = False


# Cluster learner integration: load the labelled
# PositionalClusterLearner state at module load time so AGNNCore can
# use it as a drop-in role_classifier replacement. The state file is
# generated once by ``neocortex.bootstrap_classifier.save_default_state``
# and committed to the repo so every environment boots with the same
# cluster mapping. When the file is missing or fails to load, we fall
# back to None - AGNNCore then uses the legacy SemanticRoleClassifier
# (no behaviour change).
#
# Import is wrapped in try/except so this module loads cleanly even if
# the bootstrap_classifier or positional_cluster_learner modules are
# unavailable (e.g. during partial unit testing of just AGNN/core.py).
try:
    from neocortex.bootstrap_classifier import (
        DEFAULT_STATE_PATH as _CLUSTER_LEARNER_STATE_PATH,
        load_default_state as _load_cluster_learner_state,
    )
    _CLUSTER_LEARNER_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    # Same rationale as the Phase 1 warn above: the sys.path bootstrap
    # should make this import succeed in any supported load path. If
    # it doesn't, warn so the user knows why their AGNNCore isn't
    # using the labelled PositionalClusterLearner (issue #94).
    warnings.warn(
        f"AGNNCore cluster learner unavailable "
        f"({type(exc).__name__}: {exc}). Falling back to the legacy "
        f"SemanticRoleClassifier. This may indicate a broken install "
        f"or a missing/corrupt cluster_learner_state.json.",
        RuntimeWarning,
        stacklevel=2,
    )
    _CLUSTER_LEARNER_AVAILABLE = False
    _CLUSTER_LEARNER_STATE_PATH = None  # type: ignore[assignment]
    _load_cluster_learner_state = None  # type: ignore[assignment]


class AGNNCore:
    """
    Artificial Graph Neural Network - rebuilding human brain from scratch.

    Every method, class, folder = real neuroanatomical term.

    Public methods are wired to the real neuroanatomical components that
    already exist in the AGNN package. When an underlying component is
    not yet implemented (raises ``NotImplementedError`` or any other
    exception), the corresponding AGNNCore method catches the error and
    returns a graceful fallback so callers never crash.
    """

    # How much a single reinforce()/penalize() call nudges confidence.
    _REINFORCE_DELTA = 0.1

    # Three-factor learning (R-STDP / neoHebbian) — see
    # AGNN/docs/research-spiking-neural-networks.md §1.1-1.3 + §4 + B1.
    # Per-edge eligibility trace `e_ij` is incremented each time an edge
    # is read during _build_semesomes_from_graph() (the canonical
    # edge-traversal site invoked from process()), and decays
    # exponentially per call so older traversals fade. When
    # reinforce()/penalize() fires, the global modulatory signal `M`
    # (= ±_REINFORCE_DELTA) is distributed across edges proportionally
    # to their trace — `Δw_ij = M · e_ij / Σe` — implementing the
    # textbook three-factor rule `Δw = M · e`. When the trace dict is
    # empty (cold start, no recent traversal), reinforce/penalize falls
    # back to the legacy uniform node-confidence bump so behavior is
    # bit-identical for callers that never traverse.
    _ELIGIBILITY_DECAY: float = 0.9
    _ELIGIBILITY_INCREMENT: float = 1.0
    _ELIGIBILITY_EPSILON: float = 1e-6

    # Phase 0 (Aphantasic Articulation Anchor): default system message
    # sent to Qwen3 alongside the user prompt in ``_articulate()``.
    #
    # This is the root-cause fix for the ``api``/``API`` and
    # ``air``/``air`` disambiguation failures reported during testing.
    # The previous implementation only sent a single user message
    # (``[Knowledge Graph Context]\n{chain}\nQ: {question}\nA:``) with
    # no system message at all. Without a system message anchoring the
    # language and the role of the context, Qwen3-0.6B fell back to its
    # English tech-corpus prior and read ``api`` as "Application
    # Programming Interface" and ``air`` as the English noun for the
    # atmospheric gas — even when the surrounding chain was Indonesian.
    #
    # The system message below anchors three things:
    #   1. Role: the model narrates from a knowledge graph, not from
    #      its own parametric memory — this suppresses hallucination
    #      and the ``Q: ... A: Q: ... A:`` repetition loop.
    #   2. Language: tokens in the ``[Knowledge Graph Context]`` block
    #      are Bahasa Indonesia unless explicitly tagged otherwise.
    #      This is what disambiguates ``api`` (ID: fire) from ``API``
    #      (EN: programming interface) and ``air`` (ID: water) from
    #      ``air`` (EN: atmospheric gas).
    #   3. Output contract: every claim must trace back to a node in
    #      the context; no invented facts.
    #
    # The message is deliberately short (well under 200 tokens) so it
    # fits comfortably in Qwen3-0.6B's ~8K context window alongside
    # the truncated chain (max 800 chars) and the Q/A pair. It is also
    # phrased so that ``/no_think`` (the default Qwen3 mode for
    # non-reasoning tasks like articulation) keeps the model in
    # direct-generation mode rather than Long-CoT.
    #
    # Callers can override this per-instance by setting
    # ``core._system_message = "..."`` after construction. ``None``
    # disables the system message entirely (restoring the pre-Phase-0
    # single-message behaviour) — useful for ablation tests.
    _ARTICULATE_SYSTEM_MESSAGE: str = (
        "Kamu adalah asisten penalaran berbasis knowledge graph. "
        "Konteks di bawah adalah data graf pengetahuan, bukan teks "
        "naratif bebas. Token dalam blok [Knowledge Graph Context] "
        "adalah istilah dalam Bahasa Indonesia kecuali jika ditandai "
        "lain. Jangan menginterpretasikan ulang token sebagai kata "
        "Inggris (misalnya 'api' = fenomena pembakaran, bukan "
        "'Application Programming Interface'; 'air' = cairan untuk "
        "minum, bukan gas di atmosfer). Sebutkan hanya fakta yang "
        "ada di konteks. Jawab dalam bahasa yang sama dengan "
        "pertanyaan."
    )

    # Phase 1 (Aphantasic Node Representation): cumulative confidence
    # delta above which a node's amodal definition is invalidated and
    # re-generated on the next articulate call. The user-confirmed
    # value is 0.3 — reasoning: "reinforce berarti konsep berkembang,
    # definisi lama mungkin sudah terlalu sempit." Three reinforces
    # (+0.3) cross this threshold; two reinforces (+0.2) do not. The
    # threshold is a class constant so tests can read it without
    # hard-coding the magic number.
    _DEFINITION_INVALIDATE_THRESHOLD: float = 0.3

    def __init__(
        self,
        model_path: Optional[str] = None,
        classifier_persist_path: Optional[str] = None,
        use_cluster_learner: bool = True,
        cluster_learner_state_path: Optional[str] = None,
        t_norm: str = "product",
    ):
        """
        Initialize brain-inspired memory system.

        Args:
            model_path: Path to small LLM (e.g. Qwen3-0.6B) used for
                articulation. May be ``None`` - in that case
                :meth:`_articulate` returns the chain string verbatim
                with a small note. Model loading is lazy: we only
                attempt to load the model on the first
                :meth:`_articulate` call.
            classifier_persist_path: Optional path to a JSON file used
                to persist the :class:`SemanticRoleClassifier`'s
                frequency table across process restarts. When the file
                exists at init time, its contents are loaded; after
                every confident ``classify()`` call (triggered by
                ``learn()``), the table is re-saved atomically. ``None``
                (the default) disables all persistence - behaviour is
                identical to the pre-persistence AGNNCore. The path is
                propagated down through ``TrisynapticCircuit`` to the
                ``SemanticRoleClassifier`` instance it owns.
            use_cluster_learner: When True (the default), AGNNCore
                loads the labelled :class:`PositionalClusterLearner`
                from ``cluster_learner_state_path`` (or the default
                path ``AGNN/data/cluster_learner_state.json``) and
                passes it to :class:`TrisynapticCircuit` as the
                ``role_classifier``. The cluster learner is a drop-in
                replacement for :class:`SemanticRoleClassifier` - it
                implements the same ``classify(text) -> RelationType``
                and ``spo(text) -> SPO`` contract - but its
                classifications come from the zero-bias positional
                clusters discovered on the pretrain corpus, not from
                hand-authored seed keywords. When the state file is
                missing or fails to load, AGNNCore falls back to the
                legacy :class:`SemanticRoleClassifier` (no behaviour
                change). When False, the cluster learner is bypassed
                entirely and AGNNCore behaves identically to the
                pre-cluster-learner version.
            cluster_learner_state_path: Optional path to a JSON state
                file for the cluster learner. ``None`` (the default)
                uses ``AGNN/data/cluster_learner_state.json`` (the
                file committed to the repo by
                ``neocortex.bootstrap_classifier.save_default_state``).
                Only consulted when ``use_cluster_learner=True``.
            t_norm: Which fuzzy-logic t-norm to use when
                :class:`InferiorFrontalGyrus` (BA 44) composes two
                premise weights into one inferred weight during
                deductive reasoning. One of:

                    - ``"product"``      (default)  ->  T(a, b) = a * b
                    - ``"lukasiewicz"``             ->  T(a, b) = max(0, a + b - 1)
                    - ``"godel"``                   ->  T(a, b) = min(a, b)

                The default ``"product"`` reproduces the pre-refactor
                BA 44 arithmetic exactly (e.g. ``0.7 * 0.7 = 0.49``,
                ``0.6 * 0.6 = 0.36``, ``1.0 * 1.0 = 1.0``) and is the
                only value that keeps every existing test's
                weight-related assertion unchanged. The other two
                options make BA 44 a *conscious* fuzzy-logic engine
                whose conjunction operator is configurable, as
                recommended in
                ``AGNN/docs/research-neuro-symbolic-reasoning.md`` §4 / §B1.

                **Research surface.** This parameter is intentionally
                exposed at the AGNNCore level (rather than requiring
                callers to reach into ``core.deductive.t_norm``
                directly) so that A/B t-norm experiments can be run
                from any AGNNCore entry point (the
                ``__main__.py`` CLI, ``e2e_test.py``, ad-hoc scripts).
                The parameter is plumbed straight into
                :class:`InferiorFrontalGyrus`'s constructor; if BA 44
                is unavailable (``_safe_init`` returns ``None``), the
                parameter is recorded on ``self._t_norm_requested``
                but has no effect.

                Note: the ``CAUSAL_DIFFERENTIAL_CONFLICT`` rule
                resolves its two premises via arithmetic mean, which
                is *not* a t-norm and is therefore unaffected by this
                parameter. Likewise, ``DIFFERENTIAL_INVERSION`` is a
                unary inversion and does not combine weights.
        """
        # Component wiring. Each component is wrapped in try/except so
        # AGNNCore can still be constructed even if a sibling component
        # raises on init (e.g. NotImplementedError skeleton).
        self.graph = self._safe_init("engrams.engram_complex", "EngramComplex")

        # Cluster learner integration: when use_cluster_learner=True,
        # try to load the labelled PositionalClusterLearner from the
        # state file. On any failure (file missing, corrupt, module
        # unavailable), fall back to None - TrisynapticCircuit will
        # then construct a fresh SemanticRoleClassifier (legacy
        # behaviour). This is the "graceful degradation" contract: a
        # missing state file never crashes AGNNCore.
        cluster_learner = None
        if use_cluster_learner and _CLUSTER_LEARNER_AVAILABLE:
            try:
                state_path = (
                    cluster_learner_state_path
                    or _CLUSTER_LEARNER_STATE_PATH
                )
                if state_path is not None:
                    cluster_learner = _load_cluster_learner_state(state_path)
            except Exception:
                cluster_learner = None

        # TrisynapticCircuit gets the classifier_persist_path so the
        # SemanticRoleClassifier it constructs can load + auto-save
        # its frequency table. We only pass it when we have a graph
        # to wire the circuit with - if the graph init failed, the
        # circuit kwargs stay empty (matching the pre-persistence
        # behaviour) so the test suite's "EngramComplex not available"
        # skip path keeps working.
        #
        # When cluster_learner is not None, we pass it as
        # role_classifier - TrisynapticCircuit will use it as the
        # primary edge-type inferrer (it has the same classify()/spo()
        # contract as SemanticRoleClassifier). In that case we DO NOT
        # also pass classifier_persist_path, because TrisynapticCircuit
        # only consults that path when constructing a fresh
        # SemanticRoleClassifier (which it won't do when given a
        # pre-built role_classifier).
        trisynaptic_kwargs: Dict[str, Any] = {}
        if self.graph is not None:
            trisynaptic_kwargs["engram_complex"] = self.graph
        if cluster_learner is not None:
            trisynaptic_kwargs["role_classifier"] = cluster_learner
        elif classifier_persist_path is not None:
            trisynaptic_kwargs["classifier_persist_path"] = (
                classifier_persist_path
            )
        self.trisynaptic = self._safe_init(
            "circuits.trisynaptic_circuit", "TrisynapticCircuit",
            kwargs=trisynaptic_kwargs,
        )
        self.papez = self._safe_init("circuits.papez_circuit", "PapezCircuit")
        # BA 44 (InferiorFrontalGyrus) takes a `t_norm` kwarg selecting
        # which fuzzy-logic conjunction to use when composing premise
        # weights. We plumb the AGNNCore-level `t_norm` parameter
        # straight through. The default "product" reproduces the
        # pre-refactor arithmetic bit-for-bit, so existing tests are
        # unaffected; the other two options ("lukasiewicz", "godel")
        # are research surfaces for A/B t-norm experiments (see
        # AGNN/docs/research-neuro-symbolic-reasoning.md §4 / §B1).
        #
        # Validate up-front: `_safe_init` swallows construction errors
        # (returns None), so an invalid `t_norm` value would otherwise
        # leave `self.deductive` silently None with no diagnostic. We
        # raise explicitly here so a typo surfaces at construction
        # time, not at the first `process()` call.
        _VALID_T_NORMS = ("product", "lukasiewicz", "godel")
        if t_norm not in _VALID_T_NORMS:
            raise ValueError(
                f"Unknown t_norm {t_norm!r}. "
                f"Expected one of: {list(_VALID_T_NORMS)}"
            )
        deductive_kwargs: Dict[str, Any] = {"t_norm": t_norm}
        self.deductive = self._safe_init(
            "neocortex.inferior_frontal_gyrus", "InferiorFrontalGyrus",
            kwargs=deductive_kwargs,
        )
        self.consolidation = self._safe_init(
            "plasticity.systems_consolidation", "SystemsConsolidation",
        )
        self.cingulate = self._safe_init(
            "limbic_system.cingulate_gyrus", "CingulateGyrus",
        )

        # Model loading is lazy. ``_model_load_attempted`` ensures we
        # only try to load once even if the first attempt fails, so
        # repeated ``process()`` calls don't re-attempt the (expensive)
        # HF from_pretrained call.
        self.model = None
        self._tokenizer = None
        self._model_path = model_path
        self._model_load_attempted = False

        # Episome registry - tracks every Episome returned by learn()
        # so introspect / reinforce / penalize can find them by id
        # without depending on the (graph-only) EngramComplex API.
        self._episomes: List[Any] = []

        # Remember the requested classifier persist path so callers
        # can inspect it (useful for tests + debugging). When None,
        # no persistence is active.
        self._classifier_persist_path = classifier_persist_path

        # Cluster learner integration: record which role_classifier
        # TrisynapticCircuit is actually using. Tests check this to
        # verify the cluster learner was loaded (or not). The
        # ``_cluster_learner`` attribute is the loaded
        # PositionalClusterLearner instance (or None when the legacy
        # SemanticRoleClassifier is in use). The
        # ``_use_cluster_learner_requested`` attribute records the
        # caller's intent - useful for the test that verifies
        # ``use_cluster_learner=False`` falls back identically.
        self._cluster_learner = cluster_learner
        self._use_cluster_learner_requested = use_cluster_learner

        # Record the caller's requested t_norm (post-validation, so
        # always one of "product"/"lukasiewicz"/"godel"). Useful for
        # test introspection — when `self.deductive` is None (BA 44
        # module unavailable), this attribute still surfaces what the
        # caller asked for. The actual t-norm function lives on
        # `self.deductive.t_norm` (a string, matching this attribute)
        # and `self.deductive._tnorm_fn` (the resolved callable) when
        # BA 44 is available.
        self._t_norm_requested = t_norm

        # Phase 0: per-instance override of the articulate system
        # message. ``None`` (the default) means "use the class-level
        # ``_ARTICULATE_SYSTEM_MESSAGE``". Set to a non-None string to
        # override, or set to an empty string ``""`` to explicitly
        # disable the system message (restoring pre-Phase-0 behaviour
        # for ablation). See ``_articulate`` for the resolution order.
        self._system_message: Optional[str] = None

        # Phase 0: the system message stashed by ``_articulate`` for
        # the real ``_generate`` to pick up. Initialised to ``None``
        # so the first ``_generate`` call (e.g. from a test that
        # bypasses ``_articulate``) behaves like pre-Phase-0.
        # ``_articulate`` overwrites this on every call.
        self._active_system_message: Optional[str] = None

        # Phase 1 (Aphantasic Node Representation): the definition
        # extractor owns the cross-Episome text-hash cache. Lazy
        # generation happens in ``_articulate`` — we don't generate
        # definitions at ``learn()`` time because (a) the user may
        # learn many facts in a burst and blocking 0.5–2 s per node
        # is poor UX, and (b) most nodes are never articulated. The
        # extractor borrows ``self.model`` + ``self._tokenizer`` on
        # each call, so it picks up the lazy-loaded model
        # automatically. When Phase 1 imports failed at module load
        # time (``_PHASE1_AVAILABLE == False``), ``_definition_extractor``
        # is None and ``_articulate`` falls back to the pre-Phase-1
        # surface-form-only chain.
        self._definition_extractor = (
            DefinitionExtractor() if _PHASE1_AVAILABLE else None
        )

        # Phase 1: the chain formatter is stateless, but we hold one
        # instance per AGNNCore so the rendering limits (max surface
        # chars, max anchors per node) can be configured per-brain in
        # the future without touching the formatter's constructor.
        self._chain_formatter = (
            AphantasicChainFormatter() if _PHASE1_AVAILABLE else None
        )

        # Phase 1: per-node cumulative reinforce delta tracker. Used
        # by ``reinforce()`` to decide when to invalidate the cached
        # amodal definition. ``{episome_id: cumulative_delta}`` where
        # cumulative_delta is the *positive* sum of all reinforce()
        # calls on that node since the last definition invalidation
        # (penalize() does not subtract — only growth triggers
        # re-generation, mirroring the user's reasoning: "reinforce
        # berarti konsep berkembang"). When the delta crosses
        # ``_DEFINITION_INVALIDATE_THRESHOLD``, ``reinforce()`` calls
        # ``_definition_extractor.invalidate(text)``, resets the
        # delta to 0, and sets ``episome.definition_dirty = True`` so
        # the next ``_articulate`` re-generates the definition.
        self._reinforce_deltas: Dict[Any, float] = {}

        # Three-factor learning: per-edge eligibility trace. Keyed by
        # ``(source_id, target_id, str(relation_type))`` — the same
        # triple _build_semesomes_from_graph uses for dedup. Value is
        # a non-negative float; decays exponentially per process() call
        # and is consumed (renormalised to ±_REINFORCE_DELTA) when
        # reinforce()/penalize() fires.
        self._eligibility: Dict[Any, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_init(module_name: str, class_name: str,
                   kwargs: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Import + instantiate ``module.Class(**kwargs)`` defensively.

        Returns ``None`` if the import or construction raises any
        exception (including NotImplementedError). AGNNCore callers
        must check for ``None`` before using the component.
        """
        try:
            import importlib
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(**(kwargs or {}))
        except Exception:
            return None

    # Maximum chars of the reasoning chain to include in the prompt.
    # Chains longer than this are truncated to fit Qwen3-0.6B's context
    # window (~8K tokens) while still leaving room for the question + answer.
    _CHAIN_MAX_CHARS = 800

    def _load_model(self) -> None:
        """Lazy-load Qwen3-0.6B via HuggingFace transformers.

        Model path resolution order:
            1. ``self._model_path`` (explicit constructor arg)
            2. ``QWEN_PATH`` environment variable
        If neither is set, skip loading entirely (graceful fallback).

        On success: ``self.model`` = ``AutoModelForCausalLM`` instance,
                    ``self._tokenizer`` = ``AutoTokenizer`` instance.
        On any failure (transformers not installed, path missing, OOM,
        etc.): ``self.model`` = ``None``, ``self._tokenizer`` = ``None``
        so callers fall back to the no-model path.
        """
        model_path = self._model_path or os.environ.get("QWEN_PATH")
        if not model_path:
            # Neither explicit path nor QWEN_PATH set — skip loading.
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="cpu",
            )
        except Exception:
            # Model unavailable / un-loadable - keep self.model = None.
            self.model = None
            self._tokenizer = None

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        system_message: Optional[str] = None,
    ) -> str:
        """Generate text from ``prompt`` using the loaded Qwen3 model.

        The prompt is wrapped in Qwen3's chat template via
        ``tokenizer.apply_chat_template([{role: "user", content: prompt}])``
        before tokenization. This avoids the ``Q: ...\\nA: Q: ...\\nA: ...``
        repetition loop Qwen3 falls into when fed a bare-text prompt —
        the chat template's ``<|im_start|>user`` / ``<|im_end|>`` /
        ``<|im_start|>assistant`` tokens give the model the structural
        cues it was post-trained on, so generation stops cleanly at
        ``<|im_end|>`` instead of echoing the ``A:`` cue.

        A mild ``repetition_penalty=1.1`` is also passed to
        ``model.generate()`` as a belt-and-suspenders guard against
        any residual repetition tendency on short prompts.

        Phase 0 (Aphantasic Articulation Anchor):
            When a system message is supplied — either via the
            ``system_message`` keyword argument **or** via the
            ``self._active_system_message`` instance attribute set by
            ``_articulate()`` — it is prepended to the chat messages
            as a ``{role: "system"}`` entry before the user message.
            This is the root-cause fix for the ``api``/``API`` and
            ``air``/``air`` disambiguation failures: the system message
            anchors the language and the model's role so Qwen3-0.6B
            does not fall back to its English tech-corpus prior when
            reading Indonesian tokens.

            Resolution order for the system message:
              1. Explicit ``system_message`` keyword argument (highest
                 priority — callers can override per-call).
              2. ``self._active_system_message`` instance attribute
                 (set by ``_articulate`` so existing tests that mock
                 ``_generate`` with a 2-arg lambda still work — the
                 mock never sees the keyword).
              3. ``None`` (no system message, pre-Phase-0 behaviour).

            When the resolved system message is ``None`` or empty, the
            call builds a single-message chat, preserving backward
            compatibility with callers that invoke
            ``_generate(prompt)`` or
            ``_generate(prompt, max_new_tokens=N)`` without the new
            keyword or attribute.

        Args:
            prompt: Input prompt (already formatted with the knowledge
                graph context + question).
            max_new_tokens: Maximum new tokens to generate (default 256).
                The prompt tokens are NOT counted toward this limit.
            system_message: Optional system message to prepend to the
                chat. ``None`` (the default) falls back to
                ``self._active_system_message``. Pass an empty string
                ``""`` to explicitly disable the system message even
                if the instance attribute is set (ablation path).

        Returns:
            Generated text (prompt tokens stripped, only the new tokens
            decoded). Falls back to returning the prompt itself on any
            error so the caller still gets a non-empty string.
        """
        # Resolve the system message: explicit kwarg > instance attr > None.
        if system_message is None:
            system_message = getattr(self, "_active_system_message", None)
        try:
            import torch  # noqa: WPS433
            # Wrap the formatted prompt in Qwen3's chat template. The
            # user message carries the full chain context + Q/A pair so
            # the model has the same information as before, just in the
            # structural format Qwen3 was post-trained on.
            #
            # Phase 0: when a system_message is resolved, prepend it as
            # a {role: "system"} entry. This gives Qwen3 the language +
            # role anchor it needs to disambiguate Indonesian tokens
            # like "api"/"air" without falling back to its English
            # tech-corpus prior. When system_message is None/empty,
            # build a single-message chat (pre-Phase-0 behaviour) so
            # existing callers and tests are unaffected.
            if system_message:
                chat_messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
            else:
                chat_messages = [{"role": "user", "content": prompt}]
            chat_text = self._tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(chat_text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                )
            # Slice off the prompt tokens — only decode the new tokens.
            # This avoids re-emitting the prompt in the output.
            input_len = inputs["input_ids"].shape[1]
            return self._tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=True,
            )
        except Exception:
            return prompt

    def _articulate(self, question: str, chain: str) -> str:
        """Turn a (question, chain) pair into a natural-language answer.

        Prompt template (user message; unchanged from pre-Phase-0)::

            [Knowledge Graph Context]
            {chain (truncated to 800 chars)}
            Q: {question}
            A:

        Phase 0 (Aphantasic Articulation Anchor):
            Before calling ``_generate``, this method resolves the
            system message and stashes it on
            ``self._active_system_message`` so the real ``_generate``
            prepends it as a ``{role: "system"}`` chat entry.

            Resolution order:
              1. ``self._system_message`` if it is a non-empty string
                 (per-instance override set by callers).
              2. ``self._system_message == ""`` → empty string is an
                 explicit "disable" signal → no system message is sent
                 (ablation / pre-Phase-0 behaviour).
              3. ``self._system_message is None`` (the default) →
                 fall back to the class-level
                 ``_ARTICULATE_SYSTEM_MESSAGE`` (the Indonesian anchor
                 that fixes the ``api``/``API`` and ``air``/``air``
                 disambiguation failures).

            The system message is stashed on the instance attribute
            (rather than passed as a keyword argument) so existing
            tests that mock ``_generate`` with a 2-arg lambda like
            ``lambda prompt, max_new_tokens=256: ...`` continue to
            work unchanged — the mock never receives the new keyword,
            but the real ``_generate`` reads the attribute.

        Behavior:
            - If a model path is available (``self._model_path`` **or**
              the ``QWEN_PATH`` environment variable), lazy-load the
              model on the first call. Loading is attempted **at most
              once** — repeated calls don't re-trigger the (expensive)
              ``from_pretrained`` call even if the first attempt failed.
            - If a model is available, generate via ``_generate``.
            - Otherwise (no path, load failed, generate failed) return
              the chain (truncated to 800 chars) wrapped in a short
              note so the caller still sees the graph context. This is
              the graceful-fallback path — ``process()`` never crashes
              just because the model is unavailable.
        """
        # Lazy load: only attempt once, and only if a path is available.
        model_path_available = bool(self._model_path) or bool(
            os.environ.get("QWEN_PATH")
        )
        if (
            model_path_available
            and self.model is None
            and not self._model_load_attempted
        ):
            self._model_load_attempted = True
            self._load_model()

        # Phase 0: resolve the system message and stash it for
        # ``_generate`` to pick up. See the docstring above for the
        # resolution order. We always set the attribute (even to None
        # or "") so a previous call's value doesn't leak into this one.
        if self._system_message is not None:
            # Per-instance override: "" means explicit disable, any
            # other non-None string is the override content.
            self._active_system_message = self._system_message
        else:
            # Default: use the class-level Indonesian anchor.
            self._active_system_message = self._ARTICULATE_SYSTEM_MESSAGE

        # Truncate the chain to stay within Qwen3-0.6B's context budget.
        chain_truncated = (chain or "")[: self._CHAIN_MAX_CHARS]

        if self.model is not None and self._tokenizer is not None:
            prompt = (
                f"[Knowledge Graph Context]\n{chain_truncated}\n"
                f"Q: {question}\nA:"
            )
            return self._generate(prompt)

        # Fallback: return chain as the answer (graceful, no crash).
        return f"[Graph context: {chain_truncated}] (model not loaded)"

    # ------------------------------------------------------------------
    # Phase 1 (Aphantasic Node Representation) — 3-layer articulate
    # ------------------------------------------------------------------

    def _articulate_aphantasic(
        self,
        question: str,
        episomes: List[Any],
        semesomes: Optional[List[Any]] = None,
    ) -> str:
        """Articulate with the 3-layer aphantasic chain format.

        This is the Phase 1 counterpart to ``_articulate``. Where
        ``_articulate`` reads a pre-formatted ``chain`` string,
        ``_articulate_aphantasic`` takes the *retrieved episomes* +
        *semesomes* and builds the chain itself via
        ``AphantasicChainFormatter``, after lazily populating each
        episome's ``amodal_definition`` (Layer 2) via
        ``DefinitionExtractor``.

        The 3-layer chain (KONSEP / DEFINISI / RELASI) is then passed
        to ``_articulate`` as its ``chain`` argument, so all the
        Phase 0 machinery (system message anchor, _generate chat
        template, prompt-structure tests) keeps working unchanged.

        Lazy generation contract:
          - For each retrieved episome, if ``amodal_definition`` is
            empty OR ``definition_dirty`` is True, call
            ``DefinitionExtractor.extract(text, model, tokenizer,
            force_refresh=definition_dirty)`` and store the result
            back on the episome.
          - Reset ``definition_dirty = False`` after a successful
            extract (even if the extract returned empty — we don't
            want to retry on every articulate call when the model is
            unavailable).
          - If Phase 1 helpers are unavailable (``_definition_extractor
            is None`` or ``_chain_formatter is None``), fall back to
            the pre-Phase-1 chain format (join episome.text with
            " -> "). This keeps AGNNCore robust to a broken Phase 1
            import.

        Args:
            question: User query.
            episomes: Retrieved Episome instances (typically 3–5 from
                PapezCircuit). When empty, returns an empty string.
            semesomes: Optional Semesome edges between the episomes.
                Currently passed through to the formatter for future
                use (the formatter renders Layer 3 from
                ``Episome.causal_anchors``, not from semesomes).

        Returns:
            The articulated answer string. Falls back to
            ``_articulate(question, fallback_chain)`` when Phase 1
            helpers are unavailable or when the formatter returns an
            empty string (e.g. all episomes have empty text).
        """
        if not episomes:
            return ""

        # Phase 1 unavailable → degrade gracefully to pre-Phase-1 path.
        # We build the chain as the " -> "-joined episome texts (the
        # same format ``process()`` uses for its non-deductive
        # fallback) and delegate to ``_articulate``. This preserves
        # the Phase 0 system message anchor while skipping the
        # DEFINISI/RELASI sections.
        if self._definition_extractor is None or self._chain_formatter is None:
            fallback_chain = " -> ".join(
                getattr(e, "text", str(e))[:50] for e in episomes
            )
            return self._articulate(question, fallback_chain)

        # Lazy-populate Layer 2 (amodal_definition) on each retrieved
        # episome. We do this BEFORE formatting so the formatter sees
        # the freshest definitions. ``force_refresh`` is True when
        # the episome's ``definition_dirty`` flag is set (i.e.
        # ``reinforce()`` crossed the invalidate threshold since the
        # last generation).
        for epi in episomes:
            self._populate_definition(epi)

        # Format the 3-layer chain (KONSEP / DEFINISI / RELASI).
        chain = self._chain_formatter.format(episomes, semesomes or [])
        if not chain:
            # Formatter returned empty (e.g. all episomes had empty
            # text). Fall back to the pre-Phase-1 join so the
            # articulate prompt still gets *something*.
            chain = " -> ".join(
                getattr(e, "text", str(e))[:50] for e in episomes
            )

        # Delegate to ``_articulate`` for the actual generation. This
        # reuses the Phase 0 system message, the prompt-template
        # structure, and the model-loading machinery — Phase 1 only
        # changes the *body* of the [Knowledge Graph Context] block.
        return self._articulate(question, chain)

    def _populate_definition(self, episome: Any) -> None:
        """Lazily populate ``episome.amodal_definition`` if needed.

        Called by ``_articulate_aphantasic`` for each retrieved
        episome before formatting. Three cases:
          1. ``amodal_definition`` is non-empty AND ``definition_dirty``
             is False → cache hit, do nothing.
          2. ``amodal_definition`` is empty → first-time generation,
             call ``DefinitionExtractor.extract(force_refresh=False)``.
          3. ``definition_dirty`` is True → invalidate was requested
             by ``reinforce()``, call ``extract(force_refresh=True)``
             and reset the dirty flag.

        Failure contract: any exception (model not loaded, generation
        error, extractor missing) leaves ``amodal_definition`` as-is
        and resets ``definition_dirty`` to False so we don't retry on
        every articulate call. The formatter will then omit the
        DEFINISI line for this node (graceful degradation).
        """
        if self._definition_extractor is None:
            return
        try:
            dirty = bool(getattr(episome, "definition_dirty", False))
            current = getattr(episome, "amodal_definition", "") or ""
            if current and not dirty:
                return  # cache hit
            force = dirty
            definition = self._definition_extractor.extract(
                text=getattr(episome, "text", ""),
                model=self.model,
                tokenizer=self._tokenizer,
                force_refresh=force,
            )
            if definition:
                episome.amodal_definition = definition
            # Reset the dirty flag regardless of whether the extract
            # succeeded — we don't want to retry on every call if
            # the model is unavailable.
            episome.definition_dirty = False
        except Exception:  # noqa: BLE001
            # Best-effort: leave the episome as-is. The formatter
            # will omit the DEFINISI line if amodal_definition is empty.
            try:
                episome.definition_dirty = False
            except Exception:  # noqa: BLE001
                pass

    def _find_episome(self, episome_id: Any) -> Optional[Any]:
        """Look up an episome by id in the registry (best-effort)."""
        try:
            for e in self._episomes:
                if e.id == episome_id:
                    return e
        except Exception:
            return None
        return None

    def _build_semesomes_from_graph(self, episomes: List[Any]) -> List[Any]:
        """Build ``Semesome`` edges from the typed edges in the wrapped graph.

        For every retrieved episome, look up its outgoing ``TypedEdge`` s
        in the AGNNGraph and convert each one to a ``Semesome`` whose
        ``source`` / ``target`` are the **labels** of the connected
        AGNNNodes (so BA 44's transitivity rules — which match on
        ``e1.target == e2.source`` — fire when the labels chain).

        Only edges whose both endpoints are among the retrieved episomes
        are emitted, so the deduction sees a coherent sub-graph rather
        than the entire memory store.

        The returned list is ordered as a chain whenever possible: an
        edge whose ``source`` equals the previous edge's ``target`` is
        placed next. This lets ``CategoricalTransitivity`` /
        ``CausalChain`` / ``FunctionalComposition`` fire on real
        A->B->C patterns rather than seeing the edges in arbitrary
        graph-iteration order.

        Returns an empty list if the graph is unavailable or no typed
        edges connect the retrieved episomes.
        """
        if self.graph is None or not episomes:
            return []
        try:
            from engrams.semantic_engram import Semesome  # noqa: WPS433
        except Exception:
            return []
        inner = getattr(self.graph, "_graph", None)
        if inner is None:
            return []

        # Map str(episome_id) -> episome.text for label lookup + filter.
        retrieved_ids = {str(getattr(e, "id", "")) for e in episomes}
        id_to_text = {
            str(getattr(e, "id", "")): getattr(e, "text", str(e.id))
            for e in episomes
        }
        # Also include labels from the graph (in case the episome.text
        # was a normalized form that differs from the stored label).
        for nid in retrieved_ids:
            node = inner.get_node(nid)
            if node is not None:
                id_to_text[nid] = node.label

        raw_edges: List[Any] = []
        seen_pairs: set = set()
        for nid in retrieved_ids:
            for edge in inner.get_edges_from(nid):
                if edge.target_id not in retrieved_ids:
                    continue
                pair = (edge.source_id, edge.target_id,
                        str(edge.relation_type))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                # Three-factor learning: stamp this edge with an
                # eligibility trace increment so the next
                # reinforce()/penalize() call can credit it. This is
                # the local-factor side of Δw = M · e (R-STDP).
                self._eligibility[pair] = (
                    self._eligibility.get(pair, 0.0)
                    + self._ELIGIBILITY_INCREMENT
                )
                raw_edges.append(Semesome(
                    type=str(edge.relation_type.value).upper(),
                    weight=float(edge.confidence),
                    source=id_to_text.get(edge.source_id, edge.source_id),
                    target=id_to_text.get(edge.target_id, edge.target_id),
                ))
        return self._order_chain(raw_edges)

    def _decay_eligibility(self) -> None:
        """Exponentially decay all eligibility traces, drop ~zero ones.

        Called at the start of ``process()`` so the most-recent
        traversal's edges dominate the trace — older traversals fade
        by ``_ELIGIBILITY_DECAY`` per call. This is the time-decay
        half of Izhikevich's distal-reward solution.
        """
        if not self._eligibility:
            return
        decayed = {
            k: v * self._ELIGIBILITY_DECAY
            for k, v in self._eligibility.items()
        }
        self._eligibility = {
            k: v for k, v in decayed.items()
            if v > self._ELIGIBILITY_EPSILON
        }

    def _apply_eligibility_weighted(self, modulatory_signal: float) -> bool:
        """Distribute ``modulatory_signal`` across edges by eligibility.

        Implements the three-factor rule ``Δw_ij = M · e_ij / Σe``:
        each edge whose trace > 0 receives a confidence delta
        proportional to its share of the total trace, signed by
        ``modulatory_signal`` (+_REINFORCE_DELTA for reinforce,
        -_REINFORCE_DELTA for penalize). Each edge is clipped to
        [0.0, 1.0] per TypedEdge's invariant.

        Returns True if any edge was updated (i.e. the trace was
        non-empty and the apply actually ran), False otherwise —
        callers use the return value to decide whether to also apply
        the legacy uniform node bump (False) or skip it (True), so
        the total modulatory budget is conserved.
        """
        if not self._eligibility:
            return False
        if self.graph is None:
            return False
        inner = getattr(self.graph, "_graph", None)
        if inner is None:
            return False
        total = sum(self._eligibility.values())
        if total <= self._ELIGIBILITY_EPSILON:
            return False
        # Walk a snapshot — we don't mutate the trace dict here.
        for (src, tgt, rel), trace in list(self._eligibility.items()):
            if trace <= self._ELIGIBILITY_EPSILON:
                continue
            share = trace / total
            delta = modulatory_signal * share
            # Find the actual TypedEdge by matching (src, tgt, rel) —
            # get_edges_from returns live references into _edges, so
            # in-place mutation of .confidence propagates to the graph.
            try:
                for edge in inner.get_edges_from(src):
                    if (
                        edge.target_id == tgt
                        and str(edge.relation_type) == rel
                    ):
                        edge.confidence = float(
                            max(0.0, min(1.0, edge.confidence + delta))
                        )
                        break
            except Exception:  # noqa: BLE001
                # Best-effort: a single edge-mutation failure must
                # not abort the rest of the credit assignment.
                continue
        # Consume the trace — the modulatory signal has been applied.
        # A residual decay (rather than hard reset) keeps traces
        # available for back-to-back reinforces, matching the
        # biological "lingering eligibility" model (Gerstner 2018).
        self._eligibility = {
            k: v * self._ELIGIBILITY_DECAY
            for k, v in self._eligibility.items()
            if v * self._ELIGIBILITY_DECAY > self._ELIGIBILITY_EPSILON
        }
        return True

    @staticmethod
    def _order_chain(edges: List[Any]) -> List[Any]:
        """Order ``edges`` so adjacent pairs chain (e_i.target == e_{i+1}.source).

        Greedy: pick any starting edge, then repeatedly look for an
        unused edge whose ``source`` equals the current edge's
        ``target``. If no chain-extension is possible, fall back to
        the next unused edge. Edges that don't chain with anything are
        appended at the end in their original order. This maximizes
        the number of (A->B, B->C) adjacent pairs the deductive rules
        can match against.
        """
        if len(edges) <= 1:
            return list(edges)
        remaining = list(edges)
        ordered: List[Any] = []
        # Start from the edge whose source has no incoming edge in the
        # set — i.e. a likely "head" of the chain. Fall back to edges[0].
        sources = {e.source for e in remaining}
        targets = {e.target for e in remaining}
        head_candidates = [e for e in remaining if e.source not in targets]
        current = head_candidates[0] if head_candidates else remaining[0]
        ordered.append(current)
        remaining.remove(current)

        while remaining:
            next_edge = None
            for e in remaining:
                if e.source == current.target:
                    next_edge = e
                    break
            if next_edge is None:
                # No chain extension — append the first remaining edge
                # so we still make progress.
                next_edge = remaining[0]
            ordered.append(next_edge)
            remaining.remove(next_edge)
            current = next_edge
        return ordered

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn(self, question: str, wrong: str, correction: str) -> Dict[str, Any]:
        """
        HIPPOCAMPAL_ENCODING: Encode new episome via Trisynaptic Circuit.

        Circuit flow: EC -> DG -> CA3 -> CA1 -> Sub.

        Args:
            question: Stimulus / user query.
            wrong: Error signal - what the model got wrong.
            correction: Corrected fact to store as episome.

        Returns:
            Dict with ``node_id``, ``confidence``, ``graph_size``.
            Phase 2 additions:
              - ``definition_conflict``: a :class:`DefinitionConflict`
                dataclass (or ``None``) describing any cross-node
                definition conflict detected at learn time. See
                ``_check_definition_conflict`` for the detection
                contract. ``None`` means "no conflict detected"
                (either no surface collision, or the new episome's
                definition has not been generated yet — the lazy
                populate path will check again at articulate time).
              - ``surface_collision``: bool — True if there is at
                least one pre-existing Episome with the same surface
                text (Layer 1). This is a *signal*, not a *conflict*:
                a collision is a precondition for a definition
                conflict, but a collision without divergent
                definitions is fine (e.g. the user is reinforcing the
                same fact). Callers can use this flag to trigger
                eager definition generation if they want an immediate
                conflict check.
        """
        # Default fallback if trisynaptic circuit is unavailable.
        fallback: Dict[str, Any] = {
            "node_id": None,
            "confidence": 0.0,
            "graph_size": len(self._episomes),
            # Phase 2: keep the fallback shape consistent with the
            # success path so callers don't have to special-case
            # ``None`` returns.
            "definition_conflict": None,
            "surface_collision": False,
        }
        if self.trisynaptic is None:
            return fallback
        try:
            # The "stimulus" the trisynaptic circuit encodes is the
            # question (so DG / CA3 keyword indexing matches future
            # queries); the "fact" being learned is the correction.
            episome = self.trisynaptic.encode(
                stimulus=question, correction=correction,
            )
        except Exception:
            return fallback

        # Track in the registry.
        self._episomes.append(episome)

        # Phase 2: check for cross-node definition conflict. The
        # check compares the new episome's (text, amodal_definition)
        # against every pre-existing episome with the same surface
        # text. Because amodal definitions are generated lazily at
        # articulate time, the new episome's definition is almost
        # always empty at learn time → the check returns None and
        # surfaces the collision via ``surface_collision=True``. The
        # actual conflict (if any) will be detected later when both
        # definitions have been populated. Callers that want an
        # immediate check can call ``_populate_definition`` on the
        # new episome before calling ``learn`` again.
        conflict, surface_collision = self._check_definition_conflict(episome)

        return {
            "node_id": episome.id,
            "confidence": float(episome.confidence),
            "graph_size": len(self._episomes),
            # Phase 2: surface the conflict for the caller. ``None``
            # means "no conflict detected *yet*" — see docstring.
            "definition_conflict": conflict,
            "surface_collision": surface_collision,
        }

    # ------------------------------------------------------------------
    # Phase 2 — cross-node definition consistency check
    # ------------------------------------------------------------------

    def _check_definition_conflict(
        self,
        new_episome: Any,
    ) -> tuple:
        """Check ``new_episome`` against pre-existing episomes for conflicts.

        Called by ``learn()`` after the new episome is appended to the
        registry. Scans every pre-existing episome whose surface text
        matches ``new_episome.text`` (normalized) and runs
        ``CingulateGyrus.detect_definition_conflict`` on each pair.

        Returns:
            Tuple ``(conflict, surface_collision)``:
              - ``conflict``: the first :class:`DefinitionConflict`
                with ``detected=True``, or ``None`` if no conflict
                was detected. We return the *first* conflict rather
                than a list because the typical case is 0 or 1
                pre-existing episome with the same surface — a list
                would add API surface for no real benefit. Callers
                that want the full conflict log can read
                ``self.cingulate.definition_conflict_log``.
              - ``surface_collision``: True if at least one
                pre-existing episome shares the new episome's
                surface text (regardless of whether a definition
                conflict was detected). Used to surface the
                "lazy-populate-pending" case to the caller.

        Failure contract:
            Any exception in the conflict checker is swallowed and
            logged — a broken CingulateGyrus must not crash
            ``learn()``. Returns ``(None, False)`` in that case.
        """
        if self.cingulate is None:
            return (None, False)
        try:
            # Normalize the new episome's surface for the collision
            # check. We use the same lower-case + collapse logic as
            # ``CingulateGyrus._normalize_surface`` so the collision
            # set matches what the conflict checker considers "same".
            new_text_norm = " ".join(
                (getattr(new_episome, "text", "") or "").lower().split()
            )
            if not new_text_norm:
                return (None, False)

            surface_collision = False
            first_conflict = None
            for existing in self._episomes:
                # Skip the new episome itself (it's already in the
                # registry at this point — learn() appends before
                # calling this method).
                if existing is new_episome:
                    continue
                existing_text_norm = " ".join(
                    (getattr(existing, "text", "") or "").lower().split()
                )
                if existing_text_norm != new_text_norm:
                    continue
                surface_collision = True
                result = self.cingulate.detect_definition_conflict(
                    new_episome, existing
                )
                if result.detected and first_conflict is None:
                    first_conflict = result
            return (first_conflict, surface_collision)
        except Exception:  # noqa: BLE001
            return (None, False)

    def process(self, question: str) -> Dict[str, Any]:
        """
        NEOCORTICAL_REASONING: Retrieve -> Deduce -> Articulate.

        Args:
            question: User query.

        Returns:
            Dict with ``answer``, ``chain``, ``chain_confidence``.
        """
        empty: Dict[str, Any] = {
            "answer": "",
            "chain": "",
            "chain_confidence": 0.0,
        }

        # Three-factor learning: decay existing eligibility traces so
        # the edges about to be traversed (step 2 below) dominate the
        # next reinforce()/penalize() credit assignment. Older
        # traversals fade by _ELIGIBILITY_DECAY per process() call.
        self._decay_eligibility()

        # 1. Retrieve candidate episomes via the Papez circuit.
        episomes: List[Any] = []
        if self.papez is not None and self.graph is not None:
            try:
                episomes = self.papez.retrieve(question, self.graph, top_k=3)
            except Exception:
                episomes = []
        if not episomes:
            return empty

        # 2. Build semesome chain from the *real* typed edges that
        #    TrisynapticCircuit recorded in the AGNNGraph between the
        #    retrieved episomes. The previous implementation synthesized
        #    degenerate self-edges (source == target == episome.id) which
        #    could never trigger any BA 44 rule, leaving
        #    ``deduction.confidence`` (and therefore ``chain_confidence``)
        #    permanently at 0.0. Walking the actual graph edges lets
        #    CategoricalTransitivity / CausalChain / FunctionalComposition
        #    fire on real A->B->C patterns.
        semesomes = self._build_semesomes_from_graph(episomes)

        # 3. Deduce via BA 44.
        deduction = None
        if self.deductive is not None and semesomes:
            try:
                deduction = self.deductive.deduce(semesomes)
            except Exception:
                deduction = None

        if deduction is not None and deduction.rule_count > 0:
            chain = str(deduction)
            chain_confidence = float(getattr(deduction, "confidence", 0.0))
        elif semesomes:
            # No deductive rule fired, but real typed edges exist between
            # retrieved episomes. Surface the strongest edge weight as the
            # chain confidence so connected retrieval yields > 0 (the DoD
            # requires ``chain_confidence > 0`` once the graph has nodes).
            # Fall back to the joined episome texts for the chain so the
            # caller still gets a non-empty reasoning trace.
            chain = " -> ".join(
                getattr(e, "text", str(e))[:50] for e in episomes
            )
            chain_confidence = max(
                0.0,
                max(
                    float(getattr(s, "weight", 0.0)) for s in semesomes
                ),
            )
        else:
            # Fallback: join episome texts so the chain is non-empty.
            chain = " -> ".join(
                getattr(e, "text", str(e))[:50] for e in episomes
            )
            chain_confidence = 0.5

        # 4. Articulate the answer. Phase 1: when the Phase 1 helpers
        #    (DefinitionExtractor + AphantasicChainFormatter) are
        #    available, take the aphantasic path — it lazily populates
        #    each retrieved episome's amodal_definition (Layer 2) and
        #    formats the chain into the 3-layer KONSEP/DEFINISI/RELASI
        #    structure that disambiguates ``api``/``API`` and
        #    ``air``/``air`` at the node level. When Phase 1 helpers
        #    are unavailable (e.g. import failed at module load), fall
        #    back to the pre-Phase-1 path: pass the already-built
        #    ``chain`` string to ``_articulate``. The fallback
        #    preserves the Phase 0 system message anchor — only the
        #    DEFINISI/RELASI sections are skipped.
        if (
            self._definition_extractor is not None
            and self._chain_formatter is not None
        ):
            answer = self._articulate_aphantasic(question, episomes, semesomes)
        else:
            answer = self._articulate(question, chain)
        return {
            "answer": answer,
            "chain": chain,
            "chain_confidence": chain_confidence,
        }

    def introspect(self) -> Dict[str, Any]:
        """
        APHANTASIC_INSPECT: Conceptual snapshot (no visual heatmap).

        Returns:
            Dict with ``graph_size``, ``avg_confidence``, ``top_nodes``,
            and ``deductive_rules_applied``.

        ``top_nodes`` is a list of dicts (max 5), each with the shape
        ``{"id": <int>, "text": <str>, "confidence": <float>}``,
        sorted by descending confidence. Returning plain node IDs
        (ints) broke callers that did ``n['text']`` on each entry, so
        we surface the full record here.
        """
        episomes = self._episomes
        graph_size = len(episomes)
        if graph_size == 0:
            avg_confidence = 0.0
            top_nodes: List[Any] = []
        else:
            try:
                avg_confidence = sum(
                    float(e.confidence) for e in episomes
                ) / graph_size
            except Exception:
                avg_confidence = 0.0
            try:
                top_nodes = [
                    {
                        "id": e.id,
                        "text": getattr(e, "text", str(e.id)),
                        "confidence": float(getattr(e, "confidence", 0.0)),
                    }
                    for e in sorted(
                        episomes,
                        key=lambda x: float(getattr(x, "confidence", 0.0)),
                        reverse=True,
                    )[:5]
                ]
            except Exception:
                top_nodes = []

        # Lifetime count of BA 44 rule firings (if available).
        deductive_rules_applied = 0
        if self.deductive is not None:
            deductive_rules_applied = int(
                getattr(self.deductive, "rule_count", 0)
            )

        return {
            "graph_size": graph_size,
            "avg_confidence": avg_confidence,
            "top_nodes": top_nodes,
            "deductive_rules_applied": deductive_rules_applied,
        }

    def traverse(self, question: str, max_hops: int = 2) -> str:
        """
        FORNIX: Bidirectional beam search along typed edges.

        Args:
            question: Seed query.
            max_hops: Beam depth (default 2).

        Returns:
            Reasoning chain as human-readable string.
        """
        if self.papez is None or self.graph is None:
            return ""
        try:
            top_k = max(1, int(max_hops) + 1)
            episomes = self.papez.retrieve(question, self.graph, top_k=top_k)
        except Exception:
            return ""
        if not episomes:
            return ""
        try:
            return " -> ".join(
                str(getattr(e, "text", e))[:50] for e in episomes
            )
        except Exception:
            return ""

    def consolidate(self) -> Dict[str, Any]:
        """
        SYSTEMS_CONSOLIDATION: Hippocampus -> Neocortex transfer.

        Triggers spiking neural replay, refines embeddings, converts
        episodic confidence to edge weight.

        Returns:
            Dict with ``spikes_fired``, ``graph_size``,
            ``embedding_refined``, and ``transferred`` (count).
        """
        result: Dict[str, Any] = {
            "spikes_fired": 0,
            "graph_size": len(self._episomes),
            "embedding_refined": False,
            "transferred": 0,
        }
        if self.consolidation is None or self.graph is None:
            return result
        transferred = 0
        for epi in list(self._episomes):
            try:
                semesome = self.consolidation.consolidate(epi, self.graph)
                if semesome is not None:
                    transferred += 1
            except Exception:
                continue
        result["transferred"] = transferred
        result["embedding_refined"] = transferred > 0
        # spikes_fired is a placeholder for the neural-replay integration
        # which is not yet wired into AGNNCore. Use transferred count as
        # a proxy so callers get a non-zero signal when work happened.
        result["spikes_fired"] = transferred
        return result

    def reinforce(self, episome_id: Any) -> None:
        """
        REINFORCEMENT: Positive confidence update (correct answer).

        Biologis: Dopamine (mesolimbic) -> strengthen synapses.
        AI: ``confidence += 0.1`` (capped at 1.0).

        Three-factor learning (R-STDP / neoHebbian): if a non-empty
        eligibility trace exists from a recent ``process()`` /
        ``traverse()`` traversal, the +0.1 modulatory signal is
        distributed across the traversed edges proportionally to
        their trace — ``Δw_ij = M · e_ij / Σe`` — implementing
        per-edge credit assignment. Edges never traversed receive
        zero delta. When the trace is empty (cold start, no recent
        traversal), the legacy uniform +0.1 node-confidence bump
        runs instead, preserving pre-three-factor behavior. The
        public signature is unchanged.

        Phase 1 (Aphantasic Node Representation): also tracks the
        cumulative positive delta per node. When the delta crosses
        ``_DEFINITION_INVALIDATE_THRESHOLD`` (default 0.3, i.e. three
        reinforces), the node's cached amodal definition is
        invalidated — ``DefinitionExtractor.invalidate(text)`` drops
        the cross-Episome cache entry, ``episome.definition_dirty`` is
        set to True so the next ``_articulate_aphantasic`` call
        re-generates the definition via ``force_refresh=True``, and
        the cumulative delta resets to 0. Reasoning (user-confirmed):
        "reinforce berarti konsep berkembang, definisi lama mungkin
        sudah terlalu sempit." Penalize does NOT subtract from the
        delta — only growth triggers re-generation.

        Args:
            episome_id: Node to reinforce.
        """
        epi = self._find_episome(episome_id)
        if epi is None:
            return
        try:
            # Three-factor learning: if there is a lingering
            # eligibility trace from a recent traversal, distribute
            # the +_REINFORCE_DELTA modulatory signal across the
            # traversed edges (per-edge credit assignment). Otherwise
            # fall through to the legacy uniform node bump.
            applied_edge = self._apply_eligibility_weighted(
                self._REINFORCE_DELTA
            )
            if not applied_edge:
                epi.confidence = min(
                    1.0, float(epi.confidence) + self._REINFORCE_DELTA,
                )
                # Mirror onto the graph node so retrieval sees the new value.
                self._mirror_confidence_to_graph(epi)

            # Phase 1: track cumulative positive delta + invalidate
            # the cached definition when the threshold is crossed.
            # Skip when Phase 1 helpers are unavailable (e.g. import
            # failed at module load) — the delta tracker stays empty
            # and the behavior reduces to pre-Phase-1.
            if self._definition_extractor is not None:
                try:
                    current_delta = self._reinforce_deltas.get(
                        episome_id, 0.0
                    ) + self._REINFORCE_DELTA
                    if current_delta >= self._DEFINITION_INVALIDATE_THRESHOLD:
                        # Crossed the threshold — invalidate the cache
                        # and mark the episome dirty so the next
                        # articulate call re-generates the definition.
                        self._definition_extractor.invalidate(
                            getattr(epi, "text", "")
                        )
                        epi.definition_dirty = True
                        # Reset the delta — the next invalidation
                        # cycle starts fresh from here.
                        self._reinforce_deltas[episome_id] = 0.0
                    else:
                        self._reinforce_deltas[episome_id] = current_delta
                except Exception:  # noqa: BLE001
                    # Best-effort: don't let the invalidation logic
                    # break the core reinforce() contract.
                    pass
        except Exception:
            pass

    def penalize(self, episome_id: Any) -> None:
        """
        PENALIZATION: Negative confidence update (wrong answer).

        Biologis: Serotonin (raphe nucleus) -> weaken synapses.
        AI: ``confidence -= 0.1`` (floored at 0.0).

        Three-factor learning: mirrors ``reinforce()`` — when a
        non-empty eligibility trace exists, the -0.1 modulatory
        signal is distributed across traversed edges proportionally
        to their trace; otherwise the legacy uniform -0.1
        node-confidence bump runs. Public signature unchanged.

        Args:
            episome_id: Node to penalize.
        """
        epi = self._find_episome(episome_id)
        if epi is None:
            return
        try:
            applied_edge = self._apply_eligibility_weighted(
                -self._REINFORCE_DELTA
            )
            if not applied_edge:
                epi.confidence = max(
                    0.0, float(epi.confidence) - self._REINFORCE_DELTA,
                )
                self._mirror_confidence_to_graph(epi)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Output-level feedback loop (RLHF-style, statistical substrate)
    # ------------------------------------------------------------------

    def apply_feedback(
        self,
        sentence: str,
        verdict: str,
    ) -> Dict[str, Any]:
        """Apply output-level feedback to the action↔object edge.

        Parses the sentence using the cluster learner's ``spo()``
        method, looks up the action↔object edge in the graph (if it
        exists), stamps an eligibility trace on that edge, and calls
        ``reinforce()`` or ``penalize()`` — which distributes the
        ±0.1 modulatory signal across the stamped edge via the
        existing three-factor learning path.

        This is the RLHF-style feedback loop, but the substrate is
        statistical (action_object_freq + TypedEdge.confidence),
        not neural weights. The user judges the GENERATED SENTENCE
        ("makes sense" / "doesn't make sense"); the credit-assignment
        flows to the action↔object edge that produced it.

        Zero-bias contract — what this method does NOT do:

          * It does NOT relabel the cluster. The RelationType
            assigned by ``label_clusters()`` is unchanged. The
            feedback only adjusts the edge confidence (a continuous
            weight), never the cluster membership or the label.
            This is the guard against the feedback loop silently
            injecting semantic bias into the cluster structure —
            the principle AGNN holds throughout the project.
          * It does NOT add or remove cluster members. The set
            ``action_clusters[cluster_id]`` is unchanged.
          * It does NOT touch the cluster learner's
            ``action_object_freq``. That table is the corpus
            statistics; feedback adjusts the graph edge weights,
            which are a separate, post-hoc signal.

        What it DOES do (the entire effect):

          * Finds the TypedEdge whose source.label == subject and
            target.label == object and relation_type == the action's
            cluster label (if the cluster is labelled). If multiple
            edges match, all get stamped.
          * Stamps each matching edge with an eligibility trace
            increment (``_ELIGIBILITY_INCREMENT = 1.0``).
          * Calls ``reinforce(source_episome_id)`` or
            ``penalize(source_episome_id)``. The existing
            ``_apply_eligibility_weighted`` distributes the ±0.1
            modulatory signal across the stamped edges
            proportionally to their trace (which is uniform here —
            one edge, one share = full ±0.1).

        Args:
            sentence: The generated sentence to give feedback on.
                Typically the output of
                ``PositionalClusterLearner.sample_sentence()``, but
                any 3+ token SVO sentence works. Parsed via
                ``self._cluster_learner.spo()`` (or the fallback
                SemanticRoleClassifier's spo() if no cluster learner
                is loaded).
            verdict: ``"good"`` to reinforce, ``"bad"`` to penalize.
                Any other value is silently ignored (returns
                ``{"applied": False, "reason": "invalid_verdict"}``).

        Returns:
            Dict with diagnostic fields:
              * ``applied`` (bool): True if feedback was applied
                (edge found + stamped + reinforce/penalize called).
              * ``reason`` (str): Why feedback was/wasn't applied.
                One of ``"ok"``, ``"no_cluster_learner"``,
                ``"parse_failed"``, ``"no_action_token"``,
                ``"no_object_token"``, ``"action_unclustered"``,
                ``"no_graph"``, ``"no_matching_edge"``,
                ``"invalid_verdict"``.
              * ``action`` (str|None): The action token parsed from
                the sentence.
              * ``object`` (str|None): The object token parsed from
                the sentence.
              * ``cluster_id`` (int|None): The cluster id of the
                action, or None if unclustered.
              * ``relation_type`` (str|None): The RelationType name
                of the edge that was stamped, or None.
              * ``edges_stamped`` (int): How many edges received an
                eligibility trace increment (typically 1).
        """
        result: Dict[str, Any] = {
            "applied": False,
            "reason": "unknown",
            "action": None,
            "object": None,
            "cluster_id": None,
            "relation_type": None,
            "edges_stamped": 0,
        }

        # Validate verdict early.
        if verdict not in ("good", "bad"):
            result["reason"] = "invalid_verdict"
            return result

        # Need a classifier with spo() — the cluster learner if loaded,
        # else the trisynaptic circuit's role_classifier, else bail.
        classifier = self._cluster_learner
        if classifier is None:
            # Try the trisynaptic circuit's role_classifier.
            ts = getattr(self, "trisynaptic", None)
            classifier = getattr(ts, "role_classifier", None) if ts else None
        if classifier is None:
            result["reason"] = "no_cluster_learner"
            return result

        # Parse the sentence.
        try:
            spo = classifier.spo(sentence)
        except Exception:
            result["reason"] = "parse_failed"
            return result

        action_token = self._normalize_spo_token(spo.predicate)
        object_token = self._normalize_spo_token(spo.object)
        if not action_token:
            result["reason"] = "no_action_token"
            return result
        if not object_token:
            result["reason"] = "no_object_token"
            return result
        result["action"] = action_token
        result["object"] = object_token

        # Look up the action's relation type via the stable content-
        # addressed API (issue #93: cluster IDs are not stable across
        # PCL versions, so we never expose them to callers). The new
        # ``get_relation_type_for_action`` method returns the
        # RelationType directly (or None if the action is unclustered
        # or in an unlabelled cluster), without revealing the
        # underlying cluster_id.
        #
        # We still need the cluster_id for the result dict
        # (``result["cluster_id"]`` is a documented public field of
        # ``apply_feedback``'s return value, used by
        # ``sample_feedback_loop.py`` for debug logging). Reach into
        # the internal ``cluster_id_of`` only for that one debug
        # output — never for label resolution.
        relation_type = None
        cluster_id = None
        if hasattr(classifier, "get_relation_type_for_action"):
            # Preferred path: stable, content-addressed API.
            relation_type = classifier.get_relation_type_for_action(action_token)
            # The cluster_id is still useful for debug logging, so
            # fetch it from the internal field. This is safe because
            # we don't use it for any semantic decision — only for
            # the ``result["cluster_id"]`` debug field.
            if relation_type is not None and hasattr(classifier, "cluster_id_of"):
                cluster_id = classifier.cluster_id_of.get(action_token)
                if cluster_id is not None and cluster_id >= 0:
                    result["cluster_id"] = cluster_id
                    result["relation_type"] = relation_type.name
        elif hasattr(classifier, "cluster_id_of"):
            # Fallback for older classifiers that don't yet have the
            # new method (e.g. a plain SemanticRoleClassifier without
            # PCL wrapping). Use the legacy direct-access path. This
            # branch can be removed once all classifier implementations
            # in the repo expose ``get_relation_type_for_action``.
            cluster_id = classifier.cluster_id_of.get(action_token)
            if cluster_id is not None and cluster_id >= 0:
                result["cluster_id"] = cluster_id
                if hasattr(classifier, "cluster_labels") and cluster_id in classifier.cluster_labels:
                    relation_type = classifier.cluster_labels[cluster_id]
                    result["relation_type"] = (
                        relation_type.name if relation_type else None
                    )
        if relation_type is None:
            # Action not in any labelled cluster — nothing to credit.
            # This is the guard against feedback on un-clustered
            # actions polluting the graph: we only adjust edges whose
            # relation_type the cluster learner has assigned.
            if cluster_id is None or cluster_id < 0:
                result["reason"] = "action_unclustered"
            else:
                # Clustered but unlabelled — distinct reason so debug
                # output can tell the two cases apart.
                result["reason"] = "action_in_unlabelled_cluster"
            return result

        # Need a graph to find edges in.
        if self.graph is None:
            result["reason"] = "no_graph"
            return result
        inner = getattr(self.graph, "_graph", None)
        if inner is None:
            result["reason"] = "no_graph"
            return result

        # Find edges whose source.label == subject and target.label ==
        # object and relation_type == the cluster's label. We iterate
        # over all nodes, find those whose label matches the subject,
        # then walk their outgoing edges looking for matching target
        # label + relation type.
        subject_token = self._normalize_spo_token(spo.subject)

        # Find candidate source nodes by label.
        source_node_ids: List[str] = []
        try:
            for nid in self._iter_node_ids(inner):
                node = inner.get_node(nid)
                if node is None:
                    continue
                if self._normalize_spo_token(getattr(node, "label", "")) == subject_token:
                    source_node_ids.append(nid)
        except Exception:
            pass

        # Walk outgoing edges from each source node, find matches.
        edges_stamped = 0
        for src_id in source_node_ids:
            try:
                for edge in inner.get_edges_from(src_id):
                    tgt_node = inner.get_node(edge.target_id)
                    if tgt_node is None:
                        continue
                    if self._normalize_spo_token(getattr(tgt_node, "label", "")) != object_token:
                        continue
                    # Match relation type if we have one. When the
                    # cluster is unlabelled (relation_type is None),
                    # stamp ANY edge between subject and object —
                    # this is rare (label_clusters is usually called
                    # before apply_feedback) but the contract is to
                    # credit the edge regardless of label when the
                    # cluster hasn't been named yet.
                    if relation_type is not None:
                        edge_rel = edge.relation_type
                        edge_rel_name = (
                            edge_rel.name if hasattr(edge_rel, "name")
                            else str(edge_rel)
                        )
                        if edge_rel_name != relation_type.name:
                            continue
                    # Stamp this edge with an eligibility trace
                    # increment so the upcoming reinforce/penalize
                    # credits it via _apply_eligibility_weighted.
                    pair = (
                        edge.source_id,
                        edge.target_id,
                        str(edge.relation_type),
                    )
                    self._eligibility[pair] = (
                        self._eligibility.get(pair, 0.0)
                        + self._ELIGIBILITY_INCREMENT
                    )
                    edges_stamped += 1
            except Exception:  # noqa: BLE001
                # Best-effort: a single edge-walk failure must not
                # abort the rest of the feedback loop.
                continue

        result["edges_stamped"] = edges_stamped
        if edges_stamped == 0:
            result["reason"] = "no_matching_edge"
            return result

        # Find a valid episome to pass to reinforce/penalize. The
        # reinforce/penalize code path needs a non-None episome to
        # proceed past the early-return guard. We use the first
        # source node's episome if it exists; otherwise we bail
        # because reinforce/penalize would silently no-op.
        target_episome_id: Optional[Any] = None
        for src_id in source_node_ids:
            epi = self._find_episome(src_id)
            if epi is not None:
                target_episome_id = src_id
                break
        if target_episome_id is None:
            # No registered episome for any source node. The trace
            # was stamped but reinforce/penalize would no-op. Clear
            # the stamp so we don't leak trace state.
            for src_id in source_node_ids:
                pair_keys_to_clear = [
                    k for k in self._eligibility
                    if k[0] == src_id
                ]
                for k in pair_keys_to_clear:
                    self._eligibility[k] -= self._ELIGIBILITY_INCREMENT
                    if self._eligibility[k] <= self._ELIGIBILITY_EPSILON:
                        del self._eligibility[k]
            result["reason"] = "no_matching_edge"
            result["edges_stamped"] = 0
            return result

        # Apply the verdict via the existing three-factor path.
        # The eligibility trace we just stamped will route the ±0.1
        # modulatory signal to the matching edge(s).
        if verdict == "good":
            self.reinforce(target_episome_id)
        else:
            self.penalize(target_episome_id)

        result["applied"] = True
        result["reason"] = "ok"
        return result

    @staticmethod
    def _normalize_spo_token(token: Any) -> str:
        """Lower-case + strip a token (or multi-token predicate phrase).

        ``spo().predicate`` for >3-token sentences is a space-joined
        phrase like ``"sedang makan nasi"``; we take the FIRST token
        (the actual action) for cluster lookup. For single-token
        predicates / subjects / objects this is a no-op lower-case.
        """
        if not token:
            return ""
        s = str(token).lower().strip()
        if not s:
            return ""
        # For multi-token predicate phrases, take the first token —
        # that's the action the cluster learner indexed.
        return s.split()[0]

    @staticmethod
    def _iter_node_ids(inner: Any) -> List[str]:
        """Best-effort iteration over all node ids in the wrapped graph.

        Tries common AGNNGraph APIs in order: ``get_all_node_ids()``,
        ``nodes`` attribute (dict-like), then ``_nodes`` private
        attribute. Returns an empty list if none work — the caller
        (apply_feedback) will then find no matching edges and report
        ``no_matching_edge``.
        """
        # Common API: explicit getter.
        for method_name in ("get_all_node_ids", "all_node_ids"):
            method = getattr(inner, method_name, None)
            if callable(method):
                try:
                    ids = method()
                    if ids is not None:
                        return list(ids)
                except Exception:
                    pass
        # Dict-like attribute.
        for attr in ("nodes", "_nodes", "node_map"):
            coll = getattr(inner, attr, None)
            if isinstance(coll, dict):
                return list(coll.keys())
            try:
                # Maybe it's iterable of nodes.
                if coll is not None:
                    return [getattr(n, "id", str(n)) for n in coll]
            except Exception:
                pass
        return []

    def _mirror_confidence_to_graph(self, episome: Any) -> None:
        """Best-effort: copy episome.confidence onto the graph node.

        The PapezCircuit reads confidence from the AGNNNode, so without
        this mirror a reinforce/penalize wouldn't affect future
        retrieval rankings.
        """
        if self.graph is None:
            return
        try:
            inner = getattr(self.graph, "_graph", None)
            if inner is None:
                return
            node = inner.get_node(str(episome.id))
            if node is not None:
                node.confidence = float(episome.confidence)
        except Exception:
            pass
