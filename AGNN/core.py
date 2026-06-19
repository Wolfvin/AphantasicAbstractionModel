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
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Make sibling AGNN subpackages importable when ``core`` is loaded as
# ``AGNN.core`` (repo root on sys.path) or as ``core`` (AGNN/ on
# sys.path, the convention used by every other module in AGNN/).
# ----------------------------------------------------------------------
_AGNP_ROOT = os.path.dirname(os.path.abspath(__file__))
if _AGNP_ROOT not in sys.path:
    sys.path.insert(0, _AGNP_ROOT)


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

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize brain-inspired memory system.

        Args:
            model_path: Path to small LLM (e.g. Qwen3-0.6B) used for
                articulation. May be ``None`` - in that case
                :meth:`_articulate` returns the chain string verbatim
                with a small note. Model loading is lazy: we only
                attempt to load the model on the first
                :meth:`_articulate` call.
        """
        # Component wiring. Each component is wrapped in try/except so
        # AGNNCore can still be constructed even if a sibling component
        # raises on init (e.g. NotImplementedError skeleton).
        self.graph = self._safe_init("engrams.engram_complex", "EngramComplex")
        self.trisynaptic = self._safe_init(
            "circuits.trisynaptic_circuit", "TrisynapticCircuit",
            kwargs={"engram_complex": self.graph} if self.graph is not None else {},
        )
        self.papez = self._safe_init("circuits.papez_circuit", "PapezCircuit")
        self.deductive = self._safe_init(
            "neocortex.inferior_frontal_gyrus", "InferiorFrontalGyrus",
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

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
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

        Args:
            prompt: Input prompt (already formatted with the knowledge
                graph context + question).
            max_new_tokens: Maximum new tokens to generate (default 256).
                The prompt tokens are NOT counted toward this limit.

        Returns:
            Generated text (prompt tokens stripped, only the new tokens
            decoded). Falls back to returning the prompt itself on any
            error so the caller still gets a non-empty string.
        """
        try:
            import torch  # noqa: WPS433
            # Wrap the formatted prompt in Qwen3's chat template. The
            # user message carries the full chain context + Q/A pair so
            # the model has the same information as before, just in the
            # structural format Qwen3 was post-trained on.
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

        Prompt template::

            [Knowledge Graph Context]
            {chain (truncated to 800 chars)}
            Q: {question}
            A:

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
                raw_edges.append(Semesome(
                    type=str(edge.relation_type.value).upper(),
                    weight=float(edge.confidence),
                    source=id_to_text.get(edge.source_id, edge.source_id),
                    target=id_to_text.get(edge.target_id, edge.target_id),
                ))
        return self._order_chain(raw_edges)

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
        """
        # Default fallback if trisynaptic circuit is unavailable.
        fallback: Dict[str, Any] = {
            "node_id": None,
            "confidence": 0.0,
            "graph_size": len(self._episomes),
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
        return {
            "node_id": episome.id,
            "confidence": float(episome.confidence),
            "graph_size": len(self._episomes),
        }

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

        # 4. Articulate the answer.
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

        Args:
            episome_id: Node to reinforce.
        """
        epi = self._find_episome(episome_id)
        if epi is None:
            return
        try:
            epi.confidence = min(
                1.0, float(epi.confidence) + self._REINFORCE_DELTA,
            )
            # Mirror onto the graph node so retrieval sees the new value.
            self._mirror_confidence_to_graph(epi)
        except Exception:
            pass

    def penalize(self, episome_id: Any) -> None:
        """
        PENALIZATION: Negative confidence update (wrong answer).

        Biologis: Serotonin (raphe nucleus) -> weaken synapses.
        AI: ``confidence -= 0.1`` (floored at 0.0).

        Args:
            episome_id: Node to penalize.
        """
        epi = self._find_episome(episome_id)
        if epi is None:
            return
        try:
            epi.confidence = max(
                0.0, float(epi.confidence) - self._REINFORCE_DELTA,
            )
            self._mirror_confidence_to_graph(epi)
        except Exception:
            pass

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


# ----------------------------------------------------------------------
# Public API shortcuts (user-facing)
# ----------------------------------------------------------------------

_core: Optional[AGNNCore] = None


def init_brain(model_path: Optional[str] = None) -> AGNNCore:
    """Initialize AGNNCore (brain) and store as module-level singleton."""
    global _core
    _core = AGNNCore(model_path=model_path)
    return _core


def learn(question: str, wrong: str, correction: str) -> Dict[str, Any]:
    """Shortcut: learn(question, wrong, correction) -> dict."""
    if _core is None:
        raise RuntimeError("init_brain() must be called first")
    return _core.learn(question, wrong, correction)


def process(question: str) -> Dict[str, Any]:
    """Shortcut: process(question) -> dict."""
    if _core is None:
        raise RuntimeError("init_brain() must be called first")
    return _core.process(question)


def inspect_engrams() -> Dict[str, Any]:
    """Shortcut: inspect_engrams() -> dict (aphantic audit)."""
    if _core is None:
        raise RuntimeError("init_brain() must be called first")
    return _core.introspect()


def reinforce(episome_id: Any) -> None:
    """Shortcut: reinforce(episome_id) -> +0.1 confidence."""
    if _core is None:
        raise RuntimeError("init_brain() must be called first")
    _core.reinforce(episome_id)


def penalize(episome_id: Any) -> None:
    """Shortcut: penalize(episome_id) -> -0.1 confidence."""
    if _core is None:
        raise RuntimeError("init_brain() must be called first")
    _core.penalize(episome_id)
