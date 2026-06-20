"""
INFERIOR FRONTAL GYRUS (BA 44): Deductive reasoning - rule application.

Biologis: Left inferior frontal gyrus (BA 44) = deductive reasoning.
AI: Apply transitivity rules (CATEGORICAL, CAUSAL, DIFFERENTIAL, FUNCTIONAL)
    plus conflict resolution (CAUSAL vs DIFFERENTIAL).

Reference: fMRI shows BA 44 > BA 8/9 for deduction vs induction
(PubMed 15178381).

The five rules implemented here mirror the table in
AGNN/ARCHITECTURE.md section 5 ("Deductive Reasoning Engine - Rule Types"):

    | Rule                              | Pattern                                  | Weight                |
    |-----------------------------------|------------------------------------------|-----------------------|
    | CATEGORICAL_TRANSITIVITY          | A->B (CAT), B->C (CAT) => A->C           | 1.0 * 1.0 = 1.0       |
    | CAUSAL_CHAIN                      | A->B (CAUSAL), B->C (CAUSAL) => A->C     | 0.7 * 0.7 = 0.49      |
    | DIFFERENTIAL_INVERSION            | A->B (DIFF=-0.8) => B->A (DIFF=-0.8)     | -0.8 (symmetric)      |
    | CAUSAL_DIFFERENTIAL_CONFLICT      | A->B (CAUSAL) + A->B (DIFF) => conflict  | (0.7 + -0.8)/2 = -0.05|
    | FUNCTIONAL_COMPOSITION            | A->B (FUNC), B->C (FUNC) => A->C         | 0.6 * 0.6 = 0.36      |

t-norms and fuzzy logic
-----------------------
The composition rules (CATEGORICAL_TRANSITIVITY, CAUSAL_CHAIN,
FUNCTIONAL_COMPOSITION) compute their inferred weight by *combining* the
two premise weights. The historical implementation hard-coded this
combination as a plain multiplication (`a * b`) — which is exactly the
**product t-norm** of fuzzy logic (one of the three canonical
conjunction operators, alongside Łukasiewicz `max(0, a+b-1)` and Gödel
`min(a, b)`).

This was called out in `AGNN/docs/research-neuro-symbolic-reasoning.md`
section 4 ("AGNN is *already* an (unconscious) product-t-norm fuzzy
logic"). This module makes the choice *explicit* and *configurable* via
the `t_norm` parameter of `InferiorFrontalGyrus`:

    t_norm = "product"      (default)  →  T(a, b) = a * b           (legacy)
    t_norm = "lukasiewicz"             →  T(a, b) = max(0, a + b - 1)
    t_norm = "godel"                   →  T(a, b) = min(a, b)

The default `"product"` reproduces the original BA44 arithmetic bit-for-
bit, so all existing tests pass unchanged. The other two t-norms are
useful for A/B testing different fuzzy-semantics regimes (see van
Krieken et al., AIJ 2022, on how the choice of t-norm affects gradient
steepness and rule-learning quality).

Note on the conflict rule: `CAUSAL_DIFFERENTIAL_CONFLICT` resolves the
two conflicting weights via arithmetic mean, which is *not* a t-norm
(it is a recognised fuzzy aggregation operator). It is therefore *not*
affected by the `t_norm` parameter — it deliberately keeps its
arithmetic-mean semantics regardless of the chosen conjunction t-norm.

Usage
-----
Two equivalent entry points:

1. `InferiorFrontalGyrus().deduce(edges)` - apply rules to a flat list of
   `Semesome` edges. Returns a `Deduction` with all inferences fired.
2. `InferiorFrontalGyrus().deduce_chain(chain)` - apply rules to an
   `EdgeChain` (preserved for backwards compatibility with the skeleton
   spec where `deduce(chain)` was the signature).

Design
------
- Each rule is its own `_Rule` subclass with `matches(edges)` and
  `apply(premises)` methods. This keeps the dispatch logic declarative and
  makes adding new rules trivial.
- `deduce()` walks the edge list once per rule and collects all
  inferences. A 3-node chain A->B->C with two CATEGORICAL edges will fire
  `CATEGORICAL_TRANSITIVITY` once and yield the inference A->C.
- A `DIFFERENTIAL` edge fires `DIFFERENTIAL_INVERSION` on its own (no
  partner needed). The inverted edge B->A gets the same weight as the
  original (symmetric inversion).
- A pair of edges with the same (source, target) but conflicting types
  (CAUSAL vs DIFFERENTIAL) fires `CAUSAL_DIFFERENTIAL_CONFLICT`. The
  resolved weight is the arithmetic mean of the two weights.
- All inferences are returned as `Semesome` instances wrapped in a
  `Deduction` so downstream code can use them directly as new edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence

# Re-use the existing engram dataclasses - they ARE our edge/node types.
from engrams.episodic_engram import Episome
from engrams.semantic_engram import Semesome


# ─────────────────────────────────────────────────────────────────────
# Edge-type vocabulary (kept as plain strings for ergonomics).
# ─────────────────────────────────────────────────────────────────────

CATEGORICAL = "CATEGORICAL"
CAUSAL = "CAUSAL"
DIFFERENTIAL = "DIFFERENTIAL"
FUNCTIONAL = "FUNCTIONAL"

_EDGE_TYPES = (CATEGORICAL, CAUSAL, DIFFERENTIAL, FUNCTIONAL)


# ─────────────────────────────────────────────────────────────────────
# t-norms (fuzzy-logic conjunctions)
# ─────────────────────────────────────────────────────────────────────

# The three canonical t-norms for fuzzy conjunction. Each takes two
# truth values in [0, 1] (or, in AGNN's looser convention, a weight in
# [-1, 1]) and returns their combined truth/weight. The "product"
# variant reproduces the legacy `a * b` arithmetic bit-for-bit.
TNormKind = Literal["product", "lukasiewicz", "godel"]


def product_tnorm(a: float, b: float) -> float:
    """Product t-norm: T(a, b) = a * b.

    This is the *legacy* BA44 behaviour: every composition rule
    previously hard-coded this multiplication. Selecting this t-norm
    reproduces the pre-refactor semantics exactly.
    """
    return a * b


def lukasiewicz_tnorm(a: float, b: float) -> float:
    """Łukasiewicz t-norm: T(a, b) = max(0, a + b - 1).

    Has steeper "either premise must be quite true" semantics: the
    combined truth collapses to 0 unless both premises individually
    exceed 0.5. Recommended for rule-learning regimes where we want
    weak premises to produce weak (rather than merely reduced) evidence.
    """
    return max(0.0, a + b - 1.0)


def godel_tnorm(a: float, b: float) -> float:
    """Gödel (minimum) t-norm: T(a, b) = min(a, b).

    The combined truth is bounded by the weaker premise. Conservative
    conjunction: a chain is only as strong as its weakest link.
    """
    return min(a, b)


_TNORMS: dict = {
    "product": product_tnorm,
    "lukasiewicz": lukasiewicz_tnorm,
    "godel": godel_tnorm,
}


def _resolve_tnorm(t_norm: TNormKind) -> Callable[[float, float], float]:
    """Map a t-norm name to its implementation, with a friendly error."""
    try:
        return _TNORMS[t_norm]
    except KeyError:
        raise ValueError(
            f"Unknown t_norm {t_norm!r}. "
            f"Expected one of: {sorted(_TNORMS)}"
        ) from None


# ─────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Inference:
    """A single rule firing.

    Attributes:
        rule: Name of the rule that fired (e.g. "CATEGORICAL_TRANSITIVITY").
        premises: List of input edges (Semesome) that triggered the rule.
        conclusion: The inferred edge (Semesome), or None for pure-conflict
            rules that only emit a resolved weight.
        weight: The inferred edge's weight.
        note: Optional human-readable explanation.
    """
    rule: str
    premises: List[Semesome]
    conclusion: Optional[Semesome]
    weight: float
    note: str = ""


@dataclass
class Deduction:
    """Result of `InferiorFrontalGyrus.deduce()`.

    Attributes:
        inferences: All rule firings, in order.
        inferred_edges: Just the `Semesome` conclusions (non-None), in order.
        rule_count: Number of rule firings.
        applied_rules: Distinct rule names that fired.
        confidence: Aggregate confidence in [0, 1] - product of all inferred
            edge weights clamped to [0, 1]. 0.0 if no inferences or if any
            inference has a non-positive weight (conflict / differential).
        context: Human-readable summary.
    """
    inferences: List[Inference] = field(default_factory=list)
    inferred_edges: List[Semesome] = field(default_factory=list)
    rule_count: int = 0
    applied_rules: List[str] = field(default_factory=list)
    confidence: float = 0.0
    context: str = ""


@dataclass
class EdgeChain:
    """Helper: a chain of edges passed to `deduce_chain()`.

    Kept lightweight so the public API matches the skeleton spec. The
    `edges` field is a list of `Semesome` instances (or any object with
    `type`, `weight`, `source`, `target` attributes).
    """
    edges: List[Semesome]
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────
# Rule definitions
# ─────────────────────────────────────────────────────────────────────

class _Rule:
    """Base class for a deductive rule.

    Subclasses override `matches(edges)` (returns a list of premise-tuples
    that should fire the rule) and `apply(premises)` (returns an
    `Inference` for each premise-tuple).

    Each rule receives the resolved t-norm callable at construction so
    that composition-style rules can combine premise weights through the
    configured t-norm rather than a hard-coded multiplication.
    """

    name: str = "BASE_RULE"

    def __init__(self, t_norm: Callable[[float, float], float] = product_tnorm):
        # Default to the legacy product t-norm so subclasses that don't
        # actually compose weights (e.g. DIFFERENTIAL_INVERSION) can
        # still accept the parameter without changing their behaviour.
        self._tnorm = t_norm

    def matches(self, edges: Sequence[Semesome]) -> List[tuple]:
        """Return a list of premise-tuples that should fire this rule."""
        raise NotImplementedError

    def apply(self, premises: tuple) -> Inference:
        """Apply the rule to one premise-tuple, returning an Inference."""
        raise NotImplementedError


class CategoricalTransitivity(_Rule):
    """A->B (CAT), B->C (CAT) => A->C (CAT, weight = t_norm(1.0, 1.0)).

    Under the default `"product"` t-norm this is `1.0 * 1.0 = 1.0`,
    matching the legacy semantics bit-for-bit.
    """

    name = "CATEGORICAL_TRANSITIVITY"

    def matches(self, edges: Sequence[Semesome]) -> List[tuple]:
        out: List[tuple] = []
        for i in range(len(edges) - 1):
            e1, e2 = edges[i], edges[i + 1]
            if (e1.type == CATEGORICAL and e2.type == CATEGORICAL
                    and e1.target == e2.source):
                out.append((e1, e2))
        return out

    def apply(self, premises: tuple) -> Inference:
        e1, e2 = premises
        weight = self._tnorm(e1.weight, e2.weight)
        conclusion = Semesome(
            type=CATEGORICAL,
            weight=weight,
            source=e1.source,
            target=e2.target,
        )
        return Inference(
            rule=self.name,
            premises=[e1, e2],
            conclusion=conclusion,
            weight=weight,
            note=f"{e1.source}->{e1.target} (CAT {e1.weight}), "
                 f"{e2.source}->{e2.target} (CAT {e2.weight}) "
                 f"=> {e1.source}->{e2.target} (CAT {weight})",
        )


class CausalChain(_Rule):
    """A->B (CAUSAL), B->C (CAUSAL) => A->C (CAUSAL, weight = t_norm(0.7, 0.7)).

    Under the default `"product"` t-norm this is `0.7 * 0.7 = 0.49`,
    matching the legacy semantics bit-for-bit.
    """

    name = "CAUSAL_CHAIN"

    def matches(self, edges: Sequence[Semesome]) -> List[tuple]:
        out: List[tuple] = []
        for i in range(len(edges) - 1):
            e1, e2 = edges[i], edges[i + 1]
            if (e1.type == CAUSAL and e2.type == CAUSAL
                    and e1.target == e2.source):
                out.append((e1, e2))
        return out

    def apply(self, premises: tuple) -> Inference:
        e1, e2 = premises
        weight = self._tnorm(e1.weight, e2.weight)
        conclusion = Semesome(
            type=CAUSAL,
            weight=weight,
            source=e1.source,
            target=e2.target,
        )
        return Inference(
            rule=self.name,
            premises=[e1, e2],
            conclusion=conclusion,
            weight=weight,
            note=f"{e1.source}->{e1.target} (CAUSAL {e1.weight}), "
                 f"{e2.source}->{e2.target} (CAUSAL {e2.weight}) "
                 f"=> {e1.source}->{e2.target} (CAUSAL {weight})",
        )


class DifferentialInversion(_Rule):
    """A->B (DIFF=-0.8) => B->A (DIFF=-0.8). Symmetric: weight preserved.

    This rule does NOT combine two premises — it is a unary inversion
    that preserves the original weight — so the t-norm parameter has no
    effect here. It is still accepted (and stored) for API uniformity.
    """

    name = "DIFFERENTIAL_INVERSION"

    def matches(self, edges: Sequence[Semesome]) -> List[tuple]:
        out: List[tuple] = []
        for e in edges:
            if e.type == DIFFERENTIAL:
                out.append((e,))
        return out

    def apply(self, premises: tuple) -> Inference:
        (e,) = premises
        weight = e.weight  # symmetric inversion keeps the same weight
        conclusion = Semesome(
            type=DIFFERENTIAL,
            weight=weight,
            source=e.target,   # swapped
            target=e.source,   # swapped
        )
        return Inference(
            rule=self.name,
            premises=[e],
            conclusion=conclusion,
            weight=weight,
            note=f"{e.source}->{e.target} (DIFF {e.weight}) "
                 f"=> {e.target}->{e.source} (DIFF {weight}) "
                 f"[symmetric inversion]",
        )


class CausalDifferentialConflict(_Rule):
    """A->B (CAUSAL) + A->B (DIFF) => conflict.

    Resolution: arithmetic mean of the two weights.
    Example: (0.7 + -0.8) / 2 = -0.05 (near zero = uncertain).

    Note: the arithmetic mean is *not* a t-norm (it is a recognised
    fuzzy aggregation operator but lacks the boundary / associativity
    properties of a t-norm). This rule therefore deliberately keeps its
    arithmetic-mean semantics regardless of the configured t-norm.
    """

    name = "CAUSAL_DIFFERENTIAL_CONFLICT"

    def matches(self, edges: Sequence[Semesome]) -> List[tuple]:
        out: List[tuple] = []
        seen_pairs: set = set()
        for i, e1 in enumerate(edges):
            for j, e2 in enumerate(edges):
                if i == j:
                    continue
                if (e1.source == e2.source and e1.target == e2.target
                        and {e1.type, e2.type} == {CAUSAL, DIFFERENTIAL}):
                    # Canonicalize pair ordering so we don't double-fire.
                    key = (i, j) if i < j else (j, i)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    # Always put the CAUSAL edge first in the tuple for
                    # deterministic weight aggregation.
                    if e1.type == CAUSAL:
                        out.append((e1, e2))
                    else:
                        out.append((e2, e1))
        return out

    def apply(self, premises: tuple) -> Inference:
        causal_edge, diff_edge = premises
        resolved = (causal_edge.weight + diff_edge.weight) / 2.0
        conclusion = Semesome(
            type=CAUSAL,  # resolution preserves CAUSAL type as the "winner"
            weight=resolved,
            source=causal_edge.source,
            target=causal_edge.target,
        )
        return Inference(
            rule=self.name,
            premises=[causal_edge, diff_edge],
            conclusion=conclusion,
            weight=resolved,
            note=f"CONFLICT on {causal_edge.source}->{causal_edge.target}: "
                 f"CAUSAL {causal_edge.weight} vs DIFFERENTIAL {diff_edge.weight} "
                 f"=> resolved weight = {resolved} "
                 f"(near zero = uncertain)",
        )


class FunctionalComposition(_Rule):
    """A->B (FUNC), B->C (FUNC) => A->C (FUNC, weight = t_norm(0.6, 0.6)).

    Under the default `"product"` t-norm this is `0.6 * 0.6 = 0.36`,
    matching the legacy semantics bit-for-bit.
    """

    name = "FUNCTIONAL_COMPOSITION"

    def matches(self, edges: Sequence[Semesome]) -> List[tuple]:
        out: List[tuple] = []
        for i in range(len(edges) - 1):
            e1, e2 = edges[i], edges[i + 1]
            if (e1.type == FUNCTIONAL and e2.type == FUNCTIONAL
                    and e1.target == e2.source):
                out.append((e1, e2))
        return out

    def apply(self, premises: tuple) -> Inference:
        e1, e2 = premises
        weight = self._tnorm(e1.weight, e2.weight)
        conclusion = Semesome(
            type=FUNCTIONAL,
            weight=weight,
            source=e1.source,
            target=e2.target,
        )
        return Inference(
            rule=self.name,
            premises=[e1, e2],
            conclusion=conclusion,
            weight=weight,
            note=f"{e1.source}->{e1.target} (FUNC {e1.weight}), "
                 f"{e2.source}->{e2.target} (FUNC {e2.weight}) "
                 f"=> {e1.source}->{e2.target} (FUNC {weight})",
        )


# ─────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────

class InferiorFrontalGyrus:
    """BA 44 deductive reasoning engine.

    Holds the five deductive rules and applies them to a sequence of
    `Semesome` edges. The rules are stateless, so a single
    `InferiorFrontalGyrus` instance can be reused across many `deduce()`
    calls.

    Args:
        t_norm: Which fuzzy-logic t-norm to use when a rule composes two
            premise weights into one inferred weight. One of:

                - "product"      (default)  →  T(a, b) = a * b
                - "lukasiewicz"             →  T(a, b) = max(0, a + b - 1)
                - "godel"                   →  T(a, b) = min(a, b)

            The default `"product"` reproduces the pre-refactor BA44
            arithmetic exactly (`0.7 * 0.7 = 0.49`, `0.6 * 0.6 = 0.36`,
            `1.0 * 1.0 = 1.0`). The other two options make BA44 a
            *conscious* fuzzy-logic engine whose conjunction operator is
            configurable, as recommended in
            `AGNN/docs/research-neuro-symbolic-reasoning.md` §4 / §B1.

            Note: the `CAUSAL_DIFFERENTIAL_CONFLICT` rule resolves its
            two premises via arithmetic mean, which is *not* a t-norm
            and is therefore unaffected by this parameter. Likewise,
            `DIFFERENTIAL_INVERSION` is a unary inversion and does not
            combine weights.
    """

    def __init__(self, t_norm: TNormKind = "product"):
        """Register the five deductive rules with the chosen t-norm."""
        tnorm_fn = _resolve_tnorm(t_norm)
        self.t_norm: TNormKind = t_norm
        self._tnorm_fn: Callable[[float, float], float] = tnorm_fn
        self.rules: List[_Rule] = [
            CategoricalTransitivity(tnorm_fn),
            CausalChain(tnorm_fn),
            DifferentialInversion(tnorm_fn),
            CausalDifferentialConflict(tnorm_fn),
            FunctionalComposition(tnorm_fn),
        ]
        # Lifetime counter of rule firings (for introspect / audit).
        self.rule_count: int = 0

    # ---- public API ------------------------------------------------

    def deduce(self, edges: Sequence[Semesome]) -> Deduction:
        """
        Apply all deductive rules to a sequence of edges.

        The edges are walked once per rule. Each rule collects its own
        premise-tuples and produces one `Inference` per tuple. All
        inferences are aggregated into a single `Deduction`.

        Args:
            edges: Sequence of `Semesome` instances (or any object exposing
                `type`, `weight`, `source`, `target`). Order matters for
                transitivity rules (A->B must come before B->C).

        Returns:
            Deduction with all rule firings + inferred edges.
        """
        edges = list(edges)  # defensive copy
        all_inferences: List[Inference] = []

        for rule in self.rules:
            for premises in rule.matches(edges):
                inference = rule.apply(premises)
                all_inferences.append(inference)
                self.rule_count += 1

        inferred_edges: List[Semesome] = [
            inf.conclusion for inf in all_inferences if inf.conclusion is not None
        ]
        applied_rules: List[str] = []
        seen: set = set()
        for inf in all_inferences:
            if inf.rule not in seen:
                seen.add(inf.rule)
                applied_rules.append(inf.rule)

        # Aggregate confidence: product of inferred weights clamped to [0, 1].
        # Negative weights (e.g. -0.8 from DIFFERENTIAL) drop confidence to 0.
        # Zero inferences => confidence 0.0.
        confidence = 1.0
        if not all_inferences:
            confidence = 0.0
        else:
            for inf in all_inferences:
                w = inf.weight
                if w <= 0:
                    confidence = 0.0
                    break
                confidence *= w
                if confidence <= 0:
                    break
            confidence = max(0.0, min(1.0, confidence))

        context_lines: List[str] = []
        context_lines.append(f"BA 44 deductive inference: {len(all_inferences)} firings")
        for inf in all_inferences:
            context_lines.append(f"  [{inf.rule}] {inf.note}")
        context = "\n".join(context_lines)

        return Deduction(
            inferences=all_inferences,
            inferred_edges=inferred_edges,
            rule_count=len(all_inferences),
            applied_rules=applied_rules,
            confidence=confidence,
            context=context,
        )

    def deduce_chain(self, chain: EdgeChain) -> Deduction:
        """
        Backwards-compatible entry point: deduce over an `EdgeChain`.

        Args:
            chain: EdgeChain with `.edges` list of Semesome.

        Returns:
            Same as `deduce(chain.edges)`.
        """
        return self.deduce(chain.edges)
