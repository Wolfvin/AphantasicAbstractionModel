"""
AAM Layer 3 — Hypothesis-Driven Active Reasoning

Inspirasi: Jin Sowoon (진소운) dari "모든걸 기억하는 천재무사"
(The Martial Genius Who Remembers Everything)

Jin Sowoon tidak hanya mengingat — dia MENGUJI. Ketika dia punya hipotesis
"X adalah musuh", dia secara aktif mencari bukti yang MUNGKIN MENYALAHKAN
hipotesis itu. Ini adalah **disconfirmatory reasoning** — mencari bukti yang
bisa membantah, bukan hanya yang mendukung.

Core Cycle:
    ANOMALY → HYPOTHESIZE → TEST (confirm + disconfirm) → REVISE → REPEAT

    1. ANOMALY    : PredictiveEngine detects anomaly → trigger
    2. HYPOTHESIZE: Generate 2-3 alternative hypotheses from anomaly
    3. TEST       : For each hypothesis, seek confirmatory AND disconfirmatory evidence
    4. REVISE     : Update confidence per hypothesis based on test results
    5. CONCLUDE   : If one hypothesis dominates → accept; if all weak → generate new hypotheses

Architecture:
    HypothesisDrivenReasoner WRAPS ReasoningEngine — it does NOT replace it.
    The engine handles the raw deductive chain; this module orchestrates the
    active hypothesis-testing cycle on top of it.

Integration:
    - Layer 2 PredictiveEngine: Anomaly → Hypothesis trigger
    - Layer 2 PatternOutput: PatternResult → Evidence source for testing
    - Layer 3 ReasoningEngine: DeductiveChain → Hypothesis testing backbone
    - RSVS Bridge: mcts_query() → Disconfirmatory search path
    - RSVS Bridge: appraise() → Evidence evaluation
    - RSVS Bridge: structural_similarity() → Hypothesis comparison

Supported by Compositional Neuro-Symbolic + Active Inference Tree Search:
    - Decomposition: hypothesis → sub-hypotheses (compositional)
    - Recomposition: sub-results → combined hypothesis confidence
    - Expected Free Energy: prioritize exploration that reduces uncertainty
    - Epistemic Value: prefer evidence that discriminates between hypotheses
    - Pragmatic Value: prefer evidence that supports actionable conclusions

Analogi: Jin Sowoon menemukan anomali "pencuri Snow Plum Pill tidak
mengonsumsi pil" → dia buat hipotesis: (1) Ju Jangmok bukan pencuri,
(2) pil yang dicuri bukan Snow Plum Pill asli, (3) ada pihak ketiga.
Lalu dia AKTIF mencari bukti yang bisa MENGHAPUS setiap hipotesis —
bukan hanya yang mendukung. Hipotesis yang paling tahan terhadap
pembantahan adalah yang paling mungkin benar.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# Layer 2 & 3 imports
from layer2.bridge import RsvsBridge, get_bridge
from layer2.predictive import Anomaly, PredictiveEngine
from layer2.pattern import PatternResult, PatternOutput, ReasoningStep
from layer3.reasoning import ReasoningEngine, DeductiveChain, DeductiveStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

# Minimum confidence difference between best and second-best hypothesis
# to declare a winner (conclusive result)
_DEFAULT_DECISIVENESS_THRESHOLD = 0.20

# Maximum number of hypothesis-testing cycles before forced conclusion
_DEFAULT_MAX_CYCLES = 5

# Minimum confidence for a hypothesis to be considered "viable"
_DEFAULT_VIABILITY_THRESHOLD = 0.15

# Number of alternative hypotheses to generate per anomaly
_DEFAULT_HYPOTHESIS_COUNT = 3

# Weight of confirmatory evidence (vs disconfirmatory)
# > 0.5 means confirmatory evidence is weighted more; < 0.5 favors disconfirmation
_DEFAULT_CONFIRMATORY_WEIGHT = 0.4

# Weight of disconfirmatory evidence — deliberately higher than confirmatory
# to embody the principle that "absence of expected evidence is evidence of absence"
_DEFAULT_DISCONFIRMATORY_WEIGHT = 0.6

# How much a single piece of disconfirmatory evidence reduces confidence
_DEFAULT_DISCONFIRM_IMPACT = 0.15

# How much a single piece of confirmatory evidence increases confidence
_DEFAULT_CONFIRM_IMPACT = 0.10

# Minimum number of evidence items before a hypothesis can be concluded
_DEFAULT_MIN_EVIDENCE_FOR_CONCLUSION = 2

# Valid hypothesis states
# Extended with hybrid/emergent states for Possibility Lattice support
_VALID_HYPOTHESIS_STATES = frozenset({
    "proposed", "testing", "confirmed", "refuted", "inconclusive", "superseded",
    "hybrid",       # Created from hybridization of two hypotheses (A × B)
    "emergent",     # Emerged from implication tracing of a hybrid
    "surviving",    # Survived elimination round in lattice mode
    "eliminated",   # Eliminated in lattice mode (below confidence threshold)
    "concluded",    # Final conclusion in lattice mode
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """A single piece of evidence for or against a hypothesis.

    Evidence can be confirmatory (supports the hypothesis) or
    disconfirmatory (undermines the hypothesis). Disconfirmatory
    evidence is the key innovation — it represents the Jin Sowoon
    principle of actively seeking evidence that could refute.

    Attributes:
        evidence_id: Unique identifier for this evidence.
        description: What this evidence says.
        source_node: RSVS node label where this evidence was found.
        source_sense: RSVS sense ID of the evidence source.
        direction: "confirmatory" or "disconfirmatory".
        strength: How strongly this evidence supports/undermines (0.0–1.0).
        grounding_score: RSVS grounding score of the source node.
        discovery_method: How this evidence was found
            ("mcts_search", "appraise", "structural_similarity",
             "pattern_completion", "active_test").
    """

    evidence_id: str
    description: str
    source_node: str = ""
    source_sense: str = "0"
    direction: str = "confirmatory"  # "confirmatory" | "disconfirmatory"
    strength: float = 0.5
    grounding_score: float = 0.5
    discovery_method: str = "unknown"

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "evidence_id": self.evidence_id,
            "description": self.description,
            "source_node": self.source_node,
            "source_sense": self.source_sense,
            "direction": self.direction,
            "strength": round(self.strength, 4),
            "grounding_score": round(self.grounding_score, 4),
            "discovery_method": self.discovery_method,
        }


@dataclass
class Hypothesis:
    """A testable hypothesis in the active reasoning cycle.

    Unlike a simple prediction (which is "I expect X"), a hypothesis
    is "I believe X because Y, and I can test it by checking Z".
    Each hypothesis tracks both confirmatory and disconfirmatory
    evidence, and its confidence is updated asymmetrically —
    disconfirmatory evidence has stronger impact.

    Jin Sowoon Principle: A hypothesis that has survived multiple
    attempts at disconfirmation is stronger than one supported only
    by confirmatory evidence.

    Attributes:
        hypothesis_id: Unique identifier for this hypothesis.
        statement: The hypothesis statement (e.g., "Ju Jangmok is a scapegoat").
        reasoning: Why this hypothesis might be true.
        test_criteria: What evidence would confirm or disconfirm this hypothesis.
        confidence: Current confidence level (0.0–1.0).
        confirmatory_evidence: Evidence supporting this hypothesis.
        disconfirmatory_evidence: Evidence undermining this hypothesis.
        state: Current state of the hypothesis lifecycle.
        anomaly_source: The anomaly that triggered this hypothesis.
        parent_hypothesis_id: ID of the parent hypothesis (for revision chains).
        created_at: When this hypothesis was created.
        tested_at: When this hypothesis was last tested.
        cycle_count: How many test cycles this hypothesis has undergone.
    """

    hypothesis_id: str
    statement: str
    reasoning: str = ""
    test_criteria: list[str] = field(default_factory=list)
    confidence: float = 0.5
    confirmatory_evidence: list[Evidence] = field(default_factory=list)
    disconfirmatory_evidence: list[Evidence] = field(default_factory=list)
    state: str = "proposed"
    anomaly_source: str = ""
    parent_hypothesis_id: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    tested_at: str = ""
    cycle_count: int = 0
    # Lattice mode extensions
    parent_ids: list[str] = field(default_factory=list)  # For hybrid lineage (2 parents = hybrid)
    generation: int = 0  # Lattice generation number (0 = initial)
    source: str = "anomaly"  # "anomaly", "context", "input", "hybrid", "emergent"

    @property
    def is_hybrid(self) -> bool:
        """Whether this hypothesis was created from hybridization."""
        return len(self.parent_ids) >= 2

    def __post_init__(self) -> None:
        """Validate initial state."""
        if self.state not in _VALID_HYPOTHESIS_STATES:
            raise ValueError(f"Invalid hypothesis state: {self.state!r}")

    @property
    def total_evidence_count(self) -> int:
        """Total number of evidence items (both directions)."""
        return len(self.confirmatory_evidence) + len(self.disconfirmatory_evidence)

    @property
    def net_evidence_score(self) -> float:
        """Weighted evidence score: positive = net support, negative = net undermine.

        Disconfirmatory evidence is weighted more heavily (0.6 vs 0.4)
        to embody the principle that surviving disconfirmation is more
        valuable than mere confirmation.
        """
        confirm_total = sum(
            e.strength * e.grounding_score
            for e in self.confirmatory_evidence
        )
        disconfirm_total = sum(
            e.strength * e.grounding_score
            for e in self.disconfirmatory_evidence
        )
        return (
            _DEFAULT_CONFIRMATORY_WEIGHT * confirm_total
            - _DEFAULT_DISCONFIRMATORY_WEIGHT * disconfirm_total
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "reasoning": self.reasoning,
            "test_criteria": list(self.test_criteria),
            "confidence": round(self.confidence, 4),
            "confirmatory_evidence": [e.to_dict() for e in self.confirmatory_evidence],
            "disconfirmatory_evidence": [e.to_dict() for e in self.disconfirmatory_evidence],
            "state": self.state,
            "anomaly_source": self.anomaly_source,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "created_at": self.created_at,
            "tested_at": self.tested_at,
            "cycle_count": self.cycle_count,
            "total_evidence_count": self.total_evidence_count,
            "net_evidence_score": round(self.net_evidence_score, 4),
            # Lattice mode extensions
            "parent_ids": list(self.parent_ids),
            "generation": self.generation,
            "source": self.source,
            "is_hybrid": self.is_hybrid,
        }


@dataclass
class HypothesisCycleResult:
    """Result from one complete hypothesis-testing cycle.

    Contains the full traceability of what was hypothesized, what
    evidence was found (both confirmatory and disconfirmatory),
    how confidences changed, and whether the cycle produced a
    conclusive result or requires further testing.

    Attributes:
        cycle_id: Unique identifier for this cycle.
        anomaly: The anomaly that triggered this cycle.
        hypotheses: The hypotheses generated and tested.
        winning_hypothesis: The hypothesis with highest confidence (if conclusive).
        is_conclusive: Whether the cycle produced a clear winner.
        decisiveness: Confidence gap between best and second-best hypothesis.
        reasoning_chain: The DeductiveChain produced during testing.
        cycle_number: Which cycle this is (1-based).
        total_evidence_found: Total evidence found across all hypotheses.
    """

    cycle_id: str
    anomaly: Anomaly | None = None
    hypotheses: list[Hypothesis] = field(default_factory=list)
    winning_hypothesis: Hypothesis | None = None
    is_conclusive: bool = False
    decisiveness: float = 0.0
    reasoning_chain: DeductiveChain | None = None
    cycle_number: int = 1
    total_evidence_found: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "cycle_id": self.cycle_id,
            "anomaly": self.anomaly.to_dict() if self.anomaly else None,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "winning_hypothesis": (
                self.winning_hypothesis.to_dict() if self.winning_hypothesis else None
            ),
            "is_conclusive": self.is_conclusive,
            "decisiveness": round(self.decisiveness, 4),
            "reasoning_chain": (
                self.reasoning_chain.to_dict() if self.reasoning_chain else None
            ),
            "cycle_number": self.cycle_number,
            "total_evidence_found": self.total_evidence_found,
        }


# ---------------------------------------------------------------------------
# HypothesisDrivenReasoner
# ---------------------------------------------------------------------------

class HypothesisDrivenReasoner:
    """Hypothesis-Driven Active Reasoning — the Jin Sowoon reasoning engine.

    This is the core innovation of AAM's deductive reasoning upgrade.
    Instead of the passive pipeline (EXTRACT → COMPOSE → GROUND → EXPLORE → CONCLUDE),
    this engine implements an ACTIVE cycle:

        ANOMALY → HYPOTHESIZE → TEST (confirm + disconfirm) → REVISE → REPEAT

    Key principles:
    1. **Disconfirmatory Search**: For each hypothesis, actively seek evidence
       that could REFUTE it — not just evidence that supports it.
    2. **Asymmetric Confidence Update**: Disconfirmatory evidence has stronger
       impact than confirmatory evidence (0.6 vs 0.4 weight).
    3. **Hypothesis Competition**: Multiple hypotheses compete; the one that
       survives the most disconfirmation wins.
    4. **Epistemic Value**: Prioritize exploring evidence that best discriminates
       between competing hypotheses (maximizes information gain).
    5. **Compositional Decomposition**: Complex hypotheses are decomposed into
       sub-hypotheses that can be tested independently.

    This module WRAPS ReasoningEngine — it does NOT replace it. The engine
    handles the raw deductive chain; this module orchestrates the active
    hypothesis-testing cycle on top of it.

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected.
        is_rust_core: Whether the Rust core is being used.
    """

    def __init__(
        self,
        bridge: Optional[RsvsBridge] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        predictive_engine: Optional[PredictiveEngine] = None,
        *,
        decisiveness_threshold: float = _DEFAULT_DECISIVENESS_THRESHOLD,
        max_cycles: int = _DEFAULT_MAX_CYCLES,
        viability_threshold: float = _DEFAULT_VIABILITY_THRESHOLD,
        hypothesis_count: int = _DEFAULT_HYPOTHESIS_COUNT,
    ) -> None:
        """Initialize the Hypothesis-Driven Active Reasoner.

        Args:
            bridge: Optional pre-built RsvsBridge. If None, one is created.
            reasoning_engine: Optional pre-built ReasoningEngine.
                If None, one is created using the bridge.
            predictive_engine: Optional pre-built PredictiveEngine.
                If None, one is created using the bridge.
            decisiveness_threshold: Minimum confidence gap to declare a winner.
            max_cycles: Maximum test cycles before forced conclusion.
            viability_threshold: Minimum confidence for a viable hypothesis.
            hypothesis_count: Number of alternative hypotheses to generate.
        """
        if bridge is not None:
            self._bridge = bridge
        elif reasoning_engine is not None:
            self._bridge = reasoning_engine._bridge
        elif predictive_engine is not None:
            self._bridge = predictive_engine._bridge
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core

        # Wrap or create the underlying reasoning engine
        if reasoning_engine is not None:
            self._reasoning_engine = reasoning_engine
        else:
            self._reasoning_engine = ReasoningEngine(bridge=self._bridge)

        # Wrap or create the predictive engine for anomaly detection
        if predictive_engine is not None:
            self._predictive_engine = predictive_engine
        else:
            self._predictive_engine = PredictiveEngine(bridge=self._bridge)

        # Configuration
        self._decisiveness_threshold = decisiveness_threshold
        self._max_cycles = max_cycles
        self._viability_threshold = viability_threshold
        self._hypothesis_count = hypothesis_count

        # Active hypothesis tracking
        self._active_hypotheses: dict[str, Hypothesis] = {}
        self._cycle_history: list[HypothesisCycleResult] = []

        logger.info(
            "HypothesisDrivenReasoner initialized "
            "(decisiveness=%.2f, max_cycles=%d, viability=%.2f, "
            "n_hypotheses=%d, rsvs=%s)",
            decisiveness_threshold, max_cycles, viability_threshold,
            hypothesis_count, self.rsvs_available,
        )

    # ==================================================================
    # MAIN METHOD — reason()
    # ==================================================================

    def reason(
        self,
        anomaly: Anomaly,
        context: list[str] | None = None,
        pattern_result: PatternResult | None = None,
    ) -> HypothesisCycleResult:
        """Run the full hypothesis-driven reasoning cycle.

        This is the primary entry point. Takes an anomaly (from
        PredictiveEngine), generates alternative hypotheses, tests
        them with both confirmatory and disconfirmatory evidence,
        and returns a conclusion.

        Steps:
        1. HYPOTHESIZE — Generate alternative hypotheses from the anomaly
        2. TEST — For each hypothesis, find confirmatory AND disconfirmatory evidence
        3. SCORE — Update confidence per hypothesis based on evidence
        4. CONCLUDE — Check if one hypothesis dominates; if not, iterate
        5. ITERATE — Repeat test-score-conclude up to max_cycles

        Analogi: Jin Sowoon menemukan anomali "pencuri tidak mengonsumsi pil"
        → dia buat 3 hipotesis → dia cari bukti yang MENDUKUNG dan yang
        MENGHAPUS setiap hipotesis → hipotesis yang paling tahan pembantahan menang.

        Args:
            anomaly: The anomaly to reason about.
            context: Optional context atoms for hypothesis generation.
            pattern_result: Optional PatternResult for evidence extraction.

        Returns:
            A HypothesisCycleResult with the full reasoning trace.
        """
        context = context or []
        cycle_id = uuid.uuid4().hex[:8]

        logger.info(
            "HypothesisDrivenReasoner.reason(): anomaly='%s' (delta=%.3f)",
            anomaly.concept, anomaly.delta,
        )

        # ---- Step 1: HYPOTHESIZE ----
        hypotheses = self._generate_hypotheses(anomaly, context)

        # ---- Steps 2-4: TEST → SCORE → CONCLUDE (with iteration) ----
        best_result = None
        reasoning_chain = None

        # Build initial PatternResult for reasoning engine if not provided
        if pattern_result is None:
            pattern_result = self._build_pattern_from_anomaly(anomaly, context)

        for cycle_num in range(1, self._max_cycles + 1):
            logger.debug(
                "Cycle %d/%d: testing %d hypotheses",
                cycle_num, self._max_cycles, len(hypotheses),
            )

            # Step 2: TEST — Find confirmatory AND disconfirmatory evidence
            for hyp in hypotheses:
                if hyp.state in ("confirmed", "refuted", "superseded"):
                    continue  # Skip concluded hypotheses
                self._test_hypothesis(hyp, context, pattern_result)

            # Step 3: SCORE — Update confidence based on evidence
            for hyp in hypotheses:
                if hyp.state in ("confirmed", "refuted", "superseded"):
                    continue
                self._update_hypothesis_confidence(hyp)
                hyp.cycle_count += 1
                hyp.tested_at = time.strftime("%Y-%m-%dT%H:%M:%S")

            # Build reasoning chain for this cycle
            try:
                reasoning_chain = self._reasoning_engine.build_chain(pattern_result)
            except Exception as exc:
                logger.debug("build_chain() failed in hypothesis cycle: %s", exc)

            # Step 4: CONCLUDE — Check for a winner
            result = self._evaluate_hypotheses(
                hypotheses, anomaly, cycle_id, cycle_num, reasoning_chain,
            )

            best_result = result

            if result.is_conclusive:
                logger.info(
                    "Cycle %d CONCLUSIVE: winner='%s' (confidence=%.3f, "
                    "decisiveness=%.3f)",
                    cycle_num,
                    result.winning_hypothesis.statement[:60] if result.winning_hypothesis else "none",
                    result.winning_hypothesis.confidence if result.winning_hypothesis else 0.0,
                    result.decisiveness,
                )
                break

            # Step 5: ITERATE — Generate revised hypotheses if inconclusive
            if cycle_num < self._max_cycles:
                hypotheses = self._revise_hypotheses(hypotheses, anomaly, context)

        # Store in history
        if best_result is not None:
            self._cycle_history.append(best_result)
            # Update active hypotheses
            for hyp in best_result.hypotheses:
                self._active_hypotheses[hyp.hypothesis_id] = hyp

        return best_result or HypothesisCycleResult(
            cycle_id=cycle_id,
            anomaly=anomaly,
            hypotheses=hypotheses,
            is_conclusive=False,
            cycle_number=0,
        )

    # ==================================================================
    # Step 1: HYPOTHESIZE — Generate alternative hypotheses
    # ==================================================================

    def _generate_hypotheses(
        self,
        anomaly: Anomaly,
        context: list[str],
    ) -> list[Hypothesis]:
        """Generate alternative hypotheses from an anomaly.

        Strategy:
        1. Use the anomaly's expected vs observed gap to generate hypotheses
        2. Use RSVS relate() to find related concepts as hypothesis bases
        3. Use structural_similarity to find alternative interpretations
        4. Generate at least 3 competing hypotheses

        Analogi: Jin Sowoon melihat anomali "tidak ada konsumsi pil" →
        dia pikirkan: (1) orang ini bukan pencuri, (2) pilnya bukan asli,
        (3) ada dalang lain. Setiap hipotesis punya alasan dan kriteria uji.

        Args:
            anomaly: The anomaly to hypothesize about.
            context: Context atoms for hypothesis generation.

        Returns:
            A list of Hypothesis objects (at least 2, up to hypothesis_count).
        """
        hypotheses: list[Hypothesis] = []
        concept = anomaly.concept
        expected = anomaly.expected
        observed = anomaly.observed
        description = anomaly.description

        # --- Hypothesis 1: Negation hypothesis ---
        # "The expected is wrong" — what we expected doesn't hold
        negation_id = uuid.uuid4().hex[:8]
        negation_stmt = (
            f"The expectation about '{concept}' is incorrect: "
            f"expected {expected[:3]} but observed {observed[:3]}"
        )
        negation_reasoning = (
            f"Anomaly detected with delta={anomaly.delta:.3f}. "
            f"The original expectation does not match observation, "
            f"suggesting the premise may be wrong."
        )
        negation_tests = self._generate_test_criteria(concept, "negation", expected, observed)

        hypotheses.append(Hypothesis(
            hypothesis_id=negation_id,
            statement=negation_stmt,
            reasoning=negation_reasoning,
            test_criteria=negation_tests,
            confidence=0.5,  # Start neutral
            state="proposed",
            anomaly_source=concept,
        ))

        # --- Hypothesis 2: Alternative explanation ---
        # "Something else explains the observation"
        alt_id = uuid.uuid4().hex[:8]
        alt_stmt = (
            f"An alternative factor explains the anomaly for '{concept}': "
            f"the observation {observed[:3]} is caused by something unexpected"
        )
        alt_reasoning = (
            f"The gap between expected and observed suggests an "
            f"unaccounted factor. The observation is real but the "
            f"cause differs from the original expectation."
        )
        alt_tests = self._generate_test_criteria(concept, "alternative", expected, observed)

        hypotheses.append(Hypothesis(
            hypothesis_id=alt_id,
            statement=alt_stmt,
            reasoning=alt_reasoning,
            test_criteria=alt_tests,
            confidence=0.4,  # Slightly lower — needs more support
            state="proposed",
            anomaly_source=concept,
        ))

        # --- Hypothesis 3+: RSVS-informed hypotheses ---
        # Use RSVS to find alternative interpretations
        rsvs_hypotheses = self._generate_rsvs_hypotheses(
            concept, expected, observed, description, context,
        )
        hypotheses.extend(rsvs_hypotheses)

        # Ensure we don't exceed the configured count
        hypotheses = hypotheses[:self._hypothesis_count]

        logger.info(
            "Generated %d hypotheses for anomaly '%s'",
            len(hypotheses), concept,
        )

        return hypotheses

    def _generate_test_criteria(
        self,
        concept: str,
        hypothesis_type: str,
        expected: list[str],
        observed: list[str],
    ) -> list[str]:
        """Generate test criteria for a hypothesis.

        Test criteria describe what evidence would confirm or disconfirm
        the hypothesis. This is the key difference from simple prediction —
        we explicitly state what we're looking for BEFORE testing.

        Args:
            concept: The concept under investigation.
            hypothesis_type: "negation", "alternative", or "rsvs_generated".
            expected: What was expected.
            observed: What was actually observed.

        Returns:
            A list of test criteria strings.
        """
        criteria: list[str] = []

        if hypothesis_type == "negation":
            # For negation: if expected is wrong, what would prove/disprove it?
            for exp in expected[:3]:
                criteria.append(
                    f"Find evidence that '{exp}' is NOT true for '{concept}'"
                )
            for obs in observed[:3]:
                criteria.append(
                    f"Find evidence that '{obs}' directly contradicts the original expectation"
                )
            # Disconfirmatory criterion: what would prove the original expectation RIGHT?
            criteria.append(
                f"Find evidence that the original expectation about '{concept}' "
                f"IS correct despite the anomaly"
            )

        elif hypothesis_type == "alternative":
            # For alternative: what would support a different explanation?
            for obs in observed[:3]:
                criteria.append(
                    f"Find what caused '{obs}' if not the expected reason"
                )
            criteria.append(
                f"Find nodes related to '{concept}' that have different "
                f"compositions than expected"
            )
            # Disconfirmatory: what would prove there's NO alternative?
            criteria.append(
                f"Find evidence that no alternative explanation exists "
                f"for the anomaly in '{concept}'"
            )

        elif hypothesis_type == "rsvs_generated":
            criteria.append(
                f"Verify if the structural relationship around '{concept}' "
                f"supports this interpretation"
            )
            criteria.append(
                f"Find disconfirming evidence — nodes that contradict "
                f"this interpretation of '{concept}'"
            )

        return criteria

    def _generate_rsvs_hypotheses(
        self,
        concept: str,
        expected: list[str],
        observed: list[str],
        description: str,
        context: list[str],
    ) -> list[Hypothesis]:
        """Generate hypotheses informed by RSVS graph structure.

        Uses relate(), structural_similarity(), and mcts_query() to find
        alternative interpretations of the anomaly.

        Args:
            concept: The anomalous concept.
            expected: What was expected.
            observed: What was observed.
            description: Anomaly description.
            context: Context atoms.

        Returns:
            A list of additional Hypothesis objects.
        """
        hypotheses: list[Hypothesis] = []

        if not self.rsvs_available:
            return hypotheses

        # Strategy 1: Use relate() to find related concepts as hypothesis bases
        try:
            relate_result = self._bridge.relate(concept)
            if relate_result and isinstance(relate_result, dict):
                related_nodes = relate_result.get("related_nodes", [])
                for node_entry in related_nodes[:3]:
                    if isinstance(node_entry, (list, tuple)) and len(node_entry) >= 1:
                        related_label = str(node_entry[0])
                    elif isinstance(node_entry, str):
                        related_label = node_entry
                    else:
                        continue

                    hyp_id = uuid.uuid4().hex[:8]
                    hyp_stmt = (
                        f"'{related_label}' is causally linked to the anomaly "
                        f"in '{concept}' — it may explain why expected differs from observed"
                    )
                    hyp_reasoning = (
                        f"RSVS relate() found a structural connection between "
                        f"'{concept}' and '{related_label}'. This connection may "
                        f"explain the anomaly through a causal chain."
                    )
                    hyp_tests = self._generate_test_criteria(
                        concept, "rsvs_generated", expected, observed,
                    )

                    hypotheses.append(Hypothesis(
                        hypothesis_id=hyp_id,
                        statement=hyp_stmt,
                        reasoning=hyp_reasoning,
                        test_criteria=hyp_tests,
                        confidence=0.35,  # Lower — needs RSVS evidence
                        state="proposed",
                        anomaly_source=concept,
                    ))
        except Exception as exc:
            logger.debug("relate() hypothesis generation failed: %s", exc)

        # Strategy 2: Use structural_similarity between expected and observed
        if expected and observed:
            try:
                for exp in expected[:2]:
                    for obs in observed[:2]:
                        sim = self._bridge.structural_similarity(exp, obs)
                        if sim and isinstance(sim, dict):
                            sim_val = sim.get("structural_similarity", 0.0)
                            if isinstance(sim_val, (int, float)) and float(sim_val) < 0.3:
                                # Low similarity → they are fundamentally different
                                hyp_id = uuid.uuid4().hex[:8]
                                hyp_stmt = (
                                    f"'{exp}' and '{obs}' are structurally different "
                                    f"(sim={float(sim_val):.2f}), suggesting the "
                                    f"anomaly in '{concept}' is due to a category error"
                                )
                                hyp_reasoning = (
                                    f"Structural similarity between '{exp}' and '{obs}' "
                                    f"is only {float(sim_val):.2f}, indicating they belong "
                                    f"to different categories. The anomaly may stem from "
                                    f"mistaking one category for another."
                                )
                                hypotheses.append(Hypothesis(
                                    hypothesis_id=hyp_id,
                                    statement=hyp_stmt,
                                    reasoning=hyp_reasoning,
                                    test_criteria=[
                                        f"Verify that '{exp}' and '{obs}' are "
                                        f"indeed different categories in the graph",
                                        f"Find evidence that '{concept}' was "
                                        f"misclassified",
                                    ],
                                    confidence=0.4,
                                    state="proposed",
                                    anomaly_source=concept,
                                ))
            except Exception as exc:
                logger.debug("structural_similarity hypothesis generation failed: %s", exc)

        return hypotheses[:2]  # Cap at 2 additional hypotheses

    # ==================================================================
    # Step 2: TEST — Find confirmatory AND disconfirmatory evidence
    # ==================================================================

    def _test_hypothesis(
        self,
        hypothesis: Hypothesis,
        context: list[str],
        pattern_result: PatternResult | None = None,
    ) -> None:
        """Test a hypothesis by seeking both confirmatory and disconfirmatory evidence.

        This is the core of the Jin Sowoon method: for each hypothesis,
        we don't just look for supporting evidence — we ACTIVELY seek
        evidence that could REFUTE it.

        Evidence sources:
        1. Confirmatory: RSVS senses(), relate(), structural_similarity()
        2. Disconfirmatory: RSVS appraise() (disagreement), MCTS search
           for contradicting nodes, structural_similarity (mismatch)

        Args:
            hypothesis: The hypothesis to test.
            context: Context atoms for evidence search.
            pattern_result: Optional PatternResult for evidence extraction.
        """
        hypothesis.state = "testing"

        # --- Find CONFIRMATORY evidence ---
        confirmatory = self._find_confirmatory_evidence(hypothesis, context)
        hypothesis.confirmatory_evidence.extend(confirmatory)

        # --- Find DISCONFIRMATORY evidence ---
        # THIS IS THE KEY INNOVATION — Jin Sowoon actively seeks refutation
        disconfirmatory = self._find_disconfirmatory_evidence(hypothesis, context)
        hypothesis.disconfirmatory_evidence.extend(disconfirmatory)

        # --- Extract evidence from PatternResult if available ---
        if pattern_result is not None:
            pattern_evidence = self._extract_pattern_evidence(hypothesis, pattern_result)
            for ev in pattern_evidence:
                if ev.direction == "confirmatory":
                    hypothesis.confirmatory_evidence.append(ev)
                else:
                    hypothesis.disconfirmatory_evidence.append(ev)

        logger.debug(
            "Hypothesis '%s' tested: +%d confirmatory, -%d disconfirmatory",
            hypothesis.statement[:40],
            len(confirmatory),
            len(disconfirmatory),
        )

    def _find_confirmatory_evidence(
        self,
        hypothesis: Hypothesis,
        context: list[str],
    ) -> list[Evidence]:
        """Find evidence that SUPPORTS the hypothesis.

        Uses RSVS senses(), relate(), and structural_similarity() to find
        nodes and relationships that are consistent with the hypothesis.

        Args:
            hypothesis: The hypothesis to find supporting evidence for.
            context: Context atoms.

        Returns:
            A list of confirmatory Evidence objects.
        """
        evidence: list[Evidence] = []
        concept = hypothesis.anomaly_source

        if not self.rsvs_available:
            return evidence

        # Strategy 1: Use senses() to find grounding evidence
        try:
            senses = self._bridge.senses(concept)
            if senses and isinstance(senses, list):
                for sense in senses[:3]:
                    if isinstance(sense, dict):
                        sense_idx = str(sense.get("sense_idx", 0))
                        gs = sense.get("grounding_score", 0.5)
                        core_atoms = sense.get("core_atoms", [])
                        if isinstance(core_atoms, list) and core_atoms:
                            ev_id = uuid.uuid4().hex[:8]
                            atom_labels = [
                                a[0] if isinstance(a, (list, tuple)) else str(a)
                                for a in core_atoms[:5]
                            ]
                            evidence.append(Evidence(
                                evidence_id=ev_id,
                                description=(
                                    f"RSVS sense {sense_idx} of '{concept}' contains "
                                    f"compositions: {', '.join(str(a) for a in atom_labels[:3])} "
                                    f"— consistent with hypothesis"
                                ),
                                source_node=concept,
                                source_sense=sense_idx,
                                direction="confirmatory",
                                strength=min(1.0, gs * 1.2),
                                grounding_score=gs,
                                discovery_method="senses",
                            ))
        except Exception as exc:
            logger.debug("senses() confirmatory search failed: %s", exc)

        # Strategy 2: Use relate() to find supporting connections
        try:
            relate_result = self._bridge.relate(concept)
            if relate_result and isinstance(relate_result, dict):
                related = relate_result.get("related_nodes", [])
                for node_entry in related[:5]:
                    if isinstance(node_entry, (list, tuple)) and len(node_entry) >= 2:
                        related_label = str(node_entry[0])
                        related_conf = float(node_entry[1]) if isinstance(node_entry[1], (int, float)) else 0.5
                    elif isinstance(node_entry, str):
                        related_label = node_entry
                        related_conf = 0.5
                    else:
                        continue

                    # Check if this related node supports the hypothesis
                    appraise_result = None
                    try:
                        statement = f"{related_label} supports the hypothesis that {hypothesis.statement[:50]}"
                        appraise_result = self._bridge.appraise(statement)
                    except Exception:
                        pass

                    # If appraise agrees or is neutral, consider it supporting
                    is_supporting = True
                    if appraise_result and isinstance(appraise_result, dict):
                        disagree_pct = appraise_result.get("disagree_pct", 0)
                        if isinstance(disagree_pct, (int, float)) and float(disagree_pct) > 0.6:
                            is_supporting = False

                    if is_supporting:
                        ev_id = uuid.uuid4().hex[:8]
                        evidence.append(Evidence(
                            evidence_id=ev_id,
                            description=(
                                f"Related node '{related_label}' is connected to "
                                f"'{concept}' (confidence={related_conf:.2f}) — "
                                f"supports the hypothesis"
                            ),
                            source_node=related_label,
                            direction="confirmatory",
                            strength=related_conf,
                            grounding_score=related_conf,
                            discovery_method="relate",
                        ))
        except Exception as exc:
            logger.debug("relate() confirmatory search failed: %s", exc)

        return evidence

    def _find_disconfirmatory_evidence(
        self,
        hypothesis: Hypothesis,
        context: list[str],
    ) -> list[Evidence]:
        """Find evidence that UNDERMINES the hypothesis.

        THIS IS THE KEY INNOVATION. Instead of only looking for supporting
        evidence, we actively search for evidence that could REFUTE the
        hypothesis. This is the Jin Sowoon principle: "A hypothesis that
        survives disconfirmation is stronger than one merely confirmed."

        Uses:
        - RSVS appraise() with NEGATIVE framing to find contradictions
        - RSVS mcts_query() to explore reasoning paths that could refute
        - Structural similarity to find mismatched expectations

        Args:
            hypothesis: The hypothesis to find undermining evidence for.
            context: Context atoms.

        Returns:
            A list of disconfirmatory Evidence objects.
        """
        evidence: list[Evidence] = []
        concept = hypothesis.anomaly_source

        if not self.rsvs_available:
            return evidence

        # Strategy 1: Use appraise() with NEGATIVE framing
        # Ask "Is this hypothesis WRONG?" instead of "Is it right?"
        try:
            negative_statement = (
                f"The hypothesis '{hypothesis.statement[:80]}' is incorrect"
            )
            appraise_result = self._bridge.appraise(negative_statement)

            if appraise_result and isinstance(appraise_result, dict):
                agree_pct = appraise_result.get("agree_pct", 0)
                disagree_pct = appraise_result.get("disagree_pct", 0)
                verdict = appraise_result.get("verdict", "")

                # If the graph agrees that the hypothesis might be wrong → disconfirmatory
                if isinstance(agree_pct, (int, float)) and float(agree_pct) > 0.4:
                    ev_id = uuid.uuid4().hex[:8]
                    evidence.append(Evidence(
                        evidence_id=ev_id,
                        description=(
                            f"RSVS appraise() suggests the hypothesis may be incorrect "
                            f"(agree_with_negation={float(agree_pct):.2f}, "
                            f"disagree={float(disagree_pct):.2f})"
                        ),
                        source_node=concept,
                        direction="disconfirmatory",
                        strength=float(agree_pct),
                        grounding_score=float(agree_pct),
                        discovery_method="appraise_negative",
                    ))

                # Check for clash pairs in appraise result
                clash_pairs = appraise_result.get("clash_pairs", [])
                if clash_pairs and isinstance(clash_pairs, list):
                    for clash in clash_pairs[:3]:
                        ev_id = uuid.uuid4().hex[:8]
                        evidence.append(Evidence(
                            evidence_id=ev_id,
                            description=(
                                f"Clash detected in graph: {clash} — contradicts hypothesis"
                            ),
                            source_node=concept,
                            direction="disconfirmatory",
                            strength=0.7,
                            grounding_score=0.6,
                            discovery_method="appraise_clash",
                        ))
        except Exception as exc:
            logger.debug("appraise() disconfirmatory search failed: %s", exc)

        # Strategy 2: Use mcts_query() to explore REASONING PATHS that could refute
        # This is the "disconfirmatory search" — exploring paths that might undermine
        try:
            # Search for paths that might contradict the hypothesis
            disconfirm_query = f"NOT {hypothesis.statement[:60]}"
            mcts_result = self._bridge.mcts_query(
                node_label=disconfirm_query,
                max_depth=3,
                simulations=30,
            )

            if mcts_result and isinstance(mcts_result, dict):
                scored_atoms = mcts_result.get("scored_atoms", [])
                if scored_atoms:
                    for atom_entry in scored_atoms[:5]:
                        label = ""
                        score = 0.5
                        if isinstance(atom_entry, (list, tuple)) and len(atom_entry) >= 2:
                            label = str(atom_entry[0])
                            score = float(atom_entry[1]) if isinstance(atom_entry[1], (int, float)) else 0.5
                        elif isinstance(atom_entry, str):
                            label = atom_entry

                        if label:
                            ev_id = uuid.uuid4().hex[:8]
                            evidence.append(Evidence(
                                evidence_id=ev_id,
                                description=(
                                    f"MCTS disconfirmatory search found '{label}' "
                                    f"(score={score:.3f}) — potential refutation path"
                                ),
                                source_node=label,
                                direction="disconfirmatory",
                                strength=min(1.0, score * 1.5),
                                grounding_score=score,
                                discovery_method="mcts_disconfirmatory",
                            ))
        except Exception as exc:
            logger.debug("mcts_query() disconfirmatory search failed: %s", exc)

        # Strategy 3: Check for structural mismatches between hypothesis and graph
        try:
            # Compare hypothesis criteria against actual graph structure
            for criterion in hypothesis.test_criteria[:3]:
                # Extract key concepts from criterion
                key_concepts = self._extract_key_concepts(criterion)
                for kc in key_concepts[:2]:
                    # Check if the graph contradicts this criterion
                    appraise_check = self._bridge.appraise(criterion[:200])
                    if appraise_check and isinstance(appraise_check, dict):
                        disagree_pct = appraise_check.get("disagree_pct", 0)
                        if isinstance(disagree_pct, (int, float)) and float(disagree_pct) > 0.5:
                            ev_id = uuid.uuid4().hex[:8]
                            evidence.append(Evidence(
                                evidence_id=ev_id,
                                description=(
                                    f"Test criterion '{criterion[:60]}' is contradicted "
                                    f"by graph (disagree={float(disagree_pct):.2f})"
                                ),
                                source_node=kc,
                                direction="disconfirmatory",
                                strength=float(disagree_pct),
                                grounding_score=float(disagree_pct),
                                discovery_method="criterion_appraise",
                            ))
        except Exception as exc:
            logger.debug("Structural mismatch disconfirmatory search failed: %s", exc)

        return evidence

    def _extract_pattern_evidence(
        self,
        hypothesis: Hypothesis,
        pattern_result: PatternResult,
    ) -> list[Evidence]:
        """Extract evidence from a PatternResult for hypothesis testing.

        Uses the pattern completion output as an additional evidence source,
        converting anomalies and patterns into confirmatory or disconfirmatory
        evidence for the hypothesis.

        Args:
            hypothesis: The hypothesis being tested.
            pattern_result: The PatternResult from Layer 2.

        Returns:
            A list of Evidence objects extracted from the pattern.
        """
        evidence: list[Evidence] = []

        # Convert anomalies to disconfirmatory evidence
        for anomaly_dict in pattern_result.anomalies[:5]:
            ev_id = uuid.uuid4().hex[:8]
            desc = anomaly_dict.get("description", str(anomaly_dict)[:60])
            evidence.append(Evidence(
                evidence_id=ev_id,
                description=f"Pattern anomaly: {desc}",
                source_node=hypothesis.anomaly_source,
                direction="disconfirmatory",
                strength=0.6,
                grounding_score=0.5,
                discovery_method="pattern_anomaly",
            ))

        # Convert pattern to confirmatory evidence (if pattern exists)
        if pattern_result.pattern:
            ev_id = uuid.uuid4().hex[:8]
            evidence.append(Evidence(
                evidence_id=ev_id,
                description=f"Pattern completion supports: {pattern_result.pattern[:80]}",
                source_node=hypothesis.anomaly_source,
                direction="confirmatory",
                strength=pattern_result.confidence,
                grounding_score=pattern_result.confidence,
                discovery_method="pattern_completion",
            ))

        return evidence

    # ==================================================================
    # Step 3: SCORE — Update hypothesis confidence
    # ==================================================================

    def _update_hypothesis_confidence(self, hypothesis: Hypothesis) -> None:
        """Update hypothesis confidence based on confirmatory and disconfirmatory evidence.

        The confidence update is ASYMMETRIC:
        - Confirmatory evidence increases confidence by _CONFIRM_IMPACT per item
        - Disconfirmatory evidence decreases confidence by _DISCONFIRM_IMPACT per item
        - Disconfirmatory impact is deliberately stronger (0.15 vs 0.10)

        Additionally, the net_evidence_score provides a global adjustment.

        Analogi: Setiap bukti yang mendukung hipotesis Jin Sowoon
        menaikkan keyakinannya sedikit, tapi setiap bukti yang MENGHAPUS
        menurunkan keyakinannya lebih banyak. Ini karena "absence of
        expected evidence is evidence of absence" — jika bukti yang
        seharusnya ada TIDAK ada, itu lebih bermakna daripada kehadiran
        bukti yang diharapkan.

        Args:
            hypothesis: The hypothesis to update.
        """
        # Calculate adjustments from new evidence
        confirm_adjust = 0.0
        for ev in hypothesis.confirmatory_evidence:
            confirm_adjust += _DEFAULT_CONFIRM_IMPACT * ev.strength * ev.grounding_score

        disconfirm_adjust = 0.0
        for ev in hypothesis.disconfirmatory_evidence:
            disconfirm_adjust += _DEFAULT_DISCONFIRM_IMPACT * ev.strength * ev.grounding_score

        # Net adjustment with asymmetric weighting
        net_adjust = (
            _DEFAULT_CONFIRMATORY_WEIGHT * confirm_adjust
            - _DEFAULT_DISCONFIRMATORY_WEIGHT * disconfirm_adjust
        )

        # Apply Rescorla-Wagner-style update
        old_confidence = hypothesis.confidence
        hypothesis.confidence = max(0.0, min(1.0, old_confidence + net_adjust))

        # Update hypothesis state based on confidence
        if hypothesis.confidence < self._viability_threshold:
            hypothesis.state = "refuted"
        elif hypothesis.confidence > 0.8 and hypothesis.total_evidence_count >= _DEFAULT_MIN_EVIDENCE_FOR_CONCLUSION:
            hypothesis.state = "confirmed"
        elif hypothesis.total_evidence_count > 0:
            hypothesis.state = "testing"

        logger.debug(
            "Hypothesis '%s' confidence: %.3f → %.3f "
            "(confirm_adj=%.3f, disconfirm_adj=%.3f, net=%.3f, state=%s)",
            hypothesis.statement[:40],
            old_confidence, hypothesis.confidence,
            confirm_adjust, disconfirm_adjust, net_adjust,
            hypothesis.state,
        )

    # ==================================================================
    # Step 4: CONCLUDE — Evaluate hypotheses and pick winner
    # ==================================================================

    def _evaluate_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        anomaly: Anomaly,
        cycle_id: str,
        cycle_number: int,
        reasoning_chain: DeductiveChain | None = None,
    ) -> HypothesisCycleResult:
        """Evaluate all hypotheses and determine if there's a winner.

        A hypothesis wins if:
        1. It has the highest confidence among all hypotheses
        2. The confidence gap (decisiveness) exceeds the threshold
        3. It has at least min_evidence_for_conclusion evidence items

        If no hypothesis wins, the cycle is inconclusive and further
        testing is needed.

        Args:
            hypotheses: The hypotheses to evaluate.
            anomaly: The anomaly being reasoned about.
            cycle_id: The cycle ID.
            cycle_number: Which cycle this is.
            reasoning_chain: The DeductiveChain from this cycle.

        Returns:
            A HypothesisCycleResult with evaluation results.
        """
        # Sort by confidence descending
        sorted_hyps = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

        best = sorted_hyps[0] if sorted_hyps else None
        second = sorted_hyps[1] if len(sorted_hyps) > 1 else None

        # Calculate decisiveness — gap between best and second-best
        decisiveness = 0.0
        if best and second:
            decisiveness = best.confidence - second.confidence
        elif best:
            decisiveness = best.confidence

        # Determine if conclusive
        is_conclusive = False
        winning_hypothesis = None

        if best and best.confidence > self._viability_threshold:
            if decisiveness >= self._decisiveness_threshold:
                if best.total_evidence_count >= _DEFAULT_MIN_EVIDENCE_FOR_CONCLUSION:
                    is_conclusive = True
                    winning_hypothesis = best
                    best.state = "confirmed"
                    # Mark others as superseded
                    for hyp in hypotheses:
                        if hyp.hypothesis_id != best.hypothesis_id:
                            if hyp.state not in ("refuted",):
                                hyp.state = "superseded"
            elif best.confidence > 0.7 and cycle_number >= self._max_cycles:
                # Force conclusion at max cycles
                is_conclusive = True
                winning_hypothesis = best
                best.state = "confirmed"

        # Count total evidence
        total_evidence = sum(h.total_evidence_count for h in hypotheses)

        return HypothesisCycleResult(
            cycle_id=cycle_id,
            anomaly=anomaly,
            hypotheses=list(hypotheses),
            winning_hypothesis=winning_hypothesis,
            is_conclusive=is_conclusive,
            decisiveness=decisiveness,
            reasoning_chain=reasoning_chain,
            cycle_number=cycle_number,
            total_evidence_found=total_evidence,
        )

    # ==================================================================
    # Step 5: REVISE — Generate revised hypotheses
    # ==================================================================

    def _revise_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        anomaly: Anomaly,
        context: list[str],
    ) -> list[Hypothesis]:
        """Revise hypotheses after an inconclusive cycle.

        Strategy:
        1. Keep viable hypotheses (confidence > viability_threshold)
        2. Refute non-viable hypotheses
        3. Generate new hypotheses from the gaps between expected and observed
        4. Use RSVS to find alternative interpretations

        Analogi: Jin Sowoon tidak menyerah ketika hipotesis pertamanya
        tidak cukup kuat. Dia revisi — mempertahankan yang masih viable,
        menghapus yang sudah terbantahkan, dan membuat hipotesis baru
        berdasarkan apa yang dia pelajari dari pengujian sebelumnya.

        Args:
            hypotheses: The current hypotheses to revise.
            anomaly: The original anomaly.
            context: Context atoms.

        Returns:
            A revised list of hypotheses for the next cycle.
        """
        revised: list[Hypothesis] = []

        # Keep viable hypotheses
        for hyp in hypotheses:
            if hyp.state == "refuted":
                continue  # Drop refuted hypotheses
            if hyp.confidence < self._viability_threshold:
                hyp.state = "refuted"
                continue
            revised.append(hyp)

        # Generate replacement hypotheses for refuted ones
        refuted_count = sum(1 for h in hypotheses if h.state == "refuted")
        if refuted_count > 0:
            # Create revised hypotheses based on what we learned
            for i in range(refuted_count):
                hyp_id = uuid.uuid4().hex[:8]
                # Use the strongest disconfirmatory evidence from refuted
                # hypotheses to inform new hypotheses
                refuted_evidence = []
                for h in hypotheses:
                    if h.state == "refuted":
                        refuted_evidence.extend(h.disconfirmatory_evidence)

                new_reasoning = (
                    f"Revised after {len(hypotheses)} hypotheses were tested. "
                    f"Based on {len(refuted_evidence)} disconfirmatory evidence items "
                    f"from previous cycle."
                )
                if refuted_evidence:
                    top_evidence = sorted(
                        refuted_evidence,
                        key=lambda e: e.strength,
                        reverse=True,
                    )[:2]
                    evidence_desc = "; ".join(e.description[:40] for e in top_evidence)
                    new_reasoning += f" Key refutation: {evidence_desc}."

                revised.append(Hypothesis(
                    hypothesis_id=hyp_id,
                    statement=(
                        f"Revised hypothesis {i+1}: the anomaly in "
                        f"'{anomaly.concept}' has a different cause than "
                        f"previously hypothesized"
                    ),
                    reasoning=new_reasoning,
                    test_criteria=self._generate_test_criteria(
                        anomaly.concept, "alternative",
                        anomaly.expected, anomaly.observed,
                    ),
                    confidence=0.4,  # Fresh start with moderate confidence
                    state="proposed",
                    anomaly_source=anomaly.concept,
                    parent_hypothesis_id=hypotheses[0].hypothesis_id if hypotheses else None,
                ))

        # Ensure we have at least 2 competing hypotheses
        while len(revised) < 2:
            hyp_id = uuid.uuid4().hex[:8]
            revised.append(Hypothesis(
                hypothesis_id=hyp_id,
                statement=(
                    f"Alternative: '{anomaly.concept}' anomaly is caused by "
                    f"an unknown factor not yet explored"
                ),
                reasoning="Generated to ensure hypothesis competition.",
                test_criteria=[
                    f"Find evidence of unknown factors related to '{anomaly.concept}'"
                ],
                confidence=0.3,
                state="proposed",
                anomaly_source=anomaly.concept,
            ))

        return revised[:self._hypothesis_count]

    # ==================================================================
    # Utility methods
    # ==================================================================

    def _build_pattern_from_anomaly(
        self,
        anomaly: Anomaly,
        context: list[str],
    ) -> PatternResult:
        """Build a minimal PatternResult from an anomaly for ReasoningEngine.

        When no PatternResult is provided, we create one from the anomaly
        data so that ReasoningEngine can still build a deductive chain.

        Args:
            anomaly: The anomaly to convert.
            context: Context atoms.

        Returns:
            A PatternResult suitable for ReasoningEngine.build_chain().
        """
        # Create reasoning steps from anomaly data
        trigger_step = ReasoningStep(
            step_type="trigger",
            description=f"Anomaly detected: {anomaly.description[:100]}",
            data={
                "concepts": [anomaly.concept],
                "trigger_text": f"Anomaly: {anomaly.concept}",
            },
            evidence_nodes=[anomaly.concept] + anomaly.expected[:3] + anomaly.observed[:3],
            confidence=0.5,
        )

        anomaly_step = ReasoningStep(
            step_type="anomaly",
            description=(
                f"Expected {anomaly.expected[:3]} but observed {anomaly.observed[:3]} "
                f"(delta={anomaly.delta:.3f})"
            ),
            data={
                "expected": anomaly.expected,
                "observed": anomaly.observed,
                "delta": anomaly.delta,
            },
            evidence_nodes=anomaly.observed[:5],
            confidence=max(0.3, 1.0 - anomaly.delta),
        )

        return PatternResult(
            trigger=f"Anomaly: {anomaly.concept} (delta={anomaly.delta:.3f})",
            steps=[trigger_step, anomaly_step],
            anomalies=[{
                "type": "prediction_error",
                "concept": anomaly.concept,
                "delta": anomaly.delta,
                "description": anomaly.description,
            }],
            confidence=max(0.3, 1.0 - anomaly.delta),
        )

    def _extract_key_concepts(self, text: str) -> list[str]:
        """Extract key concepts from a text string.

        Simple keyword extraction for hypothesis test criteria.

        Args:
            text: The text to extract concepts from.

        Returns:
            A list of key concept strings.
        """
        stop_words = {
            "that", "this", "with", "from", "have", "been", "they",
            "their", "which", "would", "there", "could", "about",
            "other", "into", "more", "than", "then", "some", "very",
            "also", "just", "like", "only", "over", "such", "after",
            "yang", "dan", "dari", "untuk", "dengan", "adalah", "itu",
            "ini", "ke", "di", "pada", "tidak", "akan", "telah", "oleh",
            "the", "and", "but", "for", "not", "find", "evidence",
            "hypothesis", "supports", "contradicts", "verify",
        }
        words = text.lower().replace(",", " ").replace(".", " ").replace("'", " ").split()
        return [w for w in words if len(w) > 2 and w not in stop_words][:10]

    # ==================================================================
    # Query methods
    # ==================================================================

    @property
    def reasoning_engine(self) -> ReasoningEngine:
        """Access the underlying ReasoningEngine."""
        return self._reasoning_engine

    @property
    def predictive_engine(self) -> PredictiveEngine:
        """Access the underlying PredictiveEngine."""
        return self._predictive_engine

    @property
    def active_hypotheses(self) -> dict[str, Hypothesis]:
        """Return currently active (non-terminal) hypotheses."""
        return {
            hid: h for hid, h in self._active_hypotheses.items()
            if h.state in ("proposed", "testing")
        }

    @property
    def cycle_history(self) -> list[HypothesisCycleResult]:
        """Return completed cycle history."""
        return list(self._cycle_history)

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        """Get a hypothesis by ID.

        Args:
            hypothesis_id: The hypothesis ID to look up.

        Returns:
            The Hypothesis, or None if not found.
        """
        return self._active_hypotheses.get(hypothesis_id)

    def reason_from_anomalies(
        self,
        anomalies: list[Anomaly] | None = None,
        context: list[str] | None = None,
    ) -> list[HypothesisCycleResult]:
        """Run hypothesis-driven reasoning for all current anomalies.

        Convenience method that takes anomalies from PredictiveEngine
        (if not provided) and runs the full reasoning cycle for each.

        Args:
            anomalies: Optional list of anomalies. If None, gets from
                the PredictiveEngine.
            context: Optional context atoms.

        Returns:
            A list of HypothesisCycleResult, one per anomaly.
        """
        if anomalies is None:
            anomalies = self._predictive_engine.get_anomalies()

        if not anomalies:
            logger.info("No anomalies to reason about")
            return []

        context = context or []
        results: list[HypothesisCycleResult] = []

        for anomaly in anomalies:
            result = self.reason(anomaly, context)
            results.append(result)

        return results
