"""
AAM Layer 3 — Possibility Lattice: Dynamic Hypothesis Space

Inspirasi: Strategi permainan kartu — generate semua kemungkinan,
eliminasi yang salah, hybrid yang komplementer, repeat sampai sisa 1.

Ini adalah evolusi dari Hypothesis-Driven Active Reasoning (Approach F).
Daripada ruang kemungkinan statis yang hanya menyusut melalui eliminasi,
Possibility Lattice menciptakan ruang DINAMIS yang bisa:
  - MENYUSUT melalui eliminasi (buang yang salah)
  - TUMBUH melalui hybridization (A × B → kemungkinan baru)
  - MENSTABIL melalui diminishing returns (konvergensi alami)

Core Cycle:
    GENERATE → ELIMINATE → HYBRIDIZE → DETECT NOVEL → CHECK STABILITY

    1. GENERATE  : Enumerate all possibilities from RSVS graph
                   (context-based, input-based, cross-referential)
    2. ELIMINATE : Score each against evidence; remove below threshold
    3. HYBRIDIZE : Combine complementary pairs (A × B, not A ∧ B)
                   → may reveal entirely new possibilities
    4. DETECT    : Filter genuinely novel hybrids (not rephrasings)
    5. STABILITY : If no elimination AND no novelty → fixed point reached
                   If ≤3 remain with high confidence → conclude
                   If question_mode and stuck → ask user

Key Insights:
    - Two partial hypotheses can combine into a STRONGER hypothesis
      that NEITHER parent could reach alone
    - A hybrid may open doors that neither parent could see
      ("membuka peluang baru demi kemungkinan yang lain yang baru tersadari")
    - Diminishing returns guarantee convergence: ~200 max for mature AAM
    - Question Mode = epistemic value from Active Inference

Architecture:
    PossibilityLattice WRAPS HypothesisDrivenReasoner — it adds the
    lattice mode on top of the existing hypothesis-driven cycle.
    The reasoner handles generative/eliminative single-path reasoning;
    this module handles the dynamic possibility space with hybridization.

Integration:
    - Layer 2 PossibilityGenerator: Enumerate from RSVS graph
    - Layer 2 HypothesisCombinator: Compose hybrids (A × B)
    - Layer 3 HypothesisDrivenReasoner: Base reasoning engine
    - RSVS Bridge: mcts_query() → implication traversal
    - RSVS Bridge: appraise() → novelty detection
    - RSVS Bridge: structural_similarity() → complementarity measurement
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

# Layer 2 & 3 imports
from layer2.bridge import RsvsBridge, get_bridge
from layer2.predictive import Anomaly, PredictiveEngine
from layer2.pattern import PatternResult, PatternOutput, ReasoningStep
from layer3.reasoning import ReasoningEngine, DeductiveChain, DeductiveStep
from layer3.hypothesis import (
    HypothesisDrivenReasoner,
    Hypothesis,
    Evidence,
    HypothesisCycleResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

# Maximum number of lattice generations before forced conclusion
_DEFAULT_MAX_GENERATIONS = 10

# Confidence threshold below which a possibility is eliminated
_DEFAULT_ELIMINATION_THRESHOLD = 0.15

# Number of remaining possibilities that triggers question mode
_DEFAULT_QUESTION_MODE_THRESHOLD = 3

# Confidence above which a single remaining possibility is accepted
_DEFAULT_CONCLUSION_CONFIDENCE = 0.85

# Similarity threshold above which a hybrid is considered not novel
_DEFAULT_NOVELTY_SIMILARITY_THRESHOLD = 0.85

# Complementarity threshold above which two possibilities are worth hybridizing
_DEFAULT_COMPLEMENTARITY_THRESHOLD = 0.4

# Maximum number of possibilities to generate initially
_DEFAULT_MAX_INITIAL_POSSIBILITIES = 150

# Maximum number of hybrids to produce per generation
_DEFAULT_MAX_HYBRIDS_PER_GENERATION = 30

# Maximum depth for implication tracing from a hybrid
_DEFAULT_IMPLICATION_DEPTH = 2

# How much confidence boost a hybrid gets for combining complementary evidence
_DEFAULT_HYBRID_CONFIDENCE_BOOST = 1.1


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LatticeMode(Enum):
    """Operating mode for the Possibility Lattice."""
    GENERATIVE = "generative"       # Top-K hypothesis (Approach F original)
    ELIMINATIVE = "eliminative"     # Full space → progressive elimination
    LATTICE = "lattice"             # Dynamic space: eliminate + hybridize


class PossibilityState(Enum):
    """State of a possibility in the lattice."""
    PROPOSED = "proposed"
    TESTING = "testing"
    SURVIVING = "surviving"         # Survived elimination
    HYBRID = "hybrid"               # Created from hybridization
    EMERGENT = "emergent"           # Emerged from implication tracing
    ELIMINATED = "eliminated"
    CONCLUDED = "concluded"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Possibility:
    """A single possibility in the dynamic hypothesis space.

    Unlike a Hypothesis (which is tied to an anomaly), a Possibility
    is a member of the full possibility lattice. It can be:
    - Generated from context/input/cross-reference
    - Eliminated by evidence
    - Hybridized with another possibility
    - Emergent from implication tracing of a hybrid

    Key difference from Hypothesis:
    - Possibility tracks explained_evidence (what it covers)
    - Possibility has parent_ids (for hybrid lineage)
    - Possibility has a generation number (for diminishing returns tracking)
    - Possibility can be in HYBRID or EMERGENT state

    Attributes:
        possibility_id: Unique identifier.
        statement: What this possibility claims.
        reasoning: Why this possibility might be true.
        confidence: Current confidence level (0.0–1.0).
        state: Current state in the lattice lifecycle.
        explained_evidence: Set of evidence IDs this possibility explains.
        all_evidence: Set of ALL evidence IDs in the current context.
        parent_ids: IDs of parent possibilities (empty for generated, 2 for hybrid).
        generation: Which lattice generation created this (0 = initial).
        source: How this possibility was generated
            ("context", "input", "cross_reference", "hybrid", "emergent").
        anomaly_source: The anomaly that triggered this possibility's lineage.
        confirmatory_evidence: Evidence supporting this possibility.
        disconfirmatory_evidence: Evidence undermining this possibility.
        created_at: When this possibility was created.
    """

    possibility_id: str
    statement: str
    reasoning: str = ""
    confidence: float = 0.5
    state: str = "proposed"
    explained_evidence: set[str] = field(default_factory=set)
    all_evidence: set[str] = field(default_factory=set)
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    source: str = "context"
    anomaly_source: str = ""
    confirmatory_evidence: list[Evidence] = field(default_factory=list)
    disconfirmatory_evidence: list[Evidence] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    @property
    def is_hybrid(self) -> bool:
        """Whether this possibility was created from hybridization."""
        return len(self.parent_ids) >= 2

    @property
    def coverage(self) -> float:
        """Fraction of all evidence this possibility explains."""
        if not self.all_evidence:
            return 0.0
        return len(self.explained_evidence) / len(self.all_evidence)

    @property
    def net_evidence_score(self) -> float:
        """Weighted evidence score (same logic as Hypothesis)."""
        confirm_total = sum(
            e.strength * e.grounding_score
            for e in self.confirmatory_evidence
        )
        disconfirm_total = sum(
            e.strength * e.grounding_score
            for e in self.disconfirmatory_evidence
        )
        return 0.4 * confirm_total - 0.6 * disconfirm_total

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "possibility_id": self.possibility_id,
            "statement": self.statement,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 4),
            "state": self.state,
            "explained_evidence": sorted(self.explained_evidence),
            "all_evidence_count": len(self.all_evidence),
            "coverage": round(self.coverage, 4),
            "parent_ids": list(self.parent_ids),
            "generation": self.generation,
            "source": self.source,
            "anomaly_source": self.anomaly_source,
            "is_hybrid": self.is_hybrid,
            "net_evidence_score": round(self.net_evidence_score, 4),
            "created_at": self.created_at,
        }


@dataclass
class LatticeGeneration:
    """Record of one generation in the lattice cycle.

    Tracks what happened during one round of eliminate → hybridize → detect.

    Attributes:
        generation: Which generation this is (0-based).
        pre_count: Number of possibilities at start of generation.
        post_elimination: Number after elimination.
        hybrids_created: Number of hybrids produced.
        novel_count: Number of genuinely novel possibilities found.
        eliminated_count: Number eliminated in this generation.
        emerged_count: Number of emergent possibilities from implication tracing.
        is_stable: Whether the lattice reached a fixed point.
        timestamp: When this generation was processed.
    """

    generation: int
    pre_count: int = 0
    post_elimination: int = 0
    hybrids_created: int = 0
    novel_count: int = 0
    eliminated_count: int = 0
    emerged_count: int = 0
    is_stable: bool = False
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "generation": self.generation,
            "pre_count": self.pre_count,
            "post_elimination": self.post_elimination,
            "hybrids_created": self.hybrids_created,
            "novel_count": self.novel_count,
            "eliminated_count": self.eliminated_count,
            "emerged_count": self.emerged_count,
            "is_stable": self.is_stable,
            "timestamp": self.timestamp,
        }


@dataclass
class LatticeResult:
    """Final result from the Possibility Lattice reasoning process.

    Contains the full trace of all generations, the surviving possibilities,
    the conclusion, and whether question mode was activated.

    Attributes:
        result_id: Unique identifier for this result.
        query: The original query that triggered reasoning.
        mode: Which mode was used (generative/eliminative/lattice).
        generations: Record of each generation in the lattice cycle.
        surviving_possibilities: Possibilities that survived all rounds.
        conclusion: The final concluded possibility (if any).
        is_conclusive: Whether a clear conclusion was reached.
        question_mode_activated: Whether question mode was triggered.
        questions_asked: List of questions that would be asked to the user.
        total_possibilities_generated: Total across all generations.
        total_eliminated: Total eliminated across all generations.
        total_hybrids: Total hybrids created across all generations.
        total_emergent: Total emergent possibilities found.
        confidence: Final confidence of the conclusion.
    """

    result_id: str
    query: str = ""
    mode: str = "lattice"
    generations: list[LatticeGeneration] = field(default_factory=list)
    surviving_possibilities: list[Possibility] = field(default_factory=list)
    conclusion: Possibility | None = None
    is_conclusive: bool = False
    question_mode_activated: bool = False
    questions_asked: list[str] = field(default_factory=list)
    total_possibilities_generated: int = 0
    total_eliminated: int = 0
    total_hybrids: int = 0
    total_emergent: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "result_id": self.result_id,
            "query": self.query[:200],
            "mode": self.mode,
            "generations": [g.to_dict() for g in self.generations],
            "surviving_count": len(self.surviving_possibilities),
            "conclusion": self.conclusion.to_dict() if self.conclusion else None,
            "is_conclusive": self.is_conclusive,
            "question_mode_activated": self.question_mode_activated,
            "questions_asked": list(self.questions_asked),
            "total_possibilities_generated": self.total_possibilities_generated,
            "total_eliminated": self.total_eliminated,
            "total_hybrids": self.total_hybrids,
            "total_emergent": self.total_emergent,
            "confidence": round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# PossibilityLattice
# ---------------------------------------------------------------------------

class PossibilityLattice:
    """Dynamic Possibility Space with Hybridization and Progressive Elimination.

    This is the evolution of AAM's reasoning: instead of a static possibility
    space that only shrinks, the lattice is a LIVING space that can:
    - SHRINK through elimination (buang yang salah)
    - GROW through hybridization (A × B → kemungkinan baru)
    - STABILIZE through diminishing returns (konvergensi alami)

    The process is inspired by card game strategy:
    1. Count all possible hands
    2. As cards are revealed, eliminate impossible hands
    3. Combine partial information into new insights
    4. Repeat until one possibility remains (or ask for more info)

    Key innovation: Hybridization (A × B) creates possibilities that
    NEITHER parent could reach alone — and may open entirely new
    avenues of reasoning ("membuka peluang baru").

    Three modes:
    - GENERATIVE: Top-K hypothesis (Approach F original)
    - ELIMINATIVE: Full space → progressive elimination
    - LATTICE: Dynamic space — eliminate + hybridize (DEFAULT)

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected.
        is_rust_core: Whether the Rust core is being used.
    """

    def __init__(
        self,
        bridge: Optional[RsvsBridge] = None,
        reasoner: Optional[HypothesisDrivenReasoner] = None,
        predictive_engine: Optional[PredictiveEngine] = None,
        *,
        max_generations: int = _DEFAULT_MAX_GENERATIONS,
        elimination_threshold: float = _DEFAULT_ELIMINATION_THRESHOLD,
        question_mode_threshold: int = _DEFAULT_QUESTION_MODE_THRESHOLD,
        conclusion_confidence: float = _DEFAULT_CONCLUSION_CONFIDENCE,
        novelty_threshold: float = _DEFAULT_NOVELTY_SIMILARITY_THRESHOLD,
        complementarity_threshold: float = _DEFAULT_COMPLEMENTARITY_THRESHOLD,
        max_initial_possibilities: int = _DEFAULT_MAX_INITIAL_POSSIBILITIES,
        max_hybrids_per_generation: int = _DEFAULT_MAX_HYBRIDS_PER_GENERATION,
        question_callback: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initialize the Possibility Lattice.

        Args:
            bridge: Optional pre-built RsvsBridge. If None, one is created.
            reasoner: Optional pre-built HypothesisDrivenReasoner.
            predictive_engine: Optional pre-built PredictiveEngine.
            max_generations: Maximum lattice generations before forced conclusion.
            elimination_threshold: Confidence below which a possibility is eliminated.
            question_mode_threshold: Number of remaining possibilities that triggers
                question mode.
            conclusion_confidence: Confidence above which a conclusion is accepted.
            novelty_threshold: Similarity above which a hybrid is not novel.
            complementarity_threshold: Complementarity above which two possibilities
                are worth hybridizing.
            max_initial_possibilities: Cap on initial possibility enumeration.
            max_hybrids_per_generation: Cap on hybrids produced per generation.
            question_callback: Optional callback for asking the user questions.
                If None, question mode collects questions but doesn't ask.
        """
        if bridge is not None:
            self._bridge = bridge
        elif reasoner is not None:
            self._bridge = reasoner._bridge
        elif predictive_engine is not None:
            self._bridge = predictive_engine._bridge
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core

        # Wrap or create the underlying reasoner
        if reasoner is not None:
            self._reasoner = reasoner
        else:
            self._reasoner = HypothesisDrivenReasoner(bridge=self._bridge)

        # Wrap or create the predictive engine
        if predictive_engine is not None:
            self._predictive_engine = predictive_engine
        else:
            self._predictive_engine = PredictiveEngine(bridge=self._bridge)

        # Configuration
        self._max_generations = max_generations
        self._elimination_threshold = elimination_threshold
        self._question_mode_threshold = question_mode_threshold
        self._conclusion_confidence = conclusion_confidence
        self._novelty_threshold = novelty_threshold
        self._complementarity_threshold = complementarity_threshold
        self._max_initial_possibilities = max_initial_possibilities
        self._max_hybrids_per_generation = max_hybrids_per_generation
        self._question_callback = question_callback

        # Lattice history
        self._lattice_history: list[LatticeResult] = []

        logger.info(
            "PossibilityLattice initialized "
            "(max_gen=%d, elim=%.2f, question_thresh=%d, "
            "conclusion_conf=%.2f, rsvs=%s)",
            max_generations, elimination_threshold, question_mode_threshold,
            conclusion_confidence, self.rsvs_available,
        )

    # ==================================================================
    # MAIN METHOD — reason()
    # ==================================================================

    def reason(
        self,
        query: str,
        context: list[str] | None = None,
        evidence_list: list[str] | None = None,
        anomaly: Anomaly | None = None,
        pattern_result: PatternResult | None = None,
        mode: LatticeMode = LatticeMode.LATTICE,
        question_mode: bool = False,
    ) -> LatticeResult:
        """Run the Possibility Lattice reasoning process.

        This is the primary entry point. It takes a query, generates
        all possibilities, then iteratively eliminates and hybridizes
        until a conclusion is reached.

        Process:
        1. GENERATE — Enumerate all possibilities from RSVS graph
        2. For each generation:
           a. ELIMINATE — Remove possibilities below threshold
           b. HYBRIDIZE — Combine complementary pairs
           c. DETECT NOVEL — Filter genuinely novel hybrids
           d. CHECK STABILITY — Stop if no changes
        3. CONCLUDE — Return best possibility or ask user

        Args:
            query: The query to reason about.
            context: Optional context atoms for possibility generation.
            evidence_list: Optional initial evidence to evaluate against.
            anomaly: Optional anomaly to trigger reasoning.
            pattern_result: Optional PatternResult for evidence.
            mode: Which reasoning mode to use (generative/eliminative/lattice).
            question_mode: Whether to activate question mode when stuck.

        Returns:
            A LatticeResult with the full reasoning trace.
        """
        context = context or []
        evidence_list = evidence_list or []
        result_id = uuid.uuid4().hex[:8]

        logger.info(
            "PossibilityLattice.reason(): query='%s' (mode=%s, question=%s)",
            query[:60], mode.value, question_mode,
        )

        # ---- PHASE 1: GENERATE ----
        possibilities = self._generate_possibilities(
            query, context, evidence_list, anomaly,
        )

        logger.info("Generated %d initial possibilities", len(possibilities))

        # If generative mode, delegate to HypothesisDrivenReasoner
        if mode == LatticeMode.GENERATIVE and anomaly is not None:
            return self._generative_mode(
                anomaly, context, pattern_result, result_id, query,
            )

        # ---- PHASE 2: ITERATE (eliminate → hybridize → detect) ----
        generations: list[LatticeGeneration] = []
        all_evidence = set(evidence_list)
        total_eliminated = 0
        total_hybrids = 0
        total_emergent = 0
        questions_asked: list[str] = []

        for gen_num in range(self._max_generations):
            gen_record = LatticeGeneration(generation=gen_num, pre_count=len(possibilities))

            # Step 2a: ELIMINATE — remove impossibles
            surviving, eliminated_count = self._eliminate(possibilities, all_evidence)
            gen_record.post_elimination = len(surviving)
            gen_record.eliminated_count = eliminated_count
            total_eliminated += eliminated_count

            possibilities = surviving

            # Check for single conclusion
            if len(possibilities) == 1:
                sole = possibilities[0]
                sole.state = "concluded"
                gen_record.is_stable = True
                generations.append(gen_record)
                return LatticeResult(
                    result_id=result_id,
                    query=query,
                    mode=mode.value,
                    generations=generations,
                    surviving_possibilities=possibilities,
                    conclusion=sole,
                    is_conclusive=True,
                    total_possibilities_generated=len(possibilities) + total_eliminated,
                    total_eliminated=total_eliminated,
                    total_hybrids=total_hybrids,
                    total_emergent=total_emergent,
                    confidence=sole.confidence,
                )

            # Check for zero surviving — switch to generative
            if len(possibilities) == 0:
                logger.info("All possibilities eliminated — switching to generative mode")
                if anomaly is not None:
                    return self._generative_mode(
                        anomaly, context, pattern_result, result_id, query,
                    )
                # No anomaly — return empty result
                gen_record.is_stable = True
                generations.append(gen_record)
                return LatticeResult(
                    result_id=result_id,
                    query=query,
                    mode=mode.value,
                    generations=generations,
                    confidence=0.0,
                )

            # Step 2b: HYBRIDIZE — combine complementary pairs (only in LATTICE mode)
            hybrids_created = 0
            novel_count = 0
            emerged_count = 0

            if mode == LatticeMode.LATTICE:
                hybrids, hybrid_implications = self._hybridize(
                    possibilities, all_evidence, gen_num + 1,
                )
                hybrids_created = len(hybrids)
                total_hybrids += hybrids_created

                # Step 2c: DETECT NOVEL — filter genuinely novel hybrids
                novel = self._detect_novel(hybrids, possibilities)
                novel_count = len(novel)
                possibilities.extend(novel)

                # Add emergent possibilities from implication tracing
                novel_emergent = self._detect_novel(hybrid_implications, possibilities)
                emerged_count = len(novel_emergent)
                total_emergent += emerged_count
                possibilities.extend(novel_emergent)

            gen_record.hybrids_created = hybrids_created
            gen_record.novel_count = novel_count
            gen_record.emerged_count = emerged_count

            # Step 2d: CHECK STABILITY
            eliminated = gen_record.eliminated_count
            emerged = novel_count + emerged_count
            gen_record.is_stable = (emerged == 0 and eliminated == 0)

            generations.append(gen_record)

            if gen_record.is_stable:
                logger.info(
                    "Lattice stabilized at generation %d "
                    "(%d surviving possibilities)",
                    gen_num, len(possibilities),
                )
                break

            # Step 2e: QUESTION MODE — if few remain and stuck
            if (question_mode
                    and len(possibilities) <= self._question_mode_threshold
                    and len(possibilities) > 1):
                question = self._find_best_question(possibilities)
                if question:
                    questions_asked.append(question)
                    if self._question_callback is not None:
                        user_answer = self._question_callback(question)
                        # Add user answer as evidence
                        all_evidence.add(user_answer)
                        # Re-score possibilities with new evidence
                        self._score_against_evidence(possibilities, {user_answer})
                    else:
                        # No callback — just mark that we would ask
                        logger.info(
                            "Question mode: would ask '%s'", question[:80],
                        )

            # Step 2f: Gather more evidence from RSVS for next round
            new_evidence = self._seek_discriminating_evidence(possibilities)
            all_evidence.update(new_evidence)

            # Re-score all possibilities with accumulated evidence
            self._score_against_evidence(possibilities, all_evidence)

        # ---- PHASE 3: CONCLUDE ----
        best = max(possibilities, key=lambda p: p.confidence) if possibilities else None
        is_conclusive = (
            best is not None
            and best.confidence >= self._conclusion_confidence
        )

        if best is not None and is_conclusive:
            best.state = "concluded"

        result = LatticeResult(
            result_id=result_id,
            query=query,
            mode=mode.value,
            generations=generations,
            surviving_possibilities=possibilities,
            conclusion=best,
            is_conclusive=is_conclusive,
            question_mode_activated=len(questions_asked) > 0,
            questions_asked=questions_asked,
            total_possibilities_generated=(
                len(possibilities) + total_eliminated
            ),
            total_eliminated=total_eliminated,
            total_hybrids=total_hybrids,
            total_emergent=total_emergent,
            confidence=best.confidence if best else 0.0,
        )

        self._lattice_history.append(result)
        return result

    # ==================================================================
    # PHASE 1: GENERATE — Enumerate all possibilities
    # ==================================================================

    def _generate_possibilities(
        self,
        query: str,
        context: list[str],
        evidence_list: list[str],
        anomaly: Anomaly | None,
    ) -> list[Possibility]:
        """Generate all possibilities from multiple angles.

        Strategy:
        1. Context-based: What does the RSVS graph say about the query context?
        2. Input-specific: What does the query itself suggest?
        3. Cross-referential: What emerges from combining context and input?
        4. Anomaly-driven: If an anomaly is provided, generate from it.

        For a mature AAM, this typically produces 50-150 possibilities.

        Args:
            query: The query to generate possibilities for.
            context: Context atoms from the current conversation.
            evidence_list: Initial evidence to consider.
            anomaly: Optional anomaly that triggered reasoning.

        Returns:
            A list of Possibility objects (deduplicated).
        """
        possibilities: list[Possibility] = []
        all_evidence = set(evidence_list)

        # Angle 1: Context-based possibilities
        if context and self.rsvs_available:
            ctx_possibilities = self._generate_from_context(query, context, all_evidence)
            possibilities.extend(ctx_possibilities)

        # Angle 2: Input-specific possibilities
        input_possibilities = self._generate_from_input(query, all_evidence)
        possibilities.extend(input_possibilities)

        # Angle 3: Cross-referential possibilities
        if self.rsvs_available and context:
            cross_possibilities = self._generate_from_cross_reference(
                query, context, all_evidence,
            )
            possibilities.extend(cross_possibilities)

        # Angle 4: Anomaly-driven possibilities
        if anomaly is not None:
            anomaly_possibilities = self._generate_from_anomaly(anomaly, all_evidence)
            possibilities.extend(anomaly_possibilities)

        # Deduplicate by semantic similarity
        possibilities = self._deduplicate(possibilities)

        # Cap at max initial possibilities
        possibilities = possibilities[:self._max_initial_possibilities]

        logger.info(
            "Generated %d possibilities from %d angles "
            "(context=%d, input=%d, cross=%d, anomaly=%d)",
            len(possibilities),
            sum(1 for _ in [1] if context) + sum(1 for _ in [1] if anomaly),
            sum(1 for p in possibilities if p.source == "context"),
            sum(1 for p in possibilities if p.source == "input"),
            sum(1 for p in possibilities if p.source == "cross_reference"),
            sum(1 for p in possibilities if p.source == "anomaly"),
        )

        return possibilities

    def _generate_from_context(
        self,
        query: str,
        context: list[str],
        all_evidence: set[str],
    ) -> list[Possibility]:
        """Generate possibilities based on context from RSVS graph.

        Uses relate(), senses(), and mcts_query() to find all possible
        interpretations of the query within the given context.
        """
        possibilities: list[Possibility] = []

        for ctx_atom in context[:10]:
            # Use relate() to find structurally connected concepts
            try:
                relate_result = self._bridge.relate(ctx_atom)
                if relate_result and isinstance(relate_result, dict):
                    related_nodes = relate_result.get("related_nodes", [])
                    for node_entry in related_nodes[:5]:
                        label = self._extract_label(node_entry)
                        if not label:
                            continue

                        poss_id = uuid.uuid4().hex[:8]
                        possibilities.append(Possibility(
                            possibility_id=poss_id,
                            statement=f"'{label}' is relevant to '{query}' via context '{ctx_atom}'",
                            reasoning=(
                                f"RSVS relate() found structural connection between "
                                f"'{ctx_atom}' and '{label}', suggesting relevance to the query."
                            ),
                            confidence=0.4,
                            state="proposed",
                            explained_evidence={ctx_atom},
                            all_evidence=all_evidence,
                            generation=0,
                            source="context",
                            anomaly_source="",
                        ))
            except Exception as exc:
                logger.debug("relate() context generation failed: %s", exc)

            # Use senses() to find compositional interpretations
            try:
                senses = self._bridge.senses(ctx_atom)
                if senses and isinstance(senses, list):
                    for sense in senses[:3]:
                        if not isinstance(sense, dict):
                            continue
                        sense_idx = str(sense.get("sense_idx", 0))
                        gs = sense.get("grounding_score", 0.5)
                        core_atoms = sense.get("core_atoms", [])
                        if not core_atoms:
                            continue

                        atom_labels = [
                            a[0] if isinstance(a, (list, tuple)) else str(a)
                            for a in core_atoms[:5]
                        ]

                        poss_id = uuid.uuid4().hex[:8]
                        possibilities.append(Possibility(
                            possibility_id=poss_id,
                            statement=(
                                f"Sense {sense_idx} of '{ctx_atom}' — "
                                f"compositions: {', '.join(str(a) for a in atom_labels[:3])} — "
                                f"is the interpretation for '{query}'"
                            ),
                            reasoning=(
                                f"RSVS senses() revealed a compositional structure for "
                                f"'{ctx_atom}' that may interpret the query."
                            ),
                            confidence=min(0.6, gs),
                            state="proposed",
                            explained_evidence={ctx_atom},
                            all_evidence=all_evidence,
                            generation=0,
                            source="context",
                        ))
            except Exception as exc:
                logger.debug("senses() context generation failed: %s", exc)

        return possibilities

    def _generate_from_input(
        self,
        query: str,
        all_evidence: set[str],
    ) -> list[Possibility]:
        """Generate possibilities based on the query input itself.

        Uses keyword extraction and RSVS queries to enumerate
        what the query could mean.
        """
        possibilities: list[Possibility] = []
        concepts = self._extract_key_concepts(query)

        for concept in concepts[:8]:
            # Direct query possibility
            poss_id = uuid.uuid4().hex[:8]
            possibilities.append(Possibility(
                possibility_id=poss_id,
                statement=f"'{concept}' is the key element in understanding '{query}'",
                reasoning=(
                    f"The concept '{concept}' was directly mentioned in the query, "
                    f"suggesting it plays a central role in the answer."
                ),
                confidence=0.5,
                state="proposed",
                explained_evidence=set(),
                all_evidence=all_evidence,
                generation=0,
                source="input",
            ))

            # If RSVS available, find structural alternatives
            if self.rsvs_available:
                try:
                    senses = self._bridge.senses(concept)
                    if senses and isinstance(senses, list):
                        for sense in senses[:2]:
                            if isinstance(sense, dict):
                                gs = sense.get("grounding_score", 0.5)
                                sense_idx = str(sense.get("sense_idx", 0))

                                poss_id = uuid.uuid4().hex[:8]
                                possibilities.append(Possibility(
                                    possibility_id=poss_id,
                                    statement=(
                                        f"'{concept}' (sense {sense_idx}) "
                                        f"provides a specific angle on '{query}'"
                                    ),
                                    reasoning=(
                                        f"RSVS has sense {sense_idx} for '{concept}' "
                                        f"with grounding {gs:.2f}, offering a "
                                        f"structured interpretation."
                                    ),
                                    confidence=min(0.55, gs * 1.1),
                                    state="proposed",
                                    explained_evidence={concept},
                                    all_evidence=all_evidence,
                                    generation=0,
                                    source="input",
                                ))
                except Exception as exc:
                    logger.debug("senses() input generation failed: %s", exc)

        return possibilities

    def _generate_from_cross_reference(
        self,
        query: str,
        context: list[str],
        all_evidence: set[str],
    ) -> list[Possibility]:
        """Generate possibilities from cross-referencing context and input.

        Uses mcts_query() to find reasoning paths that connect
        multiple context atoms to the query.
        """
        possibilities: list[Possibility] = []

        if not self.rsvs_available:
            return possibilities

        # Use MCTS to find connecting paths
        try:
            mcts_result = self._bridge.mcts_query(
                node_label=query[:100],
                max_depth=3,
                simulations=30,
            )
            if mcts_result and isinstance(mcts_result, dict):
                best_path = mcts_result.get("best_path", [])
                if best_path:
                    for i, node_label in enumerate(best_path[:5]):
                        if not isinstance(node_label, str):
                            continue
                        poss_id = uuid.uuid4().hex[:8]
                        possibilities.append(Possibility(
                            possibility_id=poss_id,
                            statement=(
                                f"MCTS path node '{node_label}' (step {i+1}) "
                                f"is a reasoning link for '{query[:50]}'"
                            ),
                            reasoning=(
                                f"MCTS traversal found '{node_label}' on the best "
                                f"reasoning path from the query, suggesting it "
                                f"connects context to conclusion."
                            ),
                            confidence=0.45,
                            state="proposed",
                            explained_evidence=set(context[:3]),
                            all_evidence=all_evidence,
                            generation=0,
                            source="cross_reference",
                        ))

                # Also check scored atoms from MCTS
                scored_atoms = mcts_result.get("scored_atoms", [])
                for atom_entry in scored_atoms[:5]:
                    label = self._extract_label(atom_entry)
                    if label:
                        poss_id = uuid.uuid4().hex[:8]
                        possibilities.append(Possibility(
                            possibility_id=poss_id,
                            statement=f"MCTS-scored atom '{label}' is relevant to '{query[:50]}'",
                            reasoning=(
                                f"MCTS scoring identified '{label}' as a high-value "
                                f"reasoning atom for the query."
                            ),
                            confidence=0.4,
                            state="proposed",
                            explained_evidence=set(),
                            all_evidence=all_evidence,
                            generation=0,
                            source="cross_reference",
                        ))
        except Exception as exc:
            logger.debug("mcts_query() cross-reference generation failed: %s", exc)

        return possibilities

    def _generate_from_anomaly(
        self,
        anomaly: Anomaly,
        all_evidence: set[str],
    ) -> list[Possibility]:
        """Generate possibilities from an anomaly (delegates to HypothesisDrivenReasoner).

        Uses the existing reasoner to generate hypotheses from the anomaly,
        then converts them to Possibilities.
        """
        possibilities: list[Possibility] = []

        # Generate hypotheses using the existing reasoner
        try:
            hypotheses = self._reasoner._generate_hypotheses(anomaly, [])
            for hyp in hypotheses:
                poss_id = uuid.uuid4().hex[:8]
                explained = {e.evidence_id for e in hyp.confirmatory_evidence}
                possibilities.append(Possibility(
                    possibility_id=poss_id,
                    statement=hyp.statement,
                    reasoning=hyp.reasoning,
                    confidence=hyp.confidence,
                    state="proposed",
                    explained_evidence=explained,
                    all_evidence=all_evidence,
                    generation=0,
                    source="anomaly",
                    anomaly_source=anomaly.concept,
                    confirmatory_evidence=list(hyp.confirmatory_evidence),
                    disconfirmatory_evidence=list(hyp.disconfirmatory_evidence),
                ))
        except Exception as exc:
            logger.debug("Anomaly-based generation failed: %s", exc)

        return possibilities

    # ==================================================================
    # PHASE 2a: ELIMINATE — Remove impossibles
    # ==================================================================

    def _eliminate(
        self,
        possibilities: list[Possibility],
        all_evidence: set[str],
    ) -> tuple[list[Possibility], int]:
        """Eliminate possibilities below the confidence threshold.

        Uses both evidence-based scoring and RSVS appraise() to determine
        which possibilities should be removed.

        Soft elimination: possibilities with very low confidence are removed,
        but those with moderate confidence are kept for potential hybridization.

        Args:
            possibilities: Current set of possibilities.
            all_evidence: All accumulated evidence.

        Returns:
            Tuple of (surviving possibilities, count of eliminated).
        """
        surviving: list[Possibility] = []
        eliminated_count = 0

        for poss in possibilities:
            # Skip already concluded possibilities
            if poss.state in ("concluded", "eliminated"):
                if poss.state == "eliminated":
                    eliminated_count += 1
                else:
                    surviving.append(poss)
                continue

            # Check with RSVS appraise() if available
            if self.rsvs_available:
                try:
                    appraise_result = self._bridge.appraise(poss.statement)
                    if appraise_result and isinstance(appraise_result, dict):
                        disagree_pct = appraise_result.get("disagree_pct", 0)
                        if isinstance(disagree_pct, (int, float)) and float(disagree_pct) > 0.7:
                            # Strong disconfirmation from RSVS
                            poss.confidence *= 0.5
                except Exception as exc:
                    logger.debug("appraise() elimination check failed: %s", exc)

            # Apply net evidence score
            net_score = poss.net_evidence_score
            if net_score < -0.3:
                # Net disconfirmatory evidence outweighs confirmatory
                poss.confidence *= 0.7

            # Soft elimination: remove below threshold
            if poss.confidence < self._elimination_threshold:
                poss.state = "eliminated"
                eliminated_count += 1
                logger.debug(
                    "Eliminated possibility '%s' (confidence=%.3f)",
                    poss.statement[:40], poss.confidence,
                )
            else:
                poss.state = "surviving"
                surviving.append(poss)

        logger.debug(
            "Elimination: %d → %d surviving (%d eliminated)",
            len(possibilities), len(surviving), eliminated_count,
        )

        return surviving, eliminated_count

    # ==================================================================
    # PHASE 2b: HYBRIDIZE — Combine complementary pairs
    # ==================================================================

    def _hybridize(
        self,
        possibilities: list[Possibility],
        all_evidence: set[str],
        generation: int,
    ) -> tuple[list[Possibility], list[Possibility]]:
        """Hybridize complementary possibility pairs (A × B).

        Two possibilities are complementary if:
        - They explain DIFFERENT parts of the evidence (low overlap)
        - Together they cover MORE evidence (high joint coverage)
        - Neither subsumes the other

        The hybrid is NOT A ∧ B (conjunction), but A × B (composition):
        a new possibility that combines insights from both parents
        and may reveal entirely new avenues of reasoning.

        Additionally, traces implications of each hybrid to find
        emergent possibilities that neither parent could see alone.

        Args:
            possibilities: Current surviving possibilities.
            all_evidence: All accumulated evidence.
            generation: Current lattice generation number.

        Returns:
            Tuple of (hybrid possibilities, emergent possibilities from implications).
        """
        hybrids: list[Possibility] = []
        emergent: list[Possibility] = []

        if len(possibilities) < 2:
            return hybrids, emergent

        # Find complementary pairs and hybridize
        hybrid_count = 0
        for i, p_a in enumerate(possibilities):
            if hybrid_count >= self._max_hybrids_per_generation:
                break
            for p_b in possibilities[i + 1:]:
                if hybrid_count >= self._max_hybrids_per_generation:
                    break

                complementarity = self._measure_complementarity(p_a, p_b)
                if complementarity >= self._complementarity_threshold:
                    hybrid = self._compose(p_a, p_b, all_evidence, generation)
                    hybrids.append(hybrid)
                    hybrid_count += 1

                    # Trace implications — the CRUCIAL step
                    implications = self._trace_implications(
                        hybrid, all_evidence, generation,
                    )
                    emergent.extend(implications)

        logger.debug(
            "Hybridization: %d complementary pairs → %d hybrids, %d emergent",
            hybrid_count, len(hybrids), len(emergent),
        )

        return hybrids, emergent

    def _measure_complementarity(
        self,
        a: Possibility,
        b: Possibility,
    ) -> float:
        """Measure how complementary two possibilities are.

        Two possibilities are complementary if:
        - They explain DIFFERENT parts of the evidence (low overlap)
        - Together they cover MORE evidence (high joint coverage)
        - Neither subsumes the other

        Returns:
            Complementarity score (0.0–1.0).
        """
        # Evidence overlap
        joint = a.explained_evidence | b.explained_evidence
        if not joint:
            # No evidence overlap — use structural similarity as proxy
            if self.rsvs_available:
                try:
                    sim = self._bridge.structural_similarity(
                        a.statement[:50], b.statement[:50],
                    )
                    if sim and isinstance(sim, dict):
                        sim_val = sim.get("structural_similarity", 0.5)
                        if isinstance(sim_val, (int, float)):
                            # Low similarity = high complementarity
                            return 1.0 - float(sim_val)
                except Exception:
                    pass
            # No evidence and no RSVS — moderate complementarity
            return 0.3

        overlap = a.explained_evidence & b.explained_evidence
        overlap_ratio = len(overlap) / len(joint)
        joint_coverage = len(joint) / max(len(a.all_evidence), 1)

        # High complementarity = low overlap + high joint coverage
        return joint_coverage * (1.0 - overlap_ratio)

    def _compose(
        self,
        a: Possibility,
        b: Possibility,
        all_evidence: set[str],
        generation: int,
    ) -> Possibility:
        """Create a hybrid possibility from two complementary parents.

        This is NOT A ∧ B (conjunction), but A × B (composition):
        the hybrid combines insights from both parents and may
        have properties that neither parent has alone.

        The hybrid gets a confidence boost because it explains
        more evidence than either parent alone.
        """
        poss_id = uuid.uuid4().hex[:8]
        combined_evidence = a.explained_evidence | b.explained_evidence

        # Confidence: minimum of parents * boost (covers more evidence)
        base_conf = min(a.confidence, b.confidence)
        # Boost proportional to additional coverage
        a_coverage = a.coverage
        b_coverage = b.coverage
        combined_coverage = len(combined_evidence) / max(len(all_evidence), 1)
        coverage_boost = combined_coverage / max(max(a_coverage, b_coverage), 0.01)
        hybrid_conf = min(0.95, base_conf * min(coverage_boost, _DEFAULT_HYBRID_CONFIDENCE_BOOST))

        hybrid = Possibility(
            possibility_id=poss_id,
            statement=f"{a.statement} ∘ {b.statement}",
            reasoning=(
                f"Hybrid of [{a.possibility_id}] and [{b.possibility_id}]: "
                f"combining '{a.statement[:40]}' with '{b.statement[:40]}' "
                f"creates a more complete explanation covering {len(combined_evidence)} "
                f"evidence items (vs {len(a.explained_evidence)} + {len(b.explained_evidence)} individually)."
            ),
            confidence=hybrid_conf,
            state="hybrid",
            explained_evidence=combined_evidence,
            all_evidence=all_evidence,
            parent_ids=[a.possibility_id, b.possibility_id],
            generation=generation,
            source="hybrid",
            anomaly_source=a.anomaly_source or b.anomaly_source,
            confirmatory_evidence=list(
                {e.evidence_id: e for e in a.confirmatory_evidence + b.confirmatory_evidence}.values()
            ),
            disconfirmatory_evidence=list(
                {e.evidence_id: e for e in a.disconfirmatory_evidence + b.disconfirmatory_evidence}.values()
            ),
        )

        return hybrid

    def _trace_implications(
        self,
        hybrid: Possibility,
        all_evidence: set[str],
        generation: int,
    ) -> list[Possibility]:
        """Trace implications of a hybrid to find emergent possibilities.

        The CRUCIAL step: a hybrid may open doors that neither parent
        could see alone. For example:
        - A="betrayed" + B="pride wounded"
        - Hybrid="betrayal of pride"
        - Implies="pattern of repeated trust violations"
        - Implies="childhood origin of trust issues"

        Each implication is a NEW possibility that was NOT in the
        original possibility space.

        Uses RSVS mcts_query() for implication traversal.
        """
        emergent: list[Possibility] = []

        if not self.rsvs_available:
            return emergent

        try:
            mcts_result = self._bridge.mcts_query(
                node_label=hybrid.statement[:100],
                max_depth=_DEFAULT_IMPLICATION_DEPTH,
                simulations=15,
            )
            if mcts_result and isinstance(mcts_result, dict):
                best_path = mcts_result.get("best_path", [])
                for node_label in best_path[:3]:
                    if not isinstance(node_label, str):
                        continue

                    # Check if this is genuinely new
                    poss_id = uuid.uuid4().hex[:8]
                    emergent.append(Possibility(
                        possibility_id=poss_id,
                        statement=(
                            f"Implication of hybrid: '{node_label}' — "
                            f"emerged from combining '{hybrid.statement[:40]}'"
                        ),
                        reasoning=(
                            f"Tracing implications of hybrid [{hybrid.possibility_id}] "
                            f"revealed '{node_label}' — a possibility that was not "
                            f"visible from either parent alone."
                        ),
                        confidence=hybrid.confidence * 0.8,  # Slightly lower — untested
                        state="emergent",
                        explained_evidence=set(),
                        all_evidence=all_evidence,
                        parent_ids=[hybrid.possibility_id],
                        generation=generation,
                        source="emergent",
                        anomaly_source=hybrid.anomaly_source,
                    ))
        except Exception as exc:
            logger.debug("Implication tracing failed: %s", exc)

        return emergent

    # ==================================================================
    # PHASE 2c: DETECT NOVEL — Filter genuinely novel hybrids
    # ==================================================================

    def _detect_novel(
        self,
        candidates: list[Possibility],
        existing: list[Possibility],
    ) -> list[Possibility]:
        """Filter candidates to keep only genuinely novel insights.

        A candidate is "novel" if it introduces meaning that
        no existing possibility contains. We use structural_similarity
        from RSVS as the primary filter, falling back to keyword overlap.

        Args:
            candidates: New possibilities (hybrids or emergent) to check.
            existing: Already-existing possibilities to compare against.

        Returns:
            List of genuinely novel possibilities.
        """
        novel: list[Possibility] = []

        for candidate in candidates:
            is_novel = True

            for existing_p in existing:
                # Use RSVS structural_similarity if available
                if self.rsvs_available:
                    try:
                        sim = self._bridge.structural_similarity(
                            candidate.statement[:50],
                            existing_p.statement[:50],
                        )
                        if sim and isinstance(sim, dict):
                            sim_val = sim.get("structural_similarity", 0.0)
                            if isinstance(sim_val, (int, float)):
                                if float(sim_val) > self._novelty_threshold:
                                    is_novel = False
                                    break
                            continue
                    except Exception:
                        pass

                # Fallback: keyword overlap
                cand_words = set(candidate.statement.lower().split())
                exist_words = set(existing_p.statement.lower().split())
                if cand_words and exist_words:
                    overlap = len(cand_words & exist_words) / len(cand_words | exist_words)
                    if overlap > 0.8:
                        is_novel = False
                        break

            if is_novel:
                novel.append(candidate)

        return novel

    # ==================================================================
    # PHASE 2e: QUESTION MODE — Find best discriminating question
    # ==================================================================

    def _find_best_question(self, possibilities: list[Possibility]) -> str | None:
        """Find the question that would best discriminate between remaining possibilities.

        This implements the epistemic value principle from Active Inference:
        we want to ask the question that maximizes expected information gain,
        i.e., the question whose answer would most reduce our uncertainty
        about which possibility is correct.

        Strategy:
        1. For each pair of remaining possibilities, find what differentiates them
        2. Formulate a question that would resolve the difference
        3. Pick the question that would eliminate the most possibilities

        Args:
            possibilities: The remaining possibilities (≤ question_mode_threshold).

        Returns:
            The best discriminating question, or None if no good question found.
        """
        if len(possibilities) < 2:
            return None

        # Find the two possibilities with highest confidence
        sorted_poss = sorted(possibilities, key=lambda p: p.confidence, reverse=True)
        top = sorted_poss[0]
        runner_up = sorted_poss[1]

        # Find what differentiates them
        diff_evidence = top.explained_evidence.symmetric_difference(
            runner_up.explained_evidence
        )

        if diff_evidence:
            # Ask about the differentiating evidence
            evidence_str = ", ".join(sorted(diff_evidence)[:3])
            return (
                f"Which interpretation better explains the evidence "
                f"[{evidence_str}]: "
                f"(A) {top.statement[:60]} or "
                f"(B) {runner_up.statement[:60]}?"
            )

        # No evidence difference — ask about the core claim
        return (
            f"Between these interpretations, which is more likely: "
            f"(A) {top.statement[:80]} or "
            f"(B) {runner_up.statement[:80]}?"
        )

    # ==================================================================
    # PHASE 2f: Seek discriminating evidence
    # ==================================================================

    def _seek_discriminating_evidence(
        self,
        possibilities: list[Possibility],
    ) -> set[str]:
        """Seek new evidence that would help discriminate between possibilities.

        Uses RSVS mcts_query() and appraise() to find evidence that
        would help eliminate some possibilities or strengthen others.

        Args:
            possibilities: Current surviving possibilities.

        Returns:
            Set of new evidence strings found.
        """
        new_evidence: set[str] = []

        if not self.rsvs_available:
            return set(new_evidence)

        # For the top possibilities, find evidence they DON'T explain
        for poss in possibilities[:5]:
            unexplained = poss.all_evidence - poss.explained_evidence
            if not unexplained:
                continue

            try:
                # Query RSVS for evidence about unexplained items
                for evidence_item in list(unexplained)[:2]:
                    mcts_result = self._bridge.mcts_query(
                        node_label=evidence_item[:100],
                        max_depth=2,
                        simulations=10,
                    )
                    if mcts_result and isinstance(mcts_result, dict):
                        best_path = mcts_result.get("best_path", [])
                        for node_label in best_path[:2]:
                            if isinstance(node_label, str) and node_label not in poss.all_evidence:
                                new_evidence.append(node_label)
            except Exception as exc:
                logger.debug("Discriminating evidence search failed: %s", exc)

        return set(new_evidence)

    # ==================================================================
    # Score against evidence
    # ==================================================================

    def _score_against_evidence(
        self,
        possibilities: list[Possibility],
        all_evidence: set[str],
    ) -> None:
        """Re-score all possibilities against accumulated evidence.

        For each possibility, check how well the evidence supports
        or undermines it, and update confidence accordingly.
        """
        for poss in possibilities:
            if poss.state in ("eliminated", "concluded"):
                continue

            # Update all_evidence reference
            poss.all_evidence = all_evidence

            # Count how much evidence this possibility explains
            if all_evidence:
                coverage = len(poss.explained_evidence & all_evidence) / len(all_evidence)
                # Adjust confidence based on coverage
                if coverage > 0.5:
                    poss.confidence = min(0.95, poss.confidence * 1.05)
                elif coverage < 0.1 and len(all_evidence) > 3:
                    poss.confidence *= 0.95

            # Check net evidence score
            net = poss.net_evidence_score
            if net < -0.2:
                poss.confidence = max(0.05, poss.confidence * 0.85)
            elif net > 0.2:
                poss.confidence = min(0.95, poss.confidence * 1.05)

    # ==================================================================
    # GENERATIVE MODE — Delegate to HypothesisDrivenReasoner
    # ==================================================================

    def _generative_mode(
        self,
        anomaly: Anomaly,
        context: list[str],
        pattern_result: PatternResult | None,
        result_id: str,
        query: str,
    ) -> LatticeResult:
        """Delegate to HypothesisDrivenReasoner for generative mode.

        Used when the lattice mode determines that generative reasoning
        (top-K hypothesis) is more appropriate than lattice reasoning.
        """
        cycle_result = self._reasoner.reason(
            anomaly=anomaly,
            context=context,
            pattern_result=pattern_result,
        )

        # Convert HypothesisCycleResult to LatticeResult
        possibilities = []
        for hyp in cycle_result.hypotheses:
            poss_id = uuid.uuid4().hex[:8]
            explained = {e.evidence_id for e in hyp.confirmatory_evidence}
            possibilities.append(Possibility(
                possibility_id=poss_id,
                statement=hyp.statement,
                reasoning=hyp.reasoning,
                confidence=hyp.confidence,
                state="concluded" if hyp.state == "confirmed" else "proposed",
                explained_evidence=explained,
                parent_ids=[],
                generation=0,
                source="anomaly",
                anomaly_source=anomaly.concept,
            ))

        conclusion = None
        if cycle_result.winning_hypothesis:
            conclusion = possibilities[0] if possibilities else None
            for p in possibilities:
                if p.statement == cycle_result.winning_hypothesis.statement:
                    conclusion = p
                    break

        return LatticeResult(
            result_id=result_id,
            query=query,
            mode="generative",
            generations=[LatticeGeneration(
                generation=0,
                pre_count=len(possibilities),
                post_elimination=len(possibilities),
            )],
            surviving_possibilities=possibilities,
            conclusion=conclusion,
            is_conclusive=cycle_result.is_conclusive,
            confidence=conclusion.confidence if conclusion else 0.0,
        )

    # ==================================================================
    # Deduplication
    # ==================================================================

    def _deduplicate(
        self,
        possibilities: list[Possibility],
    ) -> list[Possibility]:
        """Deduplicate possibilities by semantic similarity.

        Uses structural_similarity() when RSVS is available,
        falls back to keyword overlap.
        """
        if not possibilities:
            return possibilities

        unique: list[Possibility] = []
        seen_signatures: set[str] = set()

        for poss in possibilities:
            # Quick signature check
            sig = poss.statement[:80].lower().strip()
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            # Deeper similarity check against existing uniques
            is_duplicate = False
            for existing in unique:
                if self._are_similar(poss.statement, existing.statement):
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if poss.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(poss)
                    break

            if not is_duplicate:
                unique.append(poss)

        return unique

    def _are_similar(self, text_a: str, text_b: str) -> bool:
        """Check if two texts are semantically similar."""
        if self.rsvs_available:
            try:
                sim = self._bridge.structural_similarity(
                    text_a[:50], text_b[:50],
                )
                if sim and isinstance(sim, dict):
                    sim_val = sim.get("structural_similarity", 0.0)
                    if isinstance(sim_val, (int, float)):
                        return float(sim_val) > 0.85
            except Exception:
                pass

        # Fallback: keyword overlap
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b) / len(words_a | words_b)
        return overlap > 0.8

    # ==================================================================
    # Utility methods
    # ==================================================================

    @staticmethod
    def _extract_key_concepts(text: str) -> list[str]:
        """Extract key concepts from text for possibility generation."""
        stop = {
            "that", "this", "with", "from", "have", "been", "they",
            "their", "which", "would", "there", "could", "about",
            "yang", "dan", "dari", "untuk", "dengan", "adalah", "itu",
            "ini", "ke", "di", "pada", "tidak", "akan", "telah", "oleh",
            "adalah", "sebuah", "suatu", "atau", "juga", "karena",
        }
        words = [w.strip() for w in text.split() if len(w.strip()) > 2]
        return [w for w in words if w.lower() not in stop][:15]

    @staticmethod
    def _extract_label(entry: Any) -> str | None:
        """Extract a label from various RSVS result formats."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            return str(entry[0])
        if isinstance(entry, dict):
            return str(entry.get("label", entry.get("node", "")))
        return None

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def reasoner(self) -> HypothesisDrivenReasoner:
        """Access the underlying HypothesisDrivenReasoner."""
        return self._reasoner

    @property
    def predictive_engine(self) -> PredictiveEngine:
        """Access the underlying PredictiveEngine."""
        return self._predictive_engine

    @property
    def lattice_history(self) -> list[LatticeResult]:
        """Return history of lattice reasoning results."""
        return list(self._lattice_history)

    @property
    def total_lattice_runs(self) -> int:
        """Return total number of lattice reasoning runs."""
        return len(self._lattice_history)
