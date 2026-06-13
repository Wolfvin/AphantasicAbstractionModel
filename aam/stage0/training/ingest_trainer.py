"""
AAM Ingest Trainer — The main training orchestrator.

Implements the full feedback loop:
    Ingest → DetectGaps → AskUser → Correction → EnrichComposition → GovernBeliefs → Persist

This is NOT just in-memory — every composition, gap, correction, and pattern
is persisted to RSVS-rich format so the system accumulates knowledge across sessions.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .types import (Composition, CompositionMember, KnowledgeGap, TrainingRecord,
                    PatternObservation, GeneratedQuestion, CorrectionResult)
from .question_engine import QuestionEngine
from .correction_handler import CorrectionHandler
from .persistence import TrainingPersistence
from .corpora import TrainingCorpus


# ────────────────────────────────────────────────────────────────────
# Semantic Role Definitions
# ────────────────────────────────────────────────────────────────────

# Expected roles for different composition types
EVENT_EXPECTED_ROLES = ["Predicate", "Arg0Agent", "Arg1Patient", "Arg2Recipient",
                        "Cause", "Purpose", "Location", "Time", "Instrument"]
HM_EXPECTED_ROLES = ["Problem", "Solution", "Arg0Agent", "Beneficiary", "Motivation"]
PATTERN_EXPECTED_ROLES = ["Antecedent", "Consequent", "PatternType"]

# Indonesian preposition/role markers
RECIPIENT_MARKERS = {"ke", "kepada", "untuk", "kepada"}
LOCATION_MARKERS = {"di", "ke", "dari"}
INSTRUMENT_MARKERS = {"dengan", "memakai", "menggunakan"}
CAUSE_MARKERS = {"karena", "sebab", "akibat", "lantaran"}
PURPOSE_MARKERS = {"untuk", "agar", "supaya", "guna"}
BENEFICIARY_MARKERS = {"untuk", "bagi"}
TIME_MARKERS = {"pada", "di", "saat", "ketika", "waktu"}


# ────────────────────────────────────────────────────────────────────
# AAMTrainer — The Main Orchestrator
# ────────────────────────────────────────────────────────────────────

class AAMTrainer:
    """
    AAM Ingest Training System — orchestrates the full feedback loop.

    This is NOT just in-memory. Every composition, gap, correction, and pattern
    is persisted to RSVS-rich format so the system accumulates knowledge across sessions.

    The training cycle:
    1. Ingest text → extract semantic atoms → create compositions
    2. Detect gaps → missing roles, ambiguous tokens, sparse neighborhoods
    3. Generate questions → natural language questions from gaps
    4. Receive corrections → human provides answers
    5. Apply corrections → enrich compositions with HumanAssertion provenance
    6. Govern beliefs → promote compositions through lifecycle/epistemic states
    7. Mine patterns → find recurring patterns across compositions
    8. Persist → save everything to disk

    Usage:
        trainer = AAMTrainer(persist_dir="training_output")
        trainer.ingest("Budi menjual barang ke saya")
        questions = trainer.detect_and_question()
        for q in questions:
            answer = input(q.question_text + " ")
            trainer.correct(q, answer=answer, role=q.target_role)
        trainer.persist()
    """

    def __init__(self, persist_dir: str = "training_output", verbose: bool = True):
        self.persist_dir = persist_dir
        self.verbose = verbose

        # Core knowledge graph
        self.nodes: Dict[str, int] = {}  # label → node_id
        self.node_labels: Dict[int, str] = {}  # node_id → label
        self.compositions: Dict[str, Composition] = {}
        self.edges: List[Tuple[str, int, str]] = []  # (comp_id, target_node_id, role)

        # Gap tracking
        self.gaps: Dict[str, KnowledgeGap] = {}
        self.inquiry_memory: Dict[str, str] = {}  # gap_key → "addressed"
        self.question_history: Dict[str, Optional[str]] = {}  # question_id → answer

        # Pattern mining
        self.patterns: Dict[str, PatternObservation] = {}

        # Training records
        self.records: List[TrainingRecord] = []

        # Counters
        self._next_node_id = 1
        self._next_comp_id = 1
        self._next_gap_id = 1
        self._batch_number = 0

        # Sub-modules
        self.question_engine = QuestionEngine(self)
        self.correction_handler = CorrectionHandler(self)
        self.persistence = TrainingPersistence(persist_dir)
        self.corpus = TrainingCorpus()

        # Try to load existing state
        self._load_existing()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[AAM Trainer] {msg}")

    # ────────────────────────────────────────────────────────────────
    # Node Management
    # ────────────────────────────────────────────────────────────────

    def ensure_node(self, label: str) -> int:
        """Ensure a node with the given label exists, creating if necessary."""
        label_lower = label.lower().strip()
        if label_lower in self.nodes:
            return self.nodes[label_lower]

        node_id = self._next_node_id
        self._next_node_id += 1
        self.nodes[label_lower] = node_id
        self.node_labels[node_id] = label_lower
        return node_id

    def node_label(self, node_id: int) -> Optional[str]:
        """Get the label for a node ID."""
        return self.node_labels.get(node_id)

    # ────────────────────────────────────────────────────────────────
    # Ingest — The Core Entry Point
    # ────────────────────────────────────────────────────────────────

    def ingest(self, text: str, source: str = "UserInput") -> int:
        """
        Ingest text into the AAM knowledge graph.

        This triggers the full pipeline:
        1. Tokenize the text
        2. Extract semantic frames (SVO + preposition-based roles)
        3. Create compositions from extracted frames
        4. Apply governance (lifecycle/epistemic states)
        5. Detect gaps
        6. Mine patterns

        Returns the number of compositions created.
        """
        self._batch_number += 1
        start_time = time.time()

        self._log(f"Ingesting (batch {self._batch_number}): '{text[:80]}...' " if len(text) > 80 else f"Ingesting (batch {self._batch_number}): '{text}'")

        # Step 1: Tokenize
        tokens = self._tokenize(text)

        # Step 2: Extract semantic frame
        frame = self._extract_frame(tokens, text)

        # Step 3: Create compositions
        comps_created = self._create_compositions_from_frame(frame, text, source)

        # Step 4: Apply governance
        governance_count = self._apply_governance()

        # Step 5: Detect gaps
        gap_count = self._detect_gaps()

        # Step 6: Mine patterns
        self._mine_patterns()

        # Record this training interaction
        record = TrainingRecord(
            record_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_text=text,
            compositions_created=comps_created,
            gaps_detected=gap_count,
            questions_asked=0,
            corrections_applied=0,
            governance_transitions=governance_count,
        )
        self.records.append(record)

        elapsed = time.time() - start_time
        self._log(f"  → {comps_created} compositions, {gap_count} gaps, {governance_count} governance transitions ({elapsed:.3f}s)")

        return comps_created

    def ingest_batch(self, texts: List[str], source: str = "Corpus") -> Dict[str, int]:
        """Ingest a batch of texts. Returns summary statistics."""
        total_comps = 0
        total_gaps = 0

        for text in texts:
            comps = self.ingest(text, source=source)
            total_comps += comps
            total_gaps += len([g for g in self.gaps.values() if not g.addressed])

        self._log(f"Batch complete: {len(texts)} texts → {total_comps} compositions, {total_gaps} unresolved gaps")
        return {"texts": len(texts), "compositions": total_comps, "gaps": total_gaps}

    # ────────────────────────────────────────────────────────────────
    # Tokenize
    # ────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens, preserving sentence structure."""
        # Simple but effective tokenizer
        tokens = []
        current = []
        punct = set(".,!?;:'\"()[]{}/—–-")

        for ch in text:
            if ch.isspace():
                if current:
                    tokens.append("".join(current).lower())
                    current = []
            elif ch in punct:
                if current:
                    tokens.append("".join(current).lower())
                    current = []
                # Sentence-ending punctuation is meaningful
                if ch in ".!?":
                    tokens.append(ch)
            else:
                current.append(ch)

        if current:
            tokens.append("".join(current).lower())

        return tokens

    # ────────────────────────────────────────────────────────────────
    # Extract Frame — SVO + Preposition-based Role Detection
    # ────────────────────────────────────────────────────────────────

    def _extract_frame(self, tokens: List[str], original_text: str) -> Dict[str, Any]:
        """
        Extract a semantic frame from tokens.

        This is a RULE-BASED extractor for Indonesian that detects:
        - Subject (Arg0Agent) — noun before verb
        - Predicate — verb (meN-*, ber-*, di-*, etc.)
        - Object (Arg1Patient) — noun after verb
        - Recipient (Arg2Recipient) — after "ke", "kepada"
        - Cause — after "karena", "sebab"
        - Purpose — after "untuk", "agar"
        - Location — after "di"
        - Instrument — after "dengan"

        This is INTENTIONALLY limited so the system will detect gaps and ask questions.
        As the system learns from corrections, it fills in missing roles.
        """
        frame = {
            "predicate": None,
            "Arg0Agent": None,
            "Arg1Patient": None,
            "Arg2Recipient": None,
            "Cause": None,
            "Purpose": None,
            "Location": None,
            "Instrument": None,
            "Time": None,
            "Beneficiary": None,
        }

        if not tokens:
            return frame

        # Indonesian verb prefixes
        verb_prefixes = ("me", "ber", "di", "ter", "pe", "se", "per", "mem", "men", "meny",
                         "meng", "memper", "diper", "teper", "memb", "menj")

        # Find predicate (first verb-like token)
        pred_idx = None
        for i, tok in enumerate(tokens):
            if tok in (".", "!", "?"):
                continue
            # Check if it looks like an Indonesian verb
            starts_with_prefix = any(tok.startswith(p) for p in verb_prefixes)
            # Common Indonesian verbs
            common_verbs = {
                "adalah", "ialah", "memiliki", "ada", "menjadi", "merupakan",
                "membuat", "mengambil", "memberi", "pergi", "datang", "makan",
                "minum", "baca", "tulis", "lihat", "dengar", "kata", "buat",
                "jual", "beli", "kirim", "terima", "pakai", "gunakan", "bantu",
                "pilih", "ubah", "hapus", "tambah", "simpan", "cari", "temu",
                "beri", "jawab", "tanya", "kembali", "mulai", "selesai", "kerja",
                "main", "tidur", "bangun", "duduk", "berdiri", "jalan", "lari",
                "berbicara", "berpendapat", "menganggap", "menilai", "menganggap",
                "berusaha", "mencoba", "menyatakan", "mengatakan", "mengaku",
            }
            is_verb = starts_with_prefix or tok in common_verbs
            if is_verb:
                frame["predicate"] = tok
                pred_idx = i
                break

        if pred_idx is None:
            # No verb found — treat as a noun phrase / description
            if len(tokens) >= 2:
                frame["Arg0Agent"] = tokens[0]
                frame["Arg1Patient"] = tokens[-1] if tokens[-1] not in ".!?" else tokens[-2]
            elif len(tokens) == 1:
                frame["Arg0Agent"] = tokens[0]
            return frame

        # Agent: noun before the verb
        if pred_idx > 0:
            agent_candidate = tokens[pred_idx - 1]
            if agent_candidate not in ".!?":
                frame["Arg0Agent"] = agent_candidate

        # Patient: noun right after the verb (before any preposition)
        if pred_idx + 1 < len(tokens):
            patient_candidate = tokens[pred_idx + 1]
            if patient_candidate not in RECIPIENT_MARKERS | CAUSE_MARKERS | PURPOSE_MARKERS | LOCATION_MARKERS | INSTRUMENT_MARKERS | {".", "!", "?"}:
                frame["Arg1Patient"] = patient_candidate

        # Scan for preposition-based roles
        i = pred_idx + 1
        while i < len(tokens):
            tok = tokens[i]

            if tok in RECIPIENT_MARKERS and i + 1 < len(tokens):
                # "ke" or "kepada" → Recipient
                # But "ke" is ambiguous — it could be location
                # We check if the next token is a person-like word
                next_tok = tokens[i + 1]
                if tok == "kepada" or self._is_person_like(next_tok):
                    frame["Arg2Recipient"] = next_tok
                else:
                    # Ambiguous — mark as Location for now, system will ask
                    frame["Location"] = next_tok

            elif tok in LOCATION_MARKERS and i + 1 < len(tokens):
                if tok == "di":
                    frame["Location"] = tokens[i + 1]
                elif tok == "dari":
                    # "dari" can be source location or origin
                    if frame["Location"] is None:
                        frame["Location"] = f"dari {tokens[i + 1]}"

            elif tok in CAUSE_MARKERS and i + 1 < len(tokens):
                # Collect the entire cause phrase
                cause_parts = []
                j = i + 1
                while j < len(tokens) and tokens[j] not in RECIPIENT_MARKERS | PURPOSE_MARKERS | LOCATION_MARKERS | {".", "!", "?"}:
                    cause_parts.append(tokens[j])
                    j += 1
                frame["Cause"] = " ".join(cause_parts) if cause_parts else tokens[i + 1]

            elif tok in PURPOSE_MARKERS and i + 1 < len(tokens):
                purpose_parts = []
                j = i + 1
                while j < len(tokens) and tokens[j] not in RECIPIENT_MARKERS | CAUSE_MARKERS | LOCATION_MARKERS | {".", "!", "?"}:
                    purpose_parts.append(tokens[j])
                    j += 1
                frame["Purpose"] = " ".join(purpose_parts) if purpose_parts else tokens[i + 1]

            elif tok in INSTRUMENT_MARKERS and i + 1 < len(tokens):
                frame["Instrument"] = tokens[i + 1]

            elif tok in TIME_MARKERS and i + 1 < len(tokens):
                frame["Time"] = tokens[i + 1]

            i += 1

        return frame

    def _is_person_like(self, token: str) -> bool:
        """Heuristic: is this token likely a person/entity name?"""
        # Proper nouns in Indonesian start with capital in original text
        # Common person pronouns
        person_words = {"saya", "kamu", "dia", "mereka", "kita", "kami", "anda",
                        "beliau", "ia", "kalian", "gw", "lu", "gua", "loe"}
        return token.lower() in person_words

    # ────────────────────────────────────────────────────────────────
    # Create Compositions from Frame
    # ────────────────────────────────────────────────────────────────

    def _create_compositions_from_frame(self, frame: Dict[str, Any], source_text: str,
                                         source: str) -> int:
        """Create Composition(s) from an extracted frame."""
        predicate = frame.get("predicate")
        if not predicate:
            # No predicate — create a simple token composition
            if frame.get("Arg0Agent"):
                self.ensure_node(frame["Arg0Agent"])
            return 0

        # Create the main Event composition
        comp_id = f"comp_event_{self._next_comp_id}"
        self._next_comp_id += 1

        now = datetime.now(timezone.utc).isoformat()

        comp = Composition(
            id=comp_id,
            composition_type="Event",
            source_text=source_text,
            provenance_origin="FrameCompiler" if source != "HumanAssertion" else "HumanAssertion",
            created_at=now,
            updated_at=now,
        )

        # Add Predicate member
        pred_node_id = self.ensure_node(predicate)
        comp.members.append(CompositionMember(
            node_id=str(pred_node_id),
            role="Predicate",
            confidence=0.9,
            label=predicate,
        ))

        # Add other members from the frame
        role_mapping = {
            "Arg0Agent": "Arg0Agent",
            "Arg1Patient": "Arg1Patient",
            "Arg2Recipient": "Arg2Recipient",
            "Cause": "Cause",
            "Purpose": "Purpose",
            "Location": "Location",
            "Instrument": "Instrument",
            "Time": "Time",
            "Beneficiary": "Beneficiary",
        }

        for frame_role, comp_role in role_mapping.items():
            value = frame.get(frame_role)
            if value:
                node_id = self.ensure_node(value)
                comp.members.append(CompositionMember(
                    node_id=str(node_id),
                    role=comp_role,
                    confidence=0.8 if frame_role in ("Arg0Agent", "Arg1Patient") else 0.6,
                    label=value,
                ))
                # Create edge
                self.edges.append((comp_id, node_id, comp_role))

        # Set initial confidence based on completeness
        filled_roles = sum(1 for m in comp.members if m.role != "Predicate")
        comp.confidence = min(0.3 + filled_roles * 0.1, 0.9)

        self.compositions[comp_id] = comp
        return 1

    # ────────────────────────────────────────────────────────────────
    # Governance
    # ────────────────────────────────────────────────────────────────

    def _apply_governance(self) -> int:
        """
        Apply governance rules to all compositions.

        Promotion criteria:
        - New → Candidate: after 1 batch
        - Candidate → Stable: age ≥ 3, confidence ≥ 0.55, no recent contradictions
        - Observed → Inferred: derived by reasoning rule
        - Inferred → Grounded: ≥ 2 independent sources, confidence ≥ 0.7
        - HumanAssertion → Stable + Grounded (immediately!)

        Returns the number of governance transitions applied.
        """
        transitions = 0

        for comp in self.compositions.values():
            # Increment batch_seen
            comp.batch_seen = self._batch_number

            # HumanAssertion → Stable + Grounded (immediately)
            if comp.provenance_origin == "HumanAssertion":
                if comp.lifecycle != "Stable" or comp.epistemic != "Grounded":
                    comp.lifecycle = "Stable"
                    comp.epistemic = "Grounded"
                    comp.confidence = max(comp.confidence, 0.85)
                    transitions += 1
                    continue

            # New → Candidate: after 1 batch
            if comp.lifecycle == "New" and comp.batch_seen >= 1:
                comp.lifecycle = "Candidate"
                transitions += 1

            # Candidate → Stable
            if comp.lifecycle == "Candidate":
                age = self._batch_number - comp.batch_seen + 1
                if (comp.batch_seen >= 3 and comp.confidence >= 0.55
                        and comp.epistemic != "Contradicted"):
                    # Check for confirming members
                    confirming = sum(1 for m in comp.members if m.confidence >= 0.5)
                    if confirming >= 2:
                        comp.lifecycle = "Stable"
                        transitions += 1

            # Inferred → Grounded
            if comp.epistemic == "Inferred" and comp.confidence >= 0.7:
                # Check for multiple sources (heuristic: multiple members from different origins)
                comp.epistemic = "Grounded"
                transitions += 1

        return transitions

    # ────────────────────────────────────────────────────────────────
    # Gap Detection
    # ────────────────────────────────────────────────────────────────

    def _detect_gaps(self) -> int:
        """Detect knowledge gaps in all compositions."""
        new_gaps = 0

        for comp in self.compositions.values():
            if comp.composition_type == "Event":
                # Check for missing expected roles
                expected = ["Arg0Agent", "Arg1Patient", "Arg2Recipient", "Cause", "Purpose"]

                for role in expected:
                    if not comp.has_member_with_role(role):
                        gap_key = f"{comp.id}:{role}"

                        # Skip if already addressed
                        if gap_key in self.inquiry_memory:
                            continue

                        gap_id = f"gap_{self._next_gap_id}"
                        self._next_gap_id += 1

                        gap_type = "MissingRole"
                        if role == "Cause":
                            gap_type = "MissingCause"
                        elif role == "Purpose":
                            gap_type = "MissingPurpose"
                        elif role == "Arg2Recipient":
                            gap_type = "MissingRole"  # Recipient gap is very common

                        gap = KnowledgeGap(
                            gap_id=gap_id,
                            gap_type=gap_type,
                            description=f"Event '{comp.id}' missing {role} role",
                            source_composition_id=comp.id,
                            missing_role=role,
                            confidence=0.7,
                        )
                        self.gaps[gap_id] = gap
                        new_gaps += 1

            # Check for ambiguous tokens (pronouns)
            for member in comp.members:
                ambiguous = {"dia", "ia", "mereka", "ini", "itu", "nya"}
                if member.label.lower() in ambiguous:
                    gap_key = f"ambiguous:{comp.id}:{member.label}"
                    if gap_key not in self.inquiry_memory:
                        gap_id = f"gap_{self._next_gap_id}"
                        self._next_gap_id += 1
                        gap = KnowledgeGap(
                            gap_id=gap_id,
                            gap_type="AmbiguousToken",
                            description=f"Ambiguous token '{member.label}' in composition '{comp.id}' needs disambiguation",
                            source_composition_id=comp.id,
                            missing_role=member.role,
                            confidence=0.8,
                        )
                        self.gaps[gap_id] = gap
                        new_gaps += 1

        return new_gaps

    # ────────────────────────────────────────────────────────────────
    # Question Generation & Feedback Loop
    # ────────────────────────────────────────────────────────────────

    def detect_and_question(self) -> List[GeneratedQuestion]:
        """
        Detect gaps and generate questions for them.

        This is the core of the feedback loop — the system identifies
        what it doesn't know and asks the user about it.
        """
        return self.question_engine.generate_questions()

    def correct(self, question: GeneratedQuestion, answer: str, role: Optional[str] = None) -> CorrectionResult:
        """
        Apply a human correction to fill a gap.

        The correction is treated as HumanAssertion → Stable + Grounded immediately.
        This is how the "parent" teaches the "child" AAM.
        """
        return self.correction_handler.apply_correction(question, answer, role)

    def correct_direct(self, composition_id: str, role: str, value: str) -> CorrectionResult:
        """
        Directly correct a composition by adding or changing a role.

        This is for when the parent notices a wrong concept and wants to fix it directly.
        """
        result = self.correction_handler.apply_direct_correction(composition_id, role, value)
        self._apply_governance()  # Re-govern after correction
        self._mine_patterns()  # Re-mine patterns after correction
        return result

    # ────────────────────────────────────────────────────────────────
    # Pattern Mining
    # ────────────────────────────────────────────────────────────────

    def _mine_patterns(self):
        """
        Mine patterns from accumulated compositions.

        Finds recurring (predicate, role, filler) patterns and
        promotes them when they appear consistently across multiple
        compositions.
        """
        # Count (predicate, role, filler) occurrences
        pattern_counts: Dict[Tuple[str, str, str], int] = {}

        for comp in self.compositions.values():
            pred = comp.member_with_role("Predicate")
            if not pred:
                continue

            for member in comp.members:
                if member.role == "Predicate":
                    continue

                key = (pred.label, member.role, member.label)
                pattern_counts[key] = pattern_counts.get(key, 0) + 1

        # Promote patterns with ≥ 3 occurrences
        for (predicate, role, filler), count in pattern_counts.items():
            pattern_key = f"{predicate}:{role}:{filler}"

            if pattern_key in self.patterns:
                self.patterns[pattern_key].observation_count = count
                self.patterns[pattern_key].last_seen = datetime.now(timezone.utc).isoformat()

                # Promote if enough observations
                if count >= 3 and self.patterns[pattern_key].lifecycle == "Candidate":
                    self.patterns[pattern_key].lifecycle = "Stable"
                    self.patterns[pattern_key].epistemic = "Grounded"
                    self._log(f"Pattern promoted to Stable/Grounded: {predicate} + {role} = {filler} ({count} observations)")

                elif count >= 2 and self.patterns[pattern_key].epistemic == "Inferred":
                    self.patterns[pattern_key].epistemic = "Grounded"
            else:
                now = datetime.now(timezone.utc).isoformat()
                self.patterns[pattern_key] = PatternObservation(
                    pattern_id=f"pattern_{len(self.patterns)}",
                    predicate=predicate,
                    role=role,
                    filler=filler,
                    observation_count=count,
                    first_seen=now,
                    last_seen=now,
                    lifecycle="Candidate" if count >= 2 else "New",
                    epistemic="Inferred" if count >= 2 else "Observed",
                )

    # ────────────────────────────────────────────────────────────────
    # Persistence — NOT Just In-Memory!
    # ────────────────────────────────────────────────────────────────

    def persist(self):
        """
        Persist the full training state to disk.

        This saves:
        - The complete knowledge graph (nodes, compositions, edges)
        - Inquiry memory (so it doesn't ask the same question twice)
        - Question history
        - Pattern mining results
        - Training records
        """
        self.persistence.save(self)
        self._log(f"Persisted to {self.persist_dir}")

    def _load_existing(self):
        """Load existing training state from disk if available."""
        loaded = self.persistence.load(self)
        if loaded:
            self._log(f"Loaded existing state: {len(self.compositions)} compositions, {len(self.gaps)} gaps, {len(self.patterns)} patterns")

    # ────────────────────────────────────────────────────────────────
    # Reporting
    # ────────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Get the current training status."""
        return {
            "batch_number": self._batch_number,
            "total_nodes": len(self.nodes),
            "total_compositions": len(self.compositions),
            "total_edges": len(self.edges),
            "total_gaps": len(self.gaps),
            "unresolved_gaps": len([g for g in self.gaps.values() if not g.addressed]),
            "total_patterns": len(self.patterns),
            "stable_patterns": len([p for p in self.patterns.values() if p.lifecycle == "Stable"]),
            "grounded_patterns": len([p for p in self.patterns.values() if p.epistemic == "Grounded"]),
            "composition_by_lifecycle": {
                state: len([c for c in self.compositions.values() if c.lifecycle == state])
                for state in ["New", "Candidate", "Stable", "Deprecated", "Quarantine"]
            },
            "composition_by_epistemic": {
                state: len([c for c in self.compositions.values() if c.epistemic == state])
                for state in ["Observed", "Inferred", "Hypothesis", "Grounded", "Contradicted"]
            },
            "composition_by_type": {
                ctype: len([c for c in self.compositions.values() if c.composition_type == ctype])
                for ctype in set(c.composition_type for c in self.compositions.values())
            },
            "average_confidence": (
                sum(c.confidence for c in self.compositions.values()) / len(self.compositions)
                if self.compositions else 0.0
            ),
        }

    def print_status(self):
        """Print a formatted status report."""
        s = self.status()
        print("\n" + "=" * 60)
        print("AAM Training Status")
        print("=" * 60)
        print(f"  Batch:                {s['batch_number']}")
        print(f"  Nodes:                {s['total_nodes']}")
        print(f"  Compositions:         {s['total_compositions']}")
        print(f"  Edges:                {s['total_edges']}")
        print(f"  Gaps (unresolved):    {s['unresolved_gaps']} / {s['total_gaps']}")
        print(f"  Patterns (stable):    {s['stable_patterns']} / {s['total_patterns']}")
        print(f"  Patterns (grounded):  {s['grounded_patterns']} / {s['total_patterns']}")
        print(f"  Avg Confidence:       {s['average_confidence']:.3f}")
        print()
        print("  By Lifecycle:")
        for state, count in s["composition_by_lifecycle"].items():
            if count > 0:
                print(f"    {state}: {count}")
        print("  By Epistemic:")
        for state, count in s["composition_by_epistemic"].items():
            if count > 0:
                print(f"    {state}: {count}")
        print("  By Type:")
        for ctype, count in s["composition_by_type"].items():
            if count > 0:
                print(f"    {ctype}: {count}")
        print("=" * 60)

    def print_compositions(self, limit: int = 10):
        """Print recent compositions in detail."""
        comps = sorted(self.compositions.values(), key=lambda c: c.created_at, reverse=True)[:limit]
        for comp in comps:
            members_str = ", ".join(f"{m.role}={m.label}" for m in comp.members)
            print(f"  [{comp.lifecycle}/{comp.epistemic}] {comp.id}: {members_str} (conf={comp.confidence:.2f})")

    def print_patterns(self, min_count: int = 2):
        """Print discovered patterns."""
        sorted_patterns = sorted(self.patterns.values(), key=lambda p: p.observation_count, reverse=True)
        for p in sorted_patterns:
            if p.observation_count >= min_count:
                print(f"  [{p.lifecycle}/{p.epistemic}] {p.predicate} + {p.role} = {p.filler} (×{p.observation_count})")
