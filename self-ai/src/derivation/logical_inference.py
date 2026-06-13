# @WHO:   self-ai/src/derivation/logical_inference.py
# @WHAT:  Logical Inference Engine — property inheritance, contradiction propagation, universal instantiation
# @PART:  self-ai/derivation
# @ENTRY: LogicalInferenceEngine.infer(), LogicalInferenceEngine.derive_properties()

"""Logical Inference Engine — Idea 3 from deductive thinking.

Core principle: Knowledge should be DERIVED, not hardcoded.

Instead of storing every fact explicitly, the system should be able
to DERIVE new knowledge from existing knowledge using logical rules:

  1. Property Inheritance: If X IS_A Y and Y HAS property P, then X HAS P
     Example: "kucing IS_A hewan" + "hewan HAS bernapas" → "kucing HAS bernapas"

  2. Contradiction Propagation: If A contradicts B and B implies C,
     then A is inconsistent with C.
     Example: "rajin" contradicts "malas" + "malas" implies "tidak belajar"
     → "rajin" is inconsistent with "tidak belajar"

  3. Universal Instantiation: If ALL X are Y and Z is X, then Z is Y
     Example: "semua hewan bernapas" + "kucing IS_A hewan" → "kucing bernapas"

  4. Transitive Property: If A > B and B > C, then A > C
     (Applied to quantitative and qualitative hierarchies)

  5. Compositional Concepts: If X IS_A A and Y IS_A B,
     and A relates to B via R, then X may relate to Y via R.
     Example: "pisang IS_A buah" + "apel IS_A buah"
     → pisang and apel share properties of buah

This engine REPLACES hardcoded concept mappings with DERIVED knowledge.
When the system learns "kucing IS_A hewan", it automatically derives
all properties of "hewan" for "kucing" — no teaching required.

v18: First implementation.
"""

import re
import logging
from typing import Optional, Dict, Any, List, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class InferenceRule:
    """A logical inference rule that can be applied to derive new knowledge."""

    def __init__(self, name: str, rule_type: str, premises: List[str],
                 conclusion_template: str, confidence_modifier: float = 0.9):
        """
        Args:
            name: Human-readable name of the rule
            rule_type: Type of inference (inheritance, transitive, etc.)
            premises: List of premise patterns (each is a relation type)
            conclusion_template: Template for the derived conclusion
            confidence_modifier: How much to reduce confidence (0-1)
        """
        self.name = name
        self.rule_type = rule_type
        self.premises = premises
        self.conclusion_template = conclusion_template
        self.confidence_modifier = confidence_modifier


class LogicalInferenceEngine:
    """Derive new knowledge from existing knowledge using logical rules.

    This engine operates on the axiom store — it queries existing axioms
    and derives NEW axioms that weren't explicitly stored.

    Key principle: Every derivation must include a PROOF TRACE showing
    which axioms and rules were used. This makes all derived knowledge
    AUDITABLE — the system can always explain WHY it believes something.
    """

    # Standard relation types used in inferences
    IS_A = 'IS_A'
    HAS_PROPERTY = 'HAS'
    INSTANCE_OF = 'INSTANCE_OF'
    RELATES_TO = 'RELATES_TO'
    CONTRADICTS = 'CONTRADICTS'
    IMPLIES = 'IMPLIES'

    def __init__(self, self_core=None):
        self.self_core = self_core
        self._derived_cache = {}  # Cache derived inferences
        self._inference_log = []  # Track all inferences for auditing

    def infer(self, subject: str, relation: str = None,
              max_depth: int = 3) -> List[dict]:
        """Infer all properties of a subject through logical reasoning.

        This is the main entry point. Given a subject, it derives
        ALL properties that can be logically inferred.

        Args:
            subject: The entity to reason about
            relation: Optional specific relation to infer (if None, infer all)
            max_depth: Maximum inference chain depth (prevents infinite loops)

        Returns:
            List of derived facts, each with:
                - subject, relation, object
                - confidence
                - proof_trace (chain of axioms and rules used)
        """
        results = []
        visited = set()  # Prevent cycles

        self._infer_recursive(
            subject=subject,
            target_relation=relation,
            depth=0,
            max_depth=max_depth,
            visited=visited,
            results=results,
            proof_chain=[]
        )

        return results

    def _infer_recursive(self, subject: str, target_relation: str,
                         depth: int, max_depth: int,
                         visited: set, results: list, proof_chain: list):
        """Recursive inference with cycle detection and depth limiting."""
        if depth > max_depth:
            return

        # Cycle detection
        state_key = f"{subject}:{target_relation}:{depth}"
        if state_key in visited:
            return
        visited.add(state_key)

        # ── Rule 1: Property Inheritance ──
        # If X IS_A Y and Y HAS P, then X HAS P
        inherited = self._apply_property_inheritance(subject, proof_chain)
        for derived in inherited:
            if self._is_new_derivation(derived, results):
                results.append(derived)
                # Recurse on the derived object to find more properties
                self._infer_recursive(
                    subject=derived['object'],
                    target_relation=target_relation,
                    depth=depth + 1,
                    max_depth=max_depth,
                    visited=visited,
                    results=results,
                    proof_chain=proof_chain + [derived['proof_trace']]
                )

        # ── Rule 2: Transitive IS_A ──
        # If X IS_A Y and Y IS_A Z, then X IS_A Z
        transitive = self._apply_transitive_is_a(subject, proof_chain)
        for derived in transitive:
            if self._is_new_derivation(derived, results):
                results.append(derived)
                self._infer_recursive(
                    subject=derived['object'],
                    target_relation=target_relation,
                    depth=depth + 1,
                    max_depth=max_depth,
                    visited=visited,
                    results=results,
                    proof_chain=proof_chain + [derived['proof_trace']]
                )

        # ── Rule 3: Contradiction Propagation ──
        # If A contradicts B and B implies C, then A is inconsistent with C
        contradictions = self._apply_contradiction_propagation(subject, proof_chain)
        for derived in contradictions:
            if self._is_new_derivation(derived, results):
                results.append(derived)

        # ── Rule 4: Universal Instantiation ──
        # If ALL X are Y and Z is X, then Z is Y
        universal = self._apply_universal_instantiation(subject, proof_chain)
        for derived in universal:
            if self._is_new_derivation(derived, results):
                results.append(derived)

        # ── Rule 5: Compositional Concepts ──
        # If X IS_A A and Y IS_A A, then X and Y share A's properties
        compositional = self._apply_compositional_concepts(subject, proof_chain)
        for derived in compositional:
            if self._is_new_derivation(derived, results):
                results.append(derived)

    def _is_new_derivation(self, derived: dict, existing: list) -> bool:
        """Check if this derivation is truly new (not already derived)."""
        key = f"{derived.get('subject', '')}:{derived.get('relation', '')}:{derived.get('object', '')}"
        for e in existing:
            e_key = f"{e.get('subject', '')}:{e.get('relation', '')}:{e.get('object', '')}"
            if e_key == key:
                return False
        return True

    # ═══════════════ INFERENCE RULES ═══════════════

    def _apply_property_inheritance(self, subject: str,
                                     proof_chain: list) -> List[dict]:
        """Rule 1: If X IS_A Y and Y HAS P, then X HAS P.

        This is the most powerful rule — it allows the system to
        automatically inherit properties from parent categories.

        Example:
          "kucing IS_A hewan" + "hewan HAS bernapas"
          → "kucing HAS bernapas" (derived, not taught)
        """
        results = []
        axioms = self._get_axioms()

        # Find all IS_A parents of this subject
        is_a_parents = []
        for axiom_id, axiom in axioms.items():
            if self._relation_matches(axiom, self.IS_A):
                if self._subject_matches(axiom, subject):
                    parent = axiom.get('object', '').lower()
                    is_a_parents.append((parent, axiom_id, axiom.get('confidence', 0.5)))

        # For each parent, find their properties
        for parent, parent_axiom_id, parent_conf in is_a_parents:
            for axiom_id, axiom in axioms.items():
                if self._subject_matches(axiom, parent):
                    relation = axiom.get('predicate', '').upper()
                    if relation not in (self.IS_A, self.INSTANCE_OF):
                        # Derive that the subject also has this property
                        obj = axiom.get('object', '')
                        derived_conf = parent_conf * axiom.get('confidence', 0.5) * 0.9

                        result = {
                            'subject': subject,
                            'relation': relation,
                            'object': obj,
                            'confidence': derived_conf,
                            'source': 'inherited',
                            'proof_trace': {
                                'rule': 'property_inheritance',
                                'premises': [
                                    f"{subject} IS_A {parent} (axiom {parent_axiom_id})",
                                    f"{parent} {relation} {obj} (axiom {axiom_id})",
                                ],
                                'conclusion': f"{subject} {relation} {obj}",
                            }
                        }
                        results.append(result)
                        self._log_inference(result)

        return results

    def _apply_transitive_is_a(self, subject: str,
                                proof_chain: list) -> List[dict]:
        """Rule 2: If X IS_A Y and Y IS_A Z, then X IS_A Z.

        This allows multi-level category hierarchies.

        Example:
          "kucing IS_A mamalia" + "mamalia IS_A hewan"
          → "kucing IS_A hewan" (derived, not taught)
        """
        results = []
        axioms = self._get_axioms()

        # Find direct IS_A parents
        direct_parents = []
        for axiom_id, axiom in axioms.items():
            if self._relation_matches(axiom, self.IS_A):
                if self._subject_matches(axiom, subject):
                    direct_parents.append((axiom.get('object', '').lower(), axiom_id, axiom.get('confidence', 0.5)))

        # For each parent, find THEIR IS_A parents (grandparents)
        for parent, parent_axiom_id, parent_conf in direct_parents:
            for axiom_id, axiom in axioms.items():
                if self._relation_matches(axiom, self.IS_A):
                    if self._subject_matches(axiom, parent):
                        grandparent = axiom.get('object', '').lower()
                        if grandparent != subject and grandparent != parent:  # Avoid trivial cycles
                            derived_conf = parent_conf * axiom.get('confidence', 0.5) * 0.85

                            result = {
                                'subject': subject,
                                'relation': self.IS_A,
                                'object': grandparent,
                                'confidence': derived_conf,
                                'source': 'derived_transitive',
                                'proof_trace': {
                                    'rule': 'transitive_is_a',
                                    'premises': [
                                        f"{subject} IS_A {parent} (axiom {parent_axiom_id})",
                                        f"{parent} IS_A {grandparent} (axiom {axiom_id})",
                                    ],
                                    'conclusion': f"{subject} IS_A {grandparent}",
                                }
                            }
                            results.append(result)
                            self._log_inference(result)

        return results

    def _apply_contradiction_propagation(self, subject: str,
                                          proof_chain: list) -> List[dict]:
        """Rule 3: If A contradicts B and B implies C, then A is inconsistent with C.

        This allows the system to detect INDIRECT contradictions.

        Example:
          "rajin" CONTRADICTS "malas" + "malas" IMPLIES "tidak belajar"
          → "rajin" is inconsistent with "tidak belajar"
        """
        results = []
        axioms = self._get_axioms()

        # Find antonym/contradiction relationships
        antonym_map = self._get_antonym_map()

        # Find all properties of subject
        subject_properties = []
        for axiom_id, axiom in axioms.items():
            if self._subject_matches(axiom, subject):
                relation = axiom.get('predicate', '').upper()
                obj = axiom.get('object', '')
                if relation not in (self.IS_A, self.INSTANCE_OF):
                    subject_properties.append((obj, axiom_id, relation))

        # For each property, check if its antonym exists elsewhere
        for prop, prop_axiom_id, relation in subject_properties:
            antonyms = antonym_map.get(prop.lower(), [])
            for ant in antonyms:
                # Check if any other axiom mentions the antonym for this subject
                for other_id, other_axiom in axioms.items():
                    other_obj = other_axiom.get('object', '').lower()
                    if other_obj == ant.lower() and self._subject_matches(other_axiom, subject):
                        # Found a contradiction through propagation
                        result = {
                            'subject': subject,
                            'relation': 'CONTRADICTS',
                            'object': ant,
                            'confidence': 0.8,
                            'source': 'derived_contradiction',
                            'proof_trace': {
                                'rule': 'contradiction_propagation',
                                'premises': [
                                    f"{subject} {relation} {prop} (axiom {prop_axiom_id})",
                                    f"'{prop}' contradicts '{ant}' (antonym_map)",
                                    f"{subject} {other_axiom.get('predicate', '')} {ant} (axiom {other_id})",
                                ],
                                'conclusion': f"{subject} has contradictory properties: {prop} vs {ant}",
                            }
                        }
                        results.append(result)
                        self._log_inference(result)

        return results

    def _apply_universal_instantiation(self, subject: str,
                                        proof_chain: list) -> List[dict]:
        """Rule 4: If ALL X are Y and Z is X, then Z is Y.

        This handles universal statements ("semua", "setiap", "seluruh").

        Example:
          "semua hewan bernapas" + "kucing IS_A hewan"
          → "kucing bernapas" (derived via universal instantiation)
        """
        results = []
        axioms = self._get_axioms()

        # Find IS_A parents
        is_a_parents = []
        for axiom_id, axiom in axioms.items():
            if self._relation_matches(axiom, self.IS_A):
                if self._subject_matches(axiom, subject):
                    is_a_parents.append((axiom.get('object', '').lower(), axiom_id, axiom.get('confidence', 0.5)))

        # Find universal statements (axioms with "semua"/"setiap" markers)
        universal_axioms = []
        for axiom_id, axiom in axioms.items():
            subj = axiom.get('subject', '').lower()
            if any(universal in subj for universal in ['semua', 'setiap', 'seluruh', 'segala', 'tiap']):
                universal_axioms.append((axiom_id, axiom))

        # Instantiate universal statements for this subject
        for parent, parent_axiom_id, parent_conf in is_a_parents:
            for uax_id, uaxiom in universal_axioms:
                u_subject = uaxiom.get('subject', '').lower()
                # Check if the universal's category matches our parent
                # "semua hewan" → category = "hewan"
                for universal_prefix in ['semua ', 'setiap ', 'seluruh ', 'segala ', 'tiap ']:
                    if u_subject.startswith(universal_prefix):
                        category = u_subject[len(universal_prefix):]
                        if category == parent:
                            # Universal applies to our subject!
                            u_relation = uaxiom.get('predicate', '').upper()
                            u_object = uaxiom.get('object', '')
                            derived_conf = parent_conf * uaxiom.get('confidence', 0.9) * 0.85

                            result = {
                                'subject': subject,
                                'relation': u_relation,
                                'object': u_object,
                                'confidence': derived_conf,
                                'source': 'derived_universal',
                                'proof_trace': {
                                    'rule': 'universal_instantiation',
                                    'premises': [
                                        f"{u_subject} {u_relation} {u_object} (axiom {uax_id})",
                                        f"{subject} IS_A {parent} (axiom {parent_axiom_id})",
                                    ],
                                    'conclusion': f"{subject} {u_relation} {u_object}",
                                }
                            }
                            results.append(result)
                            self._log_inference(result)

        return results

    def _apply_compositional_concepts(self, subject: str,
                                       proof_chain: list) -> List[dict]:
        """Rule 5: If X IS_A A and Y IS_A A, then X and Y share A's properties.

        This handles analogical reasoning based on shared categories.

        Example:
          "pisang IS_A buah" + "apel IS_A buah" + "apel HAS vitamin C"
          → "pisang MAY_HAVE vitamin C" (weak derivation via shared category)

        This is the weakest form of inference — it generates HYPOTHESES
        rather than certain conclusions. Confidence is significantly reduced.
        """
        results = []
        axioms = self._get_axioms()

        # Find IS_A parents
        is_a_parents = []
        for axiom_id, axiom in axioms.items():
            if self._relation_matches(axiom, self.IS_A):
                if self._subject_matches(axiom, subject):
                    is_a_parents.append((axiom.get('object', '').lower(), axiom_id, axiom.get('confidence', 0.5)))

        # For each parent, find SIBLINGS (other things that are also IS_A that parent)
        for parent, parent_axiom_id, parent_conf in is_a_parents:
            siblings = []
            for axiom_id, axiom in axioms.items():
                if self._relation_matches(axiom, self.IS_A):
                    if self._subject_matches(axiom, parent) and not self._subject_matches(axiom, subject):
                        sibling = axiom.get('subject', '').lower()
                        siblings.append((sibling, axiom_id, axiom.get('confidence', 0.5)))

            # For each sibling, check their properties
            for sibling, sibling_axiom_id, sibling_conf in siblings:
                for axiom_id, axiom in axioms.items():
                    if self._subject_matches(axiom, sibling):
                        relation = axiom.get('predicate', '').upper()
                        obj = axiom.get('object', '')
                        if relation not in (self.IS_A, self.INSTANCE_OF):
                            # Very low confidence — this is a hypothesis
                            derived_conf = parent_conf * sibling_conf * axiom.get('confidence', 0.5) * 0.3

                            if derived_conf >= 0.1:  # Only include if above noise threshold
                                result = {
                                    'subject': subject,
                                    'relation': f'MAY_{relation}',
                                    'object': obj,
                                    'confidence': derived_conf,
                                    'source': 'derived_compositional',
                                    'proof_trace': {
                                        'rule': 'compositional_concepts',
                                        'premises': [
                                            f"{subject} IS_A {parent} (axiom {parent_axiom_id})",
                                            f"{sibling} IS_A {parent} (axiom {sibling_axiom_id})",
                                            f"{sibling} {relation} {obj} (axiom {axiom_id})",
                                        ],
                                        'conclusion': f"{subject} MAY_{relation} {obj} (hypothesis from sibling {sibling})",
                                    }
                                }
                                results.append(result)
                                self._log_inference(result)

        return results

    # ═══════════════ HIGH-LEVEL API ═══════════════

    def derive_properties(self, subject: str, property_name: str = None) -> dict:
        """Derive all known properties of a subject.

        This is the most commonly used API — given a word or concept,
        derive everything the system can infer about it.

        Args:
            subject: The concept to reason about
            property_name: Optional specific property to derive

        Returns:
            dict with:
                - direct_properties: Properties directly stated in axioms
                - inherited_properties: Properties derived via IS_A chain
                - hypotheses: Weakly derived properties (compositional)
                - contradictions: Any detected contradictions
        """
        all_inferences = self.infer(subject, property_name)

        direct = []
        inherited = []
        hypotheses = []
        contradictions = []

        for inf in all_inferences:
            source = inf.get('source', '')
            if source == 'inherited':
                inherited.append(inf)
            elif source == 'derived_compositional':
                hypotheses.append(inf)
            elif source == 'derived_contradiction':
                contradictions.append(inf)
            else:
                direct.append(inf)

        # Also include direct axiom properties
        axioms = self._get_axioms()
        for axiom_id, axiom in axioms.items():
            if self._subject_matches(axiom, subject):
                relation = axiom.get('predicate', '').upper()
                obj = axiom.get('object', '')
                if relation not in (self.IS_A, self.INSTANCE_OF):
                    already = any(d['object'] == obj and d['relation'] == relation
                                 for d in direct)
                    if not already:
                        direct.append({
                            'subject': subject,
                            'relation': relation,
                            'object': obj,
                            'confidence': axiom.get('confidence', 0.5),
                            'source': 'direct_axiom',
                        })

        return {
            'subject': subject,
            'direct_properties': direct,
            'inherited_properties': inherited,
            'hypotheses': hypotheses,
            'contradictions': contradictions,
        }

    def check_consistency(self, subject: str) -> List[dict]:
        """Check if all derived properties of a subject are consistent.

        Returns a list of any contradictions found in the derived knowledge.
        """
        all_inferences = self.infer(subject)
        contradictions = []

        # Group by relation type
        by_relation = defaultdict(list)
        for inf in all_inferences:
            rel = inf.get('relation', '')
            by_relation[rel].append(inf)

        # Check for antonym contradictions within same relation
        antonym_map = self._get_antonym_map()
        for rel, infs in by_relation.items():
            objects = [inf.get('object', '').lower() for inf in infs]
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects):
                    if i < j:
                        antonyms_of_1 = antonym_map.get(obj1, [])
                        if obj2 in antonyms_of_1:
                            contradictions.append({
                                'type': 'antonym_contradiction',
                                'relation': rel,
                                'conflicting_values': [obj1, obj2],
                                'confidence': min(infs[i].get('confidence', 0.5),
                                                 infs[j].get('confidence', 0.5)),
                            })

        return contradictions

    # ═══════════════ HELPERS ═══════════════

    def _get_axioms(self) -> dict:
        """Get all axioms from the knowledge store."""
        if self.self_core and hasattr(self.self_core, 'axioms'):
            return self.self_core.axioms
        return {}

    def _get_antonym_map(self) -> dict:
        """Get the antonym map from concept clusters."""
        try:
            from derivation.concepts import CONCEPT_CLUSTERS
            return CONCEPT_CLUSTERS.get('antonym_map', {})
        except ImportError:
            return {}

    def _relation_matches(self, axiom: dict, relation: str) -> bool:
        """Check if an axiom's predicate matches a relation type."""
        pred = axiom.get('predicate', '').upper()
        return pred == relation

    def _subject_matches(self, axiom: dict, subject: str) -> bool:
        """Check if an axiom's subject matches (case-insensitive, substring)."""
        ax_subject = axiom.get('subject', '').lower()
        return subject.lower() in ax_subject or ax_subject in subject.lower()

    def _log_inference(self, inference: dict):
        """Log an inference for auditing."""
        self._inference_log.append({
            'subject': inference.get('subject', ''),
            'relation': inference.get('relation', ''),
            'object': inference.get('object', ''),
            'confidence': inference.get('confidence', 0.0),
            'source': inference.get('source', ''),
            'rule': inference.get('proof_trace', {}).get('rule', ''),
        })
        # Keep log manageable
        if len(self._inference_log) > 1000:
            self._inference_log = self._inference_log[-500:]

    def get_inference_stats(self) -> dict:
        """Get statistics about inferences made."""
        if not self._inference_log:
            return {'total': 0}

        by_rule = defaultdict(int)
        for entry in self._inference_log:
            by_rule[entry.get('rule', 'unknown')] += 1

        return {
            'total': len(self._inference_log),
            'by_rule': dict(by_rule),
        }
