"""
Layer 3 — Deductive Coder Layer

Extends the Layer 2 CoderLayer with cross-layer reasoning capabilities
that use RSVS compositional semantics for deeper structural analysis.

Analogi: Layer 2 CoderLayer = Jin Soun bisa membaca manual teknik
bela diri dan mengerti kelemahannya dari struktur.
Layer 3 DeductiveCoderLayer = Jin Soun juga menghubungkan teknik
dari manual BERBEDA — cross-layer analysis yang menggunakan
compositional semantics untuk analisis yang lebih dalam.

Layer 3 additions over Layer 2:
  - DeductiveCoderLayer: extends CoderLayer with analyze_with_rsvs()
    which creates a full RSVS-represented code graph using
    compositional semantics (compose, structural_similarity,
    substitution_analysis, senses) for deeper structural analysis
    than the basic analyze_code() method.

All base functionality (CoderLayer, CodeElement, CodeAnalysisResult,
parsing functions, language detection, constants, etc.) is imported
from layer2.coder_layer — no duplication.
"""

from __future__ import annotations

import logging
from typing import Optional

# P1-1: Import from Layer 2 instead of duplicating
# P1-2: Cross-package imports use absolute style (layer2 is a sibling package,
# not a subpackage, so relative imports like ..layer2 don't work in this layout)
from layer2.coder_layer import (
    CoderLayer,
    CodeElement,
    CodeAnalysisResult,
    CODE_SOURCE_TRUST,
    DEFAULT_EXTENSIONS,
    ALL_SUPPORTED_EXTENSIONS,
    parse_python_code,
    _parse_code_regex,
    detect_language,
)
from layer2.bridge import RsvsBridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeductiveCoderLayer — Layer 3 extension
# ---------------------------------------------------------------------------

class DeductiveCoderLayer(CoderLayer):
    """Deductive Coder Layer — extends CoderLayer with RSVS compositional semantics.

    Adds analyze_with_rsvs() which creates a full RSVS-represented code
    graph using compositional semantics, enabling deeper structural
    analysis than the basic analyze_code() method.

    Uses:
    - bridge.compose() for code structure representation
    - bridge.structural_similarity() for code similarity analysis
    - bridge.substitution_analysis() for refactoring suggestions
    - bridge.senses() for multi-sense code element representation

    Analogi: Jin Soun tidak hanya membaca satu manual — dia
    menghubungkan teknik dari berbagai manual, membandingkan
    strukturnya, dan menemukan pola lintas-sumber. Layer 3
    = cross-layer reasoning dengan compositional semantics.

    All Layer 2 methods (ingest_code, ingest_file, ingest_directory,
    analyze_code, get_code_summary, etc.) are inherited unchanged.
    """

    # ==================================================================
    # RSVS compositional semantics integration
    # ==================================================================

    def analyze_with_rsvs(
        self,
        code: str,
        bridge: Optional[RsvsBridge] = None,
        language: str = "auto",
        source: str = "code_snippet",
    ) -> CodeAnalysisResult:
        """Analyze code using RSVS compositional semantics.

        Unlike analyze_code() which uses basic graph operations,
        this method creates a full RSVS-represented code graph using
        compositional semantics, enabling deeper structural analysis.

        Uses:
        - bridge.compose() for code structure representation
        - bridge.structural_similarity() for code similarity analysis
        - bridge.substitution_analysis() for refactoring suggestions
        - bridge.senses() for multi-sense code element representation

        Flow:
        1. Parse code into structural elements
        2. Represent each element as an RSVS node with composition references
        3. Create compositional nodes for class→method relationships
        4. Use structural_similarity for code comparison
        5. Use substitution_analysis for refactoring suggestions
        6. Detect anomalies using appraise() on code statements

        Args:
            code: Source code string to analyze.
            bridge: Optional override bridge (uses instance bridge if None).
            language: Programming language (default: "auto").
            source: Source identifier for the code.

        Returns:
            A CodeAnalysisResult with RSVS-enhanced analysis.
        """
        b = bridge or self._bridge
        result = CodeAnalysisResult(query=f"RSVS analysis of {source}")
        evidence_nodes: list[str] = []
        confidence = 0.3

        # Detect language if auto
        if language == "auto":
            language = detect_language(code, filename=source)

        # Parse code into structural elements
        if language == "python":
            elements = parse_python_code(code, source=source)
        else:
            elements = _parse_code_regex(code, source=source, language=language)

        # ---- Step 1: REPRESENT — Create RSVS nodes with composition refs ----
        # Each code element becomes a node; parent-child relationships
        # become composition references.
        compose_results: list[dict] = []
        element_node_map: dict[str, int] = {}  # element_key → node_id

        for element in elements:
            element_key = f"{element.kind}:{element.name}"
            if element.parent:
                element_key = f"{element.parent}.{element_key}"

            if b is not None and b.is_available:
                # Build composition references for this element
                comp_refs: list[tuple[str, int]] = []

                # Children become composition references
                for child_name in element.children[:10]:
                    child_key = f"method:{child_name}"
                    if child_key in element_node_map:
                        comp_refs.append((child_name, 0))

                # Parent reference
                if element.parent and element.parent in element_node_map:
                    comp_refs.append((element.parent, 0))

                # Ingest the element text first
                ingest_text = element.to_ingest_text()
                try:
                    b.ingest(ingest_text)
                except Exception:
                    pass

                # Then create a compositional node
                try:
                    node_id = b.compose(element_key, comp_refs) if comp_refs else None
                    if node_id is not None:
                        element_node_map[element_key] = node_id
                        compose_results.append({
                            "element": element_key,
                            "node_id": node_id,
                            "compositions": len(comp_refs),
                        })
                        evidence_nodes.append(element_key)
                        confidence = max(confidence, 0.5)
                except Exception as exc:
                    logger.debug("compose() failed for '%s': %s", element_key, exc)

                # Also get senses for this element
                try:
                    senses = b.senses(element.name)
                    if senses and isinstance(senses, list):
                        for sense in senses:
                            if isinstance(sense, dict):
                                gs = sense.get("grounding_score", 0.0)
                                if gs > 0.5:
                                    confidence = max(confidence, 0.6)
                except Exception:
                    pass

        # Record elements found
        for element in elements:
            result.elements_found.append(element.to_dict())

        # ---- Step 2: SIMILARITY — Use structural_similarity for comparison ----
        # Compare pairs of code elements via RSVS structural_similarity
        seen_pairs: set[tuple[str, str]] = set()
        element_names = [e.name for e in elements if e.name]
        for i, name_a in enumerate(element_names[:20]):
            for name_b in element_names[i + 1:20]:
                pair = tuple(sorted([name_a, name_b]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                if b is not None and b.is_available:
                    try:
                        sim = b.structural_similarity(name_a, name_b)
                        if sim and isinstance(sim, dict):
                            sim_value = sim.get("structural_similarity", 0.0)
                            if sim_value > 0.05:
                                result.similar_code.append({
                                    "node_a": name_a,
                                    "node_b": name_b,
                                    "similarity": sim_value,
                                    "shared": sim.get("shared", []),
                                })
                                evidence_nodes.extend([name_a, name_b])
                                confidence = max(confidence, 0.65)
                    except Exception:
                        pass

        # ---- Step 3: SUBSTITUTION — Refactoring suggestions ----
        # Use substitution_analysis to identify potential refactorings
        for i, name_a in enumerate(element_names[:10]):
            for name_b in element_names[i + 1:10]:
                if b is not None and b.is_available:
                    try:
                        sub_result = b.substitution_analysis(name_a, name_b)
                        if sub_result and isinstance(sub_result, dict):
                            substitutions = sub_result.get("substitutions", [])
                            if substitutions:
                                for sub_a, sub_b in substitutions[:3]:
                                    result.suggestions.append(
                                        f"Consider refactoring: '{sub_a}' in '{name_a}' "
                                        f"could be replaced with '{sub_b}' from '{name_b}'"
                                    )
                                    confidence = max(confidence, 0.7)
                    except Exception:
                        pass

        # ---- Step 4: ANOMALY — Detect code anomalies ----
        for element in elements[:5]:
            if element.kind in ("function", "method") and b is not None and b.is_available:
                statement = f"{element.name} is a complete and correct implementation"
                try:
                    appraise_result = b.appraise(statement)
                    if isinstance(appraise_result, dict):
                        verdict = appraise_result.get("verdict", "neutral")
                        if verdict == "disagree" or appraise_result.get("disagree_pct", 0) > 0.3:
                            result.anomalies.append({
                                "type": "code_anomaly",
                                "concept": element.name,
                                "verdict": verdict,
                                "disagree_pct": appraise_result.get("disagree_pct", 0.0),
                                "description": (
                                    f"Code element '{element.name}' may have issues: "
                                    f"appraise verdict={verdict}"
                                ),
                            })
                            evidence_nodes.append(element.name)
                            confidence = max(confidence, 0.7)
                except Exception:
                    pass

        # ---- Step 5: PATTERNS — Identify code patterns ----
        if result.similar_code:
            result.patterns.append({
                "type": "structural_similarity",
                "description": (
                    f"Found {len(result.similar_code)} structurally similar "
                    f"code pairs via RSVS compositional semantics"
                ),
                "pair_count": len(result.similar_code),
            })
            confidence = max(confidence, 0.7)

        if compose_results:
            result.patterns.append({
                "type": "compositional_representation",
                "description": (
                    f"Created {len(compose_results)} compositional nodes "
                    f"representing code structure in RSVS graph"
                ),
                "composition_count": len(compose_results),
            })
            confidence = max(confidence, 0.75)

        # ---- Step 6: OUTPUT — Finalize ----
        result.evidence_nodes = list(dict.fromkeys(evidence_nodes))
        result.confidence = min(0.95, confidence)

        logger.info(
            "DeductiveCoderLayer.analyze_with_rsvs(): source='%s', elements=%d, "
            "similar=%d, anomalies=%d, patterns=%d, confidence=%.3f",
            source, len(elements), len(result.similar_code),
            len(result.anomalies), len(result.patterns), result.confidence,
        )

        return result
