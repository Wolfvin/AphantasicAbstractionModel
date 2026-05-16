"""
AAM Layer 2 — Policy Rule Compiler

Compiles conditional patterns from the RSVS graph into executable PolicyRules.

This is the bridge between:
- Layer 1 Rust: ExtractFrame extracts Antecedent/Consequent roles from
  conditional sentences (e.g., "jika penghasilan > 500 juta, tarif 30%")
- Layer 2 Python: PolicyEngine evaluates deterministic conditions

The compiler:
1. Scans the graph for compositions with Antecedent/Consequent roles
2. Extracts the condition and consequence text
3. Parses them into structured PolicyRule objects with eval-able conditions
4. Registers the rules with a PolicyEngine

This enables automatic rule generation from regulation text — no manual coding
required for each new regulation. Just ingest the text and compile.

Analogi: Ini adalah "penerjemah hukum" — membaca teks regulasi dan
mengubahnya menjadi aturan yang bisa dievaluasi secara deterministik.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .bridge import V12PipelineBridge, get_bridge
from .policy_engine import PolicyEngine, PolicyRule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Number extraction utilities
# ---------------------------------------------------------------------------

# Pattern to extract numbers from Indonesian text (handles "500 juta", "30 persen", etc.)
_NUMBER_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "Rp500 juta" → 500_000_000
    ("rp_value", re.compile(
        r"rp\s*([\d.,]+)\s*(juta|miliar|milyar|triliun)?",
        re.IGNORECASE
    )),
    # "500 juta" → 500_000_000
    ("raw_number_unit", re.compile(
        r"([\d.,]+)\s*(juta|miliar|milyar|triliun|ribu|ratus)",
        re.IGNORECASE
    )),
    # "30 persen" / "30%" → 0.30
    ("percentage", re.compile(
        r"([\d.,]+)\s*(persen|%)",
        re.IGNORECASE
    )),
    # Plain number fallback
    ("plain_number", re.compile(r"([\d.,]+)")),
]

# Multipliers for Indonesian number units
_UNIT_MULTIPLIERS: dict[str, float] = {
    "ribu": 1_000,
    "ratus": 100,
    "juta": 1_000_000,
    "miliar": 1_000_000_000,
    "milyar": 1_000_000_000,
    "triliun": 1_000_000_000_000,
}


def extract_number(text: str) -> Optional[float]:
    """Extract a numeric value from Indonesian text.

    Handles:
    - "Rp500.000.000" → 500000000
    - "500 juta" → 500000000
    - "30 persen" / "30%" → 0.30
    - "500.000.000" → 500000000
    - Plain numbers

    Returns None if no number can be extracted.
    """
    # Try percentage first (converts to decimal)
    m = _NUMBER_PATTERNS[2][1].search(text)  # percentage
    if m:
        num_str = m.group(1).replace(".", "").replace(",", ".")
        try:
            return float(num_str) / 100.0
        except ValueError:
            pass

    # Try Rp value
    m = _NUMBER_PATTERNS[0][1].search(text)  # rp_value
    if m:
        num_str = m.group(1).replace(".", "").replace(",", ".")
        unit = (m.group(2) or "").lower()
        try:
            value = float(num_str)
            value *= _UNIT_MULTIPLIERS.get(unit, 1)
            return value
        except ValueError:
            pass

    # Try raw number with unit
    m = _NUMBER_PATTERNS[1][1].search(text)  # raw_number_unit
    if m:
        num_str = m.group(1).replace(".", "").replace(",", ".")
        unit = m.group(2).lower()
        try:
            value = float(num_str)
            value *= _UNIT_MULTIPLIERS.get(unit, 1)
            return value
        except ValueError:
            pass

    # Plain number fallback
    m = _NUMBER_PATTERNS[3][1].search(text)  # plain_number
    if m:
        num_str = m.group(1).replace(".", "").replace(",", ".")
        try:
            return float(num_str)
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Condition parser — converts text to eval-able expressions
# ---------------------------------------------------------------------------

# Tax-related keyword patterns for condition generation
# Note: capture group is non-greedy and stops at common Indonesian keywords
_THRESHOLD_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # "penghasilan di atas X" → "value > X"
    (
        "income_threshold",
        re.compile(r"penghasilan\s+(?:di atas|lebih dari|>\s*)\s*([\d.,]+\s*(?:juta|miliar|milyar|triliun|ribu)?)", re.IGNORECASE),
        "value > {threshold}",
    ),
    # "PKP di atas X" → "value > X"
    (
        "pkp_threshold",
        re.compile(r"pkp\s+(?:di atas|lebih dari|>\s*)\s*([\d.,]+\s*(?:juta|miliar|milyar|triliun|ribu)?)", re.IGNORECASE),
        "value > {threshold}",
    ),
    # "di atas X" → "value > X"
    (
        "above_threshold",
        re.compile(r"di atas\s+([\d.,]+\s*(?:juta|miliar|milyar|triliun|ribu)?)", re.IGNORECASE),
        "value > {threshold}",
    ),
    # "di bawah X" → "value < X"
    (
        "below_threshold",
        re.compile(r"di bawah\s+([\d.,]+\s*(?:juta|miliar|milyar|triliun|ribu)?)", re.IGNORECASE),
        "value < {threshold}",
    ),
]

# Rate extraction patterns for consequence
_RATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # "tarif 30 persen" → rate=0.30
    (
        "tarif_rate",
        re.compile(r"tarif\s+([\d.,]+)\s*(?:persen|%)", re.IGNORECASE),
    ),
    # "dikenakan ... 30 persen" → rate=0.30
    (
        "dikenakan_rate",
        re.compile(r"dikenakan\s+.*?([\d.,]+)\s*(?:persen|%)", re.IGNORECASE),
    ),
]


def parse_condition(antecedent_text: str) -> tuple[str, dict[str, Any]]:
    """Parse an antecedent text into an eval-able condition expression.

    Args:
        antecedent_text: The "if" part of a conditional sentence
            (e.g., "penghasilan di atas 500 juta")

    Returns:
        A tuple of (condition_expression, parameters).
        The condition is a Python expression suitable for PolicyRule.evaluate().
        The parameters dict contains extracted numeric values.
    """
    parameters: dict[str, Any] = {}

    # Try threshold patterns first (most specific)
    for pattern_name, pattern, template in _THRESHOLD_PATTERNS:
        m = pattern.search(antecedent_text)
        if m:
            value_text = m.group(1).strip()
            threshold = extract_number(value_text)
            if threshold is not None:
                parameters["threshold"] = threshold
                condition = template.format(threshold=threshold)
                return condition, parameters

    # Fallback: extract the largest non-percentage number as threshold
    # This avoids mistakenly treating a rate (30%) as a threshold
    all_numbers: list[float] = []
    for pattern_name, pattern in [
        ("rp_value", _NUMBER_PATTERNS[0][1]),
        ("raw_number_unit", _NUMBER_PATTERNS[1][1]),
    ]:
        for m in pattern.finditer(antecedent_text):
            num_str = m.group(1).replace(".", "").replace(",", ".")
            unit = (m.group(2) or "").lower() if m.lastindex and m.lastindex >= 2 else ""
            try:
                value = float(num_str)
                value *= _UNIT_MULTIPLIERS.get(unit, 1)
                all_numbers.append(value)
            except ValueError:
                pass

    if all_numbers:
        # Use the largest number as the threshold (most likely the income/PKP threshold)
        threshold = max(all_numbers)
        parameters["threshold"] = threshold
        return f"value > {threshold}", parameters

    # Last resort: always-true condition (the rule exists but won't trigger)
    return "True", parameters


def parse_consequence(consequent_text: str) -> dict[str, Any]:
    """Parse a consequent text to extract parameters.

    Args:
        consequent_text: The "then" part of a conditional sentence
            (e.g., "dikenakan tarif 30 persen")

    Returns:
        A dict of extracted parameters (rate, etc.).
    """
    parameters: dict[str, Any] = {}

    for pattern_name, pattern in _RATE_PATTERNS:
        m = pattern.search(consequent_text)
        if m:
            rate_str = m.group(1).replace(",", ".")
            try:
                rate = float(rate_str) / 100.0
                parameters["rate"] = rate
            except ValueError:
                pass
            break

    return parameters


# ---------------------------------------------------------------------------
# PolicyRuleCompiler
# ---------------------------------------------------------------------------

@dataclass
class CompiledRule:
    """A rule compiled from the RSVS graph.

    Contains both the PolicyRule and the source composition metadata
    for traceability.
    """
    rule: PolicyRule
    source_composition_id: str
    antecedent_text: str
    consequent_text: str
    compilation_confidence: float = 0.0


class PolicyRuleCompiler:
    """Compiles conditional patterns from the RSVS graph into PolicyRules.

    Usage:
        compiler = PolicyRuleCompiler(bridge)
        compiled = compiler.compile_all()
        for cr in compiled:
            print(f"Rule {cr.rule.rule_id}: {cr.rule.condition}")
            engine.add_rule(cr.rule)

    The compiler scans the graph for compositions with Antecedent/Consequent
    roles (produced by ExtractFrame's conditional marker detection and
    ReasonFrame's ConditionConsequenceRule), then converts them into
    structured PolicyRule objects with eval-able condition expressions.
    """

    def __init__(self, bridge: Optional[V12PipelineBridge] = None) -> None:
        self._bridge = bridge or get_bridge()
        self._compiled: list[CompiledRule] = []
        self._rule_counter = 0

    def compile_all(self) -> list[CompiledRule]:
        """Scan the graph and compile all conditional patterns into PolicyRules.

        Returns a list of CompiledRule objects, each containing:
        - A PolicyRule ready for registration with PolicyEngine
        - Source composition ID for traceability
        - Original antecedent/consequent text
        - Compilation confidence (based on composition confidence)
        """
        self._compiled = []
        compositions = self._bridge.compositions()

        for comp in compositions:
            compiled = self._compile_composition(comp)
            if compiled is not None:
                self._compiled.append(compiled)

        logger.info(
            "PolicyRuleCompiler: compiled %d rules from %d compositions",
            len(self._compiled), len(compositions),
        )
        return self._compiled

    def _compile_composition(self, comp: dict) -> Optional[CompiledRule]:
        """Try to compile a single composition into a PolicyRule.

        Returns None if the composition doesn't have Antecedent/Consequent
        roles or if compilation fails.
        """
        members = comp.get("members", [])

        # Look for Antecedent and Consequent roles
        antecedent = None
        consequent = None
        for m in members:
            role_str = str(m.get("role", "")).lower()
            label = m.get("label", "")
            if "antecedent" in role_str:
                antecedent = label
            elif "consequent" in role_str:
                consequent = label

        if not antecedent or not consequent:
            return None

        return self._compile_from_text(
            antecedent_text=antecedent,
            consequent_text=consequent,
            composition_id=comp.get("id", "unknown"),
            confidence=comp.get("confidence", 0.5),
        )

    def _compile_from_text(
        self,
        antecedent_text: str,
        consequent_text: str,
        composition_id: str = "unknown",
        confidence: float = 0.5,
    ) -> Optional[CompiledRule]:
        """Compile an antecedent/consequent pair into a PolicyRule.

        This is the core compilation logic:
        1. Parse the antecedent into a condition expression
        2. Parse the consequent into parameters (rate, threshold, etc.)
        3. Handle Indonesian inverted conditionals ("consequence jika condition")
        4. Create a PolicyRule with a unique ID

        Args:
            antecedent_text: The condition text (e.g., "penghasilan di atas 500 juta")
            consequent_text: The consequence text (e.g., "dikenakan tarif 30 persen")
            composition_id: Source composition ID for traceability
            confidence: Compilation confidence from the source composition

        Returns:
            A CompiledRule, or None if compilation fails completely.
        """
        # Parse condition — try antecedent first
        condition, condition_params = parse_condition(antecedent_text)

        # Indonesian inverted conditional: if the antecedent has no numeric threshold
        # but the consequent does, the "jika" clause is actually the condition.
        # Example: "dikenakan pajak jika penghasilan di atas 500 juta"
        #   antecedent = "dikenakan pajak" (no number)
        #   consequent = "penghasilan di atas 500 juta dikenakan tarif 30 persen" (has number)
        if condition == "True" and not condition_params:
            # Try parsing the consequent as the condition instead
            alt_condition, alt_params = parse_condition(consequent_text)
            if alt_condition != "True" and alt_params:
                condition = alt_condition
                condition_params = alt_params
                # Swap: what we thought was antecedent is actually the consequence
                antecedent_text, consequent_text = consequent_text, antecedent_text

        # Parse consequence
        consequence_params = parse_consequence(consequent_text)

        # Also check antecedent for rate (in case the rate is in the "if" part)
        if "rate" not in consequence_params:
            ante_rate = parse_consequence(antecedent_text)
            if "rate" in ante_rate:
                consequence_params["rate"] = ante_rate["rate"]

        # Merge parameters
        parameters = {**condition_params, **consequence_params}

        # Generate rule ID
        self._rule_counter += 1
        rule_id = f"COMPILED_{self._rule_counter:03d}"

        # Determine category from text
        category = self._infer_category(antecedent_text + " " + consequent_text)

        # Create the PolicyRule
        rule = PolicyRule(
            rule_id=rule_id,
            description=f"Jika {antecedent_text}, maka {consequent_text}",
            condition=condition,
            severity="warning",
            category=category,
            parameters=parameters,
        )

        return CompiledRule(
            rule=rule,
            source_composition_id=composition_id,
            antecedent_text=antecedent_text,
            consequent_text=consequent_text,
            compilation_confidence=confidence,
        )

    def register_with_engine(self, engine: PolicyEngine) -> int:
        """Compile rules and register them with a PolicyEngine.

        Args:
            engine: The PolicyEngine to register rules with.

        Returns:
            The number of rules successfully registered.
        """
        compiled = self.compile_all()
        count = 0
        for cr in compiled:
            try:
                engine.add_rule(cr.rule)
                count += 1
            except Exception as exc:
                logger.warning("Failed to register rule %s: %s", cr.rule.rule_id, exc)

        return count

    @staticmethod
    def _infer_category(text: str) -> str:
        """Infer the regulatory category from text content."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["pajak", "pph", "tarif", "wajib pajak", "npwp", "spt", "pkp", "ppn"]):
            return "tax"
        if any(kw in text_lower for kw in ["imigrasi", "visa", "paspor", "izin tinggal"]):
            return "immigration"
        if any(kw in text_lower for kw in ["perusahaan", "pt", "cv", "perseroan", "direksi"]):
            return "corporate"
        if any(kw in text_lower for kw in ["tenaga kerja", "upah", "gaji", "phk", "serikat"]):
            return "labor"

        return "regulation"
