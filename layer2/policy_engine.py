"""
AAM Layer 2 — Policy Engine (Base)

Rule-based compliance checking with deterministic evaluation.
Defines PolicyRules (conditions), checks compliance, and produces
PolicyViolations when rules are broken.

This is the Layer 2 base — providing core rule evaluation.
Layer 3's DeductivePolicyEngine extends this with RSVS PolicyMeta
(governance_score, status_flip_count) for trust-weighted checking.

Analogi: Layer 2 PolicyEngine = Jin Soun membuka buku hukum dan
mengecek "Apakah tindakan ini melanggar Pasal 21 UU PPh?"
Layer 3 = juga mengecek catatan pengawasan untuk verdict lebih nuanced.

Design decisions:
  - Rules are evaluated deterministically (no probabilistic logic)
  - Expressions use a restricted eval sandbox (_SAFE_EVAL_NAMES)
  - Indonesian tax rules included as default rule set
  - Violations are data objects, not exceptions
"""

from __future__ import annotations

import logging
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe eval sandbox
# ---------------------------------------------------------------------------

_SAFE_EVAL_NAMES: dict[str, Any] = {
    # Built-in functions (safe subset)
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "len": len,
    "sum": sum,
    "any": any,
    "all": all,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    # Math functions
    "math": math,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
    # Operators
    "__builtins__": {},
    # Boolean constants
    "True": True,
    "False": False,
    "None": None,
}
"""
Restricted namespace for safe eval() of rule expressions.
Only whitelisted functions and constants are available.
No access to __import__, open, exec, eval, or file operations.
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PolicyRule:
    """A compliance rule with a condition expression.

    Each rule has:
    - A unique ID for tracking
    - A natural language description
    - A condition expression (evaluated in safe sandbox)
    - A severity level ('error', 'warning', 'info')
    - A category for grouping (e.g. 'tax', 'regulation', 'corporate')
    - Parameters that the condition can reference

    The condition expression is a Python expression evaluated in the
    _SAFE_EVAL_NAMES sandbox with additional context variables:
    - 'value': The value being checked
    - 'params': The rule's parameters dict
    - 'threshold': Shortcut for params.get('threshold', 0)

    Example conditions:
        "value > threshold"
        "value < params.get('max_rate', 1.0)"
        "len(value) > 0 and value > 0"
    """

    rule_id: str
    description: str
    condition: str
    severity: str = "warning"
    category: str = "general"
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def evaluate(self, value: Any) -> bool:
        """Evaluate the rule condition against a value.

        Uses a restricted eval sandbox for safety.
        Returns False if the condition cannot be evaluated.

        Args:
            value: The value to check against this rule.

        Returns:
            True if the condition is satisfied (rule passed),
            False if the condition fails or cannot be evaluated.
        """
        if not self.enabled:
            return True  # Disabled rules always pass

        try:
            # Build safe evaluation context
            eval_context = dict(_SAFE_EVAL_NAMES)
            eval_context["value"] = value
            eval_context["params"] = self.parameters
            eval_context["threshold"] = self.parameters.get("threshold", 0)

            result = eval(self.condition, eval_context)
            return bool(result)
        except Exception as exc:
            logger.debug(
                "Rule '%s' evaluation failed: condition='%s', value=%s, error=%s",
                self.rule_id, self.condition, repr(value)[:50], exc,
            )
            return False  # Fail closed — if can't evaluate, rule doesn't pass

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "condition": self.condition,
            "severity": self.severity,
            "category": self.category,
            "parameters": self.parameters,
            "enabled": self.enabled,
        }


@dataclass
class PolicyViolation:
    """A record of a policy rule violation.

    Created when a rule's condition evaluates to False.
    Contains the rule that was violated, the value that
    triggered the violation, and suggested remediation.

    Attributes:
        rule: The PolicyRule that was violated.
        value: The value that triggered the violation.
        message: Human-readable description of the violation.
        remediation: Suggested fix for the violation.
    """

    rule: PolicyRule
    value: Any = None
    message: str = ""
    remediation: str = ""

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "rule_id": self.rule.rule_id,
            "rule_description": self.rule.description,
            "severity": self.rule.severity,
            "category": self.rule.category,
            "value": str(self.value)[:100] if self.value is not None else None,
            "message": self.message,
            "remediation": self.remediation,
        }


# ---------------------------------------------------------------------------
# Indonesian Tax Rules (Default Rule Set)
# ---------------------------------------------------------------------------

def _create_indonesian_tax_rules() -> list[PolicyRule]:
    """Create default Indonesian tax compliance rules.

    Based on UU PPh (Undang-Undang Pajak Penghasilan) and
    common compliance requirements for Indonesian entities.

    These rules are for demonstration and baseline checking.
    For production use, rules should be reviewed by tax professionals.
    """
    return [
        PolicyRule(
            rule_id="ID_TAX_PPH21_THRESHOLD",
            description="PPh 21 tarif progresif: penghasilan bruto harus di atas PTKP",
            condition="value > params.get('ptkp', 54000000)",
            severity="warning",
            category="tax_pph21",
            parameters={"ptkp": 54_000_000, "threshold": 54_000_000},
        ),
        PolicyRule(
            rule_id="ID_TAX_PPH21_RATE_CAP",
            description="PPh 21 tarif maksimal 35% untuk bracket tertinggi",
            condition="value <= params.get('max_rate', 0.35)",
            severity="error",
            category="tax_pph21",
            parameters={"max_rate": 0.35, "threshold": 0.35},
        ),
        PolicyRule(
            rule_id="ID_TAX_PPH23_RATE",
            description="PPh 23 tarif 2% untuk jasa dan 4% untuk sewa",
            condition="value >= 0 and value <= params.get('max_rate', 0.04)",
            severity="warning",
            category="tax_pph23",
            parameters={"max_rate": 0.04, "threshold": 0.04},
        ),
        PolicyRule(
            rule_id="ID_TAX_PPN_RATE",
            description="PPN tarif standar 11% (sejak 2022)",
            condition="value == params.get('standard_rate', 0.11)",
            severity="info",
            category="tax_ppn",
            parameters={"standard_rate": 0.11, "threshold": 0.11},
        ),
        PolicyRule(
            rule_id="ID_TAX_NPWP_REQUIRED",
            description="NPWP wajib untuk penghasilan di atas PTKP",
            condition="value != '' and value is not None",
            severity="error",
            category="tax_compliance",
            parameters={"threshold": 0},
        ),
        PolicyRule(
            rule_id="ID_TAX_FILING_DEADLINE",
            description="SPT Tahunan harus dilaporkan sebelum 31 Maret (OP) atau 30 April (Badan)",
            condition="value <= params.get('deadline_month', 4)",
            severity="warning",
            category="tax_compliance",
            parameters={"deadline_month": 4, "threshold": 4},
        ),
    ]


# ---------------------------------------------------------------------------
# PolicyEngine — Main class
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Rule-based compliance checking engine.

    Manages a set of PolicyRules and checks values against them.
    Produces PolicyViolations when rules are broken, and generates
    compliance reports.

    This is the Layer 2 base — providing deterministic rule evaluation.
    Layer 3's DeductivePolicyEngine extends this with RSVS PolicyMeta
    for trust-weighted, auditable compliance checking.

    Usage:
        engine = PolicyEngine()
        engine.load_tax_rules_indonesia()
        result = engine.check_compliance("some_entity", context={"rate": 0.15})
        if result["violations"]:
            for v in result["violations"]:
                print(f"VIOLATION: {v['message']}")
    """

    def __init__(self, bridge: Optional[RsvsBridge] = None) -> None:
        """Initialize the PolicyEngine.

        Args:
            bridge: Optional pre-built RsvsBridge. If None, creates one.
        """
        self._bridge = bridge or get_bridge()
        self._rules: dict[str, PolicyRule] = {}
        self._violation_history: list[PolicyViolation] = []

    # ==================================================================
    # Rule management
    # ==================================================================

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a compliance rule.

        Args:
            rule: The PolicyRule to add.
        """
        self._rules[rule.rule_id] = rule
        logger.debug("Added rule: %s (%s)", rule.rule_id, rule.description[:60])

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a compliance rule by ID.

        Args:
            rule_id: The rule ID to remove.

        Returns:
            True if the rule was found and removed.
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_rules(self, category: str = "") -> list[PolicyRule]:
        """Get all rules, optionally filtered by category.

        Args:
            category: Optional category filter.

        Returns:
            List of PolicyRule instances.
        """
        rules = list(self._rules.values())
        if category:
            rules = [r for r in rules if r.category == category]
        return rules

    def load_tax_rules_indonesia(self) -> int:
        """Load default Indonesian tax compliance rules.

        Returns:
            Number of rules loaded.
        """
        rules = _create_indonesian_tax_rules()
        for rule in rules:
            self.add_rule(rule)
        logger.info("Loaded %d Indonesian tax rules", len(rules))
        return len(rules)

    # ==================================================================
    # Compliance checking
    # ==================================================================

    def check_compliance(
        self,
        entity: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Check an entity against all active rules.

        Evaluates each enabled rule against the provided context.
        Context values are matched to rules by category — tax rules
        look for 'rate', 'income', etc. in context.

        Args:
            entity: The entity being checked (name, ID, label).
            context: Optional dict of context values for rule evaluation.
                Common keys: 'rate', 'income', 'npwp', 'month', etc.

        Returns:
            Dict with:
                - "entity": The entity checked
                - "rules_evaluated": Number of rules checked
                - "violations": List of violation dicts
                - "warnings": Count of warning-severity violations
                - "errors": Count of error-severity violations
                - "compliant": Boolean — no error-severity violations
        """
        ctx = context or {}
        violations: list[PolicyViolation] = []
        warnings_count = 0
        errors_count = 0

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            # Determine the value to check based on context
            value = self._extract_value_for_rule(rule, ctx)

            # Evaluate the rule
            try:
                passed = rule.evaluate(value)
            except Exception as exc:
                logger.debug("Rule '%s' failed to evaluate: %s", rule.rule_id, exc)
                passed = False

            if not passed:
                violation = PolicyViolation(
                    rule=rule,
                    value=value,
                    message=f"Rule '{rule.rule_id}' violated: {rule.description}",
                    remediation=self._suggest_remediation(rule, value),
                )
                violations.append(violation)
                self._violation_history.append(violation)

                if rule.severity == "error":
                    errors_count += 1
                elif rule.severity == "warning":
                    warnings_count += 1

        result = {
            "entity": entity,
            "rules_evaluated": sum(1 for r in self._rules.values() if r.enabled),
            "violations": [v.to_dict() for v in violations],
            "warnings": warnings_count,
            "errors": errors_count,
            "compliant": errors_count == 0,
        }

        logger.info(
            "check_compliance('%s'): %d rules, %d violations (%d errors, %d warnings)",
            entity, result["rules_evaluated"], len(violations),
            errors_count, warnings_count,
        )

        return result

    def get_compliance_report(self) -> dict[str, Any]:
        """Get a summary compliance report.

        Returns:
            Dict with total rules, violation history, and categories.
        """
        by_category: dict[str, int] = {}
        for rule in self._rules.values():
            by_category[rule.category] = by_category.get(rule.category, 0) + 1

        recent_violations = self._violation_history[-20:]

        return {
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "categories": by_category,
            "total_violations_recorded": len(self._violation_history),
            "recent_violations": [v.to_dict() for v in recent_violations],
        }

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _extract_value_for_rule(self, rule: PolicyRule, context: dict[str, Any]) -> Any:
        """Extract the appropriate value from context for a rule.

        Maps rule categories to context keys:
        - tax_pph21 → 'income' or 'rate'
        - tax_pph23 → 'rate'
        - tax_ppn → 'rate'
        - tax_compliance → 'npwp' or 'month'
        - default → 'value' or 'rate'
        """
        category_key_map: dict[str, list[str]] = {
            "tax_pph21": ["income", "rate", "value"],
            "tax_pph23": ["rate", "value"],
            "tax_ppn": ["rate", "value"],
            "tax_compliance": ["npwp", "month", "value"],
        }

        keys_to_try = category_key_map.get(rule.category, ["value", "rate"])

        for key in keys_to_try:
            if key in context:
                return context[key]

        # Fallback: return the first numeric value in context
        for v in context.values():
            if isinstance(v, (int, float)):
                return v

        return None

    def _suggest_remediation(self, rule: PolicyRule, value: Any) -> str:
        """Generate a remediation suggestion for a violation.

        Args:
            rule: The violated rule.
            value: The value that caused the violation.

        Returns:
            A suggestion string.
        """
        suggestions = {
            "ID_TAX_PPH21_THRESHOLD": "Pastikan penghasilan bruto melebihi PTKP sebelum menghitung PPh 21",
            "ID_TAX_PPH21_RATE_CAP": "Periksa tarif PPh 21 — tarif tidak boleh melebihi 35%",
            "ID_TAX_PPH23_RATE": "Periksa tarif PPh 23 — jasa: 2%, sewa: 4%",
            "ID_TAX_PPN_RATE": "PPN standar adalah 11% — periksa apakah ada pengecualian",
            "ID_TAX_NPWP_REQUIRED": "NPWP wajib untuk penghasilan di atas PTKP",
            "ID_TAX_FILING_DEADLINE": "Laporkan SPT sebelum batas waktu (31 Maret OP / 30 April Badan)",
        }

        return suggestions.get(rule.rule_id, f"Review rule '{rule.rule_id}': {rule.description}")
