"""
Policy Engine — Rule-based Compliance Checking for Tax/Regulation

Analogi: Jin Soun punya perfect memory (graph), tapi untuk urusan
hukum/regulasi, dia juga butuh peraturan tertulis. Graph = mengingat
konteks dan menemukan rule yang relevan. Policy Engine = mengevaluasi
apakah suatu situasi memenuhi rule — seperti merujuk ke buku hukum.

Flow:
1. Ingest rules into RSVS graph (rules become knowledge)
2. When checking compliance: graph finds relevant rules via relate()
3. Evaluate each relevant rule against the situation (deterministic)
4. Report violations with evidence, suggestions, and legal references

Key Insight: The RSVS graph provides WHICH rules apply (pattern completion),
but the Policy Engine provides WHETHER a rule is satisfied (deterministic).
This is the combination of intuitive memory + explicit regulation.

Analogi novel:
    Graph = Jin Soun mengingat semua konteks kasus
    Policy Engine = Jin Soun membuka buku hukum dan mengecek
    "Apakah tindakan ini melanggar Pasal 21 UU PPh?"

Without the policy engine, Jin Soun hanya bisa menebak apakah
sesuatu itu legal atau tidak. Dengan policy engine, dia bisa
MEMBUKTIKKAN — karena aturannya tertulis dan bisa diaudit.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed names for safe eval of string expressions
# ---------------------------------------------------------------------------

_SAFE_EVAL_NAMES: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "len": len,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
    "not": lambda a: not a,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PolicyRule:
    """A single policy rule — explicit, auditable, binary.

    Unlike predictive coding (probabilistic), a policy rule is
    deterministic — it either passes or fails.

    Analogi: Ini adalah pasal dalam buku hukum. Bukan "mungkin
    melanggar", tapi "melanggar atau tidak". Tidak ada abu-abu.

    Attributes:
        rule_id: Unique identifier (e.g., "TAX_PPH_21_001")
        domain: Domain tag (e.g., "tax_pph", "regulation_kse")
        description: Human-readable description of the rule
        condition: Callable or string expression that evaluates to True/False.
            - Callable: takes context dict and returns bool
            - String: a simple Python expression eval'd with context dict
        severity: "critical", "warning", or "info"
        reference: Legal/regulatory reference (e.g., "UU PPh Pasal 21")
        tags: Additional classification tags
    """

    rule_id: str
    domain: str
    description: str
    condition: Callable[[dict], bool] | str
    severity: str = "warning"  # "critical", "warning", "info"
    reference: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "rule_id": self.rule_id,
            "domain": self.domain,
            "description": self.description,
            "condition_type": "callable" if callable(self.condition) else "expression",
            "condition_repr": (
                getattr(self.condition, "__name__", repr(self.condition))
                if callable(self.condition)
                else str(self.condition)
            ),
            "severity": self.severity,
            "reference": self.reference,
            "tags": list(self.tags),
        }


@dataclass
class PolicyViolation:
    """A detected policy violation.

    Created by PolicyEngine when a rule's condition evaluates to False.
    Contains all the information needed for an audit trail.

    Analogi: Ini adalah "surat tilang" — bukti bahwa suatu aturan
    dilanggar, lengkap dengan pasal yang dilanggar, bukti, dan
    saran perbaikan.

    Attributes:
        rule_id: Which rule was violated
        description: Human-readable description of the violation
        severity: Severity level ("critical", "warning", "info")
        evidence: List of evidence items supporting the violation
        suggestion: Suggested corrective action
        reference: Legal/regulatory reference
    """

    rule_id: str
    description: str
    severity: str
    evidence: list[str] = field(default_factory=list)
    suggestion: str = ""
    reference: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "severity": self.severity,
            "evidence": list(self.evidence),
            "suggestion": self.suggestion,
            "reference": self.reference,
        }


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Rule-based Policy Engine for compliance checking.

    Combines RSVS pattern completion (finding WHICH rules apply)
    with deterministic evaluation (checking IF a rule is satisfied).

    Analogi: Jin Soun punya perfect memory (graph) yang bisa menemukan
    aturan mana yang relevan, dan buku hukum (policy engine) yang bisa
    mengecek apakah aturan itu dilanggar. Keduanya bekerja bersama:

    1. Graph → "Aturan tentang PPh 21 relevan untuk situasi ini"
    2. Policy Engine → "Situasi ini MELANGGAR Pasal 21 karena X"

    Flow:
    1. add_rule() → store rule + ingest description into RSVS graph
    2. check_compliance() → use RSVS relate() to find relevant rules
       → evaluate each rule's condition against the context
       → return violations
    3. get_compliance_report() → full report with violations, warnings,
       passed rules, and overall status

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected.
        is_rust_core: Whether the Rust core backend is being used.
    """

    def __init__(self, bridge: Optional[RsvsBridge] = None) -> None:
        """Initialize the Policy Engine.

        Args:
            bridge: Optional shared RsvsBridge instance. If None,
                a new bridge is created via get_bridge().
        """
        if bridge is not None:
            self._bridge = bridge
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core

        # Rule store: rule_id → PolicyRule
        self._rules: dict[str, PolicyRule] = {}

        # Violation tracking
        self._violations: list[PolicyViolation] = []

        if self.rsvs_available:
            logger.info(
                "PolicyEngine initialized with RSVS bridge (rust_core=%s)",
                self.is_rust_core,
            )
        else:
            logger.info("PolicyEngine initialized WITHOUT RSVS core (fallback mode)")

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a policy rule. Also ingests it into RSVS graph.

        The rule's description is ingested into the RSVS graph as knowledge,
        so that future relate() calls can discover this rule based on
        semantic similarity to a given context.

        Analogi: Jin Soun menulis aturan baru di buku hukumnya,
        dan juga mencatatnya di Simhyeon Pavilion agar bisa
        ditemukan kembali nanti melalui asosiasi.

        Args:
            rule: The PolicyRule to add.
        """
        self._rules[rule.rule_id] = rule

        # Ingest into RSVS graph for semantic search
        ingest_text = (
            f"Policy rule {rule.rule_id} [{rule.domain}]: {rule.description}. "
            f"Reference: {rule.reference}. "
            f"Severity: {rule.severity}. "
            f"Tags: {', '.join(rule.tags) if rule.tags else 'none'}."
        )
        if self.rsvs_available:
            try:
                self._bridge.ingest(ingest_text)
                logger.debug("Rule '%s' ingested into RSVS graph", rule.rule_id)
            except Exception as exc:
                logger.warning(
                    "Failed to ingest rule '%s' into RSVS: %s", rule.rule_id, exc
                )

        logger.info(
            "Rule added: '%s' domain=%s severity=%s",
            rule.rule_id, rule.domain, rule.severity,
        )

    def add_rules_from_dict(self, rules_data: list[dict]) -> int:
        """Batch add rules from dict format.

        Each dict should have keys matching PolicyRule fields.
        The 'condition' key can be either a callable or a string expression.

        Analogi: Jin Soun menerima buku hukum baru berisi banyak pasal
        sekaligus — bukan satu-satu.

        Args:
            rules_data: List of dicts with rule data.

        Returns:
            Count of rules successfully added.
        """
        count = 0
        for rule_dict in rules_data:
            try:
                rule = PolicyRule(
                    rule_id=rule_dict["rule_id"],
                    domain=rule_dict.get("domain", "general"),
                    description=rule_dict.get("description", ""),
                    condition=rule_dict.get("condition", lambda ctx: True),
                    severity=rule_dict.get("severity", "warning"),
                    reference=rule_dict.get("reference", ""),
                    tags=rule_dict.get("tags", []),
                )
                self.add_rule(rule)
                count += 1
            except Exception as exc:
                logger.warning(
                    "Failed to add rule from dict: %s — %s",
                    rule_dict.get("rule_id", "unknown"), exc,
                )
        return count

    def get_rule(self, rule_id: str) -> Optional[PolicyRule]:
        """Get a specific rule by ID.

        Args:
            rule_id: The rule identifier.

        Returns:
            The PolicyRule, or None if not found.
        """
        return self._rules.get(rule_id)

    def list_rules(self, domain: str | None = None) -> list[PolicyRule]:
        """List all rules, optionally filtered by domain.

        Args:
            domain: Optional domain filter.

        Returns:
            List of PolicyRule objects.
        """
        if domain is None:
            return list(self._rules.values())
        return [r for r in self._rules.values() if r.domain == domain]

    # ------------------------------------------------------------------
    # Compliance checking
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        context: str,
        context_atoms: list[str] | None = None,
    ) -> dict:
        """Check compliance against all applicable rules.

        Two-phase approach:
        1. Use RSVS relate() to find rules relevant to the context
        2. Evaluate each relevant rule's condition against the context

        If RSVS is unavailable, all rules are evaluated.

        Analogi: Jin Soun mendengar situasi, lalu otaknya
        mengaktifkan aturan mana yang relevan (RSVS relate),
        lalu mengecek satu-satu apakah aturan itu dilanggar (eval).

        Args:
            context: A text description of the situation to check.
            context_atoms: Optional list of context atoms for RSVS
                disambiguation. If provided, context_query() is used
                instead of relate() for more precise matching.

        Returns:
            A dict with:
                - "total_rules_checked": int
                - "violations": list[PolicyViolation]
                - "warnings": list[PolicyViolation]
                - "info": list[PolicyViolation]
                - "passed": list[dict]
                - "applicable_rules": list[str]  # rule_ids
        """
        context_atoms = context_atoms or []
        applicable_rules = self.get_applicable_rules(context, top_k=20)

        # If no applicable rules found via RSVS, check ALL rules
        if not applicable_rules:
            applicable_rules = list(self._rules.values())

        violations: list[PolicyViolation] = []
        warnings: list[PolicyViolation] = []
        info_list: list[PolicyViolation] = []
        passed: list[dict] = []

        for rule in applicable_rules:
            violation = self.check_single_rule(rule.rule_id, context)
            if violation is not None:
                if violation.severity == "critical":
                    violations.append(violation)
                elif violation.severity == "warning":
                    warnings.append(violation)
                else:
                    info_list.append(violation)
                # Also track in internal violations list
                self._violations.append(violation)
            else:
                passed.append({
                    "rule_id": rule.rule_id,
                    "description": rule.description,
                    "severity": rule.severity,
                })

        return {
            "total_rules_checked": len(applicable_rules),
            "violations": violations,
            "warnings": warnings,
            "info": info_list,
            "passed": passed,
            "applicable_rules": [r.rule_id for r in applicable_rules],
        }

    def check_single_rule(
        self, rule_id: str, context: str
    ) -> PolicyViolation | None:
        """Check a single specific rule against context.

        Evaluates the rule's condition against the context.
        If the condition returns False → violation.

        The context string is parsed into a dict for condition evaluation.
        Simple key=value pairs are extracted (e.g., "income=50000000"
        becomes {"income": 50000000.0}).

        Analogi: Jin Soun membuka buku hukum ke Pasal 21,
        lalu mengecek apakah situasi ini memenuhi syarat pasal tersebut.
        Tidak ada "mungkin" — hanya YA atau TIDAK.

        Args:
            rule_id: The rule to check.
            context: The situation description to check against.

        Returns:
            A PolicyViolation if the rule is violated, None if compliant.
        """
        rule = self._rules.get(rule_id)
        if rule is None:
            logger.warning("Rule '%s' not found", rule_id)
            return None

        # Parse context into a dict for condition evaluation
        context_dict = self._parse_context(context)

        try:
            result = self._evaluate_condition(rule.condition, context_dict)
        except Exception as exc:
            logger.error(
                "Error evaluating rule '%s': %s", rule_id, exc
            )
            # If evaluation fails, treat as violation (conservative)
            return PolicyViolation(
                rule_id=rule_id,
                description=f"Rule evaluation error: {exc}",
                severity=rule.severity,
                evidence=[f"Condition could not be evaluated: {exc}"],
                suggestion=f"Review the condition for rule {rule_id}",
                reference=rule.reference,
            )

        if not result:
            # Rule violated — create violation record
            evidence = self._collect_evidence(rule, context, context_dict)
            suggestion = self._generate_suggestion(rule, context_dict)

            violation = PolicyViolation(
                rule_id=rule_id,
                description=rule.description,
                severity=rule.severity,
                evidence=evidence,
                suggestion=suggestion,
                reference=rule.reference,
            )

            logger.info(
                "Violation detected: rule='%s' severity=%s context='%.80s'",
                rule_id, rule.severity, context,
            )
            self._violations.append(violation)
            return violation

        # Rule passed — no violation
        logger.debug("Rule '%s' passed for context: %.60s", rule_id, context)
        return None

    def get_applicable_rules(
        self, context: str, top_k: int = 10
    ) -> list[PolicyRule]:
        """Find rules applicable to a given context using RSVS graph.

        Uses RSVS relate() to find concepts related to the context,
        then matches those concepts against rule domains, tags,
        and descriptions.

        Analogi: Jin Soun mendengar "pajak penghasilan karyawan"
        dan otaknya secara otomatis mengaktifkan aturan tentang
        PPh 21, BPJS, dan kontrak kerja — bukan aturan tentang
        PPN atau PPh 23.

        Args:
            context: The situation description.
            top_k: Maximum number of rules to return.

        Returns:
            List of PolicyRule objects that are applicable.
        """
        if not self.rsvs_available or not self._rules:
            # Fallback: return all rules (limited by top_k)
            return list(self._rules.values())[:top_k]

        # Strategy 1: Use RSVS relate() to find related concepts
        related_labels: set[str] = set()
        try:
            relate_result = self._bridge.relate(context)
            if relate_result:
                related_labels = self._extract_labels(relate_result)
        except Exception as exc:
            logger.debug("relate() failed for context '%.60s': %s", context, exc)

        # Strategy 2: Use RSVS query() for direct lookup
        try:
            query_result = self._bridge.query(context)
            if query_result:
                query_labels = self._extract_labels(query_result)
                related_labels |= query_labels
        except Exception as exc:
            logger.debug("query() failed for context '%.60s': %s", context, exc)

        # Match related labels against rules
        scored_rules: list[tuple[float, PolicyRule]] = []

        for rule in self._rules.values():
            score = self._compute_relevance(rule, context, related_labels)
            if score > 0:
                scored_rules.append((score, rule))

        # Sort by relevance score (descending)
        scored_rules.sort(key=lambda x: -x[0])

        # If RSVS didn't find enough, supplement with domain-matching rules
        result_rules = [r for _, r in scored_rules]
        if len(result_rules) < top_k:
            # Add rules whose domain appears in the context
            context_lower = context.lower()
            for rule in self._rules.values():
                if rule not in result_rules:
                    if rule.domain.lower() in context_lower or any(
                        tag.lower() in context_lower for tag in rule.tags
                    ):
                        result_rules.append(rule)
                        if len(result_rules) >= top_k:
                            break

        # If still no results, return all rules (we'll evaluate everything)
        if not result_rules:
            result_rules = list(self._rules.values())

        return result_rules[:top_k]

    def get_violations(self) -> list[PolicyViolation]:
        """Get all detected violations.

        Returns:
            List of all PolicyViolation objects detected so far.
        """
        return list(self._violations)

    def clear_violations(self) -> None:
        """Clear the violation history."""
        self._violations = []

    # ------------------------------------------------------------------
    # Compliance report
    # ------------------------------------------------------------------

    def get_compliance_report(self, context: str) -> dict:
        """Generate a full compliance report.

        Combines compliance checking with a structured report format
        suitable for audit documentation.

        Analogi: Laporan audit lengkap — bukan hanya "melanggar/tidak",
        tapi detail lengkap: aturan apa yang dicek, mana yang dilanggar,
        buktinya apa, dan saran perbaikannya.

        Args:
            context: The situation description to check.

        Returns:
            A dict with:
                - "context": str
                - "timestamp": str
                - "total_rules_checked": int
                - "violations": list[PolicyViolation]
                - "warnings": list[PolicyViolation]
                - "passed": list[dict]
                - "overall_status": "compliant" | "non_compliant" | "warning"
                - "summary": str
        """
        compliance = self.check_compliance(context)

        critical_violations = compliance["violations"]
        warnings = compliance["warnings"]
        passed = compliance["passed"]

        # Determine overall status
        if any(v.severity == "critical" for v in critical_violations):
            overall_status = "non_compliant"
        elif warnings:
            overall_status = "warning"
        else:
            overall_status = "compliant"

        # Generate summary
        summary_parts: list[str] = []
        if critical_violations:
            summary_parts.append(
                f"{len(critical_violations)} critical violation(s)"
            )
        if warnings:
            summary_parts.append(f"{len(warnings)} warning(s)")
        if passed:
            summary_parts.append(f"{len(passed)} rule(s) passed")

        summary = (
            f"Compliance check: {overall_status}. "
            + ", ".join(summary_parts)
            if summary_parts
            else f"Compliance check: {overall_status}. No rules checked."
        )

        return {
            "context": context,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_rules_checked": compliance["total_rules_checked"],
            "violations": [v.to_dict() for v in critical_violations],
            "warnings": [v.to_dict() for v in warnings],
            "passed": passed,
            "overall_status": overall_status,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Indonesian Tax Rules Preset
    # ------------------------------------------------------------------

    def load_tax_rules_indonesia(self) -> int:
        """Load Indonesian tax rules as a preset.

        Includes sample rules for:
        - PPh 21 (income tax on employment)
        - PPh 23 (income tax on services)
        - PPN (value added tax)
        - BPJS (social security)

        These are simplified representations of actual Indonesian
        tax regulations for demonstration purposes.

        Analogi: Jin Soun menerima buku panduan pajak Indonesia
        dan mencatat semua aturan penting di buku hukumnya.

        Returns:
            Count of rules loaded.
        """
        tax_rules: list[dict] = [
            # ── PPh 21 Rules ──
            {
                "rule_id": "TAX_PPH21_001",
                "domain": "tax_pph21",
                "description": (
                    "Penghasilan karyawan di atas PTKP wajib dipotong PPh 21. "
                    "PTKP untuk WP sendiri adalah Rp 54.000.000/tahun."
                ),
                "condition": lambda ctx: ctx.get("income", 0) <= 54_000_000 or ctx.get("pph21_withheld", False),
                "severity": "critical",
                "reference": "UU PPh Pasal 21 ayat (1)",
                "tags": ["pph21", "ptkp", "karyawan", "withholding"],
            },
            {
                "rule_id": "TAX_PPH21_002",
                "domain": "tax_pph21",
                "description": (
                    "Tarif PPh 21 untuk penghasilan Rp 0 - Rp 60.000.000 "
                    "adalah 5%."
                ),
                "condition": lambda ctx: (
                    ctx.get("income", 0) > 60_000_000
                    or ctx.get("pph21_rate", 0) <= 0.05
                ),
                "severity": "critical",
                "reference": "UU PPh Pasal 17 ayat (1) huruf a",
                "tags": ["pph21", "tarif", "5persen"],
            },
            {
                "rule_id": "TAX_PPH21_003",
                "domain": "tax_pph21",
                "description": (
                    "Tarif PPh 21 untuk penghasilan Rp 60.000.000 - "
                    "Rp 250.000.000 adalah 15%."
                ),
                "condition": lambda ctx: (
                    ctx.get("income", 0) <= 60_000_000
                    or ctx.get("income", 0) > 250_000_000
                    or ctx.get("pph21_rate", 0) <= 0.15
                ),
                "severity": "critical",
                "reference": "UU PPh Pasal 17 ayat (1) huruf b",
                "tags": ["pph21", "tarif", "15persen"],
            },
            {
                "rule_id": "TAX_PPH21_004",
                "domain": "tax_pph21",
                "description": (
                    "Tarif PPh 21 untuk penghasilan Rp 250.000.000 - "
                    "Rp 500.000.000 adalah 25%."
                ),
                "condition": lambda ctx: (
                    ctx.get("income", 0) <= 250_000_000
                    or ctx.get("income", 0) > 500_000_000
                    or ctx.get("pph21_rate", 0) <= 0.25
                ),
                "severity": "critical",
                "reference": "UU PPh Pasal 17 ayat (1) huruf c",
                "tags": ["pph21", "tarif", "25persen"],
            },
            {
                "rule_id": "TAX_PPH21_005",
                "domain": "tax_pph21",
                "description": (
                    "Tarif PPh 21 untuk penghasilan di atas "
                    "Rp 500.000.000 adalah 30%."
                ),
                "condition": lambda ctx: (
                    ctx.get("income", 0) <= 500_000_000
                    or ctx.get("pph21_rate", 0) <= 0.30
                ),
                "severity": "critical",
                "reference": "UU PPh Pasal 17 ayat (1) huruf d",
                "tags": ["pph21", "tarif", "30persen"],
            },
            {
                "rule_id": "TAX_PPH21_006",
                "domain": "tax_pph21",
                "description": (
                    "Tunjangan harian (meal/transport) yang melebihi "
                    "batas wajar bukan merupakan penghasilan yang "
                    "dikecualikan. Batas wajar: Rp 500.000/hari."
                ),
                "condition": lambda ctx: (
                    ctx.get("daily_allowance", 0) <= 500_000
                    or ctx.get("allowance_taxed", False)
                ),
                "severity": "warning",
                "reference": "PMK 122/PMK.010/2015",
                "tags": ["pph21", "tunjangan", "natura"],
            },
            {
                "rule_id": "TAX_PPH21_007",
                "domain": "tax_pph21",
                "description": (
                    "Employer wajib melaporkan SPT Tahunan PPh 21 "
                    "paling lambat 31 Maret setiap tahun."
                ),
                "condition": lambda ctx: ctx.get("spt_filed", False) or ctx.get("period", "annual") != "annual",
                "severity": "critical",
                "reference": "UU KUP Pasal 3 ayat (2)",
                "tags": ["pph21", "spt", "reporting", "deadline"],
            },
            # ── PPh 23 Rules ──
            {
                "rule_id": "TAX_PPH23_001",
                "domain": "tax_pph23",
                "description": (
                    "PPh 23 dipotong atas penghasilan berupa sewa "
                    "dan penghasilan lain sebesar 2% dari jumlah bruto."
                ),
                "condition": lambda ctx: (
                    ctx.get("pph23_withheld", False)
                    or ctx.get("service_type", "") not in ("rental", "other_income")
                ),
                "severity": "critical",
                "reference": "UU PPh Pasal 23 ayat (1) huruf c",
                "tags": ["pph23", "sewa", "2persen"],
            },
            {
                "rule_id": "TAX_PPH23_002",
                "domain": "tax_pph23",
                "description": (
                    "PPh 23 dipotong atas jasa teknis, jasa konstruksi, "
                    "dan jasa konsultan sebesar 2% dari jumlah bruto "
                    "(untuk konstruksi) atau 4% (untuk jasa teknis/konsultan)."
                ),
                "condition": lambda ctx: (
                    ctx.get("pph23_withheld", False)
                    or ctx.get("service_type", "") not in ("technical", "construction", "consultant")
                ),
                "severity": "warning",
                "reference": "PP 94/2010 Pasal 2",
                "tags": ["pph23", "jasa", "teknis", "konstruksi"],
            },
            # ── PPN Rules ──
            {
                "rule_id": "TAX_PPN_001",
                "domain": "tax_ppn",
                "description": (
                    "Tarif PPN adalah 11% (sejak 1 Januari 2025 "
                    "berdasarkan UU HPP)."
                ),
                "condition": lambda ctx: (
                    ctx.get("ppn_rate", 0.11) == 0.11
                ),
                "severity": "critical",
                "reference": "UU HPP Pasal 7 ayat (1)",
                "tags": ["ppn", "tarif", "11persen"],
            },
            {
                "rule_id": "TAX_PPN_002",
                "domain": "tax_ppn",
                "description": (
                    "Pengusaha Kena Pajak (PKP) wajib memungut PPN "
                    "atas penyerahan BKP/JKP. PKP wajib dikukuhkan "
                    "jika omzet melebihi Rp 4.800.000.000/tahun."
                ),
                "condition": lambda ctx: (
                    ctx.get("annual_revenue", 0) <= 4_800_000_000
                    or ctx.get("pkp_registered", False)
                ),
                "severity": "critical",
                "reference": "UU PPN Pasal 3 ayat (1)",
                "tags": ["ppn", "pkp", "omzet"],
            },
            {
                "rule_id": "TAX_PPN_003",
                "domain": "tax_ppn",
                "description": (
                    "Faktur Pajak wajib diterbitkan untuk setiap "
                    "penyerahan BKP/JKP yang terutang PPN."
                ),
                "condition": lambda ctx: ctx.get("tax_invoice_issued", False) or ctx.get("ppn_exempt", False),
                "severity": "critical",
                "reference": "UU PPN Pasal 13 ayat (1)",
                "tags": ["ppn", "faktur_pajak", "invoice"],
            },
            # ── BPJS Rules ──
            {
                "rule_id": "TAX_BPJS_001",
                "domain": "tax_bpjs",
                "description": (
                    "Iuran BPJS Kesehatan dibayar oleh pekerja 1% "
                    "dan pemberi kerja 4% dari gaji pokok."
                ),
                "condition": lambda ctx: (
                    ctx.get("bpjs_health_employee_rate", 0) <= 0.01
                    and ctx.get("bpjs_health_employer_rate", 0) <= 0.04
                ),
                "severity": "critical",
                "reference": "UU BPJS Pasal 68 ayat (5)",
                "tags": ["bpjs", "kesehatan", "iuran"],
            },
            {
                "rule_id": "TAX_BPJS_002",
                "domain": "tax_bpjs",
                "description": (
                    "Iuran BPJS Ketenagakerjaan JHT dibayar oleh "
                    "pekerja 2% dan pemberi kerja 3.7% dari gaji pokok."
                ),
                "condition": lambda ctx: (
                    ctx.get("bpjs_jht_employee_rate", 0) <= 0.02
                    and ctx.get("bpjs_jht_employer_rate", 0) <= 0.037
                ),
                "severity": "critical",
                "reference": "PP 44/2015 Pasal 9 ayat (2)",
                "tags": ["bpjs", "jht", "iuran", "ketenagakerjaan"],
            },
            {
                "rule_id": "TAX_BPJS_003",
                "domain": "tax_bpjs",
                "description": (
                    "Iuran JKP (Jaminan Kecelakaan Kerja) ditanggung "
                    "sepenuhnya oleh pemberi kerja 0.24%-1.74% "
                    "sesuai risiko."
                ),
                "condition": lambda ctx: (
                    ctx.get("bpjs_jkp_employer_rate", 0) <= 0.0174
                ),
                "severity": "warning",
                "reference": "PP 44/2015 Pasal 15",
                "tags": ["bpjs", "jkp", "kecelakaan_kerja"],
            },
        ]

        return self.add_rules_from_dict(tax_rules)

    # ------------------------------------------------------------------
    # Internal: Condition evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_condition(
        condition: Callable[[dict], bool] | str,
        context_dict: dict,
    ) -> bool:
        """Evaluate a rule condition against a context dict.

        Supports both:
        a. Callable: directly call with context_dict
        b. String expression: safe eval with context_dict as namespace

        Args:
            condition: The condition to evaluate.
            context_dict: The context data as a dict.

        Returns:
            True if the condition passes, False if violated.

        Raises:
            Exception: If evaluation fails.
        """
        if callable(condition):
            return bool(condition(context_dict))

        if isinstance(condition, str):
            # Safe eval: only allow access to context_dict + safe names
            eval_namespace = {**_SAFE_EVAL_NAMES, **context_dict}
            result = eval(condition, {"__builtins__": {}}, eval_namespace)  # noqa: S307
            return bool(result)

        # Unknown condition type — treat as pass (conservative: don't flag)
        logger.warning("Unknown condition type: %s", type(condition))
        return True

    # ------------------------------------------------------------------
    # Internal: Context parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_context(context: str) -> dict[str, Any]:
        """Parse a context string into a dict for condition evaluation.

        Extracts key=value pairs from the string, trying to convert
        values to appropriate types (int, float, bool).

        Also extracts key phrases as boolean flags if they appear
        in the context without explicit values.

        Analogi: Jin Soun mendengar "karyawan dengan penghasilan
        80 juta, sudah potong PPh 21" dan otomatis mengkonversi
        menjadi data terstruktur: {income: 80000000, pph21_withheld: True}.

        Args:
            context: The context string to parse.

        Returns:
            A dict with parsed key-value pairs.
        """
        result: dict[str, Any] = {}

        # Strategy 1: Extract key=value pairs
        # Match patterns like: key=value, key: value, key = value
        kv_patterns = [
            r'(\w+)\s*[=:]\s*([^\s,;]+)',  # key=value or key: value
        ]

        for pattern in kv_patterns:
            for match in re.finditer(pattern, context):
                key = match.group(1).lower()
                raw_value = match.group(2).strip()

                # Try to convert value
                value = PolicyEngine._parse_value(raw_value)
                result[key] = value

        # Strategy 2: Extract numeric values with context clues
        # e.g., "penghasilan 80 juta" → income=80000000
        income_patterns = [
            (r'penghasilan\s+([\d,.]+)\s*(juta|jt)', lambda m: float(m.group(1).replace(',', '.')) * 1_000_000),
            (r'income\s+[=:]?\s*([\d,.]+)\s*(juta|jt|m)', lambda m: float(m.group(1).replace(',', '.')) * 1_000_000),
            (r'rp\.?\s*([\d,.]+)\s*(juta|jt)', lambda m: float(m.group(1).replace(',', '.')) * 1_000_000),
            (r'gaji\s+[=:]?\s*([\d,.]+)\s*(juta|jt)', lambda m: float(m.group(1).replace(',', '.')) * 1_000_000),
            (r'revenue\s+[=:]?\s*([\d,.]+)\s*(miliar|milyar|t)', lambda m: float(m.group(1).replace(',', '.')) * 1_000_000_000),
            (r'omzet\s+[=:]?\s*([\d,.]+)\s*(miliar|milyar|t)', lambda m: float(m.group(1).replace(',', '.')) * 1_000_000_000),
        ]

        for pattern, converter in income_patterns:
            for match in re.finditer(pattern, context, re.IGNORECASE):
                try:
                    value = converter(match)
                    # Map to appropriate key
                    if "penghasilan" in pattern or "income" in pattern:
                        result.setdefault("income", value)
                    elif "gaji" in pattern:
                        result.setdefault("income", value)
                    elif "revenue" in pattern or "omzet" in pattern:
                        result.setdefault("annual_revenue", value)
                except (ValueError, IndexError):
                    pass

        # Strategy 3: Extract boolean flags from common phrases
        bool_patterns = {
            "pph21_withheld": [r"(?:sudah|telah)\s+potong\s+(?:pph\s*21|pajak)", r"pph\s*21\s+(?:withheld|dipotong)"],
            "pkp_registered": [r"(?:sudah|telah)\s+(?:terdaftar|dikukuhkan)\s+(?:pkp|pengusaha\s+kena\s+pajak)", r"pkp\s+(?:registered|terdaftar)"],
            "spt_filed": [r"(?:sudah|telah)\s+(?:lapor|file)\s+(?:spt|surat\s+pemberitahuan)", r"spt\s+(?:filed|dilaporkan)"],
            "tax_invoice_issued": [r"(?:sudah|telah)\s+(?:terbit|keluar|issued)\s+(?:faktur\s+pajak|invoice)", r"faktur\s+pajak\s+(?:issued|diterbitkan)"],
            "ppn_exempt": [r"(?:bebas|exempt|tidak\s+kena)\s+(?:ppn|pajak\s+pertambahan\s+nilai)", r"ppn\s+(?:exempt|bebas)"],
            "allowance_taxed": [r"(?:tunjangan|allowance)\s+(?:dipajaki|kena\s+pajak|taxed)"],
        }

        context_lower = context.lower()
        for flag, patterns in bool_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context_lower):
                    result[flag] = True
                    break

        # Strategy 4: Extract rates
        rate_patterns = [
            (r'(?:tarif|rate)\s+(?:pph\s*21|pph21)\s*[=:]?\s*([\d.]+)\s*%', "pph21_rate"),
            (r'(?:tarif|rate)\s+(?:ppn)\s*[=:]?\s*([\d.]+)\s*%', "ppn_rate"),
            (r'(?:iuran|rate)\s+(?:bpjs\s+kesehatan|health)\s+(?:pekerja|employee)\s*[=:]?\s*([\d.]+)\s*%', "bpjs_health_employee_rate"),
            (r'(?:iuran|rate)\s+(?:bpjs\s+kesehatan|health)\s+(?:pemberi\s+kerja|employer)\s*[=:]?\s*([\d.]+)\s*%', "bpjs_health_employer_rate"),
            (r'(?:iuran|rate)\s+(?:bpjs\s+jht|jht)\s+(?:pekerja|employee)\s*[=:]?\s*([\d.]+)\s*%', "bpjs_jht_employee_rate"),
            (r'(?:iuran|rate)\s+(?:bpjs\s+jht|jht)\s+(?:pemberi\s+kerja|employer)\s*[=:]?\s*([\d.]+)\s*%', "bpjs_jht_employer_rate"),
            (r'(?:iuran|rate)\s+(?:bpjs\s+jkp|jkp)\s+(?:pemberi\s+kerja|employer)\s*[=:]?\s*([\d.]+)\s*%', "bpjs_jkp_employer_rate"),
        ]

        for pattern, key in rate_patterns:
            match = re.search(pattern, context_lower)
            if match:
                try:
                    result[key] = float(match.group(1)) / 100.0
                except ValueError:
                    pass

        # Strategy 5: Extract daily allowance
        allowance_match = re.search(r'(?:tunjangan|allowance)\s+(?:harian|daily)\s*[=:]?\s*rp\.?\s*([\d,.]+)', context_lower)
        if allowance_match:
            try:
                result["daily_allowance"] = float(allowance_match.group(1).replace(',', '.').replace('.', '', allowance_match.group(1).count('.') - 1) if allowance_match.group(1).count('.') > 1 else allowance_match.group(1).replace(',', '.'))
            except ValueError:
                pass

        return result

    @staticmethod
    def _parse_value(raw: str) -> Any:
        """Parse a raw string value into an appropriate Python type.

        Args:
            raw: The raw string value.

        Returns:
            The parsed value (bool, int, float, or str).
        """
        # Boolean
        if raw.lower() in ("true", "yes", "ya", "benar"):
            return True
        if raw.lower() in ("false", "no", "tidak", "salah"):
            return False

        # None
        if raw.lower() in ("none", "null", "kosong"):
            return None

        # Integer
        try:
            return int(raw)
        except ValueError:
            pass

        # Float
        try:
            return float(raw)
        except ValueError:
            pass

        # String (strip quotes)
        return raw.strip("\"'")

    # ------------------------------------------------------------------
    # Internal: Relevance scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_relevance(
        rule: PolicyRule,
        context: str,
        related_labels: set[str],
    ) -> float:
        """Compute how relevant a rule is to the given context.

        Uses keyword overlap and RSVS relation data to score relevance.

        Args:
            rule: The rule to score.
            context: The context string.
            related_labels: Labels found by RSVS relate() for the context.

        Returns:
            A relevance score between 0.0 and 1.0.
        """
        score = 0.0
        context_lower = context.lower()

        # Domain match
        if rule.domain.lower() in context_lower:
            score += 0.4

        # Tag match
        for tag in rule.tags:
            if tag.lower() in context_lower:
                score += 0.15

        # Description keyword overlap
        desc_words = set(rule.description.lower().split())
        context_words = set(context_lower.split())
        if desc_words and context_words:
            overlap = desc_words & context_words
            score += 0.3 * (len(overlap) / max(len(desc_words), 1))

        # RSVS relation match
        rule_id_lower = rule.rule_id.lower()
        domain_lower = rule.domain.lower()
        for label in related_labels:
            label_lower = str(label).lower()
            if domain_lower in label_lower or label_lower in domain_lower:
                score += 0.2
            if rule_id_lower in label_lower:
                score += 0.3

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Internal: Evidence & suggestion generation
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_evidence(
        rule: PolicyRule,
        context: str,
        context_dict: dict,
    ) -> list[str]:
        """Collect evidence for a violation.

        Args:
            rule: The violated rule.
            context: The original context string.
            context_dict: The parsed context dict.

        Returns:
            A list of evidence strings.
        """
        evidence: list[str] = []

        # Rule reference
        if rule.reference:
            evidence.append(f"Rule reference: {rule.reference}")

        # Context values that are relevant
        relevant_keys = set()
        condition_repr = (
            str(rule.condition) if isinstance(rule.condition, str)
            else getattr(rule.condition, "__code__", None)
        )
        if condition_repr:
            # Extract variable names from condition
            if isinstance(rule.condition, str):
                # Find variable names in expression
                var_names = re.findall(r'\b([a-zA-Z_]\w*)\b', rule.condition)
                relevant_keys.update(var_names)
            else:
                # For callables, check context_dict keys
                relevant_keys.update(context_dict.keys())

        for key in relevant_keys:
            if key in context_dict:
                evidence.append(f"{key}={context_dict[key]}")

        # Always include the raw context
        if context:
            evidence.append(f"Context: {context[:200]}")

        return evidence

    @staticmethod
    def _generate_suggestion(
        rule: PolicyRule,
        context_dict: dict,
    ) -> str:
        """Generate a corrective suggestion for a violation.

        Args:
            rule: The violated rule.
            context_dict: The parsed context dict.

        Returns:
            A suggestion string.
        """
        domain = rule.domain

        if "pph21" in domain:
            return (
                f"Pastikan PPh 21 dipotong sesuai tarif yang berlaku "
                f"berdasarkan {rule.reference}. "
                f"Periksa kembali perhitungan penghasilan kena pajak."
            )
        elif "pph23" in domain:
            return (
                f"Pastikan PPh 23 dipotong atas jasa yang terutang "
                f"sesuai {rule.reference}."
            )
        elif "ppn" in domain:
            return (
                f"Pastikan PPN dipungut dan faktur pajak diterbitkan "
                f"sesuai {rule.reference}."
            )
        elif "bpjs" in domain:
            return (
                f"Pastikan iuran BPJS dibayar sesuai ketentuan "
                f"berdasarkan {rule.reference}."
            )
        else:
            return (
                f"Review kepatuhan terhadap {rule.reference} "
                f"dan lakukan tindakan korektif."
            )

    # ------------------------------------------------------------------
    # Internal: Label extraction from RSVS results
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_labels(result: Any) -> set[str]:
        """Extract concept labels from RSVS bridge result.

        Handles both dict format (from bridge) and other formats.

        Args:
            result: Result from RsvsBridge.relate() or query().

        Returns:
            A set of concept label strings.
        """
        labels: set[str] = set()

        if isinstance(result, dict):
            # relate() result
            for key in ("related_nodes", "structural_relations"):
                items = result.get(key, [])
                for item in items:
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        labels.add(str(item[0]))
                    elif isinstance(item, str):
                        labels.add(item)

            # query() result
            for key in ("atoms", "compositions"):
                items = result.get(key, [])
                for item in items:
                    if isinstance(item, (list, tuple)) and len(item) >= 1:
                        labels.add(str(item[0]))
                    elif isinstance(item, str):
                        labels.add(item)

        return labels
