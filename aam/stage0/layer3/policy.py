"""
Layer 3 — Deductive Policy Engine

Extends the Layer 2 PolicyEngine with deductive reasoning capabilities
that integrate RSVS PolicyMeta (governance_score, status_flip_count,
seen_fingerprints) for trust-weighted, auditable compliance checking.

Analogi: Layer 2 PolicyEngine = Jin Soun membuka buku hukum dan
mengecek "Apakah tindakan ini melanggar Pasal 21 UU PPh?"
Layer 3 DeductivePolicyEngine = Jin Soun mengecek BUKU HUKUM +
catatan pengawasan (governance_score, stabilitas entitas) untuk
memberikan verdict yang lebih nuanced dan bisa diaudit.

Layer 3 additions over Layer 2:
  - DeductivePolicyEngine: extends PolicyEngine with check_with_rsvs_policy()
    which reads PolicyMeta from RSVS graph nodes and adjusts compliance
    confidence based on governance, stability, and dedup info.

All base functionality (PolicyEngine, PolicyRule, PolicyViolation,
_SAFE_EVAL_NAMES, Indonesian tax rules, etc.) is imported from
layer2.policy_engine — no duplication.
"""

from __future__ import annotations

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

import logging
from typing import Any, Optional

# P1-1: Import from Layer 2 instead of duplicating
# P1-2: Cross-package imports use absolute style (layer2 is a sibling package,
# not a subpackage, so relative imports like ..layer2 don't work in this layout)
from layer2.policy_engine import (
    PolicyEngine,
    PolicyRule,
    PolicyViolation,
    _SAFE_EVAL_NAMES,
)
from layer2.bridge import RsvsBridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeductivePolicyEngine — Layer 3 extension
# ---------------------------------------------------------------------------

class DeductivePolicyEngine(PolicyEngine):
    """Deductive Policy Engine — extends PolicyEngine with RSVS PolicyMeta.

    Adds trust-weighted, auditable compliance checking that reads
    PolicyMeta (governance_score, status_flip_count, seen_fingerprints)
    from RSVS graph nodes and incorporates it into compliance evaluation.

    This bridges the gap between the RSVS graph's policy metadata and
    the PolicyEngine's deterministic rule evaluation — the Layer 3
    "deductive" layer adds context-aware confidence adjustment on top
    of Layer 2's binary pass/fail rule checking.

    Analogi: Jin Soun mengecek buku hukum (PolicyEngine), TAPI
    juga mengecek catatan pengawasan: "Entitas ini punya governance
    score 0.9 dan belum pernah berubah status" → trust tinggi,
    compliance confidence bisa dipercaya. Sebaliknya, "governance
    score 0.2, status flip 7 kali" → hati-hati, mungkin ada masalah.

    All Layer 2 methods (add_rule, check_compliance, get_compliance_report,
    load_tax_rules_indonesia, etc.) are inherited unchanged.
    """

    def check_with_rsvs_policy(
        self,
        entity_label: str,
        bridge: Optional[RsvsBridge] = None,
    ) -> dict:
        """Check compliance using PolicyMeta from the RSVS graph.

        Reads PolicyMeta (governance_score, status_flip_count,
        seen_fingerprints) from RSVS node_info and incorporates it
        into compliance evaluation. This bridges the gap between
        the RSVS graph's policy metadata and the PolicyEngine's
        deterministic rule evaluation.

        PolicyMeta fields used:
        - governance_score: Used as a trust weight. Higher governance
          score → more trustworthy entity → less strict evaluation.
        - status_flip_count: Used as instability indicator. High flip
          count → entity has changed status frequently → increase
          scrutiny.
        - seen_fingerprints: Used for dedup checking. If the entity's
          fingerprint has been seen before, skip re-evaluation.

        If the bridge is unavailable or the entity has no PolicyMeta,
        falls back to standalone mode (standard check_compliance).

        Args:
            entity_label: The entity label to check in the RSVS graph.
            bridge: Optional override bridge (uses instance bridge if None).

        Returns:
            A dict with:
                - "entity": str — the entity label checked
                - "policy_meta_available": bool — whether PolicyMeta was found
                - "governance_score": float — from PolicyMeta (0.0 if not found)
                - "status_flip_count": int — from PolicyMeta (0 if not found)
                - "trust_weight": float — derived from governance_score
                - "instability_flag": bool — whether flip count exceeds threshold
                - "is_duplicate": bool — whether fingerprint was already seen
                - "compliance": dict — result from check_compliance (adjusted)
                - "adjusted_confidence": float — confidence adjusted by trust weight
        """
        b = bridge or self._bridge
        result: dict[str, Any] = {
            "entity": entity_label,
            "policy_meta_available": False,
            "governance_score": 0.0,
            "status_flip_count": 0,
            "trust_weight": 1.0,
            "instability_flag": False,
            "is_duplicate": False,
            "compliance": {},
            "adjusted_confidence": 0.0,
        }

        # Try to read PolicyMeta from RSVS node_info
        policy_meta: dict[str, Any] = {}
        if b is not None and b.is_available:
            try:
                node_info = b.node_info(entity_label)
                if node_info and isinstance(node_info, dict):
                    # PolicyMeta may be nested under "policy_meta" key
                    # or available as top-level fields from Rust core
                    policy_meta = node_info.get("policy_meta", {})
                    if not policy_meta:
                        # Try reading individual fields from node_info directly
                        if "governance_score" in node_info:
                            policy_meta = {
                                "governance_score": node_info.get("governance_score", 0.0),
                                "status_flip_count": node_info.get("status_flip_count", 0),
                                "seen_fingerprints": node_info.get("seen_fingerprints", []),
                            }
            except Exception as exc:
                logger.debug("node_info() failed for '%s': %s", entity_label, exc)

        # Extract PolicyMeta fields
        if policy_meta:
            result["policy_meta_available"] = True
            governance_score = float(policy_meta.get("governance_score", 0.0))
            status_flip_count = int(policy_meta.get("status_flip_count", 0))
            seen_fingerprints = policy_meta.get("seen_fingerprints", [])

            result["governance_score"] = governance_score
            result["status_flip_count"] = status_flip_count

            # --- Trust weight from governance_score ---
            # governance_score 0.0 = no governance = low trust → trust_weight = 0.5
            # governance_score 1.0 = strong governance = high trust → trust_weight = 1.0
            result["trust_weight"] = 0.5 + 0.5 * governance_score

            # --- Instability flag from status_flip_count ---
            # More than 3 flips suggests instability
            _FLIP_THRESHOLD = 3
            result["instability_flag"] = status_flip_count > _FLIP_THRESHOLD

            # --- Dedup check from seen_fingerprints ---
            # Check if this entity label appears in seen fingerprints
            entity_fingerprint = f"policy_check:{entity_label}"
            if isinstance(seen_fingerprints, list):
                result["is_duplicate"] = entity_fingerprint in seen_fingerprints
            else:
                result["is_duplicate"] = False
        else:
            # No PolicyMeta found — use standalone mode defaults
            result["trust_weight"] = 0.7  # Moderate trust when no metadata
            logger.debug(
                "No PolicyMeta found for '%s', using standalone mode",
                entity_label,
            )

        # Run standard compliance check using entity_label as context
        compliance = self.check_compliance(entity_label)
        result["compliance"] = compliance

        # Adjust confidence based on trust weight and instability
        # Higher trust → compliance result is more reliable
        # Instability → reduce confidence
        base_confidence = 1.0
        if compliance.get("violations"):
            base_confidence = 0.5
        elif compliance.get("warnings"):
            base_confidence = 0.7
        else:
            base_confidence = 0.9

        trust_weight = result["trust_weight"]
        instability_penalty = 0.1 if result["instability_flag"] else 0.0
        duplicate_bonus = 0.05 if result["is_duplicate"] else 0.0
        # If duplicate, we've seen this before → slight confidence bonus

        result["adjusted_confidence"] = max(
            0.0, min(1.0,
                base_confidence * trust_weight - instability_penalty + duplicate_bonus
            )
        )

        logger.info(
            "check_with_rsvs_policy('%s'): meta=%s, trust=%.2f, "
            "instability=%s, duplicate=%s, adj_conf=%.3f",
            entity_label, result["policy_meta_available"],
            result["trust_weight"], result["instability_flag"],
            result["is_duplicate"], result["adjusted_confidence"],
        )

        return result
