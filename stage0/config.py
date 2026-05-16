"""
AAM Stage0 — Pipeline Configuration

Centralized configuration for the AAM pipeline.
Supports defaults, environment variable overrides,
and programmatic construction.

Usage:
    config = PipelineConfig()
    config = PipelineConfig.from_env()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the AAM pipeline.

    Attributes:
        eta: Learning rate for predictive coding (default: 0.1).
        anomaly_threshold: Threshold for anomaly detection (default: 0.3).
        auto_search: Automatically search internet when confidence is low.
        use_llm: Whether to use LLM for narrative generation.
        language: Output language for narratives ("id" or "en").
        gap_detection_enabled: Whether gap detection is enabled.
        maintenance_interval: Run auto-maintenance every N ingests.
        max_narrative_length: Maximum length for generated narratives.
        confidence_floor: Minimum confidence value (floor).
    """

    eta: float = 0.1
    anomaly_threshold: float = 0.3
    auto_search: bool = False
    use_llm: bool = True
    language: str = "id"
    gap_detection_enabled: bool = True
    maintenance_interval: int = 50
    max_narrative_length: int = 2000
    confidence_floor: float = 0.05

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create a PipelineConfig from environment variables.

        Environment variables:
            AAM_ETA: Learning rate (float)
            AAM_ANOMALY_THRESHOLD: Anomaly threshold (float)
            AAM_AUTO_SEARCH: Auto search (bool string)
            AAM_USE_LLM: Use LLM (bool string)
            AAM_LANGUAGE: Language code
            AAM_GAP_DETECTION: Gap detection enabled (bool string)
            AAM_MAINTENANCE_INTERVAL: Maintenance interval (int)
        """
        def _env_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("1", "true", "yes"):
                return True
            if val in ("0", "false", "no"):
                return False
            return default

        def _env_float(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default

        def _env_int(key: str, default: int) -> int:
            try:
                return int(os.environ.get(key, default))
            except (ValueError, TypeError):
                return default

        return cls(
            eta=_env_float("AAM_ETA", 0.1),
            anomaly_threshold=_env_float("AAM_ANOMALY_THRESHOLD", 0.3),
            auto_search=_env_bool("AAM_AUTO_SEARCH", False),
            use_llm=_env_bool("AAM_USE_LLM", True),
            language=os.environ.get("AAM_LANGUAGE", "id"),
            gap_detection_enabled=_env_bool("AAM_GAP_DETECTION", True),
            maintenance_interval=_env_int("AAM_MAINTENANCE_INTERVAL", 50),
        )

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "eta": self.eta,
            "anomaly_threshold": self.anomaly_threshold,
            "auto_search": self.auto_search,
            "use_llm": self.use_llm,
            "language": self.language,
            "gap_detection_enabled": self.gap_detection_enabled,
            "maintenance_interval": self.maintenance_interval,
            "max_narrative_length": self.max_narrative_length,
            "confidence_floor": self.confidence_floor,
        }
