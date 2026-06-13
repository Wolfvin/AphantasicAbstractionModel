"""AAM Pipeline Monitoring — Metrics collection and health checks.

Provides pipeline-level metrics for latency, throughput, graph growth,
gap detection rate, and system health.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TransformMetrics:
    """Metrics for a single pipeline transform."""
    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    error_count: int = 0
    
    def record(self, duration_ms: float, success: bool = True) -> None:
        self.call_count += 1
        self.total_time_ms += duration_ms
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.avg_time_ms = self.total_time_ms / self.call_count
        if not success:
            self.error_count += 1


@dataclass
class PipelineMetrics:
    """Aggregate pipeline metrics."""
    total_ingests: int = 0
    total_ask_calls: int = 0
    total_errors: int = 0
    total_ingest_time_ms: float = 0.0
    total_ask_time_ms: float = 0.0
    
    # Graph growth
    total_atoms_created: int = 0
    total_compositions_created: int = 0
    total_edges_created: int = 0
    total_gaps_detected: int = 0
    total_governance_transitions: int = 0
    
    # Per-transform metrics
    transform_metrics: dict[str, TransformMetrics] = field(default_factory=dict)
    
    # Health
    last_ingest_time: float = 0.0
    last_error_time: float = 0.0
    last_error_message: str = ""
    
    def record_ingest(self, duration_ms: float, result: dict, success: bool = True) -> None:
        self.total_ingests += 1
        self.total_ingest_time_ms += duration_ms
        if success:
            self.total_atoms_created += result.get("atoms_created", 0)
            self.total_compositions_created += result.get("compositions_created", 0)
            self.total_edges_created += result.get("edges_created", 0)
            self.total_gaps_detected += result.get("gaps_detected", 0)
            self.total_governance_transitions += result.get("governance_transitions", 0)
        else:
            self.total_errors += 1
            self.last_error_time = time.time()
            self.last_error_message = str(result.get("error", "unknown"))
        self.last_ingest_time = time.time()
    
    def record_ask(self, duration_ms: float, success: bool = True) -> None:
        self.total_ask_calls += 1
        self.total_ask_time_ms += duration_ms
        if not success:
            self.total_errors += 1
    
    def get_transform_metrics(self, name: str) -> TransformMetrics:
        if name not in self.transform_metrics:
            self.transform_metrics[name] = TransformMetrics(name=name)
        return self.transform_metrics[name]
    
    def avg_ingest_time_ms(self) -> float:
        return self.total_ingest_time_ms / max(self.total_ingests, 1)
    
    def avg_ask_time_ms(self) -> float:
        return self.total_ask_time_ms / max(self.total_ask_calls, 1)
    
    def gap_rate(self) -> float:
        return self.total_gaps_detected / max(self.total_compositions_created, 1)
    
    def error_rate(self) -> float:
        return self.total_errors / max(self.total_ingests + self.total_ask_calls, 1)
    
    def health_status(self) -> dict:
        return {
            "status": "healthy" if self.error_rate() < 0.1 else "degraded",
            "total_ingests": self.total_ingests,
            "total_ask_calls": self.total_ask_calls,
            "error_rate": round(self.error_rate(), 4),
            "avg_ingest_ms": round(self.avg_ingest_time_ms(), 2),
            "avg_ask_ms": round(self.avg_ask_time_ms(), 2),
            "gap_rate": round(self.gap_rate(), 4),
            "graph_size": {
                "atoms": self.total_atoms_created,
                "compositions": self.total_compositions_created,
                "edges": self.total_edges_created,
            },
            "last_error": self.last_error_message if self.last_error_time > 0 else None,
        }
    
    def to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.health_status(), f, indent=2)


class PipelineMonitor:
    """Runtime pipeline monitor with timing and error tracking."""
    
    def __init__(self, metrics: Optional[PipelineMetrics] = None, enabled: bool = True):
        self.metrics = metrics or PipelineMetrics()
        self.enabled = enabled
        self._timers: dict[str, float] = {}
    
    def start_timer(self, name: str) -> None:
        if self.enabled:
            self._timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        if not self.enabled or name not in self._timers:
            return 0.0
        duration_ms = (time.time() - self._timers.pop(name)) * 1000
        tm = self.metrics.get_transform_metrics(name)
        tm.record(duration_ms)
        return duration_ms
    
    def time_ingest(self, func):
        """Decorator to time ingest operations."""
        def wrapper(*args, **kwargs):
            self.start_timer("ingest")
            try:
                result = func(*args, **kwargs)
                duration = self.stop_timer("ingest")
                if isinstance(result, dict):
                    self.metrics.record_ingest(duration, result)
                return result
            except Exception as exc:
                self.stop_timer("ingest")
                self.metrics.record_ingest(0, {"error": str(exc)}, success=False)
                raise
        return wrapper


# Global monitor instance
_monitor: Optional[PipelineMonitor] = None

def get_monitor() -> PipelineMonitor:
    global _monitor
    if _monitor is None:
        _monitor = PipelineMonitor()
    return _monitor
