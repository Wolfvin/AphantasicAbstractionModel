"""
Temporal Tracking Layer — When did things happen and what's still relevant?

Analogi: Jin Soun menghubungkan kejadian berdasarkan TANGGAL —
"siapa yang ada di Hefei kapan?" Tanpa temporal metadata,
dia tidak bisa menyaring informasi berdasarkan waktu,
dan kasus Snow Plum Pill tidak terpecahkan.

This module adds a Python-side temporal tracking layer that:
1. Records when nodes were first seen and last accessed
2. Supports temporal queries ("what was active on date X?")
3. Tracks temporal validity (some facts expire, some persist)
4. Enables cross-referencing by temporal overlap

The Rust RSVS core tracks context counters internally,
but this Python layer adds wall-clock timestamps and
date-based querying that the Rust core doesn't provide.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TemporalRecord:
    """Temporal metadata for a knowledge graph node.
    
    Analogi: Setiap catatan di Simhyeon Pavilion punya cap tanggal —
    kapan diterima, kapan terakhir dibaca, dan kapan kadaluarsa.
    
    Attributes:
        label: Node label in the knowledge graph.
        first_seen: Unix timestamp when first observed.
        last_seen: Unix timestamp when last accessed/referenced.
        last_updated: Unix timestamp when the node's content changed.
        validity_start: Optional start of temporal validity window.
        validity_end: Optional end of temporal validity window (None = still valid).
        source: Where this node came from.
        access_count: How many times this node has been referenced.
        domain: What knowledge domain this node belongs to.
    """
    label: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_updated: float = 0.0
    validity_start: Optional[float] = None
    validity_end: Optional[float] = None
    source: str = "unknown"
    access_count: int = 0
    domain: str = ""

    def is_active(self, staleness_limit: float = 300.0) -> bool:
        """Check if this record is still active (not stale)."""
        if self.validity_end is not None and time.time() > self.validity_end:
            return False
        return (time.time() - self.last_seen) < staleness_limit

    def is_valid(self) -> bool:
        """Check if this record is within its validity window."""
        now = time.time()
        if self.validity_start is not None and now < self.validity_start:
            return False
        if self.validity_end is not None and now > self.validity_end:
            return False
        return True

    def touch(self) -> None:
        """Mark this record as recently accessed."""
        self.last_seen = time.time()
        self.access_count += 1

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "last_updated": self.last_updated,
            "validity_start": self.validity_start,
            "validity_end": self.validity_end,
            "source": self.source,
            "access_count": self.access_count,
            "domain": self.domain,
            "is_active": self.is_active(),
            "is_valid": self.is_valid(),
        }


class TemporalTracker:
    """Tracks temporal metadata for knowledge graph nodes.
    
    Provides date-based querying that the Rust RSVS core doesn't offer.
    This is a Python-side overlay that enriches RSVS nodes with
    wall-clock timestamps and temporal validity windows.
    
    Analogi: Jin Soun punya catatan kapan setiap informasi diterima.
    "Laporan dari Hefei datang tanggal 15. Catatan masuk-keluar
    dari tanggal 12-18. Misi Diancang tanggal 13." Tanpa tanggal,
    cross-reference temporal tidak mungkin.
    """

    def __init__(self, staleness_limit: float = 300.0) -> None:
        self._records: dict[str, TemporalRecord] = {}
        self._staleness_limit = staleness_limit
        self._domain_index: dict[str, set[str]] = {}  # domain → set of labels

    def record_observation(
        self,
        label: str,
        source: str = "unknown",
        domain: str = "",
        validity_start: Optional[float] = None,
        validity_end: Optional[float] = None,
    ) -> TemporalRecord:
        """Record that a node was observed (created or referenced).
        
        If the node already exists, update last_seen and increment access_count.
        If new, create a fresh TemporalRecord.
        """
        now = time.time()
        if label in self._records:
            rec = self._records[label]
            rec.touch()
            rec.last_updated = now
            if source != "unknown":
                rec.source = source
            if domain:
                rec.domain = domain
        else:
            rec = TemporalRecord(
                label=label,
                first_seen=now,
                last_seen=now,
                last_updated=now,
                validity_start=validity_start,
                validity_end=validity_end,
                source=source,
                domain=domain,
            )
            self._records[label] = rec
            if domain:
                self._domain_index.setdefault(domain, set()).add(label)
        return rec

    def get_record(self, label: str) -> Optional[TemporalRecord]:
        """Get temporal record for a node, or None if not tracked."""
        rec = self._records.get(label)
        if rec:
            rec.touch()
        return rec

    def query_by_time_range(
        self, start: float, end: float, domain: str = ""
    ) -> list[TemporalRecord]:
        """Find nodes that were active within a time range.
        
        Analogi: "Siapa yang ada di Hefei antara tanggal 10 dan 15?"
        """
        results = []
        for rec in self._records.values():
            # A record is "active in range" if its [first_seen, last_seen]
            # overlaps with [start, end]
            if rec.last_seen >= start and rec.first_seen <= end:
                if not domain or rec.domain == domain:
                    results.append(rec)
        return sorted(results, key=lambda r: r.last_seen, reverse=True)

    def query_active(self, domain: str = "") -> list[TemporalRecord]:
        """Find all currently active (non-stale) nodes."""
        results = []
        for rec in self._records.values():
            if rec.is_active(self._staleness_limit):
                if not domain or rec.domain == domain:
                    results.append(rec)
        return sorted(results, key=lambda r: r.last_seen, reverse=True)

    def query_temporal_overlap(
        self, label: str, tolerance: float = 3600.0
    ) -> list[TemporalRecord]:
        """Find nodes that temporally overlap with a given node.
        
        Two nodes temporally overlap if their [first_seen, last_seen] 
        windows overlap within a tolerance window.
        
        Analogi: "Siapa yang berada di Hefei pada waktu yang sama
        dengan Ju Jangmok?" — temporal overlap = benang merah.
        """
        target = self._records.get(label)
        if not target:
            return []
        
        results = []
        for rec in self._records.values():
            if rec.label == label:
                continue
            # Check if [target.first_seen - tol, target.last_seen + tol]
            # overlaps with [rec.first_seen, rec.last_seen]
            t_start = target.first_seen - tolerance
            t_end = target.last_seen + tolerance
            if rec.last_seen >= t_start and rec.first_seen <= t_end:
                results.append(rec)
        return sorted(results, key=lambda r: r.last_seen, reverse=True)

    def find_temporal_anomalies(self) -> list[dict]:
        """Find temporal anomalies — nodes that should have expired but haven't.
        
        Analogi: "Kenapa informasi dari 10 tahun lalu masih dianggap valid?
        Harusnya sudah kadaluarsa." — temporal anomaly detection.
        """
        anomalies = []
        now = time.time()
        for rec in self._records.values():
            if rec.validity_end is not None and now > rec.validity_end:
                if rec.is_active():
                    anomalies.append({
                        "type": "expired_but_active",
                        "label": rec.label,
                        "validity_end": rec.validity_end,
                        "last_seen": rec.last_seen,
                        "description": f"Node '{rec.label}' has expired but is still active",
                    })
        return anomalies

    def prune_stale(self, max_age: float = 86400.0) -> int:
        """Remove records older than max_age seconds. Returns count pruned."""
        now = time.time()
        stale_labels = [
            label for label, rec in self._records.items()
            if (now - rec.last_seen) > max_age
        ]
        for label in stale_labels:
            rec = self._records.pop(label)
            if rec.domain in self._domain_index:
                self._domain_index[rec.domain].discard(label)
        return len(stale_labels)

    def save_to_dict(self) -> dict:
        """Serialize all temporal records."""
        return {
            "staleness_limit": self._staleness_limit,
            "records": {k: v.to_dict() for k, v in self._records.items()},
            "domain_index": {k: list(v) for k, v in self._domain_index.items()},
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore temporal records from a dict."""
        self._staleness_limit = data.get("staleness_limit", 300.0)
        records_data = data.get("records", {})
        self._records = {}
        for label, rdict in records_data.items():
            self._records[label] = TemporalRecord(
                label=rdict.get("label", label),
                first_seen=rdict.get("first_seen", 0.0),
                last_seen=rdict.get("last_seen", 0.0),
                last_updated=rdict.get("last_updated", 0.0),
                validity_start=rdict.get("validity_start"),
                validity_end=rdict.get("validity_end"),
                source=rdict.get("source", "unknown"),
                access_count=rdict.get("access_count", 0),
                domain=rdict.get("domain", ""),
            )
        domain_data = data.get("domain_index", {})
        self._domain_index = {k: set(v) for k, v in domain_data.items()}

    def predict_rising_nodes(self, top_k: int = 10) -> list[dict]:
        """Identify nodes with increasing access patterns — likely to become important.

        Analogi: Jin Soun memperhatikan "orang ini sekarang tidak penting,
        tapi 10 tahun lagi dia jadi pemimpin sect." Proactive network building.

        Uses access_count growth rate as a signal.

        Args:
            top_k: Maximum number of results to return.

        Returns:
            A list of dicts with label, access stats, and trend score.
        """
        now = time.time()
        results = []
        for rec in self._records.values():
            if rec.access_count < 2:
                continue
            age_hours = max((now - rec.first_seen) / 3600.0, 0.1)
            access_rate = rec.access_count / age_hours  # accesses per hour
            # Recent acceleration: compare recent access vs lifetime average
            recent_window = 3600.0  # last hour
            is_recent = (now - rec.last_seen) < recent_window
            acceleration = 2.0 if is_recent else 0.5  # recently active = accelerating

            trend_score = access_rate * acceleration
            results.append({
                "label": rec.label,
                "access_count": rec.access_count,
                "access_rate_per_hour": round(access_rate, 3),
                "trend_score": round(trend_score, 3),
                "is_recently_active": is_recent,
            })

        results.sort(key=lambda x: x["trend_score"], reverse=True)
        return results[:top_k]

    def identify_promotion_candidates(self, bridge, top_k: int = 10) -> list[dict]:
        """Find nodes that are low-confidence but potentially high-value.

        Analogi: Jin Soun membangun relasi dengan orang SEBELUM mereka
        jadi penting. Proactive = identify nodes worth investing in.

        Args:
            bridge: RSVS bridge to check node confidence.
            top_k: Maximum number of results to return.

        Returns:
            A list of dicts with label, confidence, access count, and reason.
        """
        candidates = []
        for label, rec in self._records.items():
            if rec.access_count < 2 or not rec.is_active():
                continue
            # Check RSVS confidence
            confidence = 0.5
            if bridge and bridge.is_available:
                try:
                    info = bridge.node_info(label)
                    if isinstance(info, dict):
                        confidence = info.get("confidence", 0.5)
                except Exception:
                    pass

            # Low confidence but high access = potential promotion candidate
            if confidence < 0.5 and rec.access_count >= 3:
                candidates.append({
                    "label": label,
                    "confidence": confidence,
                    "access_count": rec.access_count,
                    "reason": f"Low confidence ({confidence:.2f}) but frequently accessed ({rec.access_count}x)",
                })

        candidates.sort(key=lambda x: x["access_count"], reverse=True)
        return candidates[:top_k]

    def get_stats(self) -> dict:
        """Get temporal tracking statistics."""
        active = sum(1 for r in self._records.values() if r.is_active(self._staleness_limit))
        return {
            "total_records": len(self._records),
            "active_records": active,
            "domains": list(self._domain_index.keys()),
            "staleness_limit": self._staleness_limit,
        }
