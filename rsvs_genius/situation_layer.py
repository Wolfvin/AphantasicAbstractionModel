"""
Situation Layer — Chat History as Semantic Memory + Active Sense Tracking

Analogi: Jin Soun sedang berada di Hefei → semua sense tentang 
Hefei, merchant guild, dan Snow Plum Pill aktif. Dia tidak perlu 
mengakses seluruh 30 tahun — hanya yang relevan untuk situasi sekarang.

Flow:
1. Chat messages → ingest into RSVS graph
2. Query graph for relevant context
3. Track which senses are currently active
4. Provide "state of the world" to other layers
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# How many recent messages to consider for "active" context
_DEFAULT_ACTIVE_WINDOW = 10

# How many seconds before a sense is considered "stale"
_SENSE_STALENESS_SECONDS = 300.0  # 5 minutes


def _get_rsvs_instance() -> Any:
    """Attempt to get an RSVS instance, returning None if unavailable.

    Tries multiple import strategies to be resilient across
    different installation setups.
    """
    # Strategy 1: Direct import from rsvs package
    try:
        from rsvs import Rsvs
        return Rsvs()
    except Exception:
        pass

    # Strategy 2: Get singleton from rsvs_core
    try:
        from rsvs.rsvs_core import get_rsvs_instance
        return get_rsvs_instance()
    except Exception:
        pass

    return None


class SituationLayer:
    """Situation Layer — Chat History as Semantic Memory + Active Sense Tracking.

    Turns a conversation into a living semantic graph where the most
    recently-discussed concepts are automatically "active" (high attention)
    and old topics gradually fade unless reinforced.

    Analogi: Jin Soun di Hefei — semua sense tentang Hefei, merchant guild,
    dan Snow Plum Pill otomatis aktif. Dia tidak perlu mengakses seluruh
    30 tahun pengalaman — hanya yang relevan untuk situasi sekarang.

    Attributes:
        rsvs: The RSVS instance (may be None if core is unavailable).
        rsvs_available: Whether a working RSVS instance is connected.
    """

    def __init__(self, rsvs_instance: Any | None = None) -> None:
        """Initialize the Situation Layer.

        Args:
            rsvs_instance: Optional pre-built RSVS instance. If None,
                the layer will try to obtain one from rsvs.rsvs_core.
        """
        if rsvs_instance is not None:
            self.rsvs: Any = rsvs_instance
            self.rsvs_available = True
        else:
            self.rsvs = _get_rsvs_instance()
            self.rsvs_available = self.rsvs is not None

        # Session tracking
        # Analogi: Jin Soun mencatat percakapan hari ini terpisah
        # dari arsip 30 tahun — tapi semuanya saling terhubung.
        self._messages: list[dict] = []
        self._active_senses: list[dict] = []
        self._session_start: float = time.time()
        self._last_event_seq: int = 0

        if self.rsvs_available:
            # Capture the current event sequence so we only track new events
            try:
                self._last_event_seq = self.rsvs.latest_seq_v1()
            except Exception:
                self._last_event_seq = 0
            logger.info("SituationLayer initialized with RSVS core (seq=%d)",
                        self._last_event_seq)
        else:
            logger.info("SituationLayer initialized WITHOUT RSVS core (fallback mode)")

    # ------------------------------------------------------------------
    # Chat message ingestion
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> dict:
        """Ingest a chat message into the semantic graph.

        Each message is ingested as text into the RSVS graph, creating
        atoms and relationships that can be queried later. The role
        (user/assistant/system) is preserved as metadata.

        Analogi: Setiap percakapan yang Jin Soun dengar atau ucapkan
        dicatat di Simhyeon Pavilion, lengkap dengan siapa yang berkata.

        Args:
            role: The speaker role — typically "user", "assistant",
                or "system".
            content: The message text content.

        Returns:
            A dict with keys:
                - "success": bool — whether ingestion succeeded
                - "role": str — the role tag
                - "message_index": int — 0-based index in session
                - "stats": dict | None — RSVS ingest stats
                - "active_atoms": list[str] — atoms activated by this message
        """
        message_record: dict = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "index": len(self._messages),
        }
        self._messages.append(message_record)

        result: dict = {
            "success": False,
            "role": role,
            "message_index": message_record["index"],
            "stats": None,
            "active_atoms": [],
        }

        # Format for ingestion — prefix with role for context
        # Analogi: Jin Soun mencatat "Pedagang berkata: harga naik"
        # bukan hanya "harga naik" — sumber penting untuk konteks.
        ingest_text = f"[{role}] {content}"

        if self.rsvs_available:
            try:
                stats = self.rsvs.ingest(ingest_text)
                result["success"] = True
                result["stats"] = self._normalize_stats(stats)

                # After ingestion, update active senses
                self._update_active_senses()

                # Try to get activated atoms from the latest event stream
                try:
                    events = self._get_raw_events()
                    result["active_atoms"] = self._extract_new_atoms(events)
                except Exception:
                    pass

            except Exception as exc:
                logger.error("RSVS ingestion failed for message: %s", exc)
                result["success"] = False
                result["reason"] = str(exc)
        else:
            # Fallback — no RSVS, but we still track the message
            result["success"] = True
            result["stats"] = {"fallback": True}
            # Extract simple keyword-like "atoms" from content
            result["active_atoms"] = self._fallback_atomize(content)
            self._update_active_senses_fallback()

        return result

    # ------------------------------------------------------------------
    # Active sense tracking
    # ------------------------------------------------------------------

    def get_active_senses(self) -> list[dict]:
        """Return the currently active senses based on recent context.

        Active senses are concepts that have been recently discussed
        or queried. They represent the "current situation" — what the
        system is paying attention to right now.

        Analogi: Di Hefei, Jin Soun secara otomatis memikirkan
        tentang Snow Plum Pill, merchant guild, dan rute perdagangan —
        karena itu yang sedang relevan.

        Returns:
            A list of dicts, each with:
                - "label": str — the concept label
                - "sense_count": int — number of senses for this concept
                - "confidence": float — confidence score (0-1)
                - "last_seen": float — timestamp when last activated
                - "staleness": float — seconds since last activation
        """
        # Refresh from RSVS if available
        if self.rsvs_available:
            self._update_active_senses()
        return list(self._active_senses)

    # ------------------------------------------------------------------
    # Context queries
    # ------------------------------------------------------------------

    def get_relevant_context(self, query: str, top_k: int = 5) -> list[dict]:
        """Query the graph for nodes related to the query.

        Uses RSVS relate() for spreading activation — finds concepts
        that are semantically connected to the query, even if not
        directly mentioned.

        Analogi: Jin Soun mendengar "racun" dan otomatis teringat
        Snow Plum Pill, Hefei, dan semua kejadian terkait —
        bukan hanya konsep "racun" saja.

        Args:
            query: The concept or text to search for.
            top_k: Maximum number of results to return.

        Returns:
            A list of dicts, each with:
                - "label": str — the related concept label
                - "relevance": float — relevance score
                - "senses": int — number of senses
                - "source": str — where this relation came from
        """
        results: list[dict] = []

        if not self.rsvs_available:
            # Fallback — search through message history
            return self._fallback_relevant_context(query, top_k)

        try:
            # Use relate() for spreading activation
            relate_result = self.rsvs.relate(query)
            if relate_result is not None:
                results = self._parse_relate_result(relate_result, top_k)
        except Exception as exc:
            logger.warning("RSVS relate() failed: %s", exc)

        # If relate didn't give enough results, try query()
        if len(results) < top_k:
            try:
                query_result = self.rsvs.query(query, context="conversation")
                if query_result is not None:
                    extra = self._parse_query_result(query_result, top_k - len(results))
                    results.extend(extra)
            except Exception as exc:
                logger.warning("RSVS query() failed: %s", exc)

        return results[:top_k]

    # ------------------------------------------------------------------
    # Situation summary
    # ------------------------------------------------------------------

    def get_situation_summary(self) -> dict:
        """Return a summary of the current situation.

        Provides a comprehensive snapshot of the current conversational
        state — what's active, what's confident, and what's happened.

        Analogi: Sebelum mengambil keputusan, Jin Soun merangkum
        situasi: "Kita di Hefei, musuh sudah tau tentang resep,
        dan kita punya 3 hari sebelum auction."

        Returns:
            A dict with keys:
                - "active_senses": list[dict] — currently active senses
                - "confidence_map": dict[str, float] — confidence scores
                - "message_count": int — number of messages in session
                - "session_duration": float — seconds since session start
                - "recent_topics": list[str] — labels of recently discussed topics
                - "rsvs_status": dict | None — RSVS system status
        """
        now = time.time()
        summary: dict = {
            "active_senses": self.get_active_senses(),
            "confidence_map": {},
            "message_count": len(self._messages),
            "session_duration": now - self._session_start,
            "recent_topics": self._get_recent_topics(),
            "rsvs_status": None,
        }

        if self.rsvs_available:
            try:
                summary["confidence_map"] = self.rsvs.confidence_map()
            except Exception:
                pass
            try:
                summary["rsvs_status"] = self.rsvs.status()
            except Exception:
                pass

        return summary

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear_session(self) -> None:
        """Clear the current session context.

        Resets the message history and active senses. Does NOT
        clear the RSVS graph — the knowledge persists, only the
        "current situation" tracking is reset.

        Analogi: Jin Soun meninggalkan Hefei dan pergi ke kota baru.
        Pengetahuannya tentang Hefei tetap ada di Simhyeon Pavilion,
        tapi dia tidak lagi secara aktif memikirkannya.
        """
        self._messages = []
        self._active_senses = []
        self._session_start = time.time()

        # Update event sequence to skip past events
        if self.rsvs_available:
            try:
                self._last_event_seq = self.rsvs.latest_seq_v1()
            except Exception:
                self._last_event_seq = 0

        logger.info("Session cleared — fresh situation context")

    # ------------------------------------------------------------------
    # Event stream
    # ------------------------------------------------------------------

    def get_event_stream(self, after_seq: int = 0) -> list[dict]:
        """Get events since a given sequence number.

        Allows incremental consumption of graph events — useful for
        streaming updates to other layers or UI components.

        Analogi: Jin Soun bertanya "Apa yang berubah sejak terakhir
        aku periksa?" — bukan membaca ulang seluruh arsip.

        Args:
            after_seq: Only return events with sequence number > this.
                Default 0 returns all events in the current session window.

        Returns:
            A list of event dicts parsed from the RSVS event stream.
            Each dict has at least "seq" and "type" keys.
            Returns empty list if RSVS is unavailable.
        """
        if not self.rsvs_available:
            return []

        try:
            raw = self.rsvs.consume_events_v1(since_seq=after_seq)
            if raw:
                return self._parse_event_stream(raw)
        except Exception as exc:
            logger.warning("Failed to get event stream: %s", exc)

        return []

    # ------------------------------------------------------------------
    # Internal: Active sense management
    # ------------------------------------------------------------------

    def _update_active_senses(self) -> None:
        """Refresh the active senses list from the RSVS graph.

        Called after each message ingestion to keep the active
        sense tracking up to date.
        """
        if not self.rsvs_available:
            return

        now = time.time()
        new_senses: list[dict] = []

        try:
            # Get confidence map — high confidence = recently activated
            cmap = self.rsvs.confidence_map()
            # Sort by confidence (descending) and take top entries
            sorted_items = sorted(cmap.items(), key=lambda x: x[1], reverse=True)

            for label, confidence in sorted_items[:20]:
                # Skip if stale
                # (We approximate "last seen" from the current time;
                #  in a full implementation, we'd track this precisely)
                sense_count = 0
                try:
                    senses = self.rsvs.senses(label)
                    sense_count = len(senses) if senses else 0
                except Exception:
                    pass

                new_senses.append({
                    "label": label,
                    "sense_count": sense_count,
                    "confidence": confidence,
                    "last_seen": now,
                    "staleness": 0.0,
                })

        except Exception as exc:
            logger.warning("Failed to update active senses: %s", exc)
            return

        # Merge with existing — update timestamps for already-known senses
        existing_map = {s["label"]: s for s in self._active_senses}
        for sense in new_senses:
            if sense["label"] in existing_map:
                # Keep the older "last_seen" if it was more recent
                old = existing_map[sense["label"]]
                sense["last_seen"] = max(old["last_seen"], sense["last_seen"])

        # Mark staleness for senses not in the new batch
        all_labels = {s["label"] for s in new_senses}
        for old_sense in self._active_senses:
            if old_sense["label"] not in all_labels:
                stale_sense = dict(old_sense)
                stale_sense["staleness"] = now - stale_sense["last_seen"]
                if stale_sense["staleness"] < _SENSE_STALENESS_SECONDS:
                    new_senses.append(stale_sense)

        # Sort: active first, then by staleness
        new_senses.sort(key=lambda s: (s["staleness"], -s["confidence"]))
        self._active_senses = new_senses

    def _update_active_senses_fallback(self) -> None:
        """Update active senses without RSVS (fallback mode).

        Uses simple keyword extraction from recent messages.
        """
        if not self._messages:
            return

        now = time.time()

        # Get recent messages
        recent = self._messages[-_DEFAULT_ACTIVE_WINDOW:]

        # Simple keyword extraction
        word_counts: dict[str, int] = {}
        for msg in recent:
            words = msg["content"].lower().split()
            for word in words:
                # Skip short words and common stop words
                if len(word) > 3 and word not in {
                    "that", "this", "with", "from", "have", "been",
                    "they", "their", "which", "would", "there", "could",
                    "about", "other", "into", "more", "than", "then",
                }:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Build active senses from word counts
        new_senses: list[dict] = []
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1])[:20]:
            # Find if already tracked
            existing = None
            for s in self._active_senses:
                if s["label"] == word:
                    existing = s
                    break

            new_senses.append({
                "label": word,
                "sense_count": count,
                "confidence": min(count / 5.0, 1.0),
                "last_seen": existing["last_seen"] if existing else now,
                "staleness": 0.0 if existing else 0.0,
            })

        # Add stale senses that weren't in recent messages
        new_labels = {s["label"] for s in new_senses}
        for old_sense in self._active_senses:
            if old_sense["label"] not in new_labels:
                stale = dict(old_sense)
                stale["staleness"] = now - stale["last_seen"]
                if stale["staleness"] < _SENSE_STALENESS_SECONDS:
                    new_senses.append(stale)

        new_senses.sort(key=lambda s: (s["staleness"], -s["confidence"]))
        self._active_senses = new_senses

    # ------------------------------------------------------------------
    # Internal: Parsing helpers
    # ------------------------------------------------------------------

    def _get_raw_events(self) -> str:
        """Get raw event stream from RSVS since last check."""
        if not self.rsvs_available:
            return ""
        return self.rsvs.consume_events_v1(since_seq=self._last_event_seq)

    @staticmethod
    def _extract_new_atoms(events_raw: str) -> list[str]:
        """Extract new atom labels from an event stream string."""
        atoms: list[str] = []
        try:
            events = json.loads(events_raw) if events_raw else []
            for event in events:
                if isinstance(event, dict):
                    # Try common event structures
                    if event.get("type") == "new_atom" and "label" in event:
                        atoms.append(event["label"])
                    elif "atom" in event and isinstance(event["atom"], str):
                        atoms.append(event["atom"])
        except json.JSONDecodeError:
            pass
        return atoms

    @staticmethod
    def _parse_relate_result(result: Any, top_k: int) -> list[dict]:
        """Parse an RSVS relate() result into a list of related concepts.

        Args:
            result: The result from RSVS.relate().
            top_k: Maximum number of results.

        Returns:
            A list of dicts with "label", "relevance", "senses", "source".
        """
        items: list[dict] = []

        if isinstance(result, dict):
            # Dict-style result
            related = result.get("related", result.get("nodes", []))
            for item in related[:top_k]:
                if isinstance(item, dict):
                    items.append({
                        "label": item.get("label", str(item)),
                        "relevance": float(item.get("score",
                                                     item.get("relevance", 0.0))),
                        "senses": int(item.get("sense_count", 0)),
                        "source": "relate",
                    })
                elif isinstance(item, str):
                    items.append({
                        "label": item,
                        "relevance": 0.0,
                        "senses": 0,
                        "source": "relate",
                    })
        else:
            # PyO3 object — try attribute access
            try:
                for attr_name in ("related", "nodes", "atoms"):
                    related = getattr(result, attr_name, None)
                    if related is not None:
                        for item in related[:top_k]:
                            if isinstance(item, str):
                                items.append({
                                    "label": item,
                                    "relevance": 0.0,
                                    "senses": 0,
                                    "source": "relate",
                                })
                            elif hasattr(item, "label"):
                                items.append({
                                    "label": getattr(item, "label", str(item)),
                                    "relevance": float(getattr(item, "score",
                                                                getattr(item, "relevance", 0.0))),
                                    "senses": int(getattr(item, "sense_count", 0)),
                                    "source": "relate",
                                })
                        break
            except Exception:
                pass

        return items[:top_k]

    @staticmethod
    def _parse_query_result(result: Any, top_k: int) -> list[dict]:
        """Parse an RSVS query() result into related concepts.

        Args:
            result: The result from RSVS.query().
            top_k: Maximum number of results.

        Returns:
            A list of dicts with "label", "relevance", "senses", "source".
        """
        items: list[dict] = []

        if isinstance(result, dict):
            # Dict-style result
            label = result.get("label", result.get("concept", ""))
            if label:
                items.append({
                    "label": label,
                    "relevance": float(result.get("score",
                                                   result.get("confidence", 1.0))),
                    "senses": int(result.get("sense_count", 0)),
                    "source": "query",
                })
        else:
            # PyO3 object
            try:
                label = getattr(result, "label", None)
                if label:
                    items.append({
                        "label": label,
                        "relevance": float(getattr(result, "score",
                                                    getattr(result, "confidence", 1.0))),
                        "senses": int(getattr(result, "sense_count", 0)),
                        "source": "query",
                    })
            except Exception:
                pass

        return items[:top_k]

    @staticmethod
    def _parse_event_stream(raw: str) -> list[dict]:
        """Parse a raw RSVS event stream (JSON) into event dicts.

        Args:
            raw: JSON string from consume_events_v1().

        Returns:
            A list of event dicts.
        """
        try:
            events = json.loads(raw)
            if isinstance(events, list):
                return events
            if isinstance(events, dict):
                return [events]
        except json.JSONDecodeError:
            pass
        return []

    # ------------------------------------------------------------------
    # Internal: Fallback helpers (when RSVS is unavailable)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_atomize(text: str) -> list[str]:
        """Simple keyword extraction for fallback mode.

        Splits text into words and returns the "interesting" ones
        (length > 3, not common stop words).

        Args:
            text: Input text to atomize.

        Returns:
            A list of word-level "atoms".
        """
        stop_words = {
            "that", "this", "with", "from", "have", "been", "they",
            "their", "which", "would", "there", "could", "about",
            "other", "into", "more", "than", "then", "some", "very",
            "also", "just", "like", "only", "over", "such", "after",
        }
        words = text.lower().split()
        return [w for w in words if len(w) > 3 and w not in stop_words][:20]

    def _fallback_relevant_context(self, query: str, top_k: int) -> list[dict]:
        """Simple text-search fallback for relevant context.

        Searches through recent messages for mentions of query terms.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            A list of dicts with "label", "relevance", "senses", "source".
        """
        query_words = set(query.lower().split())
        results: list[dict] = []

        for msg in reversed(self._messages):
            content_words = set(msg["content"].lower().split())
            overlap = query_words & content_words
            if overlap:
                for word in overlap:
                    if not any(r["label"] == word for r in results):
                        results.append({
                            "label": word,
                            "relevance": len(overlap) / max(len(query_words), 1),
                            "senses": 1,
                            "source": "fallback_search",
                        })
            if len(results) >= top_k:
                break

        return results[:top_k]

    def _get_recent_topics(self) -> list[str]:
        """Extract topic labels from recent messages.

        Returns:
            A list of concept labels that have been recently discussed.
        """
        topics: list[str] = []

        # If RSVS is available, use the confidence map
        if self.rsvs_available:
            try:
                cmap = self.rsvs.confidence_map()
                sorted_topics = sorted(cmap.items(), key=lambda x: -x[1])
                topics = [label for label, _ in sorted_topics[:10]]
            except Exception:
                pass

        # Fallback: extract from messages
        if not topics and self._messages:
            recent = self._messages[-_DEFAULT_ACTIVE_WINDOW:]
            for msg in recent:
                atoms = self._fallback_atomize(msg["content"])
                for atom in atoms[:3]:
                    if atom not in topics:
                        topics.append(atom)
                        if len(topics) >= 10:
                            break

        return topics

    @staticmethod
    def _normalize_stats(stats: Any) -> dict:
        """Normalize RSVS ingest stats to a plain dict.

        Args:
            stats: The stats object from RSVS.ingest().

        Returns:
            A plain dict representation.
        """
        if isinstance(stats, dict):
            return stats
        try:
            return {
                k: getattr(stats, k)
                for k in dir(stats)
                if not k.startswith("_") and not callable(getattr(stats, k))
            }
        except Exception:
            return {"raw": str(stats)}
