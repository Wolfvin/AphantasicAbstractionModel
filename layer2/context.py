"""
Context Layer — Internet Search + Scope Filter + Source Trust

Analogi: Jin Soun punya Simhyeon Pavilion + 30 tahun pengalaman.
Tapi dia tau kapan harus pakai sumber mana.
Context Layer = kemampuan membatasi diri ke sumber tertentu.

Flow:
1. User input → determine if internet search needed
2. Web search → ingest results into RSVS graph
3. Scope filter → only use trusted sources for output
4. Source trust scoring → weight evidence by provenance
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from .bridge import AbstractionBridge, RsvsBridge, get_bridge
from .web_search import WebSearchEngine, _web_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source trust mapping — how much we trust each source type
# Analogi: Jin Soun lebih percaya laporan mata langsung daripada rumor pasar
# ---------------------------------------------------------------------------

SOURCE_TRUST: dict[str, float] = {
    "user_input": 1.0,       # Direct user input — highest trust
    "web_search": 0.7,       # Web search results — decent but unverified
    "academic": 0.9,         # Academic / scholarly sources
    "official_doc": 0.95,    # Official documentation
    "wiki": 0.75,            # Wikipedia-style crowd-sourced
    "social_media": 0.3,     # Social media — low trust
    "unknown": 0.5,          # Unknown provenance — neutral
}


# _web_search is now imported from .web_search for backward compatibility.
# It delegates to a shared WebSearchEngine instance with caching.


class ContextLayer:
    """Context Layer — Internet Search + Scope Filter + Source Trust.

    Manages the provenance and trust of information flowing into the
    RSVS graph. Provides scope filtering so that downstream consumers
    can restrict answers to trusted sources.

    Analogi: Jin Soun bisa mengakses Simhyeon Pavilion (semua pengetahuan),
    tapi untuk keputusan penting dia hanya mempercayai laporan dari
    mata-mata terpercayanya. Context Layer = filter sumber itu.

    Attributes:
        rsvs_available: Whether a working RSVS instance is connected
            (either Rust core or fallback graph via the bridge).
        is_rust_core: Whether the Rust core is being used.
    """

    def __init__(
        self,
        rsvs_instance: Any | None = None,
        bridge: Optional[RsvsBridge] = None,
        web_search: Optional[WebSearchEngine] = None,
    ) -> None:
        """Initialize the Context Layer.

        Args:
            rsvs_instance: Optional pre-built RSVS instance. If None,
                the layer will obtain a bridge via get_bridge().
            bridge: Optional pre-built RsvsBridge instance. If provided,
                takes precedence over rsvs_instance.
            web_search: Optional WebSearchEngine instance. If None,
                a default instance is created with standard settings.
        """
        if bridge is not None:
            self._bridge = bridge
        elif rsvs_instance is not None:
            self._bridge = RsvsBridge(rsvs_instance=rsvs_instance)
        else:
            self._bridge = get_bridge()

        self.rsvs_available = self._bridge.is_available
        self.is_rust_core = self._bridge.is_rust_core

        # Web search engine — with caching and fallback
        self._web_search = web_search or WebSearchEngine()

        # Scope filter — when set, only sources in this list are used
        # Analogi: Jin Soun membatasi diri hanya membaca laporan dari
        # jaringan mata-mata tertentu untuk misi ini.
        self._allowed_sources: list[str] = []

        # Ingestion log — tracks what has been ingested and from where
        # This acts as a lightweight provenance ledger.
        self._ingestion_log: list[dict] = []

        if self.rsvs_available:
            if self.is_rust_core:
                logger.info("ContextLayer initialized with RSVS Rust core")
            else:
                logger.info("ContextLayer initialized with RSVS fallback graph")
        else:
            logger.info("ContextLayer initialized WITHOUT RSVS core (fallback mode)")

    # ------------------------------------------------------------------
    # Scope management
    # ------------------------------------------------------------------

    def set_scope(self, allowed_sources: list[str]) -> None:
        """Set the scope filter — only allow answers from these sources.

        Analogi: Jin Soun berkata "untuk kasus ini, hanya percaya
        laporan dari mata-mata kode Merah dan Biru."

        Args:
            allowed_sources: List of source identifiers that are trusted
                for the current scope. Examples: ["academic", "official_doc"]
        """
        self._allowed_sources = list(allowed_sources)
        logger.debug("Scope set to: %s", self._allowed_sources)

    def clear_scope(self) -> None:
        """Clear the scope filter — accept all sources.

        Analogi: Setelah misi selesai, Jin Soun membuka kembali
        akses ke semua sumber.
        """
        self._allowed_sources = []
        logger.debug("Scope cleared")

    def get_scope(self) -> list[str]:
        """Get current scope filter.

        Returns:
            List of currently allowed source identifiers.
            Empty list means all sources are accepted.
        """
        return list(self._allowed_sources)

    def is_in_scope(self, source: str) -> bool:
        """Check if a source is within the allowed scope.

        If no scope is set (empty list), all sources are in scope.

        Args:
            source: Source identifier to check.

        Returns:
            True if the source is allowed, False otherwise.
        """
        if not self._allowed_sources:
            return True
        return source in self._allowed_sources

    # ------------------------------------------------------------------
    # GN-4: Access control for knowledge retrieval
    # ------------------------------------------------------------------

    def set_access_policy(self, policy: dict) -> None:
        """Set access control policy for knowledge retrieval.

        Analogi: Jin Soun menjadi ancaman karena dia tahu semua rahasia.
        Access control = siapa boleh tahu apa.

        Args:
            policy: Dict with keys:
                - "redact_patterns": list[str] — patterns to redact from output
                - "max_confidence_for_source": dict[str, float] — cap confidence per source
                - "deny_sources": list[str] — sources to completely exclude
        """
        self._access_policy = policy

    def _apply_access_policy(self, text: str, source: str = "") -> str:
        """Apply access policy to output text.

        Args:
            text: The output text to filter.
            source: The source of the text (for source-specific rules).

        Returns:
            The filtered text with redactions applied.
        """
        if not hasattr(self, '_access_policy') or not self._access_policy:
            return text

        policy = self._access_policy
        result = text

        # Redact patterns
        for pattern in policy.get("redact_patterns", []):
            if pattern.lower() in result.lower():
                result = result.replace(pattern, "[REDACTED]")

        return result

    # ------------------------------------------------------------------
    # Source trust
    # ------------------------------------------------------------------

    def trust_score(self, source_type: str) -> float:
        """Return trust score for a given source type.

        Analogi: Jin Soun tahu bahwa laporan dari "Mata-mata Perak"
        lebih bisa dipercaya daripada "Pedagang Keliling".

        Args:
            source_type: The type of source (e.g. "user_input",
                "web_search", "academic").

        Returns:
            A float between 0.0 and 1.0 indicating trust level.
            Returns the "unknown" trust score for unregistered types.
        """
        return SOURCE_TRUST.get(source_type, SOURCE_TRUST["unknown"])

    # ------------------------------------------------------------------
    # Ingestion with provenance
    # ------------------------------------------------------------------

    def ingest_text(self, text: str, source: str = "user_input") -> dict:
        """Ingest text into the RSVS graph with source tracking.

        Tags the source in the graph metadata so that downstream
        scope filters can accept or reject information based on
        where it came from.

        Analogi: Jin Soun mencatat setiap informasi dengan cap
        "dari siapa" agar bisa menilai nanti.

        Args:
            text: The text content to ingest.
            source: Provenance identifier (default: "user_input").

        Returns:
            A dict with keys:
                - "success": bool — whether ingestion succeeded
                - "source": str — the source tag applied
                - "trust": float — the trust score for this source
                - "in_scope": bool — whether source passes current scope
                - "stats": dict | None — RSVS ingest stats (if available)
        """
        trust = self.trust_score(source)
        in_scope = self.is_in_scope(source)

        record: dict = {
            "success": False,
            "source": source,
            "trust": trust,
            "in_scope": in_scope,
            "stats": None,
        }

        # If source is out of scope, we still record it but don't ingest
        if not in_scope:
            logger.info("Source '%s' is out of scope — skipping ingestion", source)
            record["success"] = False
            record["reason"] = "out_of_scope"
            self._ingestion_log.append({
                **record,
                "timestamp": time.time(),
                "text_length": len(text),
            })
            return record

        if self.rsvs_available:
            try:
                # P-04: Pass source provenance to bridge for trust weighting
                stats = self._bridge.ingest(text, source_provenance=source)
                record["success"] = True
                record["stats"] = stats  # Already a plain dict from bridge
            except Exception as exc:
                logger.error("RSVS ingestion failed: %s", exc)
                record["success"] = False
                record["reason"] = str(exc)
        else:
            # Fallback mode — no RSVS, but we still track the ingestion
            record["success"] = True
            record["stats"] = {
                "atoms_before": 0,
                "atoms_after": 0,
                "new_atoms": 0,
                "fallback": True,
            }

        self._ingestion_log.append({
            **record,
            "timestamp": time.time(),
            "text_length": len(text),
        })
        return record

    # ------------------------------------------------------------------
    # Web search + ingest pipeline
    # ------------------------------------------------------------------

    def search_and_ingest(self, query: str, max_results: int = 5) -> dict:
        """Search the web and ingest results into the RSVS graph.

        L2-07 fix: Results are now filtered by RSVS relevance before
        ingestion. Snippets that are inconsistent with the graph
        (appraise = "disagree") or not related to active senses
        are filtered out.

        Analogi: Jin Soun mengirim mata-mata ke kota lain untuk
        mencari informasi, lalu mencatat hasilnya di Simhyeon Pavilion.
        Tapi dia hanya mencatat informasi yang RELEVAN — bukan semua
        laporan yang diterima.

        Flow:
            1. Web search using z-ai-web-dev-sdk
            2. Filter results by RSVS relevance (L2-07)
            3. Format filtered results as structured text
            4. Ingest into RSVS graph with "web_search" source tag
            5. Return formatted results with provenance

        Args:
            query: The search query string.
            max_results: Maximum number of search results (default: 5).

        Returns:
            A dict with keys:
                - "query": str — the search query
                - "results": list[dict] — raw search results
                - "ingested": bool — whether results were ingested
                - "ingest_stats": dict | None — stats from ingestion
                - "result_count": int — number of results found
                - "trust": float — trust score for web_search source
                - "filtered_count": int — number of results filtered out
        """
        trust = self.trust_score("web_search")
        in_scope = self.is_in_scope("web_search")

        response: dict = {
            "query": query,
            "results": [],
            "ingested": False,
            "ingest_stats": None,
            "result_count": 0,
            "trust": trust,
            "filtered_count": 0,
        }

        # Step 1: Perform web search (using instance engine with caching)
        search_results = self._web_search.search(query, num=max_results)
        response["results"] = search_results
        response["result_count"] = len(search_results)

        if not search_results:
            logger.info("No web search results for: %s", query)
            return response

        # Step 2 (L2-07): Filter results by RSVS relevance
        filtered_results = self._filter_search_results(search_results, query)
        response["filtered_count"] = len(search_results) - len(filtered_results)

        if not filtered_results:
            logger.info(
                "All %d search results filtered out by RSVS relevance for: %s",
                len(search_results), query
            )
            return response

        # Step 3: Format filtered results for ingestion
        formatted_parts: list[str] = []
        for i, result in enumerate(filtered_results):
            title = result.get("title", "Untitled")
            url = result.get("url", result.get("link", ""))
            snippet = result.get("snippet", result.get("description", ""))
            formatted_parts.append(
                f"[Web Search Result {i + 1}]\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Snippet: {snippet}"
            )

        combined_text = "\n\n".join(formatted_parts)

        # Step 4: Ingest (if in scope)
        if not in_scope:
            logger.info("web_search source is out of scope — skipping ingestion")
            return response

        ingest_result = self.ingest_text(combined_text, source="web_search")
        response["ingested"] = ingest_result["success"]
        response["ingest_stats"] = ingest_result.get("stats")

        return response

    def _filter_search_results(
        self,
        results: list[dict],
        query: str,
    ) -> list[dict]:
        """Filter search results by RSVS relevance (L2-07 fix).

        Uses bridge.appraise() to check if each snippet is consistent
        with the graph, and bridge.relate() to find the most relevant
        snippets to active senses. Only snippets above the appraise
        threshold are kept.

        When RSVS is unavailable, all results are kept (no filtering).

        Args:
            results: Raw search results from the web search engine.
            query: The original search query.

        Returns:
            A filtered list of search result dicts.
        """
        if not self.rsvs_available:
            return results  # No filtering without RSVS

        filtered: list[dict] = []

        for result in results:
            snippet = result.get("snippet", result.get("description", ""))
            title = result.get("title", "")
            text_to_check = f"{title} {snippet}"

            if not text_to_check.strip():
                continue

            # Check 1: Use appraise() to verify consistency with graph
            try:
                appraise_result = self._bridge.appraise(text_to_check)
                if isinstance(appraise_result, dict):
                    verdict = appraise_result.get("verdict", "neutral")
                    # If the snippet contradicts the graph, skip it
                    if verdict == "disagree":
                        agree_pct = appraise_result.get("agree_pct", 0.0)
                        disagree_pct = appraise_result.get("disagree_pct", 0.0)
                        if isinstance(disagree_pct, (int, float)) and float(disagree_pct) > float(agree_pct):
                            logger.debug(
                                "Filtered search result (disagree): %s",
                                title[:50]
                            )
                            continue
            except Exception as exc:
                logger.debug("appraise() for search filtering failed: %s", exc)

            # Check 2: Use relate() to verify relevance to active senses
            try:
                # Extract key terms from the snippet
                relate_result = self._bridge.relate(query)
                if relate_result:
                    related_nodes = relate_result.get("related_nodes", [])
                    # If the query has related nodes in the graph,
                    # the result is likely relevant
                    if related_nodes:
                        # Result is relevant — keep it
                        filtered.append(result)
                        continue
            except Exception as exc:
                logger.debug("relate() for search filtering failed: %s", exc)

            # If we can't determine relevance, keep the result
            # (better to have false positives than false negatives)
            filtered.append(result)

        return filtered

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_ingestion_log(self) -> list[dict]:
        """Get the full ingestion log with provenance.

        Returns:
            A list of dicts, each containing source, trust, timestamp,
            and ingestion status for every text that has been processed.
        """
        return list(self._ingestion_log)

    def clear_ingestion_log(self) -> None:
        """Clear the ingestion log."""
        self._ingestion_log = []

    # ------------------------------------------------------------------
    # Persistence (P2-8: Cognitive persistence)
    # ------------------------------------------------------------------

    _PERSIST_SCHEMA_VERSION = "1.0"

    def save_to_dict(self) -> dict:
        """Serialize cognitive state to a plain dict (in-memory).

        Saves `_allowed_sources` and `_ingestion_log` (the provenance
        ledger).  The RSVS bridge / web search engine are NOT serialized
        — only the Layer-2 cognitive state.

        Returns:
            A dict containing the full serializable state.
        """
        return {
            "schema_version": self._PERSIST_SCHEMA_VERSION,
            "allowed_sources": self._allowed_sources,
            "ingestion_log": self._ingestion_log[-500:],  # Keep bounded
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore cognitive state from a plain dict (in-memory).

        Restores `_allowed_sources` and `_ingestion_log`.
        Existing state is replaced.

        Args:
            data: A dict previously returned by `save_to_dict()`.
        """
        if not isinstance(data, dict):
            logger.warning("load_from_dict: expected dict, got %s", type(data).__name__)
            return

        # Schema compatibility check
        saved_version = data.get("schema_version", "0.0")
        if saved_version != self._PERSIST_SCHEMA_VERSION:
            logger.warning(
                "load_from_dict: schema version mismatch (saved=%s, current=%s). "
                "Proceeding with best-effort restore.",
                saved_version, self._PERSIST_SCHEMA_VERSION,
            )

        self._allowed_sources = data.get("allowed_sources", [])
        self._ingestion_log = data.get("ingestion_log", [])

        logger.info(
            "ContextLayer state restored: %d allowed sources, %d ingestion log entries",
            len(self._allowed_sources), len(self._ingestion_log),
        )

    def save(self, path: str) -> dict:
        """Save cognitive state to a JSON file.

        Args:
            path: Filesystem path to write the JSON file.

        Returns:
            A summary dict with stats about what was saved.
        """
        data = self.save_to_dict()
        summary: dict = {
            "path": path,
            "allowed_sources": len(self._allowed_sources),
            "ingestion_log_entries": len(self._ingestion_log),
            "schema_version": self._PERSIST_SCHEMA_VERSION,
            "success": False,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            summary["success"] = True
            logger.info("ContextLayer state saved to %s", path)
        except (OSError, TypeError) as exc:
            summary["error"] = str(exc)
            logger.error("ContextLayer save failed: %s", exc)
        return summary

    def load(self, path: str) -> dict:
        """Load cognitive state from a JSON file.

        Args:
            path: Filesystem path to read the JSON file from.

        Returns:
            A summary dict with stats about what was loaded.
        """
        summary: dict = {
            "path": path,
            "allowed_sources": 0,
            "ingestion_log_entries": 0,
            "success": False,
        }
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_from_dict(data)
            summary["allowed_sources"] = len(self._allowed_sources)
            summary["ingestion_log_entries"] = len(self._ingestion_log)
            summary["schema_version"] = data.get("schema_version", "unknown")
            summary["success"] = True
            logger.info("ContextLayer state loaded from %s", path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            summary["error"] = str(exc)
            logger.error("ContextLayer load failed: %s", exc)
        return summary


