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

import logging
import time
from typing import Any, Optional

from .bridge import RsvsBridge, get_bridge
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
                stats = self._bridge.ingest(text)
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

        Analogi: Jin Soun mengirim mata-mata ke kota lain untuk
        mencari informasi, lalu mencatat hasilnya di Simhyeon Pavilion.

        Flow:
            1. Web search using z-ai-web-dev-sdk
            2. Format results as structured text
            3. Ingest into RSVS graph with "web_search" source tag
            4. Return formatted results with provenance

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
        }

        # Step 1: Perform web search (using instance engine with caching)
        search_results = self._web_search.search(query, num=max_results)
        response["results"] = search_results
        response["result_count"] = len(search_results)

        if not search_results:
            logger.info("No web search results for: %s", query)
            return response

        # Step 2: Format results for ingestion
        # Analogi: Mata-mata kembali dengan laporan mentah —
        # Jin Soun merangkumnya sebelum menyimpan di arsip.
        formatted_parts: list[str] = []
        for i, result in enumerate(search_results):
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

        # Step 3: Ingest (if in scope)
        if not in_scope:
            logger.info("web_search source is out of scope — skipping ingestion")
            return response

        ingest_result = self.ingest_text(combined_text, source="web_search")
        response["ingested"] = ingest_result["success"]
        response["ingest_stats"] = ingest_result.get("stats")

        return response

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


