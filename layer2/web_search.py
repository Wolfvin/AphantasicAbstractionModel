"""
Web Search Module — z-ai-web-dev-sdk backend with caching and fallback.

Provides the WebSearchEngine class that:
1. Tries the z-ai-web-dev-sdk directly (fast, no subprocess)
2. Falls back to a subprocess-based Node.js call
3. Caches results with configurable TTL to avoid redundant API calls
4. Uses JSON-based stdin/stdout for safe subprocess communication

The z-ai-web-dev-sdk returns results with keys:
    url, name, snippet, host_name, rank, date, favicon

For backward compatibility, results are normalized to also include
'title' (mapped from 'name') and 'link' (mapped from 'url').
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node.js script for subprocess fallback — reads JSON from stdin, writes to stdout
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = r"""
const ZAI = require('z-ai-web-dev-sdk').default;
(async () => {
  const chunks = [];
  process.stdin.on('data', (c) => chunks.push(c));
  process.stdin.on('end', async () => {
    try {
      const {query, num} = JSON.parse(Buffer.concat(chunks).toString());
      const zai = await ZAI.create();
      const r = await zai.functions.invoke("web_search", {query, num});
      process.stdout.write(JSON.stringify(r));
    } catch (e) {
      process.stderr.write(e.message || String(e));
      process.exit(1);
    }
  });
})();
"""


def _normalize_result(raw: dict) -> dict:
    """Normalize a raw SDK result dict to a canonical format.

    The SDK returns: url, name, snippet, host_name, rank, date, favicon
    We ensure backward-compatible keys: title (from name), link (from url).
    Also ensures title and name are always both present by cross-mapping.
    """
    normalized = dict(raw)

    # Establish name ↔ title bidirectional mapping
    # If neither exists, default to empty string
    if "title" not in normalized and "name" in normalized:
        normalized["title"] = normalized["name"]
    elif "name" not in normalized and "title" in normalized:
        normalized["name"] = normalized["title"]
    elif "name" not in normalized and "title" not in normalized:
        normalized["name"] = ""
        normalized["title"] = ""

    # Establish url ↔ link bidirectional mapping
    if "link" not in normalized and "url" in normalized:
        normalized["link"] = normalized["url"]
    elif "url" not in normalized and "link" in normalized:
        normalized["url"] = normalized["link"]
    elif "url" not in normalized and "link" not in normalized:
        normalized["url"] = ""
        normalized["link"] = ""

    # Ensure snippet exists
    normalized.setdefault("snippet", normalized.get("description", ""))

    return normalized


class WebSearchEngine:
    """Web search engine with caching and graceful fallback.

    Tries the z-ai-web-dev-sdk directly first (via subprocess with JSON
    stdin/stdout for safe communication), then falls back to a simpler
    subprocess call. Results are cached with a configurable TTL.

    Args:
        cache_ttl: Time-to-live in seconds for cached search results.
            Defaults to 300 (5 minutes).
    """

    def __init__(self, cache_ttl: int = 300) -> None:
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        logger.debug("WebSearchEngine initialized (cache_ttl=%ds)", cache_ttl)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, num: int = 5) -> list[dict]:
        """Search the web, returning cached results when available.

        Args:
            query: Search query string.
            num: Maximum number of results to return.

        Returns:
            A list of search result dicts with keys: url, name, snippet,
            title, link (plus any additional keys from the SDK).
        """
        # Check cache first
        cache_key = f"{query}:{num}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.info("Web search cache hit for: %s", query)
            return cached

        # Try SDK-based search first, then subprocess fallback
        results = self._search_via_sdk(query, num)
        if results is None:
            logger.info("SDK search failed, trying subprocess fallback for: %s", query)
            results = self._search_via_subprocess(query, num)

        if results is None:
            results = []

        # Normalize results
        results = [_normalize_result(r) for r in results]

        # Store in cache
        self._set_cached(cache_key, results)
        logger.info("Web search returned %d results for: %s", len(results), query)
        return results

    def clear_cache(self) -> None:
        """Clear the entire search cache."""
        self._cache.clear()
        logger.debug("Web search cache cleared")

    # ------------------------------------------------------------------
    # SDK-based search (subprocess with JSON stdin/stdout)
    # ------------------------------------------------------------------

    def _search_via_sdk(self, query: str, num: int) -> Optional[list[dict]]:
        """Search using z-ai-web-dev-sdk via subprocess with JSON communication.

        Uses stdin/stdout JSON instead of string interpolation for safe
        handling of complex queries.

        Returns:
            List of result dicts, or None if search failed.
        """
        try:
            payload = json.dumps({"query": query, "num": num})
            result = subprocess.run(
                ["node", "-e", _SUBPROCESS_SCRIPT],
                input=payload,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                parsed = json.loads(result.stdout)
                if isinstance(parsed, list):
                    return parsed
                logger.warning("SDK returned non-list: %s", type(parsed).__name__)
                return None
            if result.stderr.strip():
                logger.debug("SDK search stderr: %s", result.stderr.strip()[:200])
        except FileNotFoundError:
            logger.debug("Node.js not found — SDK search unavailable")
        except subprocess.TimeoutExpired:
            logger.warning("SDK search timed out for query: %s", query)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse SDK search results: %s", exc)
        except Exception as exc:
            logger.warning("SDK search failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Subprocess fallback (JSON stdin/stdout — safe communication)
    # ------------------------------------------------------------------

    def _search_via_subprocess(self, query: str, num: int) -> Optional[list[dict]]:
        """Fallback search using JSON stdin/stdout for safe communication.

        Uses the same _SUBPROCESS_SCRIPT as _search_via_sdk but as a
        secondary fallback when the primary SDK path fails. This avoids
        the command-injection risk of string-interpolated shell args.

        Returns:
            List of result dicts, or None if search failed.
        """
        try:
            payload = json.dumps({"query": query, "num": num})
            result = subprocess.run(
                ["node", "-e", _SUBPROCESS_SCRIPT],
                input=payload,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                parsed = json.loads(result.stdout)
                if isinstance(parsed, list):
                    return parsed
                logger.warning("Fallback returned non-list: %s", type(parsed).__name__)
            if result.stderr.strip():
                logger.debug("Fallback search stderr: %s", result.stderr.strip()[:200])
        except FileNotFoundError:
            logger.debug("Node.js not found — fallback search unavailable")
        except subprocess.TimeoutExpired:
            logger.warning("Fallback search timed out for query: %s", query)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse fallback search results: %s", exc)
        except Exception as exc:
            logger.warning("Fallback search failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_cached(self, key: str) -> Optional[list[dict]]:
        """Get cached results if they haven't expired."""
        if key in self._cache:
            timestamp, results = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return results
            # Expired — remove it
            del self._cache[key]
        return None

    def _set_cached(self, key: str, results: list[dict]) -> None:
        """Store results in cache with current timestamp."""
        self._cache[key] = (time.time(), results)


# ---------------------------------------------------------------------------
# Module-level convenience function (backward compatibility)
# ---------------------------------------------------------------------------

_shared_engine: Optional[WebSearchEngine] = None


def _get_shared_engine() -> WebSearchEngine:
    """Get or create the shared WebSearchEngine instance."""
    global _shared_engine
    if _shared_engine is None:
        _shared_engine = WebSearchEngine()
    return _shared_engine


def _web_search(query: str, num: int = 5) -> list[dict]:
    """Module-level convenience function for backward compatibility.

    Delegates to a shared WebSearchEngine instance.

    Args:
        query: Search query string.
        num: Maximum number of results to return.

    Returns:
        A list of search result dicts.
    """
    return _get_shared_engine().search(query, num=num)
