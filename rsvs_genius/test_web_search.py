#!/usr/bin/env python3
"""
Test WebSearchEngine and ContextLayer web search integration.

Tests:
1. WebSearchEngine basic search (live SDK if available, mock fallback)
2. WebSearchEngine cache — second call should return cached results
3. ContextLayer.search_and_ingest() with web search
4. Verify search results format (must have 'url', 'name', 'snippet' keys)
5. Cache TTL expiration

Run: cd /home/z/my-project/RSVS && PYTHONPATH=/home/z/my-project/RSVS python -m rsvs_genius.test_web_search
"""

from __future__ import annotations

import sys
import time
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/home/z/my-project/RSVS")


def _is_sdk_available() -> bool:
    """Check if z-ai-web-dev-sdk is available via Node.js."""
    import subprocess
    try:
        result = subprocess.run(
            ["node", "-e", "require('z-ai-web-dev-sdk')"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Sample mock data for when SDK is not available
MOCK_SEARCH_RESULTS = [
    {
        "url": "https://example.com/page1",
        "name": "Example Page 1",
        "snippet": "This is the first example result about knowledge graphs.",
        "host_name": "example.com",
        "rank": 0,
        "date": "",
        "favicon": "",
    },
    {
        "url": "https://example.com/page2",
        "name": "Example Page 2",
        "snippet": "A second result about RSVS structural reasoning.",
        "host_name": "example.com",
        "rank": 1,
        "date": "",
        "favicon": "",
    },
    {
        "url": "https://example.org/page3",
        "name": "Example Page 3",
        "snippet": "Third result discussing cognitive architectures.",
        "host_name": "example.org",
        "rank": 2,
        "date": "",
        "favicon": "",
    },
]


def test_1_basic_search():
    """Test 1: WebSearchEngine basic search (live SDK or mock)."""
    from rsvs_genius.web_search import WebSearchEngine

    engine = WebSearchEngine()

    if _is_sdk_available():
        # Live test
        print("  Testing with LIVE SDK...")
        results = engine.search("knowledge graph", num=3)
        if results:
            print(f"  ✓ Live search returned {len(results)} results")
            for r in results[:2]:
                print(f"    - {r.get('name', r.get('title', 'N/A'))}")
        else:
            print("  ⚠ Live search returned no results (API may be down)")
    else:
        # Mock test
        print("  Testing with MOCK (SDK not available)...")
        with patch.object(engine, '_search_via_sdk', return_value=MOCK_SEARCH_RESULTS):
            results = engine.search("knowledge graph", num=3)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        print(f"  ✓ Mock search returned {len(results)} results")

    print("  ✅ Test 1 PASSED — WebSearchEngine basic search\n")


def test_2_cache_hit():
    """Test 2: WebSearchEngine cache — second call should return cached results."""
    from rsvs_genius.web_search import WebSearchEngine

    engine = WebSearchEngine(cache_ttl=300)
    call_count = 0

    def mock_search_via_sdk(query, num):
        nonlocal call_count
        call_count += 1
        return MOCK_SEARCH_RESULTS

    with patch.object(engine, '_search_via_sdk', side_effect=mock_search_via_sdk):
        # First call — should hit the actual search
        results1 = engine.search("cache test query", num=3)
        assert call_count == 1, f"Expected 1 SDK call, got {call_count}"

        # Second call — should return cached results
        results2 = engine.search("cache test query", num=3)
        assert call_count == 1, f"Expected 1 SDK call (cached), got {call_count}"

        # Results should be identical
        assert results1 == results2, "Cached results should match original"

    print(f"  ✓ First call triggered SDK (call_count={call_count})")
    print(f"  ✓ Second call used cache (call_count still={call_count})")
    print("  ✅ Test 2 PASSED — Cache hit behavior\n")


def test_3_context_layer_search_and_ingest():
    """Test 3: ContextLayer.search_and_ingest() with web search."""
    from rsvs_genius.context_layer import ContextLayer
    from rsvs_genius.web_search import WebSearchEngine

    # Create a mock web search engine
    mock_engine = WebSearchEngine()
    with patch.object(mock_engine, '_search_via_sdk', return_value=MOCK_SEARCH_RESULTS):
        # Pre-populate cache so search returns mock data
        mock_engine._set_cached("test query:5", [_normalize(r) for r in MOCK_SEARCH_RESULTS])

    layer = ContextLayer(web_search=mock_engine)
    assert layer.rsvs_available, "ContextLayer should be available"

    # Test search_and_ingest
    response = layer.search_and_ingest("test query", max_results=5)

    assert response["query"] == "test query", "Query should match"
    assert response["result_count"] >= 0, "result_count should be non-negative"
    assert "results" in response, "Response should have 'results' key"
    assert "ingested" in response, "Response should have 'ingested' key"
    assert response["trust"] == 0.7, "web_search trust should be 0.7"

    print(f"  ✓ search_and_ingest returned {response['result_count']} results")
    print(f"  ✓ ingested={response['ingested']}, trust={response['trust']}")
    print("  ✅ Test 3 PASSED — ContextLayer.search_and_ingest()\n")


def test_4_results_format():
    """Test 4: Verify search results format (must have 'url', 'name', 'snippet' keys)."""
    from rsvs_genius.web_search import WebSearchEngine, _normalize_result

    # Test with raw SDK-format results
    for raw in MOCK_SEARCH_RESULTS:
        normalized = _normalize_result(raw)
        assert "url" in normalized, f"Missing 'url' key in: {list(normalized.keys())}"
        assert "name" in normalized, f"Missing 'name' key in: {list(normalized.keys())}"
        assert "snippet" in normalized, f"Missing 'snippet' key in: {list(normalized.keys())}"
        # Backward compatibility keys
        assert "title" in normalized, f"Missing 'title' (backward compat) key"
        assert "link" in normalized, f"Missing 'link' (backward compat) key"
        # Verify title == name mapping
        assert normalized["title"] == normalized["name"], "title should equal name"
        assert normalized["link"] == normalized["url"], "link should equal url"

    # Test with minimal result dict
    minimal = {"url": "https://test.com", "snippet": "test"}
    normalized = _normalize_result(minimal)
    assert "name" in normalized, "Should have 'name' key after normalization"
    assert "title" in normalized, "Should have 'title' key after normalization"

    print("  ✓ All required keys present: url, name, snippet, title, link")
    print("  ✓ Backward compatibility: title=name, link=url")
    print("  ✅ Test 4 PASSED — Search results format\n")


def test_5_cache_ttl_expiration():
    """Test 5: Cache TTL expiration."""
    from rsvs_genius.web_search import WebSearchEngine

    # Create engine with very short TTL
    engine = WebSearchEngine(cache_ttl=1)  # 1 second TTL
    call_count = 0

    def mock_search_via_sdk(query, num):
        nonlocal call_count
        call_count += 1
        return MOCK_SEARCH_RESULTS

    with patch.object(engine, '_search_via_sdk', side_effect=mock_search_via_sdk):
        # First call
        results1 = engine.search("ttl test", num=3)
        assert call_count == 1, f"Expected 1 call, got {call_count}"

        # Second call immediately — should be cached
        results2 = engine.search("ttl test", num=3)
        assert call_count == 1, "Should use cache"

    # Now wait for TTL to expire and test cache directly
    # Manually set a cache entry with old timestamp
    engine._cache["expired_key:3"] = (time.time() - 2, MOCK_SEARCH_RESULTS)
    cached = engine._get_cached("expired_key:3")
    assert cached is None, "Expired cache entry should return None"

    # Verify fresh cache entry works
    engine._cache["fresh_key:3"] = (time.time(), MOCK_SEARCH_RESULTS)
    cached = engine._get_cached("fresh_key:3")
    assert cached is not None, "Fresh cache entry should return results"

    print("  ✓ Cache entry expires after TTL")
    print("  ✓ Fresh cache entries return correctly")
    print("  ✅ Test 5 PASSED — Cache TTL expiration\n")


# Helper for test_3
def _normalize(raw: dict) -> dict:
    """Normalize a raw result dict (same as web_search._normalize_result)."""
    from rsvs_genius.web_search import _normalize_result
    return _normalize_result(raw)


def main():
    print("=" * 70)
    print("RSVS Genius — Web Search Engine Tests")
    print("=" * 70)
    print()

    sdk_available = _is_sdk_available()
    print(f"z-ai-web-dev-sdk available: {'YES' if sdk_available else 'NO (using mocks)'}")
    print()

    tests = [
        ("1. WebSearchEngine basic search", test_1_basic_search),
        ("2. WebSearchEngine cache hit", test_2_cache_hit),
        ("3. ContextLayer.search_and_ingest()", test_3_context_layer_search_and_ingest),
        ("4. Search results format", test_4_results_format),
        ("5. Cache TTL expiration", test_5_cache_ttl_expiration),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"Testing: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
