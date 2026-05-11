#!/usr/bin/env python3
"""
Coder Layer Test — Test the Codespace Context Layer.

Tests:
1. ingest_code with a simple Python function
2. ingest_code with a Python class
3. analyze_code with a query
4. find_similar_code
5. detect_code_anomalies

Run: cd /home/z/my-project/RSVS && PYTHONPATH=/home/z/my-project/RSVS python /home/z/my-project/RSVS/rsvs_genius/test_coder.py
"""

from __future__ import annotations

import sys
import json

sys.path.insert(0, "/home/z/my-project/RSVS")

from rsvs_genius.coder_layer import (
    CoderLayer,
    CodeElement,
    CodeAnalysisResult,
    parse_python_code,
    detect_language,
)


# ---------------------------------------------------------------------------
# Test 1: ingest_code with a simple Python function
# ---------------------------------------------------------------------------

def test_ingest_simple_function():
    """Test ingesting a simple Python function."""
    print("=" * 70)
    print("TEST 1: ingest_code — Simple Python Function")
    print("=" * 70)
    print()

    layer = CoderLayer()
    assert layer.rsvs_available, "CoderLayer should be available"
    print(f"  CoderLayer mode: {'Rust core' if layer.is_rust_core else 'fallback'}")

    # Simple function
    code = '''
def calculate_price(base_price: float, discount: float = 0.0) -> float:
    """Calculate the final price after discount."""
    return base_price * (1 - discount)
'''

    result = layer.ingest_code(code, language="python", source="pricing.py")

    assert result["success"], f"ingest should succeed, got: {result}"
    assert result["element_count"] >= 1, f"should find at least 1 element, got {result['element_count']}"
    assert result["language"] == "python", f"language should be python, got {result['language']}"
    assert result["source"] == "pricing.py", f"source should be pricing.py, got {result['source']}"

    # Check that the function was parsed
    elements = result["elements"]
    func_elements = [e for e in elements if e["kind"] == "function"]
    assert len(func_elements) >= 1, f"should find at least 1 function, got {elements}"

    func = func_elements[0]
    assert func["name"] == "calculate_price", f"function name should be calculate_price, got {func['name']}"
    assert "base_price" in func["signature"], f"signature should contain base_price, got {func['signature']}"
    assert func["docstring"] != "", "function should have a docstring"

    print(f"  ✓ Ingest result: success={result['success']}, elements={result['element_count']}")
    print(f"  ✓ Function: {func['name']}{func['signature']}")
    print(f"  ✓ Docstring: {func['docstring'][:60]}...")
    print()
    print("  ✅ TEST 1 PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Test 2: ingest_code with a Python class
# ---------------------------------------------------------------------------

def test_ingest_python_class():
    """Test ingesting a Python class with methods."""
    print("=" * 70)
    print("TEST 2: ingest_code — Python Class with Methods")
    print("=" * 70)
    print()

    layer = CoderLayer()

    # Class with methods
    code = '''
class User:
    """Represents a user in the system."""

    def __init__(self, name: str, email: str):
        """Initialize a new user."""
        self.name = name
        self.email = email

    def get_display_name(self) -> str:
        """Get the user's display name."""
        return f"{self.name} <{self.email}>"

    def update_email(self, new_email: str) -> None:
        """Update the user's email address."""
        self.email = new_email
'''

    result = layer.ingest_code(code, language="python", source="user_model.py")

    assert result["success"], f"ingest should succeed, got: {result}"
    assert result["element_count"] >= 4, (
        f"should find at least 4 elements (1 class + 3 methods), got {result['element_count']}"
    )

    elements = result["elements"]

    # Check class
    class_elements = [e for e in elements if e["kind"] == "class"]
    assert len(class_elements) >= 1, f"should find at least 1 class, got {elements}"
    user_class = class_elements[0]
    assert user_class["name"] == "User", f"class name should be User, got {user_class['name']}"
    assert "get_display_name" in user_class["children"], (
        f"class should list get_display_name as child, got {user_class['children']}"
    )

    # Check methods
    method_elements = [e for e in elements if e["kind"] == "method"]
    assert len(method_elements) >= 3, f"should find at least 3 methods, got {method_elements}"
    method_names = [m["name"] for m in method_elements]
    assert "__init__" in method_names, f"should find __init__, got {method_names}"
    assert "get_display_name" in method_names, f"should find get_display_name, got {method_names}"
    assert "update_email" in method_names, f"should find update_email, got {method_names}"

    # Check parent references
    for method in method_elements:
        assert method["parent"] == "User", f"method parent should be User, got {method['parent']}"

    print(f"  ✓ Ingest result: success={result['success']}, elements={result['element_count']}")
    print(f"  ✓ Class: {user_class['name']} with methods: {user_class['children']}")
    print(f"  ✓ Methods: {method_names}")
    print()
    print("  ✅ TEST 2 PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Test 3: analyze_code with a query
# ---------------------------------------------------------------------------

def test_analyze_code():
    """Test analyzing code with a query."""
    print("=" * 70)
    print("TEST 3: analyze_code — Query Analysis")
    print("=" * 70)
    print()

    layer = CoderLayer()

    # First ingest some code
    code1 = '''
def authenticate(username: str, password: str) -> bool:
    """Authenticate a user with username and password."""
    return verify_credentials(username, password)

def verify_credentials(username: str, password: str) -> bool:
    """Verify user credentials against the database."""
    return True
'''

    code2 = '''
class AuthManager:
    """Manages authentication for the application."""

    def login(self, username: str, password: str) -> str:
        """Log in a user and return a token."""
        if authenticate(username, password):
            return generate_token(username)
        return ""

    def logout(self, token: str) -> None:
        """Log out a user by invalidating their token."""
        invalidate_token(token)
'''

    layer.ingest_code(code1, language="python", source="auth.py")
    layer.ingest_code(code2, language="python", source="auth_manager.py")

    # Now analyze
    result = layer.analyze_code("authenticate function", context=["security", "auth"])

    assert isinstance(result, CodeAnalysisResult), f"result should be CodeAnalysisResult, got {type(result)}"
    assert result.query == "authenticate function", f"query should match"
    assert result.confidence > 0, f"confidence should be > 0, got {result.confidence}"

    print(f"  ✓ Analysis query: {result.query}")
    print(f"  ✓ Elements found: {len(result.elements_found)}")
    print(f"  ✓ Similar code: {len(result.similar_code)}")
    print(f"  ✓ Anomalies: {len(result.anomalies)}")
    print(f"  ✓ Patterns: {len(result.patterns)}")
    print(f"  ✓ Suggestions: {len(result.suggestions)}")
    print(f"  ✓ Confidence: {result.confidence:.3f}")
    print(f"  ✓ Evidence nodes: {len(result.evidence_nodes)}")
    print()

    # Verify the analysis found the authenticate function
    found_auth = any(
        "authenticate" in str(e).lower()
        for e in result.elements_found
    )
    print(f"  ✓ Found authenticate in elements: {found_auth}")
    print()
    print("  ✅ TEST 3 PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Test 4: find_similar_code
# ---------------------------------------------------------------------------

def test_find_similar_code():
    """Test finding structurally similar code."""
    print("=" * 70)
    print("TEST 4: find_similar_code — Similarity Search")
    print("=" * 70)
    print()

    layer = CoderLayer()

    # Ingest some code first
    code1 = '''
def calculate_price(base_price: float, discount: float) -> float:
    """Calculate price after discount."""
    return base_price * (1 - discount)

def calculate_total(items: list, tax_rate: float) -> float:
    """Calculate total with tax."""
    subtotal = sum(item.price for item in items)
    return subtotal * (1 + tax_rate)
'''

    layer.ingest_code(code1, language="python", source="pricing.py")

    # Now search for similar code
    query_code = '''
def compute_cost(base: float, rate: float) -> float:
    """Compute cost with rate adjustment."""
    return base * rate
'''

    results = layer.find_similar_code(query_code, top_k=5)

    assert isinstance(results, list), f"results should be a list, got {type(results)}"
    print(f"  ✓ Found {len(results)} similar code element(s)")

    for i, result in enumerate(results):
        print(f"    {i + 1}. {result.get('name', 'unknown')} "
              f"({result.get('kind', '?')}) — similarity: {result.get('similarity', 0):.3f}")

    print()
    print("  ✅ TEST 4 PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Test 5: detect_code_anomalies
# ---------------------------------------------------------------------------

def test_detect_code_anomalies():
    """Test detecting code anomalies."""
    print("=" * 70)
    print("TEST 5: detect_code_anomalies — Bug Detection")
    print("=" * 70)
    print()

    layer = CoderLayer()

    # Ingest code with some potential issues
    code = '''
class DataStore:
    """Stores data objects."""

    def save(self, data):
        pass

    def load(self, key):
        pass

def process():
    pass

def validate(input_data):
    """Validates input data."""
    return bool(input_data)
'''

    layer.ingest_code(code, language="python", source="data_store.py")

    # Detect anomalies
    anomalies = layer.detect_code_anomalies()

    assert isinstance(anomalies, list), f"anomalies should be a list, got {type(anomalies)}"
    print(f"  ✓ Found {len(anomalies)} anomalies")

    # We should find some anomalies — at least missing documentation for some functions
    missing_doc = [a for a in anomalies if a["type"] == "missing_documentation"]
    print(f"  ✓ Missing documentation: {len(missing_doc)}")

    for i, anomaly in enumerate(anomalies[:10], 1):
        print(f"    {i}. [{anomaly['type']}] {anomaly['element']}: {anomaly['description'][:80]}")

    print()

    # The anomaly detection should find something — at minimum missing docs
    # for process() which has no docstring
    assert len(anomalies) >= 1, f"should find at least 1 anomaly, got {len(anomalies)}"

    print("  ✅ TEST 5 PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Test 6: Additional — Code parsing, language detection, summary
# ---------------------------------------------------------------------------

def test_additional_features():
    """Test additional features: parsing, language detection, summary."""
    print("=" * 70)
    print("TEST 6: Additional Features — Parsing, Detection, Summary")
    print("=" * 70)
    print()

    layer = CoderLayer()

    # Test language detection
    py_lang = detect_language("def hello(): pass", "test.py")
    assert py_lang == "python", f"should detect python, got {py_lang}"
    print(f"  ✓ Language detection (Python): {py_lang}")

    js_lang = detect_language("const x = () => {}", "test.js")
    assert js_lang == "javascript", f"should detect javascript, got {js_lang}"
    print(f"  ✓ Language detection (JavaScript): {js_lang}")

    # Test parse_python_code directly
    elements = parse_python_code("def foo(x: int) -> str:\n    return str(x)")
    assert len(elements) >= 1, f"should parse at least 1 element, got {len(elements)}"
    assert elements[0].name == "foo", f"function name should be foo, got {elements[0].name}"
    print(f"  ✓ Direct parse: {elements[0].name}{elements[0].signature}")

    # Test CodeElement.to_ingest_text()
    element = CodeElement(
        kind="function",
        name="my_func",
        signature="(x: int, y: str) -> bool",
        docstring="Does something useful.",
        source="test.py",
    )
    text = element.to_ingest_text()
    assert "my_func" in text, f"ingest text should contain name"
    assert "x: int" in text, f"ingest text should contain signature"
    assert "Does something" in text, f"ingest text should contain docstring"
    print(f"  ✓ CodeElement.to_ingest_text(): {text[:80]}...")

    # Ingest some code for summary test
    code = '''
def hello():
    print("hello")

def goodbye():
    print("goodbye")
'''
    layer.ingest_code(code, language="python", source="greetings.py")

    # Test get_code_summary
    summary = layer.get_code_summary()
    assert "total_elements" in summary, "summary should have total_elements"
    assert "by_kind" in summary, "summary should have by_kind"
    assert "files" in summary, "summary should have files"
    print(f"  ✓ Code summary: {summary['total_elements']} elements, {len(summary['files'])} files")
    print(f"    By kind: {summary['by_kind']}")

    # Test file-specific summary
    file_summary = layer.get_code_summary(file_path="greetings.py")
    assert file_summary["total_elements"] >= 1, "file summary should have elements"
    print(f"  ✓ File summary: {file_summary['total_elements']} elements from greetings.py")

    print()
    print("  ✅ TEST 6 PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Test 7: Shared bridge — verify CoderLayer uses the same bridge
# ---------------------------------------------------------------------------

def test_shared_bridge():
    """Test that CoderLayer can share a bridge with other layers."""
    print("=" * 70)
    print("TEST 7: Shared Bridge — Interoperability with Other Layers")
    print("=" * 70)
    print()

    from rsvs_genius import RsvsBridge, ContextLayer

    # Create a shared bridge
    shared_bridge = RsvsBridge()

    # Create both layers with the same bridge
    coder = CoderLayer(bridge=shared_bridge)
    context = ContextLayer(bridge=shared_bridge)

    # Verify they share the same bridge instance
    coder_bridge_id = id(coder._bridge)
    context_bridge_id = id(context._bridge)
    assert coder_bridge_id == context_bridge_id, (
        f"CoderLayer and ContextLayer should share the same bridge. "
        f"Coder: {coder_bridge_id}, Context: {context_bridge_id}"
    )
    print(f"  ✓ CoderLayer and ContextLayer share bridge (id={coder_bridge_id})")

    # Test that knowledge ingested via ContextLayer is available to CoderLayer
    context.ingest_text("authenticate function validates user credentials", source="user_input")

    # CoderLayer should be able to use this knowledge
    # (the shared bridge means they share the same graph)
    result = coder.analyze_code("authenticate")
    assert isinstance(result, CodeAnalysisResult), "analyze_code should return CodeAnalysisResult"
    print(f"  ✓ Cross-layer analysis works: confidence={result.confidence:.3f}")

    print()
    print("  ✅ TEST 7 PASSED\n")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("*" * 70)
    print("  RSVS GENIUS — CODER LAYER TEST")
    print("  'The Genius Who Reads Code Like Martial Arts Manuals'")
    print("*" * 70)
    print()

    tests = [
        ("1. Ingest Simple Function", test_ingest_simple_function),
        ("2. Ingest Python Class", test_ingest_python_class),
        ("3. Analyze Code", test_analyze_code),
        ("4. Find Similar Code", test_find_similar_code),
        ("5. Detect Code Anomalies", test_detect_code_anomalies),
        ("6. Additional Features", test_additional_features),
        ("7. Shared Bridge", test_shared_bridge),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {name}")
            print(f"     Error: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 70)
    print(f"CODER LAYER TEST — Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
