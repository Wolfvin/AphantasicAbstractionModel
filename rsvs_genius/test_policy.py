"""
Test Policy Engine — Comprehensive test suite

Tests for:
1. add_rule and check_single_rule
2. check_compliance with a tax scenario
3. load_tax_rules_indonesia
4. get_compliance_report
5. RSVS graph correctly finds applicable rules
6. String expression conditions
7. Callable conditions
8. Context parsing
"""

from __future__ import annotations

import sys

# Ensure rsvs_genius is importable
sys.path.insert(0, "/home/z/my-project/RSVS")

from rsvs_genius.policy_engine import PolicyEngine, PolicyRule, PolicyViolation
from rsvs_genius.rsvs_bridge import RsvsBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def assert_test(condition: bool, name: str, detail: str = "") -> None:
    """Simple assertion helper without unittest dependency."""
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        extra = f" — {detail}" if detail else ""
        print(f"  ✗ {name}{extra}")


def make_engine() -> PolicyEngine:
    """Create a fresh PolicyEngine with a fresh bridge."""
    bridge = RsvsBridge()  # Will use fallback graph
    return PolicyEngine(bridge=bridge)


# ---------------------------------------------------------------------------
# Test 1: add_rule and check_single_rule
# ---------------------------------------------------------------------------

def test_add_rule_and_check_single():
    """Test adding a rule and checking it against context."""
    print("\n=== Test 1: add_rule and check_single_rule ===")
    engine = make_engine()

    # Add a simple rule with callable condition
    rule = PolicyRule(
        rule_id="TEST_001",
        domain="test",
        description="Income above threshold requires tax withholding",
        condition=lambda ctx: ctx.get("income", 0) <= 50_000_000 or ctx.get("tax_withheld", False),
        severity="critical",
        reference="Test Regulation 001",
        tags=["income", "tax"],
    )
    engine.add_rule(rule)

    # Verify rule was added
    retrieved = engine.get_rule("TEST_001")
    assert_test(retrieved is not None, "Rule was added and retrievable")
    assert_test(retrieved.rule_id == "TEST_001", "Rule ID matches")
    assert_test(retrieved.domain == "test", "Rule domain matches")

    # Check compliant case: income below threshold
    violation = engine.check_single_rule("TEST_001", "income=30000000")
    assert_test(violation is None, "No violation when income below threshold")

    # Check non-compliant case: income above threshold, no withholding
    violation = engine.check_single_rule("TEST_001", "income=80000000")
    assert_test(violation is not None, "Violation detected when income above threshold")
    if violation:
        assert_test(violation.rule_id == "TEST_001", "Violation rule_id matches")
        assert_test(violation.severity == "critical", "Violation severity is critical")
        assert_test(violation.reference == "Test Regulation 001", "Violation reference matches")

    # Check compliant case: income above threshold but withholding done
    violation = engine.check_single_rule("TEST_001", "income=80000000 tax_withheld=True")
    assert_test(violation is None, "No violation when tax is withheld")


# ---------------------------------------------------------------------------
# Test 2: check_compliance with a tax scenario
# ---------------------------------------------------------------------------

def test_check_compliance():
    """Test compliance checking with multiple rules."""
    print("\n=== Test 2: check_compliance ===")
    engine = make_engine()

    # Add multiple rules
    rules = [
        PolicyRule(
            rule_id="COMP_001",
            domain="tax_pph21",
            description="PPh 21 must be withheld for employees",
            condition=lambda ctx: ctx.get("pph21_withheld", False) or ctx.get("employment_type", "") != "employee",
            severity="critical",
            reference="UU PPh 21",
            tags=["pph21", "employment"],
        ),
        PolicyRule(
            rule_id="COMP_002",
            domain="tax_ppn",
            description="PPN must be collected for taxable supplies",
            condition=lambda ctx: ctx.get("ppn_collected", False) or ctx.get("supply_type", "") == "exempt",
            severity="warning",
            reference="UU PPN",
            tags=["ppn", "supply"],
        ),
        PolicyRule(
            rule_id="COMP_003",
            domain="general",
            description="Annual tax return must be filed",
            condition=lambda ctx: ctx.get("spt_filed", False),
            severity="critical",
            reference="UU KUP",
            tags=["spt", "reporting"],
        ),
    ]
    for r in rules:
        engine.add_rule(r)

    # Check compliance with a scenario that triggers violations
    result = engine.check_compliance("tax_pph21 employee with income, pph21_withheld=False, spt_filed=False")

    assert_test(result["total_rules_checked"] > 0, "Some rules were checked")
    assert_test(isinstance(result["violations"], list), "Violations is a list")
    assert_test(isinstance(result["warnings"], list), "Warnings is a list")
    assert_test(isinstance(result["passed"], list), "Passed is a list")

    # We should have at least some violations or warnings
    total_issues = len(result["violations"]) + len(result["warnings"]) + len(result["info"])
    assert_test(total_issues > 0, f"Issues found ({total_issues}) > 0")


# ---------------------------------------------------------------------------
# Test 3: load_tax_rules_indonesia
# ---------------------------------------------------------------------------

def test_load_tax_rules_indonesia():
    """Test loading Indonesian tax rules preset."""
    print("\n=== Test 3: load_tax_rules_indonesia ===")
    engine = make_engine()

    count = engine.load_tax_rules_indonesia()
    assert_test(count > 0, f"Rules loaded: {count} > 0")
    assert_test(count >= 15, f"At least 15 rules loaded (got {count})")

    # Verify specific rules exist
    pph21_rule = engine.get_rule("TAX_PPH21_001")
    assert_test(pph21_rule is not None, "TAX_PPH21_001 exists")
    if pph21_rule:
        assert_test(pph21_rule.domain == "tax_pph21", "PPh21 rule domain is correct")
        assert_test(pph21_rule.severity == "critical", "PPh21 rule is critical")

    ppn_rule = engine.get_rule("TAX_PPN_001")
    assert_test(ppn_rule is not None, "TAX_PPN_001 exists")

    bpjs_rule = engine.get_rule("TAX_BPJS_001")
    assert_test(bpjs_rule is not None, "TAX_BPJS_001 exists")

    # List rules by domain
    pph21_rules = engine.list_rules(domain="tax_pph21")
    assert_test(len(pph21_rules) > 0, f"PPh21 rules found: {len(pph21_rules)}")

    pph23_rules = engine.list_rules(domain="tax_pph23")
    assert_test(len(pph23_rules) > 0, f"PPh23 rules found: {len(pph23_rules)}")

    ppn_rules = engine.list_rules(domain="tax_ppn")
    assert_test(len(ppn_rules) > 0, f"PPN rules found: {len(ppn_rules)}")

    bpjs_rules = engine.list_rules(domain="tax_bpjs")
    assert_test(len(bpjs_rules) > 0, f"BPJS rules found: {len(bpjs_rules)}")


# ---------------------------------------------------------------------------
# Test 4: get_compliance_report
# ---------------------------------------------------------------------------

def test_get_compliance_report():
    """Test generating a full compliance report."""
    print("\n=== Test 4: get_compliance_report ===")
    engine = make_engine()
    engine.load_tax_rules_indonesia()

    # Scenario: employee with high income, PPh 21 not withheld
    report = engine.get_compliance_report(
        "Karyawan penghasilan 80 juta, belum potong PPh 21, "
        "pph21_withheld=False, spt_filed=False"
    )

    assert_test("context" in report, "Report has 'context' key")
    assert_test("timestamp" in report, "Report has 'timestamp' key")
    assert_test("total_rules_checked" in report, "Report has 'total_rules_checked' key")
    assert_test("violations" in report, "Report has 'violations' key")
    assert_test("warnings" in report, "Report has 'warnings' key")
    assert_test("passed" in report, "Report has 'passed' key")
    assert_test("overall_status" in report, "Report has 'overall_status' key")
    assert_test("summary" in report, "Report has 'summary' key")

    assert_test(report["overall_status"] in ("compliant", "non_compliant", "warning"),
                f"Overall status is valid: {report['overall_status']}")

    # With pph21_withheld=False and spt_filed=False, should be non-compliant
    assert_test(report["overall_status"] in ("non_compliant", "warning"),
                f"Status is non_compliant or warning: {report['overall_status']}")

    print(f"  Report summary: {report['summary']}")
    print(f"  Overall status: {report['overall_status']}")
    print(f"  Violations: {len(report['violations'])}")
    print(f"  Warnings: {len(report['warnings'])}")
    print(f"  Passed: {len(report['passed'])}")

    # Test compliant scenario
    engine2 = make_engine()
    engine2.load_tax_rules_indonesia()

    compliant_report = engine2.get_compliance_report(
        "income=30000000 pph21_withheld=True spt_filed=True "
        "ppn_rate=0.11 pkp_registered=True tax_invoice_issued=True "
        "bpjs_health_employee_rate=0.01 bpjs_health_employer_rate=0.04 "
        "bpjs_jht_employee_rate=0.02 bpjs_jht_employer_rate=0.037 "
        "bpjs_jkp_employer_rate=0.01 "
        "pph23_withheld=True service_type=other "
        "allowance_taxed=True daily_allowance=300000"
    )
    assert_test(compliant_report["overall_status"] == "compliant",
                f"Fully compliant scenario: {compliant_report['overall_status']}")


# ---------------------------------------------------------------------------
# Test 5: RSVS graph finds applicable rules
# ---------------------------------------------------------------------------

def test_rsvs_finds_applicable_rules():
    """Test that RSVS graph correctly finds applicable rules via relate()."""
    print("\n=== Test 5: RSVS graph finds applicable rules ===")
    engine = make_engine()
    engine.load_tax_rules_indonesia()

    # When we check compliance for PPh 21 context,
    # the PPh 21 rules should be in the applicable set
    applicable = engine.get_applicable_rules("pajak penghasilan karyawan PPh 21", top_k=10)
    assert_test(len(applicable) > 0, f"Applicable rules found: {len(applicable)}")

    # At least one PPh 21 rule should be in the results
    pph21_found = any(r.domain == "tax_pph21" for r in applicable)
    assert_test(pph21_found, "At least one PPh 21 rule is applicable")

    # When context is about PPN, PPN rules should be found
    applicable_ppn = engine.get_applicable_rules("PPN pajak pertambahan nilai", top_k=10)
    ppn_found = any(r.domain == "tax_ppn" for r in applicable_ppn)
    assert_test(ppn_found, "PPN rules found for PPN context")

    # When context is about BPJS, BPJS rules should be found
    applicable_bpjs = engine.get_applicable_rules("BPJS kesehatan ketenagakerjaan", top_k=10)
    bpjs_found = any(r.domain == "tax_bpjs" for r in applicable_bpjs)
    assert_test(bpjs_found, "BPJS rules found for BPJS context")


# ---------------------------------------------------------------------------
# Test 6: String expression conditions
# ---------------------------------------------------------------------------

def test_string_expression_conditions():
    """Test rules with string expression conditions."""
    print("\n=== Test 6: String expression conditions ===")
    engine = make_engine()

    # Add rule with string expression
    rule = PolicyRule(
        rule_id="STR_001",
        domain="test",
        description="Annual return must be filed",
        condition="spt_filed == True",
        severity="critical",
        reference="Test Reg",
    )
    engine.add_rule(rule)

    # Test with spt_filed=True
    violation = engine.check_single_rule("STR_001", "spt_filed=True")
    assert_test(violation is None, "String condition passes when spt_filed=True")

    # Test with spt_filed=False
    violation = engine.check_single_rule("STR_001", "spt_filed=False")
    assert_test(violation is not None, "String condition fails when spt_filed=False")
    if violation:
        assert_test(violation.severity == "critical", "Violation severity from string rule")

    # Test with another string expression
    rule2 = PolicyRule(
        rule_id="STR_002",
        domain="test",
        description="PPN rate must be 11%",
        condition="ppn_rate == 0.11",
        severity="warning",
        reference="UU HPP",
    )
    engine.add_rule(rule2)

    violation = engine.check_single_rule("STR_002", "ppn_rate=0.11")
    assert_test(violation is None, "String condition passes when ppn_rate=0.11")

    violation = engine.check_single_rule("STR_002", "ppn_rate=0.10")
    assert_test(violation is not None, "String condition fails when ppn_rate=0.10")


# ---------------------------------------------------------------------------
# Test 7: Callable conditions with complex logic
# ---------------------------------------------------------------------------

def test_callable_conditions():
    """Test rules with callable conditions and complex logic."""
    print("\n=== Test 7: Callable conditions with complex logic ===")
    engine = make_engine()

    # Rule that checks multiple fields
    rule = PolicyRule(
        rule_id="CALL_001",
        domain="tax_bpjs",
        description="BPJS Health rates must be within limits",
        condition=lambda ctx: (
            ctx.get("bpjs_health_employee_rate", 0) <= 0.01
            and ctx.get("bpjs_health_employer_rate", 0) <= 0.04
        ),
        severity="critical",
        reference="UU BPJS",
    )
    engine.add_rule(rule)

    # Compliant: correct rates
    violation = engine.check_single_rule("CALL_001", "bpjs_health_employee_rate=0.01 bpjs_health_employer_rate=0.04")
    assert_test(violation is None, "Callable condition passes with correct rates")

    # Non-compliant: employee rate too high
    violation = engine.check_single_rule("CALL_001", "bpjs_health_employee_rate=0.02 bpjs_health_employer_rate=0.04")
    assert_test(violation is not None, "Callable condition fails with excessive employee rate")

    # Non-compliant: employer rate too high
    violation = engine.check_single_rule("CALL_001", "bpjs_health_employee_rate=0.01 bpjs_health_employer_rate=0.05")
    assert_test(violation is not None, "Callable condition fails with excessive employer rate")


# ---------------------------------------------------------------------------
# Test 8: Context parsing
# ---------------------------------------------------------------------------

def test_context_parsing():
    """Test context string parsing into structured data."""
    print("\n=== Test 8: Context parsing ===")

    # Test key=value parsing
    ctx = PolicyEngine._parse_context("income=80000000 pph21_withheld=True")
    assert_test(ctx.get("income") == 80000000, f"Numeric value parsed: {ctx.get('income')}")
    assert_test(ctx.get("pph21_withheld") is True, f"Boolean value parsed: {ctx.get('pph21_withheld')}")

    # Test Indonesian income parsing
    ctx = PolicyEngine._parse_context("penghasilan 80 juta")
    assert_test(ctx.get("income") == 80_000_000, f"Indonesian income parsed: {ctx.get('income')}")

    # Test boolean flag extraction
    ctx = PolicyEngine._parse_context("sudah potong PPh 21")
    assert_test(ctx.get("pph21_withheld") is True, "Boolean flag 'pph21_withheld' extracted")

    ctx = PolicyEngine._parse_context("sudah lapor SPT")
    assert_test(ctx.get("spt_filed") is True, "Boolean flag 'spt_filed' extracted")

    # Test rate parsing
    ctx = PolicyEngine._parse_context("tarif ppn 11%")
    assert_test(ctx.get("ppn_rate") == 0.11, f"PPN rate parsed: {ctx.get('ppn_rate')}")

    # Test revenue parsing
    ctx = PolicyEngine._parse_context("revenue 5 miliar")
    assert_test(ctx.get("annual_revenue") == 5_000_000_000, f"Revenue parsed: {ctx.get('annual_revenue')}")

    # Test PKP registration flag
    ctx = PolicyEngine._parse_context("sudah terdaftar PKP")
    assert_test(ctx.get("pkp_registered") is True, "PKP registration flag extracted")

    # Test empty context
    ctx = PolicyEngine._parse_context("")
    assert_test(isinstance(ctx, dict), "Empty context returns empty dict")


# ---------------------------------------------------------------------------
# Test 9: Violation tracking and clearing
# ---------------------------------------------------------------------------

def test_violation_tracking():
    """Test violation history tracking and clearing."""
    print("\n=== Test 9: Violation tracking ===")
    engine = make_engine()

    rule = PolicyRule(
        rule_id="TRACK_001",
        domain="test",
        description="Simple rule for tracking test",
        condition=lambda ctx: ctx.get("valid", False),
        severity="warning",
    )
    engine.add_rule(rule)

    # Initially no violations
    assert_test(len(engine.get_violations()) == 0, "No violations initially")

    # Check and generate violation
    engine.check_single_rule("TRACK_001", "valid=False")
    assert_test(len(engine.get_violations()) == 1, "One violation after check")

    # Check again (another violation)
    engine.check_single_rule("TRACK_001", "valid=False")
    assert_test(len(engine.get_violations()) == 2, "Two violations after second check")

    # Clear violations
    engine.clear_violations()
    assert_test(len(engine.get_violations()) == 0, "Violations cleared")


# ---------------------------------------------------------------------------
# Test 10: PolicyRule and PolicyViolation serialization
# ---------------------------------------------------------------------------

def test_serialization():
    """Test to_dict() serialization for PolicyRule and PolicyViolation."""
    print("\n=== Test 10: Serialization ===")

    rule = PolicyRule(
        rule_id="SER_001",
        domain="test",
        description="Serialization test rule",
        condition=lambda ctx: True,
        severity="info",
        reference="Test Ref",
        tags=["test", "serialization"],
    )
    rule_dict = rule.to_dict()
    assert_test(rule_dict["rule_id"] == "SER_001", "Rule dict rule_id matches")
    assert_test(rule_dict["condition_type"] == "callable", "Rule dict condition_type is callable")
    assert_test(rule_dict["severity"] == "info", "Rule dict severity matches")
    assert_test("test" in rule_dict["tags"], "Rule dict tags included")

    # String condition type
    rule2 = PolicyRule(
        rule_id="SER_002",
        domain="test",
        description="String condition rule",
        condition="x > 5",
        severity="warning",
    )
    rule2_dict = rule2.to_dict()
    assert_test(rule2_dict["condition_type"] == "expression", "String condition type is expression")

    violation = PolicyViolation(
        rule_id="SER_001",
        description="Test violation",
        severity="critical",
        evidence=["evidence1", "evidence2"],
        suggestion="Fix it",
        reference="Test Ref",
    )
    v_dict = violation.to_dict()
    assert_test(v_dict["rule_id"] == "SER_001", "Violation dict rule_id matches")
    assert_test(len(v_dict["evidence"]) == 2, "Violation dict evidence count matches")
    assert_test(v_dict["suggestion"] == "Fix it", "Violation dict suggestion matches")


# ---------------------------------------------------------------------------
# Test 11: add_rules_from_dict batch operation
# ---------------------------------------------------------------------------

def test_add_rules_from_dict():
    """Test batch adding rules from dict format."""
    print("\n=== Test 11: add_rules_from_dict ===")
    engine = make_engine()

    rules_data = [
        {
            "rule_id": "BATCH_001",
            "domain": "test",
            "description": "Batch rule 1",
            "condition": lambda ctx: ctx.get("ok", False),
            "severity": "warning",
            "reference": "Batch Ref 1",
            "tags": ["batch", "test"],
        },
        {
            "rule_id": "BATCH_002",
            "domain": "test",
            "description": "Batch rule 2",
            "condition": "value > 100",
            "severity": "info",
            "reference": "Batch Ref 2",
        },
        {
            "rule_id": "BATCH_003",
            "domain": "other",
            "description": "Batch rule 3",
            "condition": lambda ctx: True,
            "severity": "critical",
        },
    ]

    count = engine.add_rules_from_dict(rules_data)
    assert_test(count == 3, f"All 3 rules added (count={count})")

    # Verify each rule
    assert_test(engine.get_rule("BATCH_001") is not None, "BATCH_001 exists")
    assert_test(engine.get_rule("BATCH_002") is not None, "BATCH_002 exists")
    assert_test(engine.get_rule("BATCH_003") is not None, "BATCH_003 exists")

    # Test domain filtering
    test_rules = engine.list_rules(domain="test")
    assert_test(len(test_rules) == 2, f"2 test domain rules (got {len(test_rules)})")


# ---------------------------------------------------------------------------
# Test 12: Edge cases
# ---------------------------------------------------------------------------

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Test 12: Edge cases ===")
    engine = make_engine()

    # Check non-existent rule
    violation = engine.check_single_rule("NONEXISTENT", "test context")
    assert_test(violation is None, "Non-existent rule returns None")

    # Check with no rules loaded
    applicable = engine.get_applicable_rules("some context")
    assert_test(isinstance(applicable, list), "Empty engine returns list")
    assert_test(len(applicable) == 0, "Empty engine returns empty list")

    # Check rule with missing context keys
    rule = PolicyRule(
        rule_id="EDGE_001",
        domain="test",
        description="Rule that checks missing key",
        condition=lambda ctx: ctx.get("missing_key", "default") == "expected",
        severity="info",
    )
    engine.add_rule(rule)

    # Default value should prevent crash
    violation = engine.check_single_rule("EDGE_001", "no relevant keys here")
    assert_test(violation is not None, "Rule with missing context key handled gracefully")

    # Add rule with bad string expression (should catch error)
    bad_rule = PolicyRule(
        rule_id="EDGE_002",
        domain="test",
        description="Rule with invalid expression",
        condition="undefined_var > 5",  # Will fail in eval
        severity="critical",
    )
    engine.add_rule(bad_rule)

    # This should NOT crash — should return violation with error description
    violation = engine.check_single_rule("EDGE_002", "some context")
    assert_test(violation is not None, "Bad expression handled without crash")
    if violation:
        assert_test("error" in violation.description.lower() or "evaluation" in violation.description.lower(),
                     "Bad expression violation mentions error")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    """Run all tests."""
    global _passed, _failed

    print("=" * 60)
    print("Policy Engine Tests")
    print("=" * 60)

    test_add_rule_and_check_single()
    test_check_compliance()
    test_load_tax_rules_indonesia()
    test_get_compliance_report()
    test_rsvs_finds_applicable_rules()
    test_string_expression_conditions()
    test_callable_conditions()
    test_context_parsing()
    test_violation_tracking()
    test_serialization()
    test_add_rules_from_dict()
    test_edge_cases()

    print("\n" + "=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed, {_passed + _failed} total")
    print("=" * 60)

    if _failed > 0:
        sys.exit(1)
    else:
        print("All tests passed! ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
