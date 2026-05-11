---
Task ID: sec+test
Agent: security-and-tests-updater
Task: Update SECURITY.md and expand test coverage

Work Log:
- Updated SECURITY.md with v6.1 features (inactivity TTL, cycle detection, traversal safety, xxhash)
- Added /context-query endpoint tests (7 tests)
- Added compositional architecture contract tests (6 tests)
- Added version v6.1.0 check tests (2 tests)

Stage Summary:
- SECURITY.md fully synced with v6.1
- Test coverage expanded from ~45 tests to ~60 tests
- All new endpoints and features have test coverage

---
Task ID: 4
Agent: general-purpose
Task: Implement Rule-based Policy Engine for Tax/Regulation

Work Log:
- Read all existing rsvs_genius layer files (context_layer, situation_layer, predictive_engine, pattern_output, pipeline) and bridge module to understand patterns
- Created `/home/z/my-project/RSVS/rsvs_genius/policy_engine.py` (~880 lines) with:
  - `PolicyRule` dataclass: rule_id, domain, description, condition (callable or string expr), severity, reference, tags
  - `PolicyViolation` dataclass: rule_id, description, severity, evidence, suggestion, reference
  - `PolicyEngine` class with full compliance checking pipeline:
    - `add_rule()` / `add_rules_from_dict()` — rule management + RSVS graph ingestion
    - `check_compliance()` — two-phase: RSVS relate() finds applicable rules, then deterministic evaluation
    - `check_single_rule()` — evaluates single rule condition against context
    - `get_applicable_rules()` — uses RSVS graph to find relevant rules via relate()/query()
    - `get_compliance_report()` — full audit report with violations, warnings, passed, overall_status
    - `load_tax_rules_indonesia()` — 15 real Indonesian tax rules (PPh 21, PPh 23, PPN, BPJS)
    - Context parser: extracts key=value, Indonesian income notation, boolean flags, tax rates
    - Safe eval for string expression conditions
- Created `/home/z/my-project/RSVS/rsvs_genius/test_policy.py` with 12 test suites, 80 assertions:
  - Test 1: add_rule and check_single_rule (9 assertions)
  - Test 2: check_compliance with multiple rules (5)
  - Test 3: load_tax_rules_indonesia (11)
  - Test 4: get_compliance_report with compliant/non-compliant scenarios (11)
  - Test 5: RSVS graph finds applicable rules via relate() (4)
  - Test 6: String expression conditions (5)
  - Test 7: Callable conditions with complex logic (3)
  - Test 8: Context parsing (9)
  - Test 9: Violation tracking and clearing (4)
  - Test 10: Serialization (8)
  - Test 11: add_rules_from_dict batch operation (6)
  - Test 12: Edge cases and error handling (5)
- Updated `/home/z/my-project/RSVS/rsvs_genius/__init__.py`:
  - Added PolicyEngine, PolicyRule, PolicyViolation to exports
  - Updated layer list in docstring to include PolicyEngine as layer 6
- All 80 tests pass ✓

Key Design Decisions:
1. PolicyEngine uses the shared RsvsBridge pattern like all other layers
2. Rules are stored both in-memory (dict) and ingested into RSVS graph for semantic search
3. Two-phase compliance: RSVS finds WHICH rules apply (probabilistic), then deterministic evaluation checks IF satisfied
4. String expression conditions use restricted eval with __builtins__={} for safety
5. Context parsing supports Indonesian notation ("penghasilan 80 juta", "sudah potong PPh 21")
6. Tax rules are based on actual Indonesian tax law (UU PPh, UU PPN, UU BPJS, PP 44/2015, PMK 122)
7. Violations from check_single_rule are also tracked in internal history
