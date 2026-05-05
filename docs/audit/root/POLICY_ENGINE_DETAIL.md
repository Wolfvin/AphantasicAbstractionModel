# RSVS Policy Engine Final Plan (V4.2)

## 1) Objective
Policy Engine adalah satu-satunya owner untuk field governance node:
- `tier`
- `confidence`
- `status`
- `is_seed`
- `is_locked`

Ingest/parser hanya membuat kandidat evidence. Keputusan governance harus deterministik, dapat diaudit, dan tahan terhadap input teks masif/noisy.

## 2) Interfaces
Input:
1. `candidate_node`
2. `policy_context`
3. `graph_state`
4. `evidence_batch`

Output:
- `PolicyDecision` (accept)
- `PolicyReject` (fail-fast)

Language wire assumption (active reconstruction):
- `surface_label` primary identity uses `<surface>@<lang>`.
- `language_links.same_as` is primary cross-language relation.
- `sense_state` is optional/secondary, not required as language owner.

```python
class PolicyDecision(TypedDict):
    tier: int
    confidence: float
    status: str
    is_seed: bool
    is_locked: bool
    decision_reason: str
    policy_version: str
    applied_rules: list[str]

class PolicyReject(TypedDict):
    error_code: str
    violated_rule: str
    reason: str
    policy_version: str
```

Error codes:
- `policy_invalid_input`
- `policy_conflict`
- `invariant_violation`
- `transition_not_allowed`
- `seed_unlock_forbidden`
- `policy_version_mismatch`

## 3) Decision Policy
### 3.1 Precedence (mandatory)
1. `InvariantGuard`
2. `SeedRule`
3. `ManualGovernanceRule`
4. `PromotionRule`
5. `DefaultRule`

Jika conflict, prioritas tertinggi menang.

### 3.2 Hard invariants
- `is_seed=true => is_locked=true`
- `is_seed=true => tier=1`
- `is_seed=true => confidence=1.0`
- `is_seed=true => status=stable`

Pelanggaran invariant wajib reject.

### 3.3 Status state machine
Valid statuses:
- `new`
- `candidate`
- `stable`
- `deprecated`
- `quarantine`

Allowed transitions:
- `new -> candidate`
- `candidate -> stable`
- `candidate -> quarantine`
- `stable -> deprecated`
- `candidate -> deprecated`
- `quarantine -> candidate` (setelah recovery rule)

Forbidden transitions:
- `deprecated -> stable`
- `new -> stable` tanpa memenuhi threshold

### 3.4 Confidence model
Gunakan score terkalibrasi:

`governance_score = f(evidence_strength, source_trust, recency, contradiction_penalty)`

Where:
- `evidence_strength`: frequency/coherence/diversity/consistency
- `source_trust`: bobot sumber
- `recency`: decay terhadap evidence lama
- `contradiction_penalty`: penalti untuk evidence yang bertentangan

Default mapping:
- `confidence = governance_score` untuk non-seed
- seed selalu `confidence=1.0`

### 3.5 Tier mapping
- `0.00 - 0.39 => tier=3`
- `0.40 - 0.79 => tier=2`
- `0.80 - 1.00 => tier=1` (non-seed tetap tunduk promotion rule)

### 3.6 Promotion logic (final)
- `promotion_threshold = 0.75`
- Tidak ada hard cap jumlah promote per batch.

Rules:
- `confidence >= 0.75` => boleh promote sesuai state machine.
- `< 0.75` => tetap `candidate`, evidence diakumulasi lintas batch.
- Confidence naik hanya dari evidence baru (dedup gate aktif).

Hysteresis (anti flip-flop):
- Promote threshold: `>= 0.75`
- Demote-to-candidate threshold: `< 0.60`

### 3.7 Lock policy
- Seed lock immutable (tidak bisa unlock).
- Non-seed lock/unlock hanya via manual governance action + audit reason.

## 4) High-Volume Safeguards
1. `SourceTrustWeighting` (wajib)
- `trusted_seed/governance_manual > verified_runtime > user_raw > unknown_external`.

2. `TemporalWindowing` (wajib)
- `short_window=7d`, `long_window=30d`.
- Promote ke stable butuh konsistensi dua window.

3. `ConflictBudget + Quarantine` (wajib)
- Jika `status_flip_count` melebihi budget, node masuk `quarantine`.

4. `IdempotentDedupGate` (wajib)
- Replay fingerprint tidak boleh inflate score linear.

## 5) Audit, Migration, and Regression Guards
Audit fields wajib:
- `policy_version`
- `decision_reason`
- `applied_rules`
- `previous_state`
- `new_state`
- `timestamp`
- `correlation_id`

Policy migration:
- Simpan jejak `previous_policy_version -> new_policy_version`.
- Re-evaluate node non-seed ketika policy naik versi.
- Audit history append-only.

Regression guard checklist:
1. Seed invariants selalu lolos.
2. Transition invalid selalu reject.
3. Replay duplicate tidak menaikkan confidence penuh.
4. Low-trust evidence tidak bisa sendirian promote stable.
5. Quarantine node tidak boleh dipromote sebelum recovery.
6. Hysteresis mencegah flip di sekitar threshold.
7. Same input + same state + same policy_version => same decision.

## 6) Implementation Notes for Current Runtime
- `_build_snapshot_and_events` hanya membentuk kandidat dan raw evidence.
- Semua governance fields harus ditetapkan oleh `PolicyEngine.evaluate(...)`.
- Penulisan snapshot/artifact dilakukan setelah:
  1. policy decision,
  2. invariant validator,
  3. audit record write.
