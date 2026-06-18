"""AGNN Pipeline End-to-End Benchmark.

Research question:
    Does process() with an AGNN knowledge graph produce better answers
    than a baseline (empty graph, no learned facts)?

Pipeline tested:
    learn(question, wrong, correction) → nodes enter AGNNGraph
    process(question):
        1. agnn_traverse(question) → reasoning chain
        2. Prepend "[Knowledge Graph Context]\n{chain}\nQuestion: {q}" to derivation text
        3. Qwen3-0.6B generates → answer  (or rule-based fallback if model absent)

Usage:
    python self-ai/tests/benchmark_agnn_pipeline.py

Environment variables (optional — override default model paths):
    QWEN_PATH   local directory with Qwen3-0.6B weights (default: HuggingFace cache)
    BGE_PATH    local directory with BAAI/bge-m3 weights  (default: HuggingFace cache)

When models are unavailable the script runs a MockSelfCore dry-run that
verifies script structure and scoring logic, then prints setup instructions.
"""

from __future__ import annotations

import os
import sys
import time
import math
import logging
from typing import Optional

# ── Path setup: must come before any self-ai imports ──────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

logging.basicConfig(level=logging.WARNING)  # Suppress noisy library logs
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Model availability check — exit with clear instructions if needed
# ══════════════════════════════════════════════════════════════════════════════

def _check_model_paths():
    """
    Returns (qwen_available, bge_available).
    Checks env vars first, then HuggingFace cache.
    """
    cache_root = os.path.expanduser('~/.cache/huggingface/hub')

    def _hf_cached(model_name: str) -> bool:
        dir_name = f"models--{model_name.replace('/', '--')}"
        return os.path.isdir(os.path.join(cache_root, dir_name))

    qwen_path = os.environ.get('QWEN_PATH', '')
    bge_path  = os.environ.get('BGE_PATH', '')

    qwen_ok = (bool(qwen_path) and os.path.isdir(qwen_path)) or _hf_cached('Qwen/Qwen3-0.6B')
    bge_ok  = (bool(bge_path)  and os.path.isdir(bge_path))  or _hf_cached('BAAI/bge-m3')

    return qwen_ok, bge_ok


def _print_setup_instructions(qwen_ok: bool, bge_ok: bool):
    print()
    print("=" * 68)
    print("  MODEL SETUP REQUIRED")
    print("=" * 68)
    if not qwen_ok:
        print()
        print("  Qwen3-0.6B NOT FOUND. To download:")
        print()
        print("    pip install huggingface_hub transformers")
        print("    python -c \"from huggingface_hub import snapshot_download; "
              "snapshot_download('Qwen/Qwen3-0.6B')\"")
        print()
        print("  Or set QWEN_PATH to a local directory:")
        print("    export QWEN_PATH=/path/to/Qwen3-0.6B")
    if not bge_ok:
        print()
        print("  BAAI/bge-m3 NOT FOUND. To download:")
        print()
        print("    pip install sentence-transformers")
        print("    python -c \"from sentence_transformers import SentenceTransformer; "
              "SentenceTransformer('BAAI/bge-m3')\"")
        print()
        print("  Or set BGE_PATH to a local directory:")
        print("    export BGE_PATH=/path/to/bge-m3")
    print()
    print("  After downloading, re-run:")
    print("    python self-ai/tests/benchmark_agnn_pipeline.py")
    print("=" * 68)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Knowledge Facts
# ══════════════════════════════════════════════════════════════════════════════

LEARN_FACTS = [
    {
        "question":   "Siapa presiden pertama Indonesia?",
        "wrong":      "Soeharto",
        "correction": "Sukarno adalah presiden pertama Indonesia, menjabat 1945-1967",
    },
    {
        "question":   "Di mana Sukarno lahir?",
        "wrong":      "Jakarta",
        "correction": "Sukarno lahir di Blitar, Jawa Timur",
    },
    {
        "question":   "Apa ibu kota Indonesia?",
        "wrong":      "Surabaya",
        "correction": "Ibu kota Indonesia adalah Jakarta",
    },
    {
        "question":   "Berapa lama Sukarno menjabat?",
        "wrong":      "10 tahun",
        "correction": "Sukarno menjabat selama 22 tahun dari 1945 hingga 1967",
    },
    {
        "question":   "Siapa presiden kedua Indonesia?",
        "wrong":      "Habibie",
        "correction": "Soeharto adalah presiden kedua Indonesia, menjabat 1967-1998",
    },
]

TEST_QUESTIONS = [
    # ── 1-hop: explicitly in graph ────────────────────────────────────────────
    {
        "question": "Siapa presiden pertama Indonesia?",
        "hop_type": "1-hop",
        "expected_keywords": ["sukarno", "pertama"],
        "expected_answer":   "Sukarno adalah presiden pertama Indonesia, menjabat 1945-1967",
    },
    {
        "question": "Di mana Sukarno lahir?",
        "hop_type": "1-hop",
        "expected_keywords": ["blitar", "jawa timur"],
        "expected_answer":   "Sukarno lahir di Blitar, Jawa Timur",
    },
    {
        "question": "Apa ibu kota Indonesia?",
        "hop_type": "1-hop",
        "expected_keywords": ["jakarta", "ibu kota"],
        "expected_answer":   "Ibu kota Indonesia adalah Jakarta",
    },
    {
        "question": "Berapa lama Sukarno menjabat sebagai presiden?",
        "hop_type": "1-hop",
        "expected_keywords": ["22", "1945", "1967"],
        "expected_answer":   "Sukarno menjabat selama 22 tahun dari 1945 hingga 1967",
    },
    {
        "question": "Siapa presiden kedua Indonesia?",
        "hop_type": "1-hop",
        "expected_keywords": ["soeharto", "kedua"],
        "expected_answer":   "Soeharto adalah presiden kedua Indonesia, menjabat 1967-1998",
    },
    # ── 2-hop: requires combining two facts ───────────────────────────────────
    {
        "question": "Sukarno lahir di provinsi apa?",
        "hop_type": "2-hop",
        "expected_keywords": ["jawa timur"],
        "expected_answer":   "Sukarno lahir di Blitar yang berada di Jawa Timur",
    },
    {
        "question": "Presiden pertama Indonesia menjabat sampai tahun berapa?",
        "hop_type": "2-hop",
        "expected_keywords": ["1967", "sukarno"],
        "expected_answer":   "Sukarno sebagai presiden pertama menjabat hingga 1967",
    },
    {
        "question": "Siapa presiden Indonesia sebelum Soeharto?",
        "hop_type": "2-hop",
        "expected_keywords": ["sukarno", "pertama"],
        "expected_answer":   "Sukarno adalah presiden Indonesia sebelum Soeharto",
    },
    # ── Control: tidak ada di graph ───────────────────────────────────────────
    {
        "question": "Siapa penemu telepon?",
        "hop_type": "control",
        "expected_keywords": ["bell", "alexander"],
        "expected_answer":   "Alexander Graham Bell menemukan telepon",
    },
    {
        "question": "Apa ibu kota Jepang?",
        "hop_type": "control",
        "expected_keywords": ["tokyo"],
        "expected_answer":   "Ibu kota Jepang adalah Tokyo",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════════════════════

def _keyword_hit_rate(output: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in output (case-insensitive)."""
    if not keywords:
        return 0.0
    out_lower = output.lower()
    hits = sum(1 for kw in keywords if kw.lower() in out_lower)
    return hits / len(keywords)


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Cosine similarity between two equal-length float lists."""
    import math
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _answer_alignment(output: str, expected: str, embed_fn=None) -> float:
    """
    Cosine similarity between output and expected embeddings via bge-m3.
    Falls back to keyword overlap if embed_fn is None.
    """
    if embed_fn is None:
        # Fallback: Jaccard over word sets
        a = set(output.lower().split())
        b = set(expected.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    try:
        embs = embed_fn([output, expected], show_progress_bar=False, normalize_embeddings=True)
        return float(sum(x * y for x, y in zip(embs[0], embs[1])))
    except Exception:
        return 0.0


def _semantic_score(keyword_hit: float, alignment: float) -> float:
    return 0.6 * keyword_hit + 0.4 * alignment


# ══════════════════════════════════════════════════════════════════════════════
#  Mock SelfCore — for dry-run when real models are absent
# ══════════════════════════════════════════════════════════════════════════════

class MockSelfCore:
    """
    Minimal stand-in for SelfCore when Qwen3-0.6B / bge-m3 are not available.

    Stores corrections in a dict, and produces a very simple answer by
    looking for a stored correction whose question overlaps with the query.
    This verifies script structure without needing real LLM inference.
    """

    def __init__(self):
        self._store: dict[str, str] = {}      # question → correction text
        self._agnn_chains: dict[str, str] = {}
        self._graph_size = 0

    def learn(self, question: str, wrong_answer: str, correction: str) -> dict:
        self._store[question] = correction
        self._graph_size += 1
        # Build a simple mock AGNN chain
        self._agnn_chains[question] = (
            f"[{question}] --CATEGORICAL--> [{correction[:60]}]"
        )
        return {
            'node_id': f'mock_{self._graph_size:03d}',
            'experience': correction,
            'confidence': 0.6,
            'graph_size': self._graph_size,
            'duplicate': False,
        }

    def agnn_traverse(self, query: str, max_hops: int = 2) -> str:
        """Simple keyword-match traversal over stored chains."""
        query_lower = query.lower()
        matched = []
        for q, chain in self._agnn_chains.items():
            if any(w in q.lower() for w in query_lower.split() if len(w) > 3):
                matched.append(chain)
        return " → ".join(matched[:max_hops]) if matched else ""

    def process(self, text: str) -> dict:
        """Answer by looking for stored corrections whose question matches."""
        text_lower = text.lower()
        best_answer = None
        best_score = 0
        for q, correction in self._store.items():
            overlap = sum(1 for w in q.lower().split()
                         if w in text_lower and len(w) > 3)
            if overlap > best_score:
                best_score = overlap
                best_answer = correction
        answer = best_answer or f"[MOCK: no stored fact for '{text[:40]}']"
        return {
            'derivation': {
                'answer': answer,
                'confidence': 0.7 if best_answer else 0.1,
                'method': 'mock_lookup',
            }
        }

    @property
    def _agnn(self):
        class _FakeAGNN:
            def node_count(self_inner):
                return self._graph_size
        return _FakeAGNN()


# ══════════════════════════════════════════════════════════════════════════════
#  Core benchmark logic
# ══════════════════════════════════════════════════════════════════════════════

def _extract_answer(process_result: dict) -> str:
    """Pull the answer string from a process() result dict."""
    deriv = process_result.get('derivation', {})
    if isinstance(deriv, dict):
        answer = deriv.get('answer')
        if answer is not None:
            return str(answer)
    # Some pipelines surface the answer at the top level
    if process_result.get('answer') is not None:
        return str(process_result['answer'])
    return ""


def _score_question(
    question_dict: dict,
    baseline_result: dict,
    agnn_result: dict,
    embed_fn=None,
) -> dict:
    """Compute scores for one question under both conditions."""
    expected   = question_dict['expected_answer']
    keywords   = question_dict['expected_keywords']

    baseline_text = _extract_answer(baseline_result) or ""
    agnn_text     = _extract_answer(agnn_result) or ""

    b_khr = _keyword_hit_rate(baseline_text, keywords)
    a_khr = _keyword_hit_rate(agnn_text, keywords)
    b_aln = _answer_alignment(baseline_text, expected, embed_fn)
    a_aln = _answer_alignment(agnn_text, expected, embed_fn)

    return {
        'question':        question_dict['question'],
        'hop_type':        question_dict['hop_type'],
        'baseline_score':  _semantic_score(b_khr, b_aln),
        'agnn_score':      _semantic_score(a_khr, a_aln),
        'baseline_answer': baseline_text,
        'agnn_answer':     agnn_text,
    }


def run_benchmark(dry_run: bool = False):
    """Execute the full AGNN pipeline benchmark."""

    print()
    print("=" * 68)
    print("  AGNN Pipeline Benchmark")
    print("=" * 68)

    # ── 1. Initialise SelfCore (real or mock) ─────────────────────────────────
    embed_fn = None

    if dry_run:
        print()
        print("  [DRY-RUN MODE] Using MockSelfCore — real models not available.")
        print("  This verifies script structure and scoring logic only.")
        baseline_core = MockSelfCore()
        agnn_core     = MockSelfCore()
    else:
        print()
        print("  Loading models (this may take a while on CPU)...")
        try:
            from core.self import SelfCore
            baseline_core = SelfCore()
            agnn_core     = SelfCore()
        except Exception as exc:
            print(f"\n  ERROR: Failed to initialise SelfCore: {exc}")
            print("  Falling back to dry-run mode.\n")
            dry_run = True
            baseline_core = MockSelfCore()
            agnn_core     = MockSelfCore()

        # Try to get bge-m3 for answer_alignment scoring
        try:
            from derivation.model_registry import get_shared_embedding_model
            model = get_shared_embedding_model()
            if model is not None:
                embed_fn = model.encode
                print("  bge-m3 loaded  ✓")
            else:
                print("  bge-m3 NOT available — answer_alignment uses Jaccard fallback")
        except Exception as exc:
            print(f"  bge-m3 unavailable ({exc}) — using Jaccard fallback")

    # ── 2. Baseline: process() before any learn() calls ──────────────────────
    print()
    print("  Phase 1 — Baseline (empty graph): running process() on all questions...")
    baseline_results = {}
    for tq in TEST_QUESTIONS:
        q = tq['question']
        try:
            result = baseline_core.process(q)
        except Exception as exc:
            result = {'derivation': {'answer': '', 'confidence': 0.0, 'method': f'error:{exc}'}}
        baseline_results[q] = result

    # ── 3. Load knowledge into AGNN core ──────────────────────────────────────
    print()
    print("  Phase 2 — Learning 5 facts via core.learn()...")
    for fact in LEARN_FACTS:
        res = agnn_core.learn(
            question=fact['question'],
            wrong_answer=fact['wrong'],
            correction=fact['correction'],
        )
        graph_size = res.get('graph_size', '?')
        print(f"    ✓ Learned: '{fact['question'][:50]}' (graph_size={graph_size})")

    node_count = agnn_core._agnn.node_count() if agnn_core._agnn is not None else '?'
    print(f"  Graph now has {node_count} AGNN node(s).")

    # ── 4. AGNN: process() after learn() ─────────────────────────────────────
    print()
    print("  Phase 3 — AGNN (5 facts loaded): running process() on all questions...")
    agnn_results   = {}
    agnn_chains    = {}
    for tq in TEST_QUESTIONS:
        q = tq['question']
        # Capture the AGNN chain for reporting
        try:
            chain = agnn_core.agnn_traverse(q.rstrip('?!.,;:'), max_hops=2)
        except Exception:
            chain = ""
        agnn_chains[q] = chain

        try:
            result = agnn_core.process(q)
        except Exception as exc:
            result = {'derivation': {'answer': '', 'confidence': 0.0, 'method': f'error:{exc}'}}
        agnn_results[q] = result

    # ── 5. Score all questions ────────────────────────────────────────────────
    scores = []
    for tq in TEST_QUESTIONS:
        q = tq['question']
        scored = _score_question(
            tq,
            baseline_results[q],
            agnn_results[q],
            embed_fn=embed_fn,
        )
        scores.append(scored)

    # ── 6. Print results table ────────────────────────────────────────────────
    print()
    print("=" * 68)
    mode_tag = "[DRY-RUN / MockSelfCore]" if dry_run else "Qwen3-0.6B"
    print(f"  Model: {mode_tag} | Graph: 5 nodes after learn()")
    print("=" * 68)
    print()

    col_q  = 30
    col_h  = 8
    col_s  = 10

    header = (
        f"  {'Question':<{col_q}}  {'Hops':<{col_h}}  "
        f"{'Baseline':>{col_s}}  {'+ AGNN':>{col_s}}"
    )
    sep    = "  " + "-" * (col_q + col_h + col_s * 2 + 6)
    print(header)
    print(sep)

    for s in scores:
        q_short = s['question'][:col_q]
        print(
            f"  {q_short:<{col_q}}  {s['hop_type']:<{col_h}}  "
            f"{s['baseline_score']:>{col_s}.4f}  {s['agnn_score']:>{col_s}.4f}"
        )

    print(sep)

    # ── 7. Summary statistics ─────────────────────────────────────────────────
    all_baseline = [s['baseline_score'] for s in scores]
    all_agnn     = [s['agnn_score']     for s in scores]

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def _by_hop(hop):
        bs = [s['baseline_score'] for s in scores if s['hop_type'] == hop]
        ag = [s['agnn_score']     for s in scores if s['hop_type'] == hop]
        return _avg(bs), _avg(ag)

    b_avg = _avg(all_baseline)
    a_avg = _avg(all_agnn)
    delta = a_avg - b_avg

    b_1hop, a_1hop = _by_hop('1-hop')
    b_2hop, a_2hop = _by_hop('2-hop')
    b_ctrl, a_ctrl = _by_hop('control')

    print()
    print("  Summary:")
    print(f"    Baseline avg : {b_avg:.4f}")
    print(f"    AGNN avg     : {a_avg:.4f}  (delta: {delta:+.4f})")
    print()
    print(f"    1-hop  avg   : {b_1hop:.4f}  →  {a_1hop:.4f}  (delta: {a_1hop - b_1hop:+.4f})")
    print(f"    2-hop  avg   : {b_2hop:.4f}  →  {a_2hop:.4f}  (delta: {a_2hop - b_2hop:+.4f})")
    print(f"    control avg  : {b_ctrl:.4f}  →  {a_ctrl:.4f}  (delta: {a_ctrl - b_ctrl:+.4f})")

    # ── 8. AGNN chains per question ───────────────────────────────────────────
    print()
    print("  AGNN chain shown for each question (first 200 chars):")
    for tq in TEST_QUESTIONS:
        q = tq['question']
        chain = agnn_chains.get(q, "")
        chain_preview = chain[:200] if chain else "(no chain found)"
        print(f"    Q: \"{q}\"")
        print(f"       chain: {chain_preview}")

    # ── 9. Qualitative analysis ───────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  Analysis")
    print("=" * 68)
    print()

    if delta > 0.05:
        verdict = "YES — AGNN context enrichment clearly helps."
    elif delta > 0.01:
        verdict = "MARGINALLY — small improvement from AGNN context."
    elif abs(delta) <= 0.01:
        verdict = "NEUTRAL — no measurable difference."
    else:
        verdict = "NO — AGNN context did not improve scores in this run."

    print(f"  Does AGNN context improve answers?  {verdict}")
    print()

    hop1_delta = a_1hop - b_1hop
    hop2_delta = a_2hop - b_2hop
    ctrl_delta = a_ctrl - b_ctrl

    print("  Effect by question type:")
    print(f"    1-hop  (explicit facts) : {hop1_delta:+.4f}  — "
          + ("direct retrieval should help most" if hop1_delta > 0.02
             else "limited gain — may already be answerable without graph"))
    print(f"    2-hop  (combined facts) : {hop2_delta:+.4f}  — "
          + ("graph traversal enables implicit reasoning" if hop2_delta > 0.02
             else "multi-hop connection may not surface via keyword-based edges"))
    print(f"    control (off-domain)    : {ctrl_delta:+.4f}  — "
          + ("unexpected influence (check for spurious edges)"
             if abs(ctrl_delta) > 0.05 else "correctly near-zero (expected)"))

    print()
    print("  Architecture observations:")
    print("    - agnn_traverse() uses BFS over keyword-inferred CATEGORICAL edges.")
    print("    - 1-hop gains are expected when question keywords match node labels.")
    print("    - 2-hop gains depend on whether infer_agnn_edges() built")
    print("      the cross-fact edges correctly (e.g., Blitar → Jawa Timur).")
    print("    - Control questions should show ~0 delta (no matching AGNN nodes).")
    print()

    if dry_run:
        print("  NOTE: This was a DRY-RUN with MockSelfCore.")
        print("  To get real LLM-generated scores, install models and re-run:")
        print("    python self-ai/tests/benchmark_agnn_pipeline.py")
        print()

    print("=" * 68)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    qwen_ok, bge_ok = _check_model_paths()
    models_available = qwen_ok and bge_ok

    if not models_available:
        _print_setup_instructions(qwen_ok, bge_ok)
        print("  Running DRY-RUN to verify script structure...")
        print()
        run_benchmark(dry_run=True)
    else:
        run_benchmark(dry_run=False)
