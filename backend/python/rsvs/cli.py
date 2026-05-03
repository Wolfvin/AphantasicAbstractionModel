#!/usr/bin/env python3
"""
RSVS CLI — v5.0.0

Usage:
  rsvs init [--db PATH]
  rsvs ingest <text_or_file> [--db PATH] [--domain INT]
  rsvs query <concept> <context> [--db PATH] [--top INT]
  rsvs similarity <a> <b> [--db PATH]
  rsvs status [--db PATH]
  rsvs atoms [--db PATH] [--seeds]
  rsvs senses <concept> [--db PATH]
  rsvs info <atom> [--db PATH]
  rsvs ingest-corpus --domains <d1> [<d2> ...] [--db PATH]
  rsvs eval [--db PATH] [--domains <d1> ...] [--json-out PATH]
  rsvs replay-events [--db PATH] [--after-seq N] [--limit N]

Default DB path: ./rsvs.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

DEFAULT_DB    = "rsvs.json"
VERSION       = "5.0.0"
BANNER        = f"RSVS v{VERSION} — Relational Symbolic Vocabulary System"

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _load_rsvs(db_path: str):
    """Load from disk, or create fresh if not found."""
    from rsvs import Rsvs
    p = Path(db_path)
    if p.exists():
        try:
            r = Rsvs.load(str(p))
            return r
        except Exception as e:
            _err(f"Failed to load {db_path}: {e}")
    return Rsvs()

def _save_rsvs(r, db_path: str):
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    r.save(str(p))

def _err(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)

def _ok(msg: str):
    print(msg)

def _json_out(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def _is_file(s: str) -> bool:
    return os.path.isfile(s)

def _read_input(text_or_file: str) -> str:
    """If argument is a file path, read it. Otherwise treat as literal text."""
    if _is_file(text_or_file):
        with open(text_or_file, encoding="utf-8") as f:
            return f.read()
    return text_or_file

# -----------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------

def cmd_init(args):
    """Create a fresh RSVS database."""
    db = args.db
    p  = Path(db)
    if p.exists() and not args.force:
        _err(f"{db} already exists. Use --force to overwrite.")

    from rsvs import Rsvs
    r = Rsvs(
        entity_promote_n=args.promote_n,
        theta_assign=args.theta,
        n_warm=args.n_warm,
        eta=args.eta,
    )
    _save_rsvs(r, db)
    st = r.status()
    _ok(f"✓ Initialized {db}")
    _ok(f"  seed atoms:  {int(st['total_atoms'])}")
    _ok(f"  n_warm:      {args.n_warm}")
    _ok(f"  promote_n:   {args.promote_n}")


def cmd_ingest(args):
    """Ingest text into the knowledge graph."""
    text = _read_input(args.text_or_file)
    if not text.strip():
        _err("Input text is empty.")

    r = _load_rsvs(args.db)

    if args.domain:
        r.set_domain(args.domain)

    stats = r.ingest(text)
    _save_rsvs(r, args.db)

    if args.json:
        _json_out({
            "sentences_processed": stats.sentences_processed,
            "atoms_promoted":      stats.atoms_promoted,
            "senses_created":      stats.sense_created,
            "confidence_updated":  stats.confidence_updated,
            "frozen_batches":      stats.frozen_batches,
            "db":                  args.db,
        })
    else:
        _ok(f"✓ Ingested {stats.sentences_processed} sentence(s)")
        if stats.atoms_promoted:
            _ok(f"  atoms promoted:    {stats.atoms_promoted}")
        if stats.sense_created:
            _ok(f"  senses created:    {stats.sense_created}")
        _ok(f"  confidence updated:{stats.confidence_updated}")
        if stats.frozen_batches:
            _ok(f"  ⚠ frozen batches:  {stats.frozen_batches}")
        _ok(f"  saved → {args.db}")


def cmd_query(args):
    """Context-aware query for a concept."""
    r      = _load_rsvs(args.db)
    result = r.query(args.concept, args.context)

    if result is None:
        if args.json:
            _json_out({"error": f"concept '{args.concept}' not found"})
            sys.exit(1)
        else:
            _err(f"Concept '{args.concept}' not found. Ingest more text first.")

    top_k = args.top
    atoms = result.atoms[:top_k]

    if args.json:
        _json_out({
            "concept":    args.concept,
            "context":    args.context,
            "sense_idx":  result.sense_idx,
            "sense_n":    result.sense_n,
            "atoms":      [{"label": l, "score": round(s, 4)} for l, s in atoms],
        })
    else:
        _ok(f"query: {args.concept!r} | context: {args.context!r}")
        _ok(f"  active sense: #{result.sense_idx}  (N={result.sense_n} contexts)")
        _ok(f"  top atoms:")
        for label, score in atoms:
            bar = "█" * int(score * 20)
            _ok(f"    {label:<16} {score:.3f}  {bar}")


def cmd_similarity(args):
    """Compute similarity between two concepts."""
    r   = _load_rsvs(args.db)
    sim = r.similarity(args.a, args.b)

    if sim is None:
        if args.json:
            _json_out({"error": f"one or both concepts not found"})
        else:
            _err(f"One or both concepts not found: '{args.a}', '{args.b}'")

    if args.json:
        _json_out({
            "a":       args.a,
            "b":       args.b,
            "jaccard": round(sim.jaccard, 4),
            "shared":  sim.shared,
            "only_a":  sim.only_a,
            "only_b":  sim.only_b,
        })
    else:
        bar = "█" * int(sim.jaccard * 40)
        _ok(f"sim({args.a!r}, {args.b!r})")
        _ok(f"  jaccard: {sim.jaccard:.3f}  {bar}")
        if sim.shared:
            _ok(f"  shared:  {sim.shared}")
        if sim.only_a:
            _ok(f"  only {args.a}: {sim.only_a}")
        if sim.only_b:
            _ok(f"  only {args.b}: {sim.only_b}")


def cmd_status(args):
    """Show system status."""
    db = args.db
    p  = Path(db)
    if not p.exists():
        if args.json:
            _json_out({"error": f"{db} not found"})
        else:
            _err(f"{db} not found. Run 'rsvs init' first.")

    r  = _load_rsvs(db)
    st = r.status()
    db_size = p.stat().st_size

    if args.json:
        _json_out({
            "db":             db,
            "db_size_bytes":  db_size,
            "total_nodes":    int(st["total_nodes"]),
            "total_atoms":    int(st["total_atoms"]),
            "total_contexts": int(st["total_contexts"]),
            "warmed_up":      bool(st["warmed_up"]),
            "theta_assign":   round(st["theta_assign"], 4),
            "theta_merge":    round(st["theta_merge"],  4),
            "watchlist":      int(st["watchlist_count"]),
            "changelog":      int(st["changelog_count"]),
        })
    else:
        _ok(BANNER)
        _ok(f"\nDatabase: {db}  ({db_size:,} bytes)")
        _ok(f"  total nodes:    {int(st['total_nodes'])}")
        _ok(f"  total atoms:    {int(st['total_atoms'])}")
        _ok(f"  total contexts: {int(st['total_contexts'])}")
        _ok(f"  warmed up:      {'yes' if st['warmed_up'] else 'no'}")
        _ok(f"  θ_assign:       {st['theta_assign']:.3f}")
        _ok(f"  θ_merge:        {st['theta_merge']:.3f}")
        if st["watchlist_count"]:
            _ok(f"  ⚠ watchlist:   {int(st['watchlist_count'])} atoms need review")


def cmd_atoms(args):
    """List known atoms."""
    r     = _load_rsvs(args.db)
    atoms = sorted(r.atoms(include_seeds=args.seeds))

    if not atoms:
        if args.json:
            _json_out([])
        else:
            _ok("No atoms yet. Run 'rsvs ingest' first.")
        return

    if args.json:
        cm = r.confidence_map()
        _json_out([
            {"label": a, "confidence": round(cm.get(a, 0.0), 4)}
            for a in atoms
        ])
    else:
        cm = r.confidence_map()
        _ok(f"Atoms ({len(atoms)}):")
        for atom in atoms:
            conf = cm.get(atom, 0.0)
            bar  = "█" * int(conf * 20)
            _ok(f"  {atom:<16} conf={conf:.3f}  {bar}")


def cmd_senses(args):
    """Show senses for a concept."""
    r = _load_rsvs(args.db)

    try:
        senses = r.senses(args.concept)
    except Exception as e:
        if args.json:
            _json_out({"error": str(e)})
        else:
            _err(str(e))
        return

    if args.json:
        _json_out([{
            "sense_idx":  s.sense_idx,
            "n_contexts": s.n_contexts,
            "coherence":  round(s.coherence, 4),
            "status":     s.status,
            "core_atoms": s.core_atoms,
        } for s in senses])
    else:
        _ok(f"Senses for '{args.concept}' ({len(senses)} total):")
        for s in senses:
            coh_bar = "█" * int(s.coherence * 20)
            _ok(f"\n  sense #{s.sense_idx} [{s.status}]  N={s.n_contexts}")
            _ok(f"    coherence: {s.coherence:.3f}  {coh_bar}")
            _ok(f"    core:      {s.core_atoms}")


def cmd_info(args):
    """Show detailed info about an atom."""
    r = _load_rsvs(args.db)

    try:
        info = r.atom_info(args.atom)
    except Exception as e:
        if args.json:
            _json_out({"error": str(e)})
        else:
            _err(str(e))
        return

    tier_name = {1: "Tier1 (autonomous)", 2: "Tier2 (flagged)", 3: "Tier3 (blocked)"}

    if args.json:
        _json_out({
            "label":      info.label,
            "id":         info.id,
            "confidence": round(info.confidence, 4),
            "tier":       info.tier,
            "is_stable":  info.is_stable,
        })
    else:
        conf_bar = "█" * int(info.confidence * 20)
        _ok(f"Atom: '{info.label}'  (id={info.id})")
        _ok(f"  confidence: {info.confidence:.3f}  {conf_bar}")
        _ok(f"  tier:       {tier_name.get(info.tier, str(info.tier))}")
        _ok(f"  memory:     {'stable' if info.is_stable else 'working'}")

        # Show senses if available
        try:
            senses = r.senses(args.atom)
            _ok(f"  senses:     {len(senses)}")
            for s in senses[:3]:
                _ok(f"    #{s.sense_idx} [{s.status}] N={s.n_contexts} "
                    f"coh={s.coherence:.2f} core={s.core_atoms[:3]}")
            if len(senses) > 3:
                _ok(f"    ... and {len(senses)-3} more")
        except Exception:
            pass

def cmd_ingest_corpus(args):
    from rsvs.ingest_wiki import ingest_domains
    from rsvs.corpus import domain_names
    domains = domain_names() if args.all else args.domains
    if not domains:
        _err("Specify --domains or --all for ingest-corpus")
    summary = ingest_domains(args.db, domains, chunk_size=args.chunk_size, verbose=not args.json)
    if args.json:
        _json_out({
            "api_version": "v1",
            "schema_version": "v1",
            "db": args.db,
            "domains": domains,
            "summary": summary,
        })
    else:
        _ok(f"✓ ingest-corpus done: {len(domains)} domain(s) -> {args.db}")

def cmd_eval(args):
    from rsvs.eval import run_eval, compare_with_baseline
    domains = args.domains if args.domains else None
    report = run_eval(
        db_path=args.db,
        domains=domains,
        verbose=not args.json,
        compare_adaptive=True,
    )
    payload = report.to_dict()
    payload["api_version"] = "v1"
    payload["schema_version"] = "v1"
    if args.baseline_json and Path(args.baseline_json).exists():
        baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        payload["gates"] = compare_with_baseline(payload, baseline)
    if args.json:
        _json_out(payload)
    else:
        print(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if not args.json:
            _ok(f"saved eval report -> {args.json_out}")

def cmd_replay_events(args):
    r = _load_rsvs(args.db)
    raw = r.consume_events_v1(args.after_seq, args.limit)
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"raw": raw}
    if args.json:
        _json_out(payload)
    else:
        _ok(f"latest_seq={payload.get('latest_seq')}, events={len(payload.get('events', []))}")
        for evt in payload.get("events", [])[:20]:
            _ok(f"  seq={evt.get('seq')} type={evt.get('event_type')} corr={evt.get('correlation_id')}")


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rsvs",
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"RSVS {VERSION}")

    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # --- init ---
    pi = sub.add_parser("init", help="Initialize a new RSVS database")
    pi.add_argument("--db",        default=DEFAULT_DB, metavar="PATH")
    pi.add_argument("--force",     action="store_true")
    pi.add_argument("--promote-n", dest="promote_n", type=int,   default=3)
    pi.add_argument("--theta",     type=float, default=0.12)
    pi.add_argument("--n-warm",    dest="n_warm", type=int, default=20)
    pi.add_argument("--eta",       type=float, default=0.1)

    # --- ingest ---
    pn = sub.add_parser("ingest", help="Ingest text or a file into the graph")
    pn.add_argument("text_or_file", metavar="TEXT_OR_FILE")
    pn.add_argument("--db",     default=DEFAULT_DB, metavar="PATH")
    pn.add_argument("--domain", type=int, default=None)
    pn.add_argument("--json",   action="store_true")

    # --- query ---
    pq = sub.add_parser("query", help="Context-aware query for a concept")
    pq.add_argument("concept")
    pq.add_argument("context")
    pq.add_argument("--db",  default=DEFAULT_DB, metavar="PATH")
    pq.add_argument("--top", type=int, default=6)
    pq.add_argument("--json", action="store_true")

    # --- similarity ---
    ps = sub.add_parser("similarity", help="Similarity between two concepts")
    ps.add_argument("a")
    ps.add_argument("b")
    ps.add_argument("--db",   default=DEFAULT_DB, metavar="PATH")
    ps.add_argument("--json", action="store_true")

    # --- status ---
    pst = sub.add_parser("status", help="Show system status")
    pst.add_argument("--db",   default=DEFAULT_DB, metavar="PATH")
    pst.add_argument("--json", action="store_true")

    # --- atoms ---
    pa = sub.add_parser("atoms", help="List known atoms")
    pa.add_argument("--db",    default=DEFAULT_DB, metavar="PATH")
    pa.add_argument("--seeds", action="store_true", help="Include seed atoms")
    pa.add_argument("--json",  action="store_true")

    # --- senses ---
    pse = sub.add_parser("senses", help="Show senses for a concept")
    pse.add_argument("concept")
    pse.add_argument("--db",   default=DEFAULT_DB, metavar="PATH")
    pse.add_argument("--json", action="store_true")

    # --- info ---
    pif = sub.add_parser("info", help="Detailed info about an atom")
    pif.add_argument("atom")
    pif.add_argument("--db",   default=DEFAULT_DB, metavar="PATH")
    pif.add_argument("--json", action="store_true")

    # --- ingest-corpus ---
    pic = sub.add_parser("ingest-corpus", help="Ingest embedded corpus domains")
    pic.add_argument("--db", default=DEFAULT_DB, metavar="PATH")
    pic.add_argument("--domains", nargs="+", default=None)
    pic.add_argument("--all", action="store_true")
    pic.add_argument("--chunk-size", type=int, default=5)
    pic.add_argument("--json", action="store_true")

    # --- eval ---
    pev = sub.add_parser("eval", help="Run quality + speed evaluation")
    pev.add_argument("--db", default=DEFAULT_DB, metavar="PATH")
    pev.add_argument("--domains", nargs="+", default=None)
    pev.add_argument("--json", action="store_true")
    pev.add_argument("--json-out", default=None, metavar="PATH")
    pev.add_argument("--baseline-json", default=None, metavar="PATH")

    # --- replay-events ---
    pre = sub.add_parser("replay-events", help="Replay incremental event stream")
    pre.add_argument("--db", default=DEFAULT_DB, metavar="PATH")
    pre.add_argument("--after-seq", type=int, default=None)
    pre.add_argument("--limit", type=int, default=500)
    pre.add_argument("--json", action="store_true")

    return p


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

COMMANDS = {
    "init":       cmd_init,
    "ingest":     cmd_ingest,
    "query":      cmd_query,
    "similarity": cmd_similarity,
    "status":     cmd_status,
    "atoms":      cmd_atoms,
    "senses":     cmd_senses,
    "info":       cmd_info,
    "ingest-corpus": cmd_ingest_corpus,
    "eval":       cmd_eval,
    "replay-events": cmd_replay_events,
}

def main():
    parser = build_parser()
    args   = parser.parse_args()
    COMMANDS[args.command](args)

if __name__ == "__main__":
    main()
