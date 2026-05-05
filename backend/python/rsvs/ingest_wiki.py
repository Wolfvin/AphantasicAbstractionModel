"""
RSVS v0.9 — Wikipedia ingestion pipeline.

Two modes:
  1. Embedded corpus (offline) — uses corpus.py
  2. Live Wikipedia (online)   — uses wikipedia-api if available

Usage as script:
  python3 -m rsvs.ingest_wiki --db rsvs.json --domains geology water
  python3 -m rsvs.ingest_wiki --db rsvs.json --all
  python3 -m rsvs.ingest_wiki --db rsvs.json --all --report
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from collections.abc import Iterator

from .corpus import DOMAINS, get_domain_text, domain_names


# -----------------------------------------------------------------------
# Wikipedia fetch (optional — graceful fallback to embedded)
# -----------------------------------------------------------------------

def _fetch_wikipedia(title: str) -> str | None:
    """Try to fetch a Wikipedia article. Returns None if unavailable."""
    try:
        import urllib.request
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        req = urllib.request.Request(url, headers={"User-Agent": "RSVS/0.9"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            return data.get("extract", "")
    except Exception:
        return None


# -----------------------------------------------------------------------
# Domain → article title mapping (for live mode)
# -----------------------------------------------------------------------

DOMAIN_ARTICLES = {
    "geology":   ["Rock_(geology)", "Mineral", "Igneous_rock", "Sedimentary_rock"],
    "water":     ["Water", "Ocean", "River", "Hydrological_cycle"],
    "biology":   ["Cell_(biology)", "Evolution", "Photosynthesis", "Ecosystem"],
    "physics":   ["Force", "Energy", "Wave", "Thermodynamics"],
    "materials": ["Material", "Metal", "Composite_material", "Polymer"],
}


# -----------------------------------------------------------------------
# Text chunks — yield sentence-level batches for streaming ingest
# -----------------------------------------------------------------------

def iter_domain_chunks(domain: str, chunk_size: int = 5) -> Iterator[str]:
    """Yield batches of `chunk_size` sentences for a domain."""
    sentences = DOMAINS.get(domain, [])
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i : i + chunk_size]
        yield " ".join(chunk)


# -----------------------------------------------------------------------
# Main ingestion function
# -----------------------------------------------------------------------

def ingest_domains(
    db_path: str,
    domains: list[str],
    chunk_size: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Ingest the given domains into an RSVS database.
    Returns a summary dict with stats per domain.
    """
    from rsvs import Rsvs

    p = Path(db_path)
    if p.exists():
        r = Rsvs.load(str(p))
        if verbose:
            print(f"Loaded existing DB: {db_path}")
    else:
        r = Rsvs(entity_promote_n=3, theta_assign=0.12, n_warm=20, eta=0.1)
        if verbose:
            print(f"Created new DB: {db_path}")

    summary = {}

    for domain_idx, domain in enumerate(domains, 1):
        if domain not in DOMAINS:
            print(f"  ⚠ Unknown domain: {domain!r} — skipping", file=sys.stderr)
            continue

        r.set_domain(domain_idx)
        if verbose:
            print(f"\n[{domain_idx}/{len(domains)}] Domain: {domain!r} "
                  f"({len(DOMAINS[domain])} sentences)")

        domain_stats = {
            "sentences": 0,
            "atoms_promoted": 0,
            "senses_created": 0,
            "chunks": 0,
        }

        t0 = time.time()
        for chunk in iter_domain_chunks(domain, chunk_size):
            stats = r.ingest(chunk)
            domain_stats["sentences"]     += stats.sentences_processed
            domain_stats["atoms_promoted"] += stats.atoms_promoted
            domain_stats["senses_created"] += stats.sense_created
            domain_stats["chunks"]         += 1

            if verbose:
                print(f"  chunk {domain_stats['chunks']:2d}: "
                      f"+{stats.sentences_processed} sentences  "
                      f"+{stats.atoms_promoted} atoms  "
                      f"+{stats.sense_created} senses")

        elapsed = time.time() - t0
        domain_stats["elapsed_s"] = round(elapsed, 2)
        summary[domain] = domain_stats

        if verbose:
            st = r.status()
            print(f"  → total atoms: {int(st['total_atoms'])}  "
                  f"contexts: {int(st['total_contexts'])}  "
                  f"({elapsed:.1f}s)")

    # Save
    r.save(db_path)
    if verbose:
        print(f"\nSaved → {db_path}  ({Path(db_path).stat().st_size:,} bytes)")

    return summary


# -----------------------------------------------------------------------
# Report — print knowledge summary after ingestion
# -----------------------------------------------------------------------

def print_report(db_path: str):
    from rsvs import Rsvs

    r = Rsvs.load(db_path)
    st = r.status()
    atoms = sorted(r.atoms())
    cm = r.confidence_map()

    print("\n" + "=" * 60)
    print("RSVS Knowledge Report")
    print("=" * 60)
    print(f"DB:       {db_path}  ({Path(db_path).stat().st_size:,} bytes)")
    print(f"Nodes:    {int(st['total_nodes'])}")
    print(f"Atoms:    {int(st['total_atoms'])} promoted")
    print(f"Contexts: {int(st['total_contexts'])}")
    print(f"Warmed:   {'yes' if st['warmed_up'] else 'no'}")
    print(f"θ_assign: {st['theta_assign']:.3f}  θ_merge: {st['theta_merge']:.3f}")

    print(f"\nTop atoms by confidence ({len(atoms)} total):")
    sorted_conf = sorted(cm.items(), key=lambda x: -x[1])
    for label, conf in sorted_conf[:15]:
        bar = "█" * int(conf * 25)
        print(f"  {label:<18} {conf:.4f}  {bar}")

    print("\nKey similarities:")
    pairs = [
        ("stone", "rock"),   ("stone", "metal"),  ("stone", "water"),
        ("water", "liquid"), ("rock", "mineral"),  ("heat", "temperature"),
        ("solid", "hard"),   ("metal", "hard"),    ("water", "ice"),
    ]
    for a, b in pairs:
        sim = r.similarity(a, b)
        if sim and sim.jaccard > 0:
            bar = "█" * int(sim.jaccard * 30)
            print(f"  sim({a:<8}, {b:<12}) = {sim.jaccard:.3f}  {bar}")
            if sim.shared:
                print(f"    shared: {sim.shared[:5]}")

    print("\nMulti-sense concepts:")
    for concept in ["stone", "rock", "water", "hard", "solid", "heat"]:
        if concept in atoms:
            try:
                senses = r.senses(concept)
                if len(senses) >= 2:
                    print(f"  {concept}: {len(senses)} senses")
                    for s in senses[:3]:
                        print(f"    #{s.sense_idx} [{s.status}] "
                              f"N={s.n_contexts} "
                              f"coh={s.coherence:.2f} "
                              f"core={s.core_atoms[:3]}")
            except Exception:
                pass

    print("\nContext queries:")
    queries = [
        ("stone", "hard rough texture surface"),
        ("stone", "heat pressure formation underground"),
        ("water", "liquid clear flow transparent"),
        ("rock",  "mineral crystal solid earth"),
        ("heat",  "temperature energy thermal"),
    ]
    for concept, ctx in queries:
        result = r.query(concept, ctx)
        if result:
            top = result.top_atoms(4)
            print(f"  query({concept!r}, {ctx[:30]!r}...)")
            print(f"    → sense #{result.sense_idx} (N={result.sense_n}): {top}")
    print()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        prog="rsvs-wiki",
        description="Ingest Wikipedia-style corpus into RSVS",
    )
    p.add_argument("--db",      default="rsvs.json", metavar="PATH")
    p.add_argument("--domains", nargs="+", choices=domain_names(), metavar="DOMAIN")
    p.add_argument("--all",     action="store_true", help="Ingest all domains")
    p.add_argument("--chunk",   type=int, default=5, help="Sentences per chunk")
    p.add_argument("--report",  action="store_true", help="Print knowledge report after")
    p.add_argument("--quiet",   action="store_true")
    args = p.parse_args()

    if args.all:
        domains = domain_names()
    elif args.domains:
        domains = args.domains
    else:
        p.error("Specify --domains or --all")

    summary = ingest_domains(
        args.db, domains,
        chunk_size=args.chunk,
        verbose=not args.quiet,
    )

    if args.report:
        print_report(args.db)

    # Print JSON summary
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
