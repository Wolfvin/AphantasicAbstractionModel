#!/usr/bin/env python3
"""RSVS Agent CLI

Agent-friendly CLI wrapper for RSVS backend bridge + atom artifacts.
No third-party dependencies required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error, parse, request

VALID_MODES = ("ingest", "appraise", "relate")
VALID_VIEWS = ("compact", "detail")
DEFAULT_BASE_URL = os.environ.get("RSVS_BRIDGE_URL", "http://127.0.0.1:8787")
DEFAULT_ATOM_DIR = Path(
    os.environ.get(
        "RSVS_ATOM_DIR",
        "/home/raymond/workspace/projets/skills_and_mcp/RSVS/atom",
    )
)


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url=url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach backend {url}: {exc.reason}") from exc


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _run_request(base_url: str, mode: str, text: str, correlation_id: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise RuntimeError(f"invalid mode: {mode}")
    payload = {
        "mode": mode,
        "text": text,
        "correlation_id": correlation_id,
        "options": options or {},
    }
    return _http_json("POST", f"{base_url}/run", payload)


def cmd_health(args: argparse.Namespace) -> int:
    data = _http_json("GET", f"{args.base_url}/health")
    _print_json(data)
    return 0


def cmd_latest(args: argparse.Namespace) -> int:
    url = f"{args.base_url}/latest"
    qs: dict[str, str] = {}
    if args.mode:
        qs["mode"] = args.mode
    if args.view:
        qs["view"] = args.view
    if qs:
        url = f"{url}?{parse.urlencode(qs)}"
    data = _http_json("GET", url)

    if args.summary:
        mode = data.get("mode") or args.mode or "ingest"
        result = data.get("result", {})
        if mode == "ingest" and "snapshot" in result:
            snap = result.get("snapshot", {})
            print(
                f"latest mode=ingest snapshot={snap.get('snapshot_id', '-')}"
                f" nodes={len(snap.get('nodes', []))}"
                f" edges={len(snap.get('edges', []))}"
                f" events={len(result.get('events', []))}"
            )
        elif mode == "appraise":
            stance = result.get("stance", {})
            print(
                f"latest mode=appraise verdict={result.get('verdict', '-')}"
                f" agree={stance.get('agree', '-')} disagree={stance.get('disagree', '-')}"
                f" confidence={result.get('confidence', '-')}"
            )
        elif mode == "relate":
            print(
                f"latest mode=relate nodes={len(result.get('related_nodes', []))}"
                f" edges={len(result.get('related_edges', []))}"
            )
        else:
            print(f"latest mode={mode}")
        return 0

    _print_json(data)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text or not text.strip():
        raise RuntimeError("input text is empty")

    options: dict[str, Any] = {}
    if args.top_k is not None:
        options["top_k"] = args.top_k
    if args.view:
        options["view"] = args.view

    data = _run_request(args.base_url, args.mode, text, args.correlation_id, options)

    if args.summary:
        result = data.get("result", {})
        if args.mode == "ingest":
            snap = result.get("snapshot", {})
            print(
                f"run mode=ingest corr={data.get('correlation_id', '-') }"
                f" nodes={len(snap.get('nodes', []))}"
                f" edges={len(snap.get('edges', []))}"
            )
        elif args.mode == "appraise":
            stance = result.get("stance", {})
            print(
                f"run mode=appraise verdict={result.get('verdict', '-') }"
                f" agree={stance.get('agree', '-')} disagree={stance.get('disagree', '-')}"
                f" confidence={result.get('confidence', '-')}"
            )
        else:
            print(
                f"run mode=relate nodes={len(result.get('related_nodes', []))}"
                f" edges={len(result.get('related_edges', []))}"
            )
        return 0

    _print_json(data)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    run_args = argparse.Namespace(
        base_url=args.base_url,
        mode="ingest",
        text=args.text,
        file=args.file,
        correlation_id=args.correlation_id,
        summary=args.summary,
        top_k=None,
        view="compact",
    )
    return cmd_run(run_args)


def _find_latest(atom_dir: Path, pattern: str) -> Path | None:
    files = sorted(atom_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def cmd_atom_ls(args: argparse.Namespace) -> int:
    atom_dir = Path(args.atom_dir)
    if not atom_dir.exists():
        raise RuntimeError(f"atom dir not found: {atom_dir}")

    patterns = (
        ("snapshot-*.json", "snapshot"),
        ("events-*.jsonl", "events"),
        ("report-*.json", "report"),
        ("appraise-*.json", "appraise"),
        ("relate-*.json", "relate"),
    )

    rows: list[dict[str, Any]] = []
    for pat, kind in patterns:
        for p in sorted(atom_dir.glob(pat), key=lambda x: x.stat().st_mtime, reverse=True)[: args.limit]:
            rows.append(
                {
                    "file": str(p),
                    "kind": kind,
                    "bytes": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                }
            )

    rows = sorted(rows, key=lambda r: r["mtime"], reverse=True)
    _print_json(rows)
    return 0


def cmd_atom_show(args: argparse.Namespace) -> int:
    atom_dir = Path(args.atom_dir)
    if not atom_dir.exists():
        raise RuntimeError(f"atom dir not found: {atom_dir}")

    if args.kind == "snapshot":
        p = _find_latest(atom_dir, "snapshot-*.json")
    elif args.kind == "events":
        p = _find_latest(atom_dir, "events-*.jsonl")
    elif args.kind == "report":
        p = _find_latest(atom_dir, "report-*.json")
    elif args.kind == "appraise":
        p = _find_latest(atom_dir, "appraise-*.json")
    else:
        p = _find_latest(atom_dir, "relate-*.json")

    if p is None:
        raise RuntimeError(f"no {args.kind} file found")

    if args.kind == "events":
        lines = p.read_text(encoding="utf-8").splitlines()
        if args.tail > 0:
            lines = lines[-args.tail :]
        print("\n".join(lines))
        return 0

    print(p.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rsvs-agent", description="Agent-oriented RSVS bridge CLI")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Bridge base URL (default: {DEFAULT_BASE_URL})")
    p.add_argument("--atom-dir", default=str(DEFAULT_ATOM_DIR), help=f"Atom directory (default: {DEFAULT_ATOM_DIR})")

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("health", help="Check bridge health")
    s.set_defaults(func=cmd_health)

    s = sub.add_parser("latest", help="Fetch latest bundle/artifact")
    s.add_argument("--mode", choices=VALID_MODES, help="Mode-specific latest artifact")
    s.add_argument("--view", choices=VALID_VIEWS, default="compact", help="Output view mode")
    s.add_argument("--summary", action="store_true", help="Print compact summary")
    s.set_defaults(func=cmd_latest)

    s = sub.add_parser("run", help="Run mode-aware request")
    s.add_argument("--mode", required=True, choices=VALID_MODES)
    s.add_argument("--text", default="", help="Input text")
    s.add_argument("--file", help="Read input from file")
    s.add_argument("--correlation-id", default="", help="Optional correlation id")
    s.add_argument("--top-k", type=int, help="Top-k related nodes for relate mode")
    s.add_argument("--view", choices=VALID_VIEWS, default="compact", help="Result view mode")
    s.add_argument("--summary", action="store_true", help="Print compact summary")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("ingest", help="Backward-compatible alias for run --mode ingest")
    s.add_argument("--text", default="", help="Input text")
    s.add_argument("--file", help="Read input from file")
    s.add_argument("--correlation-id", default="", help="Optional correlation id")
    s.add_argument("--summary", action="store_true", help="Print compact summary")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("atom-ls", help="List recent atom artifacts")
    s.add_argument("--limit", type=int, default=5, help="Max files per type")
    s.set_defaults(func=cmd_atom_ls)

    s = sub.add_parser("atom-show", help="Show latest atom artifact content")
    s.add_argument("kind", choices=["snapshot", "events", "report", "appraise", "relate"], help="Artifact kind")
    s.add_argument("--tail", type=int, default=20, help="Tail lines for events kind")
    s.set_defaults(func=cmd_atom_show)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
