# CLI Reference

RSVS installs a `rsvs` command-line tool after `pip install rsvs`. The CLI provides access to all core operations without writing Python code. State is persisted to a JSON file.

---

## Global Options

| Option | Default | Description |
|--------|---------|-------------|
| `--db PATH` | `./rsvs.json` | Path to the knowledge graph JSON file |
| `--json` | `false` | Output in machine-readable JSON format |

---

## Commands

### `rsvs init`

Initialize a new knowledge graph.

```bash
rsvs init --db my_graph.json --promote-n 3 --theta 0.12 --n-warm 20 --eta 0.1
```

### `rsvs ingest`

Ingest text (literal string or file path).

```bash
rsvs ingest "Batu adalah material keras dari alam" --db my_graph.json
rsvs ingest corpus.txt --db my_graph.json
```

### `rsvs ingest-corpus`

Ingest from the embedded Indonesian corpus.

```bash
rsvs ingest-corpus --domains geology materials --db my_graph.json
rsvs ingest-corpus --all --db my_graph.json
```

### `rsvs query`

Query a concept in context.

```bash
rsvs query batu "material keras" --db my_graph.json
```

### `rsvs similarity`

Compute similarity between two concepts.

```bash
rsvs similarity batu kayu --db my_graph.json
```

### `rsvs info`

Inspect a concept's details.

```bash
rsvs info batu --db my_graph.json
```

### `rsvs senses`

List all senses of a concept.

```bash
rsvs senses batu --db my_graph.json
```

### `rsvs atoms`

List all known atoms.

```bash
rsvs atoms --db my_graph.json
```

### `rsvs status`

Show system status.

```bash
rsvs status --db my_graph.json
```

### `rsvs eval`

Run quality evaluation benchmarks.

```bash
rsvs eval --db my_graph.json --json
```

### `rsvs replay-events`

Replay the incremental event stream.

```bash
rsvs replay-events --db my_graph.json --after-seq 100 --limit 50
```

### `rsvs serve`

> **Note**: This command is not yet implemented. To start the FastAPI server, use `python -m rsvs.fastapi_server` instead. See [server.md](server.md) for details.

```bash
# Not yet available as a CLI subcommand — use this instead:
python -m rsvs.fastapi_server
```
