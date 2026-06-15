# @WHO:   self-ai/src/derivation/sqlite_store.py
# @WHAT:  SQLite-backed persistence for UnderstandingGraph
# @PART:  self-ai/derivation
# @ENTRY: SQLiteGraphStore

"""SQLite Graph Store — scalable persistence for UnderstandingGraph.

Why SQLite instead of JSON flat file?
    The existing JSON store loads the ENTIRE file into memory on every
    startup and writes the entire graph on every mutation. This works for
    <100 nodes but breaks down at scale:

    - A 1000+ node graph means a multi-MB JSON file parsed every time
    - No way to query by confidence, source, or lifecycle without loading
      everything first
    - No concurrent access safety (atomic writes via .tmp help, but SQLite
      gives proper transactional guarantees)

    SQLite is part of Python's stdlib — zero new dependencies — and gives us:
    - Indexed queries on confidence, source, lifecycle
    - Row-level reads (get_node without loading everything)
    - ACID transactions
    - Efficient upserts (no full-file rewrite)

Schema:
    The `nodes` table stores both queryable columns AND a full JSON blob
    so that every UnderstandingNode field can be round-tripped. The
    explicit columns (id, name, concept, abstraction, confidence, source,
    lifecycle, condition_embedding, created_at) exist for indexed queries.
    The `data` column stores the complete to_dict() JSON for reconstruction.

Usage:
    from derivation.understanding_builder import UnderstandingGraph

    # Automatic — just use a .db or .sqlite extension
    graph = UnderstandingGraph(store_path="self.db")

    # Or .sqlite — same effect
    graph = UnderstandingGraph(store_path="knowledge.sqlite")
"""

import json
import sqlite3
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SQLiteGraphStore:
    """SQLite-backed store for UnderstandingGraph nodes.

    Implements the same interface as the JSON flat-file store so that
    UnderstandingGraph can swap backends transparently based on the
    file extension of store_path.

    Methods:
        load()          — Load all nodes from SQLite into a dict
        save(nodes)     — Upsert all nodes into SQLite
        add_node(node)  — Insert or replace a single node
        get_node(id)    — Retrieve a single node by ID
        delete_node(id) — Remove a single node by ID
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._init_db()

    # ──────────────── Schema Initialization ────────────────

    def _init_db(self):
        """Create the nodes table if it doesn't exist."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    concept TEXT,
                    abstraction TEXT,
                    confidence REAL,
                    source TEXT,
                    lifecycle TEXT,
                    condition_embedding TEXT,
                    created_at REAL,
                    data TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_confidence
                ON nodes(confidence)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_source
                ON nodes(source)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_nodes_lifecycle
                ON nodes(lifecycle)
            """)
            conn.commit()
        finally:
            conn.close()

    # ──────────────── Internal Helpers ────────────────

    def _row_to_node_dict(self, row: tuple) -> dict:
        """Convert a SQLite row tuple to an UnderstandingNode-compatible dict.

        The full node data is stored in the `data` column as JSON.
        The explicit columns are for querying only; the `data` column
        is the source of truth for reconstruction.
        """
        # row = (id, name, concept, abstraction, confidence, source,
        #        lifecycle, condition_embedding, created_at, data)
        data_json = row[9]
        return json.loads(data_json)

    def _node_to_row(self, node) -> tuple:
        """Convert an UnderstandingNode to a SQLite row tuple.

        Extracts queryable columns AND stores the full to_dict() JSON
        in the `data` column for complete round-tripping.
        """
        node_dict = node.to_dict()
        data_json = json.dumps(node_dict, ensure_ascii=False)

        condition_embedding_json = (
            json.dumps(node.condition_embedding)
            if node.condition_embedding is not None
            else None
        )

        lifecycle_value = (
            node.lifecycle.value
            if hasattr(node.lifecycle, 'value')
            else str(node.lifecycle)
        )

        return (
            node.id,
            node.name,
            node.concept,
            node.abstraction,
            node.confidence,
            node.source,
            lifecycle_value,
            condition_embedding_json,
            time.time(),
            data_json,
        )

    # ──────────────── Public Interface ────────────────

    def load(self) -> Dict[str, dict]:
        """Load all nodes from SQLite.

        Returns:
            dict mapping node_id → UnderstandingNode dict (as produced
            by to_dict()). The caller (UnderstandingGraph._load) is
            responsible for converting each dict back into an
            UnderstandingNode via from_dict().
        """
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute("SELECT * FROM nodes")
            rows = cursor.fetchall()
            result = {}
            for row in rows:
                node_dict = self._row_to_node_dict(row)
                node_id = node_dict.get('id', row[0])
                result[node_id] = node_dict
            return result
        except Exception as e:
            logger.warning("Failed to load nodes from SQLite: %s", e)
            return {}
        finally:
            conn.close()

    def save(self, nodes: dict):
        """Upsert all nodes into SQLite.

        Args:
            nodes: dict mapping node_id → UnderstandingNode objects,
                   same format as UnderstandingGraph._nodes.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("BEGIN TRANSACTION")
            # Clear and re-insert for a clean save (matches JSON store
            # semantics where _save() writes the entire graph).
            conn.execute("DELETE FROM nodes")
            for node_id, node in nodes.items():
                row = self._node_to_row(node)
                conn.execute(
                    """INSERT OR REPLACE INTO nodes
                       (id, name, concept, abstraction, confidence, source,
                        lifecycle, condition_embedding, created_at, data)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning("Failed to save nodes to SQLite: %s", e)
        finally:
            conn.close()

    def add_node(self, node):
        """Insert or replace a single node in SQLite.

        Args:
            node: An UnderstandingNode instance.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            row = self._node_to_row(node)
            conn.execute(
                """INSERT OR REPLACE INTO nodes
                   (id, name, concept, abstraction, confidence, source,
                    lifecycle, condition_embedding, created_at, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row,
            )
            conn.commit()
        except Exception as e:
            logger.warning("Failed to add node to SQLite: %s", e)
        finally:
            conn.close()

    def get_node(self, node_id: str) -> Optional[dict]:
        """Retrieve a single node by ID.

        Args:
            node_id: The node's unique identifier.

        Returns:
            An UnderstandingNode-compatible dict (as produced by to_dict()),
            or None if not found. The caller is responsible for converting
            back to an UnderstandingNode via from_dict().
        """
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "SELECT * FROM nodes WHERE id = ?", (node_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_node_dict(row)
        except Exception as e:
            logger.warning("Failed to get node from SQLite: %s", e)
            return None
        finally:
            conn.close()

    def delete_node(self, node_id: str):
        """Remove a single node by ID.

        Args:
            node_id: The node's unique identifier.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            conn.commit()
        except Exception as e:
            logger.warning("Failed to delete node from SQLite: %s", e)
        finally:
            conn.close()
