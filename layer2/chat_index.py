"""
Semantic Chat Index — Conversations as a Graph of Meaning.

The existing SituationLayer ingests chat messages as raw text into the RSVS
graph without any semantic structure. This module indexes conversations as a
graph of meaning — each utterance becomes a ChatNode with extracted semantic
atoms, and relationships between utterances become ChatEdges (reply_to,
semantic_link, topic_shift, elaboration, contradiction).

Architecture:
    ChatNode           — a single utterance with semantic atoms
    ChatEdge           — a typed, weighted link between utterances
    ConversationGraph  — a full conversation with nodes, edges, and topics
    SemanticChatIndex  — main class that indexes, retrieves, and segments

Semantic Link Detection (on every new message):
    "semantic_link"   : semantic_atoms overlap > 30% with an earlier message
    "topic_shift"     : semantic_atoms have < 10% overlap with previous message
    "elaboration"     : new atoms are a subset of previous atoms AND same role
    "contradiction"   : bridge.appraise() returns "disagree" vs earlier message

Conversation Segmentation:
    - Gap > 300 seconds between messages  → new conversation
    - topic_shift magnitude > 0.8         → new conversation segment
    - Manual override via conversation_id parameter

Analogi: Seperti Jin Soun di Simhyeon Pavilion yang mengarsipkan percakapan
bukan hanya sebagai teks, tapi sebagai jaringan makna. Setiap percakapan
adalah graf — node-nya adalah ucapan dengan atom semantik, edge-nya adalah
hubungan makna (balasan, link semantik, pergeseran topik, penjelasan,
kontradiksi). Saat dia mencari "kapan kita bicara tentang racun?", dia
tidak mencari kata "racun" — dia mencari makna di baliknya.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of ChatNodes to keep in memory (auto-prune oldest)
_MAX_NODES = 10_000

# Time gap (seconds) beyond which a new conversation is automatically started
_CONVERSATION_GAP_SECONDS = 300.0

# Semantic overlap thresholds for edge detection
_SEMANTIC_LINK_OVERLAP = 0.30   # >30% overlap → semantic_link
_TOPIC_SHIFT_OVERLAP = 0.10    # <10% overlap with previous → topic_shift
_TOPIC_SHIFT_MAGNITUDE = 0.8   # magnitude >0.8 → new conversation segment

# Number of topics per conversation
_MIN_TOPICS_PER_CONV = 5
_MAX_TOPICS_PER_CONV = 10

# Valid roles for ChatNode
_VALID_ROLES = frozenset({"user", "assistant", "system"})

# Valid edge types for ChatEdge
_VALID_EDGE_TYPES = frozenset({
    "reply_to", "semantic_link", "topic_shift", "elaboration", "contradiction",
})

# Persistence schema version
_PERSIST_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Keyword extraction helper (shared with fallback mode)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "that", "this", "with", "from", "have", "been", "they",
    "their", "which", "would", "there", "could", "about",
    "other", "into", "more", "than", "then", "some", "very",
    "also", "just", "like", "only", "over", "such", "after",
    "yang", "dan", "dari", "untuk", "dengan", "adalah", "itu",
    "ini", "ke", "di", "pada", "tidak", "akan", "telah", "oleh",
    "the", "and", "but", "for", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "are", "has",
})

# Role-prefixed artifacts to filter from semantic atoms
# These are created when we ingest "[role] content" into RSVS
_ROLE_ARTIFACTS = frozenset({
    "user", "assistant", "system",
    "[user]", "[assistant]", "[system]",
})


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text for fallback semantic atoms.

    Args:
        text: Input text.

    Returns:
        A list of lowercased keywords (length > 2, not stop words).
    """
    cleaned = text.lower()
    for ch in ",.!?;:()[]{}\"'":
        cleaned = cleaned.replace(ch, " ")
    words = cleaned.split()
    return [w for w in words if len(w) > 2 and w not in _STOP_WORDS and w not in _ROLE_ARTIFACTS][:30]


# ---------------------------------------------------------------------------
# ChatNode — a single utterance with semantic structure
# ---------------------------------------------------------------------------

@dataclass
class ChatNode:
    """A single utterance in a conversation, indexed with semantic atoms.

    Each ChatNode captures not just the raw text, but also the semantic
    concepts (atoms) extracted from RSVS, the position in the conversation,
    and an optional embedding vector for vector similarity search.

    Analogi: Seperti catatan Jin Soun tentang sebuah percakapan — bukan
    hanya "apa yang dikatakan", tapi juga "makna apa yang terkandung",
    "siapa yang mengatakannya", dan "seberapa yakin kita tentang maknanya".

    Attributes:
        node_id: Unique identifier for this utterance (8-char hex).
        role: Speaker role — "user", "assistant", or "system".
        content: The raw text content of the utterance.
        timestamp: ISO-format timestamp when the node was created.
        semantic_atoms: Extracted concept labels from RSVS.
        embedding: Optional embedding vector for similarity search.
        turn_index: Position of this utterance in the conversation.
        conversation_id: Which conversation this node belongs to.
        confidence: Average confidence of the semantic_atoms.
    """

    node_id: str
    role: str
    content: str
    timestamp: str
    semantic_atoms: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    turn_index: int = 0
    conversation_id: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "node_id": self.node_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "semantic_atoms": list(self.semantic_atoms),
            "embedding": list(self.embedding) if self.embedding is not None else None,
            "turn_index": self.turn_index,
            "conversation_id": self.conversation_id,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChatNode:
        """Deserialize from a plain dict.

        Args:
            data: A dict previously returned by to_dict().

        Returns:
            A new ChatNode instance.
        """
        embedding_data = data.get("embedding")
        return cls(
            node_id=data.get("node_id", uuid.uuid4().hex[:8]),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
            semantic_atoms=data.get("semantic_atoms", []),
            embedding=list(embedding_data) if embedding_data is not None else None,
            turn_index=data.get("turn_index", 0),
            conversation_id=data.get("conversation_id", ""),
            confidence=data.get("confidence", 0.0),
        )


# ---------------------------------------------------------------------------
# ChatEdge — a typed, weighted link between utterances
# ---------------------------------------------------------------------------

@dataclass
class ChatEdge:
    """A typed, weighted edge between two ChatNodes.

    Edge types capture the semantic relationship between utterances:
    - "reply_to": sequential reply in the conversation flow
    - "semantic_link": significant semantic overlap (>30%)
    - "topic_shift": low semantic overlap with previous (<10%)
    - "elaboration": new message's atoms are a subset of previous (same role)
    - "contradiction": appraise() detects disagreement

    Analogi: Seperti catatan Jin Soun tentang hubungan antar percakapan —
    "ucapan ini menjawab ucapan sebelumnya", "topik bergeser dari obat
    ke politik", "ini menambah penjelasan tentang apa yang tadi dibicarakan",
    atau "ini bertentangan dengan klaim sebelumnya".

    Attributes:
        edge_id: Unique identifier for this edge (8-char hex).
        from_node: The source ChatNode.node_id.
        to_node: The target ChatNode.node_id.
        edge_type: The type of semantic relationship.
        weight: Strength of the link (0.0–1.0).
        metadata: Additional information about the link.
    """

    edge_id: str
    from_node: str
    to_node: str
    edge_type: str
    weight: float = 0.5
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate edge after initialization."""
        if self.edge_type not in _VALID_EDGE_TYPES:
            raise ValueError(
                f"Invalid edge_type: {self.edge_type!r}. "
                f"Must be one of: {sorted(_VALID_EDGE_TYPES)}"
            )
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(
                f"weight must be between 0.0 and 1.0, got {self.weight}"
            )

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "edge_id": self.edge_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChatEdge:
        """Deserialize from a plain dict.

        Args:
            data: A dict previously returned by to_dict().

        Returns:
            A new ChatEdge instance.
        """
        return cls(
            edge_id=data.get("edge_id", uuid.uuid4().hex[:8]),
            from_node=data.get("from_node", ""),
            to_node=data.get("to_node", ""),
            edge_type=data.get("edge_type", "reply_to"),
            weight=data.get("weight", 0.5),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# ConversationGraph — a full conversation with structure
# ---------------------------------------------------------------------------

@dataclass
class ConversationGraph:
    """A complete conversation indexed as a graph of meaning.

    Contains all ChatNodes and ChatEdges for a conversation, plus
    extracted topics, auto-generated summary, and metadata.

    Analogi: Seperti satu gulungan arsip lengkap di Simhyeon Pavilion —
    berisi semua ucapan (node), semua hubungan (edge), topik-topik yang
    dibicarakan, dan ringkasan otomatis. Jin Soun bisa membaca ringkasan
    dulu, lalu menelusuri graf untuk detail.

    Attributes:
        conversation_id: Unique identifier for this conversation.
        nodes: List of ChatNodes in this conversation.
        edges: List of ChatEdges in this conversation.
        topics: Main topics discussed in this conversation.
        created_at: ISO timestamp when the conversation was started.
        updated_at: ISO timestamp when the conversation was last updated.
        summary: Auto-generated summary of the conversation.
        node_count: Number of nodes (derived, kept in sync).
        edge_count: Number of edges (derived, kept in sync).
    """

    conversation_id: str
    nodes: list[ChatNode] = field(default_factory=list)
    edges: list[ChatEdge] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    summary: str = ""
    node_count: int = 0
    edge_count: int = 0

    def __post_init__(self) -> None:
        """Synchronize counts after initialization."""
        self.node_count = len(self.nodes)
        self.edge_count = len(self.edges)

    def refresh_counts(self) -> None:
        """Refresh node_count and edge_count from the actual lists."""
        self.node_count = len(self.nodes)
        self.edge_count = len(self.edges)
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "conversation_id": self.conversation_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "topics": list(self.topics),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConversationGraph:
        """Deserialize from a plain dict.

        Args:
            data: A dict previously returned by to_dict().

        Returns:
            A new ConversationGraph instance.
        """
        nodes = [ChatNode.from_dict(n) for n in data.get("nodes", [])]
        edges = [ChatEdge.from_dict(e) for e in data.get("edges", [])]
        return cls(
            conversation_id=data.get("conversation_id", uuid.uuid4().hex[:8]),
            nodes=nodes,
            edges=edges,
            topics=data.get("topics", []),
            created_at=data.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
            updated_at=data.get("updated_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
            summary=data.get("summary", ""),
            node_count=len(nodes),
            edge_count=len(edges),
        )


# ---------------------------------------------------------------------------
# SemanticChatIndex — main class
# ---------------------------------------------------------------------------

class SemanticChatIndex:
    """Indexes conversations as a graph of meaning using RSVS.

    Instead of ingesting chat messages as raw text (like SituationLayer),
    this module extracts semantic atoms from each utterance, creates
    ChatNodes with structured metadata, and detects semantic links
    between utterances as ChatEdges.

    Thread Safety:
        All operations are protected by a threading.Lock to ensure
        safe concurrent access.

    Bounded State:
        Maximum 10,000 ChatNodes are kept in memory. When this limit
        is exceeded, the oldest nodes are auto-pruned (along with
        their edges).

    Analogi: Jin Soun di Simhyeon Pavilion tidak hanya menyimpan teks
    percakapan — dia membuat indeks semantik. Setiap ucapan diurai
    menjadi atom makna, dihubungkan ke ucapan lain berdasarkan
    kesamaan makna, dan dikategorikan ke dalam percakapan. Saat dia
    ingin mencari "kapan kita bicara tentang racun?", dia mencari
    berdasarkan makna, bukan kata kunci.

    Usage:
        idx = SemanticChatIndex(bridge=get_bridge())

        # Index a single message
        node = idx.index_message("user", "Tell me about Snow Plum Pill")

        # Index a full conversation
        graph = idx.index_conversation([
            {"role": "user", "content": "What is RSVS?"},
            {"role": "assistant", "content": "RSVS is a semantic graph engine."},
        ])

        # Retrieve by meaning
        results = idx.retrieve_by_meaning("semantic graph", top_k=5)

        # Get topics and shifts
        topics = idx.get_recent_topics(n=10)
        shifts = idx.detect_topic_shifts(graph.conversation_id)
    """

    def __init__(self, bridge: RsvsBridge | None = None) -> None:
        """Initialize the SemanticChatIndex.

        Args:
            bridge: Optional pre-built RsvsBridge. If None, one is
                obtained via get_bridge().
        """
        self._bridge = bridge if bridge is not None else get_bridge()
        self.rsvs_available = self._bridge.is_available

        # Internal state — conversation_id → ConversationGraph
        self._conversations: dict[str, ConversationGraph] = {}

        # Node index — node_id → ChatNode (for fast lookup)
        self._node_index: dict[str, ChatNode] = {}

        # Edge index — edge_id → ChatEdge (for fast lookup)
        self._edge_index: dict[str, ChatEdge] = {}

        # Node-to-conversation mapping — node_id → conversation_id
        self._node_to_conv: dict[str, str] = {}

        # Outgoing edges per node — node_id → [edge_id]
        self._outgoing_edges: dict[str, list[str]] = {}

        # Incoming edges per node — node_id → [edge_id]
        self._incoming_edges: dict[str, list[str]] = {}

        # Last message tracking per conversation for auto-linking
        # conversation_id → (node_id, timestamp_float, semantic_atoms)
        self._last_msg_per_conv: dict[str, tuple[str, float, list[str]]] = {}

        # Auto-incrementing conversation counter
        self._conv_counter: int = 0

        # Thread lock for all operations
        self._lock = threading.Lock()

        if self.rsvs_available:
            if self._bridge.is_rust_core:
                logger.info("SemanticChatIndex initialized with RSVS Rust core")
            else:
                logger.info("SemanticChatIndex initialized with RSVS fallback graph")
        else:
            logger.info("SemanticChatIndex initialized WITHOUT RSVS (keyword fallback)")

    # ------------------------------------------------------------------
    # index_message — ingest a single message
    # ------------------------------------------------------------------

    def index_message(
        self,
        role: str,
        content: str,
        conversation_id: str | None = None,
    ) -> ChatNode:
        """Ingest a single chat message and return a ChatNode.

        Steps:
        1. Ingest the message into the RSVS graph
        2. Extract semantic atoms using bridge.senses(), relate(), query()
        3. Create a ChatNode with semantic_atoms
        4. Auto-detect conversation_id if not provided
        5. Link to previous message with "reply_to" edge
        6. Detect semantic links to earlier messages

        Analogi: Jin Soun mendengar sebuah ucapan, mencatatnya,
        menguraikan maknanya, menghubungkannya ke ucapan sebelumnya,
        dan mendeteksi apakah ada pergeseran topik atau kontradiksi.

        Args:
            role: Speaker role — "user", "assistant", or "system".
            content: The message text content.
            conversation_id: Optional conversation ID. If None, auto-detected
                based on time gap and topic continuity.

        Returns:
            A ChatNode representing this utterance in the semantic graph.
        """
        if role not in _VALID_ROLES:
            logger.warning("Invalid role %r, defaulting to 'user'", role)
            role = "user"

        now = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))

        # Step 1 & 2: Extract semantic atoms
        semantic_atoms = self._extract_semantic_atoms(role, content)
        confidence = self._compute_node_confidence(semantic_atoms)

        # Step 4: Auto-detect conversation_id
        if conversation_id is None:
            conversation_id = self._detect_conversation_id(now, semantic_atoms)

        # Get or create the conversation graph
        conv_graph = self._get_or_create_conversation(conversation_id)

        # Determine turn index
        turn_index = len(conv_graph.nodes)

        # Step 3: Create ChatNode
        node_id = uuid.uuid4().hex[:8]
        node = ChatNode(
            node_id=node_id,
            role=role,
            content=content,
            timestamp=timestamp,
            semantic_atoms=semantic_atoms,
            embedding=None,  # Embeddings computed on demand
            turn_index=turn_index,
            conversation_id=conversation_id,
            confidence=confidence,
        )

        with self._lock:
            # Add node to conversation and indexes
            conv_graph.nodes.append(node)
            conv_graph.refresh_counts()
            self._node_index[node_id] = node
            self._node_to_conv[node_id] = conversation_id
            self._outgoing_edges[node_id] = []
            self._incoming_edges[node_id] = []

            # Step 5: Link to previous message with "reply_to" edge
            prev_info = self._last_msg_per_conv.get(conversation_id)
            if prev_info is not None:
                prev_node_id, prev_ts, prev_atoms = prev_info
                self._create_edge(
                    from_node=node_id,
                    to_node=prev_node_id,
                    edge_type="reply_to",
                    weight=1.0,
                    metadata={"time_gap": now - prev_ts},
                )

            # Step 6: Detect semantic links to earlier messages
            self._detect_semantic_links(node, conv_graph)

            # Update last message tracking
            self._last_msg_per_conv[conversation_id] = (node_id, now, semantic_atoms)

            # Update conversation topics
            self._update_conversation_topics(conversation_id)

            # Auto-prune if exceeding max nodes
            self._auto_prune()

        logger.debug(
            "Indexed message node=%s role=%s conv=%s atoms=%d turn=%d",
            node_id, role, conversation_id, len(semantic_atoms), turn_index,
        )

        return node

    # ------------------------------------------------------------------
    # index_conversation — batch-index a full conversation
    # ------------------------------------------------------------------

    def index_conversation(self, messages: list[dict]) -> ConversationGraph:
        """Batch-index a full conversation.

        Messages format: [{"role": "user", "content": "..."}, ...]

        Each message is indexed sequentially with auto-detected edges
        (reply_to, semantic_link, topic_shift, etc.).

        Analogi: Jin Soun menerima sebuah gulungan percakapan lengkap
        dan mengarsipkannya sekaligus — mengurai setiap ucapan,
        menghubungkannya, dan mengidentifikasi topik-topik utama.

        Args:
            messages: A list of dicts with "role" and "content" keys.

        Returns:
            A ConversationGraph representing the indexed conversation.
        """
        if not messages:
            conv_id = uuid.uuid4().hex[:8]
            return self._get_or_create_conversation(conv_id)

        # Use a consistent conversation ID for the batch
        conv_id = uuid.uuid4().hex[:8]
        first_message = True

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue

            # Override conversation_id for batch indexing
            node = self.index_message(role, content, conversation_id=conv_id)

            if first_message:
                first_message = False

        conv_graph = self._conversations.get(conv_id)
        if conv_graph is None:
            conv_graph = self._get_or_create_conversation(conv_id)

        # Generate summary
        conv_graph.summary = self._generate_summary(conv_graph)

        logger.info(
            "Batch-indexed conversation %s: %d nodes, %d edges, %d topics",
            conv_id, conv_graph.node_count, conv_graph.edge_count,
            len(conv_graph.topics),
        )

        return conv_graph

    # ------------------------------------------------------------------
    # retrieve_by_meaning — find messages by semantic overlap
    # ------------------------------------------------------------------

    def retrieve_by_meaning(self, query: str, top_k: int = 5) -> list[ChatNode]:
        """Find messages whose semantic_atoms overlap with the query's concepts.

        Uses RSVS relate() and structural_similarity() for matching.
        Falls back to keyword matching when RSVS is unavailable.

        Analogi: Jin Soun mencari "kapan kita bicara tentang racun?" —
        dia tidak mencari kata "racun" saja, tapi semua percakapan
        yang maknanya terkait dengan racun, termasuk yang membahas
        "bisa", "keracunan", atau "antidot".

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.

        Returns:
            A list of ChatNodes ranked by semantic relevance.
        """
        # Extract concepts from the query
        query_atoms = self._extract_semantic_atoms("user", query)

        if not query_atoms:
            # Fall back to simple keyword matching
            query_atoms = _extract_keywords(query)

        if not query_atoms:
            return []

        query_set = set(a.lower() for a in query_atoms)

        # Score each node by semantic overlap
        scored_nodes: list[tuple[float, ChatNode]] = []

        with self._lock:
            for node in self._node_index.values():
                node_set = set(a.lower() for a in node.semantic_atoms)
                if not node_set:
                    continue

                # Jaccard-like overlap
                overlap = query_set & node_set
                union = query_set | node_set
                score = len(overlap) / len(union) if union else 0.0

                # Boost by RSVS structural similarity if available
                if self.rsvs_available and score > 0.05:
                    for atom in query_atoms[:3]:
                        try:
                            sim_result = self._bridge.structural_similarity(
                                atom, node.semantic_atoms[0] if node.semantic_atoms else atom
                            )
                            if sim_result is not None:
                                struct_sim = sim_result.get("structural_similarity", 0.0)
                                score = max(score, struct_sim * 0.5)
                        except Exception:
                            pass

                if score > 0.0:
                    scored_nodes.append((score, node))

        # Also try RSVS relate() for spreading activation
        if self.rsvs_available:
            try:
                for atom in query_atoms[:5]:
                    relate_result = self._bridge.relate(atom)
                    if relate_result is None:
                        continue
                    related_nodes = relate_result.get("related_nodes", [])
                    for item in related_nodes:
                        if not isinstance(item, (list, tuple)) or len(item) < 1:
                            continue
                        related_label = str(item[0]).lower()
                        # Find nodes whose semantic_atoms contain this label
                        with self._lock:
                            for node in self._node_index.values():
                                node_labels = {a.lower() for a in node.semantic_atoms}
                                if related_label in node_labels:
                                    # Check if already scored
                                    existing = next(
                                        (s for s, n in scored_nodes if n.node_id == node.node_id),
                                        None,
                                    )
                                    if existing is None:
                                        score = 0.3  # Relate boost
                                        scored_nodes.append((score, node))
                                    else:
                                        # Boost existing score
                                        idx = next(
                                            i for i, (s, n) in enumerate(scored_nodes)
                                            if n.node_id == node.node_id
                                        )
                                        scored_nodes[idx] = (
                                            scored_nodes[idx][0] + 0.1,
                                            scored_nodes[idx][1],
                                        )
            except Exception as exc:
                logger.debug("RSVS relate() failed during retrieve_by_meaning: %s", exc)

        # Sort by score descending
        scored_nodes.sort(key=lambda x: -x[0])

        # Deduplicate by node_id
        seen: set[str] = set()
        results: list[ChatNode] = []
        for _, node in scored_nodes:
            if node.node_id not in seen:
                seen.add(node.node_id)
                results.append(node)
                if len(results) >= top_k:
                    break

        return results

    # ------------------------------------------------------------------
    # retrieve_by_topic — find messages discussing a specific topic
    # ------------------------------------------------------------------

    def retrieve_by_topic(self, topic: str, top_k: int = 10) -> list[ChatNode]:
        """Find all messages discussing a specific topic.

        A message "discusses" a topic if the topic label appears in
        the message's semantic_atoms (case-insensitive match) or if
        the RSVS relate() connects the topic to the message's atoms.

        Analogi: Jin Soun mencari semua percakapan yang membahas
        "racun" — bukan hanya yang menyebut kata itu, tapi juga
        yang membahas konsep terkait.

        Args:
            topic: The topic to search for.
            top_k: Maximum number of results.

        Returns:
            A list of ChatNodes discussing the given topic.
        """
        topic_lower = topic.lower()
        results: list[ChatNode] = []

        with self._lock:
            # First pass: direct match in semantic_atoms
            for node in self._node_index.values():
                node_atoms_lower = {a.lower() for a in node.semantic_atoms}
                if topic_lower in node_atoms_lower:
                    results.append(node)

                # Substring match
                for atom in node_atoms_lower:
                    if topic_lower in atom or atom in topic_lower:
                        if node not in results:
                            results.append(node)
                        break

        # Second pass: RSVS relate() for semantic extension
        if self.rsvs_available:
            try:
                relate_result = self._bridge.relate(topic)
                if relate_result is not None:
                    related_labels = set()
                    for item in relate_result.get("related_nodes", []):
                        if isinstance(item, (list, tuple)) and len(item) >= 1:
                            related_labels.add(str(item[0]).lower())
                        elif isinstance(item, str):
                            related_labels.add(item.lower())

                    with self._lock:
                        for node in self._node_index.values():
                            if node in results:
                                continue
                            node_atoms_lower = {a.lower() for a in node.semantic_atoms}
                            if node_atoms_lower & related_labels:
                                results.append(node)
            except Exception as exc:
                logger.debug("RSVS relate() failed during retrieve_by_topic: %s", exc)

        # Sort by confidence descending
        results.sort(key=lambda n: -n.confidence)

        return results[:top_k]

    # ------------------------------------------------------------------
    # get_conversation — retrieve a ConversationGraph
    # ------------------------------------------------------------------

    def get_conversation(self, conversation_id: str) -> ConversationGraph | None:
        """Get a ConversationGraph by its ID.

        Args:
            conversation_id: The conversation to retrieve.

        Returns:
            The ConversationGraph, or None if not found.
        """
        with self._lock:
            return self._conversations.get(conversation_id)

    # ------------------------------------------------------------------
    # get_recent_topics — top topics across all conversations
    # ------------------------------------------------------------------

    def get_recent_topics(self, n: int = 10) -> list[str]:
        """Get the most frequent topics across all conversations.

        Topics are ranked by frequency (how many conversations discuss them)
        and weighted by confidence from bridge.confidence_map().

        Analogi: Jin Soun bertanya "Apa saja topik yang paling sering
        dibicarakan di semua arsip?" — dia mendapatkan daftar topik
        yang paling sering muncul.

        Args:
            n: Maximum number of topics to return.

        Returns:
            A list of topic strings, ranked by frequency and confidence.
        """
        topic_counts: dict[str, float] = {}

        with self._lock:
            for conv in self._conversations.values():
                for topic in conv.topics:
                    topic_lower = topic.lower()
                    topic_counts[topic_lower] = topic_counts.get(topic_lower, 0.0) + 1.0

        # Boost by RSVS confidence if available
        if self.rsvs_available:
            try:
                cmap = self._bridge.confidence_map()
                for topic in list(topic_counts.keys()):
                    if topic in cmap:
                        topic_counts[topic] += cmap[topic]
            except Exception:
                pass

        # Sort by weighted count descending
        sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])
        return [t for t, _ in sorted_topics[:n]]

    # ------------------------------------------------------------------
    # get_conversation_topics — topics for a specific conversation
    # ------------------------------------------------------------------

    def get_conversation_topics(self, conversation_id: str) -> list[str]:
        """Get topics for a specific conversation.

        Args:
            conversation_id: The conversation to get topics for.

        Returns:
            A list of topic strings, or empty list if conversation not found.
        """
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return []
            return list(conv.topics)

    # ------------------------------------------------------------------
    # detect_topic_shifts — find topic transition points
    # ------------------------------------------------------------------

    def detect_topic_shifts(self, conversation_id: str) -> list[dict]:
        """Detect topic shift points in a conversation.

        Returns a list of dicts, each containing:
        - node_id: The ChatNode where the shift occurs
        - topic_before: The dominant topic before the shift
        - topic_after: The dominant topic after the shift
        - shift_magnitude: How dramatic the shift is (0.0–1.0)

        Analogi: Jin Soun menandai momen-momen ketika percakapan
        bergeser topik — "dari obat ke politik", "dari cuaca ke
        strategi". Ini membantunya memahami alur percakapan.

        Args:
            conversation_id: The conversation to analyze.

        Returns:
            A list of shift dicts, sorted by turn_index.
        """
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None or len(conv.nodes) < 2:
                return []

        shifts: list[dict] = []

        with self._lock:
            nodes = list(conv.nodes)  # Copy to avoid holding lock too long

        for i in range(1, len(nodes)):
            prev_node = nodes[i - 1]
            curr_node = nodes[i]

            prev_atoms = set(a.lower() for a in prev_node.semantic_atoms)
            curr_atoms = set(a.lower() for a in curr_node.semantic_atoms)

            if not prev_atoms or not curr_atoms:
                continue

            # Compute overlap ratio
            overlap = prev_atoms & curr_atoms
            union = prev_atoms | curr_atoms
            overlap_ratio = len(overlap) / len(union) if union else 0.0
            shift_magnitude = 1.0 - overlap_ratio

            # Only report significant shifts (> topic shift threshold)
            if overlap_ratio < _TOPIC_SHIFT_OVERLAP or shift_magnitude > 0.5:
                # Determine topic before and after
                topic_before = self._dominant_atom(prev_node.semantic_atoms)
                topic_after = self._dominant_atom(curr_node.semantic_atoms)

                shifts.append({
                    "node_id": curr_node.node_id,
                    "topic_before": topic_before,
                    "topic_after": topic_after,
                    "shift_magnitude": round(shift_magnitude, 3),
                    "turn_index": curr_node.turn_index,
                    "timestamp": curr_node.timestamp,
                })

        return shifts

    # ------------------------------------------------------------------
    # get_semantic_links — get edges for a specific node
    # ------------------------------------------------------------------

    def get_semantic_links(self, node_id: str) -> list[ChatEdge]:
        """Get all semantic link edges connected to a node.

        Returns edges of all types (not just "semantic_link") that
        have the given node as either source or target.

        Args:
            node_id: The ChatNode to get links for.

        Returns:
            A list of ChatEdges connected to this node.
        """
        with self._lock:
            edges: list[ChatEdge] = []
            for edge_id in self._outgoing_edges.get(node_id, []):
                edge = self._edge_index.get(edge_id)
                if edge is not None:
                    edges.append(edge)
            for edge_id in self._incoming_edges.get(node_id, []):
                edge = self._edge_index.get(edge_id)
                if edge is not None:
                    edges.append(edge)
            return edges

    # ------------------------------------------------------------------
    # get_statistics — summary statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Get summary statistics about the chat index.

        Returns:
            A dict with total counts and breakdowns.
        """
        with self._lock:
            total_nodes = len(self._node_index)
            total_edges = len(self._edge_index)
            total_conversations = len(self._conversations)

            # Edge type breakdown
            edge_type_counts: dict[str, int] = {}
            for edge in self._edge_index.values():
                edge_type_counts[edge.edge_type] = edge_type_counts.get(edge.edge_type, 0) + 1

            # Role breakdown
            role_counts: dict[str, int] = {}
            for node in self._node_index.values():
                role_counts[node.role] = role_counts.get(node.role, 0) + 1

            # Average confidence
            avg_confidence = 0.0
            if total_nodes > 0:
                avg_confidence = sum(n.confidence for n in self._node_index.values()) / total_nodes

            # Average atoms per node
            avg_atoms = 0.0
            if total_nodes > 0:
                avg_atoms = sum(len(n.semantic_atoms) for n in self._node_index.values()) / total_nodes

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "total_conversations": total_conversations,
                "edge_type_counts": edge_type_counts,
                "role_counts": role_counts,
                "avg_confidence": round(avg_confidence, 3),
                "avg_atoms_per_node": round(avg_atoms, 1),
                "rsvs_available": self.rsvs_available,
                "max_nodes": _MAX_NODES,
            }

    # ------------------------------------------------------------------
    # Persistence — save/load
    # ------------------------------------------------------------------

    def save_to_dict(self) -> dict:
        """Serialize all conversation graphs to a plain dict.

        Returns:
            A dict containing the full serializable state.
        """
        with self._lock:
            conversations = {
                cid: conv.to_dict()
                for cid, conv in self._conversations.items()
            }
            last_msg = {
                cid: {
                    "node_id": info[0],
                    "timestamp": info[1],
                    "atoms": info[2],
                }
                for cid, info in self._last_msg_per_conv.items()
            }

        return {
            "schema_version": _PERSIST_SCHEMA_VERSION,
            "conversations": conversations,
            "last_msg_per_conv": last_msg,
            "conv_counter": self._conv_counter,
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore state from a plain dict.

        Args:
            data: A dict previously returned by save_to_dict().
        """
        if not isinstance(data, dict):
            logger.warning("load_from_dict: expected dict, got %s", type(data).__name__)
            return

        saved_version = data.get("schema_version", "0.0")
        if saved_version != _PERSIST_SCHEMA_VERSION:
            logger.warning(
                "load_from_dict: schema version mismatch (saved=%s, current=%s). "
                "Proceeding with best-effort restore.",
                saved_version, _PERSIST_SCHEMA_VERSION,
            )

        # Clear current state
        with self._lock:
            self._conversations.clear()
            self._node_index.clear()
            self._edge_index.clear()
            self._node_to_conv.clear()
            self._outgoing_edges.clear()
            self._incoming_edges.clear()
            self._last_msg_per_conv.clear()

            # Restore conversations
            for cid, conv_data in data.get("conversations", {}).items():
                conv = ConversationGraph.from_dict(conv_data)
                self._conversations[cid] = conv

                # Rebuild indexes
                for node in conv.nodes:
                    self._node_index[node.node_id] = node
                    self._node_to_conv[node.node_id] = cid
                    self._outgoing_edges[node.node_id] = []
                    self._incoming_edges[node.node_id] = []

                for edge in conv.edges:
                    self._edge_index[edge.edge_id] = edge
                    self._outgoing_edges.setdefault(edge.from_node, []).append(edge.edge_id)
                    self._incoming_edges.setdefault(edge.to_node, []).append(edge.edge_id)

            # Restore last message tracking
            for cid, info in data.get("last_msg_per_conv", {}).items():
                self._last_msg_per_conv[cid] = (
                    info.get("node_id", ""),
                    info.get("timestamp", 0.0),
                    info.get("atoms", []),
                )

            self._conv_counter = data.get("conv_counter", 0)

        logger.info(
            "SemanticChatIndex state restored: %d conversations, %d nodes, %d edges",
            len(self._conversations), len(self._node_index), len(self._edge_index),
        )

    def save(self, path: str) -> dict:
        """Save state to a JSON file.

        Args:
            path: Filesystem path to write the JSON file.

        Returns:
            A summary dict with stats about what was saved.
        """
        data = self.save_to_dict()
        summary: dict = {
            "path": path,
            "conversations": len(self._conversations),
            "nodes": len(self._node_index),
            "edges": len(self._edge_index),
            "schema_version": _PERSIST_SCHEMA_VERSION,
            "success": False,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            summary["success"] = True
            logger.info("SemanticChatIndex state saved to %s", path)
        except (OSError, TypeError) as exc:
            summary["error"] = str(exc)
            logger.error("SemanticChatIndex save failed: %s", exc)
        return summary

    def load(self, path: str) -> dict:
        """Load state from a JSON file.

        Args:
            path: Filesystem path to read the JSON file from.

        Returns:
            A summary dict with stats about what was loaded.
        """
        summary: dict = {
            "path": path,
            "conversations": 0,
            "nodes": 0,
            "success": False,
        }
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_from_dict(data)
            summary["conversations"] = len(self._conversations)
            summary["nodes"] = len(self._node_index)
            summary["schema_version"] = data.get("schema_version", "unknown")
            summary["success"] = True
            logger.info("SemanticChatIndex state loaded from %s", path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            summary["error"] = str(exc)
            logger.error("SemanticChatIndex load failed: %s", exc)
        return summary

    # ==================================================================
    # Internal methods
    # ==================================================================

    def _extract_semantic_atoms(self, role: str, content: str) -> list[str]:
        """Extract semantic atoms from a message using RSVS.

        Steps:
        1. Ingest the message into the RSVS graph
        2. Use bridge.senses() on keywords to get concept labels
        3. Use bridge.relate() to find related concepts
        4. Use bridge.query() for composition information
        5. Collect unique concept labels as semantic_atoms

        Falls back to keyword extraction when RSVS unavailable.

        Args:
            role: The speaker role (used for ingest formatting).
            content: The message text.

        Returns:
            A list of semantic atom labels.
        """
        atoms: list[str] = []
        seen: set[str] = set()

        # Ingest into RSVS graph
        if self.rsvs_available:
            try:
                ingest_text = f"[{role}] {content}"
                self._bridge.ingest(ingest_text)
            except Exception as exc:
                logger.debug("RSVS ingest failed: %s", exc)

        # Extract keywords first (always useful)
        keywords = _extract_keywords(content)

        if not self.rsvs_available:
            return keywords

        # Use RSVS senses() on each keyword
        for keyword in keywords[:15]:
            try:
                senses = self._bridge.senses(keyword)
                if senses is not None and isinstance(senses, list):
                    for sense in senses:
                        if isinstance(sense, dict):
                            # Get core atoms from the sense
                            core_atoms = sense.get("core_atoms", [])
                            for atom in core_atoms:
                                if isinstance(atom, str) and atom not in seen:
                                    atoms.append(atom)
                                    seen.add(atom)

                                    # Also add the keyword itself
                                    if keyword not in seen:
                                        atoms.append(keyword)
                                        seen.add(keyword)
                        elif isinstance(sense, str) and sense not in seen:
                            atoms.append(sense)
                            seen.add(sense)
            except Exception:
                pass

        # Use RSVS relate() to find related concepts
        for keyword in keywords[:5]:
            try:
                relate_result = self._bridge.relate(keyword)
                if relate_result is not None:
                    for item in relate_result.get("related_nodes", []):
                        if isinstance(item, (list, tuple)) and len(item) >= 1:
                            label = str(item[0])
                            if label not in seen:
                                atoms.append(label)
                                seen.add(label)
                        elif isinstance(item, str) and item not in seen:
                            atoms.append(item)
                            seen.add(item)
            except Exception:
                pass

        # Use RSVS query() for composition information
        for keyword in keywords[:3]:
            try:
                query_result = self._bridge.query(keyword)
                if query_result is not None:
                    for comp in query_result.get("compositions", []):
                        if isinstance(comp, (list, tuple)) and len(comp) >= 1:
                            label = str(comp[0])
                            if label not in seen:
                                atoms.append(label)
                                seen.add(label)
                    for atom_entry in query_result.get("atoms", []):
                        if isinstance(atom_entry, (list, tuple)) and len(atom_entry) >= 1:
                            label = str(atom_entry[0])
                            if label not in seen:
                                atoms.append(label)
                                seen.add(label)
            except Exception:
                pass

        # If RSVS didn't produce anything, fall back to keywords
        if not atoms:
            return keywords

        # Filter out role-prefixed artifacts (e.g., "[user]", "assistant")
        atoms = [a for a in atoms if a.lower() not in _ROLE_ARTIFACTS]

        # If filtering removed everything, fall back to keywords
        if not atoms:
            return keywords

        return atoms

    def _compute_node_confidence(self, semantic_atoms: list[str]) -> float:
        """Compute the average confidence of a node's semantic atoms.

        Uses bridge.confidence_map() to look up each atom's confidence.
        Falls back to 0.5 per atom when RSVS unavailable.

        Args:
            semantic_atoms: The atoms to compute confidence for.

        Returns:
            The average confidence (0.0–1.0).
        """
        if not semantic_atoms:
            return 0.0

        if not self.rsvs_available:
            return 0.5

        try:
            cmap = self._bridge.confidence_map()
            if not cmap:
                return 0.5

            total = 0.0
            count = 0
            for atom in semantic_atoms:
                if atom in cmap:
                    total += cmap[atom]
                    count += 1

            return total / count if count > 0 else 0.5
        except Exception:
            return 0.5

    def _detect_conversation_id(
        self,
        now: float,
        semantic_atoms: list[str],
    ) -> str:
        """Auto-detect which conversation a message belongs to.

        Segmentation rules:
        1. If there's a gap > 300 seconds since the last message → new conversation
        2. If topic_shift magnitude > 0.8 → new conversation segment
        3. Otherwise, continue the most recent conversation

        Args:
            now: Current timestamp (float).
            semantic_atoms: The new message's semantic atoms.

        Returns:
            A conversation_id string.
        """
        if not self._last_msg_per_conv:
            # First message ever → create a new conversation
            return self._new_conversation_id()

        # Find the most recently active conversation
        latest_conv_id: str | None = None
        latest_ts: float = 0.0
        latest_atoms: list[str] = []

        with self._lock:
            for cid, (nid, ts, atoms) in self._last_msg_per_conv.items():
                if ts > latest_ts:
                    latest_ts = ts
                    latest_conv_id = cid
                    latest_atoms = atoms

        if latest_conv_id is None:
            return self._new_conversation_id()

        # Rule 1: Time gap > 300 seconds
        if now - latest_ts > _CONVERSATION_GAP_SECONDS:
            return self._new_conversation_id()

        # Rule 2: Topic shift magnitude > 0.8
        if latest_atoms:
            prev_set = set(a.lower() for a in latest_atoms)
            curr_set = set(a.lower() for a in semantic_atoms)
            union = prev_set | curr_set
            overlap = prev_set & curr_set
            if union:
                overlap_ratio = len(overlap) / len(union)
                shift_magnitude = 1.0 - overlap_ratio
                if shift_magnitude > _TOPIC_SHIFT_MAGNITUDE:
                    return self._new_conversation_id()

        # Continue the most recent conversation
        return latest_conv_id

    def _new_conversation_id(self) -> str:
        """Generate a new unique conversation ID.

        Returns:
            An 8-char hex conversation ID.
        """
        with self._lock:
            self._conv_counter += 1
        return uuid.uuid4().hex[:8]

    def _get_or_create_conversation(self, conversation_id: str) -> ConversationGraph:
        """Get an existing ConversationGraph or create a new one.

        Must be called within a lock context or the result used
        within the same lock acquisition.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The ConversationGraph for this conversation.
        """
        conv = self._conversations.get(conversation_id)
        if conv is None:
            conv = ConversationGraph(conversation_id=conversation_id)
            self._conversations[conversation_id] = conv
        return conv

    def _create_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: str,
        weight: float = 0.5,
        metadata: dict | None = None,
    ) -> ChatEdge:
        """Create a ChatEdge and register it in the indexes.

        Must be called within a lock context.

        Args:
            from_node: Source node ID.
            to_node: Target node ID.
            edge_type: Type of semantic relationship.
            weight: Strength of the link.
            metadata: Additional info.

        Returns:
            The created ChatEdge.
        """
        edge_id = uuid.uuid4().hex[:8]
        edge = ChatEdge(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node,
            edge_type=edge_type,
            weight=max(0.0, min(1.0, weight)),
            metadata=metadata or {},
        )

        self._edge_index[edge_id] = edge

        # Add to conversation graph
        conv_id = self._node_to_conv.get(from_node)
        if conv_id:
            conv = self._conversations.get(conv_id)
            if conv:
                conv.edges.append(edge)
                conv.refresh_counts()

        # Update adjacency indexes
        self._outgoing_edges.setdefault(from_node, []).append(edge_id)
        self._incoming_edges.setdefault(to_node, []).append(edge_id)

        return edge

    def _detect_semantic_links(
        self,
        new_node: ChatNode,
        conv_graph: ConversationGraph,
    ) -> None:
        """Detect semantic links between the new node and earlier nodes.

        Must be called within a lock context.

        Detection rules:
        - "semantic_link": semantic_atoms overlap > 30% with an earlier message
        - "topic_shift": semantic_atoms have < 10% overlap with previous message
        - "elaboration": new atoms are a subset of previous atoms AND same role
        - "contradiction": bridge.appraise() returns "disagree"

        Args:
            new_node: The newly created ChatNode.
            conv_graph: The ConversationGraph it belongs to.
        """
        new_atoms = set(a.lower() for a in new_node.semantic_atoms)
        if not new_atoms:
            return

        # Check against all previous nodes in this conversation
        for prev_node in conv_graph.nodes:
            if prev_node.node_id == new_node.node_id:
                continue

            prev_atoms = set(a.lower() for a in prev_node.semantic_atoms)
            if not prev_atoms:
                continue

            # Compute Jaccard overlap
            intersection = new_atoms & prev_atoms
            union = new_atoms | prev_atoms
            overlap_ratio = len(intersection) / len(union) if union else 0.0

            # "semantic_link": overlap > 30%
            if overlap_ratio > _SEMANTIC_LINK_OVERLAP:
                self._create_edge(
                    from_node=new_node.node_id,
                    to_node=prev_node.node_id,
                    edge_type="semantic_link",
                    weight=overlap_ratio,
                    metadata={"overlap_ratio": round(overlap_ratio, 3)},
                )

            # "topic_shift": < 10% overlap with the immediately previous node
            # (only check against the most recent previous node)
            if (
                prev_node.turn_index == new_node.turn_index - 1
                and overlap_ratio < _TOPIC_SHIFT_OVERLAP
            ):
                self._create_edge(
                    from_node=new_node.node_id,
                    to_node=prev_node.node_id,
                    edge_type="topic_shift",
                    weight=1.0 - overlap_ratio,
                    metadata={"overlap_ratio": round(overlap_ratio, 3)},
                )

            # "elaboration": new atoms are a subset of previous AND same role
            if (
                new_atoms.issubset(prev_atoms)
                and new_node.role == prev_node.role
                and len(new_atoms) > 0
                and len(new_atoms) < len(prev_atoms)
            ):
                subset_ratio = len(new_atoms) / len(prev_atoms)
                self._create_edge(
                    from_node=new_node.node_id,
                    to_node=prev_node.node_id,
                    edge_type="elaboration",
                    weight=subset_ratio,
                    metadata={"subset_ratio": round(subset_ratio, 3)},
                )

            # "contradiction": appraise() returns "disagree"
            if self.rsvs_available and overlap_ratio > 0.1:
                try:
                    appraise_result = self._bridge.appraise(
                        f"{new_node.content} vs {prev_node.content}"
                    )
                    if isinstance(appraise_result, dict):
                        verdict = appraise_result.get("verdict", "")
                        disagree_pct = appraise_result.get("disagree_pct", 0.0)
                        if verdict == "disagree" or disagree_pct > 0.5:
                            self._create_edge(
                                from_node=new_node.node_id,
                                to_node=prev_node.node_id,
                                edge_type="contradiction",
                                weight=disagree_pct,
                                metadata={
                                    "verdict": verdict,
                                    "disagree_pct": disagree_pct,
                                },
                            )
                except Exception as exc:
                    logger.debug("Appraise for contradiction detection failed: %s", exc)

    def _update_conversation_topics(self, conversation_id: str) -> None:
        """Update the topics list for a conversation.

        Topics are extracted by:
        1. Collecting all semantic_atoms from all nodes
        2. Counting frequency
        3. Using bridge.confidence_map() to rank by confidence
        4. Keeping top 5-10 topics

        Must be called within a lock context or the conversation
        graph is already acquired.

        Args:
            conversation_id: The conversation to update topics for.
        """
        conv = self._conversations.get(conversation_id)
        if conv is None:
            return

        # Collect all semantic atoms
        atom_counts: dict[str, float] = {}
        for node in conv.nodes:
            for atom in node.semantic_atoms:
                atom_lower = atom.lower()
                # Skip role artifacts
                if atom_lower in _ROLE_ARTIFACTS:
                    continue
                atom_counts[atom_lower] = atom_counts.get(atom_lower, 0.0) + 1.0

        # Boost by RSVS confidence
        if self.rsvs_available:
            try:
                cmap = self._bridge.confidence_map()
                for atom in list(atom_counts.keys()):
                    if atom in cmap:
                        atom_counts[atom] += cmap[atom] * 2.0
            except Exception:
                pass

        # Sort by weighted count
        sorted_atoms = sorted(atom_counts.items(), key=lambda x: -x[1])

        # Determine number of topics
        n_topics = min(_MAX_TOPICS_PER_CONV, max(_MIN_TOPICS_PER_CONV, len(sorted_atoms)))

        conv.topics = [atom for atom, _ in sorted_atoms[:n_topics]]

    def _generate_summary(self, conv_graph: ConversationGraph) -> str:
        """Auto-generate a summary for a conversation.

        Uses the conversation's topics and first/last messages to
        create a brief summary string.

        Analogi: Jin Soun menulis ringkasan di kepala gulungan arsip:
        "Percakapan tentang X, Y, Z antara user dan assistant."

        Args:
            conv_graph: The conversation to summarize.

        Returns:
            A summary string.
        """
        if not conv_graph.nodes:
            return "Empty conversation."

        topics_str = ", ".join(conv_graph.topics[:5]) if conv_graph.topics else "no clear topics"

        # Get first message preview
        first_msg = conv_graph.nodes[0]
        first_preview = first_msg.content[:80] + ("..." if len(first_msg.content) > 80 else "")

        # Count roles
        role_counts: dict[str, int] = {}
        for node in conv_graph.nodes:
            role_counts[node.role] = role_counts.get(node.role, 0) + 1

        role_str = ", ".join(f"{count} {role}" for role, count in role_counts.items())

        return (
            f"Conversation about {topics_str}. "
            f"Started with: \"{first_preview}\" "
            f"({role_str}, {conv_graph.node_count} messages)."
        )

    @staticmethod
    def _dominant_atom(semantic_atoms: list[str]) -> str:
        """Get the most representative atom from a list.

        For simplicity, returns the first atom. In a more sophisticated
        implementation, this would use frequency or confidence weighting.

        Args:
            semantic_atoms: The list of atoms.

        Returns:
            The dominant atom, or "unknown" if empty.
        """
        return semantic_atoms[0] if semantic_atoms else "unknown"

    def _auto_prune(self) -> None:
        """Auto-prune oldest nodes when exceeding _MAX_NODES.

        Removes the oldest conversation's nodes and edges until
        total node count is within the limit.

        Must be called within a lock context.
        """
        while len(self._node_index) > _MAX_NODES:
            # Find the oldest conversation
            oldest_conv_id: str | None = None
            oldest_time = float("inf")

            for cid, conv in self._conversations.items():
                if conv.nodes and conv.created_at < oldest_time:
                    oldest_time = conv.created_at
                    oldest_conv_id = cid

            if oldest_conv_id is None:
                break

            conv = self._conversations.get(oldest_conv_id)
            if conv is None:
                break

            # Remove nodes one at a time from the oldest conversation
            if conv.nodes:
                node = conv.nodes.pop(0)
                self._remove_node_from_indexes(node)
                conv.refresh_counts()
            else:
                # Empty conversation — remove it
                del self._conversations[oldest_conv_id]
                self._last_msg_per_conv.pop(oldest_conv_id, None)

    def _remove_node_from_indexes(self, node: ChatNode) -> None:
        """Remove a ChatNode and its associated edges from all indexes.

        Must be called within a lock context.

        Args:
            node: The ChatNode to remove.
        """
        node_id = node.node_id

        # Remove all outgoing edges
        for edge_id in list(self._outgoing_edges.get(node_id, [])):
            edge = self._edge_index.pop(edge_id, None)
            if edge:
                # Remove from incoming edges of target
                incoming = self._incoming_edges.get(edge.to_node, [])
                if edge_id in incoming:
                    incoming.remove(edge_id)

        # Remove all incoming edges
        for edge_id in list(self._incoming_edges.get(node_id, [])):
            edge = self._edge_index.pop(edge_id, None)
            if edge:
                # Remove from outgoing edges of source
                outgoing = self._outgoing_edges.get(edge.from_node, [])
                if edge_id in outgoing:
                    outgoing.remove(edge_id)

        # Remove from indexes
        self._node_index.pop(node_id, None)
        self._node_to_conv.pop(node_id, None)
        self._outgoing_edges.pop(node_id, None)
        self._incoming_edges.pop(node_id, None)
