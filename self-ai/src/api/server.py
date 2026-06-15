# @WHO:   self-ai/src/api/server.py
# @WHAT:  Thin HTTP API layer on top of SelfCore — FastAPI app
# @PART:  self-ai/api
# @ENTRY: uvicorn src.api.server:app

"""FastAPI server — HTTP API layer for SELF-AI.

Exposes 4 endpoints over SelfCore:

    POST /answer       → ask a question, get an answer
    POST /learn        → store a correction as a new UnderstandingNode
    POST /reinforce    → strengthen nodes that produced a correct answer
    GET  /introspect   → snapshot of the understanding graph

Startup behaviour:
    Models (Qwen3-0.6B + bge-m3) and SelfCore are **not** loaded at import
    time.  On the first request the singleton initializer fires; subsequent
    requests reuse the cached instances.  Until loading completes every
    endpoint returns HTTP 503 with ``{"error": "model not ready, retry"}``.

Environment variables:
    QWEN_PATH   — path to Qwen3-0.6B model directory
                  (default: <repo_root>/dependencies/models/Qwen3-0.6B)
    BGE_PATH    — path to bge-m3 model directory
                  (default: <repo_root>/dependencies/models/bge-m3)

Run:
    cd self-ai
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
#  Resolve repo root so we can set default model paths
# ──────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

# ──────────────────────────────────────────────────────────
#  Lazy-loaded singleton state
# ──────────────────────────────────────────────────────────

_core: Optional[object] = None      # SelfCore instance
_loading: bool = False               # True while first-load is in progress
_ready: bool = False                 # True once SelfCore is usable


def _ensure_core():  # noqa: C901 — simple sequential init, not complex
    """Initialise SelfCore (and its models) on first call.

    On success, sets module-level ``_core`` and ``_ready``.
    Returns the core instance, or *None* if loading fails.
    """
    global _core, _loading, _ready

    if _ready and _core is not None:
        return _core

    if _loading:
        # Another coroutine / thread is already loading — bail out
        return None

    _loading = True
    try:
        # ── 1. Set model paths via env vars if not already set ──
        qwen_path = os.environ.get(
            "QWEN_PATH",
            os.path.join(_REPO_ROOT, "dependencies", "models", "Qwen3-0.6B"),
        )
        bge_path = os.environ.get(
            "BGE_PATH",
            os.path.join(_REPO_ROOT, "dependencies", "models", "bge-m3"),
        )

        # Point HuggingFace at local directories so model_registry /
        # SensoryLayer can find the weights without downloading.
        # We only override if the local directory actually exists.
        if os.path.isdir(qwen_path):
            os.environ.setdefault("QWEN_PATH", qwen_path)
            logger.info("QWEN_PATH set to %s", qwen_path)
        else:
            logger.warning("QWEN_PATH directory not found: %s", qwen_path)

        if os.path.isdir(bge_path):
            os.environ.setdefault("BGE_PATH", bge_path)
            logger.info("BGE_PATH set to %s", bge_path)
        else:
            logger.warning("BGE_PATH directory not found: %s", bge_path)

        # ── 2. Ensure src/ is on sys.path so internal imports work ──
        src_dir = os.path.join(_REPO_ROOT, "self-ai", "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        config_dir = os.path.join(_REPO_ROOT, "self-ai")
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)

        # ── 3. Import and instantiate SelfCore ──
        from core.self import SelfCore  # type: ignore[import-untyped]

        logger.info("Initializing SelfCore (first request) …")
        _core = SelfCore()
        _ready = True
        logger.info("SelfCore ready.")
        return _core

    except Exception:
        logger.exception("Failed to initialize SelfCore")
        _core = None
        _ready = False
        return None
    finally:
        _loading = False


# ──────────────────────────────────────────────────────────
#  FastAPI app & request / response schemas
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title="SELF-AI API",
    version="0.1.0",
    description="Thin HTTP layer over SelfCore — answer, learn, reinforce, introspect",
)


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=2048,
        description="Override max_new_tokens for generation",
    )


class AnswerResponse(BaseModel):
    answer: str


class LearnRequest(BaseModel):
    question: str = Field(..., min_length=1)
    wrong_answer: str = Field(..., min_length=1)
    correction: str = Field(..., min_length=1)


class LearnResponse(BaseModel):
    node_id: Optional[str] = None
    experience: str
    confidence: float
    graph_size: int
    duplicate: bool


class ReinforceRequest(BaseModel):
    question: str = Field(..., min_length=1)
    confirmed_answer: str = Field(..., min_length=1)


class ReinforceResponse(BaseModel):
    reinforced_count: int
    node_ids: list[str]
    new_confidences: dict[str, float]


class IntrospectResponse(BaseModel):
    graph_size: int
    avg_confidence: float
    top_nodes: list[dict]
    recent_nodes: list[dict]
    sources: dict[str, int]
    status: str


# ──────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    """Ask SELF a question and get an answer string.

    Internally calls ``SelfCore.process(question)`` which runs the full
    8-layer pipeline (sensory → curiosity) and returns the derivation
    result.  The *answer* field is extracted from the derivation layer.
    """
    core = _ensure_core()
    if core is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not ready, retry"},
        )

    try:
        result = core.process(req.question)
    except Exception as exc:
        logger.exception("/answer — process() raised")
        return JSONResponse(
            status_code=500,
            content={"error": f"process failed: {exc}"},
        )

    # The answer lives in the derivation sub-dict
    derivation = result.get("derivation", result)
    answer_text = derivation.get("answer")

    if answer_text is None:
        answer_text = ""

    return AnswerResponse(answer=str(answer_text))


@app.post("/learn", response_model=LearnResponse)
def learn(req: LearnRequest):
    """Store a correction as a new UnderstandingNode.

    When a user corrects SELF's answer, this endpoint records the
    (question, wrong_answer, correction) triple as a new experience
    node in the understanding graph.  Duplicate corrections are
    idempotent — the existing node is touched instead of re-added.
    """
    core = _ensure_core()
    if core is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not ready, retry"},
        )

    try:
        result = core.learn(
            question=req.question,
            wrong_answer=req.wrong_answer,
            correction=req.correction,
        )
    except Exception as exc:
        logger.exception("/learn — learn() raised")
        return JSONResponse(
            status_code=500,
            content={"error": f"learn failed: {exc}"},
        )

    return LearnResponse(
        node_id=result.get("node_id"),
        experience=result.get("experience", ""),
        confidence=result.get("confidence", 0.0),
        graph_size=result.get("graph_size", 0),
        duplicate=result.get("duplicate", False),
    )


@app.post("/reinforce", response_model=ReinforceResponse)
def reinforce(req: ReinforceRequest):
    """Strengthen nodes that contributed to a confirmed-correct answer.

    Finds nodes in the understanding graph whose experience text
    contains the confirmed answer and bumps their confidence,
    making them more influential in future unconscious injections.
    """
    core = _ensure_core()
    if core is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not ready, retry"},
        )

    try:
        result = core.reinforce(
            question=req.question,
            confirmed_answer=req.confirmed_answer,
        )
    except Exception as exc:
        logger.exception("/reinforce — reinforce() raised")
        return JSONResponse(
            status_code=500,
            content={"error": f"reinforce failed: {exc}"},
        )

    return ReinforceResponse(
        reinforced_count=result.get("reinforced_count", 0),
        node_ids=result.get("node_ids", []),
        new_confidences=result.get("new_confidences", {}),
    )


@app.get("/introspect", response_model=IntrospectResponse)
def introspect():
    """Return a snapshot of what SELF currently knows.

    No model call, no tokenizer, no network — purely reads state
    from the UnderstandingGraph (graph size, avg confidence,
    top / recent nodes, source distribution).
    """
    core = _ensure_core()
    if core is None:
        return JSONResponse(
            status_code=503,
            content={"error": "model not ready, retry"},
        )

    try:
        result = core.introspect()
    except Exception as exc:
        logger.exception("/introspect — introspect() raised")
        return JSONResponse(
            status_code=500,
            content={"error": f"introspect failed: {exc}"},
        )

    return IntrospectResponse(
        graph_size=result.get("graph_size", 0),
        avg_confidence=result.get("avg_confidence", 0.0),
        top_nodes=result.get("top_nodes", []),
        recent_nodes=result.get("recent_nodes", []),
        sources=result.get("sources", {}),
        status=result.get("status", "empty"),
    )
