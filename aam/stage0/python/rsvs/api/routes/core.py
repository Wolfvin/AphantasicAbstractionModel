"""Core RSVS API routes — run, ingest, query, compose, appraise, relate, etc.

These are the primary CRUD-style endpoints that drive the main application flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..._version import __version__
from ...exceptions import RustCoreUnavailableError
from ...modes import run_mode
from ...rsvs_core import get_rsvs_instance
from ..deps import (
    _enriched_node_info,
    _enriched_query_result,
    _enriched_sense_info,
    _verify_api_key,
    limiter,
)
from ..schemas import (
    AppraiseRequest,
    ComposeRequest,
    IngestRequest,
    NodeInfoRequest,
    QueryRequest,
    RelateRequest,
    RunRequest,
    SensesRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/run")
@limiter.limit("30/minute")
async def run_endpoint(request: Request, req: RunRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    rsvs = get_rsvs_instance()
    return await asyncio.to_thread(
        run_mode, rsvs, req.mode,
        text=req.text, target=req.target, source=req.source, options=req.options,
    )


@router.post("/ingest")
@limiter.limit("30/minute")
async def ingest_endpoint(request: Request, req: IngestRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    rsvs = get_rsvs_instance()
    return await asyncio.to_thread(run_mode, rsvs, "ingest", text=req.text, source=req.source)


@router.post("/query")
@limiter.limit("30/minute")
async def query_endpoint(request: Request, req: QueryRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    rsvs = get_rsvs_instance()
    result = await asyncio.to_thread(rsvs.query, req.text, req.context or "")
    if result is None:
        return {"ok": True, "result": None}
    enriched = _enriched_query_result(result)
    return {"ok": True, "result": enriched}


@router.post("/appraise")
@limiter.limit("30/minute")
async def appraise_endpoint(request: Request, req: AppraiseRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    rsvs = get_rsvs_instance()
    return await asyncio.to_thread(run_mode, rsvs, "appraise", text=req.target)


@router.post("/relate")
@limiter.limit("30/minute")
async def relate_endpoint(request: Request, req: RelateRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    rsvs = get_rsvs_instance()
    return await asyncio.to_thread(run_mode, rsvs, "relate", text=req.source, target=req.target)


@router.post("/compose", tags=["compose"])
@limiter.limit("30/minute")
async def compose_node(request: Request, body: ComposeRequest, _auth: None = Depends(_verify_api_key)):
    """Create a composite node from explicit compositions or atom IDs."""
    rsvs = get_rsvs_instance()
    if rsvs is None:
        raise RustCoreUnavailableError()

    req = body

    if req.compositions and req.atom_ids:
        raise HTTPException(status_code=400, detail="Provide either 'compositions' or 'atom_ids', not both.")

    if not req.compositions and not req.atom_ids:
        raise HTTPException(status_code=400, detail="Either 'compositions' or 'atom_ids' must be provided.")

    sense_count_before = 0
    try:
        sense_count_before = len(await asyncio.to_thread(rsvs.senses, req.label))
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Could not read sense count before compose: %s", exc)

    if req.compositions:
        compositions_tuples = [(c.label, c.sense_id) for c in req.compositions]
        node_id = await asyncio.to_thread(rsvs.compose, req.label, compositions_tuples, req.lang)
        composed_from = [{"label": c.label, "sense_id": c.sense_id} for c in req.compositions]
        composed_type = "compositions"
    else:
        node_id = await asyncio.to_thread(rsvs.compose_from_ids, req.label, req.atom_ids, req.lang)
        composed_from = req.atom_ids
        composed_type = "atom_ids"

    if req.condition_label:
        try:
            sense_count_after = len(await asyncio.to_thread(rsvs.senses, req.label))
            target_idx = sense_count_after - 1 if sense_count_after > sense_count_before else 0
            await asyncio.to_thread(rsvs.set_sense_label, req.label, target_idx, req.condition_label)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("Could not set sense label after compose: %s", exc)

    snapshot = await asyncio.to_thread(rsvs.snapshot_v1)
    events = await asyncio.to_thread(rsvs.consume_events_v1)
    return {
        "ok": True,
        "status": "ok",
        "node_id": node_id,
        "label": req.label,
        composed_type: composed_from,
        "lang": req.lang,
        "snapshot": snapshot,
        "events": events,
    }


@router.post("/node-info", tags=["node"])
@limiter.limit("30/minute")
async def node_info_endpoint(request: Request, req: NodeInfoRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    """Get enriched node info (v5.0: includes layer field)."""
    rsvs = get_rsvs_instance()
    info = await asyncio.to_thread(rsvs.node_info, req.label)
    if info is None:
        return {"ok": True, "result": None}
    enriched = _enriched_node_info(info)
    return {"ok": True, "result": enriched}


@router.post("/senses", tags=["node"])
@limiter.limit("30/minute")
async def senses_endpoint(request: Request, req: SensesRequest, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    """Get sense info for a label (v5.0: includes layer, grounding_score, compositions)."""
    rsvs = get_rsvs_instance()
    senses = await asyncio.to_thread(rsvs.senses, req.label)
    enriched_senses = [_enriched_sense_info(s) for s in senses]
    return {"ok": True, "label": req.label, "senses": enriched_senses}


@router.get("/snapshot")
@limiter.limit("60/minute")
async def snapshot_endpoint(request: Request, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    rsvs = get_rsvs_instance()
    raw = await asyncio.to_thread(rsvs.snapshot_v1)
    return json.loads(raw)


@router.get("/events")
@limiter.limit("60/minute")
async def events_endpoint(request: Request, _auth: None = Depends(_verify_api_key)) -> dict[str, Any]:
    rsvs = get_rsvs_instance()
    raw = await asyncio.to_thread(rsvs.consume_events_v1)
    return json.loads(raw)


@router.get("/health")
@limiter.limit("60/minute")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/")
@limiter.limit("60/minute")
async def root(request: Request) -> dict[str, str]:
    return {"name": "RSVS", "version": __version__, "docs": "/docs"}
