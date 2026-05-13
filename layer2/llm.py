"""
LLM Bridge — Generate natural narrative FROM graph reasoning chain.

KEY INSIGHT: The LLM doesn't generate from nothing. It generates FROM the graph.
Graph = structural memory, LLM = narrative voice.
Jin Soun = graph, his body = limited LLM.

This module provides two backends:
1. z-ai-web-dev-sdk: Uses the hosted AI model to generate natural narrative
   from structured reasoning chain data. The LLM receives the graph's
   reasoning steps as context — it doesn't hallucinate, it narrates.
   P1-4 fix: Now uses Python SDK directly instead of Node.js subprocess
   to eliminate shell injection risk.

2. Fallback: Generates a structured investigation report template
   when the SDK is unavailable.

Analogi: Jin Soun punya kenangan sempurna (graph) tapi tubuh third-rate (LLM).
Tubuhnya terbatas, tapi karena dia mengakses graph secara langsung,
dia bisa mengeluarkan kesimpulan yang tepat. LLM = tubuh yang mengungkapkan
apa yang graph sudah ketahui.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# z-ai-web-dev-sdk backend
# ---------------------------------------------------------------------------

# Lazy import cache for the Python SDK
_zai_sdk: Any = None
_zai_sdk_checked: bool = False


def _get_zai_sdk() -> Any:
    """Lazy-import the z-ai-web-dev-sdk Python package.

    Returns the ZAI class if available, otherwise None.
    Caches the result to avoid repeated import attempts.
    """
    global _zai_sdk, _zai_sdk_checked
    if _zai_sdk_checked:
        return _zai_sdk
    _zai_sdk_checked = True
    try:
        from z_ai_web_dev_sdk import ZAI  # type: ignore[import]
        _zai_sdk = ZAI
        logger.debug("z-ai-web-dev-sdk Python package available")
    except ImportError:
        _zai_sdk = None
        logger.debug("z-ai-web-dev-sdk Python package not available")
    return _zai_sdk


async def _call_llm_async(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call the LLM via the Python z-ai-web-dev-sdk.

    Args:
        system_prompt: The system prompt for the LLM.
        user_prompt: The user prompt containing graph context.

    Returns:
        The generated narrative text, or None on failure.
    """
    ZAI = _get_zai_sdk()
    if ZAI is None:
        return None

    zai = await ZAI.create()
    completion = await zai.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1500,
    )

    content = ""
    if hasattr(completion, "choices") and completion.choices:
        choice = completion.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            content = choice.message.content or ""

    return content if content else None


def _call_llm_sync(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Synchronously call the LLM by running the async SDK in an event loop.

    Args:
        system_prompt: The system prompt for the LLM.
        user_prompt: The user prompt containing graph context.

    Returns:
        The generated narrative text, or None on failure.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're already in an async context — create a new thread
        # to run the async function to avoid "cannot run from running loop"
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                _call_llm_async(system_prompt, user_prompt),
            )
            return future.result(timeout=90)

    return asyncio.run(_call_llm_async(system_prompt, user_prompt))


def generate_narrative_via_sdk(
    trigger: str,
    reasoning_chain: list[dict],
    pattern: str,
    evidence_nodes: list[str],
    confidence: float,
    anomalies: list[dict],
    language: str = "id",
) -> Optional[str]:
    """Generate natural narrative using z-ai-web-dev-sdk.

    The LLM receives the graph's reasoning chain as structured context.
    It does NOT generate from nothing — it narrates from the graph.

    P1-4 fix: Now uses the Python SDK directly instead of spawning a
    Node.js subprocess with string-interpolated user data. This eliminates
    the shell injection risk from the previous implementation.

    Strategy:
    1. Try the Python z-ai-web-dev-sdk (no subprocess, no injection risk)
    2. If unavailable, fall back to Node.js but with safe JSON stdin/stdout
       instead of string-interpolated command-line args
    3. Retry up to 3 times with exponential backoff

    Args:
        trigger: The original trigger text.
        reasoning_chain: List of reasoning step dicts with keys:
            step_type, description, confidence, evidence_nodes.
        pattern: The completed pattern description.
        evidence_nodes: List of evidence node labels.
        confidence: Overall confidence score (0.0 - 1.0).
        anomalies: List of anomaly dicts.
        language: Output language code ("id" for Indonesian, "en" for English).

    Returns:
        A narrative string, or None if generation failed.
    """
    # Build a structured prompt FROM the graph
    # This is the KEY: we feed graph data to the LLM, not just a question.
    chain_text = _format_reasoning_chain_for_llm(reasoning_chain)
    anomaly_text = _format_anomalies_for_llm(anomalies)
    evidence_text = ", ".join(evidence_nodes[:20]) if evidence_nodes else "none"

    if language == "id":
        system_prompt = (
            "Kamu adalah asisten AI yang menghasilkan narasi investigasi "
            "berdasarkan rantai penalaran dari knowledge graph. "
            "Kamu TIDAK mengarang — kamu hanya menarasikan apa yang sudah "
            "ditentukan oleh analisis graph. Setiap klaim harus bisa "
            "ditelusuri ke evidence node. Gunakan bahasa Indonesia yang "
            "natural dan jelas."
        )
        user_prompt = (
            f"TRIGGER: {trigger}\n\n"
            f"REASONING CHAIN:\n{chain_text}\n\n"
            f"PATTERN:\n{pattern}\n\n"
            f"ANOMALIES:\n{anomaly_text}\n\n"
            f"EVIDENCE NODES: {evidence_text}\n"
            f"CONFIDENCE: {confidence:.0%}\n\n"
            f"Berdasarkan data graph di atas, hasilkan narasi investigasi "
            f"yang menjelaskan: (1) Apa yang memicu analisis ini, "
            f"(2) Apa yang ditemukan di setiap langkah, "
            f"(3) Anomali apa yang terdeteksi, "
            f"(4) Pola apa yang terbentuk, dan "
            f"(5) Kesimpulan dengan tingkat keyakinan. "
            f"Jangan tambahkan informasi yang tidak ada di graph."
        )
    else:
        system_prompt = (
            "You are an AI assistant that generates investigation narratives "
            "based on reasoning chains from a knowledge graph. "
            "You do NOT hallucinate — you only narrate what the graph analysis "
            "has determined. Every claim must be traceable to an evidence node. "
            "Write in clear, natural language."
        )
        user_prompt = (
            f"TRIGGER: {trigger}\n\n"
            f"REASONING CHAIN:\n{chain_text}\n\n"
            f"PATTERN:\n{pattern}\n\n"
            f"ANOMALIES:\n{anomaly_text}\n\n"
            f"EVIDENCE NODES: {evidence_text}\n"
            f"CONFIDENCE: {confidence:.0%}\n\n"
            f"Based on the graph data above, produce an investigation narrative "
            f"that explains: (1) What triggered this analysis, "
            f"(2) What was found at each step, "
            f"(3) What anomalies were detected, "
            f"(4) What pattern emerged, and "
            f"(5) The conclusion with confidence level. "
            f"Do not add information not present in the graph."
        )

    # --- Strategy 1: Try Python SDK (no subprocess, no injection risk) ---
    ZAI = _get_zai_sdk()
    if ZAI is not None:
        for attempt in range(3):
            try:
                result = _call_llm_sync(system_prompt, user_prompt)
                if result:
                    logger.info("LLM narrative generated via Python SDK: %d chars", len(result))
                    return result
                logger.debug("Python SDK returned empty result (attempt %d/3)", attempt + 1)
            except Exception as exc:
                logger.debug("Python SDK attempt %d/3 failed: %s", attempt + 1, exc)
            if attempt < 2:
                time.sleep(2 ** attempt)

    # --- Strategy 2: Fall back to Node.js with safe JSON stdin/stdout ---
    # This avoids shell injection by passing prompts via JSON stdin
    # instead of interpolating them into a command-line string.
    try:
        import subprocess

        # Build the JS script that reads prompts from stdin as JSON
        js_script = """
const ZAI = require('z-ai-web-dev-sdk').default;
(async () => {
  const chunks = [];
  process.stdin.on('data', (c) => chunks.push(c));
  process.stdin.on('end', async () => {
    try {
      const input = JSON.parse(Buffer.concat(chunks).toString());
      const zai = await ZAI.create();
      const completion = await zai.chat.completions.create({
        messages: [
          { role: "system", content: input.system_prompt },
          { role: "user", content: input.user_prompt }
        ],
        temperature: 0.3,
        max_tokens: 1500
      });
      const content = completion.choices[0]?.message?.content || "";
      console.log(JSON.stringify({ narrative: content }));
    } catch (err) {
      process.stderr.write(err.message || String(err));
      process.exit(1);
    }
  });
})();
"""
        # Pass prompts as JSON via stdin — no string interpolation into shell
        stdin_payload = json.dumps({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })

        for attempt in range(3):
            try:
                result = subprocess.run(
                    ["node", "-e", js_script],
                    input=stdin_payload,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )

                if result.returncode == 0 and result.stdout.strip():
                    parsed = json.loads(result.stdout.strip())
                    narrative = parsed.get("narrative", "")
                    if narrative:
                        logger.info("LLM narrative generated via Node.js (safe stdin): %d chars", len(narrative))
                        return narrative

                if result.stderr:
                    logger.debug("LLM SDK stderr: %s", result.stderr[:200])

            except subprocess.TimeoutExpired:
                logger.warning("LLM narrative generation timed out (attempt %d/3)", attempt + 1)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse LLM response: %s (attempt %d/3)", exc, attempt + 1)
            except Exception as exc:
                logger.debug("Node.js attempt %d/3 failed: %s", attempt + 1, exc)

            if attempt < 2:
                time.sleep(2 ** attempt)

    except FileNotFoundError:
        logger.debug("Node.js not found — LLM narrative unavailable")
    except Exception as exc:
        logger.warning("Node.js LLM fallback failed: %s", exc)

    return None


def _format_reasoning_chain_for_llm(chain: list[dict]) -> str:
    """Format reasoning chain steps for LLM context.

    Args:
        chain: List of reasoning step dicts.

    Returns:
        A formatted string summarizing each step.
    """
    if not chain:
        return "(no reasoning steps)"

    lines: list[str] = []
    for i, step in enumerate(chain, 1):
        step_type = step.get("step_type", "unknown")
        description = step.get("description", "")
        confidence = step.get("confidence", 0.0)
        evidence = step.get("evidence_nodes", [])

        line = f"Step {i} [{step_type.upper()}]: {description}"
        if evidence:
            line += f" (evidence: {', '.join(str(e) for e in evidence[:5])})"
        line += f" [confidence: {confidence:.0%}]"
        lines.append(line)

    return "\n".join(lines)


def _format_anomalies_for_llm(anomalies: list[dict]) -> str:
    """Format anomalies for LLM context.

    Args:
        anomalies: List of anomaly dicts.

    Returns:
        A formatted string describing each anomaly.
    """
    if not anomalies:
        return "(no anomalies detected)"

    lines: list[str] = []
    for i, anomaly in enumerate(anomalies, 1):
        anomaly_type = anomaly.get("type", "unknown")
        description = anomaly.get("description", str(anomaly))
        nodes = anomaly.get("nodes", [])

        line = f"Anomaly {i} [{anomaly_type}]: {description}"
        if nodes:
            line += f" (nodes: {', '.join(str(n) for n in nodes[:3])})"
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fallback narrative generator (when LLM SDK is unavailable)
# ---------------------------------------------------------------------------

def generate_narrative_fallback(
    trigger: str,
    reasoning_chain: list[dict],
    pattern: str,
    evidence_nodes: list[str],
    confidence: float,
    anomalies: list[dict],
) -> str:
    """Generate a structured investigation report without LLM.

    Used when the z-ai-web-dev-sdk is not available. Produces a
    structured narrative template based on the graph reasoning chain.

    This is NOT as good as the LLM version — it's template-based,
    not natural language. But it's always available.

    Analogi: Jin Soun tanpa kemampuan bicara — dia masih bisa
    mengkomunikasikan kesimpulannya melalui catatan terstruktur,
    meski tidak se-eloquen saat dia bisa berbicara.

    Args:
        trigger: The original trigger text.
        reasoning_chain: List of reasoning step dicts.
        pattern: The completed pattern description.
        evidence_nodes: List of evidence node labels.
        confidence: Overall confidence score.
        anomalies: List of anomaly dicts.

    Returns:
        A structured investigation report string.
    """
    parts: list[str] = []

    # === Section 1: Trigger ===
    if trigger:
        parts.append(
            f"## Trigger\n\nInput: \"{trigger}\"\n\n"
            f"This triggered a pattern completion analysis across the knowledge graph."
        )

    # === Section 2: Reasoning Chain ===
    if reasoning_chain:
        chain_lines: list[str] = ["## Reasoning Chain"]
        for i, step in enumerate(reasoning_chain, 1):
            step_type = step.get("step_type", "unknown")
            description = step.get("description", "")
            step_conf = step.get("confidence", 0.0)
            evidence = step.get("evidence_nodes", [])

            chain_lines.append(
                f"\n**Step {i}: {step_type.upper()}**\n"
                f"{description}\n"
                f"Confidence: {step_conf:.0%}"
            )
            if evidence:
                chain_lines.append(
                    f"Evidence: {', '.join(str(e) for e in evidence[:8])}"
                )
        parts.append("\n".join(chain_lines))

    # === Section 3: Pattern ===
    if pattern:
        parts.append(f"## Pattern\n\n{pattern}")

    # === Section 4: Anomalies ===
    if anomalies:
        anomaly_lines: list[str] = ["## Anomalies Detected"]
        for i, anomaly in enumerate(anomalies, 1):
            anomaly_type = anomaly.get("type", "unknown")
            description = anomaly.get("description", str(anomaly))
            anomaly_lines.append(
                f"\n**Anomaly {i}** [{anomaly_type}]: {description}"
            )
        parts.append("\n".join(anomaly_lines))

    # === Section 5: Evidence ===
    if evidence_nodes:
        unique_evidence = list(dict.fromkeys(str(e) for e in evidence_nodes))
        evidence_str = ", ".join(unique_evidence[:15])
        parts.append(
            f"## Evidence\n\n"
            f"Grounded in {len(unique_evidence)} knowledge node(s): {evidence_str}."
        )

    # === Section 6: Confidence ===
    confidence_desc = "low"
    if confidence >= 0.7:
        confidence_desc = "high"
    elif confidence >= 0.4:
        confidence_desc = "moderate"

    parts.append(
        f"## Confidence\n\n"
        f"Overall: **{confidence:.0%}** ({confidence_desc}).\n\n"
        f"Each claim in this narrative can be traced back to specific nodes "
        f"in the knowledge graph. This is not probabilistic text generation — "
        f"it is structured reasoning from graph evidence."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Unified narrative generation
# ---------------------------------------------------------------------------

def generate_narrative(
    trigger: str,
    reasoning_chain: list[dict],
    pattern: str,
    evidence_nodes: list[str],
    confidence: float,
    anomalies: list[dict],
    language: str = "id",
    use_llm: bool = True,
) -> str:
    """Generate narrative — try LLM first, fall back to template.

    This is the main entry point for narrative generation. It tries
    the z-ai-web-dev-sdk first (for natural language), then falls
    back to a structured template.

    KEY INSIGHT: Whether LLM or template, the output is always
    derived FROM the graph. The LLM doesn't hallucinate — it
    narrates what the graph has already determined.

    Analogi: Jin Soun bisa mengungkapkan kesimpulannya secara
    lisan (LLM) atau tertulis (template). Apapun medianya,
    isinya sama — karena sumbernya sama: graph kenangannya.

    Args:
        trigger: The original trigger text.
        reasoning_chain: List of reasoning step dicts.
        pattern: The completed pattern description.
        evidence_nodes: List of evidence node labels.
        confidence: Overall confidence score.
        anomalies: List of anomaly dicts.
        language: Output language ("id" or "en").
        use_llm: Whether to try LLM generation (default: True).

    Returns:
        A narrative string — either LLM-generated natural text
        or a structured investigation report template.
    """
    # Try LLM first
    if use_llm:
        llm_narrative = generate_narrative_via_sdk(
            trigger=trigger,
            reasoning_chain=reasoning_chain,
            pattern=pattern,
            evidence_nodes=evidence_nodes,
            confidence=confidence,
            anomalies=anomalies,
            language=language,
        )
        if llm_narrative:
            return llm_narrative

    # Fallback to template
    logger.info("Using fallback narrative generator (LLM unavailable)")
    return generate_narrative_fallback(
        trigger=trigger,
        reasoning_chain=reasoning_chain,
        pattern=pattern,
        evidence_nodes=evidence_nodes,
        confidence=confidence,
        anomalies=anomalies,
    )
