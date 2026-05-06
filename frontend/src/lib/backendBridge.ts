import type { ChatMessage, GraphSnapshot, RSVSEvent, ComposeResult, StructuralSimilarityResult, SubstitutionAnalysisResult, CompositionPair } from '@/lib/types';
export type RSVSMode = 'ingest' | 'appraise' | 'relate' | 'compose' | 'structural_similarity' | 'substitution_analysis' | 'grounding_info' | 'context_query';

export interface BackendIngestResponse {
  ok: boolean;
  correlation_id: string;
  snapshot: GraphSnapshot;
  events: RSVSEvent[];
  messages: ChatMessage[];
  files?: {
    snapshot?: string;
    events?: string;
    report?: string;
  };
  error?: string;
}

interface BackendRunEnvelope {
  ok: boolean;
  mode: RSVSMode;
  correlation_id?: string;
  timestamp: string;
  result: {
    snapshot?: GraphSnapshot;
    events?: RSVSEvent[];
    stats?: {
      token_count: number;
      node_count: number;
      edge_count: number;
      batch_id: string;
    };
    compose?: ComposeResult;
  };
  messages: ChatMessage[];
  files?: {
    snapshot?: string;
    events?: string;
    report?: string;
    appraise?: string;
    relate?: string;
  };
  meta?: {
    version: string;
    atom_dir: string;
    latency_ms: number;
  };
  error?: string;
}

/**
 * Build proxy URL for server-side API routing.
 * All backend calls go through /api/proxy/<path> so the API key
 * is never exposed to the browser (handled server-side in route.ts).
 */
function proxyUrl(path: string): string {
  return `/api/proxy/${path}`;
}

function getAuthHeaders(): Record<string, string> {
  // API key is injected server-side by the proxy route — never sent from client
  return { 'Content-Type': 'application/json' };
}

export async function runModeToBackend(
  mode: RSVSMode,
  text: string,
  correlationId: string,
  options: Record<string, unknown> = {},
): Promise<BackendRunEnvelope> {
  const url = proxyUrl('run');
  const res = await fetch(url, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({
      mode,
      text,
      correlation_id: correlationId,
      options,
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Backend run failed (${res.status}): ${body}`);
  }

  return (await res.json()) as BackendRunEnvelope;
}

export async function ingestToBackend(text: string, correlationId: string): Promise<BackendIngestResponse> {
  const payload = await runModeToBackend('ingest', text, correlationId);
  const snapshot = payload.result?.snapshot || {
    snapshot_id: '',
    generated_at: new Date().toISOString(),
    context: { domain: 'unknown', batch_id: '', input_message_id: correlationId },
    nodes: [],
    edges: [],
  };
  return {
    ok: payload.ok,
    correlation_id: payload.correlation_id || correlationId,
    snapshot,
    events: payload.result?.events || [],
    messages: payload.messages || [],
    files: payload.files,
    error: payload.error,
  };
}

export async function composeToBackend(
  label: string,
  atomIds: number[],
  lang: string,
  correlationId: string,
): Promise<BackendRunEnvelope> {
  return runModeToBackend('compose', label, correlationId, {
    label,
    atom_ids: atomIds,
    lang,
  });
}

/**
 * v6.0: Compose with composition pairs instead of atom IDs.
 * POST /compose accepts `compositions` (list of {label, sense_id} pairs) OR `atom_ids`.
 */
export async function composeWithCompositions(
  label: string,
  compositions: CompositionPair[],
  lang: string,
  correlationId: string,
): Promise<BackendRunEnvelope> {
  return runModeToBackend('compose', label, correlationId, {
    label,
    compositions,
    lang,
  });
}

/**
 * v6.0: Fetch structural similarity between two nodes.
 * GET /structural-similarity?a=raja&b=ratu
 */
export async function fetchStructuralSimilarity(
  labelA: string,
  labelB: string,
): Promise<StructuralSimilarityResult> {
  const url = proxyUrl(`structural-similarity?a=${encodeURIComponent(labelA)}&b=${encodeURIComponent(labelB)}`);
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Structural similarity failed (${res.status}): ${body}`);
  }
  return (await res.json()) as StructuralSimilarityResult;
}

/**
 * v6.0: Fetch substitution analysis between two nodes.
 * GET /substitution-analysis?a=raja&b=ratu
 */
export async function fetchSubstitutionAnalysis(
  labelA: string,
  labelB: string,
): Promise<SubstitutionAnalysisResult> {
  const url = proxyUrl(`substitution-analysis?a=${encodeURIComponent(labelA)}&b=${encodeURIComponent(labelB)}`);
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Substitution analysis failed (${res.status}): ${body}`);
  }
  return (await res.json()) as SubstitutionAnalysisResult;
}

/**
 * v6.1: Context-aware depth-controlled query.
 * POST /context-query with concept, context_atoms, and optional traversal params.
 *
 * Uses P(a|S,q) scoring, cycle detection, and adaptive halting
 * for recursive composition expansion.
 *
 * Depth presets:
 * - Shallow (max_depth=1): Fast appraise-style lookup
 * - Medium (max_depth=2): Relate-style one-hop expansion
 * - Deep (max_depth=5): Full grounding verification
 */
export interface ContextQueryOptions {
  concept: string;
  context_atoms: string[];
  max_depth?: number;
  gamma?: number;
  halt_confidence?: number;
  tau_relevance?: number;
}

export interface ContextQueryResult {
  ok: boolean;
  concept: string;
  result: {
    active_sense_idx: number | null;
    total_senses: number;
    scored_atoms: [string, number][];
    depth_reached: number;
    halt_reason: string;
    cycles_detected: number;
    layer: number;
    grounding_score: number;
    /** v8.2: Convergent nodes that contributed to this query result. */
    convergence_contributors?: [string, number][];
  } | null;
}

export async function contextQuery(
  options: ContextQueryOptions,
): Promise<ContextQueryResult> {
  const url = proxyUrl('context-query');
  const res = await fetch(url, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify(options),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Context query failed (${res.status}): ${body}`);
  }
  return (await res.json()) as ContextQueryResult;
}

/**
 * Fetch the latest snapshot from the backend.
 * Uses /snapshot endpoint instead of the removed /latest endpoint.
 */
export async function fetchLatestFromBackend(): Promise<BackendIngestResponse> {
  const url = proxyUrl('snapshot');
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Backend snapshot failed (${res.status}): ${body}`);
  }
  const snapshot = (await res.json()) as GraphSnapshot;
  return {
    ok: true,
    correlation_id: '',
    snapshot,
    events: [],
    messages: [],
  };
}
