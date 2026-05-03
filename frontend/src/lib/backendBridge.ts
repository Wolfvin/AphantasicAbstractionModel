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
 * Get the backend base URL.
 *
 * The URL MUST be provided via NEXT_PUBLIC_RSVS_BACKEND_URL environment variable.
 * In development, defaults to http://localhost:8000.
 * In production (docker-compose), this is set to http://backend:8000.
 *
 * IMPORTANT: Never hardcode a production URL here. Always use environment variables.
 */
export function getBackendBaseUrl(): string {
  const url = process.env.NEXT_PUBLIC_RSVS_BACKEND_URL;
  if (!url) {
    // Development fallback only — warns in production builds
    if (process.env.NODE_ENV === 'production') {
      console.error(
        'NEXT_PUBLIC_RSVS_BACKEND_URL is not set! ' +
        'Set this environment variable before deploying. ' +
        'Falling back to localhost (will not work in production).'
      );
    }
    return 'http://localhost:8000';
  }
  return url;
}

/**
 * Get the API key header if configured.
 * Reads from NEXT_PUBLIC_RSVS_API_KEY (client-side) or injects via server proxy.
 */
function getAuthHeaders(): Record<string, string> {
  const apiKey = process.env.NEXT_PUBLIC_RSVS_API_KEY;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  return headers;
}

export async function runModeToBackend(
  mode: RSVSMode,
  text: string,
  correlationId: string,
  options: Record<string, unknown> = {},
): Promise<BackendRunEnvelope> {
  const url = `${getBackendBaseUrl()}/run`;
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
  return runModeToBackend('compose', `${label} = ${atomIds.map(() => '+').join(' ')}`, correlationId, {
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
  return runModeToBackend('compose', `${label} = ${compositions.map(c => c.label).join(' + ')}`, correlationId, {
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
  const url = `${getBackendBaseUrl()}/structural-similarity?a=${encodeURIComponent(labelA)}&b=${encodeURIComponent(labelB)}`;
  const headers: Record<string, string> = {};
  const apiKey = process.env.NEXT_PUBLIC_RSVS_API_KEY;
  if (apiKey) headers['X-API-Key'] = apiKey;
  const res = await fetch(url, { method: 'GET', headers });
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
  const url = `${getBackendBaseUrl()}/substitution-analysis?a=${encodeURIComponent(labelA)}&b=${encodeURIComponent(labelB)}`;
  const headers: Record<string, string> = {};
  const apiKey = process.env.NEXT_PUBLIC_RSVS_API_KEY;
  if (apiKey) headers['X-API-Key'] = apiKey;
  const res = await fetch(url, { method: 'GET', headers });
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
  } | null;
}

export async function contextQuery(
  options: ContextQueryOptions,
): Promise<ContextQueryResult> {
  const url = `${getBackendBaseUrl()}/context-query`;
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
  const url = `${getBackendBaseUrl()}/snapshot`;
  const headers: Record<string, string> = {};
  const apiKey = process.env.NEXT_PUBLIC_RSVS_API_KEY;
  if (apiKey) headers['X-API-Key'] = apiKey;
  const res = await fetch(url, { method: 'GET', headers });
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
