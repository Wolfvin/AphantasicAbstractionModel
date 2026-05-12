import type { ChatMessage, GraphSnapshot, RSVSEvent, ComposeResult, StructuralSimilarityResult, SubstitutionAnalysisResult, CompositionPair, RSVSNode } from '@/lib/types';
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
// ── F-04: Label Resolution Cache ──
// When the Rust core is active, relate() returns numeric node IDs (u32).
// We resolve these to labels using the /node-info API endpoint and cache them.
const labelCache = new Map<number, string>();
const labelCacheExpiry = new Map<number, number>(); // node_id → timestamp
const LABEL_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

/**
 * F-04: Resolve a numeric node ID to its label string.
 * Uses a local cache to avoid redundant API calls.
 * Falls back to "#<id>" if resolution fails.
 */
async function resolveNodeIdToLabel(nodeId: number): Promise<string> {
  // Check cache first
  const cached = labelCache.get(nodeId);
  const cachedExpiry = labelCacheExpiry.get(nodeId);
  if (cached !== undefined && cachedExpiry !== undefined && Date.now() < cachedExpiry) {
    return cached;
  }

  // Try to resolve via API
  try {
    const url = proxyUrl(`node-info?node_id=${nodeId}`);
    const res = await fetch(url, { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      const label = data?.label ?? data?.node?.label ?? null;
      if (label && typeof label === 'string') {
        labelCache.set(nodeId, label);
        labelCacheExpiry.set(nodeId, Date.now() + LABEL_CACHE_TTL_MS);
        return label;
      }
    }
  } catch {
    // API call failed — use fallback
  }

  // Fallback: check graph store for the node
  const fallback = `#${nodeId}`;
  labelCache.set(nodeId, fallback);
  labelCacheExpiry.set(nodeId, Date.now() + LABEL_CACHE_TTL_MS);
  return fallback;
}

/**
 * F-04: Resolve numeric IDs in relate results to labels.
 * Processes the related_nodes and structural_relations arrays.
 */
async function resolveRelateLabels(result: {
  related_nodes: Array<[string | number, number]>;
  structural_relations?: Array<[string | number, number]>;
  _pyo3_object?: boolean;
}): Promise<void> {
  // If not from PyO3, no resolution needed
  if (!result._pyo3_object) return;

  // Resolve related_nodes
  if (Array.isArray(result.related_nodes)) {
    const resolved = await Promise.all(
      result.related_nodes.map(async (item) => {
        if (Array.isArray(item) && item.length >= 2) {
          const id = item[0];
          const score = item[1];
          // If id is numeric, resolve to label
          if (typeof id === 'number') {
            const label = await resolveNodeIdToLabel(id);
            return [label, score] as [string, number];
          }
          return item as [string, number];
        }
        return item;
      })
    );
    result.related_nodes = resolved;
  }

  // Resolve structural_relations
  if (Array.isArray(result.structural_relations)) {
    const resolved = await Promise.all(
      result.structural_relations.map(async (item) => {
        if (Array.isArray(item) && item.length >= 2) {
          const id = item[0];
          const score = item[1];
          if (typeof id === 'number') {
            const label = await resolveNodeIdToLabel(id);
            return [label, score] as [string, number];
          }
          return item as [string, number];
        }
        return item;
      })
    );
    result.structural_relations = resolved;
  }

  // Also try to resolve labels from nodes already in the graph store
  // This is a sync fallback — if the node is already loaded locally,
  // we don't need to hit the API.
  try {
    const { useGraphStore } = await import('@/store/aamStore');
    const nodes = useGraphStore.getState().nodes;
    for (const item of result.related_nodes) {
      if (Array.isArray(item) && typeof item[0] === 'string' && item[0].startsWith('#')) {
        const numericId = parseInt(item[0].slice(1), 10);
        if (!isNaN(numericId)) {
          const graphNode = nodes.get(numericId);
          if (graphNode?.label) {
            item[0] = graphNode.label;
            labelCache.set(numericId, graphNode.label);
            labelCacheExpiry.set(numericId, Date.now() + LABEL_CACHE_TTL_MS);
          }
        }
      }
    }
  } catch {
    // Store not available — ignore
  }
}

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
