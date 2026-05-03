import type { ChatMessage, GraphSnapshot, RSVSEvent, ComposeResult } from '@/lib/types';
export type RSVSMode = 'ingest' | 'appraise' | 'relate' | 'compose';

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

export function getBackendBaseUrl(): string {
  return process.env.NEXT_PUBLIC_RSVS_BACKEND_URL || 'http://127.0.0.1:8000';
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
    headers: { 'Content-Type': 'application/json' },
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

export async function fetchLatestFromBackend(): Promise<BackendIngestResponse> {
  const url = `${getBackendBaseUrl()}/latest`;
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Backend latest failed (${res.status}): ${body}`);
  }
  return (await res.json()) as BackendIngestResponse;
}
