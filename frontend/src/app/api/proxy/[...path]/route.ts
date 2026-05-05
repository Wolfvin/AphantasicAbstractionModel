import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.RSVS_BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.RSVS_API_KEY || '';

/**
 * Allowlist of backend endpoints that the proxy will forward to.
 * Any path not in this set is rejected with 403 to prevent
 * unauthenticated access to sensitive backend operations.
 */
const ALLOWED_ENDPOINTS = new Set([
  'run',
  'ingest',
  'query',
  'snapshot',
  'events',
  'structural-similarity',
  'substitution-analysis',
  'context-query',
  'context-similarity',
  'health',
]);

function validatePath(pathSegments: string[]): string | null {
  if (pathSegments.length === 0) return null;
  const endpoint = pathSegments[0];
  if (!ALLOWED_ENDPOINTS.has(endpoint)) return null;
  return pathSegments.join('/');
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: pathSegments } = await params;
  const backendPath = validatePath(pathSegments);
  if (!backendPath) {
    return NextResponse.json({ error: 'forbidden_path' }, { status: 403 });
  }

  const body = await request.json();

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }

  try {
    const res = await fetch(`${BACKEND_URL}/${backendPath}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    return NextResponse.json({ error: 'backend_unavailable' }, { status: 502 });
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: pathSegments } = await params;
  const backendPath = validatePath(pathSegments);
  if (!backendPath) {
    return NextResponse.json({ error: 'forbidden_path' }, { status: 403 });
  }

  const searchParams = request.nextUrl.search;

  const headers: Record<string, string> = {};
  if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }

  try {
    const res = await fetch(`${BACKEND_URL}/${backendPath}${searchParams}`, {
      method: 'GET',
      headers,
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error) {
    return NextResponse.json({ error: 'backend_unavailable' }, { status: 502 });
  }
}
