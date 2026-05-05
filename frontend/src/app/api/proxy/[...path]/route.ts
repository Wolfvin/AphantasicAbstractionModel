import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.RSVS_BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.RSVS_API_KEY || '';
const SESSION_COOKIE = 'rsvs_session';

/**
 * Allowed frontend origins for CSRF protection.
 * In production, set RSVS_ALLOWED_ORIGINS to a comma-separated list.
 * Defaults to localhost:3000 for development.
 */
const ALLOWED_ORIGINS = (
  process.env.RSVS_ALLOWED_ORIGINS || 'http://localhost:3000'
).split(',').map((o: string) => o.trim().replace(/\/+$/, ''));

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

/**
 * Validate that the request has a session cookie and a valid Origin/Referer.
 * This prevents random HTTP clients from "borrowing" the server-side API key
 * through the proxy without having visited the frontend first.
 */
function validateAuth(request: NextRequest): NextResponse | null {
  // 1. Session cookie must exist (set by middleware on first page visit)
  const session = request.cookies.get(SESSION_COOKIE)?.value;
  if (!session) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  // 2. CSRF: Origin or Referer must match an allowed origin
  const origin = request.headers.get('origin');
  const referer = request.headers.get('referer');
  const source = origin || (referer ? new URL(referer).origin : '');

  // For same-origin requests (no Origin header in same-site navigation),
  // the cookie presence is sufficient. Origin is sent on POST/fetch calls.
  if (source && !ALLOWED_ORIGINS.includes(source)) {
    return NextResponse.json({ error: 'forbidden_origin' }, { status: 403 });
  }

  return null; // Auth OK
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const authError = validateAuth(request);
  if (authError) return authError;

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
  const authError = validateAuth(request);
  if (authError) return authError;

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
