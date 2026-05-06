import { NextRequest, NextResponse } from 'next/server';
import {
  SESSION_COOKIE,
  ALLOWED_ORIGINS,
  verifySignedCookie,
} from '@/lib/proxyAuth';

const BACKEND_URL = process.env.RSVS_BACKEND_URL || 'http://localhost:8000';
const API_KEY = process.env.RSVS_API_KEY || '';

/**
 * Allowlist of backend endpoints that the proxy will forward to.
 * Any path not in this set is rejected with 403.
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
 * Extract the origin from Origin or Referer header, safely.
 * Returns null if neither header is present.
 * Returns 'invalid' if Referer is malformed (cannot be parsed as URL).
 */
function extractOrigin(request: NextRequest): string | null | 'invalid' {
  const origin = request.headers.get('origin');
  if (origin) return origin;

  const referer = request.headers.get('referer');
  if (referer) {
    try {
      return new URL(referer).origin;
    } catch {
      return 'invalid';
    }
  }

  return null; // Neither header present
}

/**
 * Validate auth for state-changing (POST) requests:
 *
 *   1. Session cookie must exist AND have a valid HMAC signature.
 *      This prevents forged cookies — the HMAC secret is server-side only.
 *
 *   2. Origin header is REQUIRED on POST requests.
 *      Browsers always send Origin on fetch/POST. Direct HTTP clients
 *      that omit it are rejected — they must go through the frontend.
 *
 *   3. If Origin is present (or extracted from Referer), it must match
 *      the allowed origins list. Malformed Referer → 400.
 */
function validatePostAuth(request: NextRequest): NextResponse | null {
  // 1. Signed session cookie
  const cookieValue = request.cookies.get(SESSION_COOKIE)?.value;
  if (!cookieValue || !verifySignedCookie(cookieValue)) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  // 2. Origin is mandatory for POST (browsers always send it)
  const source = extractOrigin(request);

  if (source === 'invalid') {
    return NextResponse.json({ error: 'invalid_referer' }, { status: 400 });
  }

  if (source === null) {
    // No Origin and no Referer — reject POST.
    // Legitimate browsers always send Origin on POST/fetch.
    return NextResponse.json({ error: 'origin_required' }, { status: 403 });
  }

  // 3. Origin must match allowlist
  if (!ALLOWED_ORIGINS.includes(source)) {
    return NextResponse.json({ error: 'forbidden_origin' }, { status: 403 });
  }

  return null; // Auth OK
}

/**
 * Validate auth for read-only (GET) requests:
 *
 *   1. Signed session cookie required.
 *   2. If Origin/Referer is present, it must match allowlist.
 *      Absence is tolerated on GET (browsers don't always send Origin
 *      on navigation, and GET is non-mutating).
 */
function validateGetAuth(request: NextRequest): NextResponse | null {
  // 1. Signed session cookie
  const cookieValue = request.cookies.get(SESSION_COOKIE)?.value;
  if (!cookieValue || !verifySignedCookie(cookieValue)) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  // 2. If Origin/Referer present, validate it
  const source = extractOrigin(request);

  if (source === 'invalid') {
    return NextResponse.json({ error: 'invalid_referer' }, { status: 400 });
  }

  if (source !== null && !ALLOWED_ORIGINS.includes(source)) {
    return NextResponse.json({ error: 'forbidden_origin' }, { status: 403 });
  }

  return null; // Auth OK
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const authError = validatePostAuth(request);
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
  const authError = validateGetAuth(request);
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
