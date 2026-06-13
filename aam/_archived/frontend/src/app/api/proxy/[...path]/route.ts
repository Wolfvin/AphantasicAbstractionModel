import { NextRequest, NextResponse } from 'next/server';
import {
  SESSION_COOKIE,
  ALLOWED_ORIGINS,
  verifySignedCookie,
} from '@/lib/proxyAuth';

const IS_DEMO_MODE = process.env.RSVS_DEMO_MODE === '1' || !process.env.RSVS_BACKEND_URL;

if (!process.env.RSVS_BACKEND_URL && process.env.RSVS_DEMO_MODE !== '1') {
  console.warn(
    '⚠️ RSVS_BACKEND_URL is not set. Running in demo mode. ' +
    'Set RSVS_BACKEND_URL or explicitly set RSVS_DEMO_MODE=1 to suppress this warning.'
  );
}

const BACKEND_URL = process.env.RSVS_BACKEND_URL || '';
const API_KEY = process.env.RSVS_API_KEY || '';

/**
 * Allowlist of backend endpoints that the proxy will forward to.
 * Any path not in this set is rejected with 403.
 */
const ALLOWED_ENDPOINTS = new Set([
  'run',
  'ingest',
  'appraise',
  'relate',
  'compose',
  'query',
  'snapshot',
  'events',
  'structural-similarity',
  'substitution-analysis',
  'context-query',
  'context-similarity',
  'health',
  'node-info',
  'senses',
  'similarity',
]);

function validatePath(pathSegments: string[]): string | null {
  if (pathSegments.length === 0) return null;
  const endpoint = pathSegments[0];
  if (!ALLOWED_ENDPOINTS.has(endpoint)) return null;
  return pathSegments.join('/');
}

/**
 * Extract the origin from Origin or Referer header, safely.
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

  return null;
}

/**
 * In demo mode (no backend), skip auth entirely.
 * There's no API key to protect, so auth is unnecessary.
 */
function isDemoRequest(): boolean {
  return IS_DEMO_MODE || !BACKEND_URL;
}

function validatePostAuth(request: NextRequest): NextResponse | null {
  if (isDemoRequest()) return null; // No backend = no auth needed

  // 1. Signed session cookie
  const cookieValue = request.cookies.get(SESSION_COOKIE)?.value;
  if (!cookieValue || !verifySignedCookie(cookieValue)) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  // 2. Origin is mandatory for POST
  const source = extractOrigin(request);
  if (source === 'invalid') {
    return NextResponse.json({ error: 'invalid_referer' }, { status: 400 });
  }
  if (source === null) {
    return NextResponse.json({ error: 'origin_required' }, { status: 403 });
  }

  // 3. Origin must match allowlist
  if (ALLOWED_ORIGINS.length > 0 && !ALLOWED_ORIGINS.includes(source)) {
    return NextResponse.json({ error: 'forbidden_origin' }, { status: 403 });
  }

  return null;
}

function validateGetAuth(request: NextRequest): NextResponse | null {
  if (isDemoRequest()) return null;

  const cookieValue = request.cookies.get(SESSION_COOKIE)?.value;
  if (!cookieValue || !verifySignedCookie(cookieValue)) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  const source = extractOrigin(request);
  if (source === 'invalid') {
    return NextResponse.json({ error: 'invalid_referer' }, { status: 400 });
  }
  if (source !== null && ALLOWED_ORIGINS.length > 0 && !ALLOWED_ORIGINS.includes(source)) {
    return NextResponse.json({ error: 'forbidden_origin' }, { status: 403 });
  }

  return null;
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

  // Demo mode: no backend available
  if (isDemoRequest()) {
    return NextResponse.json(
      { error: 'demo_mode', message: 'No backend configured. The app is running in demo mode with simulated data.' },
      { status: 503 }
    );
  }

  const body = await request.json();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (API_KEY) headers['X-API-Key'] = API_KEY;

  try {
    const res = await fetch(`${BACKEND_URL}/${backendPath}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
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

  // Demo mode: no backend available
  if (isDemoRequest()) {
    return NextResponse.json(
      { error: 'demo_mode', message: 'No backend configured. Running in demo mode.' },
      { status: 503 }
    );
  }

  const searchParams = request.nextUrl.search;
  const headers: Record<string, string> = {};
  if (API_KEY) headers['X-API-Key'] = API_KEY;

  try {
    const res = await fetch(`${BACKEND_URL}/${backendPath}${searchParams}`, {
      method: 'GET',
      headers,
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ error: 'backend_unavailable' }, { status: 502 });
  }
}
