import { NextRequest, NextResponse } from 'next/server';

/**
 * Next.js middleware: ensures every visitor has an HttpOnly session cookie.
 *
 * The proxy route (`/api/proxy/*`) checks for this cookie before injecting
 * the backend API key, preventing unauthenticated abuse of the proxy.
 *
 * Cookie-based sessions are sufficient for single-user / internal deployments.
 * For multi-tenant production, replace with a real auth provider (NextAuth, etc.).
 */

const SESSION_COOKIE = 'rsvs_session';
const SESSION_MAX_AGE = 60 * 60 * 24 * 7; // 7 days

function generateSessionToken(): string {
  // Cryptographic random hex string (32 bytes = 64 chars)
  const bytes = new Uint8Array(32);
  // Node.js crypto available in Edge Runtime
  const crypto = globalThis.crypto;
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

export function middleware(request: NextRequest) {
  const response = NextResponse.next();

  // Set session cookie if missing
  const existing = request.cookies.get(SESSION_COOKIE)?.value;
  if (!existing) {
    const token = generateSessionToken();
    response.cookies.set(SESSION_COOKIE, token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: SESSION_MAX_AGE,
    });
  }

  return response;
}

export const config = {
  // Run on all paths except static assets and Next.js internals
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
