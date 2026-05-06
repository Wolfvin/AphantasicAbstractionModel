import { NextRequest, NextResponse } from 'next/server';
import {
  SESSION_COOKIE,
  SESSION_MAX_AGE,
  generateSessionToken,
  createSignedCookieValue,
  verifySignedCookie,
} from '@/lib/proxyAuth';

/**
 * Next.js middleware: ensures every visitor has a valid, HMAC-signed
 * HttpOnly session cookie.
 *
 * Runs in Node.js runtime (not Edge) because it uses Node's crypto
 * module for HMAC-SHA256 signing.
 */

export function middleware(request: NextRequest) {
  const response = NextResponse.next();

  try {
    const existing = request.cookies.get(SESSION_COOKIE)?.value;

    if (!existing || !verifySignedCookie(existing)) {
      // No cookie or tampered cookie — issue a fresh signed one
      const token = generateSessionToken();
      const signedValue = createSignedCookieValue(token);
      response.cookies.set(SESSION_COOKIE, signedValue, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        path: '/',
        maxAge: SESSION_MAX_AGE,
      });
    }
  } catch {
    // If crypto fails for any reason, just continue without setting cookie.
    // The proxy route will handle auth gracefully.
  }

  return response;
}

export const config = {
  // Run on all paths except static assets and Next.js internals
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
  // Use Node.js runtime to access crypto module
  runtime: 'nodejs',
};
