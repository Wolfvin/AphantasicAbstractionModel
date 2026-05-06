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
 * On first visit: generates a random token, signs it with HMAC-SHA256,
 * and stores `token.hmac` as the cookie value.
 *
 * On subsequent visits: validates the existing cookie's HMAC signature.
 * If the signature is invalid (tampered), regenerates the cookie.
 */

export function middleware(request: NextRequest) {
  const response = NextResponse.next();

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

  return response;
}

export const config = {
  // Run on all paths except static assets and Next.js internals
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
