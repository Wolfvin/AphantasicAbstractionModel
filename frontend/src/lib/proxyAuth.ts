/**
 * Server-side authentication utilities for the RSVS proxy.
 *
 * Provides HMAC-SHA256 signed session cookies and Origin/Referer validation
 * to prevent non-browser clients from abusing the backend API key injection.
 *
 * Used by:
 *   - src/middleware.ts  (sets signed session cookie)
 *   - src/app/api/proxy/[...path]/route.ts  (validates cookie + origin)
 *
 * Security model:
 *   1. A cryptographic secret (RSVS_SESSION_SECRET) signs every session cookie.
 *      Forgers cannot produce a valid HMAC without the secret.
 *   2. POST requests MUST include a valid Origin header (browsers always do).
 *      Direct HTTP clients that omit Origin are rejected.
 *   3. GET requests tolerate absent Origin (browsers don't always send it),
 *      but if present it must match the allowlist.
 */

import { createHmac, timingSafeEqual } from 'crypto';

// --- Configuration ---

const SESSION_COOKIE = 'rsvs_session';
const SESSION_MAX_AGE = 60 * 60 * 24 * 7; // 7 days

/**
 * HMAC signing key. In production, set RSVS_SESSION_SECRET to a
 * cryptographically random string (≥32 bytes). Falls back to a
 * derived key from RSVS_API_KEY; if neither is set, generates a
 * random one at startup (valid for single-instance deployments only).
 */
const SESSION_SECRET = process.env.RSVS_SESSION_SECRET
  || process.env.RSVS_API_KEY
  || (() => {
      // Ephemeral random secret — works for single-instance dev only
      const bytes = Buffer.alloc(32);
      require('crypto').randomFillSync(bytes);
      return bytes.toString('hex');
    })();

// --- Cookie signing ---

/**
 * Compute HMAC-SHA256 of the session token using the server secret.
 * Returns the hex-encoded MAC.
 */
export function signToken(token: string): string {
  return createHmac('sha256', SESSION_SECRET).update(token).digest('hex');
}

/**
 * Create a signed cookie value: `token.hmac`.
 */
export function createSignedCookieValue(token: string): string {
  return `${token}.${signToken(token)}`;
}

/**
 * Validate a signed cookie value. Returns true only if:
 *   - The value has the format `token.hmac`
 *   - The HMAC matches a fresh computation over the token
 * Uses timing-safe comparison to prevent timing attacks.
 */
export function verifySignedCookie(cookieValue: string | undefined): boolean {
  if (!cookieValue) return false;

  const dotIndex = cookieValue.lastIndexOf('.');
  if (dotIndex <= 0 || dotIndex >= cookieValue.length - 1) return false;

  const token = cookieValue.slice(0, dotIndex);
  const providedHmac = cookieValue.slice(dotIndex + 1);
  const expectedHmac = signToken(token);

  // timing-safe comparison
  if (providedHmac.length !== expectedHmac.length) return false;
  return timingSafeEqual(Buffer.from(providedHmac), Buffer.from(expectedHmac));
}

// --- Session token generation ---

/**
 * Generate a cryptographically random session token (32 bytes = 64 hex chars).
 */
export function generateSessionToken(): string {
  const bytes = new Uint8Array(32);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

// --- Origin / Referer validation ---

/**
 * Allowed frontend origins for CSRF protection.
 * Set RSVS_ALLOWED_ORIGINS to a comma-separated list in production.
 * Defaults to localhost:3000 for development.
 */
const ALLOWED_ORIGINS = (
  process.env.RSVS_ALLOWED_ORIGINS || 'http://localhost:3000'
).split(',').map((o: string) => o.trim().replace(/\/+$/, ''));

export { SESSION_COOKIE, SESSION_MAX_AGE, ALLOWED_ORIGINS };
