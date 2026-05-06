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
 * HMAC signing key. Security hierarchy:
 *
 *   1. RSVS_SESSION_SECRET (preferred — dedicated signing key, independent
 *      from API key for proper key separation)
 *   2. RSVS_API_KEY (fallback — works but not ideal key separation)
 *   3. Ephemeral random (DEV ONLY — breaks sessions on restart,
 *      incompatible with multi-instance)
 *
 * In production (NODE_ENV=production), we HARD-FAIL if neither
 * RSVS_SESSION_SECRET nor RSVS_API_KEY is set. An ephemeral secret
 * in production is unacceptable: it invalidates all sessions on every
 * restart and makes multi-instance deployments impossible.
 */
const SESSION_SECRET = (() => {
  const explicit = process.env.RSVS_SESSION_SECRET;
  const apiKey = process.env.RSVS_API_KEY;
  const isProd = process.env.NODE_ENV === 'production';

  if (explicit) return explicit;

  // Fallback to API key (acceptable but not ideal key separation)
  if (apiKey) {
    if (isProd) {
      console.warn(
        'RSVS_SESSION_SECRET not set — using RSVS_API_KEY as signing key. ' +
        'Set RSVS_SESSION_SECRET explicitly for proper key separation.'
      );
    }
    return apiKey;
  }

  // No secret at all — ephemeral random
  if (isProd) {
    // v8.3.1: HARD-FAIL in production instead of just warning.
    // Ephemeral secrets break sessions on restart and are incompatible
    // with multi-instance deployments. This is a security misconfiguration.
    throw new Error(
      'FATAL: RSVS_SESSION_SECRET (or RSVS_API_KEY) is required in production ' +
      'but neither is set. Set RSVS_SESSION_SECRET to a cryptographically ' +
      'random string (≥32 bytes). Generate with: ' +
      'python -c "import secrets; print(secrets.token_hex(32))"'
    );
  }

  // Development only — ephemeral is fine for local single-instance
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
