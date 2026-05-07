/**
 * Server-side authentication utilities for the RSVS proxy.
 *
 * Provides HMAC-SHA256 signed session cookies and Origin/Referer validation
 * to prevent non-browser clients from abusing the backend API key injection.
 *
 * In demo/Vercel deployments without RSVS_SESSION_SECRET, the system
 * auto-generates an ephemeral secret with a warning. This means sessions
 * won't survive a server cold start, but the app will work for demo purposes.
 * For production with a real backend, always set RSVS_SESSION_SECRET.
 */

import { createHmac, timingSafeEqual } from 'crypto';

// --- Configuration ---

const SESSION_COOKIE = 'rsvs_session';
const SESSION_MAX_AGE = 60 * 60 * 24 * 7; // 7 days

/**
 * Lazy-initialized HMAC signing key.
 *
 * Priority:
 *   1. RSVS_SESSION_SECRET (preferred)
 *   2. RSVS_API_KEY (fallback)
 *   3. Auto-generated ephemeral key (demo/Vercel mode — logged as warning)
 *
 * The secret is resolved on first use, not at import time, so the app
 * doesn't crash if it's missing.
 */
let _resolvedSecret: string | null = null;

function getSessionSecret(): string {
  if (_resolvedSecret) return _resolvedSecret;

  const explicit = process.env.RSVS_SESSION_SECRET;
  const apiKey = process.env.RSVS_API_KEY;

  if (explicit) {
    _resolvedSecret = explicit;
    return _resolvedSecret;
  }

  if (apiKey) {
    console.warn(
      'RSVS_SESSION_SECRET not set — using RSVS_API_KEY as signing key. ' +
      'Set RSVS_SESSION_SECRET explicitly for proper key separation.'
    );
    _resolvedSecret = apiKey;
    return _resolvedSecret;
  }

  // No secret configured — auto-generate for demo mode
  console.warn(
    'RSVS_SESSION_SECRET not set — using auto-generated ephemeral key. ' +
    'Sessions will not survive server restarts. ' +
    'For production deployments with a backend, set RSVS_SESSION_SECRET.'
  );
  const bytes = Buffer.alloc(32);
  require('crypto').randomFillSync(bytes);
  _resolvedSecret = bytes.toString('hex');
  return _resolvedSecret;
}

// --- Cookie signing ---

/**
 * Compute HMAC-SHA256 of the session token using the server secret.
 * Returns the hex-encoded MAC.
 */
export function signToken(token: string): string {
  return createHmac('sha256', getSessionSecret()).update(token).digest('hex');
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
 *
 * In demo mode (no backend), origins are permissive since there's
 * no API key to protect.
 */
const IS_DEMO_MODE = !process.env.RSVS_BACKEND_URL;

const ALLOWED_ORIGINS = IS_DEMO_MODE
  ? [] // Empty = no origin validation in demo mode (no backend to protect)
  : (process.env.RSVS_ALLOWED_ORIGINS || 'http://localhost:3000')
      .split(',')
      .map((o: string) => o.trim().replace(/\/+$/, ''));

export { SESSION_COOKIE, SESSION_MAX_AGE, ALLOWED_ORIGINS, IS_DEMO_MODE };
