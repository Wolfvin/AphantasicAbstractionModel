/**
 * Vitest global setup file.
 *
 * This file runs before each test suite. It configures:
 *   - @testing-library/jest-dom matchers (toBeInTheDocument, etc.)
 *   - Global mocks for browser APIs not available in jsdom
 *
 * v8.3.1: Created this missing file — Vitest was configured with
 * setupFiles: ['./src/test/setup.ts'] but the file didn't exist,
 * causing ALL test suites to fail before execution.
 */

import '@testing-library/jest-dom/vitest';

const noop = (): void => { /* intentional no-op */ };

// Mock Web Crypto API for tests (jsdom doesn't provide it)
if (typeof globalThis.crypto === 'undefined' || !(globalThis.crypto as Record<string, unknown>).randomUUID) {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { webcrypto } = require('crypto');
  Object.defineProperty(globalThis, 'crypto', {
    value: webcrypto.crypto,
    writable: true,
  });
}

// Mock ResizeObserver (not available in jsdom)
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class ResizeObserver {
    observe = noop;
    unobserve = noop;
    disconnect = noop;
  } as unknown as typeof ResizeObserver;
}

// Mock IntersectionObserver (not available in jsdom)
if (typeof globalThis.IntersectionObserver === 'undefined') {
  globalThis.IntersectionObserver = class IntersectionObserver {
    root = null;
    rootMargin = '';
    thresholds: number[] = [];
    observe = noop;
    unobserve = noop;
    disconnect = noop;
    takeRecords = (): IntersectionObserverEntry[] => [];
  } as unknown as typeof IntersectionObserver;
}

// Mock matchMedia (not available in jsdom)
if (typeof window !== 'undefined' && !window.matchMedia) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: noop,
      removeListener: noop,
      addEventListener: noop,
      removeEventListener: noop,
      dispatchEvent: () => false,
    }),
  });
}
