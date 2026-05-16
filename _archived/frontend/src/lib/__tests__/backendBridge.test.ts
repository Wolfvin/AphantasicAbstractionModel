import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  runModeToBackend,
  fetchLatestFromBackend,
} from '../backendBridge';

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
});

describe('runModeToBackend', () => {
  it('sends POST request to /api/proxy/run with correct payload', async () => {
    const mockResponse = {
      ok: true,
      mode: 'ingest',
      correlation_id: 'corr_1',
      timestamp: new Date().toISOString(),
      result: {},
      messages: [],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    const result = await runModeToBackend('ingest', 'hello world', 'corr_1');
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/proxy/run',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    expect(result.ok).toBe(true);
    expect(result.mode).toBe('ingest');
  });

  it('throws on non-ok response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => 'Internal Server Error',
    });

    await expect(
      runModeToBackend('ingest', 'test', 'corr_1'),
    ).rejects.toThrow('Backend run failed (500)');
  });

  it('passes options in the request body', async () => {
    const mockResponse = {
      ok: true,
      mode: 'appraise',
      correlation_id: 'corr_2',
      timestamp: new Date().toISOString(),
      result: {},
      messages: [],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    await runModeToBackend('appraise', 'test node', 'corr_2', { node_id: 42 });
    const callArgs = mockFetch.mock.calls[0][1];
    const body = JSON.parse(callArgs.body);
    expect(body.options).toEqual({ node_id: 42 });
  });

  it('does not include X-API-Key in headers (handled by proxy)', async () => {
    const mockResponse = {
      ok: true,
      mode: 'ingest',
      correlation_id: 'corr_3',
      timestamp: new Date().toISOString(),
      result: {},
      messages: [],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    await runModeToBackend('ingest', 'test', 'corr_3');
    const callArgs = mockFetch.mock.calls[0][1];
    expect(callArgs.headers).not.toHaveProperty('X-API-Key');
  });
});

describe('fetchLatestFromBackend', () => {
  it('sends GET request to /api/proxy/snapshot', async () => {
    const mockSnapshot = {
      snapshot_id: 'snap_1',
      generated_at: new Date().toISOString(),
      context: { domain: 'test', batch_id: 'b1', input_message_id: 'm1' },
      nodes: [],
      edges: [],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockSnapshot,
    });

    const result = await fetchLatestFromBackend();
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/proxy/snapshot',
      expect.objectContaining({ method: 'GET' }),
    );
    expect(result.ok).toBe(true);
  });

  it('throws on non-ok response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      text: async () => 'Not Found',
    });

    await expect(fetchLatestFromBackend()).rejects.toThrow(
      'Backend snapshot failed (404)',
    );
  });
});
