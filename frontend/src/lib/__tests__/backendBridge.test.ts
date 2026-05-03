import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  getBackendBaseUrl,
  runModeToBackend,
  fetchLatestFromBackend,
} from '../backendBridge';

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
});

describe('getBackendBaseUrl', () => {
  it('returns default URL when env var is not set', () => {
    const url = getBackendBaseUrl();
    expect(url).toBe('http://127.0.0.1:8000');
  });
});

describe('runModeToBackend', () => {
  it('sends POST request to /run with correct payload', async () => {
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
      expect.stringContaining('/run'),
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
});

describe('fetchLatestFromBackend', () => {
  it('sends GET request to /latest', async () => {
    const mockResponse = {
      ok: true,
      correlation_id: 'corr_latest',
      snapshot: {
        snapshot_id: 'snap_1',
        generated_at: new Date().toISOString(),
        context: { domain: 'test', batch_id: 'b1', input_message_id: 'm1' },
        nodes: [],
        edges: [],
      },
      events: [],
      messages: [],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    const result = await fetchLatestFromBackend();
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/latest'),
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
      'Backend latest failed (404)',
    );
  });
});
