import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import RelatePanel from '../RelatePanel';
import type { RelateResult } from '@/lib/types';

function makeRelateResult(overrides: Partial<RelateResult> = {}): RelateResult {
  return {
    query_terms: ['rock', 'erosion'],
    related_nodes: [
      { node_id: 1, label: 'mineral', score: 0.9, tier: 1, kind: 'atom' },
      { node_id: 2, label: 'sediment', score: 0.6, tier: 2, kind: 'atom' },
    ],
    related_edges: [
      { edge_id: 'e1', source: 1, target: 2, weight: 0.75, label: 'causal' },
    ],
    ...overrides,
  };
}

describe('RelatePanel', () => {
  it('renders query terms as badges', () => {
    const result = makeRelateResult();
    render(<RelatePanel result={result} />);
    expect(screen.getByText('rock')).toBeInTheDocument();
    expect(screen.getByText('erosion')).toBeInTheDocument();
  });

  it('renders related nodes with labels', () => {
    const result = makeRelateResult();
    render(<RelatePanel result={result} />);
    expect(screen.getByText('mineral')).toBeInTheDocument();
    expect(screen.getByText('sediment')).toBeInTheDocument();
  });

  it('renders related edges', () => {
    const result = makeRelateResult();
    render(<RelatePanel result={result} />);
    expect(screen.getByText('1')).toBeInTheDocument(); // source
    expect(screen.getByText('2')).toBeInTheDocument(); // target
  });

  it('shows node tier badges', () => {
    const result = makeRelateResult();
    render(<RelatePanel result={result} />);
    expect(screen.getByText('T1')).toBeInTheDocument();
    expect(screen.getByText('T2')).toBeInTheDocument();
  });

  it('shows node kind badges', () => {
    const result = makeRelateResult();
    render(<RelatePanel result={result} />);
    // Both nodes are atoms
    const atomBadges = screen.getAllByText('Atom');
    expect(atomBadges.length).toBeGreaterThanOrEqual(1);
  });

  it('renders edge labels when present', () => {
    const result = makeRelateResult();
    render(<RelatePanel result={result} />);
    expect(screen.getByText('causal')).toBeInTheDocument();
  });

  it('shows empty state when no related nodes', () => {
    const result = makeRelateResult({
      related_nodes: [],
      related_edges: [],
    });
    render(<RelatePanel result={result} />);
    expect(screen.getByText('No related nodes found')).toBeInTheDocument();
  });

  it('does not show related edges section when empty', () => {
    const result = makeRelateResult({
      related_nodes: [{ node_id: 1, label: 'test', score: 0.5, tier: 1, kind: 'atom' }],
      related_edges: [],
    });
    render(<RelatePanel result={result} />);
    expect(screen.queryByText(/Related Edges/)).not.toBeInTheDocument();
  });
});
