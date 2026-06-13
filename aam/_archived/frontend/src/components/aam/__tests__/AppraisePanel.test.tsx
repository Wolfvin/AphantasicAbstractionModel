import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import AppraisePanel from '../AppraisePanel';
import type { AppraiseResult } from '@/lib/types';

function makeAppraiseResult(overrides: Partial<AppraiseResult> = {}): AppraiseResult {
  return {
    verdict: 'agree',
    stance: { agree: 70, disagree: 20, neutral: 10 },
    confidence: 0.85,
    rationale: 'The evidence supports the claim.',
    evidence_nodes: [
      { node_id: 1, label: 'mineral', confidence: 0.9, role: 'support' },
      { node_id: 2, label: 'sediment', confidence: 0.6, role: 'neutral' },
    ],
    conflict_nodes: [
      { node_id: 3, label: 'erosion', confidence: 0.4, role: 'conflict' },
    ],
    evidence_paths: [
      { path: [1, 2, 3], weight: 0.75, label: 'causal' },
    ],
    target_node_id: 1,
    ...overrides,
  };
}

describe('AppraisePanel', () => {
  it('renders verdict correctly for agree', () => {
    const result = makeAppraiseResult({ verdict: 'agree' });
    render(<AppraisePanel result={result} />);
    expect(screen.getByText('Agree')).toBeInTheDocument();
  });

  it('renders verdict correctly for mixed', () => {
    const result = makeAppraiseResult({ verdict: 'mixed' });
    render(<AppraisePanel result={result} />);
    expect(screen.getByText('Mixed')).toBeInTheDocument();
  });

  it('renders verdict correctly for disagree', () => {
    const result = makeAppraiseResult({ verdict: 'disagree' });
    render(<AppraisePanel result={result} />);
    expect(screen.getByText('Disagree')).toBeInTheDocument();
  });

  it('displays evidence nodes (support)', () => {
    const result = makeAppraiseResult();
    render(<AppraisePanel result={result} />);
    expect(screen.getByText('mineral')).toBeInTheDocument();
  });

  it('displays conflict nodes', () => {
    const result = makeAppraiseResult();
    render(<AppraisePanel result={result} />);
    expect(screen.getByText('erosion')).toBeInTheDocument();
  });

  it('displays rationale text', () => {
    const result = makeAppraiseResult();
    render(<AppraisePanel result={result} />);
    expect(screen.getByText('The evidence supports the claim.')).toBeInTheDocument();
  });

  it('displays confidence percentage', () => {
    const result = makeAppraiseResult({ confidence: 0.85 });
    render(<AppraisePanel result={result} />);
    expect(screen.getByText('85.0%')).toBeInTheDocument();
  });

  it('shows evidence paths', () => {
    const result = makeAppraiseResult();
    render(<AppraisePanel result={result} />);
    expect(screen.getByText('1 → 2 → 3')).toBeInTheDocument();
  });

  it('shows target node link when target_node_id is set', () => {
    const result = makeAppraiseResult({ target_node_id: 1 });
    render(<AppraisePanel result={result} />);
    expect(screen.getByText('View target node in graph')).toBeInTheDocument();
  });

  it('renders empty state — no conflict nodes', () => {
    const result = makeAppraiseResult({ conflict_nodes: [], evidence_nodes: [] });
    render(<AppraisePanel result={result} />);
    // No "Conflict Nodes" heading should appear
    expect(screen.queryByText(/Conflict Nodes/)).not.toBeInTheDocument();
    // No "Support Nodes" heading should appear
    expect(screen.queryByText(/Support Nodes/)).not.toBeInTheDocument();
  });
});
