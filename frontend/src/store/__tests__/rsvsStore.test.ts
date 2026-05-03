import { describe, it, expect, beforeEach } from 'vitest';
import {
  useGraphStore,
  useUIStore,
  useTimelineStore,
  useChatStore,
  useModeResultStore,
} from '../rsvsStore';
import type {
  RSVSNode,
  RSVSEdge,
  AppraiseResult,
  RelateResult,
} from '@/lib/types';

// ── Helper factories ──

function makeNode(overrides: Partial<RSVSNode> = {}): RSVSNode {
  return {
    id: 1,
    label: 'test-node',
    kind: 'atom',
    tier: 1,
    confidence: 0.8,
    status: 'stable',
    ...overrides,
  };
}

function makeEdge(overrides: Partial<RSVSEdge> = {}): RSVSEdge {
  return {
    id: '1->2',
    source: 1,
    target: 2,
    direction: 'undirected',
    weight: 0.5,
    source_type: 'learned',
    status: 'stable',
    ...overrides,
  };
}

// ── Graph Store ──

describe('useGraphStore', () => {
  beforeEach(() => {
    useGraphStore.setState({
      nodes: new Map(),
      edges: new Map(),
      events: [],
    });
  });

  it('initializes with empty maps', () => {
    const state = useGraphStore.getState();
    expect(state.nodes.size).toBe(0);
    expect(state.edges.size).toBe(0);
    expect(state.events).toHaveLength(0);
  });

  it('addNode adds a node to the store', () => {
    const node = makeNode();
    useGraphStore.getState().addNode(node);
    expect(useGraphStore.getState().nodes.get(1)).toEqual(node);
  });

  it('addEdge adds an edge to the store', () => {
    const edge = makeEdge();
    useGraphStore.getState().addEdge(edge);
    expect(useGraphStore.getState().edges.get('1->2')).toEqual(edge);
  });

  it('updateNode updates an existing node', () => {
    const node = makeNode();
    useGraphStore.getState().addNode(node);
    useGraphStore.getState().updateNode(1, { confidence: 0.95 });
    const updated = useGraphStore.getState().nodes.get(1);
    expect(updated?.confidence).toBe(0.95);
    expect(updated?.label).toBe('test-node'); // unchanged fields preserved
  });

  it('removeNode removes a node', () => {
    useGraphStore.getState().addNode(makeNode());
    useGraphStore.getState().removeNode(1);
    expect(useGraphStore.getState().nodes.has(1)).toBe(false);
  });

  it('removeEdge removes an edge', () => {
    useGraphStore.getState().addEdge(makeEdge());
    useGraphStore.getState().removeEdge('1->2');
    expect(useGraphStore.getState().edges.has('1->2')).toBe(false);
  });

  it('loadSnapshot replaces all nodes and edges', () => {
    const nodes = [makeNode({ id: 10 }), makeNode({ id: 20, label: 'other' })];
    const edges = [makeEdge({ id: '10->20', source: 10, target: 20 })];
    useGraphStore.getState().loadSnapshot(nodes, edges);
    const state = useGraphStore.getState();
    expect(state.nodes.size).toBe(2);
    expect(state.edges.size).toBe(1);
  });

  it('getNode returns the node by id', () => {
    const node = makeNode();
    useGraphStore.getState().addNode(node);
    expect(useGraphStore.getState().getNode(1)).toEqual(node);
    expect(useGraphStore.getState().getNode(999)).toBeUndefined();
  });

  it('getNodeNeighbors returns connected nodes and edges', () => {
    const n1 = makeNode({ id: 1 });
    const n2 = makeNode({ id: 2, label: 'neighbor' });
    const edge = makeEdge({ id: '1->2', source: 1, target: 2 });
    useGraphStore.getState().addNode(n1);
    useGraphStore.getState().addNode(n2);
    useGraphStore.getState().addEdge(edge);

    const neighbors = useGraphStore.getState().getNodeNeighbors(1);
    expect(neighbors.nodes).toHaveLength(1);
    expect(neighbors.edges).toHaveLength(1);
    expect(neighbors.nodes[0].id).toBe(2);
  });

  it('pushEvent appends an event', () => {
    const event = {
      event_id: 'evt_1',
      timestamp: new Date().toISOString(),
      correlation_id: 'corr_1',
      event_type: 'atom_created' as const,
      payload: {},
      animation_hint: { priority: 'normal' as const, focus_node_id: null, burst_group: null },
    };
    useGraphStore.getState().pushEvent(event);
    expect(useGraphStore.getState().events).toHaveLength(1);
  });
});

// ── UI Store ──

describe('useUIStore', () => {
  beforeEach(() => {
    useUIStore.setState({
      selectedNodeId: null,
      isDrawerOpen: false,
      viewMode: 'explore',
      isSearchOpen: false,
      searchQuery: '',
      isLeftRailCollapsed: false,
      focusedNodeId: null,
      pinnedNodeIds: new Set(),
    });
  });

  it('initializes with default values', () => {
    const state = useUIStore.getState();
    expect(state.selectedNodeId).toBeNull();
    expect(state.viewMode).toBe('explore');
    expect(state.isDrawerOpen).toBe(false);
  });

  it('selectNode sets selectedNodeId and opens drawer', () => {
    useUIStore.getState().selectNode(42);
    const state = useUIStore.getState();
    expect(state.selectedNodeId).toBe(42);
    expect(state.isDrawerOpen).toBe(true);
  });

  it('selectNode(null) closes drawer', () => {
    useUIStore.getState().selectNode(42);
    useUIStore.getState().selectNode(null);
    expect(useUIStore.getState().isDrawerOpen).toBe(false);
  });

  it('toggleDrawer flips drawer state', () => {
    expect(useUIStore.getState().isDrawerOpen).toBe(false);
    useUIStore.getState().toggleDrawer();
    expect(useUIStore.getState().isDrawerOpen).toBe(true);
  });

  it('setViewMode switches mode', () => {
    useUIStore.getState().setViewMode('analyze');
    expect(useUIStore.getState().viewMode).toBe('analyze');
  });

  it('closeDrawer resets selectedNodeId', () => {
    useUIStore.getState().selectNode(5);
    useUIStore.getState().closeDrawer();
    expect(useUIStore.getState().selectedNodeId).toBeNull();
  });
});

// ── Timeline Store ──

describe('useTimelineStore', () => {
  beforeEach(() => {
    useTimelineStore.setState({
      timelineState: 'live',
      playbackSpeed: 1,
      currentEventIndex: -1,
      timelineEvents: [],
    });
  });

  it('initializes with default timeline state', () => {
    const state = useTimelineStore.getState();
    expect(state.timelineState).toBe('live');
    expect(state.playbackSpeed).toBe(1);
    expect(state.currentEventIndex).toBe(-1);
    expect(state.timelineEvents).toHaveLength(0);
  });

  it('pause sets state to paused', () => {
    useTimelineStore.getState().pause();
    expect(useUIStore.getState().isDrawerOpen).toBe(false); // sanity
    expect(useTimelineStore.getState().timelineState).toBe('paused');
  });

  it('play sets state to live', () => {
    useTimelineStore.getState().pause();
    useTimelineStore.getState().play();
    expect(useTimelineStore.getState().timelineState).toBe('live');
  });

  it('togglePlayPause flips between live and paused', () => {
    expect(useTimelineStore.getState().timelineState).toBe('live');
    useTimelineStore.getState().togglePlayPause();
    expect(useTimelineStore.getState().timelineState).toBe('paused');
    useTimelineStore.getState().togglePlayPause();
    expect(useTimelineStore.getState().timelineState).toBe('live');
  });

  it('addTimelineEvent adds event and updates index', () => {
    const event = {
      event_id: 'tl_1',
      timestamp: new Date().toISOString(),
      event_type: 'atom_created' as const,
      label: 'Atom created',
    };
    useTimelineStore.getState().addTimelineEvent(event);
    const state = useTimelineStore.getState();
    expect(state.timelineEvents).toHaveLength(1);
    expect(state.currentEventIndex).toBe(0);
  });

  it('seekTo sets index and pauses', () => {
    const events = [
      { event_id: 'tl_1', timestamp: new Date().toISOString(), event_type: 'atom_created' as const, label: '1' },
      { event_id: 'tl_2', timestamp: new Date().toISOString(), event_type: 'edge_created' as const, label: '2' },
    ];
    events.forEach((e) => useTimelineStore.getState().addTimelineEvent(e));
    useTimelineStore.getState().seekTo(0);
    expect(useTimelineStore.getState().currentEventIndex).toBe(0);
    expect(useTimelineStore.getState().timelineState).toBe('paused');
  });

  it('resetTimeline clears events and index', () => {
    const event = {
      event_id: 'tl_1',
      timestamp: new Date().toISOString(),
      event_type: 'atom_created' as const,
      label: 'Atom created',
    };
    useTimelineStore.getState().addTimelineEvent(event);
    useTimelineStore.getState().resetTimeline();
    const state = useTimelineStore.getState();
    expect(state.timelineEvents).toHaveLength(0);
    expect(state.currentEventIndex).toBe(-1);
    expect(state.timelineState).toBe('live');
  });
});

// ── Mode Result Store ──

describe('useModeResultStore', () => {
  beforeEach(() => {
    useModeResultStore.setState({
      currentResult: null,
      isResultLoading: false,
      resultError: null,
    });
  });

  it('initializes with null result', () => {
    const state = useModeResultStore.getState();
    expect(state.currentResult).toBeNull();
    expect(state.isResultLoading).toBe(false);
    expect(state.resultError).toBeNull();
  });

  it('setAppraiseResult stores appraise data', () => {
    const appraiseResult: AppraiseResult = {
      verdict: 'agree',
      stance: { agree: 80, disagree: 10, neutral: 10 },
      confidence: 0.85,
      rationale: 'Test rationale',
      evidence_nodes: [],
      conflict_nodes: [],
      evidence_paths: [],
    };
    useModeResultStore.getState().setAppraiseResult(appraiseResult);
    const state = useModeResultStore.getState();
    expect(state.currentResult).not.toBeNull();
    expect(state.currentResult!.type).toBe('appraise');
    expect(state.isResultLoading).toBe(false);
    expect(state.resultError).toBeNull();
  });

  it('setRelateResult stores relate data', () => {
    const relateResult: RelateResult = {
      query_terms: ['test'],
      related_nodes: [],
      related_edges: [],
    };
    useModeResultStore.getState().setRelateResult(relateResult);
    const state = useModeResultStore.getState();
    expect(state.currentResult).not.toBeNull();
    expect(state.currentResult!.type).toBe('relate');
  });

  it('clearResult resets to null', () => {
    const relateResult: RelateResult = {
      query_terms: ['test'],
      related_nodes: [],
      related_edges: [],
    };
    useModeResultStore.getState().setRelateResult(relateResult);
    useModeResultStore.getState().clearResult();
    expect(useModeResultStore.getState().currentResult).toBeNull();
  });

  it('setResultLoading updates loading state', () => {
    useModeResultStore.getState().setResultLoading(true);
    expect(useModeResultStore.getState().isResultLoading).toBe(true);
  });

  it('setResultError stores error and clears loading', () => {
    useModeResultStore.getState().setResultLoading(true);
    useModeResultStore.getState().setResultError('Network error');
    const state = useModeResultStore.getState();
    expect(state.resultError).toBe('Network error');
    expect(state.isResultLoading).toBe(false);
  });
});
