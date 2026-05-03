// RSVS Zustand Stores

import { create } from 'zustand';
import type {
  RSVSNode,
  RSVSEdge,
  RSVSEvent,
  ChatMessage,
  ViewMode,
  TimelineState,
  TimelineEvent,
  FilterState,
  AnimationQueueItem,
  NodeStatus,
  Tier,
} from '@/lib/types';

// ── Graph Domain State ──

interface GraphState {
  nodes: Map<number, RSVSNode>;
  edges: Map<string, RSVSEdge>;
  events: RSVSEvent[];

  addNode: (node: RSVSNode) => void;
  updateNode: (id: number, updates: Partial<RSVSNode>) => void;
  removeNode: (id: number) => void;
  getNode: (id: number) => RSVSNode | undefined;
  addEdge: (edge: RSVSEdge) => void;
  updateEdge: (id: string, updates: Partial<RSVSEdge>) => void;
  removeEdge: (id: string) => void;
  getEdge: (id: string) => RSVSEdge | undefined;
  pushEvent: (event: RSVSEvent) => void;
  loadSnapshot: (nodes: RSVSNode[], edges: RSVSEdge[]) => void;
  getNodeNeighbors: (id: number) => { nodes: RSVSNode[]; edges: RSVSEdge[] };
}

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: new Map(),
  edges: new Map(),
  events: [],

  addNode: (node) =>
    set((state) => {
      const nodes = new Map(state.nodes);
      nodes.set(node.id, node);
      return { nodes };
    }),

  updateNode: (id, updates) =>
    set((state) => {
      const nodes = new Map(state.nodes);
      const existing = nodes.get(id);
      if (existing) {
        nodes.set(id, { ...existing, ...updates });
      }
      return { nodes };
    }),

  removeNode: (id) =>
    set((state) => {
      const nodes = new Map(state.nodes);
      nodes.delete(id);
      return { nodes };
    }),

  getNode: (id) => get().nodes.get(id),

  addEdge: (edge) =>
    set((state) => {
      const edges = new Map(state.edges);
      edges.set(edge.id, edge);
      return { edges };
    }),

  updateEdge: (id, updates) =>
    set((state) => {
      const edges = new Map(state.edges);
      const existing = edges.get(id);
      if (existing) {
        edges.set(id, { ...existing, ...updates });
      }
      return { edges };
    }),

  removeEdge: (id) =>
    set((state) => {
      const edges = new Map(state.edges);
      edges.delete(id);
      return { edges };
    }),

  getEdge: (id) => get().edges.get(id),

  pushEvent: (event) =>
    set((state) => ({ events: [...state.events, event] })),

  loadSnapshot: (nodes, edges) =>
    set(() => {
      const nodeMap = new Map<number, RSVSNode>();
      nodes.forEach((n) => nodeMap.set(n.id, n));
      const edgeMap = new Map<string, RSVSEdge>();
      edges.forEach((e) => edgeMap.set(e.id, e));
      return { nodes: nodeMap, edges: edgeMap };
    }),

  getNodeNeighbors: (id) => {
    const { nodes, edges } = get();
    const neighborNodes: RSVSNode[] = [];
    const neighborEdges: RSVSEdge[] = [];
    edges.forEach((edge) => {
      if (edge.source === id) {
        const target = nodes.get(edge.target);
        if (target) neighborNodes.push(target);
        neighborEdges.push(edge);
      } else if (edge.target === id) {
        const source = nodes.get(edge.source);
        if (source) neighborNodes.push(source);
        neighborEdges.push(edge);
      }
    });
    return { nodes: neighborNodes, edges: neighborEdges };
  },
}));

// ── UI State ──

interface UIState {
  selectedNodeId: number | null;
  isDrawerOpen: boolean;
  viewMode: ViewMode;
  isSearchOpen: boolean;
  searchQuery: string;
  isLeftRailCollapsed: boolean;
  focusedNodeId: number | null;
  pinnedNodeIds: Set<number>;

  selectNode: (id: number | null) => void;
  toggleDrawer: () => void;
  openDrawer: () => void;
  closeDrawer: () => void;
  setViewMode: (mode: ViewMode) => void;
  toggleSearch: () => void;
  setSearchQuery: (q: string) => void;
  toggleLeftRail: () => void;
  focusNode: (id: number | null) => void;
  togglePinNode: (id: number) => void;
}

export const useUIStore = create<UIState>((set, get) => ({
  selectedNodeId: null,
  isDrawerOpen: false,
  viewMode: 'explore',
  isSearchOpen: false,
  searchQuery: '',
  isLeftRailCollapsed: false,
  focusedNodeId: null,
  pinnedNodeIds: new Set(),

  selectNode: (id) =>
    set({ selectedNodeId: id, isDrawerOpen: id !== null }),

  toggleDrawer: () =>
    set((s) => ({ isDrawerOpen: !s.isDrawerOpen })),

  openDrawer: () => set({ isDrawerOpen: true }),
  closeDrawer: () => set({ isDrawerOpen: false, selectedNodeId: null }),

  setViewMode: (mode) => set({ viewMode: mode }),
  toggleSearch: () => set((s) => ({ isSearchOpen: !s.isSearchOpen })),
  setSearchQuery: (q) => set({ searchQuery: q }),
  toggleLeftRail: () => set((s) => ({ isLeftRailCollapsed: !s.isLeftRailCollapsed })),
  focusNode: (id) => set({ focusedNodeId: id }),
  togglePinNode: (id) =>
    set((s) => {
      const pins = new Set(s.pinnedNodeIds);
      if (pins.has(id)) pins.delete(id);
      else pins.add(id);
      return { pinnedNodeIds: pins };
    }),
}));

// ── Timeline State ──

interface TimelineStore {
  timelineState: TimelineState;
  playbackSpeed: number;
  currentEventIndex: number;
  timelineEvents: TimelineEvent[];

  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  setSpeed: (speed: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  seekTo: (index: number) => void;
  addTimelineEvent: (event: TimelineEvent) => void;
  resetTimeline: () => void;
}

export const useTimelineStore = create<TimelineStore>((set, get) => ({
  timelineState: 'live',
  playbackSpeed: 1,
  currentEventIndex: -1,
  timelineEvents: [],

  play: () => set({ timelineState: 'live' }),
  pause: () => set({ timelineState: 'paused' }),
  togglePlayPause: () =>
    set((s) => ({ timelineState: s.timelineState === 'live' ? 'paused' : 'live' })),
  setSpeed: (speed) => set({ playbackSpeed: speed }),
  stepForward: () =>
    set((s) => ({
      currentEventIndex: Math.min(s.currentEventIndex + 1, s.timelineEvents.length - 1),
      timelineState: 'paused',
    })),
  stepBackward: () =>
    set((s) => ({
      currentEventIndex: Math.max(s.currentEventIndex - 1, 0),
      timelineState: 'paused',
    })),
  seekTo: (index) =>
    set({ currentEventIndex: index, timelineState: 'paused' }),
  addTimelineEvent: (event) =>
    set((s) => ({
      timelineEvents: [...s.timelineEvents, event],
      currentEventIndex: s.timelineEvents.length,
    })),
  resetTimeline: () =>
    set({ timelineEvents: [], currentEventIndex: -1, timelineState: 'live' }),
}));

// ── Chat State ──

interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  addMessage: (msg: ChatMessage) => void;
  setLoading: (loading: boolean) => void;
  clearMessages: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isLoading: false,

  addMessage: (msg) =>
    set((s) => ({ messages: [...s.messages, msg] })),
  setLoading: (loading) => set({ isLoading: loading }),
  clearMessages: () => set({ messages: [] }),
}));

// ── Filter State ──

interface FilterStoreState {
  filters: FilterState;
  setTiers: (tiers: Tier[]) => void;
  setConfidenceRange: (range: [number, number]) => void;
  setNodeKinds: (kinds: RSVSNode['kind'][]) => void;
  setSourceBatch: (batch: string | null) => void;
  setRecentActivity: (seconds: number | null) => void;
  resetFilters: () => void;
}

const defaultFilters: FilterState = {
  tiers: [1, 2, 3],
  confidenceRange: [0, 1],
  nodeKinds: ['atom', 'composite'],
  sourceBatch: null,
  recentActivity: null,
};

export const useFilterStore = create<FilterStoreState>((set) => ({
  filters: { ...defaultFilters },

  setTiers: (tiers) =>
    set((s) => ({ filters: { ...s.filters, tiers } })),
  setConfidenceRange: (range) =>
    set((s) => ({ filters: { ...s.filters, confidenceRange: range } })),
  setNodeKinds: (kinds) =>
    set((s) => ({ filters: { ...s.filters, nodeKinds: kinds } })),
  setSourceBatch: (batch) =>
    set((s) => ({ filters: { ...s.filters, sourceBatch: batch } })),
  setRecentActivity: (seconds) =>
    set((s) => ({ filters: { ...s.filters, recentActivity: seconds } })),
  resetFilters: () => set({ filters: { ...defaultFilters } }),
}));

// ── Animation Queue State ──

interface AnimationQueueStore {
  queue: AnimationQueueItem[];
  activeAnimations: Map<string, AnimationQueueItem>;
  reducedMotion: boolean;
  addToQueue: (item: AnimationQueueItem) => void;
  removeFromQueue: (id: string) => void;
  setActiveAnimation: (id: string, item: AnimationQueueItem) => void;
  removeActiveAnimation: (id: string) => void;
  setReducedMotion: (reduced: boolean) => void;
}

export const useAnimationStore = create<AnimationQueueStore>((set) => ({
  queue: [],
  activeAnimations: new Map(),
  reducedMotion: false,

  addToQueue: (item) =>
    set((s) => ({ queue: [...s.queue, item] })),
  removeFromQueue: (id) =>
    set((s) => ({ queue: s.queue.filter((i) => i.id !== id) })),
  setActiveAnimation: (id, item) =>
    set((s) => {
      const active = new Map(s.activeAnimations);
      active.set(id, item);
      return { activeAnimations: active };
    }),
  removeActiveAnimation: (id) =>
    set((s) => {
      const active = new Map(s.activeAnimations);
      active.delete(id);
      return { activeAnimations: active };
    }),
  setReducedMotion: (reduced) => set({ reducedMotion: reduced }),
}));
