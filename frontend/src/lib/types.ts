// RSVS Core Type Definitions

export type NodeKind = 'atom' | 'composite';
export type Tier = 1 | 2 | 3;
export type NodeStatus = 'new' | 'stable' | 'decaying' | 'removed';
export type EdgeStatus = 'new' | 'stable' | 'updated' | 'removing';
export type EdgeDirection = 'directed' | 'undirected';
export type SourceType = 'bootstrap' | 'learned';
export type AnimationPriority = 'low' | 'normal' | 'high';
export type ViewMode = 'explore' | 'analyze' | 'presentation';
export type TimelineState = 'live' | 'paused';

export type EventType =
  | 'atom_created'
  | 'atom_removed'
  | 'edge_created'
  | 'edge_removed'
  | 'edge_weight_changed'
  | 'tier_changed'
  | 'confidence_changed'
  | 'sense_changed';

export interface NodeSense {
  count: number;
  active_index: number | null;
  coherence: number | null;
}

export interface NodeMetrics {
  degree: number;
  in_degree: number;
  out_degree: number;
  last_updated_at: string;
}

export interface CompositionAtom {
  atom_id: number;
  weight: number;
}

export interface CompositionComposite {
  composite_id: number;
  weight: number;
}

export interface NodeComposition {
  atoms: CompositionAtom[];
  related_composites: CompositionComposite[];
}

export interface NodeRender {
  position: { x: number; y: number; z: number };
  size: number;
  color: string;
  glow: number;
}

export interface NodeProvenance {
  source_batch_id: string | null;
  source_domain: string | null;
  source_type: SourceType;
}

export interface RSVSNode {
  id: number;
  label: string;
  kind: NodeKind;
  tier: Tier;
  confidence: number;
  status: NodeStatus;
  sense?: NodeSense;
  metrics?: NodeMetrics;
  composition?: NodeComposition;
  render?: NodeRender;
  provenance?: NodeProvenance;
}

export interface EdgeMetrics {
  cooc: number | null;
  npmi: number | null;
  jaccard: number | null;
  last_updated_at: string;
}

export interface EdgeRender {
  thickness: number;
  color: string;
  opacity: number;
  pulse: number;
}

export interface RSVSEdge {
  id: string;
  source: number;
  target: number;
  direction: EdgeDirection;
  weight: number;
  source_type: SourceType;
  status: EdgeStatus;
  metrics?: EdgeMetrics;
  render?: EdgeRender;
}

export interface AnimationHint {
  priority: AnimationPriority;
  focus_node_id: number | null;
  burst_group: string | null;
}

export interface RSVSEvent {
  event_id: string;
  timestamp: string;
  correlation_id: string;
  event_type: EventType;
  payload: {
    node?: Partial<RSVSNode>;
    edge?: Partial<RSVSEdge>;
    before?: Record<string, unknown>;
    after?: Record<string, unknown>;
  };
  animation_hint: AnimationHint;
}

export interface GraphSnapshot {
  snapshot_id: string;
  generated_at: string;
  context: {
    domain: string;
    batch_id: string;
    input_message_id: string;
  };
  nodes: RSVSNode[];
  edges: RSVSEdge[];
}

export type MessageType = 'user_input' | 'system_ingest_status' | 'system_promoted_atoms' | 'system_warnings';

export interface ChatMessage {
  id: string;
  type: MessageType;
  content: string;
  timestamp: string;
  correlation_id?: string;
  mode?: 'ingest' | 'appraise' | 'relate';
}

export interface TimelineEvent {
  event_id: string;
  timestamp: string;
  event_type: EventType;
  label: string;
  data?: Record<string, unknown>;
}

export interface FilterState {
  tiers: Tier[];
  confidenceRange: [number, number];
  nodeKinds: NodeKind[];
  sourceBatch: string | null;
  recentActivity: number | null; // seconds
}

export interface AnimationQueueItem {
  id: string;
  type: EventType;
  startTime: number;
  duration: number;
  nodeId?: number;
  edgeId?: string;
  completed: boolean;
}

// 3D layout types
export interface ForceNode extends RSVSNode {
  fx?: number;
  fy?: number;
  fz?: number;
  vx?: number;
  vy?: number;
  vz?: number;
}
