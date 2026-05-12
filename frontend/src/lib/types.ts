// RSVS Core Type Definitions

export type NodeKind = 'node';
export type Tier = 1 | 2 | 3;
export type NodeStatus = 'new' | 'stable' | 'decaying' | 'removed' | 'candidate' | 'deprecated' | 'quarantine';
export type EdgeStatus = 'new' | 'stable' | 'updated' | 'removing' | 'deprecated';
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
  // v6.0 compositional fields
  layer?: number;
  grounding_score?: number;
  grounding_evidence?: GroundingEvidence;
  compositions?: CompositionPair[];
  // Individual sense entries for multi-sense display
  senses?: SenseEntry[];
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

// v6.0: Composition pair — [label, sense_id] used in compose API
export interface CompositionPair {
  label: string;
  sense_id: string;
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

export type CompressionState = 'raw' | 'compressed' | 'composed';

// v6.0: Grounding evidence for a sense
export type GroundingVerdict = 'well_grounded' | 'needs_review' | 'needs_revision';

export interface GroundingEvidence {
  confirming_contexts: number;
  contradicting_contexts: number;
  last_contradiction: string | null;
  revision_count: number;
  score: number;
  verdict: GroundingVerdict;
}

// v6.0: Transformer Bridge configuration
export interface TransformerBridgeConfig {
  similarity_threshold: number;
  max_compositions: number;
  use_attention_weights: boolean;
}

export interface PolicyMeta {
  governance_score: number;
  status_flip_count: number;
  auto_promote: boolean;
  max_tier: Tier;
  min_confidence: number;
}

// v8.0: Language link matching the Rust backend's LanguageLink struct.
// link_type is typically "structural_equivalence" from convergence detection.
// target_id references another node that this node is structurally equivalent to.
export interface LanguageLink {
  link_type: string;
  target_id: number;
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
  // v4.2 fields
  is_seed?: boolean;
  semantic?: {
    compression_state: CompressionState;
    derived_from_node_ids: number[];
    compression_reason?: string;
  };
  compression_state?: CompressionState;
  derived_from_node_ids?: number[];
  policy_meta?: PolicyMeta;
  language_links?: LanguageLink[];
  // Composition visualization fields
  atoms?: number[];
  compression_reason?: string;
  // v6.0 compositional architecture fields
  layer?: number; // 0=primitive, 1=internal representation, N=compositional
  grounding_score?: number;
  grounding_evidence?: GroundingEvidence;
  compositions?: CompositionPair[]; // list of [label, sense_id] pairs
  // v8.0: Whether this node is an internal representation (layer 1 bridge)
  internal_representation?: boolean;
  // Structural information fields for enhanced display
  composition_references?: CompositionReference[];
  substitution_pairs?: SubstitutionPairInfo[];
  convergence_links?: ConvergenceLinkInfo[];
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

export interface EdgeEvidence {
  source_node_ids: number[];
  path: number[][];
  strength: number;
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
  // v4.2 fields
  label?: string;
  evidence?: EdgeEvidence[];
  // Structural edge types for visual differentiation
  edge_type?: EdgeType;
}

// Structural edge types for visual differentiation in the graph
export type EdgeType =
  | 'regular'          // Default graph edge
  | 'composition'      // Atom → Composite reference
  | 'convergence'      // Structural equivalence link (dashed)
  | 'substitution';    // Substitution pair connection

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

export type MessageType = 'user_input' | 'system_ingest_status' | 'system_promoted_atoms' | 'system_warnings' | 'system_compose_result';

export interface ChatMessage {
  id: string;
  type: MessageType;
  content: string;
  timestamp: string;
  correlation_id?: string;
  mode?: 'ingest' | 'appraise' | 'relate' | 'compose' | 'grounding_info';
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

// ── Mode Result State ──

export interface ComposeResult {
  composite_node: RSVSNode;
  atom_nodes: RSVSNode[];
  jaccard_similarities: Array<{
    composite_id: number;
    composite_label: string;
    similarity: number;
  }>;
  // v5.0: compositions used (if any)
  compositions?: CompositionPair[];
}

export type ModeResult =
  | { type: 'appraise'; data: AppraiseResult }
  | { type: 'relate'; data: RelateResult }
  | { type: 'compose'; data: ComposeResult }
  | { type: 'structural_similarity'; data: StructuralSimilarityResult }
  | { type: 'substitution_analysis'; data: SubstitutionAnalysisResult }
  | { type: 'grounding_info'; data: GroundingInfoResult }
  | null;

// ── v6.0: Grounding Info Result ──

export interface GroundingInfoResult {
  label: string;
  sense_id: number;
  grounding_evidence: GroundingEvidence;
  composition_details: Array<{
    label: string;
    sense_id: number;
    confirmed: boolean;
  }>;
}

// ── v6.0: Structural Similarity ──

export interface SharedComposition {
  label: string;
  sense_id_a: string;
  sense_id_b: string;
  similarity: number;
}

export interface DifferingComposition {
  label: string;
  present_in: 'a' | 'b';
  sense_id: string;
  weight: number;
}

export interface StructuralSimilarityResult {
  node_a: { label: string; id: number };
  node_b: { label: string; id: number };
  similarity_score: number;
  shared_compositions: SharedComposition[];
  differing_compositions: DifferingComposition[];
}

// ── v6.0: Substitution Analysis ──

export interface SubstitutionPair {
  atom_a: { label: string; id: number };
  atom_b: { label: string; id: number };
  substitution_score: number;
  semantic_shift: string;
}

export interface SubstitutionAnalysisResult {
  node_a: { label: string; id: number };
  node_b: { label: string; id: number };
  substitution_pairs: SubstitutionPair[];
}

// ── Structural Information Types (F-02) ──

/** Individual sense entry for multi-sense display */
export interface SenseEntry {
  sense_id: number;
  label: string;
  composition: string[];  // labels of composition atoms
  confidence: number;
  status: 'mature' | 'fragile' | 'emerging' | 'deprecated';
  coherence: number;
  grounding_score?: number;
}

/** Composition reference: a link from this node to another node it's composed of */
export interface CompositionReference {
  ref_node_id: number;
  ref_label: string;
  weight: number;
  sense_id?: number;
}

/** Substitution pair info: edge difference highlighting */
export interface SubstitutionPairInfo {
  atom_a_id: number;
  atom_a_label: string;
  atom_b_id: number;
  atom_b_label: string;
  substitution_score: number;
  semantic_shift: string;
}

/** Convergence link: structural equivalence between nodes */
export interface ConvergenceLinkInfo {
  target_id: number;
  target_label: string;
  link_type: string;
  strength: number;
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

// ── Appraise Result ──

export type AppraiseVerdict = 'agree' | 'mixed' | 'disagree';

export interface AppraiseStance {
  agree: number;
  disagree: number;
  neutral: number;
}

export interface AppraiseEvidenceNode {
  node_id: number;
  label: string;
  confidence: number;
  role: 'support' | 'conflict' | 'neutral';
}

export interface EvidencePath {
  path: number[];
  weight: number;
  label?: string;
}

export interface AppraiseResult {
  verdict: AppraiseVerdict;
  stance: AppraiseStance;
  confidence: number;
  rationale: string;
  evidence_nodes: AppraiseEvidenceNode[];
  conflict_nodes: AppraiseEvidenceNode[];
  evidence_paths: EvidencePath[];
  target_node_id?: number;
  /** v8.2: Convergent nodes that contributed to the appraise score. */
  convergence_contributors?: Array<{ label: string; boost: number }>;
}

// ── Relate Result ──

export interface RelateNode {
  node_id: number;
  label: string;
  score: number;
  tier: Tier;
  kind: NodeKind;
  // v6.0 fields
  layer?: number;
  grounding_score?: number;
  grounding_evidence?: GroundingEvidence;
  compositions?: CompositionPair[];
}

export interface RelateEdge {
  edge_id: string;
  source: number;
  target: number;
  weight: number;
  label?: string;
}

export interface StructuralRelation {
  relation_type: string;
  source_label: string;
  target_label: string;
  weight: number;
  description?: string;
}

export interface RelateResult {
  query_terms: string[];
  related_nodes: RelateNode[];
  related_edges: RelateEdge[];
  // v6.0 field
  structural_relations?: StructuralRelation[];
}
