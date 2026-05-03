// RSVS Mock Data Generator
// Generates realistic graph snapshots and event streams for demo

import type {
  RSVSNode, RSVSEdge, RSVSEvent, ChatMessage, TimelineEvent,
  NodeKind, Tier, SourceType, NodeStatus, EdgeStatus,
} from '@/lib/types';

const ATOM_LABELS = [
  'neural', 'synapse', 'cortex', 'quantum', 'entropy', 'vector', 'tensor',
  'graph', 'topology', 'manifold', 'gradient', 'embedding', 'attention',
  'transformer', 'latent', 'diffusion', 'energy', 'harmonic', 'spectral',
  'coherence', 'resonance', 'oscillation', 'dimension', 'projection',
  'kernel', 'matrix', 'eigenvalue', 'convergence', 'divergence', 'flux',
];

const COMPOSITE_LABELS = [
  'neural-graph', 'quantum-tensor', 'spectral-embedding', 'harmonic-resonance',
  'diffusion-manifold', 'attention-transformer', 'latent-projection',
  'energy-gradient', 'coherence-flux', 'topology-kernel',
];

const DOMAINS = ['nlp', 'vision', 'reasoning', 'memory', 'learning'];

let eventIdCounter = 0;
let nodeIdCounter = 100;
let correlationCounter = 0;

function rand(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function randInt(min: number, max: number): number {
  return Math.floor(rand(min, max));
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function generateNodeId(): number {
  return ++nodeIdCounter;
}

function generateEventId(): string {
  return `evt_${++eventIdCounter}`;
}

function generateCorrelationId(): string {
  return `msg_demo_${++correlationCounter}`;
}

function generatePosition(): { x: number; y: number; z: number } {
  return {
    x: rand(-20, 20),
    y: rand(-15, 15),
    z: rand(-20, 20),
  };
}

const TIER_COLORS: Record<Tier, string> = {
  1: '#00E5FF',  // Cyan - stable, cool
  2: '#FFB74D',  // Warm amber
  3: '#FF5252',  // Critical red
};

const COMPOSITE_COLORS = [
  '#B388FF', '#69F0AE', '#FF80AB', '#80D8FF', '#FFD740',
];

function generateNode(kind: NodeKind = 'atom', tier?: Tier): RSVSNode {
  const id = generateNodeId();
  const t = tier || (randInt(1, 4) as Tier);
  const isAtom = kind === 'atom';

  return {
    id,
    label: isAtom ? pick(ATOM_LABELS) : pick(COMPOSITE_LABELS),
    kind,
    tier: t,
    confidence: rand(0.3, 1.0),
    status: 'new' as NodeStatus,
    sense: {
      count: randInt(1, 6),
      active_index: 0,
      coherence: rand(0.2, 1.0),
    },
    metrics: {
      degree: randInt(1, 12),
      in_degree: randInt(0, 6),
      out_degree: randInt(1, 8),
      last_updated_at: new Date().toISOString(),
    },
    composition: {
      atoms: [],
      related_composites: [],
    },
    render: {
      position: generatePosition(),
      size: isAtom ? rand(0.5, 1.2) : rand(1.2, 2.0),
      color: isAtom ? TIER_COLORS[t] : pick(COMPOSITE_COLORS),
      glow: rand(0.1, 0.6),
    },
    provenance: {
      source_batch_id: `ingest_${String(randInt(1, 100)).padStart(5, '0')}`,
      source_domain: pick(DOMAINS),
      source_type: (Math.random() > 0.3 ? 'learned' : 'bootstrap') as SourceType,
    },
  };
}

function generateEdge(source: number, target: number, sourceType?: SourceType): RSVSEdge {
  return {
    id: `${source}->${target}`,
    source,
    target,
    direction: Math.random() > 0.4 ? 'directed' : 'undirected',
    weight: rand(0.2, 1.0),
    source_type: sourceType || (Math.random() > 0.3 ? 'learned' : 'bootstrap') as SourceType,
    status: 'new' as EdgeStatus,
    metrics: {
      cooc: rand(0.1, 0.9),
      npmi: rand(0.0, 0.8),
      jaccard: rand(0.05, 0.7),
      last_updated_at: new Date().toISOString(),
    },
    render: {
      thickness: rand(0.5, 3.0),
      color: '#89D7FF',
      opacity: rand(0.3, 0.8),
      pulse: rand(0, 0.5),
    },
  };
}

export function generateInitialSnapshot(count: number = 30): { nodes: RSVSNode[]; edges: RSVSEdge[] } {
  const nodes: RSVSNode[] = [];
  const edges: RSVSEdge[] = [];

  // Generate atoms
  for (let i = 0; i < count; i++) {
    const node = generateNode('atom');
    node.status = 'stable';
    node.render!.glow = rand(0.05, 0.3);
    nodes.push(node);
  }

  // Generate composites
  for (let i = 0; i < Math.floor(count * 0.2); i++) {
    const node = generateNode('composite');
    node.status = 'stable';
    node.render!.glow = rand(0.05, 0.3);
    // Link some atoms to this composite
    const memberCount = randInt(2, 5);
    for (let j = 0; j < memberCount; j++) {
      const atom = nodes[randInt(0, count)];
      if (atom) {
        node.composition!.atoms.push({ atom_id: atom.id, weight: rand(0.3, 1.0) });
        const edge = generateEdge(atom.id, node.id, 'learned');
        edge.status = 'stable';
        edge.render!.opacity = rand(0.3, 0.7);
        edges.push(edge);
      }
    }
    nodes.push(node);
  }

  // Generate extra edges for density
  const extraEdges = Math.floor(count * 0.8);
  for (let i = 0; i < extraEdges; i++) {
    const a = nodes[randInt(0, nodes.length)];
    const b = nodes[randInt(0, nodes.length)];
    if (a && b && a.id !== b.id) {
      const edgeId = `${a.id}->${b.id}`;
      if (!edges.find((e) => e.id === edgeId || e.id === `${b.id}->${a.id}`)) {
        const edge = generateEdge(a.id, b.id);
        edge.status = 'stable';
        edges.push(edge);
      }
    }
  }

  return { nodes, edges };
}

export function generateEventStream(
  existingNodes: RSVSNode[],
  count: number = 15,
): RSVSEvent[] {
  const events: RSVSEvent[] = [];
  const corrId = generateCorrelationId();

  for (let i = 0; i < count; i++) {
    const eventType = pick([
      'atom_created', 'edge_created', 'confidence_changed',
      'edge_weight_changed', 'tier_changed',
    ]) as RSVSEvent['event_type'];

    const timestamp = new Date(Date.now() + i * rand(200, 800)).toISOString();

    switch (eventType) {
      case 'atom_created': {
        const node = generateNode();
        node.id = generateNodeId();
        events.push({
          event_id: generateEventId(),
          timestamp,
          correlation_id: corrId,
          event_type: 'atom_created',
          payload: { node },
          animation_hint: {
            priority: 'high',
            focus_node_id: node.id,
            burst_group: corrId,
          },
        });
        break;
      }
      case 'edge_created': {
        const a = existingNodes.length > 0 ? pick(existingNodes) : { id: 100, label: 'root' };
        const edge = generateEdge(a.id, generateNodeId());
        events.push({
          event_id: generateEventId(),
          timestamp,
          correlation_id: corrId,
          event_type: 'edge_created',
          payload: { edge },
          animation_hint: {
            priority: 'normal',
            focus_node_id: a.id,
            burst_group: corrId,
          },
        });
        break;
      }
      case 'confidence_changed': {
        const node = existingNodes.length > 0 ? pick(existingNodes) : { id: 100, label: 'root' };
        const before = node.confidence;
        const after = Math.min(1.0, Math.max(0.0, before + rand(-0.15, 0.2)));
        events.push({
          event_id: generateEventId(),
          timestamp,
          correlation_id: corrId,
          event_type: 'confidence_changed',
          payload: {
            node: { id: node.id, label: node.label },
            before: { confidence: before },
            after: { confidence: after },
          },
          animation_hint: {
            priority: 'normal',
            focus_node_id: node.id,
            burst_group: corrId,
          },
        });
        break;
      }
      case 'edge_weight_changed': {
        const node = existingNodes.length > 0 ? pick(existingNodes) : { id: 100, label: 'root' };
        const before = rand(0.2, 0.5);
        const after = Math.min(1.0, before + rand(0.1, 0.4));
        events.push({
          event_id: generateEventId(),
          timestamp,
          correlation_id: corrId,
          event_type: 'edge_weight_changed',
          payload: {
            edge: { id: `${node.id}->300`, source: node.id, target: 300 },
            before: { weight: before },
            after: { weight: after },
          },
          animation_hint: {
            priority: 'low',
            focus_node_id: node.id,
            burst_group: corrId,
          },
        });
        break;
      }
      case 'tier_changed': {
        const node = existingNodes.length > 0 ? pick(existingNodes) : { id: 100, label: 'root' };
        const oldTier = randInt(1, 4) as Tier;
        const newTier = (oldTier % 3) + 1 as Tier;
        events.push({
          event_id: generateEventId(),
          timestamp,
          correlation_id: corrId,
          event_type: 'tier_changed',
          payload: {
            node: { id: node.id, label: node.label },
            before: { tier: oldTier },
            after: { tier: newTier },
          },
          animation_hint: {
            priority: 'high',
            focus_node_id: node.id,
            burst_group: corrId,
          },
        });
        break;
      }
    }
  }

  return events;
}

export function generateChatMessages(): ChatMessage[] {
  return [
    {
      id: 'chat_1',
      type: 'user_input',
      content: 'Initialize RSVS with geology domain corpus',
      timestamp: new Date(Date.now() - 30000).toISOString(),
    },
    {
      id: 'chat_2',
      type: 'system_ingest_status',
      content: 'Ingesting batch ingest_00047 — 1,247 tokens processed, 12 atoms promoted, 3 composites formed.',
      timestamp: new Date(Date.now() - 28000).toISOString(),
      correlation_id: 'msg_demo_1',
    },
    {
      id: 'chat_3',
      type: 'system_promoted_atoms',
      content: 'New atoms: mineral (T2, c=0.50), sediment (T2, c=0.43), erosion (T1, c=0.71)',
      timestamp: new Date(Date.now() - 26000).toISOString(),
      correlation_id: 'msg_demo_1',
    },
    {
      id: 'chat_4',
      type: 'user_input',
      content: 'Add physics domain — quantum mechanics concepts',
      timestamp: new Date(Date.now() - 15000).toISOString(),
    },
    {
      id: 'chat_5',
      type: 'system_ingest_status',
      content: 'Ingesting batch ingest_00048 — 2,891 tokens, cross-domain edges forming. 8 atoms promoted.',
      timestamp: new Date(Date.now() - 12000).toISOString(),
      correlation_id: 'msg_demo_2',
    },
  ];
}

export function generateTimelineEvents(events: RSVSEvent[]): TimelineEvent[] {
  return events.map((evt, i) => ({
    event_id: evt.event_id,
    timestamp: evt.timestamp,
    event_type: evt.event_type,
    label: formatEventLabel(evt),
    data: evt.payload,
  }));
}

function formatEventLabel(evt: RSVSEvent): string {
  const nodeLabel = evt.payload.node?.label ?? '?';
  const edgeId = evt.payload.edge?.id ?? '?';
  switch (evt.event_type) {
    case 'atom_created':
      return `Atom "${nodeLabel}" created`;
    case 'edge_created':
      return `Edge ${edgeId} connected`;
    case 'confidence_changed':
      return `Confidence updated for ${nodeLabel}`;
    case 'edge_weight_changed':
      return `Weight changed for edge ${edgeId}`;
    case 'tier_changed':
      return `Tier changed for ${nodeLabel}`;
    case 'atom_removed':
      return `Atom ${nodeLabel} removed`;
    default:
      return evt.event_type;
  }
}
