/**
 * Compute visual properties for graph nodes and edges based on their semantic data.
 * This replaces render metadata that was previously provided by the bridge.
 */

export interface NodeRenderProps {
  size: number;
  color: string;
  glow: number;
  opacity: number;
}

export interface EdgeRenderProps {
  thickness: number;
  opacity: number;
  color: string;
}

const TIER_SIZES: Record<number, number> = {
  1: 1.4,
  2: 1.0,
  3: 0.7,
};

const STATUS_COLORS: Record<string, string> = {
  stable: '#69F0AE',      // green
  candidate: '#00E5FF',   // cyan
  new: '#B388FF',         // purple
  deprecated: '#FFB74D',  // orange
  quarantine: '#FF5252',  // red
  decaying: '#FFB74D',    // orange (legacy compat)
  removed: '#FF5252',     // red (legacy compat)
};

/**
 * Compute visual render properties for a graph node based on its semantic data.
 * Seed nodes get larger sizing. Tier determines base size. Status determines color.
 * Confidence affects glow intensity and opacity.
 */
/**
 * Determine if a node is an atom (primitive, non-composed) based on its compression state.
 */
export function isAtomNode(node: {
  compression_state?: string;
  semantic?: { compression_state?: string };
  derived_from_node_ids?: number[];
  atoms?: number[];
}): boolean {
  const cs = node.compression_state ?? node.semantic?.compression_state ?? 'raw';
  const derived = node.derived_from_node_ids ?? node.semantic?.derived_from_node_ids ?? [];
  return cs === 'raw' && derived.length === 0 && (!node.atoms || node.atoms.length === 0);
}

/**
 * Determine if a node is a composite (built from other nodes).
 */
export function isCompositeNode(node: {
  compression_state?: string;
  semantic?: { compression_state?: string };
  derived_from_node_ids?: number[];
  atoms?: number[];
}): boolean {
  const cs = node.compression_state ?? node.semantic?.compression_state ?? 'raw';
  const derived = node.derived_from_node_ids ?? node.semantic?.derived_from_node_ids ?? [];
  return cs === 'compressed' || cs === 'composed' || derived.length > 0 || (node.atoms !== undefined && node.atoms.length > 0);
}

/**
 * Get the number of atoms in a composite node.
 */
export function getAtomCount(node: {
  atoms?: number[];
  derived_from_node_ids?: number[];
  semantic?: { derived_from_node_ids?: number[] };
  composition?: { atoms: Array<{ atom_id: number }>; related_composites: Array<{ composite_id: number }> };
}): number {
  if (node.atoms && node.atoms.length > 0) return node.atoms.length;
  if (node.derived_from_node_ids && node.derived_from_node_ids.length > 0) return node.derived_from_node_ids.length;
  if (node.semantic?.derived_from_node_ids && node.semantic.derived_from_node_ids.length > 0) return node.semantic.derived_from_node_ids.length;
  if (node.composition?.atoms) return node.composition.atoms.length;
  return 0;
}

const COMPOSITE_COLORS: Record<string, string> = {
  composed: '#FF80AB',    // Pink for composed
  compressed: '#B388FF',  // Purple for compressed
};

export function computeNodeRenderProps(node: {
  tier?: number;
  status?: string;
  is_seed?: boolean;
  confidence?: number;
  kind?: string;
  compression_state?: string;
  semantic?: { compression_state?: string };
  derived_from_node_ids?: number[];
  atoms?: number[];
  composition?: { atoms: Array<{ atom_id: number }>; related_composites: Array<{ composite_id: number }> };
}): NodeRenderProps {
  const tier = node.tier ?? 3;
  const status = node.status ?? 'new';
  const confidence = node.confidence ?? 0.5;
  const composite = isCompositeNode(node);
  const atomCount = getAtomCount(node);

  // Composites are slightly larger, scaling with atom count
  const baseSize = node.is_seed
    ? 1.6
    : (TIER_SIZES[tier] ?? 0.7);

  const size = composite
    ? baseSize * (1 + Math.min(atomCount * 0.1, 0.5))
    : baseSize;

  // Composites get a special color, atoms use status-based color
  const cs = node.compression_state ?? node.semantic?.compression_state ?? 'raw';
  const color = composite
    ? (COMPOSITE_COLORS[cs] ?? '#B388FF')
    : (STATUS_COLORS[status] ?? '#B388FF');

  const glow = composite
    ? Math.max(0.4, Math.min(0.95, confidence + 0.1))
    : Math.max(0.3, Math.min(0.9, confidence));
  const opacity = Math.max(0.4, Math.min(1.0, confidence + 0.3));

  return { size, color, glow, opacity };
}

/**
 * Compute visual render properties for a graph edge based on its weight.
 */
export function computeEdgeRenderProps(weight: number): EdgeRenderProps {
  return {
    thickness: Math.max(0.3, Math.min(2.0, weight * 2)),
    opacity: Math.max(0.2, Math.min(0.8, weight)),
    color: '#80D8FF',
  };
}

/**
 * Get status color for a given node status string.
 * Returns the hex color code.
 */
export function getStatusColor(status: string): string {
  return STATUS_COLORS[status] ?? '#B388FF';
}

/**
 * Get tier size multiplier for a given tier number.
 */
export function getTierSize(tier: number): number {
  return TIER_SIZES[tier] ?? 0.7;
}
