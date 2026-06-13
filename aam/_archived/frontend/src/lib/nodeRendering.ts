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
  semantic?: { compression_state?: string; derived_from_node_ids?: number[] };
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
  semantic?: { compression_state?: string; derived_from_node_ids?: number[] };
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

// ── v8.0: Layer Color Map ──
// Layer 0 = gold (primitive seeds), Layer 1 = cyan (internal representation bridge),
// Layer 2 = green (first composed), Layer 3 = orange, Layer 4+ = deeper colors
const LAYER_COLORS: Record<number, string> = {
  0: '#FFD54F',   // Gold — primitive seed nodes
  1: '#00BCD4',   // Cyan — internal representation (layer 1 bridge)
  2: '#66BB6A',   // Green — first-level composed
  3: '#FFA726',   // Orange — second-level composed
  4: '#AB47BC',   // Purple — third-level composed
  5: '#EF5350',   // Red — fourth-level composed
};

const DEFAULT_LAYER_COLOR = '#EC407A'; // Pink for layer 6+

const LAYER_Y_OFFSET = 6; // Y-axis spacing between layers in 3D space

/**
 * v5.0: Get the color for a compositional layer.
 * Layer 0 = blue (primitive), Layer 1 = green, Layer 2 = orange, etc.
 */
export function getLayerColor(layer: number): string {
  return LAYER_COLORS[layer] ?? DEFAULT_LAYER_COLOR;
}

/**
 * v5.0: Get the Y-axis offset for a layer in the 3D visualization.
 * Layer 0 at bottom (y=0), higher layers stacked up.
 */
export function getLayerYOffset(layer: number): number {
  return layer * LAYER_Y_OFFSET;
}

/**
 * v8.0: Get the layer label for display.
 * Layer 0 = Seeds (language-agnostic primitives)
 * Layer 1 = Internal Representation (bridge between tokens and seeds)
 * Layer 2+ = Composed Concepts
 */
export function getLayerLabel(layer: number): string {
  if (layer === 0) return 'Layer 0 — Seeds';
  if (layer === 1) return 'Layer 1 — Internal Repr';
  return `Layer ${layer} — Composed`;
}

/**
 * v5.0: Get all defined layer color entries for legend rendering.
 */
export function getLayerColorEntries(): Array<{ layer: number; color: string; label: string }> {
  return Object.entries(LAYER_COLORS).map(([layer, color]) => ({
    layer: Number(layer),
    color,
    label: getLayerLabel(Number(layer)),
  }));
}

/**
 * v8.0: Check if a node is an internal representation (layer 1 bridge).
 * These nodes have compositions that reference ONLY layer 0 seed primitives.
 */
export function isInternalRepresentation(node: {
  internal_representation?: boolean;
  layer?: number;
}): boolean {
  return node.internal_representation === true || node.layer === 1;
}

/**
 * v8.0: Check if a node has convergence links (structural equivalence with other nodes).
 */
export function hasConvergenceLinks(node: {
  language_links?: Array<{ link_type: string; target_id: number }>;
}): boolean {
  return (node.language_links ?? []).some(ll => ll.link_type === 'structural_equivalence');
}

/**
 * v8.0: Get convergence link targets for a node.
 * Returns target IDs of structurally equivalent nodes.
 */
export function getConvergenceTargets(node: {
  language_links?: Array<{ link_type: string; target_id: number }>;
}): number[] {
  return (node.language_links ?? [])
    .filter(ll => ll.link_type === 'structural_equivalence')
    .map(ll => ll.target_id);
}

/**
 * v5.0: Compute the effective layer for a node.
 * Uses the `layer` field if available, otherwise infers from composition state.
 */
export function computeNodeLayer(node: {
  layer?: number;
  compression_state?: string;
  semantic?: { compression_state?: string };
  derived_from_node_ids?: number[];
  atoms?: number[];
  compositions?: Array<unknown>;
}): number {
  // If layer is explicitly set, use it
  if (node.layer !== undefined && node.layer !== null) return node.layer;
  // Otherwise, infer from composition state
  if (isCompositeNode(node)) return 1;
  return 0;
}

/**
 * v5.0: Build a composition chain string for display.
 * E.g., "raja = tahta_tertinggi + laki_laki + kerajaan"
 */
export function buildCompositionChain(
  nodeLabel: string,
  compositions?: Array<{ label: string }>,
  derivedFromNodeIds?: number[],
  allNodes?: Map<number, { label: string }>,
): string | null {
  // Try compositions first (v5.0)
  if (compositions && compositions.length > 0) {
    const parts = compositions.map(c => c.label).join(' + ');
    return parts.length > 0 ? `${nodeLabel} = ${parts}` : null;
  }

  // Fall back to derived_from_node_ids
  if (derivedFromNodeIds && derivedFromNodeIds.length > 0 && allNodes) {
    const labels = derivedFromNodeIds
      .map(id => allNodes.get(id)?.label ?? `#${id}`);
    if (labels.length > 0) {
      return `${nodeLabel} = ${labels.join(' + ')}`;
    }
  }

  return null;
}

export function computeNodeRenderProps(node: {
  tier?: number;
  status?: string;
  is_seed?: boolean;
  confidence?: number;
  kind?: string;
  compression_state?: string;
  semantic?: { compression_state?: string; derived_from_node_ids?: number[]; internal_representation?: boolean };
  derived_from_node_ids?: number[];
  atoms?: number[];
  composition?: { atoms: Array<{ atom_id: number }>; related_composites: Array<{ composite_id: number }> };
  // v5.0 fields
  layer?: number;
  compositions?: Array<unknown>;
  // v8.0 fields
  internal_representation?: boolean;
}): NodeRenderProps {
  const tier = node.tier ?? 3;
  const status = node.status ?? 'new';
  const confidence = node.confidence ?? 0.5;
  const composite = isCompositeNode(node);
  const atomCount = getAtomCount(node);

  // v5.0: Determine effective layer
  const layer = computeNodeLayer(node);

  // Composites are slightly larger, scaling with atom count
  const baseSize = node.is_seed
    ? 1.6
    : (TIER_SIZES[tier] ?? 0.7);

  const size = composite
    ? baseSize * (1 + Math.min(atomCount * 0.1, 0.5))
    : baseSize;

  // v8.0: Internal representation nodes get a distinctive cyan color with semi-transparent appearance
  const isInternalRepr = node.internal_representation === true || node.semantic?.internal_representation === true;

  // v5.0: Color-code by layer (overrides composite/status color when layer > 0)
  // Layer 0 nodes (seeds) get gold color
  // Layer 1 internal representation gets cyan with semi-transparency
  // Higher layers use the layer color directly
  const layerColor = getLayerColor(layer);
  let color: string;
  if (layer === 0 && node.is_seed) {
    // Seeds: gold color
    color = layerColor;
  } else if (isInternalRepr) {
    // Internal representation: cyan, the bridge color
    color = layerColor;
  } else if (layer === 0) {
    // Primitive: blend status color with a blue tint
    color = composite
      ? (COMPOSITE_COLORS[node.compression_state ?? node.semantic?.compression_state ?? 'raw'] ?? layerColor)
      : (STATUS_COLORS[status] ?? layerColor);
  } else {
    // Compositional: use layer color
    color = layerColor;
  }

  const glow = composite
    ? Math.max(0.4, Math.min(0.95, confidence + 0.1))
    : Math.max(0.3, Math.min(0.9, confidence));
  // v8.0: Internal representation nodes are slightly transparent
  const opacity = isInternalRepr
    ? Math.max(0.5, Math.min(0.85, confidence + 0.2))
    : Math.max(0.4, Math.min(1.0, confidence + 0.3));

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
