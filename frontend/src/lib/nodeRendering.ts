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
export function computeNodeRenderProps(node: {
  tier?: number;
  status?: string;
  is_seed?: boolean;
  confidence?: number;
  kind?: string;
}): NodeRenderProps {
  const tier = node.tier ?? 3;
  const status = node.status ?? 'new';
  const confidence = node.confidence ?? 0.5;

  const size = node.is_seed
    ? 1.6
    : (TIER_SIZES[tier] ?? 0.7);

  const color = STATUS_COLORS[status] ?? '#B388FF';
  const glow = Math.max(0.3, Math.min(0.9, confidence));
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
