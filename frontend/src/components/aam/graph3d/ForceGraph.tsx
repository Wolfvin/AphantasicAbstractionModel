'use client';

import { useEffect, useRef } from 'react';
import { useGraphStore } from '@/store/aamStore';
import { computeNodeRenderProps, computeEdgeRenderProps, isCompositeNode, isAtomNode, getAtomCount, computeNodeLayer, getLayerYOffset, isInternalRepresentation } from '@/lib/nodeRendering';

// ── Simulation Constants ──
const REPULSION_CONSTANT = 600;
const ATTRACTION_CONSTANT = 0.012;
const CENTER_GRAVITY = 0.004;
const DAMPING = 0.88;
const MIN_DISTANCE = 1;
const ENERGY_THRESHOLD = 0.005;
const IDEAL_EDGE_LENGTH = 8;
const SEED_SPHERE_RADIUS = 5;
const MAX_VELOCITY = 2.0;

// v8.0: Layer 1 ring radius — internal representation nodes orbit around the seed cluster
const LAYER1_RING_RADIUS = 10;
// v8.0: Layer 2+ outer orbit radius
const LAYER2_ORBIT_RADIUS = 18;

// Composition clustering: extra attraction between composites and their atoms
const COMPOSITION_ATTRACTION_CONSTANT = 0.025;
const COMPOSITION_IDEAL_DISTANCE = 4;

// v5.0: Layer Y-force — gently pulls nodes toward their layer's Y-offset
const LAYER_Y_GRAVITY = 0.008;

// ── Barnes-Hut Optimization Constants ──
// Theta parameter: if s/d < theta, treat cluster as single body.
// Lower = more accurate, higher = faster. 0.5 is typical.
const BARNES_HUT_THETA = 0.5;

// Frame budget in ms — skip force calc if previous frame took too long
const FRAME_BUDGET_MS = 16; // ~60fps target

// Maximum delta time cap to avoid instability after tab switch
const MAX_DELTA_MS = 50;

// ── Barnes-Hut Octree Implementation ──

interface OctreeNode {
  // Bounding region center and half-size
  cx: number;
  cy: number;
  cz: number;
  half: number;

  // Aggregate mass and center-of-mass
  mass: number;
  comX: number;
  comY: number;
  comZ: number;

  // Children (8 octants), null = empty
  children: (OctreeNode | null)[];
  // Whether this node contains a single body (leaf with body)
  bodyIndex: number; // -1 if internal or empty
}

function createOctreeNode(cx: number, cy: number, cz: number, half: number): OctreeNode {
  return {
    cx, cy, cz, half,
    mass: 0,
    comX: 0, comY: 0, comZ: 0,
    children: new Array(8).fill(null),
    bodyIndex: -1,
  };
}

function octantIndex(node: OctreeNode, x: number, y: number, z: number): number {
  let idx = 0;
  if (x >= node.cx) idx |= 1;
  if (y >= node.cy) idx |= 2;
  if (z >= node.cz) idx |= 4;
  return idx;
}

function childCenter(parent: OctreeNode, octant: number): { cx: number; cy: number; cz: number; half: number } {
  const qh = parent.half * 0.5;
  return {
    cx: parent.cx + (octant & 1 ? qh : -qh),
    cy: parent.cy + (octant & 2 ? qh : -qh),
    cz: parent.cz + (octant & 4 ? qh : -qh),
    half: qh,
  };
}

function insertIntoOctree(root: OctreeNode, x: number, y: number, z: number, mass: number, bodyIdx: number): void {
  // If this node is empty, just place the body here
  if (root.bodyIndex === -1 && root.mass === 0) {
    root.bodyIndex = bodyIdx;
    root.mass = mass;
    root.comX = x;
    root.comY = y;
    root.comZ = z;
    return;
  }

  // If this is a leaf with an existing body, subdivide
  if (root.bodyIndex !== -1) {
    const existingIdx = root.bodyIndex;
    // We need the existing body's position — but we don't store it here.
    // Instead, we store it at insertion time via the positions array.
    // For now, we'll use the COM which equals the single body's position.
    const ex = root.comX;
    const ey = root.comY;
    const ez = root.comZ;
    const eMass = root.mass;

    root.bodyIndex = -1; // Mark as internal node

    // Re-insert the existing body into the appropriate child
    const existingOctant = octantIndex(root, ex, ey, ez);
    const cc = childCenter(root, existingOctant);
    root.children[existingOctant] = createOctreeNode(cc.cx, cc.cy, cc.cz, cc.half);
    insertIntoOctree(root.children[existingOctant]!, ex, ey, ez, eMass, existingIdx);
  }

  // Insert the new body into the appropriate child
  const octant = octantIndex(root, x, y, z);
  if (root.children[octant] === null) {
    const cc = childCenter(root, octant);
    root.children[octant] = createOctreeNode(cc.cx, cc.cy, cc.cz, cc.half);
  }
  insertIntoOctree(root.children[octant]!, x, y, z, mass, bodyIdx);

  // Update aggregate mass and center-of-mass
  const totalMass = root.mass + mass;
  root.comX = (root.comX * root.mass + x * mass) / totalMass;
  root.comY = (root.comY * root.mass + y * mass) / totalMass;
  root.comZ = (root.comZ * root.mass + z * mass) / totalMass;
  root.mass = totalMass;
}

/**
 * Compute repulsive force on a body at (x,y,z) from the octree.
 * Uses the Barnes-Hut approximation: if the node is far enough (s/d < theta),
 * treat it as a single body at its center-of-mass.
 */
function computeRepulsiveForce(
  node: OctreeNode,
  x: number, y: number, z: number,
  theta: number,
  repulsionConst: number,
  tierMultiplier: number,
  // Output: accumulate into these
  fxRef: { fx: number; fy: number; fz: number },
  // Positions and tier multipliers for leaf lookups
  positions: Array<{ x: number; y: number; z: number }>,
  tierMultipliers: Float64Array,
): void {
  if (node.mass === 0) return;

  const dx = x - node.comX;
  const dy = y - node.comY;
  const dz = z - node.comZ;
  const distSq = dx * dx + dy * dy + dz * dz;

  // s/d ratio: size of node / distance to node
  const s = node.half * 2; // full width
  const dist = Math.sqrt(Math.max(distSq, MIN_DISTANCE * MIN_DISTANCE));

  if (node.bodyIndex !== -1) {
    // Leaf node: direct body-body interaction
    // Don't compute force on self
    const otherTierMult = tierMultipliers[node.bodyIndex];
    const force = repulsionConst * tierMultiplier * otherTierMult / Math.max(distSq, MIN_DISTANCE * MIN_DISTANCE);
    const invDist = 1 / dist;
    fxRef.fx += force * dx * invDist;
    fxRef.fy += force * dy * invDist;
    fxRef.fz += force * dz * invDist;
    return;
  }

  // Internal node: check if it's far enough to approximate
  if (s / dist < theta) {
    // Treat as single body at center-of-mass
    // Use average tier multiplier for the cluster (approximation)
    const avgTierMult = 1.0; // Safe default for cluster approximation
    const force = repulsionConst * tierMultiplier * avgTierMult / Math.max(distSq, MIN_DISTANCE * MIN_DISTANCE);
    const invDist = 1 / dist;
    fxRef.fx += force * dx * invDist;
    fxRef.fy += force * dy * invDist;
    fxRef.fz += force * dz * invDist;
    return;
  }

  // Too close: recurse into children
  for (let i = 0; i < 8; i++) {
    const child = node.children[i];
    if (child !== null) {
      computeRepulsiveForce(child, x, y, z, theta, repulsionConst, tierMultiplier, fxRef, positions, tierMultipliers);
    }
  }
}

// ── Spatial Grid for Neighbor Lookup ──

const GRID_CELL_SIZE = 10; // Size of each grid cell

interface SpatialGrid {
  cells: Map<number, number[]>; // cell hash → list of node indices
  invCellSize: number;
}

function spatialHash(x: number, y: number, z: number, invCellSize: number): number {
  // Simple spatial hash using cell coordinates
  const ix = Math.floor(x * invCellSize);
  const iy = Math.floor(y * invCellSize);
  const iz = Math.floor(z * invCellSize);
  // Mix the coordinates into a single hash
  return ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) | 0;
}

function buildSpatialGrid(
  positions: Array<{ x: number; y: number; z: number }>,
  cellSize: number,
): SpatialGrid {
  const invCellSize = 1 / cellSize;
  const cells = new Map<number, number[]>();

  for (let i = 0; i < positions.length; i++) {
    const hash = spatialHash(positions[i].x, positions[i].y, positions[i].z, invCellSize);
    let cell = cells.get(hash);
    if (!cell) {
      cell = [];
      cells.set(hash, cell);
    }
    cell.push(i);
  }

  return { cells, invCellSize };
}

/**
 * Get neighbor indices from spatial grid (within adjacent cells).
 */
function getNeighborIndices(
  grid: SpatialGrid,
  x: number, y: number, z: number,
): number[] {
  const inv = grid.invCellSize;
  const ix = Math.floor(x * inv);
  const iy = Math.floor(y * inv);
  const iz = Math.floor(z * inv);

  const neighbors: number[] = [];
  // Check the 27 neighboring cells (including own cell)
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dz = -1; dz <= 1; dz++) {
        const hash = spatialHash(
          (ix + dx) * (1 / inv),
          (iy + dy) * (1 / inv),
          (iz + dz) * (1 / inv),
          inv,
        );
        // Recompute hash properly
        const nix = ix + dx;
        const niy = iy + dy;
        const niz = iz + dz;
        const nHash = ((nix * 73856093) ^ (niy * 19349663) ^ (niz * 83492791)) | 0;
        const cell = grid.cells.get(nHash);
        if (cell) {
          for (const idx of cell) {
            neighbors.push(idx);
          }
        }
      }
    }
  }
  return neighbors;
}

interface VelocityMap {
  [nodeId: number]: { vx: number; vy: number; vz: number };
}

/**
 * Position seed nodes on a sphere at the center of the graph.
 * Uses golden angle distribution for even spacing on a sphere.
 */
function computeSeedPosition(index: number, total: number): { x: number; y: number; z: number } {
  if (total <= 1) return { x: 0, y: 0, z: 0 };

  // Golden angle distribution for even sphere coverage
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const theta = goldenAngle * index;
  const phi = Math.acos(1 - (2 * (index + 0.5)) / total);

  return {
    x: SEED_SPHERE_RADIUS * Math.sin(phi) * Math.cos(theta),
    y: SEED_SPHERE_RADIUS * Math.sin(phi) * Math.sin(theta),
    z: SEED_SPHERE_RADIUS * Math.cos(phi),
  };
}

/**
 * v8.0: Position layer 1 nodes on a ring around the seed cluster.
 * Uses golden angle distribution for even spacing on a circle.
 */
function computeLayer1RingPosition(index: number, total: number): { x: number; z: number } {
  if (total <= 1) return { x: LAYER1_RING_RADIUS, z: 0 };

  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const theta = goldenAngle * index;

  return {
    x: LAYER1_RING_RADIUS * Math.cos(theta),
    z: LAYER1_RING_RADIUS * Math.sin(theta),
  };
}

/**
 * v8.0: Position layer 2+ nodes in an outer orbit.
 */
function computeLayer2OrbitPosition(index: number, total: number): { x: number; z: number } {
  if (total <= 1) return { x: LAYER2_ORBIT_RADIUS, z: 0 };

  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const theta = goldenAngle * index;
  const radiusVariation = LAYER2_ORBIT_RADIUS + (Math.random() - 0.5) * 6;

  return {
    x: radiusVariation * Math.cos(theta),
    z: radiusVariation * Math.sin(theta),
  };
}
function getAtomIds(node: {
  atoms?: number[];
  derived_from_node_ids?: number[];
  semantic?: { derived_from_node_ids?: number[] };
  composition?: { atoms: Array<{ atom_id: number }>; related_composites: Array<{ composite_id: number }> };
}): number[] {
  if (node.atoms && node.atoms.length > 0) return node.atoms;
  if (node.derived_from_node_ids && node.derived_from_node_ids.length > 0) return node.derived_from_node_ids;
  if (node.semantic?.derived_from_node_ids && node.semantic.derived_from_node_ids.length > 0) return node.semantic.derived_from_node_ids;
  if (node.composition?.atoms && node.composition.atoms.length > 0) return node.composition.atoms.map(a => a.atom_id);
  return [];
}

/**
 * Enhanced n-body force-directed layout simulation with:
 * - Seed node sphere positioning at the center
 * - Barnes-Hut approximation for O(n log n) repulsive force calculation
 * - Spatial grid/hashing for neighbor lookup optimization
 * - Tier-weighted repulsion (higher tier = less repulsion)
 * - Edge weight-based attraction
 * - Composition-aware clustering: composites attract their atom nodes
 * - Batch state updates to reduce re-renders
 * - Frame budget limiting (skip force calc if frame took too long)
 * - Delta time capping for stability after tab switches
 * - Computed visual properties from nodeRendering.ts
 * - Velocity clamping for stability
 *
 * Runs in a requestAnimationFrame loop and updates node positions in the store.
 */
export function useForceLayout(): void {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);

  const velocitiesRef = useRef<VelocityMap>({});
  const rafRef = useRef<number | null>(null);
  const tickFnRef = useRef<(() => void) | null>(null);
  const prevNodeIdsRef = useRef<Set<number>>(new Set());
  // Track last frame time for budget limiting
  const lastFrameTimeRef = useRef<number>(0);

  const nodeCount = nodes.size;
  const edgeCount = edges.size;

  useEffect(() => {
    tickFnRef.current = () => {
      const frameStart = performance.now();
      const currentNodes = useGraphStore.getState().nodes;
      const currentEdges = useGraphStore.getState().edges;
      const currentBatchUpdatePositions = useGraphStore.getState().batchUpdatePositions;
      const currentUpdateNode = useGraphStore.getState().updateNode;

      const nodeList = Array.from(currentNodes.values());
      const edgeList = Array.from(currentEdges.values());

      if (nodeList.length === 0) {
        rafRef.current = null;
        return;
      }

      const velocities = velocitiesRef.current;
      const prevNodeIds = prevNodeIdsRef.current;
      const currentNodeIds = new Set(nodeList.map((n) => n.id));

      // Detect new nodes and assign initial positions
      const seedNodes = nodeList.filter((n) => n.is_seed);
      const layer1Nodes = nodeList.filter((n) => !n.is_seed && isInternalRepresentation(n));
      const nonSeedNewNodes: number[] = [];

      for (const node of nodeList) {
        if (!velocities[node.id]) {
          // New node — initialize velocity and position
          if (node.is_seed) {
            const seedIndex = seedNodes.indexOf(node);
            const seedPos = computeSeedPosition(seedIndex, seedNodes.length);
            velocities[node.id] = { vx: 0, vy: 0, vz: 0 };

            // Seed nodes start at their sphere position (unless already positioned)
            if (!node.render?.position || (node.render.position.x === 0 && node.render.position.y === 0 && node.render.position.z === 0)) {
              const renderProps = computeNodeRenderProps(node);
              currentUpdateNode(node.id, {
                render: {
                  position: seedPos,
                  size: renderProps.size,
                  color: renderProps.color,
                  glow: renderProps.glow,
                },
              });
            }
          } else {
            // Non-seed nodes: start near a connected seed or at random offset from center
            velocities[node.id] = {
              vx: (Math.random() - 0.5) * 0.3,
              vy: (Math.random() - 0.5) * 0.3,
              vz: (Math.random() - 0.5) * 0.3,
            };

            // v8.0: Layer-aware initial positioning
            const nodeLayer = computeNodeLayer(node);
            const isInternalRepr = isInternalRepresentation(node);

            // v8.0: Layer 1 nodes start on a ring around the seed cluster
            if (isInternalRepr && !node.render?.position) {
              const layer1Index = layer1Nodes.indexOf(node);
              const ringPos = computeLayer1RingPosition(layer1Index, layer1Nodes.length);
              const renderProps = computeNodeRenderProps(node);
              currentUpdateNode(node.id, {
                render: {
                  position: {
                    x: ringPos.x,
                    y: getLayerYOffset(1),
                    z: ringPos.z,
                  },
                  size: renderProps.size,
                  color: renderProps.color,
                  glow: renderProps.glow,
                },
              });
            } else if (nodeLayer >= 2 && !node.render?.position) {
              // v8.0: Layer 2+ nodes start in outer orbit
              const layer2PlusNodes = nodeList.filter(
                (n) => !n.is_seed && !isInternalRepresentation(n) && computeNodeLayer(n) >= 2
              );
              const outerIndex = layer2PlusNodes.indexOf(node);
              const orbitPos = computeLayer2OrbitPosition(outerIndex, layer2PlusNodes.length);
              const renderProps = computeNodeRenderProps(node);
              currentUpdateNode(node.id, {
                render: {
                  position: {
                    x: orbitPos.x,
                    y: getLayerYOffset(nodeLayer),
                    z: orbitPos.z,
                  },
                  size: renderProps.size,
                  color: renderProps.color,
                  glow: renderProps.glow,
                },
              });
            } else {
              const renderProps = computeNodeRenderProps(node);
              const pos = node.render?.position;
              if (!pos || (pos.x === 0 && pos.y === 0 && pos.z === 0)) {
                currentUpdateNode(node.id, {
                  render: {
                    position: {
                      x: (Math.random() - 0.5) * 20,
                      y: (Math.random() - 0.5) * 15,
                      z: (Math.random() - 0.5) * 20,
                    },
                    size: renderProps.size,
                    color: renderProps.color,
                    glow: renderProps.glow,
                  },
                });
              }
            }
          }
          nonSeedNewNodes.push(node.id);
        }
      }

      // Clean up velocities for removed nodes
      for (const id of Object.keys(velocities)) {
        if (!currentNodeIds.has(Number(id))) {
          delete velocities[Number(id)];
        }
      }
      prevNodeIdsRef.current = currentNodeIds;

      // Compute forces
      const forces: { [nodeId: number]: { fx: number; fy: number; fz: number } } = {};
      for (const node of nodeList) {
        forces[node.id] = { fx: 0, fy: 0, fz: 0 };
      }

      // ── Frame budget check ──
      // If previous frame was too slow, skip force calculation this frame
      // (still apply damping to slow things down gradually)
      const elapsedSinceLast = frameStart - lastFrameTimeRef.current;
      const skipForceCalc = elapsedSinceLast > FRAME_BUDGET_MS * 3;

      if (!skipForceCalc) {
        // ── Repulsion: Barnes-Hut approximation for O(n log n) ──
        // Build positions array for octree construction
        const n = nodeList.length;
        const positions: Array<{ x: number; y: number; z: number }> = new Array(n);
        const tierMultipliers = new Float64Array(n);

        for (let i = 0; i < n; i++) {
          const node = nodeList[i];
          positions[i] = node.render?.position ?? { x: 0, y: 0, z: 0 };
          tierMultipliers[i] = node.tier === 1 ? 1.3 : node.tier === 2 ? 1.0 : 0.7;
        }

        // Build octree
        let minCoord = Infinity, maxCoord = -Infinity;
        for (let i = 0; i < n; i++) {
          const p = positions[i];
          minCoord = Math.min(minCoord, p.x, p.y, p.z);
          maxCoord = Math.max(maxCoord, p.x, p.y, p.z);
        }
        const extent = Math.max(maxCoord - minCoord, 1);
        const center = (minCoord + maxCoord) / 2;
        const half = extent / 2 + 1; // Add margin

        const octree = createOctreeNode(center, center, center, half);
        for (let i = 0; i < n; i++) {
          insertIntoOctree(octree, positions[i].x, positions[i].y, positions[i].z, 1.0, i);
        }

        // Compute repulsive forces using Barnes-Hut traversal
        const forceAccum = { fx: 0, fy: 0, fz: 0 };
        for (let i = 0; i < n; i++) {
          forceAccum.fx = 0;
          forceAccum.fy = 0;
          forceAccum.fz = 0;
          computeRepulsiveForce(
            octree,
            positions[i].x, positions[i].y, positions[i].z,
            BARNES_HUT_THETA,
            REPULSION_CONSTANT,
            tierMultipliers[i],
            forceAccum,
            positions,
            tierMultipliers,
          );
          forces[nodeList[i].id].fx += forceAccum.fx;
          forces[nodeList[i].id].fy += forceAccum.fy;
          forces[nodeList[i].id].fz += forceAccum.fz;
        }

        // Attraction along edges (Hooke's law) — weight-based
        // Build spatial grid for edge optimization
        const spatialGrid = buildSpatialGrid(positions, GRID_CELL_SIZE);

        for (const edge of edgeList) {
          if (edge.source === edge.target) continue;
          const a = currentNodes.get(edge.source);
          const b = currentNodes.get(edge.target);
          if (!a || !b) continue;

          const posA = a.render?.position ?? { x: 0, y: 0, z: 0 };
          const posB = b.render?.position ?? { x: 0, y: 0, z: 0 };

          const dx = posB.x - posA.x;
          const dy = posB.y - posA.y;
          const dz = posB.z - posA.z;
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), MIN_DISTANCE);

          const displacement = dist - IDEAL_EDGE_LENGTH;
          // Stronger attraction for higher-weight edges
          const f = ATTRACTION_CONSTANT * displacement * (0.3 + edge.weight * 0.7);
          const fx = (f * dx) / dist;
          const fy = (f * dy) / dist;
          const fz = (f * dz) / dist;

          forces[edge.source].fx += fx;
          forces[edge.source].fy += fy;
          forces[edge.source].fz += fz;
          forces[edge.target].fx -= fx;
          forces[edge.target].fy -= fy;
          forces[edge.target].fz -= fz;
        }

        // ── Composition attraction force ──
        // Composites attract their constituent atoms closer (in addition to edge attraction)
        for (const node of nodeList) {
          if (!isCompositeNode(node)) continue;
          const compositePos = node.render?.position;
          if (!compositePos) continue;

          const atomIds = getAtomIds(node);
          for (const aid of atomIds) {
            const atomNode = currentNodes.get(aid);
            if (!atomNode?.render?.position) continue;

            const atomPos = atomNode.render.position;
            const dx = compositePos.x - atomPos.x;
            const dy = compositePos.y - atomPos.y;
            const dz = compositePos.z - atomPos.z;
            const dist = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), MIN_DISTANCE);

            const displacement = dist - COMPOSITION_IDEAL_DISTANCE;
            const f = COMPOSITION_ATTRACTION_CONSTANT * displacement;
            const fx = (f * dx) / dist;
            const fy = (f * dy) / dist;
            const fz = (f * dz) / dist;

            // Pull atom toward composite
            forces[aid].fx += fx;
            forces[aid].fy += fy;
            forces[aid].fz += fz;
            // Pull composite toward atom (weaker)
            forces[node.id].fx -= fx * 0.3;
            forces[node.id].fy -= fy * 0.3;
            forces[node.id].fz -= fz * 0.3;
          }
        }

        // Center gravity — stronger for seed nodes
        for (const node of nodeList) {
          const pos = node.render?.position ?? { x: 0, y: 0, z: 0 };
          const gravityStrength = node.is_seed ? CENTER_GRAVITY * 1.5 : CENTER_GRAVITY;
          forces[node.id].fx -= pos.x * gravityStrength;
          forces[node.id].fy -= pos.y * gravityStrength;
          forces[node.id].fz -= pos.z * gravityStrength;
        }

        // ── v8.0: Seed immovable constraint ──
        // Seed nodes are locked in their sphere positions — zero out all forces
        for (const node of nodeList) {
          if (node.is_seed) {
            forces[node.id].fx = 0;
            forces[node.id].fy = 0;
            forces[node.id].fz = 0;
          }
        }

        // ── v8.0: Layer 1 ring gravity ──
        // Internal representation nodes are gently pulled toward a ring
        // at LAYER1_RING_RADIUS around the Y axis at their layer height
        for (const node of nodeList) {
          if (node.is_seed) continue;
          const isInternalRepr = isInternalRepresentation(node);
          const layer = computeNodeLayer(node);
          const pos = node.render?.position ?? { x: 0, y: 0, z: 0 };

          if (isInternalRepr) {
            // Pull toward ring at radius LAYER1_RING_RADIUS on the XZ plane
            const distFromCenter = Math.sqrt(pos.x * pos.x + pos.z * pos.z);
            if (distFromCenter > 0.1) {
              const targetX = (pos.x / distFromCenter) * LAYER1_RING_RADIUS;
              const targetZ = (pos.z / distFromCenter) * LAYER1_RING_RADIUS;
              const ringForce = 0.005;
              forces[node.id].fx += (targetX - pos.x) * ringForce;
              forces[node.id].fz += (targetZ - pos.z) * ringForce;
            } else {
              // Node at center — push out to ring
              const angle = Math.random() * Math.PI * 2;
              forces[node.id].fx += Math.cos(angle) * LAYER1_RING_RADIUS * 0.01;
              forces[node.id].fz += Math.sin(angle) * LAYER1_RING_RADIUS * 0.01;
            }
          } else if (layer >= 2) {
            // Layer 2+ nodes: gently pushed outward from center
            const distFromCenter = Math.sqrt(pos.x * pos.x + pos.z * pos.z);
            if (distFromCenter < LAYER2_ORBIT_RADIUS - 3) {
              // Too close to center — push outward
              const pushForce = 0.002;
              if (distFromCenter > 0.1) {
                forces[node.id].fx += (pos.x / distFromCenter) * pushForce * LAYER2_ORBIT_RADIUS;
                forces[node.id].fz += (pos.z / distFromCenter) * pushForce * LAYER2_ORBIT_RADIUS;
              }
            }
          }
        }

        // ── v5.0: Layer Y-gravity ──
        // Gently pull nodes toward their layer's Y-offset
        for (const node of nodeList) {
          if (node.is_seed) continue; // Seeds are already positioned
          const layer = computeNodeLayer(node);
          const targetY = getLayerYOffset(layer);
          const pos = node.render?.position ?? { x: 0, y: 0, z: 0 };
          const yOffset = targetY - pos.y;
          forces[node.id].fy += yOffset * LAYER_Y_GRAVITY;
        }
      }

      // Update velocities with damping and apply forces
      let totalKineticEnergy = 0;
      const positionUpdates = new Map<number, { x: number; y: number; z: number }>();

      for (const node of nodeList) {
        const vel = velocities[node.id];
        if (!vel) continue;

        // v8.0: Seed nodes are immovable — skip velocity/position update
        if (node.is_seed) {
          // Still update render props in case layer colors changed
          const renderProps = computeNodeRenderProps(node);
          const existingPos = node.render?.position ?? { x: 0, y: 0, z: 0 };
          currentUpdateNode(node.id, {
            render: {
              position: existingPos,
              size: renderProps.size,
              color: renderProps.color,
              glow: renderProps.glow,
            },
          });
          continue;
        }

        vel.vx = (vel.vx + forces[node.id].fx) * DAMPING;
        vel.vy = (vel.vy + forces[node.id].fy) * DAMPING;
        vel.vz = (vel.vz + forces[node.id].fz) * DAMPING;

        // Clamp velocity for stability
        const speed = Math.sqrt(vel.vx * vel.vx + vel.vy * vel.vy + vel.vz * vel.vz);
        if (speed > MAX_VELOCITY) {
          const scale = MAX_VELOCITY / speed;
          vel.vx *= scale;
          vel.vy *= scale;
          vel.vz *= scale;
        }

        totalKineticEnergy += vel.vx * vel.vx + vel.vy * vel.vy + vel.vz * vel.vz;

        const pos = node.render?.position ?? { x: 0, y: 0, z: 0 };
        const newPos = {
          x: pos.x + vel.vx,
          y: pos.y + vel.vy,
          z: pos.z + vel.vz,
        };

        // Collect position for batch update
        positionUpdates.set(node.id, newPos);
      }

      // Apply all position updates in a single Zustand batch
      if (positionUpdates.size > 0) {
        currentBatchUpdatePositions(positionUpdates);
      }

      // Update render props (size, color, glow) for non-seed nodes that moved
      // Only update render props every other frame to reduce re-renders
      const shouldUpdateRenderProps = !skipForceCalc;
      if (shouldUpdateRenderProps) {
        for (const node of nodeList) {
          if (node.is_seed) continue;
          const newPos = positionUpdates.get(node.id);
          if (!newPos) continue;

          // Compute render props from nodeRendering utility
          const renderProps = computeNodeRenderProps(node);
          currentUpdateNode(node.id, {
            render: {
              ...node.render,
              position: newPos,
              size: renderProps.size,
              color: renderProps.color,
              glow: renderProps.glow,
            },
          });
        }
      }

      // Record frame time for budget check
      lastFrameTimeRef.current = performance.now();

      // Stop when energy is low
      if (totalKineticEnergy < ENERGY_THRESHOLD) {
        rafRef.current = null;
        return;
      }

      rafRef.current = requestAnimationFrame(tickFnRef.current!);
    };
  }, []);

  // Start/restart simulation when graph changes
  useEffect(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (nodeCount > 0 && tickFnRef.current) {
      rafRef.current = requestAnimationFrame(tickFnRef.current);
    }

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [nodeCount, edgeCount]);
}
