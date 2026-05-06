'use client';

import { useEffect, useRef } from 'react';
import { useGraphStore } from '@/store/rsvsStore';
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
 * - Tier-weighted repulsion (higher tier = less repulsion)
 * - Edge weight-based attraction
 * - Composition-aware clustering: composites attract their atom nodes
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

  const nodeCount = nodes.size;
  const edgeCount = edges.size;

  useEffect(() => {
    tickFnRef.current = () => {
      const currentNodes = useGraphStore.getState().nodes;
      const currentEdges = useGraphStore.getState().edges;
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

      // Repulsion (Coulomb's law) — tier-weighted: lower tier nodes repel more
      for (let i = 0; i < nodeList.length; i++) {
        const a = nodeList[i];
        const posA = a.render?.position ?? { x: 0, y: 0, z: 0 };

        for (let j = i + 1; j < nodeList.length; j++) {
          const b = nodeList[j];
          const posB = b.render?.position ?? { x: 0, y: 0, z: 0 };

          const dx = posA.x - posB.x;
          const dy = posA.y - posB.y;
          const dz = posA.z - posB.z;
          const distSq = Math.max(dx * dx + dy * dy + dz * dz, MIN_DISTANCE * MIN_DISTANCE);
          const dist = Math.sqrt(distSq);

          // Tier weight: Tier1 nodes are more "solid" and repel more
          const tierMultiplierA = a.tier === 1 ? 1.3 : a.tier === 2 ? 1.0 : 0.7;
          const tierMultiplierB = b.tier === 1 ? 1.3 : b.tier === 2 ? 1.0 : 0.7;
          const force = REPULSION_CONSTANT * tierMultiplierA * tierMultiplierB / distSq;
          const fx = (force * dx) / dist;
          const fy = (force * dy) / dist;
          const fz = (force * dz) / dist;

          forces[a.id].fx += fx;
          forces[a.id].fy += fy;
          forces[a.id].fz += fz;
          forces[b.id].fx -= fx;
          forces[b.id].fy -= fy;
          forces[b.id].fz -= fz;
        }
      }

      // Attraction along edges (Hooke's law) — weight-based
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
      // Layer 0 at bottom (y=0), higher layers stacked up
      for (const node of nodeList) {
        if (node.is_seed) continue; // Seeds are already positioned
        const layer = computeNodeLayer(node);
        const targetY = getLayerYOffset(layer);
        const pos = node.render?.position ?? { x: 0, y: 0, z: 0 };
        const yOffset = targetY - pos.y;
        forces[node.id].fy += yOffset * LAYER_Y_GRAVITY;
      }

      // Update velocities with damping and apply forces
      let totalKineticEnergy = 0;

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

        // Compute render props from nodeRendering utility
        const renderProps = computeNodeRenderProps(node);
        const edgeRenderProps = computeEdgeRenderProps(node.confidence);

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
