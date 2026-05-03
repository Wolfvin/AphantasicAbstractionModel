'use client';

import { useEffect, useRef } from 'react';
import { useGraphStore } from '@/store/rsvsStore';
import { computeNodeRenderProps, computeEdgeRenderProps, isCompositeNode, isAtomNode, getAtomCount } from '@/lib/nodeRendering';

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

// Composition clustering: extra attraction between composites and their atoms
const COMPOSITION_ATTRACTION_CONSTANT = 0.025;
const COMPOSITION_IDEAL_DISTANCE = 4;

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
 * Get atom IDs for a node (from any of the composition fields).
 */
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

            // Try to position near a connected seed node
            let placedNearSeed = false;
            for (const edge of edgeList) {
              const connectedId = edge.source === node.id ? edge.target : edge.target === node.id ? edge.source : null;
              if (connectedId !== null) {
                const connectedNode = currentNodes.get(connectedId);
                if (connectedNode?.is_seed && connectedNode.render?.position) {
                  const renderProps = computeNodeRenderProps(node);
                  const offset = {
                    x: (Math.random() - 0.5) * 4,
                    y: (Math.random() - 0.5) * 4,
                    z: (Math.random() - 0.5) * 4,
                  };
                  currentUpdateNode(node.id, {
                    render: {
                      position: {
                        x: connectedNode.render.position.x + offset.x,
                        y: connectedNode.render.position.y + offset.y,
                        z: connectedNode.render.position.z + offset.z,
                      },
                      size: renderProps.size,
                      color: renderProps.color,
                      glow: renderProps.glow,
                    },
                  });
                  placedNearSeed = true;
                  break;
                }
              }
            }

            // Composition-aware: try to position new composites near their atom nodes
            if (!placedNearSeed && isCompositeNode(node)) {
              const atomIds = getAtomIds(node);
              let avgX = 0, avgY = 0, avgZ = 0;
              let count = 0;
              for (const aid of atomIds) {
                const atomNode = currentNodes.get(aid);
                if (atomNode?.render?.position) {
                  avgX += atomNode.render.position.x;
                  avgY += atomNode.render.position.y;
                  avgZ += atomNode.render.position.z;
                  count++;
                }
              }
              if (count > 0) {
                const renderProps = computeNodeRenderProps(node);
                currentUpdateNode(node.id, {
                  render: {
                    position: {
                      x: avgX / count + (Math.random() - 0.5) * 2,
                      y: avgY / count + (Math.random() - 0.5) * 2,
                      z: avgZ / count + (Math.random() - 0.5) * 2,
                    },
                    size: renderProps.size,
                    color: renderProps.color,
                    glow: renderProps.glow,
                  },
                });
                placedNearSeed = true;
              }
            }

            // Composition-aware: try to position atom nodes near their composite
            if (!placedNearSeed && isAtomNode(node)) {
              // Find composites that reference this node as an atom
              for (const otherNode of nodeList) {
                if (isCompositeNode(otherNode) && otherNode.render?.position) {
                  const otherAtomIds = getAtomIds(otherNode);
                  if (otherAtomIds.includes(node.id)) {
                    const renderProps = computeNodeRenderProps(node);
                    const offset = {
                      x: (Math.random() - 0.5) * 3,
                      y: (Math.random() - 0.5) * 3,
                      z: (Math.random() - 0.5) * 3,
                    };
                    currentUpdateNode(node.id, {
                      render: {
                        position: {
                          x: otherNode.render.position.x + offset.x,
                          y: otherNode.render.position.y + offset.y,
                          z: otherNode.render.position.z + offset.z,
                        },
                        size: renderProps.size,
                        color: renderProps.color,
                        glow: renderProps.glow,
                      },
                    });
                    placedNearSeed = true;
                    break;
                  }
                }
              }
            }

            // Fallback: random position further from center
            if (!placedNearSeed) {
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

      // Update velocities with damping and apply forces
      let totalKineticEnergy = 0;

      for (const node of nodeList) {
        const vel = velocities[node.id];
        if (!vel) continue;

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
