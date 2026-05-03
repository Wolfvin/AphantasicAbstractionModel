'use client';

import { useEffect, useRef } from 'react';
import { useGraphStore } from '@/store/rsvsStore';

// ── Simulation Constants ──
const REPULSION_CONSTANT = 500;
const ATTRACTION_CONSTANT = 0.01;
const CENTER_GRAVITY = 0.005;
const DAMPING = 0.9;
const MIN_DISTANCE = 1;
const ENERGY_THRESHOLD = 0.01;
const IDEAL_EDGE_LENGTH = 6;

interface VelocityMap {
  [nodeId: number]: { vx: number; vy: number; vz: number };
}

/**
 * Simple n-body force-directed layout simulation.
 * Runs in a requestAnimationFrame loop and updates node positions in the store.
 */
export function useForceLayout(): void {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);

  const velocitiesRef = useRef<VelocityMap>({});
  const rafRef = useRef<number | null>(null);
  const tickFnRef = useRef<(() => void) | null>(null);

  const nodeCount = nodes.size;
  const edgeCount = edges.size;

  useEffect(() => {
    // Build tick function
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

      // Initialize velocities for new nodes
      for (const node of nodeList) {
        if (!velocities[node.id]) {
          velocities[node.id] = {
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            vz: (Math.random() - 0.5) * 0.5,
          };
        }
      }

      // Clean up velocities for removed nodes
      for (const id of Object.keys(velocities)) {
        if (!currentNodes.has(Number(id))) {
          delete velocities[Number(id)];
        }
      }

      // For each pair, compute repulsion (Coulomb's law)
      const forces: { [nodeId: number]: { fx: number; fy: number; fz: number } } = {};
      for (const node of nodeList) {
        forces[node.id] = { fx: 0, fy: 0, fz: 0 };
      }

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

          const force = REPULSION_CONSTANT / distSq;
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

      // Attraction along edges (Hooke's law)
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
        const f = ATTRACTION_CONSTANT * displacement * (0.5 + edge.weight * 0.5);
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

      // Center gravity
      for (const node of nodeList) {
        const pos = node.render?.position ?? { x: 0, y: 0, z: 0 };
        forces[node.id].fx -= pos.x * CENTER_GRAVITY;
        forces[node.id].fy -= pos.y * CENTER_GRAVITY;
        forces[node.id].fz -= pos.z * CENTER_GRAVITY;
      }

      // Update velocities with damping and apply forces
      let totalKineticEnergy = 0;

      for (const node of nodeList) {
        const vel = velocities[node.id];
        if (!vel) continue;

        vel.vx = (vel.vx + forces[node.id].fx) * DAMPING;
        vel.vy = (vel.vy + forces[node.id].fy) * DAMPING;
        vel.vz = (vel.vz + forces[node.id].fz) * DAMPING;

        totalKineticEnergy += vel.vx * vel.vx + vel.vy * vel.vy + vel.vz * vel.vz;

        const pos = node.render?.position ?? { x: 0, y: 0, z: 0 };
        const newPos = {
          x: pos.x + vel.vx,
          y: pos.y + vel.vy,
          z: pos.z + vel.vz,
        };

        currentUpdateNode(node.id, {
          render: {
            ...node.render,
            position: newPos,
            size: node.render?.size ?? 1,
            color: node.render?.color ?? '#00E5FF',
            glow: node.render?.glow ?? 0,
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
