/* eslint-disable react-hooks/refs */
'use client';

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import { Line2, LineMaterial } from 'three-stdlib';
import type { RSVSEdge } from '@/lib/types';
import { useGraphStore } from '@/store/rsvsStore';
import { isCompositeNode, isAtomNode } from '@/lib/nodeRendering';

// ── Constants ──
const LERP_FACTOR = 0.08;
const SPAWN_DURATION_MS = 600;
const ARROW_SIZE = 0.25;
const ARROW_CONE_RADIUS = 0.12;
const ARROW_CONE_HEIGHT = 0.35;

// Composition edge color — cyan/blue for composition edges
const COMPOSITION_EDGE_COLOR = '#00BCD4';
// Regular edge color fallback
const REGULAR_EDGE_COLOR = '#1a3a5c';

interface GraphEdgeProps {
  edge: RSVSEdge;
  sourcePos: THREE.Vector3;
  targetPos: THREE.Vector3;
  isCompositionEdge?: boolean;
}

/**
 * A single edge rendered as a Line with optional arrow for directed edges.
 * Composition edges (atom → composite) render differently: thinner, cyan, pulsing, dashed-style.
 */
const GraphEdgeComponent: React.FC<GraphEdgeProps> = ({
  edge,
  sourcePos,
  targetPos,
  isCompositionEdge = false,
}) => {
  const lineRef = useRef<Line2>(null);
  const coneRef = useRef<THREE.Mesh>(null);
  const dashLineRef = useRef<Line2>(null);

  // Spawn animation tracking
  const spawnStartRef = useRef<number | null>(null);
  const [spawnComplete, setSpawnComplete] = useState(false);
  const drawProgressRef = useRef(0);

  // Brightness flash for "updated" edges
  const flashRef = useRef(0);

  // Current animated opacity
  const currentOpacityRef = useRef(0);

  // Derived render properties — composition edges get their own color
  const edgeColor = isCompositionEdge
    ? COMPOSITION_EDGE_COLOR
    : (edge.render?.color ?? REGULAR_EDGE_COLOR);
  const edgeThickness = isCompositionEdge
    ? Math.max(0.3, (edge.render?.thickness ?? 1) * 0.6)
    : (edge.render?.thickness ?? 1);
  const baseOpacity = isCompositionEdge
    ? Math.max(0.15, (edge.render?.opacity ?? 0.6) * 0.5)
    : (edge.render?.opacity ?? 0.6);
  const pulseAmount = isCompositionEdge
    ? Math.max(edge.render?.pulse ?? 0, 0.3)
    : (edge.render?.pulse ?? 0);
  const edgeStatus = edge.status ?? 'stable';
  const isDirected = edge.direction === 'directed';

  // Compute midpoint and direction for arrow
  const direction = useMemo(() => {
    const dir = new THREE.Vector3().subVectors(targetPos, sourcePos);
    const len = dir.length();
    if (len < 0.001) return new THREE.Vector3(1, 0, 0);
    return dir.normalize();
  }, [sourcePos, targetPos]);

  const arrowPosition = useMemo(() => {
    // Place arrow at ~80% along the edge
    return new THREE.Vector3().lerpVectors(sourcePos, targetPos, 0.82);
  }, [sourcePos, targetPos]);

  // Arrow rotation: look along direction
  const arrowQuaternion = useMemo(() => {
    const quat = new THREE.Quaternion();
    quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    return quat;
  }, [direction]);

  // Cone geometry for arrow (shared)
  const coneGeometry = useMemo(
    () => new THREE.ConeGeometry(ARROW_CONE_RADIUS, ARROW_CONE_HEIGHT, 8),
    [],
  );

  // Arrow material ref (mutable for useFrame)
  const arrowMaterialRef = useRef<THREE.MeshBasicMaterial | null>(null);

  useEffect(() => {
    if (arrowMaterialRef.current == null) {
      arrowMaterialRef.current = new THREE.MeshBasicMaterial({
        color: new THREE.Color(edgeColor),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
    }
  }, [edgeColor]);

  // Track previous weight for flash detection
  const prevWeightRef = useRef(edge.weight);

  // Per-frame animation
  useFrame((state) => {
    const time = state.clock.elapsedTime;

    // ── Spawn animation ──
    if (!spawnComplete) {
      if (spawnStartRef.current === null) {
        spawnStartRef.current = time;
      }
      const elapsed = time - spawnStartRef.current;
      const progress = Math.min(elapsed / (SPAWN_DURATION_MS / 1000), 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      drawProgressRef.current = eased;

      if (progress >= 1) {
        setSpawnComplete(true);
        drawProgressRef.current = 1;
      }
    }

    // ── Weight change flash ──
    if (edge.weight !== prevWeightRef.current) {
      flashRef.current = 1;
      prevWeightRef.current = edge.weight;
    }

    // Decay flash
    if (flashRef.current > 0) {
      flashRef.current = Math.max(0, flashRef.current - 0.02);
    }

    // ── Compute target opacity ──
    let targetOpacity = baseOpacity;

    // Pulse animation — more pronounced for composition edges
    if (pulseAmount > 0) {
      const pulseStrength = isCompositionEdge ? 0.15 : 0.05;
      const pulse = pulseStrength * pulseAmount * Math.sin(time * (isCompositionEdge ? 3 : 2) * Math.PI);
      targetOpacity += pulse;
    }

    // Brightness flash for weight changes
    if (flashRef.current > 0) {
      targetOpacity = Math.min(1, targetOpacity + flashRef.current * 0.4);
    }

    // New edges: fade in
    if (edgeStatus === 'new' && !spawnComplete) {
      targetOpacity *= drawProgressRef.current;
    }

    // Removing edges: fade out
    if (edgeStatus === 'removing') {
      targetOpacity *= 0.3;
    }

    // Lerp opacity
    currentOpacityRef.current = lerp(currentOpacityRef.current, targetOpacity, LERP_FACTOR);

    // ── Update line material ──
    if (lineRef.current) {
      const mat = lineRef.current.material as LineMaterial;
      if (mat) {
        mat.opacity = Math.max(0, Math.min(1, currentOpacityRef.current));
        mat.transparent = true;
      }
    }

    // ── Update composition dash line ──
    if (dashLineRef.current && isCompositionEdge) {
      const dashMat = dashLineRef.current.material as LineMaterial;
      if (dashMat) {
        dashMat.opacity = Math.max(0, Math.min(1, currentOpacityRef.current * 0.6));
        dashMat.transparent = true;
        // Update dash offset for animation
        dashMat.dashOffset = -(time * 0.5);
      }
    }

    // ── Update arrow ──
    if (coneRef.current && isDirected && arrowMaterialRef.current) {
      const arrowOpacity = Math.max(0, Math.min(1, currentOpacityRef.current * 0.8));
      arrowMaterialRef.current.opacity = arrowOpacity;
      coneRef.current.visible = spawnComplete || drawProgressRef.current > 0.7;
    }
  });

  // Compute animated points for spawn (draw in from source)
  const linePoints = useMemo(() => {
    if (edgeStatus === 'new' && !spawnComplete) {
      // Will be dynamically updated in useFrame via the drawProgress ref
    }
    return [sourcePos, targetPos];
  }, [sourcePos, targetPos, edgeStatus]);

  return (
    <group>
      {/* Main edge line */}
      <Line
        ref={lineRef}
        points={linePoints}
        color={edgeColor}
        lineWidth={edgeThickness}
        transparent
        opacity={baseOpacity}
        // Line2 / LineMaterial props
        dashed={false}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />

      {/* Composition edge: dashed overlay for visual distinction */}
      {isCompositionEdge && (
        <Line
          ref={dashLineRef}
          points={linePoints}
          color={edgeColor}
          lineWidth={edgeThickness * 1.5}
          transparent
          opacity={baseOpacity * 0.6}
          dashed
          dashSize={0.5}
          gapSize={0.3}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      )}

      {/* Arrow for directed edges — hidden for composition edges */}
      {isDirected && !isCompositionEdge && (
        <mesh
          ref={coneRef}
          geometry={coneGeometry}
          material={arrowMaterialRef.current ?? undefined}
          position={arrowPosition}
          quaternion={arrowQuaternion}
        />
      )}

      {/* Inner glow line (slightly thicker, more transparent, for luminance) */}
      <Line
        points={linePoints}
        color={edgeColor}
        lineWidth={edgeThickness * (isCompositionEdge ? 3 : 2.5)}
        transparent
        opacity={baseOpacity * (isCompositionEdge ? 0.08 : 0.12)}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </group>
  );
};

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export const GraphEdge = React.memo(GraphEdgeComponent);
