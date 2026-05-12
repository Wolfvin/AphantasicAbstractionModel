/* eslint-disable react-hooks/refs */
'use client';

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import { Line2, LineMaterial } from 'three-stdlib';
import type { RSVSEdge, EdgeType } from '@/lib/types';
import { useGraphStore } from '@/store/aamStore';
import { isCompositeNode, isAtomNode } from '@/lib/nodeRendering';
import { lerp } from '@/lib/constants';

// ── Constants ──
const LERP_FACTOR = 0.08;
const SPAWN_DURATION_MS = 600;
const ARROW_SIZE = 0.25;
const ARROW_CONE_RADIUS = 0.12;
const ARROW_CONE_HEIGHT = 0.35;

// Edge type colors
const COMPOSITION_EDGE_COLOR = '#00BCD4';   // Cyan for composition
const CONVERGENCE_EDGE_COLOR = '#E040FB';    // Purple for convergence
const SUBSTITUTION_EDGE_COLOR = '#FFB74D';   // Orange for substitution
const REGULAR_EDGE_COLOR = '#1a3a5c';

// ── Curved line helper for composition references ──
function computeCurvedPoints(
  start: THREE.Vector3,
  end: THREE.Vector3,
  curvature: number = 0.3,
  segments: number = 20,
): THREE.Vector3[] {
  const mid = new THREE.Vector3().lerpVectors(start, end, 0.5);
  // Offset the midpoint perpendicular to the line
  const direction = new THREE.Vector3().subVectors(end, start).normalize();
  const up = new THREE.Vector3(0, 1, 0);
  const perpendicular = new THREE.Vector3().crossVectors(direction, up).normalize();
  // If parallel to up, use a different reference
  if (perpendicular.lengthSq() < 0.01) {
    perpendicular.crossVectors(direction, new THREE.Vector3(1, 0, 0)).normalize();
  }
  mid.add(perpendicular.multiplyScalar(curvature * start.distanceTo(end)));

  const points: THREE.Vector3[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    // Quadratic Bezier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
    const p = new THREE.Vector3();
    p.x = (1 - t) * (1 - t) * start.x + 2 * (1 - t) * t * mid.x + t * t * end.x;
    p.y = (1 - t) * (1 - t) * start.y + 2 * (1 - t) * t * mid.y + t * t * end.y;
    p.z = (1 - t) * (1 - t) * start.z + 2 * (1 - t) * t * mid.z + t * t * end.z;
    points.push(p);
  }
  return points;
}

interface GraphEdgeProps {
  edge: RSVSEdge;
  sourcePos: THREE.Vector3;
  targetPos: THREE.Vector3;
  isCompositionEdge?: boolean;
}

/**
 * A single edge rendered as a Line with optional arrow for directed edges.
 * Supports multiple edge types:
 * - Regular: standard straight line
 * - Composition: curved arrow (atom → composite reference)
 * - Convergence: dashed purple line (structural equivalence)
 * - Substitution: orange dotted line
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

  // Determine edge type — use the edge_type field, fall back to isCompositionEdge prop
  const edgeType: EdgeType = edge.edge_type ?? (isCompositionEdge ? 'composition' : 'regular');
  const isConvergenceEdge = edgeType === 'convergence';
  const isSubstitutionEdge = edgeType === 'substitution';
  const isCompositionType = edgeType === 'composition';

  // Derived render properties based on edge type
  const edgeColor = isConvergenceEdge
    ? CONVERGENCE_EDGE_COLOR
    : isSubstitutionEdge
      ? SUBSTITUTION_EDGE_COLOR
      : isCompositionType
        ? COMPOSITION_EDGE_COLOR
        : (edge.render?.color ?? REGULAR_EDGE_COLOR);

  const edgeThickness = isConvergenceEdge || isSubstitutionEdge
    ? Math.max(0.4, (edge.render?.thickness ?? 1) * 0.7)
    : isCompositionType
      ? Math.max(0.3, (edge.render?.thickness ?? 1) * 0.6)
      : (edge.render?.thickness ?? 1);

  const baseOpacity = isConvergenceEdge
    ? Math.max(0.2, (edge.render?.opacity ?? 0.5) * 0.6)
    : isSubstitutionEdge
      ? Math.max(0.2, (edge.render?.opacity ?? 0.5) * 0.6)
      : isCompositionType
        ? Math.max(0.15, (edge.render?.opacity ?? 0.6) * 0.5)
        : (edge.render?.opacity ?? 0.6);

  const pulseAmount = isCompositionType
    ? Math.max(edge.render?.pulse ?? 0, 0.3)
    : isConvergenceEdge
      ? Math.max(edge.render?.pulse ?? 0, 0.4)
      : (edge.render?.pulse ?? 0);

  const edgeStatus = edge.status ?? 'stable';
  const isDirected = edge.direction === 'directed';

  // Compute curved points for composition edges
  const curvedPoints = useMemo(() => {
    if (isCompositionType) {
      return computeCurvedPoints(sourcePos, targetPos, 0.25);
    }
    return null;
  }, [sourcePos, targetPos, isCompositionType]);

  // Line points — curved for composition, straight for others
  const linePoints = useMemo(() => {
    if (isCompositionType && curvedPoints) {
      return curvedPoints;
    }
    return [sourcePos, targetPos];
  }, [sourcePos, targetPos, isCompositionType, curvedPoints]);

  // Compute midpoint and direction for arrow
  const direction = useMemo(() => {
    const dir = new THREE.Vector3().subVectors(targetPos, sourcePos);
    const len = dir.length();
    if (len < 0.001) return new THREE.Vector3(1, 0, 0);
    return dir.normalize();
  }, [sourcePos, targetPos]);

  const arrowPosition = useMemo(() => {
    if (isCompositionType && curvedPoints) {
      // Place arrow at ~80% along the curve
      const idx = Math.floor(curvedPoints.length * 0.8);
      return curvedPoints[Math.min(idx, curvedPoints.length - 1)].clone();
    }
    // Place arrow at ~80% along the edge
    return new THREE.Vector3().lerpVectors(sourcePos, targetPos, 0.82);
  }, [sourcePos, targetPos, isCompositionType, curvedPoints]);

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

    // Pulse animation — type-specific speeds
    if (pulseAmount > 0) {
      const pulseSpeed = isConvergenceEdge ? 4 : isCompositionType ? 3 : 2;
      const pulseStrength = isConvergenceEdge ? 0.2 : isCompositionType ? 0.15 : 0.05;
      const pulse = pulseStrength * pulseAmount * Math.sin(time * pulseSpeed * Math.PI);
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

    // ── Update composition/convergence dash line ──
    if (dashLineRef.current && (isCompositionType || isConvergenceEdge || isSubstitutionEdge)) {
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
      // Show arrow for composition and regular directed edges, not for convergence
      const shouldShowArrow = !isConvergenceEdge && !isSubstitutionEdge;
      const arrowOpacity = shouldShowArrow
        ? Math.max(0, Math.min(1, currentOpacityRef.current * 0.8))
        : 0;
      arrowMaterialRef.current.opacity = arrowOpacity;
      coneRef.current.visible = shouldShowArrow && (spawnComplete || drawProgressRef.current > 0.7);
    }
  });

  return (
    <group>
      {/* Main edge line — curved for composition, straight for others */}
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

      {/* Convergence: dashed overlay for structural equivalence */}
      {isConvergenceEdge && (
        <Line
          ref={dashLineRef}
          points={[sourcePos, targetPos]}
          color={edgeColor}
          lineWidth={edgeThickness * 1.2}
          transparent
          opacity={baseOpacity * 0.6}
          dashed
          dashSize={0.6}
          gapSize={0.35}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      )}

      {/* Substitution: dotted overlay */}
      {isSubstitutionEdge && (
        <Line
          ref={dashLineRef}
          points={[sourcePos, targetPos]}
          color={edgeColor}
          lineWidth={edgeThickness * 0.8}
          transparent
          opacity={baseOpacity * 0.6}
          dashed
          dashSize={0.3}
          gapSize={0.2}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      )}

      {/* Composition: dashed overlay for visual distinction */}
      {isCompositionType && (
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

      {/* Arrow for directed edges — shown for regular and composition, hidden for convergence/substitution */}
      {isDirected && !isConvergenceEdge && !isSubstitutionEdge && (
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
        lineWidth={edgeThickness * (isCompositionType ? 3 : isConvergenceEdge ? 3.5 : 2.5)}
        transparent
        opacity={baseOpacity * (isCompositionType ? 0.08 : isConvergenceEdge ? 0.1 : 0.12)}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </group>
  );
};

export const GraphEdge = React.memo(GraphEdgeComponent);
