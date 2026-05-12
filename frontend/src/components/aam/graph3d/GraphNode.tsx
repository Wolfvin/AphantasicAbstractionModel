/* eslint-disable react-hooks/immutability */
'use client';

import React, { useRef, useMemo, useState, useCallback } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import type { RSVSNode } from '@/lib/types';
import { useGraphStore, useUIStore } from '@/store/aamStore';
import { isCompositeNode, isAtomNode, getAtomCount, computeNodeLayer, getLayerColor, buildCompositionChain, isInternalRepresentation, hasConvergenceLinks, getConvergenceTargets } from '@/lib/nodeRendering';
import { lerp } from '@/lib/constants';

// ── Animation constants ──
const SPAWN_DURATION_MS = 500;
const PULSE_SPEED = 2.5;
const DECAY_FLICKER_SPEED = 8;
const LERP_FACTOR = 0.1;

interface GraphNodeProps {
  node: RSVSNode;
  isSelected: boolean;
  isHovered: boolean;
  onClick: (nodeId: number) => void;
  onHover: (nodeId: number | null) => void;
  onDoubleClick: (nodeId: number) => void;
}

const GraphNodeComponent: React.FC<GraphNodeProps> = ({
  node,
  isSelected,
  isHovered,
  onClick,
  onHover,
  onDoubleClick,
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const glowMeshRef = useRef<THREE.Mesh>(null);
  const torusRef = useRef<THREE.Mesh>(null);
  const compositeHaloRef = useRef<THREE.Mesh>(null);

  // Derived render properties
  const nodeColor = node.render?.color ?? '#00E5FF';
  const nodeSize = node.render?.size ?? 1;
  const nodeGlow = node.render?.glow ?? 0;
  const nodeStatus = node.status ?? 'stable';
  const pos = node.render?.position ?? { x: 0, y: 0, z: 0 };

  // Composition detection
  const composite = isCompositeNode(node);
  const atom = isAtomNode(node);
  const atomCount = getAtomCount(node);

  // v8.0: Internal representation and convergence detection
  const internalRepr = isInternalRepresentation(node);
  const hasConvergence = hasConvergenceLinks(node);
  const convergenceTargets = getConvergenceTargets(node);

  // v5.0: Compute effective layer
  const layer = computeNodeLayer(node);
  const layerColor = getLayerColor(layer);

  // v5.0: Build composition chain string for hover tooltip
  const compositionChain = useMemo(() => {
    return buildCompositionChain(
      node.label,
      node.compositions,
      node.derived_from_node_ids ?? node.semantic?.derived_from_node_ids,
      useGraphStore.getState().nodes as Map<number, { label: string }>,
    );
  }, [node.label, node.compositions, node.derived_from_node_ids, node.semantic?.derived_from_node_ids]);

  // Get atom IDs for tendril lines
  const atomIds = useMemo(() => {
    if (!composite) return [];
    if (node.atoms && node.atoms.length > 0) return node.atoms;
    if (node.derived_from_node_ids && node.derived_from_node_ids.length > 0) return node.derived_from_node_ids;
    if (node.semantic?.derived_from_node_ids && node.semantic.derived_from_node_ids.length > 0) return node.semantic.derived_from_node_ids;
    if (node.composition?.atoms) return node.composition.atoms.map(a => a.atom_id);
    return [];
  }, [composite, node.atoms, node.derived_from_node_ids, node.semantic?.derived_from_node_ids, node.composition?.atoms]);

  // Materials created with useMemo (mutable for useFrame, but created once)
  const baseMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(nodeColor),
        emissive: new THREE.Color(nodeColor),
        emissiveIntensity: 0.3,
        metalness: 0.4,
        roughness: 0.3,
        transparent: true,
        opacity: 0.95,
      }),
    [nodeColor],
  );

  const glowMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(nodeColor),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [nodeColor],
  );

  const selectionRingMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: new THREE.Color('#ffffff'),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const compositeHaloMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        // v8.0: Internal repr gets cyan, composites get pink
        color: new THREE.Color(internalRepr ? '#00BCD4' : composite ? '#FF80AB' : '#00E5FF'),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    [composite, internalRepr],
  );

  // Spawn animation tracking
  const spawnStartRef = useRef<number | null>(null);
  const [spawnComplete, setSpawnComplete] = useState(false);

  // Smooth scale tracking
  const targetScaleRef = useRef(1);
  const currentScaleRef = useRef(0.01);

  // Shared geometry (memoized)
  const sphereGeometry = useMemo(
    () => new THREE.SphereGeometry(atom ? 0.7 : 1, 16, 16),
    [atom],
  );

  const glowGeometry = useMemo(
    () => new THREE.SphereGeometry(atom ? 0.7 : 1, 16, 16),
    [atom],
  );

  const torusGeometry = useMemo(
    () => new THREE.TorusGeometry(1, 0.06, 8, 32),
    [],
  );

  // Composite halo geometry — torus for the ring effect
  const haloGeometry = useMemo(
    () => new THREE.TorusGeometry(1.5, 0.04, 8, 48),
    [],
  );

  // View mode for label visibility
  const viewMode = useUIStore((s) => s.viewMode);

  // Event handlers
  const handleClick = useCallback(
    (e: THREE.Event) => {
      (e as { stopPropagation?: () => void }).stopPropagation?.();
      onClick(node.id);
    },
    [onClick, node.id],
  );

  const handlePointerOver = useCallback(
    (e: THREE.Event) => {
      (e as { stopPropagation?: () => void }).stopPropagation?.();
      onHover(node.id);
      if (meshRef.current) {
        document.body.style.cursor = 'pointer';
      }
    },
    [onHover, node.id],
  );

  const handlePointerOut = useCallback(() => {
    onHover(null);
    document.body.style.cursor = 'auto';
  }, [onHover]);

  const handleDoubleClick = useCallback(
    (e: THREE.Event) => {
      (e as { stopPropagation?: () => void }).stopPropagation?.();
      onDoubleClick(node.id);
    },
    [onDoubleClick, node.id],
  );

  // Build tendril positions for composite → atom connections
  const nodes = useGraphStore((s) => s.nodes);

  const tendrilData = useMemo(() => {
    if (!composite || atomIds.length === 0) return [];
    const result: { startPos: THREE.Vector3; endPos: THREE.Vector3; atomId: number }[] = [];
    for (const aid of atomIds) {
      const atomNode = nodes.get(aid);
      if (!atomNode?.render?.position) continue;
      result.push({
        startPos: new THREE.Vector3(pos.x, pos.y, pos.z),
        endPos: new THREE.Vector3(atomNode.render.position.x, atomNode.render.position.y, atomNode.render.position.z),
        atomId: aid,
      });
    }
    return result;
  }, [composite, atomIds, nodes, pos.x, pos.y, pos.z]);

  // Per-frame animation
  useFrame((state, delta) => {
    if (!groupRef.current || !meshRef.current) return;

    const time = state.clock.elapsedTime;

    // ── Spawn animation ──
    if (!spawnComplete) {
      if (spawnStartRef.current === null) {
        spawnStartRef.current = time;
      }
      const elapsed = time - spawnStartRef.current;
      const spawnProgress = Math.min(elapsed / (SPAWN_DURATION_MS / 1000), 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - spawnProgress, 3);

      currentScaleRef.current = lerp(0.01, 1, eased);

      if (spawnProgress >= 1) {
        setSpawnComplete(true);
        currentScaleRef.current = 1;
      }
    }

    // ── Compute target scale ──
    let targetScale = 1;

    // Hover: scale up
    if (isHovered) {
      targetScale = 1.15;
    }

    // Status animations
    if (nodeStatus === 'new' && spawnComplete) {
      // Gentle pulse
      const pulse = 1 + 0.08 * Math.sin(time * PULSE_SPEED * Math.PI * 2);
      targetScale *= pulse;
    } else if (nodeStatus === 'decaying') {
      // Flicker/desaturation - subtle scale jitter
      const flicker = 0.92 + 0.08 * Math.sin(time * DECAY_FLICKER_SPEED * Math.PI * 2);
      targetScale *= flicker;
    }

    targetScaleRef.current = targetScale;

    // Lerp current scale toward target
    if (!spawnComplete) {
      // Don't lerp during spawn - use eased value directly
    } else {
      currentScaleRef.current = lerp(
        currentScaleRef.current,
        targetScaleRef.current,
        LERP_FACTOR,
      );
    }

    const finalScale = nodeSize * currentScaleRef.current;
    groupRef.current.scale.setScalar(finalScale);
    groupRef.current.position.set(pos.x, pos.y, pos.z);

    // ── Emissive intensity ──
    const baseEmissive = 0.3 + nodeGlow * 0.3;

    if (isHovered) {
      baseMaterial.emissiveIntensity = lerp(
        baseMaterial.emissiveIntensity,
        0.8,
        LERP_FACTOR,
      );
    } else if (isSelected) {
      baseMaterial.emissiveIntensity = lerp(
        baseMaterial.emissiveIntensity,
        0.6,
        LERP_FACTOR,
      );
    } else {
      baseMaterial.emissiveIntensity = lerp(
        baseMaterial.emissiveIntensity,
        baseEmissive,
        LERP_FACTOR,
      );
    }

    // Decaying: desaturate emissive (reduce intensity)
    if (nodeStatus === 'decaying') {
      const decayPulse = 0.15 + 0.15 * Math.abs(Math.sin(time * DECAY_FLICKER_SPEED));
      baseMaterial.emissiveIntensity = Math.min(
        baseMaterial.emissiveIntensity,
        decayPulse,
      );
    }

    // ── Glow sphere ──
    if (glowMeshRef.current) {
      const glowScale = atom
        ? 1.3 + (isSelected ? 0.3 : 0) + (isHovered ? 0.15 : 0)
        : 1.6 + (isSelected ? 0.4 : 0) + (isHovered ? 0.2 : 0);
      glowMeshRef.current.scale.setScalar(glowScale);

      const targetGlowOpacity = isSelected
        ? 0.2
        : isHovered
          ? 0.12
          : baseEmissive * 0.15;

      glowMaterial.opacity = lerp(glowMaterial.opacity, targetGlowOpacity, LERP_FACTOR);

      // Pulse the glow on new nodes
      if (nodeStatus === 'new') {
        const glowPulse = 0.08 + 0.08 * Math.sin(time * PULSE_SPEED * Math.PI * 2);
        glowMaterial.opacity = lerp(glowMaterial.opacity, glowPulse, LERP_FACTOR);
      }

      // Atom nodes: slightly brighter glow
      if (atom) {
        glowMaterial.opacity = Math.min(glowMaterial.opacity * 1.2, 0.25);
      }
    }

    // ── Selection ring ──
    if (torusRef.current) {
      if (isSelected) {
        const ringScale = 1.8 + 0.1 * Math.sin(time * 3);
        torusRef.current.scale.setScalar(ringScale);
        selectionRingMaterial.opacity = lerp(
          selectionRingMaterial.opacity,
          0.6,
          LERP_FACTOR,
        );
        // Slowly rotate the ring
        torusRef.current.rotation.z += delta * 0.5;
        torusRef.current.rotation.x = Math.sin(time * 0.7) * 0.3;
      } else {
        selectionRingMaterial.opacity = lerp(
          selectionRingMaterial.opacity,
          0,
          LERP_FACTOR * 2,
        );
      }
    }

    // ── Composite / Internal Repr halo ──
    if (compositeHaloRef.current) {
      // v8.0: Internal repr has a subtle pulsing ring, composites have stronger halo
      const baseHaloOpacity = internalRepr ? 0.08 : 0.15;
      const haloTargetOpacity = isSelected || isHovered
        ? (internalRepr ? 0.25 : 0.35) + 0.1 * Math.sin(time * 2)
        : baseHaloOpacity + 0.05 * Math.sin(time * 1.5);

      const shouldShowHalo = composite || internalRepr;
      compositeHaloMaterial.opacity = lerp(compositeHaloMaterial.opacity, shouldShowHalo ? haloTargetOpacity : 0, LERP_FACTOR);

      // Rotate the halo slowly
      compositeHaloRef.current.rotation.z += delta * 0.3;
      compositeHaloRef.current.rotation.x = Math.sin(time * 0.5) * 0.4;

      // Scale based on atom count (internal repr typically has fewer atoms)
      const haloScale = internalRepr
        ? 1.3 + Math.min(atomCount * 0.1, 0.4)
        : 1.5 + Math.min(atomCount * 0.15, 0.8);
      compositeHaloRef.current.scale.setScalar(haloScale);
    }
  });

  // Determine label visibility
  const showLabel = viewMode === 'analyze' || isSelected || isHovered || composite;

  // Truncate long labels
  const displayLabel =
    node.label.length > 20 ? node.label.slice(0, 18) + '…' : node.label;

  // v5.0: Layer prefix — v8.0 updated labels
  const layerPrefix = layer === 0 ? '● ' : internalRepr ? '◆L1 ' : `◆L${layer} `;

  // v5.0: Composition chain tooltip (shown on hover for compositional nodes)
  const showCompositionChain = isHovered && compositionChain !== null;

  return (
    <group ref={groupRef}>
      {/* Main node sphere — smaller for atoms */}
      <mesh
        ref={meshRef}
        geometry={sphereGeometry}
        material={baseMaterial}
        onClick={handleClick}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
        onDoubleClick={handleDoubleClick}
      />

      {/* Inner glow sphere (additive) */}
      <mesh ref={glowMeshRef} geometry={glowGeometry} material={glowMaterial} />

      {/* Selection ring (torus) */}
      <mesh ref={torusRef} geometry={torusGeometry} material={selectionRingMaterial} />

      {/* v8.0: Internal representation ring — semi-transparent cyan ring for layer 1 bridge nodes */}
      {internalRepr && (
        <mesh ref={compositeHaloRef} geometry={haloGeometry} material={compositeHaloMaterial} />
      )}

      {/* v8.0: Convergence link tendrils — special dashed lines to structurally equivalent nodes */}
      {(isSelected || isHovered) && hasConvergence && convergenceTargets.map((targetId) => {
        const targetNode = nodes.get(targetId);
        if (!targetNode?.render?.position) return null;
        return (
          <ConvergenceTendrilLine
            key={`conv-${node.id}-${targetId}`}
            startPos={new THREE.Vector3(pos.x, pos.y, pos.z)}
            endPos={new THREE.Vector3(targetNode.render.position.x, targetNode.render.position.y, targetNode.render.position.z)}
            visible={isSelected || isHovered}
          />
        );
      })}

      {/* Composite halo ring — only visible for composites (non-internal-repr) */}
      {composite && !internalRepr && (
        <mesh ref={compositeHaloRef} geometry={haloGeometry} material={compositeHaloMaterial} />
      )}

      {/* Composition tendrils — thin lines from composite to its atoms when selected/hovered */}
      {(isSelected || isHovered) && composite && tendrilData.map((t) => (
        <TendrilLine
          key={`tendril-${node.id}-${t.atomId}`}
          startPos={t.startPos}
          endPos={t.endPos}
          color={nodeColor}
          visible={isSelected || isHovered}
        />
      ))}

      {/* Label */}
      {showLabel && (
        <Billboard follow lockX={false} lockY lockZ={false}>
          <Html
            position={[0, -(atom ? 1.2 : 1.6) * nodeSize, 0]}
            center
            distanceFactor={12}
            style={{
              pointerEvents: 'none',
              userSelect: 'none',
            }}
          >
            <div
              style={{
                background: 'rgba(10, 14, 26, 0.85)',
                border: `1px solid ${composite ? '#FF80AB40' : atom ? '#69F0AE40' : `${layerColor}40`}`,
                borderRadius: '6px',
                padding: '2px 8px',
                color: composite ? '#FF80AB' : atom ? '#69F0AE' : layerColor,
                fontSize: '11px',
                fontFamily: 'monospace',
                whiteSpace: 'nowrap',
                textOverflow: 'ellipsis',
                overflow: 'hidden',
                maxWidth: '140px',
                boxShadow: `0 0 8px ${composite ? '#FF80AB20' : atom ? '#69F0AE20' : `${layerColor}20`}`,
                backdropFilter: 'blur(4px)',
              }}
            >
              {layerPrefix}{displayLabel}
            </div>
          </Html>
        </Billboard>
      )}

      {/* v5.0: Composition chain tooltip on hover */}
      {showCompositionChain && (
        <Billboard follow lockX={false} lockY lockZ={false}>
          <Html
            position={[0, (atom ? 1.5 : 2.0) * nodeSize, 0]}
            center
            distanceFactor={12}
            style={{
              pointerEvents: 'none',
              userSelect: 'none',
            }}
          >
            <div
              style={{
                background: 'rgba(10, 14, 26, 0.92)',
                border: `1px solid ${layerColor}60`,
                borderRadius: '8px',
                padding: '4px 10px',
                color: '#e2e8f0',
                fontSize: '11px',
                fontFamily: 'monospace',
                whiteSpace: 'nowrap',
                boxShadow: `0 0 12px ${layerColor}30`,
                backdropFilter: 'blur(8px)',
              }}
            >
              <span style={{ color: layerColor, fontWeight: 'bold' }}>{compositionChain}</span>
            </div>
          </Html>
        </Billboard>
      )}
    </group>
  );
};

// ── Tendril Line Component ──
// Renders a thin glowing line from a composite node to one of its constituent atoms
function TendrilLine({
  startPos,
  endPos,
  color,
  visible,
}: {
  startPos: THREE.Vector3;
  endPos: THREE.Vector3;
  color: string;
  visible: boolean;
}) {
  const lineRef = useRef<THREE.Line>(null);

  const material = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        color: new THREE.Color(color),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [color],
  );

  const geometry = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints([startPos, endPos]);
  }, [startPos, endPos]);

  useFrame((state) => {
    const time = state.clock.elapsedTime;
    const targetOpacity = visible ? 0.3 + 0.1 * Math.sin(time * 3) : 0;
    material.opacity = lerp(material.opacity, targetOpacity, LERP_FACTOR);
  });

  return (
    <primitive ref={lineRef} object={new THREE.Line(geometry, material)} />
  );
}

// ── Convergence Tendril Line Component ──
// Renders a special dashed line between convergent nodes (e.g., "anjing" ↔ "dog")
// Uses a distinct purple/white color to differentiate from composition tendrils
function ConvergenceTendrilLine({
  startPos,
  endPos,
  visible,
}: {
  startPos: THREE.Vector3;
  endPos: THREE.Vector3;
  visible: boolean;
}) {
  const lineRef = useRef<THREE.Line>(null);

  const CONVERGENCE_COLOR = '#E040FB'; // Bright purple for convergence links

  const material = useMemo(
    () =>
      new THREE.LineDashedMaterial({
        color: new THREE.Color(CONVERGENCE_COLOR),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        dashSize: 0.4,
        gapSize: 0.25,
      }),
    [],
  );

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry().setFromPoints([startPos, endPos]);
    // computeLineDistances is needed for LineDashedMaterial
    const positions = geo.getAttribute('position');
    const lineDistances = new Float32Array(positions.count);
    lineDistances[0] = 0;
    for (let i = 1; i < positions.count; i++) {
      const x = positions.getX(i) - positions.getX(i - 1);
      const y = positions.getY(i) - positions.getY(i - 1);
      const z = positions.getZ(i) - positions.getZ(i - 1);
      lineDistances[i] = lineDistances[i - 1] + Math.sqrt(x * x + y * y + z * z);
    }
    geo.setAttribute('lineDistance', new THREE.BufferAttribute(lineDistances, 1));
    return geo;
  }, [startPos, endPos]);

  useFrame((state) => {
    const time = state.clock.elapsedTime;
    const targetOpacity = visible ? 0.5 + 0.2 * Math.sin(time * 4) : 0;
    material.opacity = lerp(material.opacity, targetOpacity, LERP_FACTOR);
  });

  return (
    <primitive ref={lineRef} object={new THREE.Line(geometry, material)} />
  );
}

export const GraphNode = React.memo(GraphNodeComponent);
