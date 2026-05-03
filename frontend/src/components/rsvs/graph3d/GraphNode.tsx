/* eslint-disable react-hooks/immutability */
'use client';

import React, { useRef, useMemo, useState, useCallback, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import type { RSVSNode, NodeStatus } from '@/lib/types';
import { useUIStore } from '@/store/rsvsStore';

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

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
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

  // Materials stored in refs (mutable for useFrame)
  const baseMaterialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const glowMaterialRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const selectionRingMaterialRef = useRef<THREE.MeshBasicMaterial | null>(null);

  // Initialize materials once in useEffect
  useEffect(() => {
    if (baseMaterialRef.current == null) {
      baseMaterialRef.current = new THREE.MeshStandardMaterial({
        color: new THREE.Color(nodeColor),
        emissive: new THREE.Color(nodeColor),
        emissiveIntensity: 0.3,
        metalness: 0.4,
        roughness: 0.3,
        transparent: true,
        opacity: 0.95,
      });
    }
    if (glowMaterialRef.current == null) {
      glowMaterialRef.current = new THREE.MeshBasicMaterial({
        color: new THREE.Color(nodeColor),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
    }
    if (selectionRingMaterialRef.current == null) {
      selectionRingMaterialRef.current = new THREE.MeshBasicMaterial({
        color: new THREE.Color('#ffffff'),
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
    }
  }, [nodeColor]);

  // Spawn animation tracking
  const spawnStartRef = useRef<number | null>(null);
  const [spawnComplete, setSpawnComplete] = useState(false);

  // Smooth scale tracking
  const targetScaleRef = useRef(1);
  const currentScaleRef = useRef(0.01);

  // Derived render properties
  const nodeColor = node.render?.color ?? '#00E5FF';
  const nodeSize = node.render?.size ?? 1;
  const nodeGlow = node.render?.glow ?? 0;
  const nodeStatus = node.status ?? 'stable';
  const pos = node.render?.position ?? { x: 0, y: 0, z: 0 };

  // Shared geometry (memoized)
  const sphereGeometry = useMemo(
    () => new THREE.SphereGeometry(1, 16, 16),
    [],
  );

  const glowGeometry = useMemo(
    () => new THREE.SphereGeometry(1, 16, 16),
    [],
  );

  const torusGeometry = useMemo(
    () => new THREE.TorusGeometry(1, 0.06, 8, 32),
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

  // Per-frame animation
  useFrame((state, delta) => {
    const baseMat = baseMaterialRef.current;
    const glowMat = glowMaterialRef.current;
    const ringMat = selectionRingMaterialRef.current;
    if (!groupRef.current || !meshRef.current || !baseMat || !glowMat || !ringMat) return;

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
      baseMat.emissiveIntensity = lerp(
        baseMat.emissiveIntensity,
        0.8,
        LERP_FACTOR,
      );
    } else if (isSelected) {
      baseMat.emissiveIntensity = lerp(
        baseMat.emissiveIntensity,
        0.6,
        LERP_FACTOR,
      );
    } else {
      baseMat.emissiveIntensity = lerp(
        baseMat.emissiveIntensity,
        baseEmissive,
        LERP_FACTOR,
      );
    }

    // Decaying: desaturate emissive (reduce intensity)
    if (nodeStatus === 'decaying') {
      const decayPulse = 0.15 + 0.15 * Math.abs(Math.sin(time * DECAY_FLICKER_SPEED));
      baseMat.emissiveIntensity = Math.min(
        baseMat.emissiveIntensity,
        decayPulse,
      );
    }

    // ── Glow sphere ──
    if (glowMeshRef.current) {
      const glowScale = 1.6 + (isSelected ? 0.4 : 0) + (isHovered ? 0.2 : 0);
      glowMeshRef.current.scale.setScalar(glowScale);

      const targetGlowOpacity = isSelected
        ? 0.2
        : isHovered
          ? 0.12
          : baseEmissive * 0.15;

      glowMat.opacity = lerp(glowMat.opacity, targetGlowOpacity, LERP_FACTOR);

      // Pulse the glow on new nodes
      if (nodeStatus === 'new') {
        const glowPulse = 0.08 + 0.08 * Math.sin(time * PULSE_SPEED * Math.PI * 2);
        glowMat.opacity = lerp(glowMat.opacity, glowPulse, LERP_FACTOR);
      }
    }

    // ── Selection ring ──
    if (torusRef.current) {
      if (isSelected) {
        const ringScale = 1.8 + 0.1 * Math.sin(time * 3);
        torusRef.current.scale.setScalar(ringScale);
        ringMat.opacity = lerp(
          ringMat.opacity,
          0.6,
          LERP_FACTOR,
        );
        // Slowly rotate the ring
        torusRef.current.rotation.z += delta * 0.5;
        torusRef.current.rotation.x = Math.sin(time * 0.7) * 0.3;
      } else {
        ringMat.opacity = lerp(
          ringMat.opacity,
          0,
          LERP_FACTOR * 2,
        );
      }
    }
  });

  // Determine label visibility
  const showLabel = viewMode === 'analyze' || isSelected || isHovered;

  // Truncate long labels
  const displayLabel =
    node.label.length > 20 ? node.label.slice(0, 18) + '…' : node.label;

  return (
    <group ref={groupRef}>
      {/* Main node sphere */}
      <mesh
        ref={meshRef}
        geometry={sphereGeometry}
        material={baseMaterialRef.current}
        onClick={handleClick}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
        onDoubleClick={handleDoubleClick}
      />

      {/* Inner glow sphere (additive) */}
      <mesh ref={glowMeshRef} geometry={glowGeometry} material={glowMaterialRef.current} />

      {/* Selection ring (torus) */}
      <mesh ref={torusRef} geometry={torusGeometry} material={selectionRingMaterialRef.current} />

      {/* Label */}
      {showLabel && (
        <Billboard follow lockX={false} lockY lockZ={false}>
          <Html
            position={[0, -1.6 * nodeSize, 0]}
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
                border: `1px solid ${nodeColor}40`,
                borderRadius: '6px',
                padding: '2px 8px',
                color: nodeColor,
                fontSize: '11px',
                fontFamily: 'monospace',
                whiteSpace: 'nowrap',
                textOverflow: 'ellipsis',
                overflow: 'hidden',
                maxWidth: '120px',
                boxShadow: `0 0 8px ${nodeColor}20`,
                backdropFilter: 'blur(4px)',
              }}
            >
              {displayLabel}
            </div>
          </Html>
        </Billboard>
      )}
    </group>
  );
};

export const GraphNode = React.memo(GraphNodeComponent);
