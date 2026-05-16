'use client';

import React, { Suspense, useRef, useCallback, useMemo, useEffect } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Stars } from '@react-three/drei';
import { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import * as THREE from 'three';
import { useGraphStore, useUIStore } from '@/store/aamStore';
import { GraphNode } from './GraphNode';
import { GraphEdge } from './GraphEdge';
import { useForceLayout } from './ForceGraph';
import { isCompositeNode, isAtomNode, computeNodeLayer, getLayerColor, getLayerLabel, isInternalRepresentation, hasConvergenceLinks, getConvergenceTargets } from '@/lib/nodeRendering';
import { ErrorBoundary } from '@/components/ErrorBoundary';

// ── Scene Constants ──
const BG_COLOR = '#0a0e1a';
const FOG_NEAR = 30;
const FOG_FAR = 80;
const FOCUS_LERP_SPEED = 0.05;

// ── Loading Fallback ──
function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshStandardMaterial
        color="#00E5FF"
        emissive="#00E5FF"
        emissiveIntensity={0.5}
        wireframe
      />
    </mesh>
  );
}

// ── Scene (lights, fog, background) ──
function SceneSetup() {
  return (
    <>
      {/* Background color & fog as R3F elements */}
      <color attach="background" args={[BG_COLOR]} />
      <fog attach="fog" args={[BG_COLOR, FOG_NEAR, FOG_FAR]} />

      {/* Ambient light - subtle base illumination */}
      <ambientLight intensity={0.25} color="#4466aa" />

      {/* Main directional light - warm from above-right */}
      <directionalLight
        position={[10, 15, 8]}
        intensity={0.8}
        color="#aaccff"
        castShadow={false}
      />

      {/* Secondary fill light from below-left */}
      <directionalLight
        position={[-8, -5, -6]}
        intensity={0.2}
        color="#6644aa"
      />

      {/* Point light at origin for central glow */}
      <pointLight
        position={[0, 0, 0]}
        intensity={0.4}
        color="#00E5FF"
        distance={30}
        decay={2}
      />

      {/* Background stars */}
      <Stars
        radius={60}
        depth={50}
        count={2000}
        factor={3}
        saturation={0.2}
        fade
        speed={0.5}
      />
    </>
  );
}

// ── Graph Content (nodes + edges) ──
function GraphContent() {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const selectedNodeId = useUIStore((s) => s.selectedNodeId);
  const hoveredNodeId = useUIStore((s) => s.focusedNodeId);
  const selectNode = useUIStore((s) => s.selectNode);

  // Run force layout
  useForceLayout();

  // Build neighbor set for hover highlighting
  const neighborIds = useMemo(() => {
    if (!hoveredNodeId) return new Set<number>();
    const neighborSet = new Set<number>();
    edges.forEach((edge) => {
      if (edge.source === hoveredNodeId) neighborSet.add(edge.target);
      if (edge.target === hoveredNodeId) neighborSet.add(edge.source);
    });
    return neighborSet;
  }, [hoveredNodeId, edges]);

  // Handlers
  const handleNodeClick = useCallback(
    (nodeId: number) => {
      selectNode(nodeId);
    },
    [selectNode],
  );

  const handleNodeHover = useCallback(
    (nodeId: number | null) => {
      // Use focusedNodeId as hover state proxy
      useUIStore.getState().focusNode(nodeId);
    },
    [],
  );

  const handleNodeDoubleClick = useCallback(
    (nodeId: number) => {
      // Will be connected to camera focus via controlsRef in parent
      useUIStore.getState().focusNode(nodeId);
    },
    [],
  );

  // Convert nodes map to array
  const nodeList = useMemo(() => Array.from(nodes.values()), [nodes]);
  const edgeList = useMemo(() => Array.from(edges.values()), [edges]);

  return (
    <group>
      {/* Render edges first (behind nodes) */}
      {edgeList.map((edge) => {
        const sourceNode = nodes.get(edge.source);
        const targetNode = nodes.get(edge.target);
        if (!sourceNode || !targetNode) return null;

        const sourcePos = new THREE.Vector3(
          sourceNode.render?.position?.x ?? 0,
          sourceNode.render?.position?.y ?? 0,
          sourceNode.render?.position?.z ?? 0,
        );
        const targetPos = new THREE.Vector3(
          targetNode.render?.position?.x ?? 0,
          targetNode.render?.position?.y ?? 0,
          targetNode.render?.position?.z ?? 0,
        );

        // Dim edges that are not connected to hovered node
        const isConnectedToHover =
          hoveredNodeId === null ||
          edge.source === hoveredNodeId ||
          edge.target === hoveredNodeId;

        // Detect composition edges: atom → composite or composite → atom
        const isCompositionEdge =
          (isAtomNode(sourceNode) && isCompositeNode(targetNode)) ||
          (isCompositeNode(sourceNode) && isAtomNode(targetNode));

        return (
          <group key={edge.id} visible={isConnectedToHover || hoveredNodeId === null}>
            <GraphEdge edge={edge} sourcePos={sourcePos} targetPos={targetPos} isCompositionEdge={isCompositionEdge} />
          </group>
        );
      })}

      {/* Render nodes */}
      {nodeList.map((node) => {
        const isSelected = node.id === selectedNodeId;
        const isHovered = node.id === hoveredNodeId;
        const isNeighbor = neighborIds.has(node.id);

        // Dim nodes that are neither hovered, selected, nor neighbors
        const isDimmed =
          hoveredNodeId !== null && !isSelected && !isHovered && !isNeighbor;

        return (
          <group
            key={node.id}
            visible={!isDimmed}
          >
            <GraphNode
              node={node}
              isSelected={isSelected}
              isHovered={isHovered}
              onClick={handleNodeClick}
              onHover={handleNodeHover}
              onDoubleClick={handleNodeDoubleClick}
            />
          </group>
        );
      })}
    </group>
  );
}

// ── Camera Focus Controller ──
function CameraFocusController({ controlsRef }: { controlsRef: React.RefObject<OrbitControlsImpl | null> }) {
  const focusedNodeId = useUIStore((s) => s.focusedNodeId);
  const nodes = useGraphStore((s) => s.nodes);
  const targetRef = useRef(new THREE.Vector3(0, 0, 0));
  const isFocusing = useRef(false);

  useFrame(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    if (focusedNodeId !== null) {
      const node = nodes.get(focusedNodeId);
      if (node?.render?.position) {
        targetRef.current.set(
          node.render.position.x,
          node.render.position.y,
          node.render.position.z,
        );
        isFocusing.current = true;
      }
    }

    if (isFocusing.current) {
      controls.target.lerp(targetRef.current, FOCUS_LERP_SPEED);

      // Move camera closer to the focused node
      const cameraOffset = new THREE.Vector3(5, 3, 5);
      const desiredCameraPos = targetRef.current.clone().add(cameraOffset);
      controls.object.position.lerp(desiredCameraPos, FOCUS_LERP_SPEED * 0.5);

      // Stop when close enough
      if (controls.target.distanceTo(targetRef.current) < 0.05) {
        isFocusing.current = false;
      }
    }
  });

  return null;
}

// ── HUD Overlay ──
function GraphHUD() {
  const selectedNodeId = useUIStore((s) => s.selectedNodeId);
  const nodes = useGraphStore((s) => s.nodes);
  const selectedNode = selectedNodeId !== null ? nodes.get(selectedNodeId) : undefined;

  return (
    <Html fullscreen style={{ pointerEvents: 'none' }}>
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
        }}
      >
        {/* Top-left: Title */}
        <div
          style={{
            position: 'absolute',
            top: '16px',
            left: '20px',
            color: '#00E5FF',
            fontSize: '13px',
            fontFamily: 'monospace',
            opacity: 0.6,
            letterSpacing: '2px',
            textTransform: 'uppercase',
          }}
        >
          RSVS · Recursive Symbolic Vector Space
        </div>

        {/* Bottom-left: Selected node info */}
        {selectedNode && (
          <div
            style={{
              position: 'absolute',
              bottom: '20px',
              left: '20px',
              background: 'rgba(10, 14, 26, 0.8)',
              border: `1px solid ${selectedNode.render?.color ?? '#00E5FF'}40`,
              borderRadius: '8px',
              padding: '10px 16px',
              color: '#e0e0e0',
              fontSize: '12px',
              fontFamily: 'monospace',
              backdropFilter: 'blur(8px)',
              maxWidth: '300px',
            }}
          >
            <div style={{ color: selectedNode.render?.color ?? '#00E5FF', fontWeight: 'bold', marginBottom: '4px' }}>
              {selectedNode.label}
            </div>
            <div style={{ opacity: 0.6, fontSize: '11px' }}>
              ID: {selectedNode.id} · Tier {selectedNode.tier} · {selectedNode.kind}
            </div>
            <div style={{ opacity: 0.5, fontSize: '11px', marginTop: '2px' }}>
              Confidence: {(selectedNode.confidence * 100).toFixed(1)}%
            </div>
            {/* v5.0: Layer info */}
            <div style={{ opacity: 0.6, fontSize: '11px', marginTop: '2px', color: getLayerColor(computeNodeLayer(selectedNode)) }}>
              {getLayerLabel(computeNodeLayer(selectedNode))}
              {selectedNode.internal_representation && ' · Internal Repr'}
              {selectedNode.grounding_score !== undefined && ` · Grounding: ${(selectedNode.grounding_score * 100).toFixed(1)}%`}
            </div>
            {/* v8.0: Convergence links info */}
            {hasConvergenceLinks(selectedNode) && (
              <div style={{ opacity: 0.7, fontSize: '11px', marginTop: '2px', color: '#E040FB' }}>
                ↔ Convergent with: {getConvergenceTargets(selectedNode).map(tid => {
                  const targetNode = nodes.get(tid);
                  return targetNode ? targetNode.label : `#${tid}`;
                }).join(', ')}
              </div>
            )}
            {/* v5.0: Composition chain */}
            {selectedNode.compositions && selectedNode.compositions.length > 0 && (
              <div style={{ opacity: 0.7, fontSize: '11px', marginTop: '4px', color: getLayerColor(computeNodeLayer(selectedNode)) }}>
                {selectedNode.label} = {selectedNode.compositions.map(c => c.label).join(' + ')}
              </div>
            )}
          </div>
        )}

        {/* Bottom-right: Controls hint */}
        <div
          style={{
            position: 'absolute',
            bottom: '20px',
            right: '20px',
            color: '#4466aa',
            fontSize: '11px',
            fontFamily: 'monospace',
            opacity: 0.4,
            textAlign: 'right',
            lineHeight: '1.6',
          }}
        >
          <div>drag to rotate</div>
          <div>scroll to zoom</div>
          <div>click node to select</div>
          <div>double-click to focus</div>
        </div>
      </div>
    </Html>
  );
}

// ── Main Export ──
export default function GraphScene3D() {
  const controlsRef = useRef<OrbitControlsImpl>(null);

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        background: BG_COLOR,
      }}
    >
      <ErrorBoundary name="WebGL-Canvas">
        <Canvas
          camera={{
            position: [18, 12, 18],
            fov: 55,
            near: 0.1,
            far: 200,
          }}
          gl={{
            antialias: true,
            alpha: false,
            toneMapping: THREE.ACESFilmicToneMapping,
            toneMappingExposure: 1.2,
          }}
          dpr={[1, 2]}
          style={{ background: BG_COLOR }}
        >
          <Suspense fallback={<LoadingFallback />}>
            <SceneSetup />
            <GraphContent />
            <CameraFocusController controlsRef={controlsRef} />
            <GraphHUD />
            <OrbitControls
              ref={controlsRef}
              enableDamping
              dampingFactor={0.08}
              rotateSpeed={0.6}
              zoomSpeed={0.8}
              panSpeed={0.5}
              minDistance={5}
              maxDistance={60}
              makeDefault
            />
          </Suspense>
        </Canvas>
      </ErrorBoundary>
    </div>
  );
}
