'use client';

import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, Crosshair, Pin, GitCompare, Download, Brain,
  Link2, Activity, Eye, Sparkles, AlertTriangle,
  Shield, Layers, Globe, GitBranch, ArrowUpDown,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useUIStore, useGraphStore, useModeResultStore } from '@/store/rsvsStore';
import type { RSVSNode, Tier, PolicyMeta, LanguageLink } from '@/lib/types';
import { getStatusColor } from '@/lib/nodeRendering';
import AppraisePanel from '@/components/rsvs/AppraisePanel';
import RelatePanel from '@/components/rsvs/RelatePanel';

const TIER_COLORS: Record<Tier, string> = {
  1: '#00E5FF',
  2: '#FFB74D',
  3: '#FF5252',
};

const TIER_LABELS: Record<Tier, string> = {
  1: 'T1 — Stable',
  2: 'T2 — Alert',
  3: 'T3 — Critical',
};

const STATUS_STYLES: Record<string, { color: string; label: string }> = {
  new: { color: '#B388FF', label: 'New' },
  stable: { color: '#69F0AE', label: 'Stable' },
  candidate: { color: '#00E5FF', label: 'Candidate' },
  decaying: { color: '#FFB74D', label: 'Decaying' },
  deprecated: { color: '#FFB74D', label: 'Deprecated' },
  removed: { color: '#FF5252', label: 'Removed' },
  quarantine: { color: '#FF5252', label: 'Quarantine' },
};

function ConfidenceRing({ value }: { value: number }) {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - value);
  const color = value > 0.7 ? '#69F0AE' : value > 0.4 ? '#FFB74D' : '#FF5252';

  return (
    <div className="relative w-24 h-24 flex items-center justify-center">
      <svg width="96" height="96" className="-rotate-90">
        <circle
          cx="48" cy="48" r={radius}
          fill="none" stroke="#1e293b" strokeWidth="6"
        />
        <circle
          cx="48" cy="48" r={radius}
          fill="none" stroke={color} strokeWidth="6"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.6s ease' }}
        />
      </svg>
      <span className="absolute text-lg font-bold" style={{ color }}>
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="bg-[#0d1520] rounded-lg p-3 text-center border border-[#1e293b]">
      <div className="text-lg font-bold text-[#e2e8f0]">{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-[#64748b]">{label}</div>
    </div>
  );
}

function WeightBar({ label, weight }: { label: string; weight: number }) {
  const color = weight > 0.7 ? '#69F0AE' : weight > 0.4 ? '#FFB74D' : '#64748b';
  return (
    <div className="flex items-center gap-2 py-1">
      <span className="text-xs text-[#94a3b8] w-20 truncate font-mono">{label}</span>
      <div className="flex-1 h-2 bg-[#0d1520] rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${weight * 100}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />
      </div>
      <span className="text-[10px] text-[#64748b] w-10 text-right font-mono">{weight.toFixed(2)}</span>
    </div>
  );
}

// ── v4.2 Compression State Section ──
function CompressionStateSection({ state }: { state: string }) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <Layers className="w-3.5 h-3.5 text-[#FFB74D]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Compression State</h3>
      </div>
      <StatCard label="State" value={state === 'compressed' ? 'Compressed' : 'Raw'} />
    </div>
  );
}

// ── v4.2 Policy Meta Section ──
function PolicyMetaSection({ meta }: { meta: PolicyMeta }) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <Shield className="w-3.5 h-3.5 text-[#69F0AE]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Policy & Governance</h3>
      </div>
      <div className="grid grid-cols-2 gap-2 mb-2">
        <StatCard label="Governance Score" value={meta.governance_score.toFixed(2)} />
        <StatCard label="Status Flips" value={meta.status_flip_count} />
      </div>
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-[#0d1520] rounded-lg p-2 text-center border border-[#1e293b]">
          <div className="text-xs font-bold text-[#e2e8f0]">{meta.auto_promote ? '✓' : '✗'}</div>
          <div className="text-[9px] uppercase tracking-wider text-[#64748b]">Auto-Promote</div>
        </div>
        <div className="bg-[#0d1520] rounded-lg p-2 text-center border border-[#1e293b]">
          <div className="text-xs font-bold text-[#e2e8f0]">T{meta.max_tier}</div>
          <div className="text-[9px] uppercase tracking-wider text-[#64748b]">Max Tier</div>
        </div>
        <div className="bg-[#0d1520] rounded-lg p-2 text-center border border-[#1e293b]">
          <div className="text-xs font-bold text-[#e2e8f0]">{meta.min_confidence.toFixed(2)}</div>
          <div className="text-[9px] uppercase tracking-wider text-[#64748b]">Min Conf</div>
        </div>
      </div>
    </div>
  );
}

// ── v4.2 Language Links Section ──
function LanguageLinksSection({ links }: { links: LanguageLink[] }) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <Globe className="w-3.5 h-3.5 text-[#80D8FF]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Language Links</h3>
      </div>
      <div className="space-y-1">
        {links.map((link, i) => (
          <div key={i} className="flex items-center gap-2 px-2 py-1 rounded-md bg-[#0d1520] border border-[#1e293b]">
            <Badge variant="outline" className="text-[9px] border-[#334155] text-[#64748b] shrink-0 px-1">
              {link.lang}
            </Badge>
            <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">{link.label}</span>
            <span className="text-[10px] text-[#64748b] font-mono shrink-0">{link.confidence.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── v4.2 Derived From Section ──
function DerivedFromSection({ nodeIds, onSelectNode }: { nodeIds: number[]; onSelectNode: (id: number) => void }) {
  const nodes = useGraphStore((s) => s.nodes);

  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <GitBranch className="w-3.5 h-3.5 text-[#B388FF]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Derived From</h3>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {nodeIds.map((id) => {
          const sourceNode = nodes.get(id);
          const label = sourceNode?.label ?? `#${id}`;
          return (
            <button
              key={id}
              type="button"
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-mono bg-[#B388FF10] border border-[#B388FF30] text-[#B388FF] hover:bg-[#B388FF20] transition-colors cursor-pointer"
              onClick={() => onSelectNode(id)}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function RightNodeDrawer() {
  const selectedNodeId = useUIStore((s) => s.selectedNodeId);
  const isDrawerOpen = useUIStore((s) => s.isDrawerOpen);
  const closeDrawer = useUIStore((s) => s.closeDrawer);
  const focusNode = useUIStore((s) => s.focusNode);
  const togglePinNode = useUIStore((s) => s.togglePinNode);
  const pinnedNodeIds = useUIStore((s) => s.pinnedNodeIds);
  const getNode = useGraphStore((s) => s.getNode);
  const getNodeNeighbors = useGraphStore((s) => s.getNodeNeighbors);
  const selectNode = useUIStore((s) => s.selectNode);
  const modeResult = useModeResultStore((s) => s.currentResult);

  const node = selectedNodeId !== null ? getNode(selectedNodeId) : undefined;
  const neighbors = useMemo(
    () => (selectedNodeId !== null ? getNodeNeighbors(selectedNodeId) : { nodes: [], edges: [] }),
    [selectedNodeId, getNodeNeighbors]
  );

  const handleClose = useCallback(() => {
    closeDrawer();
  }, [closeDrawer]);

  const handleFocus = useCallback(() => {
    if (selectedNodeId !== null) focusNode(selectedNodeId);
  }, [selectedNodeId, focusNode]);

  const handlePin = useCallback(() => {
    if (selectedNodeId !== null) togglePinNode(selectedNodeId);
  }, [selectedNodeId, togglePinNode]);

  const handleSelectNode = useCallback((id: number) => {
    selectNode(id);
    focusNode(id);
  }, [selectNode, focusNode]);

  const isPinned = selectedNodeId !== null && pinnedNodeIds.has(selectedNodeId);

  const strongestEdges = useMemo(() => {
    return [...neighbors.edges].sort((a, b) => b.weight - a.weight).slice(0, 5);
  }, [neighbors.edges]);

  return (
    <>
      {/* Backdrop overlay */}
      <AnimatePresence>
        {isDrawerOpen && (
          <motion.div
            className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={handleClose}
          />
        )}
      </AnimatePresence>

      {/* Drawer */}
      <AnimatePresence>
        {isDrawerOpen && node && (
          <motion.aside
            className="fixed right-0 top-0 h-full z-50 flex flex-col border-l border-[#1e293b] overflow-hidden"
            style={{ width: 'min(38vw, 460px)', backgroundColor: '#0a0e18' }}
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
            role="complementary"
            aria-label="Node detail panel"
          >
            {/* Header */}
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2 min-w-0">
                  <Brain className="w-5 h-5 text-[#00E5FF] shrink-0" />
                  <h2 className="text-lg font-bold text-[#e2e8f0] truncate">{node.label}</h2>
                </div>
                <Button
                  variant="ghost" size="icon"
                  className="shrink-0 text-[#64748b] hover:text-[#e2e8f0] hover:bg-[#1e293b]"
                  onClick={handleClose}
                  aria-label="Close panel"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>

              {/* Badges row */}
              <div className="flex flex-wrap gap-2 mb-3">
                <Badge variant="outline" className="text-[10px] font-mono border-[#334155] text-[#94a3b8]">
                  ID: {node.id}
                </Badge>
                <Badge
                  className="text-[10px] font-semibold"
                  style={{
                    backgroundColor: '#0d3b66',
                    color: '#69F0AE',
                    borderColor: 'transparent',
                  }}
                >
                  <Sparkles className="w-3 h-3 mr-1" />
                  {node.kind.toUpperCase()}
                </Badge>
                <Badge
                  className="text-[10px] font-semibold"
                  style={{
                    backgroundColor: `${TIER_COLORS[node.tier]}15`,
                    color: TIER_COLORS[node.tier],
                    borderColor: `${TIER_COLORS[node.tier]}40`,
                  }}
                >
                  {TIER_LABELS[node.tier]}
                </Badge>
                {node.status && (
                  <Badge
                    className="text-[10px]"
                    style={{
                      backgroundColor: `${(STATUS_STYLES[node.status] ?? STATUS_STYLES.new).color}15`,
                      color: (STATUS_STYLES[node.status] ?? STATUS_STYLES.new).color,
                      borderColor: `${(STATUS_STYLES[node.status] ?? STATUS_STYLES.new).color}40`,
                    }}
                  >
                    {(STATUS_STYLES[node.status] ?? STATUS_STYLES.new).label}
                  </Badge>
                )}
                {node.is_seed && (
                  <Badge className="text-[10px] bg-[#FFD74015] text-[#FFD740] border-[#FFD74040]">
                    ★ Seed
                  </Badge>
                )}
              </div>

              {/* Provenance */}
              {node.provenance && (
                <div className="text-[10px] text-[#475569] font-mono flex gap-3">
                  {node.provenance.source_domain && <span>domain: {node.provenance.source_domain}</span>}
                  {node.provenance.source_type && <span>type: {node.provenance.source_type}</span>}
                </div>
              )}

              {/* Semantic tag (v4.2) */}
              {node.semantic && (
                <div className="text-[10px] text-[#80D8FF] font-mono mt-1">
                  semantic: {node.semantic.compression_state}
                  {node.semantic.derived_from_node_ids.length > 0 && ` · derived from [${node.semantic.derived_from_node_ids.join(', ')}]`}
                </div>
              )}
            </div>

            <Separator className="bg-[#1e293b]" />

            <ScrollArea className="flex-1 px-5 py-4">
              {/* Confidence */}
              <div className="flex items-center gap-4 mb-5">
                <ConfidenceRing value={node.confidence} />
                <div>
                  <div className="text-sm font-semibold text-[#e2e8f0] mb-1">Confidence</div>
                  <div className="text-xs text-[#64748b]">
                    {node.confidence > 0.7 ? 'High confidence' :
                     node.confidence > 0.4 ? 'Moderate confidence' :
                     'Low confidence — needs more data'}
                  </div>
                </div>
              </div>

              {/* Sense Block */}
              {node.sense && (
                <div className="mb-5">
                  <div className="flex items-center gap-2 mb-2">
                    <Eye className="w-3.5 h-3.5 text-[#80D8FF]" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Sense</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <StatCard label="Count" value={node.sense.count} />
                    <StatCard label="Active" value={node.sense.active_index ?? 0} />
                  </div>
                  {node.sense.coherence !== null && (
                    <div className="mt-2">
                      <div className="flex justify-between text-[10px] mb-1">
                        <span className="text-[#64748b]">Coherence</span>
                        <span className="text-[#94a3b8] font-mono">{node.sense.coherence.toFixed(2)}</span>
                      </div>
                      <div className="h-1.5 bg-[#0d1520] rounded-full overflow-hidden">
                        <motion.div
                          className="h-full rounded-full bg-[#80D8FF]"
                          initial={{ width: 0 }}
                          animate={{ width: `${(node.sense.coherence ?? 0) * 100}%` }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}

              <Separator className="bg-[#1e293b] mb-5" />

              {/* v4.2: Compression State */}
              {node.compression_state === 'compressed' && (
                <>
                  <CompressionStateSection state={node.compression_state} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* v4.2: Policy & Governance */}
              {node.policy_meta && (
                <>
                  <PolicyMetaSection meta={node.policy_meta} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* v4.2: Derived From */}
              {node.derived_from_node_ids && node.derived_from_node_ids.length > 0 && (
                <>
                  <DerivedFromSection nodeIds={node.derived_from_node_ids} onSelectNode={handleSelectNode} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* v4.2: Language Links */}
              {node.language_links && node.language_links.length > 0 && (
                <>
                  <LanguageLinksSection links={node.language_links} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* Composition */}
              {node.composition && (
                <div className="mb-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Activity className="w-3.5 h-3.5 text-[#B388FF]" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">
                      {node.kind === 'node' ? 'Related Composites' : 'Member Nodes'}
                    </h3>
                  </div>
                  {node.composition.related_composites.map((item, i) => {
                    const id: number = 'atom_id' in item ? item.atom_id : item.composite_id;
                    const label = `#${id}`;
                    return (
                      <div
                        key={i}
                        className="cursor-pointer hover:bg-[#1e293b]/50 rounded px-1 -mx-1 transition-colors"
                        onClick={() => selectNode(id)}
                      >
                        <WeightBar label={label} weight={item.weight} />
                      </div>
                    );
                  })}
                </div>
              )}

              <Separator className="bg-[#1e293b] mb-5" />

              {/* Connectivity */}
              <div className="mb-5">
                <div className="flex items-center gap-2 mb-3">
                  <Link2 className="w-3.5 h-3.5 text-[#FFD740]" />
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Connectivity</h3>
                </div>
                <div className="grid grid-cols-3 gap-2 mb-3">
                  <StatCard label="Total" value={node.metrics?.degree ?? neighbors.nodes.length} />
                  <StatCard label="In" value={node.metrics?.in_degree ?? 0} />
                  <StatCard label="Out" value={node.metrics?.out_degree ?? 0} />
                </div>
                {strongestEdges.length > 0 && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-[#475569] mb-2">Strongest Connections</div>
                    {strongestEdges.map((edge) => (
                      <div key={edge.id} className="flex items-center justify-between py-1 text-xs">
                        <span className="text-[#94a3b8] font-mono truncate">
                          {edge.source} → {edge.target}
                        </span>
                        <span className="text-[#64748b] font-mono ml-2">{edge.weight.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Mode Result Panels (Appraise / Relate) */}
              {modeResult && modeResult.type === 'appraise' && (
                <>
                  <Separator className="bg-[#1e293b] mb-5" />
                  <AppraisePanel result={modeResult.data} />
                </>
              )}
              {modeResult && modeResult.type === 'relate' && (
                <>
                  <Separator className="bg-[#1e293b] mb-5" />
                  <RelatePanel result={modeResult.data} />
                </>
              )}

              {node.metrics?.last_updated_at && (
                <div className="text-[10px] text-[#475569] font-mono">
                  Last updated: {new Date(node.metrics.last_updated_at).toLocaleTimeString()}
                </div>
              )}
            </ScrollArea>

            {/* Actions Bar */}
            <div className="px-5 py-3 border-t border-[#1e293b] bg-[#080c14]">
              <div className="flex gap-2">
                <Button
                  variant="ghost" size="sm"
                  className="flex-1 text-xs text-[#94a3b8] hover:text-[#00E5FF] hover:bg-[#00E5FF10]"
                  onClick={handleFocus}
                >
                  <Crosshair className="w-3.5 h-3.5 mr-1.5" /> Focus
                </Button>
                <Button
                  variant="ghost" size="sm"
                  className="flex-1 text-xs hover:bg-[#FFB74D10]"
                  style={{ color: isPinned ? '#FFB74D' : '#94a3b8' }}
                  onClick={handlePin}
                >
                  <Pin className={`w-3.5 h-3.5 mr-1.5 ${isPinned ? 'fill-current' : ''}`} />
                  {isPinned ? 'Pinned' : 'Pin'}
                </Button>
                <Button
                  variant="ghost" size="sm"
                  className="flex-1 text-xs text-[#94a3b8] hover:text-[#B388FF] hover:bg-[#B388FF10]"
                >
                  <GitCompare className="w-3.5 h-3.5 mr-1.5" /> Compare
                </Button>
                <Button
                  variant="ghost" size="sm"
                  className="flex-1 text-xs text-[#94a3b8] hover:text-[#69F0AE] hover:bg-[#69F0AE10]"
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" /> Export
                </Button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </>
  );
}
