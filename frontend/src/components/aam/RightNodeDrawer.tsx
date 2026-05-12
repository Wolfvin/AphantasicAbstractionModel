'use client';

import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, Crosshair, Pin, GitCompare, Download, Brain,
  Link2, Activity, Eye, Sparkles, AlertTriangle,
  Shield, Layers, Globe, GitBranch, ArrowUpDown,
  Atom, Hexagon, Shuffle, BarChart3, ArrowLeftRight,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useUIStore, useGraphStore, useModeResultStore } from '@/store/aamStore';
import type { RSVSNode, Tier, PolicyMeta, LanguageLink, StructuralSimilarityResult, SubstitutionAnalysisResult, CompositionReference, SubstitutionPairInfo, ConvergenceLinkInfo, SenseEntry, GroundingEvidence } from '@/lib/types';
import { getStatusColor, isCompositeNode, isAtomNode, getAtomCount, computeNodeLayer, getLayerColor, getLayerLabel, buildCompositionChain } from '@/lib/nodeRendering';
import { NUMERIC_TIER_COLORS as TIER_COLORS, NUMERIC_TIER_LABELS as TIER_LABELS } from '@/lib/constants';
import AppraisePanel from '@/components/aam/AppraisePanel';
import RelatePanel from '@/components/aam/RelatePanel';
import ComposePanel from '@/components/aam/ComposePanel';

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
  const label = state === 'composed' ? 'Composed' : state === 'compressed' ? 'Compressed' : 'Raw';
  const color = state === 'composed' ? '#FF80AB' : state === 'compressed' ? '#B388FF' : '#69F0AE';
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <Layers className="w-3.5 h-3.5 text-[#FFB74D]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Compression State</h3>
      </div>
      <div className="bg-[#0d1520] rounded-lg p-3 text-center border border-[#1e293b]">
        <div className="text-lg font-bold" style={{ color }}>{label}</div>
        <div className="text-[10px] uppercase tracking-wider text-[#64748b]">State</div>
      </div>
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
              {link.link_type}
            </Badge>
            <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">→ #{link.target_id}</span>
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

// ── v5.0: Layer Info Section ──
function LayerInfoSection({ node }: { node: RSVSNode }) {
  const layer = computeNodeLayer(node);
  const layerColor = getLayerColor(layer);
  const chain = buildCompositionChain(
    node.label,
    node.compositions,
    node.derived_from_node_ids ?? node.semantic?.derived_from_node_ids,
    useGraphStore.getState().nodes as Map<number, { label: string }>,
  );

  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <Layers className="w-3.5 h-3.5" style={{ color: layerColor }} />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Compositional Layer</h3>
      </div>
      <div className="grid grid-cols-2 gap-2 mb-2">
        <div className="bg-[#0d1520] rounded-lg p-2 text-center border border-[#1e293b]">
          <div className="text-lg font-bold" style={{ color: layerColor }}>L{layer}</div>
          <div className="text-[9px] uppercase tracking-wider text-[#64748b]">{layer === 0 ? 'Primitive' : 'Composed'}</div>
        </div>
        {node.grounding_score !== undefined && (
          <div className="bg-[#0d1520] rounded-lg p-2 text-center border border-[#1e293b]">
            <div className="text-lg font-bold text-[#80D8FF]">{(node.grounding_score * 100).toFixed(0)}%</div>
            <div className="text-[9px] uppercase tracking-wider text-[#64748b]">Grounding</div>
          </div>
        )}
      </div>
      {/* Composition chain */}
      {chain && (
        <div className="bg-[#0d1520] rounded-lg p-3 border border-[#1e293b]">
          <div className="text-[9px] uppercase tracking-wider text-[#64748b] mb-1.5">Composition Chain</div>
          <div className="text-sm font-mono font-bold" style={{ color: layerColor }}>{chain}</div>
        </div>
      )}
    </div>
  );
}

// ── v5.0: Structural Similarity Section ──
function StructuralSimilaritySection({ result, onSelectNode }: { result: StructuralSimilarityResult; onSelectNode: (id: number) => void }) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-3">
        <GitCompare className="w-3.5 h-3.5 text-[#00BCD4]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Structural Similarity</h3>
      </div>

      {/* Comparison header */}
      <div className="flex items-center justify-center gap-3 mb-3">
        <button
          type="button"
          className="px-3 py-1.5 rounded-md bg-[#42A5F510] border border-[#42A5F530] text-[#42A5F5] text-sm font-mono font-bold hover:bg-[#42A5F520] transition-colors"
          onClick={() => onSelectNode(result.node_a.id)}
        >
          {result.node_a.label}
        </button>
        <ArrowLeftRight className="w-4 h-4 text-[#64748b]" />
        <button
          type="button"
          className="px-3 py-1.5 rounded-md bg-[#42A5F510] border border-[#42A5F530] text-[#42A5F5] text-sm font-mono font-bold hover:bg-[#42A5F520] transition-colors"
          onClick={() => onSelectNode(result.node_b.id)}
        >
          {result.node_b.label}
        </button>
      </div>

      {/* Similarity score */}
      <div className="bg-[#0d1520] rounded-lg p-3 text-center border border-[#1e293b] mb-3">
        <div className="text-2xl font-bold text-[#00BCD4]">{(result.similarity_score * 100).toFixed(1)}%</div>
        <div className="text-[9px] uppercase tracking-wider text-[#64748b]">Similarity Score</div>
      </div>

      {/* Shared compositions */}
      {result.shared_compositions.length > 0 && (
        <div className="mb-3">
          <div className="text-[10px] uppercase tracking-wider text-[#69F0AE] mb-1.5 flex items-center gap-1.5">
            <Link2 className="w-3 h-3" />
            Shared Compositions ({result.shared_compositions.length})
          </div>
          <div className="space-y-1">
            {result.shared_compositions.map((comp, i) => (
              <div key={i} className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-[#0d1520] border border-[#1e293b]">
                <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">{comp.label}</span>
                <div className="w-16 h-1.5 bg-[#0a0e18] rounded-full overflow-hidden shrink-0">
                  <div
                    className="h-full rounded-full bg-[#69F0AE]"
                    style={{ width: `${comp.similarity * 100}%` }}
                  />
                </div>
                <span className="text-[10px] text-[#69F0AE] font-mono shrink-0 w-8 text-right">
                  {comp.similarity.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Differing compositions */}
      {result.differing_compositions.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-[#FFB74D] mb-1.5 flex items-center gap-1.5">
            <Shuffle className="w-3 h-3" />
            Differing Compositions ({result.differing_compositions.length})
          </div>
          <div className="space-y-1">
            {result.differing_compositions.map((comp, i) => (
              <div key={i} className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-[#0d1520] border border-[#1e293b]">
                <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">{comp.label}</span>
                <Badge
                  className="text-[9px] px-1 py-0 shrink-0"
                  style={{
                    backgroundColor: comp.present_in === 'a' ? '#42A5F515' : '#66BB6A15',
                    color: comp.present_in === 'a' ? '#42A5F5' : '#66BB6A',
                    borderColor: 'transparent',
                  }}
                >
                  {comp.present_in === 'a' ? result.node_a.label : result.node_b.label}
                </Badge>
                <span className="text-[10px] text-[#FFB74D] font-mono shrink-0">
                  {comp.weight.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── v5.0: Substitution Analysis Section ──
function SubstitutionAnalysisSection({ result, onSelectNode }: { result: SubstitutionAnalysisResult; onSelectNode: (id: number) => void }) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-3">
        <Shuffle className="w-3.5 h-3.5 text-[#AB47BC]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Substitution Analysis</h3>
      </div>

      {/* Comparison header */}
      <div className="flex items-center justify-center gap-3 mb-3">
        <button
          type="button"
          className="px-3 py-1.5 rounded-md bg-[#42A5F510] border border-[#42A5F530] text-[#42A5F5] text-sm font-mono font-bold hover:bg-[#42A5F520] transition-colors"
          onClick={() => onSelectNode(result.node_a.id)}
        >
          {result.node_a.label}
        </button>
        <ArrowLeftRight className="w-4 h-4 text-[#64748b]" />
        <button
          type="button"
          className="px-3 py-1.5 rounded-md bg-[#66BB6A10] border border-[#66BB6A30] text-[#66BB6A] text-sm font-mono font-bold hover:bg-[#66BB6A20] transition-colors"
          onClick={() => onSelectNode(result.node_b.id)}
        >
          {result.node_b.label}
        </button>
      </div>

      {/* Substitution pairs */}
      {result.substitution_pairs.length > 0 ? (
        <div className="space-y-2">
          {result.substitution_pairs.map((pair, i) => (
            <div key={i} className="bg-[#0d1520] rounded-lg p-3 border border-[#1e293b]">
              <div className="flex items-center justify-center gap-2 mb-2">
                <button
                  type="button"
                  className="px-2 py-1 rounded-md bg-[#42A5F510] border border-[#42A5F520] text-[#42A5F5] text-xs font-mono hover:bg-[#42A5F520] transition-colors"
                  onClick={() => onSelectNode(pair.atom_a.id)}
                >
                  {pair.atom_a.label}
                </button>
                <ArrowLeftRight className="w-3 h-3 text-[#AB47BC]" />
                <button
                  type="button"
                  className="px-2 py-1 rounded-md bg-[#66BB6A10] border border-[#66BB6A20] text-[#66BB6A] text-xs font-mono hover:bg-[#66BB6A20] transition-colors"
                  onClick={() => onSelectNode(pair.atom_b.id)}
                >
                  {pair.atom_b.label}
                </button>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <div className="w-20 h-1.5 bg-[#0a0e18] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-[#AB47BC]"
                      style={{ width: `${pair.substitution_score * 100}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-[#AB47BC] font-mono">{pair.substitution_score.toFixed(2)}</span>
                </div>
              </div>
              <div className="text-[10px] text-[#94a3b8] mt-1.5 font-mono italic">{pair.semantic_shift}</div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-[#475569] italic">No substitution pairs found</p>
      )}
    </div>
  );
}

// ── F-02: Composition References Section ──
function CompositionReferencesSection({ refs, onSelectNode }: { refs: CompositionReference[]; onSelectNode: (id: number) => void }) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <GitBranch className="w-3.5 h-3.5 text-[#00BCD4]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Composition References</h3>
        <Badge className="text-[9px] ml-auto bg-[#00BCD415] text-[#00BCD4] border-[#00BCD430] px-1.5 py-0">
          {refs.length}
        </Badge>
      </div>
      <div className="space-y-1">
        {refs.map((ref, i) => (
          <button
            key={i}
            type="button"
            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-[#1e293b]/50 transition-colors text-left"
            onClick={() => onSelectNode(ref.ref_node_id)}
          >
            <GitBranch className="w-3 h-3 text-[#00BCD4] shrink-0" />
            <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">{ref.ref_label}</span>
            <div className="w-12 h-1.5 bg-[#0d1520] rounded-full overflow-hidden shrink-0">
              <div
                className="h-full rounded-full bg-[#00BCD4]"
                style={{ width: `${ref.weight * 100}%` }}
              />
            </div>
            <span className="text-[10px] text-[#64748b] font-mono shrink-0">{ref.weight.toFixed(2)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── F-02: Multi-Sense Information Section ──
function MultiSenseSection({ node }: { node: RSVSNode }) {
  const senses = node.sense?.senses;
  const senseCount = node.sense?.count ?? 0;
  if (!senses || senses.length === 0) return null;

  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <Eye className="w-3.5 h-3.5 text-[#80D8FF]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Multi-Sense Information</h3>
        <Badge className="text-[9px] ml-auto bg-[#80D8FF15] text-[#80D8FF] border-[#80D8FF30] px-1.5 py-0">
          {senseCount} sense{senseCount !== 1 ? 's' : ''}
        </Badge>
      </div>
      <div className="space-y-2">
        {senses.map((sense) => {
          const statusColor = sense.status === 'mature' ? '#69F0AE' : sense.status === 'fragile' ? '#FFB74D' : sense.status === 'emerging' ? '#00E5FF' : '#FF5252';
          return (
            <div key={sense.sense_id} className="bg-[#0d1520] rounded-lg p-2.5 border border-[#1e293b]">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-mono font-bold text-[#e2e8f0]">S{sense.sense_id}: {sense.label}</span>
                <Badge
                  className="text-[9px] px-1 py-0 shrink-0"
                  style={{
                    backgroundColor: `${statusColor}15`,
                    color: statusColor,
                    borderColor: 'transparent',
                  }}
                >
                  {sense.status}
                </Badge>
              </div>
              {/* Composition atoms */}
              {sense.composition.length > 0 && (
                <div className="text-[10px] text-[#64748b] font-mono mb-1">
                  atoms: {sense.composition.join(' + ')}
                </div>
              )}
              {/* Confidence & Coherence bars */}
              <div className="flex gap-3">
                <div className="flex-1">
                  <div className="flex justify-between text-[9px] mb-0.5">
                    <span className="text-[#64748b]">Confidence</span>
                    <span className="text-[#94a3b8] font-mono">{sense.confidence.toFixed(2)}</span>
                  </div>
                  <div className="h-1 bg-[#0a0e18] rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-[#69F0AE]" style={{ width: `${sense.confidence * 100}%` }} />
                  </div>
                </div>
                <div className="flex-1">
                  <div className="flex justify-between text-[9px] mb-0.5">
                    <span className="text-[#64748b]">Coherence</span>
                    <span className="text-[#94a3b8] font-mono">{sense.coherence.toFixed(2)}</span>
                  </div>
                  <div className="h-1 bg-[#0a0e18] rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-[#80D8FF]" style={{ width: `${sense.coherence * 100}%` }} />
                  </div>
                </div>
                {sense.grounding_score !== undefined && (
                  <div className="flex-1">
                    <div className="flex justify-between text-[9px] mb-0.5">
                      <span className="text-[#64748b]">Grounding</span>
                      <span className="text-[#94a3b8] font-mono">{(sense.grounding_score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1 bg-[#0a0e18] rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          backgroundColor: sense.grounding_score > 0.7 ? '#69F0AE' : sense.grounding_score > 0.4 ? '#FFB74D' : '#FF5252',
                          width: `${sense.grounding_score * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── F-02: Substitution Pairs Section ──
function SubstitutionPairsSection({ pairs, onSelectNode }: { pairs: SubstitutionPairInfo[]; onSelectNode: (id: number) => void }) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <Shuffle className="w-3.5 h-3.5 text-[#FFB74D]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Substitution Pairs</h3>
        <Badge className="text-[9px] ml-auto bg-[#FFB74D15] text-[#FFB74D] border-[#FFB74D30] px-1.5 py-0">
          {pairs.length}
        </Badge>
      </div>
      <div className="space-y-2">
        {pairs.map((pair, i) => (
          <div key={i} className="bg-[#0d1520] rounded-lg p-2.5 border border-[#1e293b]">
            <div className="flex items-center justify-center gap-2 mb-1.5">
              <button
                type="button"
                className="px-2 py-0.5 rounded-md bg-[#42A5F510] border border-[#42A5F520] text-[#42A5F5] text-xs font-mono hover:bg-[#42A5F520] transition-colors"
                onClick={() => onSelectNode(pair.atom_a_id)}
              >
                {pair.atom_a_label}
              </button>
              <ArrowLeftRight className="w-3 h-3 text-[#FFB74D]" />
              <button
                type="button"
                className="px-2 py-0.5 rounded-md bg-[#66BB6A10] border border-[#66BB6A20] text-[#66BB6A] text-xs font-mono hover:bg-[#66BB6A20] transition-colors"
                onClick={() => onSelectNode(pair.atom_b_id)}
              >
                {pair.atom_b_label}
              </button>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-[#0a0e18] rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-[#FFB74D]"
                  style={{ width: `${pair.substitution_score * 100}%` }}
                />
              </div>
              <span className="text-[10px] text-[#FFB74D] font-mono shrink-0">{pair.substitution_score.toFixed(2)}</span>
            </div>
            {/* Highlighted semantic shift */}
            <div className="text-[10px] text-[#94a3b8] mt-1 font-mono italic">
              shift: {pair.semantic_shift}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── F-02: Convergence Links Section ──
function ConvergenceLinksSection({ links, onSelectNode }: { links: ConvergenceLinkInfo[]; onSelectNode: (id: number) => void }) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <GitCompare className="w-3.5 h-3.5 text-[#E040FB]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Convergence Links</h3>
        <Badge className="text-[9px] ml-auto bg-[#E040FB15] text-[#E040FB] border-[#E040FB30] px-1.5 py-0">
          {links.length}
        </Badge>
      </div>
      <div className="space-y-1">
        {links.map((link, i) => (
          <button
            key={i}
            type="button"
            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-[#1e293b]/50 transition-colors text-left"
            onClick={() => onSelectNode(link.target_id)}
          >
            <GitCompare className="w-3 h-3 text-[#E040FB] shrink-0" />
            <Badge variant="outline" className="text-[9px] border-[#E040FB40] text-[#E040FB] shrink-0 px-1">
              {link.link_type}
            </Badge>
            <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">↔ {link.target_label}</span>
            <div className="w-12 h-1.5 bg-[#0d1520] rounded-full overflow-hidden shrink-0">
              <div
                className="h-full rounded-full bg-[#E040FB]"
                style={{ width: `${link.strength * 100}%` }}
              />
            </div>
            <span className="text-[10px] text-[#64748b] font-mono shrink-0">{link.strength.toFixed(2)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── F-02: Grounding Evidence Section ──
function GroundingEvidenceSection({ evidence }: { evidence: GroundingEvidence }) {
  const verdictColor = evidence.verdict === 'well_grounded' ? '#69F0AE' : evidence.verdict === 'needs_review' ? '#FFB74D' : '#FF5252';
  const verdictLabel = evidence.verdict === 'well_grounded' ? 'Well Grounded' : evidence.verdict === 'needs_review' ? 'Needs Review' : 'Needs Revision';

  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-2">
        <Shield className="w-3.5 h-3.5" style={{ color: verdictColor }} />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Grounding Evidence</h3>
        <Badge
          className="text-[9px] ml-auto px-1.5 py-0"
          style={{
            backgroundColor: `${verdictColor}15`,
            color: verdictColor,
            borderColor: `${verdictColor}40`,
          }}
        >
          {verdictLabel}
        </Badge>
      </div>
      <div className="grid grid-cols-2 gap-2 mb-2">
        <StatCard label="Confirming" value={evidence.confirming_contexts} />
        <StatCard label="Contradicting" value={evidence.contradicting_contexts} />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-[#0d1520] rounded-lg p-2 text-center border border-[#1e293b]">
          <div className="text-sm font-bold" style={{ color: verdictColor }}>{(evidence.score * 100).toFixed(0)}%</div>
          <div className="text-[9px] uppercase tracking-wider text-[#64748b]">Score</div>
        </div>
        <div className="bg-[#0d1520] rounded-lg p-2 text-center border border-[#1e293b]">
          <div className="text-sm font-bold text-[#e2e8f0]">{evidence.revision_count}</div>
          <div className="text-[9px] uppercase tracking-wider text-[#64748b]">Revisions</div>
        </div>
      </div>
      {evidence.last_contradiction && (
        <div className="mt-2 text-[10px] text-[#FF5252] font-mono italic">
          Last contradiction: {evidence.last_contradiction}
        </div>
      )}
    </div>
  );
}

// ── Composition Section (new for compose visualization) ──
function CompositionSection({ node, onSelectNode }: { node: RSVSNode; onSelectNode: (id: number) => void }) {
  const nodes = useGraphStore((s) => s.nodes);
  const getAtomNodes = useGraphStore((s) => s.getAtomNodes);
  const getCompositeNodesForAtom = useGraphStore((s) => s.getCompositeNodesForAtom);
  const computeJaccardSimilarity = useGraphStore((s) => s.computeJaccardSimilarity);

  // Collect atom IDs from any source
  const atomIds = useMemo(() => {
    const ids: number[] = [];
    if (node.atoms && node.atoms.length > 0) ids.push(...node.atoms);
    else if (node.derived_from_node_ids && node.derived_from_node_ids.length > 0) ids.push(...node.derived_from_node_ids);
    else if (node.semantic?.derived_from_node_ids && node.semantic.derived_from_node_ids.length > 0) ids.push(...node.semantic.derived_from_node_ids);
    else if (node.composition?.atoms) ids.push(...node.composition.atoms.map(a => a.atom_id));
    return ids;
  }, [node]);

  // Resolve atom nodes
  const atomNodes = useMemo(() => {
    return atomIds
      .map(id => nodes.get(id))
      .filter((n): n is RSVSNode => n !== undefined);
  }, [atomIds, nodes]);

  // Find composites that share atoms with this one (for Jaccard similarity)
  const similarComposites = useMemo(() => {
    if (atomIds.length === 0) return [];
    const compositeIds = new Set<number>();
    for (const aid of atomIds) {
      const composites = getCompositeNodesForAtom(aid);
      for (const c of composites) {
        if (c.id !== node.id) compositeIds.add(c.id);
      }
    }
    return Array.from(compositeIds)
      .map(id => {
        const cNode = nodes.get(id);
        if (!cNode) return null;
        const similarity = computeJaccardSimilarity(node.id, id);
        return { node: cNode, similarity };
      })
      .filter((x): x is { node: RSVSNode; similarity: number } => x !== null && x.similarity > 0)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5);
  }, [atomIds, node.id, nodes, getCompositeNodesForAtom, computeJaccardSimilarity]);

  return (
    <div className="mb-5">
      <div className="flex items-center gap-2 mb-3">
        <Hexagon className="w-3.5 h-3.5 text-[#FF80AB]" />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">Composition</h3>
        <Badge className="text-[9px] ml-auto bg-[#FF80AB15] text-[#FF80AB] border-[#FF80AB30] px-1.5 py-0">
          {atomIds.length} atom{atomIds.length !== 1 ? 's' : ''}
        </Badge>
      </div>

      {/* Atom list */}
      {atomNodes.length > 0 ? (
        <div className="space-y-1 mb-3">
          {atomNodes.map((atomNode) => (
            <button
              key={atomNode.id}
              type="button"
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-[#1e293b]/50 transition-colors text-left group"
              onClick={() => onSelectNode(atomNode.id)}
            >
              <Atom className="w-3 h-3 text-[#69F0AE] shrink-0" />
              <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">{atomNode.label}</span>
              <Badge
                className="text-[9px] px-1 py-0 shrink-0"
                style={{
                  backgroundColor: `${TIER_COLORS[atomNode.tier]}15`,
                  color: TIER_COLORS[atomNode.tier],
                  borderColor: 'transparent',
                }}
              >
                T{atomNode.tier}
              </Badge>
              <span className="text-[10px] text-[#64748b] font-mono shrink-0">
                c={atomNode.confidence.toFixed(2)}
              </span>
              <GitBranch className="w-3 h-3 text-[#475569] group-hover:text-[#00E5FF] transition-colors shrink-0" />
            </button>
          ))}
        </div>
      ) : (
        <p className="text-xs text-[#475569] italic mb-3">No atom data available</p>
      )}

      {/* Jaccard similarities */}
      {similarComposites.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-[#80D8FF] mb-2 flex items-center gap-1.5">
            <GitCompare className="w-3 h-3" />
            Shared Atoms (Jaccard)
          </div>
          <div className="space-y-1">
            {similarComposites.map(({ node: cNode, similarity }) => (
              <button
                key={cNode.id}
                type="button"
                className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-[#1e293b]/50 transition-colors text-left group"
                onClick={() => onSelectNode(cNode.id)}
              >
                <Hexagon className="w-3 h-3 text-[#FF80AB] shrink-0" />
                <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">{cNode.label}</span>
                <div className="w-16 h-1.5 bg-[#0d1520] rounded-full overflow-hidden shrink-0">
                  <motion.div
                    className="h-full rounded-full bg-[#00BCD4]"
                    initial={{ width: 0 }}
                    animate={{ width: `${similarity * 100}%` }}
                    transition={{ duration: 0.4 }}
                  />
                </div>
                <span className="text-[10px] text-[#00BCD4] font-mono shrink-0 w-8 text-right">
                  {similarity.toFixed(2)}
                </span>
                <GitBranch className="w-3 h-3 text-[#475569] group-hover:text-[#00E5FF] transition-colors shrink-0" />
              </button>
            ))}
          </div>
        </div>
      )}
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

  // Detect composite/atom
  const isComposite = node ? isCompositeNode(node) : false;
  const isAtom = node ? isAtomNode(node) : false;

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
                  {isComposite ? (
                    <Hexagon className="w-5 h-5 text-[#FF80AB] shrink-0" />
                  ) : isAtom ? (
                    <Atom className="w-5 h-5 text-[#69F0AE] shrink-0" />
                  ) : (
                    <Brain className="w-5 h-5 text-[#00E5FF] shrink-0" />
                  )}
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
                {isComposite && (
                  <Badge className="text-[10px] bg-[#FF80AB15] text-[#FF80AB] border-[#FF80AB40]">
                    ◆ Composite
                  </Badge>
                )}
                {isAtom && (
                  <Badge className="text-[10px] bg-[#69F0AE15] text-[#69F0AE] border-[#69F0AE40]">
                    ● Atom
                  </Badge>
                )}
                {/* v5.0: Layer badge */}
                {node.layer !== undefined && (
                  <Badge
                    className="text-[10px]"
                    style={{
                      backgroundColor: `${getLayerColor(node.layer)}15`,
                      color: getLayerColor(node.layer),
                      borderColor: `${getLayerColor(node.layer)}40`,
                    }}
                  >
                    L{node.layer} {node.layer === 0 ? 'Primitive' : 'Composed'}
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

              {/* v5.0: Compositional Layer Info (always show for any node) */}
              <>
                <LayerInfoSection node={node} />
                <Separator className="bg-[#1e293b] mb-5" />
              </>

              {/* Composition Section (for composite nodes) */}
              {isComposite && (
                <>
                  <CompositionSection node={node} onSelectNode={handleSelectNode} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* F-02: Composition References */}
              {node.composition_references && node.composition_references.length > 0 && (
                <>
                  <CompositionReferencesSection refs={node.composition_references} onSelectNode={handleSelectNode} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* F-02: Multi-Sense Information (enhanced sense display) */}
              {node.sense?.senses && node.sense.senses.length > 1 && (
                <>
                  <MultiSenseSection node={node} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* F-02: Grounding Evidence */}
              {node.grounding_evidence && (
                <>
                  <GroundingEvidenceSection evidence={node.grounding_evidence} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* F-02: Substitution Pairs */}
              {node.substitution_pairs && node.substitution_pairs.length > 0 && (
                <>
                  <SubstitutionPairsSection pairs={node.substitution_pairs} onSelectNode={handleSelectNode} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* F-02: Convergence Links */}
              {node.convergence_links && node.convergence_links.length > 0 && (
                <>
                  <ConvergenceLinksSection links={node.convergence_links} onSelectNode={handleSelectNode} />
                  <Separator className="bg-[#1e293b] mb-5" />
                </>
              )}

              {/* v4.2: Compression State */}
              {(node.compression_state === 'compressed' || node.compression_state === 'composed') && (
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

              {/* Composition (legacy) */}
              {node.composition && !isComposite && (
                <div className="mb-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Activity className="w-3.5 h-3.5 text-[#B388FF]" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#94a3b8]">
                      {node.kind === 'node' ? 'Related Composites' : 'Member Nodes'}
                    </h3>
                  </div>
                  {node.composition.related_composites.map((item, i) => {
                    const id: number = 'atom_id' in item ? (item.atom_id as number) : (item.composite_id as number);
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

              {/* Mode Result Panels (Appraise / Relate / Compose / Structural Similarity / Substitution) */}
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
              {modeResult && modeResult.type === 'compose' && (
                <>
                  <Separator className="bg-[#1e293b] mb-5" />
                  <ComposePanel result={modeResult.data} />
                </>
              )}
              {modeResult && modeResult.type === 'structural_similarity' && (
                <>
                  <Separator className="bg-[#1e293b] mb-5" />
                  <StructuralSimilaritySection result={modeResult.data} onSelectNode={handleSelectNode} />
                </>
              )}
              {modeResult && modeResult.type === 'substitution_analysis' && (
                <>
                  <Separator className="bg-[#1e293b] mb-5" />
                  <SubstitutionAnalysisSection result={modeResult.data} onSelectNode={handleSelectNode} />
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
