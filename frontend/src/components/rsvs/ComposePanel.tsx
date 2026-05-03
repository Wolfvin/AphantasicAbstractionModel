'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Hexagon, Atom, Plus, X, CheckCircle2, ChevronRight, Layers,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { useUIStore, useGraphStore } from '@/store/rsvsStore';
import type { ComposeResult, RSVSNode, Tier, CompositionPair } from '@/lib/types';

// ── Tier Colors ──
const TIER_COLORS: Record<Tier, string> = {
  1: '#00E5FF',
  2: '#FFB74D',
  3: '#FF5252',
};

// ── Compose Result Panel (shown in RightNodeDrawer) ──
interface ComposePanelProps {
  result?: ComposeResult;
  className?: string;
}

export default function ComposePanel({ result, className }: ComposePanelProps) {
  const selectNode = useUIStore((s) => s.selectNode);
  const focusNode = useUIStore((s) => s.focusNode);

  const handleSelectNode = (id: number) => {
    selectNode(id);
    focusNode(id);
  };

  // If we have a result, show it
  if (result) {
    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className={className}
        >
          <Card className="bg-[#0a0e18] border-[#1e293b] shadow-xl">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Hexagon className="w-5 h-5 text-[#FF80AB]" />
                <CardTitle className="text-sm font-bold text-[#e2e8f0]">Compose Result</CardTitle>
              </div>
            </CardHeader>

            <CardContent className="pt-0 space-y-4">
              {/* New composite node */}
              <div>
                <div className="text-[10px] uppercase tracking-wider text-[#FF80AB] mb-1.5 flex items-center gap-1.5">
                  <Hexagon className="w-3 h-3" />
                  New Composite
                </div>
                <button
                  type="button"
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-md bg-[#FF80AB10] border border-[#FF80AB30] hover:bg-[#FF80AB20] transition-colors text-left"
                  onClick={() => handleSelectNode(result.composite_node.id)}
                >
                  <span className="text-sm text-[#FF80AB] font-mono font-bold">{result.composite_node.label}</span>
                  <span className="text-[10px] text-[#64748b] font-mono ml-auto">#{result.composite_node.id}</span>
                  {/* v5.0: Layer info */}
                  {result.composite_node.layer !== undefined && (
                    <Badge className="text-[9px] px-1 py-0 shrink-0 bg-[#66BB6A15] text-[#66BB6A] border-transparent">
                      L{result.composite_node.layer}
                    </Badge>
                  )}
                  <ChevronRight className="w-3.5 h-3.5 text-[#475569]" />
                </button>
              </div>

              <Separator className="bg-[#1e293b]" />

              {/* v5.0: Compositions used (if available) */}
              {result.compositions && result.compositions.length > 0 && (
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-[#AB47BC] mb-1.5 flex items-center gap-1.5">
                    <Layers className="w-3 h-3" />
                    Compositions ({result.compositions.length})
                  </div>
                  <div className="space-y-1">
                    {result.compositions.map((comp, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-[#0d1520] border border-[#1e293b]"
                      >
                        <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">{comp.label}</span>
                        <span className="text-[10px] text-[#64748b] font-mono shrink-0">{comp.sense_id}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Atom nodes */}
              {result.atom_nodes.length > 0 && (
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-[#69F0AE] mb-1.5 flex items-center gap-1.5">
                    <Atom className="w-3 h-3" />
                    Atom Nodes ({result.atom_nodes.length})
                  </div>
                  <div className="space-y-1">
                    {result.atom_nodes.map((atom) => (
                      <button
                        key={atom.id}
                        type="button"
                        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-[#1e293b]/50 transition-colors text-left group"
                        onClick={() => handleSelectNode(atom.id)}
                      >
                        <Atom className="w-3 h-3 text-[#69F0AE] shrink-0" />
                        <span className="text-xs text-[#e2e8f0] font-mono truncate">{atom.label}</span>
                        <Badge
                          className="text-[9px] px-1 py-0 shrink-0"
                          style={{
                            backgroundColor: `${TIER_COLORS[atom.tier]}15`,
                            color: TIER_COLORS[atom.tier],
                            borderColor: 'transparent',
                          }}
                        >
                          T{atom.tier}
                        </Badge>
                        <ChevronRight className="w-3 h-3 text-[#475569] group-hover:text-[#00E5FF] transition-colors shrink-0" />
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Jaccard similarities */}
              {result.jaccard_similarities.length > 0 && (
                <>
                  <Separator className="bg-[#1e293b]" />
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-[#00BCD4] mb-1.5">
                      Shared Atoms (Jaccard Similarity)
                    </div>
                    <div className="space-y-1">
                      {result.jaccard_similarities.map((item, i) => (
                        <button
                          key={i}
                          type="button"
                          className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-[#1e293b]/50 transition-colors text-left group"
                          onClick={() => handleSelectNode(item.composite_id)}
                        >
                          <Hexagon className="w-3 h-3 text-[#FF80AB] shrink-0" />
                          <span className="text-xs text-[#e2e8f0] font-mono truncate">{item.composite_label}</span>
                          <div className="w-16 h-1.5 bg-[#0d1520] rounded-full overflow-hidden shrink-0 ml-auto">
                            <motion.div
                              className="h-full rounded-full bg-[#00BCD4]"
                              initial={{ width: 0 }}
                              animate={{ width: `${item.similarity * 100}%` }}
                              transition={{ duration: 0.4 }}
                            />
                          </div>
                          <span className="text-[10px] text-[#00BCD4] font-mono shrink-0 w-8 text-right">
                            {item.similarity.toFixed(2)}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </AnimatePresence>
    );
  }

  return null;
}

// ── Compose Form Panel (shown in LeftInputRail) ──
interface ComposeFormPanelProps {
  onCompose: (label: string, atomIds: number[], lang: string) => void;
  onCancel: () => void;
  isLoading?: boolean;
}

export function ComposeFormPanel({ onCompose, onCancel, isLoading }: ComposeFormPanelProps) {
  const nodes = useGraphStore((s) => s.nodes);
  const [label, setLabel] = useState('');
  const [selectedAtomIds, setSelectedAtomIds] = useState<Set<number>>(new Set());
  const [lang, setLang] = useState('id'); // default Indonesian
  const [searchQuery, setSearchQuery] = useState('');

  // Get available atom nodes (non-composite nodes)
  const availableAtoms = useMemo(() => {
    return Array.from(nodes.values()).filter((n) => {
      // Show nodes that are atoms (or not clearly composites)
      const cs = n.compression_state ?? n.semantic?.compression_state ?? 'raw';
      const derived = n.derived_from_node_ids ?? n.semantic?.derived_from_node_ids ?? [];
      const isComp = cs === 'compressed' || cs === 'composed' || derived.length > 0 || (n.atoms && n.atoms.length > 0);
      // Filter by search query
      if (searchQuery && !n.label.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      return true; // show all nodes as potential atoms
    }).sort((a, b) => a.label.localeCompare(b.label));
  }, [nodes, searchQuery]);

  const toggleAtom = useCallback((id: number) => {
    setSelectedAtomIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handleCompose = useCallback(() => {
    if (!label.trim() || selectedAtomIds.size === 0) return;
    onCompose(label.trim(), Array.from(selectedAtomIds), lang);
  }, [label, selectedAtomIds, lang, onCompose]);

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.2 }}
      className="overflow-hidden"
    >
      <div className="p-3 bg-[#0d1520] border border-[#1e293b] rounded-xl space-y-3">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Hexagon className="w-4 h-4 text-[#FF80AB]" />
            <span className="text-sm font-semibold text-[#e2e8f0]">Compose Node</span>
          </div>
          <button
            type="button"
            className="p-1 rounded-md hover:bg-white/5 text-[#64748b] hover:text-[#e2e8f0] transition-colors"
            onClick={onCancel}
            aria-label="Cancel compose"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Label input */}
        <div>
          <label className="text-[10px] uppercase tracking-wider text-[#94a3b8] mb-1 block">Composite Label</label>
          <Input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="e.g., raja, ratu"
            className="bg-[#0a0e18] border-[#1e293b] text-[#e2e8f0] text-sm font-mono placeholder:text-[#475569]"
          />
        </div>

        {/* Language selector */}
        <div>
          <label className="text-[10px] uppercase tracking-wider text-[#94a3b8] mb-1 block">Language</label>
          <div className="flex gap-1.5">
            {['id', 'en', 'jv', 'su'].map((l) => (
              <button
                key={l}
                type="button"
                className={`px-2.5 py-1 rounded-md text-xs font-mono transition-colors ${
                  lang === l
                    ? 'bg-[#FF80AB20] text-[#FF80AB] border border-[#FF80AB40]'
                    : 'bg-[#0a0e18] text-[#64748b] border border-[#1e293b] hover:text-[#94a3b8]'
                }`}
                onClick={() => setLang(l)}
              >
                {l.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Atom selection */}
        <div>
          <label className="text-[10px] uppercase tracking-wider text-[#94a3b8] mb-1 block">
            Select Atoms ({selectedAtomIds.size} selected)
          </label>
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search nodes..."
            className="bg-[#0a0e18] border-[#1e293b] text-[#e2e8f0] text-xs font-mono placeholder:text-[#475569] mb-2 h-8"
          />
          <ScrollArea className="max-h-40">
            <div className="space-y-0.5">
              {availableAtoms.slice(0, 20).map((n) => (
                <button
                  key={n.id}
                  type="button"
                  className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-colors ${
                    selectedAtomIds.has(n.id)
                      ? 'bg-[#FF80AB15] border border-[#FF80AB30]'
                      : 'hover:bg-[#1e293b]/50 border border-transparent'
                  }`}
                  onClick={() => toggleAtom(n.id)}
                >
                  <div className={`w-3 h-3 rounded-sm border flex items-center justify-center shrink-0 ${
                    selectedAtomIds.has(n.id)
                      ? 'bg-[#FF80AB] border-[#FF80AB]'
                      : 'border-[#334155]'
                  }`}>
                    {selectedAtomIds.has(n.id) && (
                      <CheckCircle2 className="w-2.5 h-2.5 text-white" />
                    )}
                  </div>
                  <span className="text-xs text-[#e2e8f0] font-mono truncate">{n.label}</span>
                  <Badge
                    className="text-[9px] px-1 py-0 shrink-0 ml-auto"
                    style={{
                      backgroundColor: `${TIER_COLORS[n.tier]}15`,
                      color: TIER_COLORS[n.tier],
                      borderColor: 'transparent',
                    }}
                  >
                    T{n.tier}
                  </Badge>
                </button>
              ))}
              {availableAtoms.length > 20 && (
                <p className="text-[10px] text-[#475569] px-2 py-1">
                  +{availableAtoms.length - 20} more nodes...
                </p>
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            className="flex-1 text-xs text-[#64748b] hover:text-[#e2e8f0] hover:bg-[#1e293b]"
            onClick={onCancel}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            className="flex-1 text-xs bg-[#FF80AB15] text-[#FF80AB] border border-[#FF80AB30] hover:bg-[#FF80AB25] disabled:opacity-40"
            disabled={!label.trim() || selectedAtomIds.size === 0 || isLoading}
            onClick={handleCompose}
          >
            {isLoading ? (
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 border-2 border-[#FF80AB40] border-t-[#FF80AB] rounded-full animate-spin" />
                Composing...
              </span>
            ) : (
              <span className="flex items-center gap-1.5">
                <Plus className="w-3.5 h-3.5" />
                Compose
              </span>
            )}
          </Button>
        </div>

        {/* Inline compose hint */}
        <p className="text-[10px] text-[#475569] text-center">
          Or type: <span className="text-[#FF80AB] font-mono">/compose label = atom1 + atom2 + atom3</span>
        </p>
      </div>
    </motion.div>
  );
}
