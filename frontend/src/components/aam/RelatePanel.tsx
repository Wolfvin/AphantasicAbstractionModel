'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search, ArrowRight, Link2, ChevronRight,
  Circle, Zap,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import type { RelateResult, RelateNode, RelateEdge, Tier, StructuralRelation } from '@/lib/types';
import { useUIStore } from '@/store/aamStore';

// ── Tier / Kind Styling ──
const TIER_COLORS: Record<Tier, string> = {
  1: '#00E5FF',
  2: '#FFB74D',
  3: '#FF5252',
};

const KIND_STYLE = { color: '#69F0AE', label: 'Node' };

// ── Structural Relation Row (v5.0) ──
function StructuralRelationRow({ relation }: { relation: StructuralRelation }) {
  const weightColor = relation.weight > 0.7 ? '#69F0AE' : relation.weight > 0.4 ? '#FFB74D' : '#64748b';
  return (
    <div className="flex items-center gap-2 px-2 py-1.5 text-xs">
      <Zap className="w-3.5 h-3.5 shrink-0" style={{ color: weightColor }} />
      <span className="text-[#94a3b8] font-mono truncate">{relation.source_label}</span>
      <span className="text-[#64748b] shrink-0">—{relation.relation_type}→</span>
      <span className="text-[#94a3b8] font-mono truncate">{relation.target_label}</span>
      <div className="w-12 h-1.5 bg-[#0d1520] rounded-full overflow-hidden shrink-0 ml-auto">
        <div
          className="h-full rounded-full"
          style={{ backgroundColor: weightColor, width: `${relation.weight * 100}%` }}
        />
      </div>
      <span className="text-[10px] text-[#64748b] font-mono shrink-0 w-8 text-right">{relation.weight.toFixed(2)}</span>
    </div>
  );
}

// ── Related Node Row ──
function RelatedNodeRow({ node, onSelectNode }: { node: RelateNode; onSelectNode: (id: number) => void }) {
  const tierColor = TIER_COLORS[node.tier] ?? '#64748b';
  const kindStyle = KIND_STYLE;

  return (
    <button
      type="button"
      className="w-full flex items-center gap-2.5 px-2 py-2 rounded-md hover:bg-[#1e293b]/50 transition-colors text-left group"
      onClick={() => onSelectNode(node.node_id)}
    >
      {/* Rank indicator dot */}
      <Circle className="w-2.5 h-2.5 shrink-0 fill-current" style={{ color: tierColor }} />

      {/* Label + badges */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-[#e2e8f0] font-mono truncate">{node.label}</span>
        </div>
        <div className="flex items-center gap-1.5 mt-0.5">
          <Badge
            className="text-[9px] px-1 py-0"
            style={{
              backgroundColor: `${kindStyle.color}15`,
              color: kindStyle.color,
              borderColor: 'transparent',
            }}
          >
            {kindStyle.label}
          </Badge>
          <Badge
            className="text-[9px] px-1 py-0"
            style={{
              backgroundColor: `${tierColor}15`,
              color: tierColor,
              borderColor: 'transparent',
            }}
          >
            T{node.tier}
          </Badge>
        </div>
      </div>

      {/* Score */}
      <div className="flex items-center gap-1.5 shrink-0">
        <div className="w-12 h-1.5 bg-[#0d1520] rounded-full overflow-hidden">
          <motion.div
            className="h-full rounded-full"
            style={{ backgroundColor: tierColor }}
            initial={{ width: 0 }}
            animate={{ width: `${node.score * 100}%` }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
          />
        </div>
        <span className="text-[10px] text-[#64748b] font-mono w-8 text-right">{node.score.toFixed(2)}</span>
      </div>

      {/* Navigate arrow */}
      <ChevronRight className="w-3.5 h-3.5 text-[#475569] group-hover:text-[#00E5FF] transition-colors shrink-0" />
    </button>
  );
}

// ── Related Edge Row ──
function RelatedEdgeRow({ edge }: { edge: RelateEdge }) {
  return (
    <div className="flex items-center gap-2 px-2 py-1.5 text-xs">
      <Link2 className="w-3.5 h-3.5 text-[#80D8FF] shrink-0" />
      <span className="text-[#94a3b8] font-mono">{edge.source}</span>
      <ArrowRight className="w-3 h-3 text-[#475569] shrink-0" />
      <span className="text-[#94a3b8] font-mono">{edge.target}</span>
      {edge.label && (
        <Badge variant="outline" className="text-[9px] border-[#334155] text-[#64748b] shrink-0">
          {edge.label}
        </Badge>
      )}
      <span className="text-[10px] text-[#64748b] font-mono ml-auto shrink-0">
        {edge.weight.toFixed(2)}
      </span>
    </div>
  );
}

// ── Props ──
interface RelatePanelProps {
  result: RelateResult;
  className?: string;
}

// ── Main Component ──
export default function RelatePanel({ result, className }: RelatePanelProps) {
  const selectNode = useUIStore((s) => s.selectNode);
  const focusNode = useUIStore((s) => s.focusNode);

  const handleSelectNode = (id: number) => {
    selectNode(id);
    focusNode(id);
  };

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
          {/* Header */}
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Search className="w-5 h-5 text-[#00E5FF]" />
              <CardTitle className="text-sm font-bold text-[#e2e8f0]">Relate Result</CardTitle>
            </div>
          </CardHeader>

          <CardContent className="pt-0 space-y-4">
            {/* Query Terms */}
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#475569] mb-1.5">Query Terms</div>
              <div className="flex flex-wrap gap-1.5">
                {result.query_terms.map((term, i) => (
                  <Badge
                    key={i}
                    className="text-xs font-mono bg-[#00E5FF10] text-[#00E5FF] border-[#00E5FF30] px-2 py-0.5"
                  >
                    {term}
                  </Badge>
                ))}
              </div>
            </div>

            <Separator className="bg-[#1e293b]" />

            {/* Related Nodes */}
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[#69F0AE] mb-1.5 flex items-center gap-1.5">
                <Zap className="w-3 h-3" />
                Related Nodes ({result.related_nodes.length})
              </div>
              {result.related_nodes.length > 0 ? (
                <ScrollArea className="max-h-64">
                  <div className="space-y-0.5">
                    {result.related_nodes.map((node) => (
                      <RelatedNodeRow key={node.node_id} node={node} onSelectNode={handleSelectNode} />
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <p className="text-xs text-[#475569] italic">No related nodes found</p>
              )}
            </div>

            {/* Related Edges */}
            {result.related_edges.length > 0 && (
              <>
                <Separator className="bg-[#1e293b]" />
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-[#80D8FF] mb-1.5 flex items-center gap-1.5">
                    <Link2 className="w-3 h-3" />
                    Related Edges ({result.related_edges.length})
                  </div>
                  <ScrollArea className="max-h-40">
                    <div className="space-y-0.5">
                      {result.related_edges.map((edge) => (
                        <RelatedEdgeRow key={edge.edge_id} edge={edge} />
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              </>
            )}

            {/* v5.0: Structural Relations */}
            {result.structural_relations && result.structural_relations.length > 0 && (
              <>
                <Separator className="bg-[#1e293b]" />
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-[#AB47BC] mb-1.5 flex items-center gap-1.5">
                    <Zap className="w-3 h-3" />
                    Structural Relations ({result.structural_relations.length})
                  </div>
                  <ScrollArea className="max-h-40">
                    <div className="space-y-0.5">
                      {result.structural_relations.map((rel, i) => (
                        <StructuralRelationRow key={i} relation={rel} />
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  );
}
