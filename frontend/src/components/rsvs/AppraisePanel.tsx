'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ShieldCheck, ShieldAlert, ShieldX, TrendingUp,
  ChevronRight, GitBranch, AlertTriangle, CheckCircle2, Link2,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import type { AppraiseResult, AppraiseVerdict, AppraiseEvidenceNode, EvidencePath } from '@/lib/types';
import { useUIStore } from '@/store/rsvsStore';

// ── Verdict Config ──
const VERDICT_CONFIG: Record<AppraiseVerdict, { icon: React.ComponentType<React.SVGProps<SVGSVGElement>>; color: string; bg: string; label: string }> = {
  agree: { icon: ShieldCheck, color: '#69F0AE', bg: '#69F0AE15', label: 'Agree' },
  mixed: { icon: ShieldAlert, color: '#FFB74D', bg: '#FFB74D15', label: 'Mixed' },
  disagree: { icon: ShieldX, color: '#FF5252', bg: '#FF525215', label: 'Disagree' },
};

// ── Evidence Node Row ──
function EvidenceNodeRow({ node, onSelectNode }: { node: AppraiseEvidenceNode; onSelectNode: (id: number) => void }) {
  const isSupport = node.role === 'support';
  const isConflict = node.role === 'conflict';
  const Icon = isSupport ? CheckCircle2 : isConflict ? AlertTriangle : TrendingUp;
  const color = isSupport ? '#69F0AE' : isConflict ? '#FF5252' : '#94a3b8';

  return (
    <button
      type="button"
      className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-[#1e293b]/50 transition-colors text-left"
      onClick={() => onSelectNode(node.node_id)}
    >
      <Icon className="w-3.5 h-3.5 shrink-0" style={{ color }} />
      <span className="text-xs text-[#e2e8f0] font-mono truncate flex-1">{node.label}</span>
      <span className="text-[10px] font-mono shrink-0" style={{ color }}>
        {node.confidence.toFixed(2)}
      </span>
    </button>
  );
}

// ── Evidence Path Row ──
function EvidencePathRow({ path }: { path: EvidencePath }) {
  return (
    <div className="flex items-center gap-2 px-2 py-1.5 text-xs">
      <GitBranch className="w-3.5 h-3.5 text-[#80D8FF] shrink-0" />
      <span className="text-[#94a3b8] font-mono truncate">
        {path.path.join(' → ')}
      </span>
      {path.label && (
        <Badge variant="outline" className="text-[9px] border-[#334155] text-[#64748b] shrink-0">
          {path.label}
        </Badge>
      )}
      <span className="text-[10px] text-[#64748b] font-mono ml-auto shrink-0">
        {path.weight.toFixed(2)}
      </span>
    </div>
  );
}

// ── Props ──
interface AppraisePanelProps {
  result: AppraiseResult;
  className?: string;
}

// ── Main Component ──
export default function AppraisePanel({ result, className }: AppraisePanelProps) {
  const selectNode = useUIStore((s) => s.selectNode);
  const focusNode = useUIStore((s) => s.focusNode);

  const verdictConfig = VERDICT_CONFIG[result.verdict] ?? VERDICT_CONFIG.mixed;
  const VerdictIcon = verdictConfig.icon;

  const handleSelectNode = (id: number) => {
    selectNode(id);
    focusNode(id);
  };

  const supportNodes = result.evidence_nodes.filter((n) => n.role === 'support');
  const neutralNodes = result.evidence_nodes.filter((n) => n.role === 'neutral');

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
          {/* Header: Verdict */}
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <VerdictIcon className="w-5 h-5" style={{ color: verdictConfig.color }} />
                <CardTitle className="text-sm font-bold text-[#e2e8f0]">Appraise Result</CardTitle>
              </div>
              <Badge
                className="text-xs font-semibold px-2.5 py-0.5"
                style={{
                  backgroundColor: verdictConfig.bg,
                  color: verdictConfig.color,
                  borderColor: `${verdictConfig.color}40`,
                }}
              >
                {verdictConfig.label}
              </Badge>
            </div>
          </CardHeader>

          <CardContent className="pt-0 space-y-4">
            {/* Stance Bar */}
            <div>
              <div className="flex items-center justify-between text-[10px] mb-1.5">
                <span className="text-[#69F0AE] font-medium">Agree {result.stance.agree}%</span>
                <span className="text-[#FF5252] font-medium">Disagree {result.stance.disagree}%</span>
              </div>
              <div className="h-2.5 bg-[#0d1520] rounded-full overflow-hidden flex">
                <motion.div
                  className="h-full bg-[#69F0AE] rounded-l-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${result.stance.agree}%` }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                />
                <motion.div
                  className="h-full bg-[#475569]"
                  initial={{ width: 0 }}
                  animate={{ width: `${result.stance.neutral}%` }}
                  transition={{ duration: 0.6, ease: 'easeOut', delay: 0.1 }}
                />
                <motion.div
                  className="h-full bg-[#FF5252] rounded-r-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${result.stance.disagree}%` }}
                  transition={{ duration: 0.6, ease: 'easeOut', delay: 0.2 }}
                />
              </div>
            </div>

            {/* Confidence Score */}
            <div>
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="text-[#94a3b8]">Confidence</span>
                <span className="text-[#e2e8f0] font-mono font-bold">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <Progress value={result.confidence * 100} className="h-1.5 bg-[#0d1520]" />
            </div>

            <Separator className="bg-[#1e293b]" />

            {/* Rationale */}
            {result.rationale && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-[#475569] mb-1.5">Rationale</div>
                <p className="text-xs text-[#94a3b8] leading-relaxed">{result.rationale}</p>
              </div>
            )}

            {/* Support Nodes */}
            {supportNodes.length > 0 && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-[#69F0AE] mb-1.5 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3 h-3" />
                  Support Nodes ({supportNodes.length})
                </div>
                <div className="max-h-32 overflow-y-auto space-y-0.5 rsvs-scrollbar">
                  {supportNodes.map((node) => (
                    <EvidenceNodeRow key={node.node_id} node={node} onSelectNode={handleSelectNode} />
                  ))}
                </div>
              </div>
            )}

            {/* Conflict Nodes */}
            {result.conflict_nodes && result.conflict_nodes.length > 0 && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-[#FF5252] mb-1.5 flex items-center gap-1.5">
                  <AlertTriangle className="w-3 h-3" />
                  Conflict Nodes ({result.conflict_nodes.length})
                </div>
                <div className="max-h-32 overflow-y-auto space-y-0.5 rsvs-scrollbar">
                  {result.conflict_nodes.map((node) => (
                    <EvidenceNodeRow key={node.node_id} node={node} onSelectNode={handleSelectNode} />
                  ))}
                </div>
              </div>
            )}

            {/* Neutral Nodes (if any) */}
            {neutralNodes.length > 0 && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-[#94a3b8] mb-1.5 flex items-center gap-1.5">
                  <TrendingUp className="w-3 h-3" />
                  Other Evidence ({neutralNodes.length})
                </div>
                <div className="max-h-32 overflow-y-auto space-y-0.5 rsvs-scrollbar">
                  {neutralNodes.map((node) => (
                    <EvidenceNodeRow key={node.node_id} node={node} onSelectNode={handleSelectNode} />
                  ))}
                </div>
              </div>
            )}

            {/* Evidence Paths */}
            {result.evidence_paths && result.evidence_paths.length > 0 && (
              <>
                <Separator className="bg-[#1e293b]" />
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-[#80D8FF] mb-1.5 flex items-center gap-1.5">
                    <GitBranch className="w-3 h-3" />
                    Evidence Paths ({result.evidence_paths.length})
                  </div>
                  <ScrollArea className="max-h-40">
                    <div className="space-y-0.5">
                      {result.evidence_paths.map((path, i) => (
                        <EvidencePathRow key={i} path={path} />
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              </>
            )}

            {/* v8.2: Convergence Contributors */}
            {result.convergence_contributors && result.convergence_contributors.length > 0 && (
              <>
                <Separator className="bg-[#1e293b]" />
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-[#CE93D8] mb-1.5 flex items-center gap-1.5">
                    <Link2 className="w-3 h-3" />
                    Convergence Boost ({result.convergence_contributors.length})
                  </div>
                  <div className="space-y-0.5">
                    {result.convergence_contributors.map((contrib, i) => (
                      <div key={i} className="flex items-center gap-2 px-2 py-1 text-xs">
                        <Link2 className="w-3 h-3 text-[#CE93D8] shrink-0" />
                        <span className="text-[#e2e8f0] font-mono truncate flex-1">{contrib.label}</span>
                        <span className="text-[10px] font-mono text-[#CE93D8] shrink-0">
                          +{(contrib.boost * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* Target node link */}
            {result.target_node_id != null && (
              <>
                <Separator className="bg-[#1e293b]" />
                <button
                  type="button"
                  className="w-full flex items-center justify-center gap-1.5 text-xs text-[#00E5FF] hover:text-[#00E5FF]/80 transition-colors py-1"
                  onClick={() => handleSelectNode(result.target_node_id!)}
                >
                  View target node in graph
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  );
}
