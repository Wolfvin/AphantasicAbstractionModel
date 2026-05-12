'use client';

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  Brain,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Send,
  Paperclip,
  Bot,
  User,
  AlertTriangle,
  Loader2,
  Sparkles,
  Trash2,
  Hexagon,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { cn } from '@/lib/utils';
import { useChatStore, useUIStore, useGraphStore, useTimelineStore, useModeResultStore } from '@/store/aamStore';
import { generateChatMessages, generateTimelineEvents } from '@/lib/mockData';
import { runModeToBackend, composeToBackend } from '@/lib/backendBridge';
import type { ChatMessage, MessageType, AppraiseResult, AppraiseVerdict, RelateResult, ComposeResult } from '@/lib/types';
import { useIsMobile } from '@/hooks/use-mobile';
import { ComposeFormPanel } from '@/components/aam/ComposePanel';

// ── Relative Time ──

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

// ── Message Type Styling Map ──

interface MessageStyleConfig {
  bubble: string;
  icon: React.ComponentType<{ className?: string }>;
  iconColor: string;
  align: string;
}
type RSVSMode = 'ingest' | 'appraise' | 'relate' | 'compose';

const MODE_LABEL: Record<RSVSMode, string> = {
  ingest: 'Ingest',
  appraise: 'Appraise',
  relate: 'Relate',
  compose: 'Compose',
};

const MODE_PREFIX_HINT = '/ingest /appraise /relate /compose';

function parseModeFromInput(raw: string, fallback: RSVSMode): { mode: RSVSMode; text: string; atomNames?: string[] } {
  const trimmed = raw.trim();
  const m = trimmed.match(/^\/(ingest|appraise|relate|compose|i|a|r|c)\b\s*/i);
  if (!m) {
    // Check for inline compose syntax: /compose label = atom1 + atom2 + atom3
    return { mode: fallback, text: trimmed };
  }
  const token = m[1].toLowerCase();
  let mode: RSVSMode;
  switch (token) {
    case 'ingest':
    case 'i':
      mode = 'ingest';
      break;
    case 'appraise':
    case 'a':
      mode = 'appraise';
      break;
    case 'relate':
    case 'r':
      mode = 'relate';
      break;
    case 'compose':
    case 'c':
      mode = 'compose';
      break;
    default:
      mode = fallback;
  }

  const remaining = trimmed.slice(m[0].length).trim();

  // Parse inline compose: "raja = tahta_tertinggi + laki_laki + kerajaan"
  if (mode === 'compose') {
    const composeMatch = remaining.match(/^(\S+)\s*=\s*(.+)$/);
    if (composeMatch) {
      const label = composeMatch[1];
      const atomNames = composeMatch[2].split('+').map(s => s.trim()).filter(Boolean);
      return { mode: 'compose', text: label, atomNames };
    }
  }

  return { mode, text: remaining };
}

const MESSAGE_STYLES: Record<MessageType, MessageStyleConfig> = {
  user_input: {
    bubble:
      'ml-auto bg-[#00E5FF10] border border-[#00E5FF30] rounded-2xl rounded-br-md',
    icon: User,
    iconColor: 'text-[#00E5FF]',
    align: 'justify-end',
  },
  system_ingest_status: {
    bubble:
      'mr-auto bg-emerald-500/10 border border-emerald-500/20 rounded-2xl rounded-bl-md',
    icon: Bot,
    iconColor: 'text-emerald-400',
    align: 'justify-start',
  },
  system_promoted_atoms: {
    bubble:
      'mr-auto bg-blue-500/10 border border-blue-500/20 rounded-2xl rounded-bl-md',
    icon: Sparkles,
    iconColor: 'text-blue-400',
    align: 'justify-start',
  },
  system_warnings: {
    bubble:
      'mr-auto bg-amber-500/10 border border-amber-500/20 rounded-2xl rounded-bl-md',
    icon: AlertTriangle,
    iconColor: 'text-amber-400',
    align: 'justify-start',
  },
  system_compose_result: {
    bubble:
      'mr-auto bg-[#FF80AB10] border border-[#FF80AB20] rounded-2xl rounded-bl-md',
    icon: Hexagon,
    iconColor: 'text-[#FF80AB]',
    align: 'justify-start',
  },
};

// ── Message Bubble ──

function MessageBubble({ message }: { message: ChatMessage }) {
  const style = MESSAGE_STYLES[message.type];
  const IconComponent = style.icon;

  const parsedAtoms = useMemo(() => {
    if (message.type !== 'system_promoted_atoms') return null;
    // Parse patterns like "mineral (T2, c=0.50)" from content
    const atomRegex = /(\w+)\s*\(T(\d),\s*c=([\d.]+)\)/g;
    const atoms: Array<{ label: string; tier: string; confidence: string }> =
      [];
    let match: RegExpExecArray | null;
    while ((match = atomRegex.exec(message.content)) !== null) {
      atoms.push({
        label: match[1],
        tier: match[2],
        confidence: match[3],
      });
    }
    return atoms;
  }, [message.content, message.type]);

  return (
    <div className={cn('flex flex-col gap-1', style.align)}>
      <div className="flex items-start gap-2 max-w-[90%]">
        {message.type !== 'user_input' && (
          <div className="mt-1 shrink-0">
            <IconComponent className={cn('size-4', style.iconColor)} />
          </div>
        )}
        <div
          className={cn(
            'px-3 py-2 shadow-sm transition-shadow hover:shadow-md',
            style.bubble,
          )}
        >
          {message.mode && (
            <div className="mb-1">
              <Badge variant="outline" className="text-[10px] border-white/15 bg-white/5 text-muted-foreground">
                {MODE_LABEL[message.mode as RSVSMode] ?? message.mode}
              </Badge>
            </div>
          )}
          {message.type === 'system_promoted_atoms' && parsedAtoms ? (
            <div className="flex flex-col gap-1.5">
              <span className="text-xs text-muted-foreground/80">
                New promoted atoms:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {parsedAtoms.map((atom) => (
                  <Badge
                    key={atom.label}
                    variant="outline"
                    className="border-blue-500/30 bg-blue-500/10 text-blue-300 text-xs gap-1"
                  >
                    {atom.label}
                    <span className="text-blue-400/70 font-mono">
                      T{atom.tier}
                    </span>
                    <span className="text-blue-400/50 font-mono">
                      {atom.confidence}
                    </span>
                  </Badge>
                ))}
              </div>
              {/* Show any extra content beyond the parsed atoms */}
              {message.content.replace(
                /New atoms:\s*.*?(?=\n|$)/,
                '',
              ).trim() && (
                <p className="text-xs text-muted-foreground mt-1">
                  {message.content.replace(/New atoms:\s*.*?(?=\n|$)/, '').trim()}
                </p>
              )}
            </div>
          ) : message.type === 'system_compose_result' ? (
            <p className="text-sm text-[#FF80AB]/90 font-mono leading-relaxed">{message.content}</p>
          ) : message.type === 'system_ingest_status' ? (
            <p className="text-sm text-emerald-300/90 font-mono leading-relaxed">
              {message.content}
            </p>
          ) : message.type === 'system_warnings' ? (
            <p className="text-sm text-amber-300/90">{message.content}</p>
          ) : (
            <p className="text-sm text-foreground/90">{message.content}</p>
          )}
        </div>
        {message.type === 'user_input' && (
          <div className="mt-1 shrink-0">
            <User className="size-4 text-[#00E5FF]" />
          </div>
        )}
      </div>
      <span
        className={cn(
          'text-[10px] text-muted-foreground/50 px-1',
          message.type === 'user_input' ? 'text-right pr-7' : 'pl-7',
        )}
      >
        {relativeTime(message.timestamp)}
      </span>
    </div>
  );
}

// ── Loading Indicator ──

function IngestingIndicator() {
  return (
    <div className="flex items-start gap-2 justify-start">
      <div className="mt-1 shrink-0">
        <Bot className="size-4 text-emerald-400" />
      </div>
      <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
        <div className="flex items-center gap-2">
          <Loader2 className="size-3.5 text-emerald-400 animate-spin" />
          <span className="text-sm text-emerald-300/80">Ingesting...</span>
        </div>
        <div className="flex gap-1 mt-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400/60 animate-bounce [animation-delay:0ms]" />
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400/60 animate-bounce [animation-delay:150ms]" />
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400/60 animate-bounce [animation-delay:300ms]" />
        </div>
      </div>
    </div>
  );
}

function ComposingIndicator() {
  return (
    <div className="flex items-start gap-2 justify-start">
      <div className="mt-1 shrink-0">
        <Hexagon className="size-4 text-[#FF80AB]" />
      </div>
      <div className="bg-[#FF80AB10] border border-[#FF80AB20] rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
        <div className="flex items-center gap-2">
          <Loader2 className="size-3.5 text-[#FF80AB] animate-spin" />
          <span className="text-sm text-[#FF80AB]/80">Composing...</span>
        </div>
      </div>
    </div>
  );
}

// ── Quick Action Chips ──

function QuickActions({
  onLoadSampleData,
  onClearGraph,
  disabled,
}: {
  onLoadSampleData: () => void;
  onClearGraph: () => void;
  disabled: boolean;
}) {
  return (
    <div className="flex gap-1.5 px-3 pb-2 overflow-x-auto">
      <button
        type="button"
        onClick={onLoadSampleData}
        disabled={disabled}
        className={cn(
          'inline-flex items-center gap-1.5 whitespace-nowrap rounded-full',
          'border border-[#00E5FF20] bg-[#00E5FF08] text-[#00E5FF]/80',
          'px-2.5 py-1 text-xs font-medium',
          'transition-colors hover:bg-[#00E5FF15] hover:text-[#00E5FF]',
          'disabled:opacity-40 disabled:pointer-events-none',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#00E5FF]/40',
        )}
      >
        <Sparkles className="size-3" />
        Load Sample Data
      </button>
      <button
        type="button"
        onClick={onClearGraph}
        disabled={disabled}
        className={cn(
          'inline-flex items-center gap-1.5 whitespace-nowrap rounded-full',
          'border border-destructive/20 bg-destructive/5 text-destructive/80',
          'px-2.5 py-1 text-xs font-medium',
          'transition-colors hover:bg-destructive/10 hover:text-destructive',
          'disabled:opacity-40 disabled:pointer-events-none',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-destructive/40',
        )}
      >
        <Trash2 className="size-3" />
        Clear Graph
      </button>
    </div>
  );
}

// ── Collapsed Icon Strip ──

function CollapsedStrip({
  onExpand,
  isMobile,
}: {
  onExpand: () => void;
  isMobile: boolean;
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center py-3 gap-4',
        isMobile ? 'w-full flex-row justify-center' : 'w-12',
        'bg-[#0d1117] border-[#1b2332]',
        isMobile ? 'border-t' : 'border-r',
      )}
    >
      <TooltipButton
        tooltip="Expand Input Panel"
        side={isMobile ? 'top' : 'right'}
        onClick={onExpand}
      >
        <Brain className="size-4 text-[#00E5FF]" />
      </TooltipButton>
      <div className="w-6 h-px bg-white/10" />
      <TooltipButton
        tooltip="Expand"
        side={isMobile ? 'top' : 'right'}
        onClick={onExpand}
      >
        {isMobile ? (
          <ChevronUp className="size-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-4 text-muted-foreground" />
        )}
      </TooltipButton>
    </div>
  );
}

function TooltipButton({
  tooltip,
  side,
  onClick,
  children,
}: {
  tooltip: string;
  side: 'top' | 'right';
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          className="p-1.5 rounded-md hover:bg-white/5 transition-colors text-muted-foreground hover:text-foreground"
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent side={side} className="text-xs">
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

// ── Main Component ──

interface LeftInputRailProps {
  onLoadSampleData?: () => void;
  onClearGraph?: () => void;
}

export default function LeftInputRail({
  onLoadSampleData,
  onClearGraph,
}: LeftInputRailProps) {
  const { messages, isLoading, addMessage, setLoading } =
    useChatStore();
  const loadSnapshot = useGraphStore((s) => s.loadSnapshot);
  const pushEvent = useGraphStore((s) => s.pushEvent);
  const addTimelineEvent = useTimelineStore((s) => s.addTimelineEvent);
  const resetTimeline = useTimelineStore((s) => s.resetTimeline);
  const { isLeftRailCollapsed, toggleLeftRail } = useUIStore();

  const [inputValue, setInputValue] = useState('');
  const [activeMode, setActiveMode] = useState<RSVSMode>('ingest');
  const [showComposePanel, setShowComposePanel] = useState(false);
  const [isComposing, setIsComposing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const isMobile = useIsMobile();

  // ── Load mock data on first mount if store is empty ──
  // Works in both development and production (demo mode)
  const hasInitialized = useRef(false);
  useEffect(() => {
    if (!hasInitialized.current && messages.length === 0) {
      const mockMessages = generateChatMessages();
      mockMessages.forEach((msg) => addMessage(msg));
      hasInitialized.current = true;
    }
  }, [messages.length, addMessage]);

  // ── Auto-scroll to bottom on new messages ──
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages.length, isLoading]);

  // ── Auto-resize textarea ──
  const adjustTextareaHeight = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const lineHeight = 24;
    const minRows = 3;
    const maxRows = 6;
    const minHeight = lineHeight * minRows;
    const maxHeight = lineHeight * maxRows;
    const newHeight = Math.min(Math.max(el.scrollHeight, minHeight), maxHeight);
    el.style.height = `${newHeight}px`;
  }, []);

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputValue, adjustTextareaHeight]);

  // ── Simulated system response (development/demo only) ──
  const ingestTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const simulateIngestResponse = useCallback(
    (correlationId: string) => {
      if (ingestTimeoutRef.current) {
        clearTimeout(ingestTimeoutRef.current);
      }

      const delay = 1000 + Math.random() * 1000;

      ingestTimeoutRef.current = setTimeout(() => {
        const atomCount = Math.floor(Math.random() * 15) + 3;
        const tokenCount = Math.floor(Math.random() * 2000) + 500;
        const compositeCount = Math.floor(Math.random() * 4) + 1;
        const batchId = `ingest_${String(Math.floor(Math.random() * 99999)).padStart(5, '0')}`;

        addMessage({
          id: `resp_${correlationId}_status`,
          type: 'system_ingest_status',
          content: `Ingesting batch ${batchId} — ${tokenCount.toLocaleString()} tokens processed, ${atomCount} atoms promoted, ${compositeCount} composite${compositeCount > 1 ? 's' : ''} formed.`,
          timestamp: new Date().toISOString(),
          correlation_id: correlationId,
        });

        if (Math.random() > 0.3) {
          const sampleAtoms = [
            { label: 'concept', tier: 1 },
            { label: 'relation', tier: 2 },
            { label: 'domain', tier: 1 },
          ];
          const selected = sampleAtoms
            .sort(() => Math.random() - 0.5)
            .slice(0, 2 + Math.floor(Math.random() * 3));
          const atomsStr = selected
            .map((a) => `${a.label} (T${a.tier}, c=${(Math.random() * 0.5 + 0.3).toFixed(2)})`)
            .join(', ');

          addMessage({
            id: `resp_${correlationId}_atoms`,
            type: 'system_promoted_atoms',
            content: `New atoms: ${atomsStr}`,
            timestamp: new Date().toISOString(),
            correlation_id: correlationId,
          });
        }

        setLoading(false);
        ingestTimeoutRef.current = null;
      }, delay);
    },
    [addMessage, setLoading],
  );

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (ingestTimeoutRef.current) {
        clearTimeout(ingestTimeoutRef.current);
      }
    };
  }, []);

  // ── Simulated appraise response (demo mode) ──
  const simulateAppraiseResponse = useCallback(
    (correlationId: string, text: string) => {
      const delay = 800 + Math.random() * 800;

      setTimeout(() => {
        const nodes = useGraphStore.getState().nodes;
        const allNodes = Array.from(nodes.values());

        // Pick some random nodes as evidence
        const shuffled = [...allNodes].sort(() => Math.random() - 0.5);
        const supportCount = Math.min(Math.floor(Math.random() * 4) + 1, shuffled.length);
        const conflictCount = Math.min(Math.floor(Math.random() * 3), Math.max(0, shuffled.length - supportCount));
        const supportNodes = shuffled.slice(0, supportCount);
        const conflictNodes = shuffled.slice(supportCount, supportCount + conflictCount);

        const agree = Math.floor(Math.random() * 60) + 20;
        const disagree = Math.floor(Math.random() * 30) + 5;
        const neutral = 100 - agree - disagree;
        const verdict: AppraiseVerdict = agree > 60 ? 'agree' : disagree > 40 ? 'disagree' : 'mixed';
        const confidence = agree / 100;

        const appraiseResult: AppraiseResult = {
          verdict,
          stance: { agree, disagree, neutral },
          confidence,
          rationale: `Demo appraise for "${text.slice(0, 50)}": ${verdict} verdict based on ${supportCount} supporting and ${conflictCount} conflicting evidence nodes.`,
          evidence_nodes: supportNodes.map((n) => ({
            node_id: n.id,
            label: n.label,
            confidence: n.confidence,
            role: 'support' as const,
          })),
          conflict_nodes: conflictNodes.map((n) => ({
            node_id: n.id,
            label: n.label,
            confidence: n.confidence,
            role: 'conflict' as const,
          })),
          evidence_paths: supportNodes.slice(0, 3).map((n, i) => ({
            path: [n.id, shuffled[(i + 1) % shuffled.length]?.id ?? n.id],
            weight: Math.random() * 0.5 + 0.3,
          })),
        };

        useModeResultStore.getState().setAppraiseResult(appraiseResult);

        addMessage({
          id: `resp_${correlationId}_appraise`,
          type: 'system_ingest_status',
          content: `Appraise: ${verdict} (${agree}% agree / ${disagree}% disagree). ${supportCount} support, ${conflictCount} conflict. [Demo]`,
          timestamp: new Date().toISOString(),
          correlation_id: correlationId,
          mode: 'appraise',
        });

        setLoading(false);
      }, delay);
    },
    [addMessage, setLoading],
  );

  // ── Simulated relate response (demo mode) ──
  const simulateRelateResponse = useCallback(
    (correlationId: string, text: string) => {
      const delay = 800 + Math.random() * 800;

      setTimeout(() => {
        const nodes = useGraphStore.getState().nodes;
        const edges = useGraphStore.getState().edges;
        const allNodes = Array.from(nodes.values());
        const allEdges = Array.from(edges.values());

        // Pick random related nodes
        const relatedCount = Math.min(Math.floor(Math.random() * 8) + 3, allNodes.length);
        const relatedNodes = allNodes.slice(0, relatedCount);
        const relatedEdges = allEdges.slice(0, Math.min(relatedCount * 2, allEdges.length));

        const relateResult: RelateResult = {
          query_terms: text.split(/\s+/).slice(0, 5),
          related_nodes: relatedNodes.map((n) => ({
            node_id: n.id,
            label: n.label,
            score: Math.random() * 0.7 + 0.3,
            tier: n.tier,
            kind: n.kind,
            layer: n.layer,
            grounding_score: n.grounding_score,
          })),
          related_edges: relatedEdges.map((e) => ({
            edge_id: e.id,
            source: e.source,
            target: e.target,
            weight: e.weight,
            label: e.label,
          })),
          structural_relations: relatedNodes.slice(0, 4).map((n, i) => ({
            relation_type: ['composes', 'depends_on', 'similar_to', 'contrasts_with'][i % 4],
            source_label: text.split(/\s+/)[0] || 'query',
            target_label: n.label,
            weight: Math.random() * 0.6 + 0.2,
            description: `Structural relation between query and ${n.label}`,
          })),
        };

        useModeResultStore.getState().setRelateResult(relateResult);

        addMessage({
          id: `resp_${correlationId}_relate`,
          type: 'system_ingest_status',
          content: `Relate: found ${relatedNodes.length} nodes and ${relatedEdges.length} edges. ${relateResult.structural_relations?.length ?? 0} structural relations. [Demo]`,
          timestamp: new Date().toISOString(),
          correlation_id: correlationId,
          mode: 'relate',
        });

        setLoading(false);
      }, delay);
    },
    [addMessage, setLoading],
  );

  // ── Handle compose action ──
  const handleCompose = useCallback(async (label: string, atomIds: number[], lang: string) => {
    const correlationId = `msg_${Date.now()}`;
    setIsComposing(true);

    const userMsg: ChatMessage = {
      id: `chat_${correlationId}`,
      type: 'user_input',
      content: `/compose ${label} = ${atomIds.join(' + ')}`,
      timestamp: new Date().toISOString(),
      correlation_id: correlationId,
      mode: 'compose',
    };
    addMessage(userMsg);
    setShowComposePanel(false);

    try {
      const res = await composeToBackend(label, atomIds, lang, correlationId);
      if (!res.ok) {
        throw new Error(res.error || 'compose failed');
      }

      const composeResult = res.result?.compose;
      if (composeResult) {
        useModeResultStore.getState().setComposeResult(composeResult);

        addMessage({
          id: `resp_${correlationId}_compose`,
          type: 'system_compose_result',
          content: `Composed "${composeResult.composite_node.label}" (ID: ${composeResult.composite_node.id}) from ${composeResult.atom_nodes.length} atom${composeResult.atom_nodes.length !== 1 ? 's' : ''}${composeResult.jaccard_similarities.length > 0 ? `. ${composeResult.jaccard_similarities.length} similar composite${composeResult.jaccard_similarities.length !== 1 ? 's' : ''} found.` : ''}.`,
          timestamp: new Date().toISOString(),
          correlation_id: correlationId,
          mode: 'compose',
        });

        // If snapshot returned, load it
        if (res.result?.snapshot) {
          loadSnapshot(res.result.snapshot.nodes, res.result.snapshot.edges);
        }
      }

      (res.messages || []).forEach((msg) => addMessage({ ...msg, mode: 'compose' }));
    } catch {
      // Backend not available — simulate compose result (demo mode)
      const nodes = useGraphStore.getState().nodes;
      const atomNodes = atomIds.map(id => nodes.get(id)).filter(Boolean);
      addMessage({
        id: `resp_${correlationId}_compose_fallback`,
        type: 'system_compose_result',
        content: `Composed "${label}" from ${atomNodes.length} atom${atomNodes.length !== 1 ? 's' : ''} (${atomNodes.map(n => n?.label).join(', ')}). [Demo]`,
        timestamp: new Date().toISOString(),
        correlation_id: correlationId,
        mode: 'compose',
      });
    } finally {
      setIsComposing(false);
    }
  }, [addMessage, loadSnapshot]);

  // ── Submit handler ──
  const handleSubmit = useCallback(async () => {
    const parsed = parseModeFromInput(inputValue, activeMode);
    const payloadText = parsed.text.trim();
    if (!payloadText || isLoading) return;

    // Handle compose mode specially
    if (parsed.mode === 'compose') {
      if (parsed.atomNames && parsed.atomNames.length > 0) {
        // Inline compose syntax: /compose label = atom1 + atom2 + atom3
        const nodes = useGraphStore.getState().nodes;
        const atomIds: number[] = [];
        for (const name of parsed.atomNames) {
          // Try to find matching nodes by label
          let found = false;
          nodes.forEach((n) => {
            if (n.label.toLowerCase() === name.toLowerCase() && !found) {
              atomIds.push(n.id);
              found = true;
            }
          });
          if (!found) {
            // Try partial match
            nodes.forEach((n) => {
              if (n.label.toLowerCase().includes(name.toLowerCase()) && !atomIds.includes(n.id) && !found) {
                atomIds.push(n.id);
                found = true;
              }
            });
          }
        }

        if (atomIds.length > 0) {
          await handleCompose(payloadText, atomIds, 'id');
        } else {
          addMessage({
            id: `msg_${Date.now()}_error`,
            type: 'system_warnings',
            content: `Could not find atom nodes matching: ${parsed.atomNames.join(', ')}`,
            timestamp: new Date().toISOString(),
            mode: 'compose',
          });
        }
        setInputValue('');
        return;
      } else {
        // Just "/compose" — show the panel
        setShowComposePanel(true);
        setActiveMode('compose');
        setInputValue('');
        return;
      }
    }

    const correlationId = `msg_${Date.now()}`;
    const userMsg: ChatMessage = {
      id: `chat_${correlationId}`,
      type: 'user_input',
      content: payloadText,
      timestamp: new Date().toISOString(),
      correlation_id: correlationId,
      mode: parsed.mode,
    };

    addMessage(userMsg);
    setInputValue('');
    setLoading(true);
    setActiveMode(parsed.mode);

    // Focus back on textarea
    setTimeout(() => textareaRef.current?.focus(), 0);

    try {
      const res = await runModeToBackend(parsed.mode, payloadText, correlationId);
      if (!res.ok) {
        throw new Error(res.error || 'backend returned not ok');
      }

      if (parsed.mode === 'ingest' && res.result?.snapshot) {
        const snapshot = res.result.snapshot;
        const events = res.result.events || [];
        loadSnapshot(snapshot.nodes, snapshot.edges);
        resetTimeline();
        events.forEach((evt) => {
          pushEvent(evt);
          generateTimelineEvents([evt]).forEach((tle) => addTimelineEvent(tle));
        });
      }

      if (parsed.mode === 'appraise') {
        const appraiseResult = res.result as AppraiseResult | undefined;
        if (appraiseResult?.verdict) {
          useModeResultStore.getState().setAppraiseResult(appraiseResult);
        }
        const stance = appraiseResult?.stance ?? { agree: 0, disagree: 0, neutral: 0 };
        const verdict = appraiseResult?.verdict ?? 'mixed';
        addMessage({
          id: `resp_${correlationId}_appraise`,
          type: 'system_ingest_status',
          content: `Appraise: ${verdict} (${stance.agree ?? 0}% agree / ${stance.disagree ?? 0}% disagree).`,
          timestamp: new Date().toISOString(),
          correlation_id: correlationId,
          mode: parsed.mode,
        });
      }

      if (parsed.mode === 'relate') {
        const relateResult = res.result as RelateResult | undefined;
        if (relateResult?.related_nodes) {
          useModeResultStore.getState().setRelateResult(relateResult);
        }
        const relatedNodes = relateResult?.related_nodes || [];
        const relatedEdges = relateResult?.related_edges || [];
        addMessage({
          id: `resp_${correlationId}_relate`,
          type: 'system_ingest_status',
          content: `Relate: found ${relatedNodes.length} nodes and ${relatedEdges.length} edges.`,
          timestamp: new Date().toISOString(),
          correlation_id: correlationId,
          mode: parsed.mode,
        });
      }

      (res.messages || []).forEach((msg) => addMessage({ ...msg, mode: parsed.mode }));
      setLoading(false);
    } catch (err) {
      // Backend not available — use simulated response (demo mode)
      console.error('[RSVS] Backend call failed, mode=', parsed.mode, 'error=', err);
      if (parsed.mode === 'ingest') {
        simulateIngestResponse(correlationId);
      } else if (parsed.mode === 'appraise') {
        simulateAppraiseResponse(correlationId, payloadText);
      } else if (parsed.mode === 'relate') {
        simulateRelateResponse(correlationId, payloadText);
      } else {
        addMessage({
          id: `resp_${correlationId}_fallback`,
          type: 'system_warnings',
          content: `Backend unavailable for ${MODE_LABEL[parsed.mode]} mode. Running in demo mode.`,
          timestamp: new Date().toISOString(),
          correlation_id: correlationId,
          mode: parsed.mode,
        });
        setLoading(false);
      }
    }
  }, [
    inputValue,
    activeMode,
    isLoading,
    addMessage,
    setLoading,
    simulateIngestResponse,
    simulateAppraiseResponse,
    simulateRelateResponse,
    loadSnapshot,
    resetTimeline,
    pushEvent,
    addTimelineEvent,
    handleCompose,
  ]);

  // ── Keyboard handler ──
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  // ── Quick action handlers ──
  const handleLoadSampleData = useCallback(() => {
    if (onLoadSampleData) {
      onLoadSampleData();
    } else {
      window.dispatchEvent(new CustomEvent('rsvs:load-sample-data'));
    }
  }, [onLoadSampleData]);

  const handleClearGraph = useCallback(() => {
    if (onClearGraph) {
      onClearGraph();
    } else {
      window.dispatchEvent(new CustomEvent('rsvs:clear-graph'));
    }
  }, [onClearGraph]);

  // ── Collapsed state ──
  if (isLeftRailCollapsed) {
    return <CollapsedStrip onExpand={toggleLeftRail} isMobile={isMobile} />;
  }

  return (
    <aside
      className={cn(
        'relative flex flex-col h-full',
        'bg-[#0d1117] border-r border-[#1b2332]',
        'w-[24%] min-w-[280px] max-w-[400px]',
        'transition-all duration-300 ease-in-out',
      )}
    >
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-[#1b2332] shrink-0">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center size-7 rounded-md bg-[#00E5FF10] border border-[#00E5FF20]">
            <Brain className="size-4 text-[#00E5FF]" />
          </div>
          <h2 className="text-sm font-semibold text-foreground/90 tracking-tight">
            RSVS Input
          </h2>
        </div>
        <TooltipButton tooltip="Collapse" side="right" onClick={toggleLeftRail}>
          <ChevronLeft className="size-4" />
        </TooltipButton>
      </div>

      {/* ── Messages ── */}
      <div
        ref={scrollContainerRef}
        className={cn(
          'flex-1 overflow-y-auto px-3 py-3 space-y-3',
          // Custom thin scrollbar
          '[&::-webkit-scrollbar]:w-1.5',
          '[&::-webkit-scrollbar-track]:bg-transparent',
          '[&::-webkit-scrollbar-thumb]:bg-white/10',
          '[&::-webkit-scrollbar-thumb]:rounded-full',
          '[&::-webkit-scrollbar-thumb:hover]:bg-white/20',
          // Firefox scrollbar
          'scrollbar-thin scrollbar-color-white/10 transparent',
        )}
      >
        {messages.length === 0 && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-center px-4 gap-3">
            <div className="size-12 rounded-full bg-[#00E5FF08] border border-[#00E5FF15] flex items-center justify-center">
              <Brain className="size-6 text-[#00E5FF]/40" />
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">
                No messages yet
              </p>
              <p className="text-xs text-muted-foreground/50">
                Type a message or use a quick action below to get started.
              </p>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {isLoading && <IngestingIndicator />}
        {isComposing && <ComposingIndicator />}
      </div>

      {/* ── Compose Panel (shown when /compose is triggered) ── */}
      {showComposePanel && (
        <div className="px-3 pb-2">
          <ComposeFormPanel
            onCompose={handleCompose}
            onCancel={() => setShowComposePanel(false)}
            isLoading={isComposing}
          />
        </div>
      )}

      {/* ── Quick Actions ── */}
      <QuickActions
        onLoadSampleData={handleLoadSampleData}
        onClearGraph={handleClearGraph}
        disabled={isLoading || isComposing}
      />

      {/* ── Input Area ── */}
      <div className="shrink-0 border-t border-[#1b2332] px-3 pb-3 pt-2">
        <div className="relative flex items-end gap-2 bg-[#161b22] border border-[#1b2332] rounded-xl focus-within:border-[#00E5FF30] focus-within:ring-2 focus-within:ring-[#00E5FF10] transition-all">
          <button
            type="button"
            className="shrink-0 p-2 text-muted-foreground hover:text-foreground transition-colors rounded-lg hover:bg-white/5 mb-1.5 ml-1"
            aria-label="Attach file"
          >
            <Paperclip className="size-4" />
          </button>

          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Send text in ${MODE_LABEL[activeMode]} mode...`}
            rows={3}
            disabled={isLoading || isComposing}
            className={cn(
              'flex-1 resize-none bg-transparent py-2 text-sm text-foreground placeholder:text-muted-foreground/40',
              'focus:outline-none',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'scrollbar-thin scrollbar-color-white/10 transparent',
              '[&::-webkit-scrollbar]:w-1',
              '[&::-webkit-scrollbar-track]:bg-transparent',
              '[&::-webkit-scrollbar-thumb]:bg-white/10',
              '[&::-webkit-scrollbar-thumb]:rounded-full',
            )}
          />

          <Button
            type="button"
            size="icon"
            onClick={handleSubmit}
            disabled={isLoading || isComposing || !inputValue.trim()}
            className={cn(
              'shrink-0 mb-1.5 mr-1.5',
              activeMode === 'compose'
                ? 'bg-[#FF80AB15] border border-[#FF80AB30] text-[#FF80AB] hover:bg-[#FF80AB25]'
                : 'bg-[#00E5FF15] border border-[#00E5FF30] text-[#00E5FF] hover:bg-[#00E5FF25]',
              'disabled:opacity-30 disabled:pointer-events-none',
              'transition-colors',
            )}
            aria-label="Send message"
          >
            <Send className="size-4" />
          </Button>
        </div>
        <div className="mt-1.5 flex items-center justify-between px-1">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                className="inline-flex h-6 items-center gap-1 rounded-md border border-[#2a3345] bg-[#0f1420] px-2 text-[10px] text-[#d1d5db] hover:bg-[#151c2b] transition-colors"
              >
                <span style={{ color: activeMode === 'compose' ? '#FF80AB' : '#facc15' }}>●</span>
                {MODE_LABEL[activeMode]}
                <ChevronDown className="size-3 text-muted-foreground" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-44">
              {(['ingest', 'appraise', 'relate', 'compose'] as RSVSMode[]).map((m) => (
                <DropdownMenuItem key={m} onClick={() => { setActiveMode(m); if (m === 'compose') setShowComposePanel(true); }}>
                  /{m}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <p className="text-[10px] text-muted-foreground/35">
            {MODE_PREFIX_HINT}
          </p>
        </div>
        <p className="text-[10px] text-muted-foreground/30 mt-1 px-1 text-center">
          Press <kbd className="px-1 py-0.5 rounded border border-white/10 bg-white/5 text-[9px]">Enter</kbd> to send &middot; <kbd className="px-1 py-0.5 rounded border border-white/10 bg-white/5 text-[9px]">Shift+Enter</kbd> for newline
        </p>
      </div>
    </aside>
  );
}

// Re-export ChevronDown that's used inline
function ChevronDown({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}
