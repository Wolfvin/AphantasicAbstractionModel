'use client';

import React, { useMemo } from 'react';
import {
  Home, Layers, Eye, Presentation, Search, Maximize2,
  Filter, Zap,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useUIStore, useGraphStore } from '@/store/aamStore';
import { DemoControls } from './TimelineBar';
import type { ViewMode, Tier } from '@/lib/types';
import { getLayerColorEntries } from '@/lib/nodeRendering';
import { NUMERIC_TIER_COLORS as TIER_COLORS, NUMERIC_TIER_LABELS as TIER_LABELS } from '@/lib/constants';

const VIEW_MODES: { mode: ViewMode; label: string; icon: React.ReactNode }[] = [
  { mode: 'explore', label: 'Explore', icon: <Eye className="w-3.5 h-3.5" /> },
  { mode: 'analyze', label: 'Analyze', icon: <Layers className="w-3.5 h-3.5" /> },
  { mode: 'presentation', label: 'Present', icon: <Presentation className="w-3.5 h-3.5" /> },
];

export default function GraphHUD() {
  const viewMode = useUIStore((s) => s.viewMode);
  const setViewMode = useUIStore((s) => s.setViewMode);
  const toggleSearch = useUIStore((s) => s.toggleSearch);
  const focusNode = useUIStore((s) => s.focusNode);
  const nodeCount = useGraphStore((s) => s.nodes.size);
  const edgeCount = useGraphStore((s) => s.edges.size);

  return (
    <div className="absolute inset-0 pointer-events-none z-10" aria-hidden="true">
      {/* Top-left: Title + View Modes */}
      <div className="absolute top-4 left-4 pointer-events-auto flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-[#00E5FF15] border border-[#00E5FF30] flex items-center justify-center">
            <Zap className="w-4 h-4 text-[#00E5FF]" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-[#e2e8f0] tracking-wide">RSVS</h1>
            <p className="text-[9px] text-[#475569] tracking-widest uppercase">Recursive Symbolic Vector Space</p>
          </div>
        </div>

        {/* View Mode Switcher */}
        <div className="flex gap-1 bg-[#0a0e18]/80 backdrop-blur-md rounded-lg p-1 border border-[#1e293b]">
          {VIEW_MODES.map(({ mode, label, icon }) => (
            <button
              key={mode}
              className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[10px] font-medium transition-all ${
                viewMode === mode
                  ? 'bg-[#00E5FF15] text-[#00E5FF] shadow-[0_0_12px_#00E5FF20]'
                  : 'text-[#64748b] hover:text-[#94a3b8] hover:bg-[#1e293b]'
              }`}
              onClick={() => setViewMode(mode)}
            >
              {icon}
              {label}
            </button>
          ))}
        </div>

        {/* Stats */}
        <div className="flex gap-3 text-[10px] font-mono text-[#475569]">
          <span>{nodeCount} nodes</span>
          <span className="text-[#1e293b]">|</span>
          <span>{edgeCount} edges</span>
        </div>
      </div>

      {/* Top-right: Actions */}
      <div className="absolute top-4 right-4 pointer-events-auto flex flex-col gap-2 items-end">
        <div className="flex gap-1.5">
          <Button
            variant="ghost" size="icon"
            className="w-8 h-8 bg-[#0a0e18]/80 backdrop-blur-md border border-[#1e293b] text-[#64748b] hover:text-[#00E5FF] hover:border-[#00E5FF40]"
            onClick={toggleSearch}
            aria-label="Quick search"
            title="Quick Search (Ctrl+K)"
          >
            <Search className="w-3.5 h-3.5" />
          </Button>
          <Button
            variant="ghost" size="icon"
            className="w-8 h-8 bg-[#0a0e18]/80 backdrop-blur-md border border-[#1e293b] text-[#64748b] hover:text-[#94a3b8] hover:border-[#334155]"
            onClick={() => focusNode(null)}
            aria-label="Reset camera"
            title="Reset Camera"
          >
            <Home className="w-3.5 h-3.5" />
          </Button>
          <DemoControls />
        </div>
      </div>

      {/* Bottom-left: Legend */}
      <div className="absolute bottom-14 left-4 pointer-events-auto">
        <div className="bg-[#0a0e18]/80 backdrop-blur-md rounded-lg p-3 border border-[#1e293b]">
          <div className="text-[9px] uppercase tracking-widest text-[#475569] mb-2">Legend</div>
          <div className="flex flex-col gap-1.5">
            {/* v5.0: Layer colors (replacing tier-based legend) */}
            {getLayerColorEntries().map(({ layer, color, label }) => (
              <div key={layer} className="flex items-center gap-2 text-[10px] text-[#94a3b8]">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                {label}
              </div>
            ))}
            <div className="w-full h-px bg-[#1e293b] my-1" />
            <div className="flex items-center gap-2 text-[10px] text-[#94a3b8]">
              <div className="w-2.5 h-2.5 rounded-md" style={{ backgroundColor: '#B388FF' }} />
              Seed node
            </div>
            <div className="flex items-center gap-2 text-[10px] text-[#94a3b8]">
              <div className="w-5 h-0.5 rounded-full" style={{ backgroundColor: '#89D7FF', opacity: 0.7 }} />
              Bootstrap edge
            </div>
            <div className="flex items-center gap-2 text-[10px] text-[#94a3b8]">
              <div className="w-5 h-0.5 rounded-full" style={{ backgroundColor: '#69F0AE', opacity: 0.7 }} />
              Learned edge
            </div>
            <div className="flex items-center gap-2 text-[10px] text-[#94a3b8]">
              <div className="w-5 h-0.5 rounded-full" style={{ backgroundColor: '#FF80AB', opacity: 0.7 }} />
              Composition link
            </div>
          </div>
        </div>
      </div>

      {/* Bottom-right: Keyboard hints */}
      <div className="absolute bottom-14 right-4 pointer-events-none">
        <div className="text-[9px] text-[#334155] font-mono text-right space-y-0.5">
          <div>Double-click → Focus node</div>
          <div>Click → Inspect</div>
          <div>Ctrl+K → Search</div>
          <div>Scroll → Zoom</div>
        </div>
      </div>
    </div>
  );
}
