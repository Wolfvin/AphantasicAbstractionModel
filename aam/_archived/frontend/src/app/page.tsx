'use client';

import React, { useEffect, useCallback, useState, useRef, Suspense } from 'react';
import dynamic from 'next/dynamic';
import { Skeleton } from '@/components/ui/skeleton';
import LeftInputRail from '@/components/aam/LeftInputRail';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import GraphHUD from '@/components/aam/GraphHUD';
import { useUIStore, useGraphStore, useAnimationStore, useTimelineStore, useChatStore, useModeResultStore } from '@/store/aamStore';
import { fetchLatestFromBackend } from '@/lib/backendBridge';
// Mock data is used as a fallback when the backend is unreachable (demo mode).
import { generateTimelineEvents } from '@/lib/timelineHelpers';
import { generateInitialSnapshot, generateEventStream, generateTimelineEvents as generateMockTimelineEvents } from '@/lib/mockData';

// Dynamic imports for heavy components (lazy loaded for performance)
const GraphScene3D = dynamic(
  () => import('@/components/aam/graph3d/GraphScene3D'),
  {
    ssr: false,
    loading: () => <GraphSkeleton />,
  }
);

const RightNodeDrawer = dynamic(() => import('@/components/aam/RightNodeDrawer'));
const TimelineBar = dynamic(() => import('@/components/aam/TimelineBar'));

// ── Skeleton loading components ──
function GraphSkeleton() {
  return (
    <div className="flex-1 flex items-center justify-center bg-[#060a12]" aria-label="Loading 3D graph" role="status">
      <div className="text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full border-2 border-[#00E5FF30] border-t-[#00E5FF] animate-spin" />
        <p className="text-sm text-[#64748b] font-mono">Initializing 3D engine...</p>
      </div>
    </div>
  );
}

function RailSkeleton() {
  return (
    <div className="h-full w-[300px] bg-[#0d1117] border-r border-[#1b2332] p-4 space-y-4" role="status" aria-label="Loading input panel">
      <Skeleton className="h-8 w-32" />
      <div className="space-y-3">
        <Skeleton className="h-16 w-full rounded-xl" />
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
      </div>
      <Skeleton className="h-20 w-full rounded-xl" />
    </div>
  );
}

function LoadingGraph() {
  return (
    <div className="flex-1 flex items-center justify-center rsvs-bg rsvs-grid relative" aria-label="Loading graph" role="status">
      {/* Animated loading rings */}
      <div className="relative">
        <div className="w-32 h-32 rounded-full border border-[#00E5FF15] animate-ping" style={{ animationDuration: '3s' }} />
        <div className="w-24 h-24 rounded-full border border-[#B388FF15] animate-ping absolute top-4 left-4" style={{ animationDuration: '2s' }} />
        <div className="w-16 h-16 rounded-full border border-[#69F0AE15] animate-ping absolute top-8 left-8" style={{ animationDuration: '1.5s' }} />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-2xl font-bold text-[#00E5FF] mb-1 animate-pulse">RSVS</div>
            <p className="text-[10px] text-[#475569] tracking-widest uppercase">Loading Neural Graph</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  const nodeCount = useGraphStore((s) => s.nodes.size);
  if (nodeCount > 0) return null;

  return (
    <div className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none">
      <div className="text-center max-w-md">
        <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-[#00E5FF08] border border-[#00E5FF20] flex items-center justify-center">
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
            <circle cx="20" cy="20" r="6" fill="#00E5FF" opacity="0.6" />
            <circle cx="8" cy="10" r="3" fill="#B388FF" opacity="0.5" />
            <circle cx="32" cy="10" r="3" fill="#69F0AE" opacity="0.5" />
            <circle cx="8" cy="30" r="3" fill="#FFB74D" opacity="0.5" />
            <circle cx="32" cy="30" r="3" fill="#FF5252" opacity="0.5" />
            <line x1="20" y1="20" x2="8" y2="10" stroke="#334155" strokeWidth="1" />
            <line x1="20" y1="20" x2="32" y2="10" stroke="#334155" strokeWidth="1" />
            <line x1="20" y1="20" x2="8" y2="30" stroke="#334155" strokeWidth="1" />
            <line x1="20" y1="20" x2="32" y2="30" stroke="#334155" strokeWidth="1" />
          </svg>
        </div>
        <h2 className="text-lg font-semibold text-[#e2e8f0] mb-2">No Graph Data Loaded</h2>
        <p className="text-sm text-[#64748b] leading-relaxed mb-4">
          Start by sending text through the input panel on the left, or click
          <span className="text-[#69F0AE] font-medium"> &quot;Load Demo&quot; </span>
          in the top-right to populate the graph with sample data.
        </p>
        <div className="flex justify-center gap-4 text-[10px] text-[#475569]">
          <span>Type text →</span>
          <span>Watch atoms form</span>
          <span>Explore connections</span>
        </div>
      </div>
    </div>
  );
}

function SearchDialog() {
  const isOpen = useUIStore((s) => s.isSearchOpen);
  const toggleSearch = useUIStore((s) => s.toggleSearch);
  const searchQuery = useUIStore((s) => s.searchQuery);
  const setSearchQuery = useUIStore((s) => s.setSearchQuery);
  const nodes = useGraphStore((s) => s.nodes);
  const selectNode = useUIStore((s) => s.selectNode);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Ctrl+K shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        toggleSearch();
      }
      if (e.key === 'Escape' && isOpen) {
        toggleSearch();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, toggleSearch]);

  const results = searchQuery.length > 0
    ? Array.from(nodes.values()).filter(
        (n) =>
          n.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
          String(n.id).includes(searchQuery)
      ).slice(0, 8)
    : [];

  return (
    <>
      {isOpen && (
        <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={toggleSearch} />
          <div className="relative w-full max-w-lg bg-[#0d1117] border border-[#1e293b] rounded-xl shadow-2xl overflow-hidden">
            <div className="flex items-center gap-3 px-4 py-3 border-b border-[#1e293b]">
              <svg className="w-4 h-4 text-[#64748b]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                ref={inputRef}
                type="text"
                className="flex-1 bg-transparent text-sm text-[#e2e8f0] placeholder-[#475569] outline-none font-mono"
                placeholder="Search nodes by ID or label..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                aria-label="Search nodes by ID or label"
              />
              <kbd className="text-[10px] text-[#475569] bg-[#1e293b] px-1.5 py-0.5 rounded font-mono">ESC</kbd>
            </div>
            {results.length > 0 && (
              <div className="max-h-64 overflow-y-auto rsvs-scrollbar">
                {results.map((node) => (
                  <button
                    key={node.id}
                    className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-[#1e293b] transition-colors text-left"
                    onClick={() => {
                      selectNode(node.id);
                      toggleSearch();
                      setSearchQuery('');
                    }}
                  >
                    <div
                      className="w-3 h-3 rounded-full shrink-0"
                      style={{ backgroundColor: node.render?.color || '#64748b' }}
                    />
                    <span className="text-sm text-[#e2e8f0] font-mono">{node.label}</span>
                    <span className="text-[10px] text-[#475569] font-mono ml-auto">#{node.id}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-mono"
                      style={{
                        backgroundColor: `${node.render?.color || '#64748b'}15`,
                        color: node.render?.color || '#64748b',
                      }}
                    >
                      T{node.tier}
                    </span>
                  </button>
                ))}
              </div>
            )}
            {searchQuery.length > 0 && results.length === 0 && (
              <div className="px-4 py-6 text-center text-sm text-[#475569]">
                No nodes found for &quot;{searchQuery}&quot;
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export default function RSVSApp() {
  const isLeftRailCollapsed = useUIStore((s) => s.isLeftRailCollapsed);
  const toggleLeftRail = useUIStore((s) => s.toggleLeftRail);
  const setReducedMotion = useAnimationStore((s) => s.setReducedMotion);
  const loadSnapshot = useGraphStore((s) => s.loadSnapshot);
  const pushEvent = useGraphStore((s) => s.pushEvent);
  const addTimelineEvent = useTimelineStore((s) => s.addTimelineEvent);
  const resetTimeline = useTimelineStore((s) => s.resetTimeline);
  const addMessage = useChatStore((s) => s.addMessage);

  const [isBackendLoading, setIsBackendLoading] = useState(true);

  // Detect reduced motion preference
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
      setReducedMotion(mq.matches);
      const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
      mq.addEventListener('change', handler);
      return () => mq.removeEventListener('change', handler);
    }
  }, [setReducedMotion]);

  useEffect(() => {
    let cancelled = false;
    setIsBackendLoading(true);
    (async () => {
      try {
        const latest = await fetchLatestFromBackend();
        if (!latest.ok || cancelled) throw new Error('no snapshot');

        loadSnapshot(latest.snapshot.nodes, latest.snapshot.edges);
        resetTimeline();
        latest.events.forEach((evt) => {
          pushEvent(evt);
          generateTimelineEvents([evt]).forEach((tle) => addTimelineEvent(tle));
        });
        (latest.messages || []).forEach((msg) => addMessage(msg));
      } catch {
        // Backend not reachable — load demo data
        if (!cancelled) {
          const { nodes, edges } = generateInitialSnapshot(25);
          loadSnapshot(nodes, edges);
          resetTimeline();
          const events = generateEventStream(nodes, 10);
          events.forEach((evt) => {
            pushEvent(evt);
            generateMockTimelineEvents([evt]).forEach((tle) => addTimelineEvent(tle));
          });
        }
      } finally {
        if (!cancelled) setIsBackendLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [loadSnapshot, resetTimeline, pushEvent, addTimelineEvent, addMessage]);

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden rsvs-bg">
      {/* Main 3-column layout */}
      <div className="flex-1 flex overflow-hidden relative" role="main">
        {/* Left Rail */}
        <div
          className="shrink-0 z-30 transition-all duration-300 ease-out"
          style={{
            width: isLeftRailCollapsed ? '0px' : 'clamp(280px, 22vw, 360px)',
            minWidth: isLeftRailCollapsed ? '0px' : '280px',
          }}
        >
          {!isLeftRailCollapsed && (
            <div className="h-full border-r border-[#1e293b]">
              <ErrorBoundary name="LeftInputRail">
                <LeftInputRail />
              </ErrorBoundary>
            </div>
          )}
        </div>

        {/* Center Stage — 3D Graph */}
        <div className="flex-1 relative min-w-0" aria-label="3D graph visualization">
          {isBackendLoading ? (
            <LoadingGraph />
          ) : (
            <ErrorBoundary name="Graph3D">
              <Suspense fallback={<GraphSkeleton />}>
                <GraphScene3D />
              </Suspense>
            </ErrorBoundary>
          )}
          <GraphHUD />
          <EmptyState />
          {/* Mode result live region for accessibility */}
          <div aria-live="polite" className="sr-only" />
        </div>

        {/* Right Drawer (rendered by RightNodeDrawer component, overlays) */}
        <ErrorBoundary name="RightNodeDrawer">
          <RightNodeDrawer />
        </ErrorBoundary>
      </div>

      {/* Timeline Bar */}
      <ErrorBoundary name="TimelineBar">
        <TimelineBar />
      </ErrorBoundary>

      {/* Search Dialog */}
      <SearchDialog />

      {/* Left rail toggle (visible when collapsed) */}
      {isLeftRailCollapsed && (
        <button
          className="fixed top-4 left-4 z-40 w-10 h-10 rounded-lg bg-[#0a0e18]/90 backdrop-blur-md border border-[#1e293b] text-[#64748b] hover:text-[#00E5FF] hover:border-[#00E5FF40] transition-all flex items-center justify-center"
          onClick={toggleLeftRail}
          aria-label="Open input panel"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M2 3h12v1.5H2V3zm0 4.25h12v1.5H2v-1.5zm0 4.25h12V13H2v-1.5z" />
          </svg>
        </button>
      )}
    </div>
  );
}
