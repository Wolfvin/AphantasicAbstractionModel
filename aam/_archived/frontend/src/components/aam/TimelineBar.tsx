'use client';

import React, { useCallback, useMemo, useRef, useEffect } from 'react';
import {
  Play, Pause, SkipForward, SkipBack, Home, Gauge,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useTimelineStore, useGraphStore, useChatStore, useUIStore } from '@/store/aamStore';
import { generateInitialSnapshot, generateEventStream, generateChatMessages, generateTimelineEvents } from '@/lib/mockData';
import type { RSVSEvent, RSVSNode, RSVSEdge } from '@/lib/types';

const EVENT_COLORS: Record<string, string> = {
  atom_created: '#69F0AE',
  atom_removed: '#FF5252',
  edge_created: '#80D8FF',
  edge_removed: '#FF5252',
  edge_weight_changed: '#FFD740',
  tier_changed: '#FFB74D',
  confidence_changed: '#B388FF',
  sense_changed: '#80CBC4',
};

const EVENT_ICONS: Record<string, string> = {
  atom_created: '+',
  atom_removed: '−',
  edge_created: '⇌',
  edge_removed: '✂',
  edge_weight_changed: '↕',
  tier_changed: '△',
  confidence_changed: '◈',
  sense_changed: '◎',
};

export default function TimelineBar() {
  const timelineState = useTimelineStore((s) => s.timelineState);
  const playbackSpeed = useTimelineStore((s) => s.playbackSpeed);
  const currentEventIndex = useTimelineStore((s) => s.currentEventIndex);
  const timelineEvents = useTimelineStore((s) => s.timelineEvents);
  const togglePlayPause = useTimelineStore((s) => s.togglePlayPause);
  const setSpeed = useTimelineStore((s) => s.setSpeed);
  const stepForward = useTimelineStore((s) => s.stepForward);
  const stepBackward = useTimelineStore((s) => s.stepBackward);
  const seekTo = useTimelineStore((s) => s.seekTo);
  const addTimelineEvent = useTimelineStore((s) => s.addTimelineEvent);

  const trackRef = useRef<HTMLDivElement>(null);

  const progress = timelineEvents.length > 0
    ? ((currentEventIndex + 1) / timelineEvents.length) * 100
    : 0;

  const handleTrackClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!trackRef.current || timelineEvents.length === 0) return;
      const rect = trackRef.current.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const index = Math.floor(ratio * timelineEvents.length);
      seekTo(index);
    },
    [timelineEvents.length, seekTo]
  );

  const speeds = [0.5, 1, 2];

  return (
    <div
      className="flex items-center gap-3 px-4 py-2 border-t border-[#1e293b]"
      style={{ backgroundColor: '#080c14' }}
      role="region"
      aria-label="Timeline controls"
    >
      {/* Play / Pause */}
      <Button
        variant="ghost" size="icon"
        className="w-8 h-8 text-[#94a3b8] hover:text-[#00E5FF] hover:bg-[#00E5FF10]"
        onClick={togglePlayPause}
        aria-label={timelineState === 'live' ? 'Pause' : 'Play'}
      >
        {timelineState === 'live' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
      </Button>

      {/* Step Back / Forward */}
      <Button
        variant="ghost" size="icon"
        className="w-7 h-7 text-[#64748b] hover:text-[#94a3b8] hover:bg-[#1e293b]"
        onClick={stepBackward}
        aria-label="Step backward"
        disabled={currentEventIndex <= 0}
      >
        <SkipBack className="w-3.5 h-3.5" />
      </Button>
      <Button
        variant="ghost" size="icon"
        className="w-7 h-7 text-[#64748b] hover:text-[#94a3b8] hover:bg-[#1e293b]"
        onClick={stepForward}
        aria-label="Step forward"
        disabled={currentEventIndex >= timelineEvents.length - 1}
      >
        <SkipForward className="w-3.5 h-3.5" />
      </Button>

      {/* Timeline Track */}
      <div className="flex-1 flex flex-col gap-1">
        <div
          ref={trackRef}
          className="relative h-2 bg-[#1e293b] rounded-full cursor-pointer group"
          onClick={handleTrackClick}
          role="slider"
          aria-valuemin={0}
          aria-valuemax={timelineEvents.length - 1}
          aria-valuenow={currentEventIndex}
          aria-label="Event timeline"
        >
          {/* Event dots */}
          {timelineEvents.map((evt, i) => (
            <div
              key={evt.event_id}
              className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full transition-all"
              style={{
                left: `${(i / timelineEvents.length) * 100}%`,
                backgroundColor: EVENT_COLORS[evt.event_type] || '#64748b',
                opacity: i <= currentEventIndex ? 1 : 0.3,
                transform: `translateY(-50%) scale(${i <= currentEventIndex ? 1 : 0.7})`,
              }}
            />
          ))}
          {/* Progress fill */}
          <div
            className="absolute top-0 left-0 h-full rounded-full transition-all duration-200"
            style={{
              width: `${progress}%`,
              background: 'linear-gradient(90deg, #00E5FF, #B388FF)',
            }}
          />
          {/* Playhead */}
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-[#00E5FF] shadow-[0_0_8px_#00E5FF80] transition-all duration-200"
            style={{ left: `calc(${progress}% - 6px)` }}
          />
        </div>
        {/* Event label */}
        {currentEventIndex >= 0 && currentEventIndex < timelineEvents.length && (
          <div className="text-[10px] text-[#64748b] font-mono truncate">
            <span style={{ color: EVENT_COLORS[timelineEvents[currentEventIndex].event_type] || '#64748b' }}>
              {EVENT_ICONS[timelineEvents[currentEventIndex].event_type] || '●'}
            </span>
            {' '}{timelineEvents[currentEventIndex].label}
          </div>
        )}
      </div>

      {/* Event count */}
      <div className="text-[10px] text-[#475569] font-mono min-w-[60px] text-right">
        {currentEventIndex + 1} / {timelineEvents.length}
      </div>

      {/* Speed selector */}
      <div className="flex gap-0.5">
        {speeds.map((s) => (
          <button
            key={s}
            className={`px-1.5 py-0.5 text-[10px] font-mono rounded transition-colors ${
              playbackSpeed === s
                ? 'bg-[#00E5FF20] text-[#00E5FF]'
                : 'text-[#475569] hover:text-[#94a3b8] hover:bg-[#1e293b]'
            }`}
            onClick={() => setSpeed(s)}
          >
            {s}x
          </button>
        ))}
      </div>

      {/* Live indicator */}
      {timelineState === 'live' && (
        <Badge className="text-[9px] bg-[#69F0AE20] text-[#69F0AE] border-[#69F0AE40] px-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[#69F0AE] mr-1 animate-pulse" />
          LIVE
        </Badge>
      )}
    </div>
  );
}

export function DemoControls() {
  const addNode = useGraphStore((s) => s.addNode);
  const addEdge = useGraphStore((s) => s.addEdge);
  const updateNode = useGraphStore((s) => s.updateNode);
  const loadSnapshot = useGraphStore((s) => s.loadSnapshot);
  const pushEvent = useGraphStore((s) => s.pushEvent);
  const nodes = useGraphStore((s) => s.nodes);
  const addMessage = useChatStore((s) => s.addMessage);
  const addTimelineEvent = useTimelineStore((s) => s.addTimelineEvent);
  const resetTimeline = useTimelineStore((s) => s.resetTimeline);
  const selectNode = useUIStore((s) => s.selectNode);
  const closeDrawer = useUIStore((s) => s.closeDrawer);

  const handleLoadSample = useCallback(() => {
    // Generate initial snapshot
    const { nodes: newNodes, edges: newEdges } = generateInitialSnapshot(30);
    loadSnapshot(newNodes, newEdges);

    // Reset timeline
    resetTimeline();

    // Generate events
    const events = generateEventStream(newNodes, 20);
    events.forEach((evt) => {
      pushEvent(evt);
      const tlEvents = generateTimelineEvents([evt]);
      tlEvents.forEach((tle) => addTimelineEvent(tle));

      // Apply event to graph
      switch (evt.event_type) {
        case 'atom_created': {
          // Demo events always contain complete node objects
          const node = evt.payload.node as RSVSNode | undefined;
          if (node?.id) {
            addNode(node);
          }
          break;
        }
        case 'edge_created': {
          const edge = evt.payload.edge as RSVSEdge | undefined;
          if (edge) {
            addEdge(edge);
          }
          break;
        }
        case 'confidence_changed': {
          const nodeId = evt.payload.node?.id;
          const newConf = evt.payload.after?.confidence as number | undefined;
          if (nodeId != null && newConf != null) {
            updateNode(nodeId, { confidence: newConf });
          }
          break;
        }
      }
    });

    // Add chat message
    addMessage({
      id: `chat_demo_${Date.now()}`,
      type: 'system_ingest_status',
      content: `Demo loaded: ${newNodes.length} nodes, ${newEdges.length} edges, ${events.length} events queued.`,
      timestamp: new Date().toISOString(),
    });
  }, [loadSnapshot, resetTimeline, generateEventStream, pushEvent, addNode, addEdge, updateNode, addMessage, addTimelineEvent]);

  const handleClearGraph = useCallback(() => {
    loadSnapshot([], []);
    resetTimeline();
    closeDrawer();
    selectNode(null);
  }, [loadSnapshot, resetTimeline, closeDrawer, selectNode]);

  return (
    <div className="flex gap-2">
      <Button
        variant="ghost" size="sm"
        className="text-[10px] text-[#69F0AE] hover:bg-[#69F0AE10] border border-[#69F0AE30] rounded-full px-3"
        onClick={handleLoadSample}
      >
        <Play className="w-3 h-3 mr-1" /> Load Demo
      </Button>
      <Button
        variant="ghost" size="sm"
        className="text-[10px] text-[#FF5252] hover:bg-[#FF525210] border border-[#FF525230] rounded-full px-3"
        onClick={handleClearGraph}
      >
        Clear
      </Button>
    </div>
  );
}
