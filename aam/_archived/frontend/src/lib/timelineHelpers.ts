/**
 * Timeline event helpers — converts backend RSVSEvent objects into UI TimelineEvent format.
 */

import type { RSVSEvent, TimelineEvent } from '@/lib/types';

/**
 * Convert backend RSVSEvent array into UI TimelineEvent array.
 */
export function generateTimelineEvents(events: RSVSEvent[]): TimelineEvent[] {
  return events.map((evt) => ({
    event_id: evt.event_id,
    timestamp: evt.timestamp,
    event_type: evt.event_type,
    label: formatEventLabel(evt),
    data: evt.payload,
  }));
}

/**
 * Format a human-readable label for a timeline event.
 */
function formatEventLabel(evt: RSVSEvent): string {
  const nodeLabel = evt.payload?.node?.label ?? '?';
  const edgeId = evt.payload?.edge?.id ?? '?';
  switch (evt.event_type) {
    case 'atom_created':
      return `Node "${nodeLabel}" created`;
    case 'edge_created':
      return `Edge ${edgeId} connected`;
    case 'confidence_changed':
      return `Confidence updated for ${nodeLabel}`;
    case 'edge_weight_changed':
      return `Weight changed for edge ${edgeId}`;
    case 'tier_changed':
      return `Tier changed for ${nodeLabel}`;
    case 'atom_removed':
      return `Node ${nodeLabel} removed`;
    case 'sense_changed':
      return `Sense updated for ${nodeLabel}`;
    default:
      return evt.event_type;
  }
}
