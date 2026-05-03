# RSVS Frontend

3D knowledge graph visualization for the RSVS (Relational Symbolic Vector Space) system.

## Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 16 (App Router) |
| Language | TypeScript 5 (strict mode) |
| Styling | Tailwind CSS 4 + shadcn/ui |
| 3D Rendering | Three.js / React Three Fiber |
| State | Zustand (client) |
| Animations | Framer Motion |
| Testing | Vitest + Testing Library |

## Development Setup

```bash
# Install dependencies
bun install

# Start dev server (port 3000)
bun run dev

# Run linting
bun run lint
```

### Environment Variables

Copy `.env.example` to `.env.local` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend REST API base URL |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000` | Backend WebSocket URL |
| `NEXT_PUBLIC_RSVS_BACKEND_URL` | `http://127.0.0.1:8787` | RSVS bridge server URL |

## Testing

```bash
# Run tests once
bun run test

# Run tests in watch mode
bun run test:watch

# Run tests with coverage
bun run test:coverage
```

Tests are located in `src/**/__tests__/` directories alongside the code they test.

## Architecture Decisions

### Frontend-Only Rendering
The frontend computes all visual properties (node size, color, glow, position) independently from the backend. The bridge server no longer provides `render` keys — the `nodeRendering.ts` utility handles this using node semantic data (tier, status, confidence).

### Mock Data Gating
Mock data (`mockData.ts`) is only loaded in development mode (`NODE_ENV === 'development'`). In production, all data comes from the RSVS bridge server. The `simulateIngestResponse` fallback is also gated behind a development check.

### Force-Directed Layout
The 3D graph uses a custom force-directed layout (`ForceGraph.tsx`) with tier-weighted repulsion, edge-weight-based attraction, and seed-node center gravity. This replaces any bridge-provided positions.

### Store Architecture
Zustand stores are split by domain:
- **GraphStore** — nodes, edges, events, snapshots
- **UIStore** — selection, drawer, view mode, search
- **TimelineStore** — playback, events, seeking
- **ChatStore** — message list
- **FilterStore** — tier/confidence/kind filters
- **AnimationStore** — animation queue
- **ModeResultStore** — appraise/relate results
