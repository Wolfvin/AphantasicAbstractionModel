# RSVS UI Interface Design Prompt (For External UI Agent)

## THINK Gate
- judgment_summary: Tujuan user jelas: UI RSVS bergaya neural brain dengan 3D graph interaktif, event animation lifecycle, panel detail kanan slide-in, input/chat kiri, dan spesifikasi cukup detail untuk dieksekusi agent lain.
- decision: Berikan design prompt yang preskriptif (bukan ide umum), lengkap sampai layout, state model, animation choreography, interaction contract, visual system, performance budget, accessibility, dan acceptance criteria.
- confidence: high
- next_step: Deliver prompt siap-copy untuk UI agent termasuk struktur deliverables dan definisi done.
- recommended_stance: agree
- goal_alignment: pass
- assumption_risk: medium (stack final frontend belum dipatok; prompt dibuat framework-agnostic dengan rekomendasi Three.js + React)

## SMART-PLAN (Compact)
- goal: Menghasilkan dokumen prompt UI detail untuk implementasi antarmuka RSVS 3D graph.
- scope_in:
  - IA/layout: left input chat, center 3D graph, right detail drawer
  - 3D graph visual language + motion system
  - event-driven animation: atom created, edge created, atom removed, edge updated
  - interaction patterns, filters, timeline, playback
  - component contract + data contract + performance + a11y
  - acceptance criteria implementable
- scope_out:
  - implementasi kode final UI
  - keputusan backend API final di luar kebutuhan kontrak UI
- phases:
  - phase_1 (done): Definisikan objective UX, mental model, dan layout utama
  - phase_2 (done): Definisikan behavior graph 3D + animation choreography
  - phase_3 (done): Definisikan interaction contract + panel behavior + chat flow
  - phase_4 (done): Definisikan design tokens, perf budget, a11y, dan definition of done
- risks:
  - overload visual (terlalu ramai sehingga sulit dibaca)
  - frame drop saat node/edge besar
  - event burst membuat animasi tabrakan
  - ambiguity antara "semantic change" vs "cosmetic update"
- verification:
  - checklist acceptance criteria terpenuhi
  - walkthrough 5 task utama user sukses tanpa bantuan
  - FPS target tercapai untuk dataset medium
  - keyboard navigation + reduced motion bekerja
- next_step: Gunakan prompt di bawah untuk minta agent UI membuat wireframe, motion spec, dan implementasi awal.

---

## MASTER PROMPT FOR UI AGENT

Design and implement a production-ready interface for **RSVS (Recursive Symbolic Vector Space)** with a strong “living neural brain” visual identity.

### 1) Product Intent
Build a UI where users can:
- Feed text/corpus into RSVS from a left-side chat/input lane.
- Watch atoms and relationships form in real-time as a **3D connected graph** in the center.
- Click any ID/atom/composite node to open a **right-side slide-in detail panel**.
- Understand change over time (created, connected, removed, confidence/tier shifts) through clear, meaningful animation.

The graph is not decorative; it is the main instrument panel.

### 2) Core Layout (Desktop First)
Use a 3-column shell:
- Left Rail (Input + Controls): 22-26% width
- Center Stage (3D Graph): flexible dominant area
- Right Drawer (Node Detail): 28-34% width, hidden by default, slide-in on node select

Desktop behavior:
- Left rail is persistent.
- Center graph is always visible.
- Right drawer overlays/occupies space with smooth transition.

Mobile/tablet behavior:
- Single canvas focus mode by default.
- Left and right panels become bottom sheets/drawers.
- Keep interaction parity (tap select node, open details, submit input).

### 3) Visual Direction
Target mood: “cognitive engine” + “organic computation”.
- Background: deep gradient + subtle volumetric haze + low-contrast grid/noise.
- Graph palette:
  - Tier 1 atoms: stable cool color
  - Tier 2 atoms: alert/warm neutral
  - Tier 3 atoms: constrained/critical color
  - Composite IDs: distinct hue family from atoms
- Edge style:
  - Thickness by weight/confidence
  - Pulse intensity by recent activity
- Use bloom/glow conservatively; readability first.
- Typography must feel technical and intentional.

### 4) 3D Graph System (Three.js)
Implement a force-directed 3D graph with:
- Node types: Atom, Composite
- Node state: id, label, tier, confidence, status (new/stable/decaying/removed)
- Edge state: source, target, weight, source_type (bootstrap/learned), status

Camera:
- Orbit controls with damping
- Double-click node = focus/zoom to node cluster
- “Home” action to reset camera

Interaction:
- Hover node: highlight immediate neighbors + show mini tooltip
- Click node: freeze highlight + open right drawer with full metadata
- Shift+click multi-select for comparison mode (optional but preferred)

### 5) Animation Choreography (Mandatory)
#### A. Atom Created
- Spawn effect: node appears from low-scale (0.2 -> 1.0), opacity fade-in, short glow burst
- Duration: 400-700ms
- Then settle into physics simulation

#### B. Atom Connected (new edge)
- Draw-on edge animation from source to target (traveling light)
- Edge alpha ramps 0 -> target opacity
- Small synchronized pulse on both nodes
- Duration: 300-600ms

#### C. Atom Removed
- Node enters “decay” phase first (desaturate + flicker low amplitude)
- Connected edges retract/fade first, then node shrinks and dissolves
- Duration: 600-1100ms
- Optional ghost trace for 1-2s in timeline mode

#### D. Edge Updated
- If weight increases: thickness + brightness pulse upward
- If weight decreases: soften thickness + color cooling
- Avoid harsh blinking; keep semantic smoothness

#### E. Global Event Burst Handling
When many events arrive at once:
- Queue and batch animations in micro-windows (e.g., 120-200ms)
- Prioritize visibility near selected/focused node
- Defer off-screen heavy effects

### 6) Right Drawer (Slide-In Detail)
Open when user selects node.

Sections:
1. Header: label, ID, type badge, tier badge
2. Confidence block: value + sparkline/trend + status text
3. Atom/Composite composition:
   - If composite: member atoms with weights
   - If atom: top related composites
4. Sense block (if available): active sense, sense count, coherence
5. Connectivity block: degree, strongest edges, recent changes
6. Actions:
   - Focus in graph
   - Pin node
   - Compare
   - Export node snapshot

Animation:
- Slide from right with opacity + slight blur-to-sharp
- 220-320ms, ease-out

### 7) Left Rail (Input as Chat)
Provide chat-like ingestion flow:
- Input box (multiline)
- Send button
- Optional attachment/file ingest
- History bubbles with system feedback

Message types:
- user_input
- system_ingest_status
- system_promoted_atoms
- system_warnings

Each submission can trigger graph updates in real-time.
Include clear loading/progress state for ingest.

### 8) Timeline + Playback (Required)
Add a timeline control to inspect evolution:
- Live mode (default)
- Pause/resume stream
- Step forward/back event
- Playback speed 0.5x/1x/2x

Timeline events:
- atom_created
- edge_created
- atom_removed
- edge_weight_changed
- tier_changed
- confidence_changed

### 9) Filtering & Modes
Provide graph filters:
- By tier
- By confidence range
- By node type
- By domain/source batch
- By recent activity window

Modes:
- Explore mode (default)
- Analyze mode (more metrics overlays)
- Presentation mode (clean cinematic view)

### 10) Information Architecture / Component Contract
Design components with clear ownership:
- `AppShell`
- `LeftInputRail`
- `GraphScene3D`
- `GraphHUD` (legend, metrics, camera actions)
- `RightNodeDrawer`
- `TimelineBar`
- `EventToAnimationOrchestrator`
- `StateStore` (graph state + ui state)

State separation:
- Graph domain state (nodes, edges, events)
- UI state (selected node, drawer open, filters, camera presets)
- Animation transient state (queues, active effects)

### 11) Data Contract (UI-facing)
Assume incoming event stream like:
- `event_id`
- `timestamp`
- `event_type`
- `payload`
- `correlation_id` (input message to resulting graph changes)

Node payload minimal:
- `id`, `label`, `kind`, `tier`, `confidence`, `sense_count`, `coherence`

Edge payload minimal:
- `source`, `target`, `weight`, `source_type`

### 12) Performance Targets
- 60 FPS target for 500-1,500 nodes on desktop GPU class menengah
- 30+ FPS minimum graceful mode for larger graphs
- Use instancing/batching where possible
- LOD strategy:
  - Full detail near focus
  - Simplified rendering for distant clusters
- Throttle expensive labels; show labels contextually

### 13) Accessibility + Safety
- Keyboard support for all core actions (node traversal, drawer open/close, timeline)
- Reduced motion mode:
  - Disable burst effects
  - Replace with low-motion fades
- Color contrast minimum for text and HUD
- Tooltips readable, not hover-only on touch devices

### 14) Micro-Interactions
- Hover halos for neighbor discovery
- Context breadcrumbs when drilling into cluster
- Quick-search ID/label (Ctrl/Cmd+K)
- Mini-map/overview optional for large graphs

### 15) Error & Empty States
- Empty graph state with onboarding instructions
- Ingest error state with retry and diagnostics
- Partial data state handling (missing fields should not crash scene)

### 16) Deliverables from UI Agent
Return all of these:
1. High-fidelity screen spec (desktop + mobile)
2. Motion spec sheet (timings, easing, per event type)
3. Component tree + state diagram
4. Event-to-animation mapping table
5. Interaction map for 5 critical user flows
6. Implementation scaffold recommendation (e.g., React + Three.js + Zustand)
7. Risk list + mitigation plan

### 17) Definition of Done
Consider the design done only if:
- 3D graph is interactive and semantically readable
- All mandatory lifecycle animations are specified and demonstrated
- Right slide-in detail panel works from node selection
- Left chat input drives visible graph updates
- Timeline playback works for graph evolution events
- Desktop + mobile behavior specified
- Performance and accessibility constraints addressed explicitly

### 18) Non-Negotiables
- No generic dashboard look.
- Graph must feel alive but controlled (not chaotic VFX).
- Animation must communicate meaning, not decoration.
- UI must support real analysis workflow, not just visual spectacle.

### 19) Detailed Data Shapes (YAML + JSON)
Use these contracts as canonical UI wire format.

#### 19.1 Atom / Node Schema (YAML)
```yaml
node:
  id: uint32                 # required, integer RSVS ID
  label: string              # required, human-readable name
  kind: atom|composite       # required
  tier: 1|2|3                # required
  confidence: float          # required, 0.0..1.0
  status: new|stable|decaying|removed  # required for animation state

  sense:
    count: int               # number of senses
    active_index: int|null   # selected sense index for current context
    coherence: float|null    # 0.0..1.0

  metrics:
    degree: int              # total connected edges
    in_degree: int
    out_degree: int
    last_updated_at: string  # ISO-8601

  composition:               # for composite or atom detail enrichment
    atoms:                   # used when kind=composite
      - atom_id: uint32
        weight: float        # local contribution weight
    related_composites:      # used when kind=atom
      - composite_id: uint32
        weight: float

  render:
    position:
      x: float
      y: float
      z: float
    size: float              # node radius base
    color: string            # hex color
    glow: float              # 0..1

  provenance:
    source_batch_id: string|null
    source_domain: string|null
    source_type: bootstrap|learned
```

#### 19.2 Edge Schema (YAML)
```yaml
edge:
  id: string                 # e.g. "12->88"
  source: uint32             # atom/composite ID
  target: uint32             # atom/composite ID
  direction: directed|undirected
  weight: float              # 0.0..1.0
  source_type: bootstrap|learned
  status: new|stable|updated|removing

  metrics:
    cooc: float|null
    npmi: float|null
    jaccard: float|null
    last_updated_at: string

  render:
    thickness: float
    color: string
    opacity: float
    pulse: float             # 0..1
```

#### 19.3 Event Stream Schema (YAML)
```yaml
event:
  event_id: string
  timestamp: string          # ISO-8601
  correlation_id: string     # ties to one user input/chat message
  event_type:                # animation driver
    atom_created|
    atom_removed|
    edge_created|
    edge_removed|
    edge_weight_changed|
    tier_changed|
    confidence_changed|
    sense_changed

  payload:
    node: {}                 # Node snapshot (when node-affecting event)
    edge: {}                 # Edge snapshot (when edge-affecting event)
    before: {}               # optional previous values
    after: {}                # optional new values

  animation_hint:
    priority: low|normal|high
    focus_node_id: uint32|null
    burst_group: string|null
```

#### 19.4 Graph Snapshot Schema (JSON)
```json
{
  "snapshot_id": "snap_2026-04-21T10:15:30.000Z",
  "generated_at": "2026-04-21T10:15:30.000Z",
  "context": {
    "domain": "geology",
    "batch_id": "ingest_00047",
    "input_message_id": "msg_01JS9VYJ"
  },
  "nodes": [
    {
      "id": 41,
      "label": "stone",
      "kind": "atom",
      "tier": 2,
      "confidence": 0.82,
      "status": "stable",
      "sense": {
        "count": 2,
        "active_index": 0,
        "coherence": 0.71
      },
      "metrics": {
        "degree": 7,
        "in_degree": 2,
        "out_degree": 5,
        "last_updated_at": "2026-04-21T10:15:27.218Z"
      },
      "composition": {
        "atoms": [],
        "related_composites": [
          { "composite_id": 301, "weight": 0.66 }
        ]
      },
      "render": {
        "position": { "x": -12.4, "y": 2.1, "z": 8.7 },
        "size": 1.15,
        "color": "#5BC0FF",
        "glow": 0.4
      },
      "provenance": {
        "source_batch_id": "ingest_00047",
        "source_domain": "geology",
        "source_type": "learned"
      }
    }
  ],
  "edges": [
    {
      "id": "41->301",
      "source": 41,
      "target": 301,
      "direction": "directed",
      "weight": 0.66,
      "source_type": "learned",
      "status": "stable",
      "metrics": {
        "cooc": 0.54,
        "npmi": 0.37,
        "jaccard": 0.42,
        "last_updated_at": "2026-04-21T10:15:27.218Z"
      },
      "render": {
        "thickness": 1.8,
        "color": "#89D7FF",
        "opacity": 0.72,
        "pulse": 0.2
      }
    }
  ]
}
```

#### 19.5 Event Examples (JSON)
```json
[
  {
    "event_id": "evt_1001",
    "timestamp": "2026-04-21T10:16:02.100Z",
    "correlation_id": "msg_01JS9VYJ",
    "event_type": "atom_created",
    "payload": {
      "node": {
        "id": 88,
        "label": "mineral",
        "kind": "atom",
        "tier": 2,
        "confidence": 0.50,
        "status": "new"
      }
    },
    "animation_hint": {
      "priority": "high",
      "focus_node_id": 88,
      "burst_group": "ingest_00048"
    }
  },
  {
    "event_id": "evt_1002",
    "timestamp": "2026-04-21T10:16:02.420Z",
    "correlation_id": "msg_01JS9VYJ",
    "event_type": "edge_created",
    "payload": {
      "edge": {
        "id": "88->301",
        "source": 88,
        "target": 301,
        "direction": "directed",
        "weight": 0.61,
        "source_type": "learned",
        "status": "new"
      }
    },
    "animation_hint": {
      "priority": "high",
      "focus_node_id": 88,
      "burst_group": "ingest_00048"
    }
  },
  {
    "event_id": "evt_1003",
    "timestamp": "2026-04-21T10:16:04.050Z",
    "correlation_id": "msg_01JS9VYJ",
    "event_type": "confidence_changed",
    "payload": {
      "node": { "id": 41, "label": "stone" },
      "before": { "confidence": 0.82 },
      "after": { "confidence": 0.86 }
    },
    "animation_hint": {
      "priority": "normal",
      "focus_node_id": 41,
      "burst_group": "ingest_00048"
    }
  }
]
```

#### 19.6 UI Mapping Rules (Wiring Contract)
- `event_type=atom_created` -> run Spawn animation (scale-in + glow burst)
- `event_type=edge_created` -> run Draw-on edge animation + dual node pulse
- `event_type=atom_removed` -> run Decay then dissolve sequence
- `event_type=edge_weight_changed` -> adjust edge thickness/color with easing
- `event_type=tier_changed` -> color family transition + badge update in drawer
- `event_type=confidence_changed` -> node ring meter pulse + drawer sparkline update

#### 19.7 Validation Rules
- Reject node if missing required: `id`, `label`, `kind`, `tier`, `confidence`
- Clamp confidence and weight to `0..1`
- Unknown `event_type` must not crash renderer; log + ignore
- If `source/target` node not present yet, hold edge event in pending queue up to `2s`
- Payload versioning (recommended): include `schema_version` in snapshot/event envelope
