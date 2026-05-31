<!-- NOTE: Design document. Folder names below (e.g. `Persistent/`) are illustrative of
the development-time layout. In an installed plugin your data root is `.continuity/` in your
project (or `$CONTINUITY_HOME`) — see the top-level README.md. The concepts are identical. -->

# Persistent Continuity Architecture — v3 (Cognition-Native Memory)

**Status:** design + core MVP (2026-05-31). Supersedes ARCHITECTURE_V2.md (kept for
lineage). v3 is an *evolution*, not a rewrite: the v2 event spine, runtime projection,
retrieval, reflection and replay-determinism all remain. v3 changes the **organizing
primitive** — from *sessions over a filesystem* to a *self-maintaining cognitive graph
driven by attention*.

> v2 thesis: "memory is rebuilt from events, not preserved as history."
> v3 thesis: "**cognition is reconstructed from a graph, focused by attention** —
> sessions are just a time-window view, not a unit of memory."

---

## 0. What changes vs v2 (one screen)

| Dimension | v2 | v3 |
|---|---|---|
| Primary primitive | session | **cognitive thread** (long-lived, cross-session, overlapping) |
| Organization | session/task grouping | **event graph + thread graph + causal edges** |
| Retrieval | salience-ranked top-K (BM25 × salience) | **attention via spreading activation** (diffusion) over the graph |
| Focus | static composite salience score | **attention state**: decay + reinforcement, persistence, cross-thread activation, dormant resurfacing, competing-hypothesis inhibition |
| Working context | stored file `working_context.md` | **ephemeral projection of the active subgraph at time T** (markdown = debug/export only) |
| Restoration | derive state → render | **graph reconstruction → active-subgraph extraction → synthesize** |
| Replay | chronological fold | **graph state @ T + attention → active subgraph** (determinism at the *event* layer) |
| Storage split | (implicit) | **cognitive layer ⟂ execution layer** kept separate in storage + indexing |
| What replaces session snapshots | YAML session file | **thread digests** (auto-generated per thread) |

**Determinism is preserved**, but moves down a layer: it is now *event-layer* determinism
(same `events.jsonl` + same intent seed ⇒ same reconstructed subgraph), not session replay.

---

## 1. Layered model (revised)

```
┌──────────────────────────────────────────────────────────────────┐
│ L5  PROJECTION (markdown)   cognitive/ execution/ threads/ exports  │ ← regenerated
├──────────────────────────────────────────────────────────────────┤
│ L4  WORKSPACE (ephemeral)   attention-synthesized lens over the     │ ← NOT stored;
│                             active subgraph (replaces working_ctx)  │   regenerated
├──────────────────────────────────────────────────────────────────┤
│ L3  ATTENTION (cognition.db) activation state, diffusion, thread    │ ← rebuilt from
│                             state, active-subgraph extraction       │   L1+L2 (cache)
├──────────────────────────────────────────────────────────────────┤
│ L2  GRAPH (state.db +        nodes(events) · thread membership ·    │ ← replayable
│     cognition.db)            causal edges · thread graph            │   from L1
├──────────────────────────────────────────────────────────────────┤
│ L1  EVENT LOG (events.jsonl) append-only, immutable TRUTH           │ ← only source
│                              (now carries event_id/thread_ids/causes)│   of truth
└──────────────────────────────────────────────────────────────────┘
```

**Invariant (unchanged):** L2–L5 are derivable. Delete them and `replay`/`cognition build`
rebuilds them from L1. Only `events.jsonl` is irreplaceable.

---

## 2. Cognitive threads (Upgrade #1, critical)

A **thread** is a long-lived line of reasoning that spans sessions, may overlap other
threads, and can merge or split. It replaces "session" as the unit memory is organized by.

Threads are themselves **events** (truth stays in the log), so the thread set is replayable:

```json
{"t":"...","event_id":"evt_44","type":"thread_open","thread_id":"salience-ranking",
 "title":"Salience & attention model","topics":["attention","salience"]}
{"t":"...","event_id":"evt_61","type":"thread_merge","thread_id":"attention-model",
 "into":"salience-ranking","reason":"same line of reasoning"}
{"t":"...","event_id":"evt_77","type":"thread_split","thread_id":"salience-ranking",
 "spawn":"dormant-resurfacing","reason":"distinct sub-problem"}
```

Membership is per-event and may be multiple/overlapping:

```json
{"t":"...","event_id":"evt_50","type":"decision",
 "thread_ids":["salience-ranking","memory-reconstruction"],
 "causes":["evt_44"],"layer":"cognitive",
 "msg":"Attention = spreading activation, not top-K"}
```

**Thread lifecycle (derived state, not new truth):** `active → dormant → (resurfaced) →
merged/closed`. Dormancy is computed from `last_activity_t`; resurfacing is an attention
event (§5). Merge/split are explicit events; the graph honors them when building thread nodes.

**Why this satisfies the constraint:** threads improve *long-horizon continuity* (a line of
reasoning survives across many sessions) and *cognitive coherence* (related events cohere
under a stable handle instead of scattering across session files).

---

## 3. Graph model (Upgrade #2)

Session is removed as a cognitive primitive. The graph is built from the event log:

- **Nodes** = events (`event_id`), typed and layered.
- **Edges**:
  - `causal` — `event.causes: [event_id,...]` → directed event→event dependency.
  - `membership` — event ↔ thread (from `thread_ids`).
  - `thread_rel` — thread↔thread: explicit (`thread_merge`/`thread_split`) **plus** derived
    co-membership weight (how often two threads share events).
- **Thread nodes** = derived aggregates (centroid topics, member events, state, salience).

`session_start`/`session_end` still exist but are demoted to **meta-markers** (a time
window label); nothing in restoration keys off them. `active_context.json` continues to
record the *current* project/branch for convenience, derived from the latest markers.

Restoration becomes **graph reconstruction + active-subgraph extraction** (§7), not replay
of a session.

---

## 4. Cognitive ⟂ Execution layers (Upgrade #6)

Every node carries a `layer`:

- **cognitive** — reasoning artifacts: `objective, decision, observation, reflection,
  hypothesis, contradiction, assumption, assumption_invalidated, open_loop, loop_closed,
  topic_shift`.
- **execution** — outputs/side-effects: `artifact, output, api_call, error, command`.
- **meta** — `session_start, session_end, checkpoint, thread_open/merge/split`.

`layer` is inferred from `type` at emit time (overridable per event). Storage and indexing
keep them separate:

- L5 projection dirs split: `cognitive/` vs `execution/`.
- Attention diffusion runs primarily over the **cognitive** subgraph; execution nodes
  attach to it as evidence (an `artifact` lifts the decision that produced it, but reasoning
  does not "spread" through code). This keeps retrieval precise: a code file never out-ranks
  the hypothesis it serves.

---

## 5. Attention system (Upgrades #4 + #5, critical)

Attention replaces the static v2 salience score with a **dynamic activation state** over the
graph. It is *cognitive focus management*, not retrieval.

### 5.1 Base activation
Per node, an emit-time/decaying base (the v2 §8 salience, reused): importance, recency,
unresolved boost, linked degree. This seeds the diffusion.

### 5.2 Spreading activation / diffusion (replaces top-K)
Activation flows outward from seeds along edges, attenuating per hop:

```
A₀ = seed(intent)                         # query BM25 hits + all unresolved loops + active threads
Aₙ₊₁(v) = base(v)·β
        + γ · Σ_{u→v ∈ edges} w(u,v)·Aₙ(u)·decay_hop
```

- seeds: current intent (BM25 candidates), **every unresolved open loop** (bias to finishing),
  and nodes in currently-active threads.
- hop limit ≤ 2–3; edge weights: causal > membership > derived thread_rel.
- thread membership lets a thread's events co-activate (cross-thread activation when an event
  belongs to two threads — the bridge carries activation between them).

### 5.3 Temporal decay + reinforcement
- **decay**: `recency = 0.5 ** (age_days / half_life)` (half-life 7d).
- **reinforcement**: each time a node is *attended* (appears in a generated workspace), its
  reinforcement counter increments → raises its floor. Frequently-revisited reasoning resists
  decay. Reinforcement is persisted as a lightweight `attended` event so it is replayable.

### 5.4 Attention persistence (oscillation control)
The new activation blends with the previous cycle: `A = α·A_new + (1−α)·A_prev`. This gives
focus *inertia* — attention doesn't teleport every cycle, and competing hypotheses oscillate
rather than flip-flop.

### 5.5 Competing-hypothesis inhibition
`contradiction` edges between two hypotheses cause **mutual inhibition**: each subtracts a
fraction of the other's activation. The graph thus *holds* a live tension between rival ideas
and surfaces the currently-stronger one without deleting the weaker.

### 5.6 Dormant thread resurfacing
A thread untouched past a horizon goes **dormant**. When current intent activates a node that
is *structurally* close (≤2 hops) to a dormant thread, that thread gets a **resurfacing bonus**
and re-enters the workspace — "I worked on this months ago and it's relevant again," automatically.

**Why this satisfies the constraint:** diffusion improves *retrieval precision* (relevance by
structure, not keyword overlap) and *reconstruction fidelity* (the workspace reflects what the
reasoning actually connects to); resurfacing + persistence improve *long-horizon continuity*.

---

## 6. Working memory as a dynamic runtime object (Upgrade #3)

`working_context.md` stops being storage. The **workspace** is an ephemeral projection
generated on demand from the active subgraph:

1. compute attention state for the current intent (§5),
2. take the top-activation nodes, expand ≤2 hops → **active subgraph**,
3. synthesize a tiny, high-signal lens (≤~1,500 tokens), grouped by active thread.

It is regenerated each cycle (session start, drift, explicit refresh). Writing it to
`runtime/working_context.md` is **optional debug/export** — the source of truth for "what am I
focused on" is the attention state, which is itself rebuildable from L1. Shape:

```markdown
# Workspace @ 2026-05-31T14:02 (active subgraph: 11 nodes / 3 threads)

## Active threads
1. [0.91] salience-ranking — attention model (last touched 2h ago)
2. [0.62] memory-reconstruction — graph restoration
3. [0.48 ↑resurfaced] dormant-resurfacing — was dormant 9d

## Focus (highest activation)
- decision evt_50: Attention = spreading activation, not top-K  (causes: evt_44)
- open_loop evt_77: competing-hypothesis inhibition tuning  [unresolved]

## Live tension (competing hypotheses)
- evt_71 dense-embeddings  ⟂  evt_72 bm25-only   (currently leaning bm25-only, 0.61 vs 0.39)

## Next concrete action
Tune inhibition coefficient; wire reinforcement event on attend.

_Projection of active subgraph; truth = events.jsonl_
```

---

## 7. Restoration in the v3 model

```
restore(intent?) :
  1. graph = build_graph(events ≤ now)            # L2 from L1
  2. intent = detect_intent()                      # latest objective + unresolved loops + active threads
  3. A = spread_activation(graph, seed=intent)     # L3 attention
  4. subgraph = active_subgraph(A, top_n, hops≤2)  # extract
  5. workspace = synthesize(subgraph)              # L4 ephemeral lens
  6. (optional) write working_context.md           # L5 debug/export
```

No session is replayed. `detect_intent` needs no human input: most-recent `objective`,
all unresolved `open_loop`s, and threads with recent activity form the seed.

---

## 8. Replay model (Upgrade #8)

Replay is no longer chronological reconstruction of a session. To reconstruct cognition **as of
time T**:

```
replay_at(T, intent) :
  events_T = [e for e in events if e.t <= T]
  graph_T  = build_graph(events_T)
  A        = spread_activation(graph_T, seed=intent)
  return active_subgraph(A)
```

Determinism: identical `events.jsonl` (≤T) + identical intent seed ⇒ identical subgraph and
workspace. The v2 `replay.py verify` (double-rebuild content digest of `state.db`) still holds
for the L2 relational projection; v3 adds a graph-digest check for L2 graph + a fixed-seed
activation check for L3.

---

## 9. Multi-agent readiness (Upgrade #7 — design only)

Designed-for, not implemented:

- **Shared event bus** — `events.jsonl` *is* the bus. Add an optional `agent` field on events
  (`planner|coder|researcher|memory`). Absent ⇒ single-agent (today).
- **Agent-scoped attention windows** — each agent computes its own attention state over the
  *shared* graph, seeded by its role + assigned threads. Same `cognition.py` engine, different
  seed/filter. No per-agent store.
- **Conflict resolution for memory writes** — the log is append-only so writes never collide;
  *semantic* conflicts (two agents assert contradictory decisions) are represented as
  `contradiction` edges and resolved by the §5.5 inhibition + a `memory` agent's reflection
  pass, which can emit `assumption_invalidated`/`loop_closed`. No locks; truth is the union,
  coherence is a projection.

Compatibility hooks added now: nullable `agent` field accepted by `append`; attention seed/
filter parameterized so an agent window is just a constrained seed. Nothing else required.

---

## 10. Folder / system structure (target)

```
Persistent/                         # NEW HOME (was "LLM Wiki 2")
├── ARCHITECTURE_V3.md              # this doc (supersedes V2)
├── ARCHITECTURE_V2.md  README.md   # lineage
├── runtime/                        # L1 truth + L4 ephemeral
│   ├── events.jsonl                #   L1 TRUTH (event_id/thread_ids/causes/layer)
│   ├── state.db                    #   L2 relational  (replay.py)
│   ├── cognition.db                #   L2 graph + L3 attention state (cognition.py)
│   ├── working_context.md          #   L5 debug/export (optional)
│   └── active_context.json · session_state.json · open_loops.json · threads.json
├── index/                          # L3 retrieval cache (FTS5)  (retrieval.py)
├── cognitive/                      # L5 cognitive-layer projections (decisions, reflections)
├── execution/                      # L5 execution-layer projections (artifacts, outputs)
├── threads/                        # thread digests  (REPLACES sessions/)
├── identity/ procedural/ semantic/ reference/ summaries/   # stable + compression tiers
├── system/                         # tooling
│   ├── runtime.py                  #   append + restore + checkpoint (event_id/thread/layer)
│   ├── replay.py                   #   L2 relational state.db
│   ├── cognition.py                #   NEW: graph build + attention/diffusion + workspace
│   ├── retrieval.py                #   L3 FTS5/BM25 + salience + Claude rerank + drift
│   ├── semantic_search.py          #   TF-IDF offline fallback
│   └── checkpoint.py resume.py validate.py …   # v1 legacy, rewired
└── _archive/                       # downgraded / out of the cognition path
```

`sessions/` is retired into `threads/` (digests) + the graph; the directory is archived.

---

## 11. Data schemas (implementation-ready)

### Event (L1) — superset of v2, backward compatible
```jsonc
{
  "t": "2026-05-31T14:02:11+10:00",     // required
  "type": "decision",                    // required
  "event_id": "evt_50",                  // assigned if absent (evt_<seq>)
  "layer": "cognitive",                  // inferred from type if absent
  "thread_ids": ["salience-ranking"],    // optional; [] if none
  "causes": ["evt_44"],                  // optional causal parents (event_ids)
  "agent": null,                          // optional; multi-agent readiness (§9)
  // ── domain fields by type ──
  "msg": "...", "path": "...", "id": "loop-x", "title": "...", "into": "...",
  "spawn": "...", "links": [...], "topics": [...]
}
```
Old events (no `event_id`/`thread_ids`/`causes`/`layer`) remain valid: ids are derived from
line position, layer from type, membership empty. **No migration required.**

### Graph (L2, `cognition.db`)
```sql
nodes(event_id PK, seq, t, type, layer, body, importance)
membership(event_id, thread_id)                       -- event ↔ thread
edges(src, dst, kind, weight)                          -- kind: causal|membership|thread_rel|contradiction
threads(thread_id PK, title, opened_t, last_activity_t, status, merged_into, n_events)
```

### Attention (L3, `cognition.db`, transient)
```sql
activation(event_id PK, base, activation, reinforced)  -- recomputed each cycle
thread_activation(thread_id PK, activation, resurfaced)
```

---

## 12. Removed legacy assumptions

1. **Session as the unit of memory** — gone; threads + graph replace it. `session_*` events are
   demoted to time-window meta-markers.
2. **`working_context.md` as stored state** — gone; the workspace is an ephemeral projection of
   the active subgraph. Markdown is debug/export.
3. **Top-K retrieval as the retrieval model** — replaced by attention/diffusion (top-K still
   exists inside `retrieval.py` as the seed generator, not the final answer).
4. **A single flat salience number = priority** — replaced by an attention *state* with dynamics
   (decay, reinforcement, persistence, inhibition, resurfacing).
5. **Chronological replay = restoration** — replaced by graph reconstruction + active-subgraph
   extraction; determinism re-based to the event layer.
6. **`sessions/` directory as primary projection** — replaced by `threads/` digests.

---

## 13. Build status & roadmap

- **Core MVP built (2026-05-31):** `system/cognition.py` — event-graph build (nodes/membership/
  edges incl. causal + derived thread_rel + contradiction), thread state (active/dormant/merged),
  spreading-activation attention (decay, unresolved-seed, persistence, contradiction inhibition,
  dormant resurfacing), active-subgraph extraction, and dynamic workspace synthesis. `runtime.py`
  extended for `event_id/thread_ids/causes/layer/agent` (backward compatible).
- **Reuses:** v2 salience (`retrieval.salience`) as base activation; `runtime.read_events` as the
  one read contract; `replay.py` L2 unchanged.
- **Attention tuning (built):** iterative lateral inhibition (`INHIBITION=0.35` × 4 rounds) —
  equal rivals stay tied, evidence asymmetry amplifies to winner-take-most; reinforcement-on-attend
  (`context --reinforce` emits replayable `attended` events; `build` folds the count into a
  base-activation floor, `REINFORCE_STEP=0.06` cap `0.30`).
- **Adaptive refresh / Phase-4 hook (built):** `cognition.py refresh [--hook]` — rebuild → drift
  check → on a genuine topic *turn* emit `topic_shift` + auto-checkpoint (keep-open) + re-inject the
  fresh workspace. Wired as a `UserPromptSubmit` hook (silent unless drift). Drift compares the
  recent window vs the *preceding* block (a turn, not accumulated novelty), filters meta events,
  threshold `0.82`.
- **Phase 5 — reflection + compression (built):** `system/reflect.py` — `reflect` scans threads/
  loops/tensions for stale branches + aging work and emits a replayable `reflection` event;
  `compress` writes per-thread **digests** to `threads/` (these *are* the session-snapshot
  replacement; cold/dormant threads' digests stand in for originals).
- **Dense embeddings (built, off by default):** pluggable `_Embedder` behind `CONTINUITY_DENSE`;
  attention seeding is a graceful 3-tier chain dense → BM25 → lexical. Stdlib-default invariant
  intact; live dense path needs `sentence-transformers` installed (not present here, fallback verified).
- **Deferred (designed, not built):** multi-agent runtime (§9 — schema + seed hooks ready).

Every v3 addition maps to at least one required gain: threads/graph → long-horizon continuity +
coherence; attention/diffusion → retrieval precision + reconstruction fidelity; layer split →
retrieval precision; ephemeral workspace → reconstruction fidelity.
