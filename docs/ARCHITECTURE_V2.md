---
id: architecture-v2
status: design
created: 2026-05-31
updated: 2026-05-31
tags: [system, architecture, v2, runtime, event-sourcing]
supersedes: README.md (v1 deterministic markdown model)
---

# Persistent Continuity Architecture v2

**From** a deterministic markdown project memory
**To** an autonomous, event-sourced, reconstructable cognitive-state runtime.

> **Goal restated:** Not conversation continuity. Reconstruct the *minimum viable
> cognitive state* required to continue useful work — automatically, every session,
> without manual task management, manual summaries, or manual restore procedures.

This is an **evolution of the existing system, not a rebuild.** The v1 layout
(`00_FOUNDATION` … `13_PROCEDURAL`) and its Python tooling
(`semantic_search.py`, `indexer.py`, `checkpoint.py`, `resume.py`,
`episodic_logger.py`, `memory_manager.py`, `validate.py`) are kept and **rewired**:
markdown stops being the source of truth and becomes a *projection*; an append-only
event log becomes truth; restoration becomes a generated synthesis rather than a
hand-ordered file load.

---

## 0. The Core Inversion

| Dimension | v1 (current) | v2 (target) |
|-----------|--------------|-------------|
| Primary unit | Task (`TASK-001`) | Session / event |
| Source of truth | Markdown files | `runtime/events.jsonl` (append-only) |
| Markdown role | Truth | Projection / inspection / interchange |
| Restoration | Deterministic 7-step manual load order | Retrieval-driven synthesis → `working_context.md` |
| What loads | Human decides; load whole files | System decides; load synthesized signal |
| Checkpoint | `python checkpoint.py` (manual) | Autonomous triggers (context %, decision, idle, artifact) |
| Lifecycle | `active/blocked/completed` folders | Open-loop salience; no folder bureaucracy |
| Continuity | Re-read files | Reconstruct cognitive state from scored memory |

**One sentence:** *Events are truth; markdown is a view; the runtime reconstructs
working memory by retrieval and salience, and regenerates a tiny synthetic
bootstrap every session.*

---

## 1. Revised Architecture

Five layers, from durable to disposable:

```
┌─────────────────────────────────────────────────────────────┐
│ L5  PROJECTION (markdown)   human inspection, exports, docs    │  ← regenerated
├─────────────────────────────────────────────────────────────┤
│ L4  RUNTIME (runtime/)      working_context.md, open_loops,    │  ← disposable
│                             active_context, session_state       │     per session
├─────────────────────────────────────────────────────────────┤
│ L3  RETRIEVAL (index/)      embeddings, session/topic/artifact  │  ← rebuilt from L1
│                             graphs, salience scores             │     (cache)
├─────────────────────────────────────────────────────────────┤
│ L2  DERIVED STATE (sqlite)  materialized views of events:       │  ← replayable
│                             sessions, decisions, artifacts,     │     from L1
│                             loops, salience                      │
├─────────────────────────────────────────────────────────────┤
│ L1  EVENT LOG (events.jsonl)  append-only, immutable TRUTH       │  ← the only
│                                                                  │     source of truth
└─────────────────────────────────────────────────────────────┘
```

**Invariant:** L2–L5 are *derivable*. If they are deleted, `replay(events.jsonl)`
rebuilds them. Only L1 is irreplaceable. Back up L1; treat everything else as cache.

The v1 memory taxonomy survives as **semantic roles inside this stack**, not as the
load order:

- **Identity (stable):** project goals, constraints, principles, user prefs
  → projected from `00_FOUNDATION` + `13_PROCEDURAL`, rarely changes.
- **Working (dynamic):** current focus, open loops, latest decisions, active artifacts
  → lives in `runtime/`, regenerated each session.
- **Episodic (retrieved):** session snapshots and prior reasoning → L1 events + L3 search.
- **Semantic (retrieved):** facts → existing `12_SEMANTIC` + `semantic_search.py`.

---

## 2. Recommended Storage Stack

Local-first, zero external services, mostly stdlib.

| Concern | Technology | Why |
|---------|-----------|-----|
| Event truth | **JSONL** append-only file | Trivially appendable, greppable, replayable, diffable, never locks readers |
| Derived state / queries | **SQLite** (`runtime/state.db`) | Single-file, stdlib `sqlite3`, transactional materialized views, fast filters |
| Default retrieval | **SQLite FTS5 / BM25** (stdlib) | Zero-dep keyword candidate generation; works fully offline |
| Semantic rerank (optional) | **Claude API rerank** over BM25 candidates | Anthropic has no embeddings endpoint, so reranking is how we "use Claude" for meaning; cloud-first per the 2026-05-31 decision |
| Salience graph | **SQLite** tables (`salience`, `links`) | Scores + entity links are relational; avoid a second store |
| Dense embeddings (pluggable, off) | behind an `Embedder` interface (Voyage / OpenAI / Ollama / sentence-transformers) | Optional upgrade selected by env var; not required and not default |
| Topic / artifact graph | **JSON adjacency** (`index/*_graph.json`) | Small, human-inspectable, regenerated by indexer |
| Working context | **Markdown** (`runtime/working_context.md`) | The one file the model reads at boot; high-signal, disposable |
| Projections / docs | **Markdown** (existing folders) | Human inspection + git-friendly history |

**Dependency policy (refined per the 2026-05-31 decision — cloud-first, not
Ollama-reliant):** the retrieval *base* must run on **pure stdlib** (SQLite FTS5/BM25),
so the system never breaks offline or without an API key. The preferred *semantic*
path is **BM25 candidate generation → optional Claude API rerank** — there is no
Anthropic embeddings endpoint, so reranking is how Claude contributes meaning.
**Dense embeddings are a pluggable, off-by-default upgrade** behind an `Embedder`
interface, chosen by `SEMANTIC_BACKEND` / `EMBED_PROVIDER` env vars (Voyage, OpenAI,
Ollama, or local sentence-transformers) — none is assumed present. The existing
`semantic_search.py` TF-IDF mode remains as the last-resort offline fallback.

---

## 3. Runtime Design (`runtime/`)

```
runtime/
├── events.jsonl          # L1 truth — append-only, never edited
├── state.db              # L2 materialized views (SQLite)
├── active_context.json   # machine state: current project, branch, focus, session id
├── session_state.json    # this session: started, last_checkpoint, context_pct, drift
├── open_loops.json       # unresolved reasoning chains with salience
├── retrieval_cache.json  # last retrieval set (avoid recompute mid-session)
├── salience_graph.db     # (or table in state.db) scores + entity links
└── working_context.md    # SYNTHESIZED bootstrap — the file the model reads first
```

The runtime layer is **lightweight, continuously updated, autonomously maintained,
and optimized for instant restoration.** It is the only thing read at session start.

### Event schema (L1)

Append one JSON object per line. Required: `t` (ISO time), `type`. Common types:

```json
{"t":"2026-05-31T14:02:11","type":"session_start","session":"2026-05-31_14-02","project":"continuity-v2"}
{"t":"2026-05-31T14:02:40","type":"objective","msg":"Design event-sourced runtime"}
{"t":"2026-05-31T14:11:03","type":"decision","msg":"events.jsonl is source of truth; markdown is projection","links":["artifact:ARCHITECTURE_V2.md"]}
{"t":"2026-05-31T14:18:20","type":"artifact","path":"runtime/replay.py","action":"created"}
{"t":"2026-05-31T14:22:50","type":"open_loop","id":"loop-7","msg":"Choose sqlite-vec vs flat embeddings"}
{"t":"2026-05-31T14:40:00","type":"loop_closed","id":"loop-7","resolution":"flat embeddings for MVP"}
{"t":"2026-05-31T15:05:00","type":"checkpoint","reason":"context_pct>0.75","session":"2026-05-31_14-02"}
{"t":"2026-05-31T15:05:01","type":"session_end","session":"2026-05-31_14-02","status":"paused"}
```

Event types: `session_start`, `session_end`, `objective`, `decision`, `artifact`,
`open_loop`, `loop_closed`, `checkpoint`, `topic_shift`, `reflection`, `error`,
`assumption`, `assumption_invalidated`. Extensible — unknown types are stored and
ignored by views that don't understand them.

### Session snapshot (projection, written on `session_end`)

The v1 task file is replaced by an append-only session snapshot — exactly the YAML
form from the upgrade brief, generated *from events*, not hand-written:

```yaml
---
session: 2026-05-31_14-02
type: code
topics: [continuity-v2, event-sourcing, retrieval]
importance: high
status: paused
salience: 0.88
---
# Objective
Design autonomous continuity architecture.
# Key Outcomes
- Inverted truth source to events.jsonl
- Defined runtime layer + restoration engine
# Open Loops
- Embedding store choice (loop-7 → closed: flat)
- Autonomous checkpoint thresholds (loop-9, open)
# Next Action
Implement replay.py + restore.py MVP.
```

---

## 4. Restoration Engine Design

Replaces the manual 7-step `RESTORE_ORDER.md`. Fully automatic, four stages.

### Stage 0 — Continuous background indexing (after each interaction)
embed → tag → score → compress → link → update salience. No manual maintenance.
Implemented by extending the existing `indexer.py` to consume new events since the
last `index/cursor` offset (incremental, not full re-scan).

### Stage 1 — Intent detection (at session start)
Infer, *without asking the human*:
- current project + continuation branch (from latest `session_start`/`active_context.json`)
- unresolved reasoning chains (from `open_loops.json`, ranked by salience)
- active architecture area (topic graph centroid of recent sessions)
- likely objective (most recent `objective` event + highest-salience open loop)

Signal = `w1·semantic_similarity + w2·recency + w3·salience + w4·unresolved + w5·artifact_refs`.

### Stage 2 — Cognitive state reconstruction
- **Identity layer** — load stable foundation + user profile (cheap, always).
- **Working memory layer** — current focus, open loops, latest decisions, active artifacts.
- **Episodic layer** — retrieve *only* the top-K semantically related prior sessions,
  linked decisions, and artifacts for the detected intent. Not whole histories.

### Stage 3 — Synthetic working context generation
Do **not** reload raw memory. Synthesize `runtime/working_context.md`: tiny,
high-signal, disposable, regenerated every session. This file *is* the bootstrap.
Target budget: **≤ ~1,500 tokens.** Shape:

```markdown
# Working Context — regenerated 2026-05-31T14:02

## You are continuing
Project: continuity-v2 · Branch: event-sourced-runtime · Last status: paused

## Identity (stable)
- Goal: autonomous reconstructable cognitive runtime
- Hard constraints: stdlib-default; events.jsonl is truth; markdown is projection

## Where work stopped
Last decision: events.jsonl = source of truth (2026-05-31T14:11)
Last artifact: runtime/replay.py

## Open loops (do these)
1. [0.91] Autonomous checkpoint thresholds — unresolved
2. [0.77] Reflection cadence — unresolved

## Next concrete action
Implement replay.py + restore.py MVP.

## Retrieved (load on demand only)
- session 2026-05-30_21-12 (sim 0.83) · ADR-events-vs-markdown · facts: tf-idf-fallback
```

### Stage 4 — Adaptive refresh (during the session)
Periodically re-run retrieval; detect topic drift (cosine distance of recent
messages vs. session centroid > threshold); inject newly-relevant memories;
checkpoint automatically; reprioritize salience. Continuity is maintained *during*
work, not only at boot.

---

## 5. Retrieval Architecture (`index/`)

```
index/
├── embeddings.db        # vectors (sqlite-vec) or flat fallback
├── session_index.json   # session_id → {topics, salience, vector_ref, links}
├── topic_graph.json     # topic ↔ topic adjacency (co-occurrence weighted)
├── artifact_graph.json  # artifact ↔ session/decision references
└── cursor               # last event offset indexed (incremental)
```

Capabilities: semantic search, relevance retrieval, relationship mapping (graph
walks from a session to its decisions/artifacts/related sessions), contextual
reconstruction. **Default pipeline: SQLite FTS5/BM25 candidate generation → optional
Claude API rerank** (cloud-first, per the 2026-05-31 decision). Dense embeddings
(`embeddings.db`) are a pluggable, off-by-default upgrade behind the `Embedder`
interface; `semantic_search.py` TF-IDF remains the offline fallback.

**Scale strategy — must hold for thousands of sessions without context collapse:**
- Never load all sessions; retrieval returns top-K (default K=5–8).
- Hierarchical compression: old sessions roll into warm→cold *summaries* (existing
  `08_SUMMARIES` tiers) whose embeddings stand in for the originals.
- Graph hop limit (≤2) keeps relationship expansion bounded.
- Salience-gated indexing: trivial events get low salience and are excluded from
  the hot retrieval set.

---

## 6. Checkpointing System (autonomous)

No manual checkpoint commands. Fire `checkpoint` (extend existing `checkpoint.py`
into a triggered function) when **any** of:

| Trigger | Condition |
|---------|-----------|
| Context pressure | session context usage > 75% |
| Major decision | a `decision` event is appended |
| Architecture mutation | change touching `00_FOUNDATION` / ADR / this file |
| Idle timeout | no events for N minutes |
| Artifact generated | an `artifact` event is appended |
| Open-loop created | an `open_loop` event is appended |

A checkpoint: validates state → appends a `checkpoint` event → updates derived views
→ regenerates the HOT projection and `working_context.md` → is itself replayable.
Because truth is the event log, a checkpoint is **cheap** (it's mostly a marker plus
a projection refresh), so frequent autonomous checkpoints are affordable.

**Integration with the harness:** wire triggers via Claude Code **hooks** (the
existing `continuity-checkpoint` skill becomes the hook target). A `PostToolUse` /
stop hook appends events; a context-threshold hook fires the checkpoint. This is the
"runs automatically inside every session" requirement — autonomy lives in hooks, not
in the model remembering to act.

---

## 7. Reflection System (drift prevention)

A periodic `reflection` pass (end of session + every N checkpoints) evaluates:
- Which unresolved reasoning chains still matter? (re-score open loops; expire dead ones)
- What assumptions changed? (emit `assumption_invalidated`; flag dependent decisions)
- What branches are stale? (sessions untouched > horizon → demote salience)
- What decisions conflict? (detect contradictory `decision` events on same entity)
- What context should decay? (apply decay curve — see §8)
- What should be compressed? (roll cold sessions into summaries)

Output is itself an event (`reflection`) plus updated salience — fully replayable and
auditable. This is the immune system against recursive summary degradation, memory
bloat, stale-context pollution, hallucinated continuity, and silent loop loss.

---

## 8. Salience Model

Every session / memory / artifact carries:

```json
{ "importance": 0.91, "recency": 0.74, "relevance": 0.88,
  "unresolved": true, "linked_entities": 12 }
```

Composite (retrieval priority):

```
salience = 0.30·importance
         + 0.20·recency
         + 0.25·relevance_to_current_intent
         + 0.15·(1 if unresolved else 0)
         + 0.10·normalize(linked_entities)
```

- **importance** — set at emit time (decisions/architecture = high; routine = low),
  later nudged by reflection.
- **recency** — exponential decay: `recency = 0.5 ** (age_days / half_life)`,
  half-life ≈ 7 days (tunable).
- **relevance** — cosine similarity to detected current intent (recomputed each session).
- **unresolved** — open loops get a hard priority boost; this is the system's bias
  toward *finishing* over *reviewing*.
- **linked_entities** — graph degree; well-connected nodes resurface.

Restoration prioritizes **unresolved loops + high salience + recent architectural
decisions + active artifacts**, never entire histories.

---

## 9. Compression Strategy

Three-tier, mirroring (and reusing) the existing `08_SUMMARIES/{hot,warm,cold}`:

| Tier | Holds | Form | Reload policy |
|------|-------|------|---------------|
| HOT | current + last session | full detail / working_context | every session |
| WARM | recent architectural arc | per-session snapshots | on demand (intent match) |
| COLD | everything older | rolled summaries + embeddings | rarely; summary stands in for originals |

**Roll-up, not deletion.** Cold sessions are summarized; the *summary* is embedded so
retrieval still reaches the knowledge without loading raw events. Originals stay in
`events.jsonl` (truth is never destroyed) but leave the hot retrieval set.
Guard against **recursive summary degradation**: summaries are always generated from
*source events*, never from prior summaries (no summary-of-summary chains).

---

## 10. Failure Modes & Drift Prevention

| Failure | Mechanism | Guard |
|---------|-----------|-------|
| Truth corruption | bad write to events.jsonl | append-only + per-line JSON validate on write; never edit existing lines; back up L1 |
| Derived/state desync | state.db disagrees with events | `replay.py` rebuilds state.db from scratch; treat L2–L5 as cache |
| Recursive summary rot | summary-of-summary | summaries only ever derive from source events |
| Memory bloat | unbounded growth | salience decay + cold roll-up + retrieval top-K |
| Stale context pollution | old branch resurfaces | recency decay + reflection demotes stale branches |
| Hallucinated continuity | model invents resumed state | working_context cites event ids/timestamps; unverifiable claims excluded |
| Open-loop loss | unresolved chain forgotten | open_loops.json persisted as events; unresolved gets salience boost |
| Conflicting decisions | two ADRs disagree | reflection contradiction check emits a flag event |
| Embedding unavailability | no model / no deps | TF-IDF fallback (already in semantic_search.py) |
| Wrong intent at boot | misdetected continuation | working_context shows detected intent explicitly; one-line correction re-runs restore |

**Determinism where it matters:** v1's prized determinism is preserved as
*replay determinism* — given the same `events.jsonl`, the system rebuilds the same
state every time. Retrieval is heuristic, but truth and reconstruction are reproducible.

---

## 11. Scalability Considerations

- **Append-only writes** are O(1) and never block readers.
- **Incremental indexing** via `index/cursor` — only new events are processed.
- **Bounded retrieval** — top-K + graph hop limit ⇒ working-context size is constant
  regardless of history length (thousands of sessions, flat boot cost).
- **SQLite** comfortably handles 10⁵–10⁶ event rows on a laptop.
- **Tiered compression** keeps the hot set small; cold history is reachable but not loaded.
- **Vector store swap** — flat fallback for small N; sqlite-vec/FAISS when N grows,
  behind the same retrieval interface so callers don't change.
- **Optional log segmentation** — roll `events.jsonl` into `events/YYYY-MM.jsonl` if a
  single file gets unwieldy; replay concatenates in order.

---

## 12. Example Filesystem Layout (v2 — as executed 2026-05-31)

The v1 numbered folders were renamed to semantic names and the lifecycle/task
folders archived. This is the layout **now on disk**:

```
Persistent/                     # (formerly LLM Wiki 2/; moved 2026-05-31)
├── ARCHITECTURE_V2.md          # this document (supersedes README.md)
├── runtime/                    # L4 — the cognitive runtime (NEW)
│   ├── events.jsonl            #   L1 truth (append-only)
│   ├── working_context.md      #   the boot file (regenerated)
│   ├── active_context.json     #   project/branch/session/status
│   ├── session_state.json      #   counts, last event, checkpoints
│   ├── open_loops.json         #   ranked unresolved loops
│   ├── state.db                #   L2 derived views        (BUILT, Phase 1)
│   ├── retrieval_cache.json    #   last retrieval set       (planned)
│   └── salience_graph.db       #   scores + links           (folded into index.db)
├── index/                      # L3 — retrieval (BUILT, Phase 3)
│   ├── index.db                #   SQLite FTS5 events + salience (rebuildable cache)
│   ├── artifact_graph.json  topic_graph.json  cursor
│   └── embeddings.db           #   (planned) dense upgrade, off by default
├── identity/      ← 00_FOUNDATION   # Identity projection (stable)
├── procedural/    ← 13_PROCEDURAL   # user profile / prefs (stable)
├── semantic/      ← 12_SEMANTIC     # facts + semantic_search.py
├── episodic/      ← 11_EPISODIC     # event logs (cold)
├── decisions/     ← 04_DECISIONS    # decision projections (also events)
├── artifacts/     ← 05_ARTIFACTS    # artifact store
├── summaries/     ← 08_SUMMARIES    # hot/warm/cold compression tiers
├── sessions/      ← 06_SESSIONS     # session snapshots (generated from events)
├── reference/     ← 07_REFERENCE    # static reference
├── system/        ← 09_SYSTEM       # tooling (v1 scripts rewired + v2 runtime.py)
│   ├── runtime.py              #   NEW (MVP): append + restore + checkpoint, stdlib
│   ├── retrieval.py            #   NEW (Phase 3): incremental FTS5 index + BM25
│   │                           #   search + salience (§8) over events.jsonl
│   ├── semantic_search.py      #   REUSE: TF-IDF offline fallback (pluggable dense)
│   ├── indexer.py              #   v1 markdown rglob index (legacy)
│   ├── checkpoint.py resume.py validate.py memory_manager.py episodic_logger.py
│   │                           #   v1 legacy, rewired to new paths (kept working)
│   └── (planned) reflect.py · Claude rerank wiring · Stage-4 adaptive refresh
├── _archive/                   # DOWNGRADED lifecycle (out of restore path)
│   ├── 01_STATE/  02_MEMORY/  03_TASKS/  10_WORKING_MEMORY/
└── README.md                   # v1 doc (retained for history; superseded)
```

(The wiki-ingestion folders `sources/ wiki/ strata/ papers/ raw/` are unrelated to
continuity and were left untouched.)

**Downgrade, don't delete.** The rigid lifecycle/task folders moved to `_archive/`
for backward inspection but left the restoration path. Open loops replace them.
`03_TASKS` is no longer load-bearing — `runtime/open_loops.json` is.

---

## 13. Suggested Implementation Roadmap

**Phase 1 — Event spine (truth). ✅ DONE (core).**
Append + per-line validate live in `runtime.py` (the `eventlog.py` role). `replay.py`
materializes events → `runtime/state.db` (sessions/decisions/artifacts/loops tables),
drop-and-rebuild from zero. *Exit met:* every action appends an event and
`replay.py rebuild` reconstructs state from zero; `replay.py verify` proves
replay determinism via a double-rebuild content digest. *Deferred (chosen):* the
one-shot v1-markdown→events migration (lossy, low value now the v2 log is truth) and
the `session_end` YAML snapshot projection.

**Phase 2 — Restoration engine.**
`restore.py` implementing Stages 1–3 → `working_context.md`. `resume.py` becomes a
thin wrapper. *Exit:* opening a session reads only `working_context.md` and the model
can answer "what was I doing / what's next" without manual file loading.

**Phase 3 — Retrieval + salience. ✅ DONE (MVP).**
`retrieval.py`: incremental indexer (consumes events past `index/cursor` into SQLite
FTS5), BM25 candidate generation blended with the §8 salience model
(importance + recency-decay + relevance + unresolved + linked), and `artifact_graph`/
`topic_graph` adjacency. `runtime.restore` prefers salience-ranked `open_loops.json`
ordering, with recency-only fallback when no index exists. *Exit met:* top-K
retrieval (`search`) works and loops are salience-ranked. **Claude rerank wired**
(`claude_rerank`: real Anthropic-API relevance scoring folded into salience, off by
default via `--rerank`/`CONTINUITY_RERANK`, degrades to BM25 order when SDK/key absent)
and **Stage 4 drift detection** (`detect_drift`: lexical cosine of recent-window vs
session baseline → optional replayable `topic_shift`). *Deferred:* mid-session
auto-refresh hook wiring (Phase 4) and dense `embeddings.db` (off by default).

**Phase 4 — Autonomy (hooks).**
Wire checkpoint triggers and post-interaction indexing into Claude Code hooks via the
existing `continuity-*` skills. *Exit:* zero manual checkpoint/restore commands in a
normal session.

**Phase 5 — Reflection + compression.**
`reflect.py` (drift, conflicts, decay, expiry) + cold roll-up into summaries. *Exit:*
system self-maintains across many sessions without bloat or stale resurfacing.

---

## 14. Minimal Viable Prototype (MVP)

Smallest thing that proves the inversion. **Pure stdlib. Three files. One day.**

1. **`eventlog.py`** — `append(type, **fields)` writes a validated JSON line to
   `runtime/events.jsonl`; `read()` yields events. ~40 lines.
2. **`restore.py`** — read events, take latest `session_start`/`objective`, all open
   `open_loop` events not yet `loop_closed`, last 3 `decision`s, last 5 `artifact`s;
   rank loops by recency; write `runtime/working_context.md` from the §4 template.
   No embeddings — recency + unresolved only. ~120 lines.
3. **`checkpoint.py` (triggered fn)** — append a `checkpoint` event, append
   `session_end`, call `restore.py` to refresh working_context. ~30 lines.

**MVP demo loop:**
```
append objective  →  append 2 decisions  →  append 1 open_loop  →  append artifact
→ run checkpoint  →  (new "session") run restore  →  read working_context.md
```
Success = the regenerated `working_context.md` correctly states the objective, the
open loop, the last decision, and the next action — with **no markdown files in the
load path and no human choosing what to load.** That single demo validates the entire
architectural shift; everything in Phases 3–5 is enhancement, not prerequisite.

---

### Appendix: what changed vs. v1, in one breath
Tasks → sessions/loops. Markdown-truth → event-truth + markdown-projection. Manual
restore order → synthesized `working_context.md`. Human-chosen loads → salience-ranked
retrieval. Manual checkpoints → hook-fired autonomous checkpoints. Static folders →
disposable runtime + replayable truth. Determinism preserved as *replay* determinism.
