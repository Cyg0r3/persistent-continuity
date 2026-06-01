# Persistent Continuity — Architecture V4

**Hybrid Cognitive Memory Stack: from persistent memory infrastructure to a continuously reconstructed cognitive runtime.**

V3 made memory cognition-native: an append-only event log as the only truth, a self-maintaining
cognitive graph, spreading-activation attention, and an ephemeral working-context lens. V4 keeps
every V3 invariant and closes the gap to a *full hybrid stack* — adding the memory **types** the
brain actually separates (episodic / semantic / procedural), the **scaling** machinery needed for
unlimited timescales (incremental build + snapshots + a persistent index), and the **consolidation**
that turns raw episodes into reusable knowledge.

> **Invariants preserved from V3 (non-negotiable):** pure Python stdlib by default · local-first ·
> `runtime/events.jsonl` is the *only* source of truth · everything else is a rebuildable projection ·
> all optional accelerators degrade gracefully to the stdlib path · single-agent behavior unchanged
> unless a feature is explicitly opted into.

---

## 0. The target reference architecture

```text
                    ATTENTION SYSTEM          ← spreading activation + resonance + incubation + oscillation
                           ↓
                ACTIVE WORKING MEMORY         ← in-runtime object, incrementally mutated (md = export only)
                           ↓
      ┌─────────────────────────────────────┐
      │      COGNITIVE GRAPH LAYER           │ ← typed nodes + causal/contradiction/dependency/concept edges
      └─────────────────────────────────────┘
          ↓            ↓             ↓
   EVENT MEMORY   SEMANTIC MEMORY   PROCEDURAL MEMORY
   (episodic)     (concepts)        (workflows/heuristics)
          ↓            ↓             ↓
        VECTOR INDEX / SEARCH / COMPRESSION   ← persistent index + snapshots + log compaction
```

PCA today implements the **top three bands well** and the **bottom three bands partially**. V4 is
mostly about the bottom: separating semantic and procedural memory from episodic, and making the
substrate scale.

---

## 1. Full architectural comparison

| Ideal layer | PCA today (V3.1) | Verdict | V4 action |
|---|---|---|---|
| **Attention system** | `cognition.attend()` — spreading activation, decay (`_recency`), reinforcement (`attended` floor), persistence (`ALPHA` blend), competing-hypothesis inhibition, dormant resurfacing | **Strong**, missing resonance/incubation/oscillation | Add resonance edges, an incubation pass, oscillation to break fixation |
| **Active working memory** | `cognition.workspace()` → ephemeral lens; `working_context.md` is already debug/export | **Good** (V3 already inverted this) | Promote to an in-runtime `WorkingMemory` object with incremental mutation + budget control |
| **Cognitive graph** | `cognition.db`: `nodes`, `membership`, `edges` (causal/membership/thread_rel/contradiction), `threads` | **Good foundation**, but *every node is an episodic event* | Add typed memory classes + `dependency`, `concept`, `resonance` edges |
| **Event (episodic) memory** | `events.jsonl`, `event_id`, `causes`, `layer`, append-only | **Strong** | Add branching/versioning + snapshots + compaction; harden the causal DAG |
| **Semantic memory** | BM25 (`retrieval.search`) + optional dense embedder (`_Embedder`, off by default) over event bodies | **Weak** — no concept nodes; semantics = keyword match over episodes | Introduce **concept/entity nodes** distilled by reflection; persistent embeddings |
| **Procedural memory** | `templates/procedural/USER_PROFILE.md` (a static doc) | **Absent as cognition** | Introduce **procedure nodes**: reasoning chains, workflows, heuristics |
| **Vector index / search / compression** | In-memory BM25; dense recomputed per call; `reflect.compress` writes thread digests | **Partial** | Persistent vector index table; snapshot-based compaction; digests feed semantic layer |

**One-line diagnosis:** PCA is a strong *episodic* cognitive runtime with a good attention engine,
bottlenecked by (a) treating all memory as undifferentiated event nodes, (b) a full drop-and-rebuild
graph that does not scale, and (c) no consolidation that converts episodes into durable concepts or
procedures.

---

## 2. Identified weaknesses, gaps, and bottlenecks

### 2.1 Structural bottlenecks (these block "unlimited timescales")

1. **O(N) rebuild on every prompt.** `cognition.build()` (cognition.py:174) drops the DB and replays
   the *entire* `events.jsonl` from zero. It runs inside `refresh()` on **every `UserPromptSubmit`**
   (hooks.json). At 10k+ events this dominates latency. → *Incremental build with a watermark.*
2. **O(N) per append → O(N²) per session.** `runtime.append()` computes the next id as
   `f"evt_{len(read_events()) + 1}"` (runtime.py:111), reading the whole log on **every write**.
   → *Maintain a persisted sequence counter; never read the full log to write one line.*
3. **Attention recomputed over all nodes every call.** Fine at hundreds of nodes, quadratic-ish at
   scale because base activation + seed + spread touch the full node set. → *Bound the working set
   to the active subgraph + a candidate frontier from the index.*
4. **The log only grows.** No snapshots, no compaction. Replay fidelity is great; replay *cost* is
   unbounded. → *Periodic snapshot events + a compacted base state; cold episodes summarized.*

### 2.2 Missing cognitive primitives

- **No semantic memory.** "Redis is fast" appears only as the body text of episodic events; it is
  never distilled into a *concept* that persists after those events go cold.
- **No procedural memory.** Successful reasoning chains and debugging heuristics are not captured,
  so the system cannot get *better at how it works* — only at *what it remembers*.
- **No abstraction formation.** Reflection (`reflect.py`) summarizes and prunes but never creates a
  higher-level node from a recurring episodic pattern.
- **Weak associative recall.** Retrieval is intent-seeded spreading + keyword; there is no
  embedding-backed "this reminds me of…" linkage that survives wording changes.

### 2.3 Legacy assumptions to retire

- *"A node is an event."* → A node is a **memory item** with a *class* (episodic / semantic /
  procedural); events remain the only *truth*, but the graph hosts distilled nodes too.
- *"Rebuild = correctness."* → Correctness comes from the **log**; the graph may be maintained
  incrementally and reconciled with a periodic full rebuild.
- *"Working context is a file."* → Already half-retired in V3; finish it — working memory is a
  runtime object, markdown is an export.

---

## 3. Revised architecture (V4)

### 3.1 Memory typing — one log, three memory classes

Everything is still a projection of `events.jsonl`. We add a **`mem_class`** dimension to graph
nodes and two new *truth* event types so distilled knowledge is itself replayable:

| Class | Source of truth | Graph table | Lifecycle |
|---|---|---|---|
| **Episodic** | raw events (today's nodes) | `nodes` (`mem_class='episodic'`) | created on append; compacted when cold |
| **Semantic** | `concept_formed` events emitted by reflection | `concepts` | formed by abstraction; reinforced by recurrence; pruned when unsupported |
| **Procedural** | `procedure_learned` events emitted by reflection | `procedures` | learned from successful causal chains; scored by outcome |

Because concepts and procedures are *emitted as events*, deleting `cognition.db` still rebuilds them
— the log remains the single truth. This is the central V4 move: **semantic and procedural memory are
not new stores of truth, they are new projections plus two new event types.**

### 3.2 Cognitive graph schema (additions to V3's `cognition.db`)

```sql
-- existing V3 tables unchanged: nodes, membership, edges, threads, activation, thread_activation
-- nodes gains a class tag (default keeps every existing node 'episodic' => backward compatible):
ALTER TABLE nodes ADD COLUMN mem_class TEXT DEFAULT 'episodic';

-- SEMANTIC MEMORY: distilled concepts/entities
CREATE TABLE concepts (
    concept_id TEXT PRIMARY KEY,     -- cpt_<slug>
    label      TEXT,
    summary    TEXT,                 -- one-line distillation
    salience   REAL DEFAULT 0.5,     -- recalibrated by consolidation
    support    INTEGER DEFAULT 1,    -- # episodic events that evidence it
    last_seen_t TEXT,
    embedding_id INTEGER             -- FK into vectors (nullable; stdlib path = NULL)
);
-- which episodes evidence which concept (abstraction provenance)
CREATE TABLE concept_evidence (concept_id TEXT, event_id TEXT);

-- PROCEDURAL MEMORY: reusable reasoning/operational patterns
CREATE TABLE procedures (
    procedure_id TEXT PRIMARY KEY,   -- prc_<slug>
    label      TEXT,
    steps      TEXT,                 -- JSON: ordered step descriptors
    trigger    TEXT,                 -- when this applies (context cue)
    outcome_score REAL DEFAULT 0.5,  -- success rate, updated on reuse
    uses       INTEGER DEFAULT 0,
    last_used_t TEXT
);

-- PERSISTENT VECTOR INDEX (optional; populated only when dense is enabled)
CREATE TABLE vectors (
    vector_id INTEGER PRIMARY KEY,
    ref_kind  TEXT,                  -- 'node' | 'concept' | 'procedure'
    ref_id    TEXT,
    dim       INTEGER,
    vec       BLOB                    -- packed float32; or external index id
);

-- INCREMENTAL BUILD watermark + snapshot bookkeeping
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);  -- e.g. last_seq, snapshot_seq
```

New **edge kinds** (the `edges` table is already `(src,dst,kind,weight)`):

- `dependency` — A must precede / is required by B (turns the implicit causal order into explicit
  blocking structure for planning).
- `concept` — episodic node ↔ concept it evidences (associative bridge between episodic and semantic).
- `resonance` — concept ↔ concept co-activation strength (the substrate for spreading *associations*,
  not just causal/thread proximity).

`EDGE_WEIGHT` (cognition.py:70) extends to: `dependency: 0.8`, `concept: 0.6`, `resonance: 0.5`.

### 3.3 Event-memory upgrades (episodic / truth layer)

- **New truth event types** (added to `KNOWN_TYPES` in runtime.py): `concept_formed`,
  `procedure_learned`, `snapshot`, `branch_open`, `branch_merge`. All optional/back-compatible.
  The V4.1 adaptive layer (§13–16) adds `pattern_detected`, `proc_executed`, `procedure_retired`,
  `meta_assessment`, and `pattern_promoted` — same rule: optional, curator-owned where they are
  conclusions, absent ⇒ current behavior.
- **Branching/versioning.** An optional `branch` field on events + a `branch_open`/`branch_merge`
  pair lets the runtime reconstruct *hypothetical* cognition ("what if we'd chosen Postgres?") without
  forking the file: a branch is just a labeled subset of the one log, and the graph build can be asked
  to include/exclude branches. Default branch = `main`; absent ⇒ `main` (back-compat).
- **Causal DAG hardening.** `causes` already exists; V4 validates it forms a DAG at build time and
  exposes `dependency` edges for planning/traceability.

---

## 4. Updated runtime model

### 4.1 Active cognitive graph reconstruction (replaces "session restoration")

V3 already reconstructs from the graph, not from sessions. V4 makes reconstruction **incremental and
attention-prioritized**:

```text
on session start / resume:
  1. load snapshot base state  (meta.snapshot_seq)         ── O(snapshot size), not O(log)
  2. apply event deltas since watermark (meta.last_seq)     ── O(new events)
  3. seed attention from: open loops + active threads + last working set + intent
  4. spreading activation → active subgraph (bounded working set)
  5. synthesize WorkingMemory object (not a file)
  6. adaptive reconstruction depth: shallow lens by default; deepen on demand
```

**Adaptive reconstruction depth** is the key new runtime knob: a cheap shallow pass (top-N active
nodes only) for routine prompts, deepening (more hops, concept expansion, procedure lookup) when the
prompt signals a hard or novel task. Depth is chosen from drift score + query complexity, reusing the
existing `retrieval.detect_drift` signal.

### 4.2 Working memory as a runtime object

```python
class WorkingMemory:               # lives in-process; markdown is export only
    active:   dict[node_id, activation]      # current focus
    threads:  dict[thread_id, activation]
    concepts: list[concept_id]               # semantic context in scope
    procedures: list[procedure_id]           # applicable how-to knowledge
    budget:   int                            # token/char budget (WORKING_CONTEXT_BUDGET_CHARS)

    def mutate(self, event):  ...    # incremental update on each new event (no full regen)
    def reprioritize(self): ...      # attention-driven re-rank within budget
    def render(self) -> str: ...     # export/debug → working_context.md
```

`workspace()` becomes `WorkingMemory.render()`. The object **mutates incrementally** as events land,
rather than re-synthesizing from scratch each prompt — this is what makes per-prompt upkeep cheap.

---

## 5. New memory subsystem definitions

### 5.1 Semantic memory
- **What:** durable concepts/entities (e.g. `cpt_session-store`, `cpt_jwt`) with a one-line summary,
  salience, and evidence links back to the episodes that formed them.
- **How formed:** the consolidation pass (§9) clusters recurring episodic content (by co-membership +
  embedding similarity when enabled) and emits a `concept_formed` event. Concepts gain `support` and
  `salience` each time new evidence recurs; they decay and are pruned when unsupported.
- **Why separate from episodic:** lets old episodes compact away while the *meaning* persists — the
  mechanism that keeps the active graph small over unlimited time.

### 5.2 Procedural memory
- **What:** reusable *how* — successful reasoning chains, debugging heuristics, execution strategies,
  operational patterns, with a trigger cue and an outcome score.
- **How formed:** consolidation detects causal chains that ended in success (e.g. a `loop_closed` or a
  passing `command`/`artifact` chain) and emits `procedure_learned`. Reuse updates `outcome_score`.
- **Retrieval:** when the current intent matches a procedure `trigger`, the procedure is injected into
  working memory ("last time this worked: …"), distinct from semantic facts and episodic recall.

### 5.3 Episodic memory
- Unchanged in spirit (today's event nodes), but now compactable: cold episodes summarized into a
  snapshot + their concepts/procedures retained, so episodic detail can age out without losing meaning.

---

## 6. Suggested storage technologies

Held to the **stdlib-default, local-first** invariant. Optional accelerators are *additive* and must
degrade gracefully (exactly like V3's `_Embedder`).

| Concern | Default (required) | Optional accelerator (graceful) |
|---|---|---|
| Truth log | `events.jsonl` (append-only) | — (never replaced; truth must stay diffable/greppable) |
| Graph + projections | **SQLite** (`cognition.db`) — already used | — |
| Vector index | packed float32 in the `vectors` table + numpy cosine | **sqlite-vec** extension, or **FAISS** / **LanceDB** for large corpora |
| Embeddings model | none (BM25/lexical) | **sentence-transformers** (`CONTINUITY_DENSE`, already wired) |
| Snapshots | JSON snapshot blob under `runtime/snapshots/` | — |

**Recommendation:** stay on SQLite as the canonical cache; add `sqlite-vec` as the *first* optional
vector accelerator because it keeps everything in one local file and one dependency-optional path,
consistent with the project's ethos. FAISS/LanceDB only matter past ~10⁵ vectors.

---

## 7. Retrieval pipeline redesign

V3 retrieval is a 3-tier seed (dense → BM25 → lexical) feeding spreading activation. V4 makes it a
**hybrid graph+vector pipeline with memory-type awareness**:

```text
query / intent
   │
   ├─►  candidate generation (cheap, bounded):
   │       • BM25 / lexical over episodic bodies        (retrieval.search)
   │       • vector ANN over concepts + episodes        (vectors table; optional)
   │       • always-on seeds: open loops, active threads (cognition._seed)
   │
   ├─►  graph expansion (spreading activation over the active subgraph):
   │       • causal / dependency / membership / thread_rel / concept / resonance edges
   │       • competing-hypothesis inhibition (unchanged)
   │
   ├─►  memory-type fusion:
   │       • episodic  → what happened
   │       • semantic  → what it means (concepts in scope)
   │       • procedural→ how to proceed (matched triggers)
   │
   └─►  WorkingMemory.reprioritize()  → budget-bounded lens
```

**Division of labor (explicit, to avoid "vectors = cognition" drift):**
- **Graph dominates** for causal/temporal/dependency reasoning and contradiction handling.
- **Vectors accelerate** *candidate recall* — finding "what might be relevant" by meaning when wording
  differs. They feed the seed; they never decide focus.
- **BM25/lexical** remains the zero-dependency floor and the determinism anchor.

Vectors are **retrieval acceleration infrastructure, not the cognition system.**

---

## 8. Attention system design

Keep the V3 spreading-activation core (decay, reinforcement, persistence, inhibition, resurfacing —
cognition.py:400-490). Add four dynamics:

1. **Resonance scoring.** When two concepts are co-active across many cycles, strengthen their
   `resonance` edge (Hebbian: "fire together → wire together"). Resonance edges then spread
   activation, giving genuine *associative* recall beyond causal/thread proximity. Bounded and
   decaying so the graph doesn't saturate.
2. **Subconscious incubation.** A background (PreCompact / idle / session-end) pass that lets
   activation diffuse *without* a query — dormant-but-well-connected nodes accrue a small lift, so an
   unsolved problem can "resurface with an idea" next session. Implemented as an extra low-gain
   diffusion round writing to a separate `incubation` activation channel.
3. **Priority oscillation.** To prevent fixation on a single high-activation cluster, periodically
   apply a mild inhibition to the current winner and a lift to the runner-up cluster, rotating focus.
   This surfaces neglected-but-relevant threads (anti-tunnel-vision), parameterized and off by default.
4. **Attention persistence across sessions.** V3 persists within the `activation` table; V4 ensures the
   *last working set* is part of resume seeding (§4.1 step 3), so cognition continues mid-thought
   rather than re-deriving from cold.

All four are **parameters/passes layered on the existing engine** — no rewrite. Per-agent attention
windows from V3.1 are unchanged and inherit these dynamics.

---

## 9. Reflection & consolidation architecture

Today's `reflect.py` does `reflect` (scan + emit a `reflection` event), `compress` (thread digests),
and `resolve` (contradiction resolution, V3.1). V4 organizes these into a **tiered, scheduled
consolidation pipeline** — analogous to fast attention vs. slow "sleep" consolidation:

| Tier | Trigger | Cost | Work |
|---|---|---|---|
| **T0 reflexive** | every `UserPromptSubmit` (existing `refresh`) | cheap | drift check, re-lens on topic turn |
| **T1 reflective** | session end / periodic | medium | `reflect` (stale/aging/tension scan), `resolve` decisive contradictions |
| **T2 consolidative** | idle / N events / "sleep cycle" | heavier | **abstraction**, **procedure learning**, **pruning**, **salience recalibration**, **snapshot/compaction** |

**New T2 passes:**
- **Abstraction formation** → cluster recurring episodic content; emit `concept_formed`; link evidence.
- **Procedure learning** → detect successful causal chains; emit `procedure_learned`; score outcomes.
- **Stale belief pruning** → demote/retire concepts whose `support` decayed and that no active thread
  references; demote stale threads (extends current `STALE_DAYS` logic).
- **Salience recalibration** → recompute concept/thread salience from recent activation history
  (counteracts drift where old high-importance items dominate forever).
- **Memory consolidation / snapshot** → write a snapshot of derived state at `last_seq`, compact cold
  episodes behind their concepts/digests, advance `meta.snapshot_seq` (the scaling mechanism).
- **Entropy management** → cap edge/concept growth; merge near-duplicate concepts; bound resonance.

**Scheduling:** T2 runs opportunistically (PreCompact already fires under context pressure; add an
idle/threshold trigger) so consolidation never blocks interactive turns — exactly how biological
consolidation is offline.

---

## 10. Multi-agent architecture readiness

V3.1 already shipped: shared event bus (`events.jsonl` + `agent` field), role-biased agent attention
windows over the one shared graph, per-agent attention persistence, and a `memory`-agent contradiction
resolver (`reflect.py resolve`). V4 extends the *design* (not full implementation) toward:

- **Agent-local working memory** — each agent owns a `WorkingMemory` object seeded by its role +
  assigned threads (windows already exist); V4 formalizes them as first-class runtime objects.
- **Coordination protocol** — agents communicate *only* by appending events to the shared bus; no side
  channels. Truth is the union of writes; coherence is a projection (already the model).
- **Memory arbitration / conflict resolution** — the **memory curator agent** owns T1/T2: it is the
  only agent that emits `assumption_invalidated`, `loop_closed`, `concept_formed`, `procedure_learned`.
  Other agents *propose* (e.g. `hypothesis`, `contradiction`); the curator *consolidates*. This gives a
  single, auditable arbitration point without locks.
- **Attention synchronization** — agents share the graph and its base activation, but each keeps a
  scoped `activation` channel (V3.1 `(agent, …)` keys); a periodic broadcast of high-salience concepts
  keeps windows loosely aligned without forcing identical focus.

No locks anywhere — append-only writes never collide; arbitration is a consolidation projection.

---

## 11. Migration path from current PCA

Each phase is independently shippable, preserves all invariants, and is backward-compatible
(absent fields/tables ⇒ current behavior). Ordered by **bottleneck-relief first**, then **new
cognition**, then **scale-out**.

> **Phase 0 — Performance foundation (no new cognition, pure speedup). ✅ BUILT.**
> Fixed the O(N²): `runtime.append` now allocates `event_id`s from a persisted O(1) counter
> (`runtime/seq`, self-healing if lost/stale) instead of re-reading the whole log on every write.
> `cognition.build()` records a watermark (`meta.event_count`) and **skips** the rebuild when the log
> hasn't grown — the graph is a pure function of the log, so an unchanged log means the existing
> `cognition.db` is already correct. `build(force=True)` / `build --force` always rebuilds (integrity
> fallback), and a pre-v4 cache without the `meta` table rebuilds automatically. Truth still rests on
> the log; the caches only avoid redundant work. Covered by `tests/test_phase0_scaling.py` (7 tests).
> *Outcome: per-prompt upkeep drops from O(log) reads to O(1) when idle, O(new events) when it grew.*

> **Phase 1 — Working memory as a runtime object. ✅ BUILT.**
> Introduced `WorkingMemory` (`cognition.py`): an in-process attention lens with
> `active`/`threads` focus, a char `budget`, and reserved `concepts`/`procedures` fields
> (empty until Phases 2/3 ⇒ current behavior). `WorkingMemory.from_graph()` seeds it from
> the active subgraph; `workspace()` is now the thin wrapper `from_graph(...).render()`,
> so the projection is byte-for-byte unchanged. `mutate(event)` folds a freshly-appended
> event into focus at peak activation **without** rebuilding the graph (a later
> `build()`+`from_graph()` reconciles to the exact projection); `reprioritize()` re-ranks
> by activation and drops the coldest nodes until `render()` fits the budget. Markdown is
> demoted to export only. Covered by `tests/test_phase1_working_memory.py` (4 tests).
> *Outcome: markdown is fully demoted to export; live reprioritization within budget.*

> **Phase 2 — Semantic memory. ✅ BUILT.**
> Added the `concepts`/`concept_evidence` tables and the `concept` edge to the graph schema
> (gated by a new `meta.schema_version` so pre-Phase-2 caches rebuild instead of skipping),
> the `concept_formed` event (curator-emitted, agent="memory"), and the **T2 abstraction
> pass** (`reflect.py abstract`): a deterministic, stdlib-only clustering of recurring
> episodic terms into durable concepts (dry-run by default, `--apply` to form, idempotent). A
> specificity (IDF) gate drops corpus-saturating generic terms and ranks the rest by tf-idf, engaging
> only once the corpus exceeds `ABSTRACT_IDF_MIN_DOCS` (below that, document frequency is uninformative).
> `cognition._build_concepts` projects those events into semantic memory — each concept is
> also materialized as a graph node (`type='concept'`, `layer='semantic'`) wired to its
> evidence episodes, so concepts **seed attention** (scaled by salience) and surface in
> `WorkingMemory.concepts` + the rendered lens. Concept-free brains render byte-for-byte as
> before. Covered by `tests/test_phase2_semantic.py` (5 tests).
> *Outcome: meaning persists after episodes cool.*

> **Phase 3 — Procedural memory. ✅ BUILT.**
> Added the `procedures`/`procedure_evidence` tables and the `procedure` edge to the graph schema
> (gated by `meta.schema_version=3` so pre-Phase-3 caches rebuild instead of skipping), the
> `procedure_learned` event (curator-emitted, agent="memory"), and the **T2 procedure-learning pass**
> (`reflect.py learn`): a deterministic, stdlib-only distillation of recurring *successful* workflows —
> it considers only threads that reached a `loop_closed`, signatures each by its ordered step terms
> (command/artifact/api_call/decision), and learns a procedure for any signature recurring across
> `PROC_MIN_SUPPORT` successful threads, drawing the `trigger` cue from those threads'
> objective/open_loop/error bodies (dry-run by default, `--apply` to learn, idempotent).
> `cognition._build_procedures` projects those events into procedural memory — each procedure is
> also materialized as a graph node (`type='procedure'`, `layer='procedural'`) wired to its evidence
> episodes. Unlike always-on concepts, a procedure is **trigger-gated**: it seeds attention and
> surfaces in `WorkingMemory.procedures` + the rendered lens ("Procedures (applicable how-to)") only
> when its trigger matches the current situation (intent query + live open loops). Procedure-free
> brains render byte-for-byte as before. Covered by `tests/test_phase3_procedural.py` (7 tests).
> *Outcome: the system improves at how it works, not just what it knows.*

> **Phase 4 — Persistent hybrid retrieval. ✅ BUILT.**
> Added the `vectors` table to the graph schema (gated by `meta.schema_version=4` so pre-Phase-4
> caches rebuild instead of skipping): `build()` now computes **one embedding per node and persists
> it** (`cognition._build_vectors`), so a query embeds only *itself* and runs a nearest-neighbour
> scan over the stored vectors instead of re-encoding the whole corpus on every call. Two backends
> share one persisted format and one query path (`_vector_search`): the **stdlib default** is a
> deterministic L2-normalized TF-IDF sparse vector (no dependency; its idf is persisted in `meta` so
> query embedding needs no corpus re-scan), and the **dense upgrade** stores sentence-transformers
> float vectors when `CONTINUITY_DENSE` is set *and* the package is installed. Vector hits seed
> attention (`_seed`) as the first tier of a graceful chain — **persisted vectors → BM25 → lexical
> LIKE** — so when no usable vectors exist (empty table, or dense vectors with the model unavailable
> at query time) candidate generation degrades silently to keyword retrieval. Vectors only *accelerate*
> and *associate*; spreading activation over the graph still decides. Covered by
> `tests/test_phase4_hybrid.py` (8 tests). *Outcome: associative recall; vectors accelerate, graph
> still decides.*

> **Phase 5 — Consolidation pipeline + snapshots. ✅ BUILT.**
> Formalized the T2 "sleep cycle" as one entry point, `reflect.py consolidate` (gated by
> `meta.schema_version=5` so pre-Phase-5 caches rebuild instead of skipping): it runs
> **abstraction → procedure learning → stale-belief pruning → snapshot/compaction** in order,
> dry-run by default, `--apply` to write, never blocking an interactive turn. New pieces:
> (a) **snapshot/compaction** — `cognition.snapshot` records a `snapshot` event (curator-emitted,
> agent="memory") + a derived-state blob under `runtime/snapshots/` and advances `meta.snapshot_seq`;
> `cognition._apply_compaction` then marks episodes behind the snapshot that no active thread cites
> as `nodes.cold` and **drops them from attention seeding** (`_seed`), so the *active* working set
> stays bounded over unlimited time while the nodes remain in the graph (spreading-reachable, fully
> replayable) and their meaning persists as concepts/procedures — open loops are never compacted.
> (b) **Stale-belief pruning** — `reflect.py prune` emits `concept_retired` for concepts whose
> evidence has all gone cold and which no active thread cites; `_build_concepts` drops them (a later
> `concept_formed` revives). (c) **Salience recalibration** — derived concept salience is now
> recency-weighted against its freshest evidence, so cold meaning fades while recurring concepts stay
> strong (an explicit curator salience still wins). (d) **Entropy caps** — `PRUNE_MAX` plus the
> existing `ABSTRACT_MAX_NEW`/`PROC_MAX_NEW`. The snapshot blob is a CACHE: a full replay reproduces
> the graph (incl. the cold set) byte-for-byte; the log stays whole and remains the only truth.
> Covered by `tests/test_phase5_consolidation.py` (11 tests). *Outcome: unlimited-timescale scaling —
> the active graph stays small while the log grows.*

> **Phase 6 — Attention dynamics. ✅ BUILT.**
> Layered the four §8 dynamics onto the V3 spreading-activation engine with no rewrite (gated by
> `meta.schema_version=6` so pre-Phase-6 caches rebuild instead of skipping): (1) **Resonance** —
> `cognition._build_resonance` adds Hebbian concept↔concept `resonance` edges (added to `EDGE_WEIGHT`)
> when two concepts' evidence co-occurs across ≥ `RESONANCE_MIN_COOC` threads ("fire together → wire
> together"); the weight saturates at `RESONANCE_NORM` and is capped at `RESONANCE_MAX_PER` per concept
> so the graph never saturates, and the edges spread like any other — activating one concept
> associatively lifts its resonant peers beyond causal/thread proximity. It is a deterministic
> projection of the log (fully replayable). (2) **Incubation** — `cognition.incubate` runs a query-less,
> low-gain diffusion that lifts dormant-but-well-connected nodes into a separate `incubation` channel,
> which `_seed` folds in *after* cold-filtering, so a neglected idea can "resurface with an idea" next
> session (even one a snapshot compacted). (3) **Oscillation** — `attend(..., oscillate=True)`
> (`--oscillate`, OFF by default) dampens the current winner thread-cluster and lifts the runner-up to
> break fixation; default-off reproduces V3.1 ranking exactly. (4) **Cross-session persistence** —
> the existing per-agent `activation` table plus the incubation seed carry the last working set into
> resume seeding (§4.1 step 3). Incubation is wired into `reflect.py consolidate` as the offline pass.
> Covered by `tests/test_phase6_attention.py` (10 tests). *Outcome: associative + anti-fixation
> cognition.*

> **Phase 7 — Branching/versioning + multi-agent formalization. ✅ BUILT.**
> Closed out V4 with the evolvability layer, all over the one append-only log (gated by
> `meta.schema_version=7` so pre-Phase-7 caches rebuild instead of skipping). (1) **Branching/
> versioning** — events carry an optional `branch` field (absent ⇒ `main`, so every existing log
> projects byte-for-byte; `runtime.append` drops a redundant `branch="main"`), bracketed by the new
> `branch_open`/`branch_merge` truth events. A branch is a **labeled subset of the one log, never a
> fork of the file**: `cognition._events_for_branch` reconstructs a branch as *main + every merged
> branch + that branch*, so the default `build()` excludes an **unmerged** what-if (a hypothetical
> never pollutes the real timeline) while `branch_merge` folds it back in for the next build.
> `cognition.reconstruct(branch)` replays any branch into an **isolated** db (in-memory by default)
> and diffs it against main **without touching `runtime/cognition.db`** — "what if we'd chosen
> Postgres?" becomes a measurable replay, not a forked repo. `build()` and `reconstruct()` share one
> projection core (`_project`) so both are the same deterministic function of the log. CLI: `branches`,
> `reconstruct <branch>`, `build --branch <name>`. (2) **Agent-local working memory** —
> `WorkingMemory.for_agent(agent)` formalizes the §10 per-agent lens as a first-class object over the
> ONE shared graph, seeded by the agent's role focus (`ROLE_PROFILES`) **and** the threads in its own
> attention lane (`_agent_threads`, reading the V3.1 per-agent `thread_activation` channel); `agent=''`
> reduces exactly to the single-agent `from_graph`. (3) **Curator arbitration** — the memory curator
> (`CURATOR_AGENT="memory"`) is the single consolidation authority: the consolidation event types
> (`concept_formed`, `procedure_learned`, `concept_retired`, `assumption_invalidated`, `loop_closed`,
> `snapshot`) are CONCLUSIONS only it emits; other agents *propose*. Append-only means a violation is
> never rejected (no lost writes) but made **auditable** — `runtime.append` flags a curator-owned event
> written under a named non-curator agent, and `cognition audit` (`arbitration_violations`) lists them.
> No locks: arbitration is a consolidation projection, coordination stays event-only. Covered by
> `tests/test_phase7_branching.py` (13 tests). *Outcome: hypothetical reconstruction and clean
> multi-agent evolution — all from the one log that remains the only truth.*

> **Phase 8 — Pattern recognition layer. ✅ BUILT.**
> Promoted *patterns* to first-class memory objects (§13), all over the one append-only log
> (gated by `meta.schema_version=8` so pre-Phase-8 caches rebuild instead of skipping). A T2
> mining pass (`reflect.py mine`, dry-run by default, `--apply` to write; also stage [3/6] of
> `reflect.py consolidate`) reads the projected graph and runs four deterministic, stdlib
> detectors: **reasoning** (a contiguous n-gram of cognitive event *types* recurring across
> threads — a repeated reasoning structure), **failure** (a failure signal —
> `assumption_invalidated`/`contradiction`/`error`/aging open loop — whose leading term recurs),
> **success** (a thread reaching `loop_closed` whose step leading-term sequence recurs — a
> reusable execution path, the §16 promotion hook), and **bottleneck** (a node with many causal
> *dependents* inside a still-unresolved thread). Each detection emits a curator-owned
> `pattern_detected` event (`agent="memory"`); mining is idempotent (existing pattern_ids are
> skipped) and entropy-capped (`PATTERN_MAX_NEW`). `cognition._build_patterns` projects these into
> the new `patterns`/`pattern_evidence` tables with `confidence`/`frequency`/`recommended`, and
> materializes each pattern as a graph **node** (`type='pattern'`, `layer='pattern'`) wired to its
> evidence by `pattern` edges (`EDGE_WEIGHT['pattern']`). Patterns then seed attention scaled by
> confidence (`PATTERN_SEED`), so a live context *surfaces the pattern it is repeating* and its
> recommended actions. CLI: `cognition patterns`. Absent `pattern_detected` events ⇒ empty pattern
> layer (exactly pre-Phase-8 behavior). Covered by `tests/test_phase8_patterns.py`. *Outcome:
> pattern-aware cognition — the system recognizes what it is doing again.*

> **Phase 9 — Adaptive procedural learning. ✅ BUILT.**
> Turns the static `procedures` table (Phase 3) into self-improving procedural cognition (§14). The
> new episodic truth event `proc_executed` `{procedure_id, outcome: success|failure, thread_id, t}`
> records each reuse — *anyone* may emit it (NOT curator-owned), the score it feeds is a curator
> projection. `cognition._build_procedures` folds executions chronologically into a
> reinforcement-weighted `outcome_score` via a bounded delta-rule
> (`score += PROC_LEARN_RATE * (target − score)`, target 1.0 success / 0.0 failure, `PROC_LEARN_RATE
> = 0.25`, clamped [0,1]) so effective strategies strengthen and failing ones decay with recency;
> each execution also counts as a use and advances `last_used_t` (the Phase 3 `uses ≥ 1` invariant
> holds). Consolidation gains a `reflect.py retire-procedures` pass (T2 stage [5/7]) that emits a
> curator `procedure_retired` for any procedure whose `outcome_score` fell below `PROC_RETIRE_SCORE`
> (0.25) after at least `PROC_RETIRE_MIN_USES` (3) executions; the curator-owned `procedure_retired`
> drops it from the projection and a later `procedure_learned` revives it. Retrieval already prefers
> high-scoring, context-matched procedures (`PROCEDURE_SEED · outcome_score`). SCHEMA_VERSION 8→9,
> append-only and replayable: the score is a deterministic projection of the execution log. Tests:
> `tests/test_phase9_adaptive_procedures.py`. *Outcome: the workflows that consistently succeed get
> stronger; the ones that fail fade.*

> **Phase 10 — Meta-cognition layer. ✅ BUILT.**
> A lightweight self-evaluation pass (§15): `reflect.py introspect` reads five *already-derived*
> signals — retrieval effectiveness (hit-rate of seeded nodes the turn reinforced, `activation`⋈
> `nodes.reinforced`), procedural success (mean reinforced `outcome_score` of executed procedures,
> §14), cognitive drift (the Stage-4 `detect_drift` divergence, quantified), context pollution
> (share of cold / low-activation nodes in the active window, `META_POLLUTION_FLOOR 0.05`), and
> attention instability (a `topic_shift`-churn proxy over the recent window). It folds them into a
> single `confidence` scalar (mean health, lower-is-better metrics inverted) and emits ONE
> curator-owned `meta_assessment` event carrying `confidence`, per-metric readings, and declarative
> strategy `nudges` (`widen_retrieval` / `review_procedures` / `rerun_retrieval` /
> `trigger_compaction` / `enable_incubation`) the next pass MAY honour — never auto-mutating here.
> Bounded and single-pass: a `meta_assessment` is never itself an input to meta-cognition (no
> recursion). A metric with no signal reads null and is simply excluded from the fold (an empty
> graph self-assesses to null and proposes nothing). Curator-owned in both `runtime` and
> `cognition` arbitration sets; the assessment projects as a replayable `layer='meta'` node. Runs as
> the final stage `[8/8]` of the `consolidate` sleep cycle (so it grades the state that cycle
> leaves behind) and is also exposed as `reflect.py introspect [--apply]` (dry-run by default). No
> SCHEMA_VERSION bump — the event folds through the generic node projection, so existing caches stay
> valid. Tests: `tests/test_phase10_metacognition.py`. *Outcome: the system monitors its own
> reasoning quality and adapts, without instability.*

> **Phase 11 — Pattern-to-procedure pipeline. ✅ BUILT.**
> Closes the loop (§16): a mined `kind="success"` pattern (Phase 8) whose confidence clears
> `PROMOTE_CONF_FLOOR` (0.70) is promoted into a procedural heuristic (Phase 9). `reflect.py promote`
> emits ONE curator `pattern_promoted` event linking `pattern_id → procedure_id` and carrying the
> seed; `cognition._build_procedures` then materializes the procedure with its `steps` taken from the
> pattern's defining recurring sequence, its `trigger` from the pattern's recurring entry context (the
> contributing threads' objective/open_loop/error cues), and its initial `outcome_score` **seeded from
> the pattern confidence** (the promotion itself is not counted as a `use` — only `proc_executed`/learn
> reinforce). Subsequent real uses emit `proc_executed` (§14), whose reinforcement then *optimizes* the
> promoted heuristic — observation → pattern → procedure → optimized strategy. Guards: only `success`
> patterns promote; promotion is **idempotent** (one procedure per pattern — a pattern already linked by
> a prior `pattern_promoted` is skipped, so a promotion §14-retirement withdrew is NOT silently revived;
> the self-correction is sticky). `pattern_promoted` is curator-owned in both arbitration sets and
> projects as a `layer='meta'` linkage node. Runs as the `[4/9]` stage of the `consolidate` sleep cycle
> (right after pattern mining, so a pattern just recorded can close the loop within the same cycle) and
> is also exposed as `reflect.py promote [--apply]` (dry-run by default). No SCHEMA_VERSION bump — the
> event folds through the existing procedural projection, so existing caches stay valid. Tests:
> `tests/test_phase11_promotion.py`. *Outcome: emergent operational intelligence from the one log.*

Every phase maps to a measurable gain — Phases 0/5 to scalability, 2/3/4 to retrieval quality and
reasoning coherence, 1/6 to continuity fidelity, 7 to evolvability, 8–11 to adaptive cognition
(pattern awareness, self-improving procedures, self-evaluation, emergent operational intelligence) —
and none adds an always-on dependency.

---

## 12. What V4 deliberately does NOT do

- No replacement of the event log as truth, and no non-stdlib hard dependency.
- No full multi-agent runtime (design only, per the brief).
- No speculative AGI constructs: every primitive here maps to a concrete table, event type, or pass,
  and to one of the five gains (continuity, coherence, scalability, retrieval quality, persistence).

**End state:** a continuously reconstructed cognitive runtime that separates remembering *what
happened* (episodic), *what it means* (semantic), and *how to proceed* (procedural); focuses them with
a real attention system; consolidates episodes into durable knowledge offline; and scales to unlimited
timescales by snapshotting the past while keeping the active graph small — all from one append-only log
that remains the only truth.

---

# V4.1 extension — Adaptive cognition

Sections 13–16 extend V4 from *persistent information storage* toward *persistent adaptive
cognition*: the system should not only remember, but **learn, adapt, optimize, and predict**. Every
addition here holds the V4 invariants verbatim — pure-stdlib default, local-first, `runtime/events.jsonl`
as the only truth, every derived structure a rebuildable projection, optional accelerators degrade
gracefully, single-agent behavior unchanged unless explicitly opted in, and absent fields/tables ⇒
exactly current behavior. Patterns, execution outcomes, and meta-assessments are all **derived
projections of new truth events**, never a second source of truth.

## 13. Pattern recognition layer

Memory retrieval recalls *what happened*. Pattern-aware cognition recognizes *what is happening
again*. Patterns become **first-class memory objects** — a fourth `mem_class` ('pattern') alongside
episodic/semantic/procedural — derived from the projected graph, never hand-authored.

**What is mined** (a T2 offline pass, `patterns.py mine`, dry-run by default, `--apply` to emit):
- **recurring reasoning structures** — repeated subgraph motifs in the causal DAG (same shape of
  decision→action→outcome chain across threads).
- **repeated failure modes** — causal chains that recurrently terminate in `assumption_invalidated`,
  contradiction, or an unclosed loop.
- **successful execution paths** — chains that recurrently reach `loop_closed` / passing artifacts.
- **temporal correlations** — event types that reliably co-occur or follow within a thread window.
- **cognitive bottlenecks / workflow inefficiencies** — nodes with many causal *dependents*
  (a prerequisite many events rely on) inside a still-unresolved thread; loops that re-open
  after closing (`loop recurrence`).
- **cross-project conceptual similarity** — concepts whose evidence spans projects/branches and
  cluster by resonance + vector similarity.

**Storage (schema additions, `SCHEMA_VERSION` bumped 7→8 so older caches rebuild):**
```sql
CREATE TABLE patterns (
    pattern_id   TEXT PRIMARY KEY,    -- pat_<kind>_<slug>
    kind         TEXT,                -- 'reasoning' | 'failure' | 'success' | 'bottleneck' | 'temporal' | 'similarity'
    label        TEXT,
    confidence   REAL DEFAULT 0.5,    -- support-weighted, in [0,1]
    frequency    INTEGER DEFAULT 1,   -- # of observed instances
    recommended  TEXT,                -- JSON: recommended_actions
    last_seen_t  TEXT
);
CREATE TABLE pattern_evidence (pattern_id TEXT, ref_kind TEXT, ref_id TEXT);  -- threads/events/concepts
```
(The 'pattern' memory class is carried by the existing `nodes.layer` column — `layer='pattern'` —
joining episodic/semantic/procedural, rather than a separate `mem_class` column.) A
`pattern_detected` truth event (curator-owned, `agent="memory"`) carries the conclusion;
`cognition._build_patterns` projects it. Each pattern is also a node, joined to its evidence by
`pattern` edges (`EDGE_WEIGHT['pattern']`), so spreading activation **surfaces the pattern a live
context is repeating** — e.g. activating a debugging thread lifts `pat_fail_timeout`. Pure stdlib:
motif and sequence detection run over the existing `nodes`/`edges`/`membership`/`causes` tables;
no new dependency. Example object:
```json
{ "pattern_id": "pat_debugging-loop", "kind": "reasoning", "confidence": 0.91,
  "frequency": 14, "associated_threads": ["thr_…"], "recommended_actions": ["bisect before patching"] }
```

## 14. Adaptive procedural learning

Phase 3 stores procedures statically. V4.1 makes them **self-improving**: procedures learn from
execution, reinforce what works, weaken what fails, and evolve.

- **Execution tracking.** A new `proc_executed` truth event records each reuse:
  `{procedure_id, outcome: "success"|"failure", thread_id, t}`. It is *episodic truth* — anyone may
  emit it; the score it feeds is a curator projection.
- **Reinforcement weighting.** `cognition._build_procedures` folds executions into `outcome_score`
  via a bounded reinforcement update (success raises, failure lowers, with recency decay so old
  outcomes fade) and updates `uses` / `last_used_t`. The score is a deterministic replay of the
  execution log — no hidden state.
- **Strategy optimization / execution-path scoring.** Competing procedures for the same `trigger`
  are ranked by `outcome_score`; retrieval injects the best context-matched one ("last time this
  worked: …").
- **Adaptive workflow evolution + procedural abstraction.** Consolidation (§9 T2) may **synthesize**
  a new procedure from a recurring successful pattern (the §16 pipeline), **generalize** a family of
  near-identical successful chains into one heuristic, and **retire** chronically-failing procedures
  (`procedure_retired`, mirroring `concept_retired`). Entropy caps bound new/generalized procedures
  per pass, exactly like `ABSTRACT_MAX_NEW`.

This answers, from the log alone: which workflows consistently succeed, which reasoning chains fail,
which retrieval/planning approaches scale best — and steers future retrieval toward the winners.

## 15. Meta-cognition layer

A **lightweight, bounded** self-evaluation subsystem — *thinking about its own thinking* without
runaway recursion. `reflect.py introspect` runs one pass (T1/T2, never inside an interactive turn)
over already-derived signals:

| Monitored | Source signal |
|---|---|
| retrieval effectiveness | hit-rate of seeded nodes that the turn actually used (activation → citation) |
| procedural success rate | mean `outcome_score` of recently executed procedures (§14) |
| cognitive drift | topic divergence between consecutive working sets (existing drift check, quantified) |
| context pollution | share of cold / low-relevance nodes in the active window |
| attention instability | activation-rank churn across refreshes (oscillation proxy) |

It emits a `meta_assessment` event with a `confidence` scalar and per-metric readings — a curator
projection, fully replayable. **Strategy adaptation** consumes it as bounded, declarative nudges:
poor recall ⇒ widen retrieval (lower the seed threshold / add a lexical tier); detected fixation ⇒
enable incubation/oscillation next pass; high pollution ⇒ trigger compaction. Single pass, no
self-referential loop: meta-assessments are never themselves inputs to meta-assessment.

## 16. Pattern-to-procedure pipeline

The capstone that turns observation into operational skill:

```
observed pattern (§13)  →  procedural knowledge (§14)  →  optimized strategy
```

When a §13 pattern of `kind="success"` (a recurring successful sequence) exceeds a confidence
threshold, the T2 consolidation pass promotes it into a procedure: it emits a `pattern_promoted`
event linking `pattern_id → procedure_id`, and `_build_procedures` materializes the procedure with
its `trigger` taken from the pattern's recurring entry context and its initial `outcome_score`
seeded from the pattern confidence. Subsequent real uses emit `proc_executed` (§14), whose
reinforcement then **optimizes** the promoted heuristic — closing the loop. Example:
a recurring successful debugging sequence is detected → promoted into `prc_bisect-before-patch` →
reinforced over future executions. Guards: only `success` patterns promote; promotion is idempotent
(one procedure per pattern); a promoted procedure that then chronically fails is retired by §14, so
a bad promotion self-corrects. This is the mechanism by which the system grows **emergent operational
intelligence** — adaptive, optimizing, and predictive — rather than static memory retrieval, while
every step remains a replayable projection of the one append-only log.
