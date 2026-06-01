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

> **Phase 7 — Branching/versioning + multi-agent formalization.**
> Optional `branch` field + `branch_open/merge`; agent-local `WorkingMemory`; curator arbitration.
> *Outcome: hypothetical reconstruction and clean multi-agent evolution.*

Every phase maps to a measurable gain — Phases 0/5 to scalability, 2/3/4 to retrieval quality and
reasoning coherence, 1/6 to continuity fidelity, 7 to evolvability — and none adds an always-on
dependency.

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
