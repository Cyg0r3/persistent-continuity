#!/usr/bin/env python3
"""
cognition.py - Persistent Continuity Architecture v3 (cognition-native core).

Implements the v3 inversion from ARCHITECTURE_V3.md: memory is a self-maintaining
COGNITIVE GRAPH focused by ATTENTION, not a collection of sessions/documents.

Layers built here (all CACHE - rebuildable from runtime/events.jsonl, the only truth):
  - L2 graph   : nodes(events) + thread membership + causal/thread/contradiction edges
  - L3 attention: spreading activation (diffusion) with decay, reinforcement, attention
                  persistence, competing-hypothesis inhibition, dormant resurfacing
  - L4 workspace: an ephemeral attention-synthesized lens over the ACTIVE SUBGRAPH
                  (replaces the stored working_context.md; markdown = debug/export)

Sessions are NOT a primitive here. Restoration = graph reconstruction + active-subgraph
extraction (NOT session replay). Determinism is at the event layer: same events + same
intent seed => same subgraph.

Subcommands:
    python cognition.py build [--force]       Build the graph (cognition.db) from events
                                              (skips when up-to-date; --force always rebuilds)
    python cognition.py attend [query] [-n N] [--oscillate]  Compute attention; show top
                                              active nodes/threads (--oscillate: rotate focus)
    python cognition.py incubate              Subconscious pass: lift dormant well-connected
                                              nodes into the channel that seeds the next pass
    python cognition.py context [query] [--write]  Synthesize the dynamic workspace lens
    python cognition.py threads                List threads with state (active/dormant/merged)
    python cognition.py subgraph [query] [-n N]    Show the extracted active subgraph
    python cognition.py roles                  List multi-agent roles and their focus types
    python cognition.py refresh [--hook]       Stage-4 adaptive refresh: on topic drift,
                                               checkpoint + re-inject fresh workspace
    python cognition.py snapshot [--apply]     T2 consolidation checkpoint: record a snapshot
                                               so cold episodes compact (dry-run unless --apply)
    python cognition.py build --branch <name>  Reconstruct a labeled branch into the cache
    python cognition.py branches               List branches (main/merged/open) + event counts
    python cognition.py reconstruct <branch>   Replay a hypothetical branch in isolation
                                               (mainline cache untouched) and diff vs main
    python cognition.py audit                  Curator-arbitration audit: flag consolidation
                                               events emitted by a non-curator agent
    python cognition.py patterns               List recognized patterns (v4.1 §13) with
                                               confidence, frequency, and recommended actions

Multi-agent (§9): attend/context/subgraph accept --agent <planner|coder|researcher|memory>
and --threads <t1,t2> to compute a role-biased attention WINDOW over the one shared graph
(constrained seed, full-graph spread; per-agent attention persistence). No --agent => the
single-agent path, unchanged.

Pure stdlib. Reuses retrieval.salience as base activation and runtime.read_events as the
single event-reading contract; both degrade to local fallbacks if unavailable.
"""

import os
import sys
import json
import math
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import data_root
# Data root is decoupled from this code (see _paths.py): per-project .continuity/
# by default, or $CONTINUITY_HOME. Plugin code itself is shared/read-only.
ROOT = data_root()
RUNTIME = ROOT / "runtime"
EVENTS = RUNTIME / "events.jsonl"
COG_DB = RUNTIME / "cognition.db"
THREADS_JSON = RUNTIME / "threads.json"
WORKING_CONTEXT = RUNTIME / "working_context.md"
SNAPSHOTS = RUNTIME / "snapshots"          # v4 Phase 5: derived-state snapshot blobs (cache)

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── attention parameters (ARCHITECTURE_V3.md §5) ─────────────────────────────
HOP_LIMIT = 2          # <=2-3: bounded relationship expansion
BETA = 0.6             # base-activation retention per cycle
GAMMA = 0.5            # spread gain from neighbours
HOP_DECAY = 0.6        # attenuation per hop
ALPHA = 0.7            # attention persistence: A = α.A_new + (1-α).A_prev
INHIBITION = 0.35      # competing-hypothesis mutual inhibition fraction (per round)
INHIBITION_ROUNDS = 4  # iterative lateral inhibition: amplifies evidence asymmetry
                       # (winner-take-most). Equal inputs stay equal (correct: no lean).
UNRESOLVED_SEED = 0.9  # open loops are always seeded (bias to finishing)
RESURFACE_BONUS = 0.25 # lift for a dormant thread reached by current intent
DORMANT_DAYS = 14.0    # thread inactivity horizon -> dormant
HALF_LIFE_DAYS = 7.0
REINFORCE_STEP = 0.06  # base-activation floor added per `attended` reinforcement
REINFORCE_CAP = 0.30   # ceiling on cumulative reinforcement (avoids runaway)
REINFORCE_TOP_N = 3    # how many focus nodes get an `attended` event on reinforce

EDGE_WEIGHT = {"causal": 1.0, "membership": 0.7, "thread_rel": 0.4,
               "concept": 0.6,        # v4 Phase 2: concept <-> evidence episode (semantic recall)
               "procedure": 0.6,      # v4 Phase 3: procedure <-> evidence episode (how-to recall)
               "resonance": 0.5,      # v4 Phase 6: concept <-> concept association (Hebbian)
               "pattern": 0.55,       # v4.1 Phase 8: pattern <-> evidence (surfaces a repeat)
               "contradiction": 0.0}  # contradiction handled by inhibition, not spread
CONCEPT_SEED = 0.55       # v4 Phase 2: in-scope concepts seed attention (scaled by salience)
PROCEDURE_SEED = 0.6      # v4 Phase 3: a procedure seeds attention only when its trigger
                          # matches the current situation (scaled by outcome_score)
PATTERN_SEED = 0.5        # v4.1 Phase 8: a recurring pattern seeds attention scaled by its
                          # confidence, so an active context surfaces the pattern it repeats
PROC_LEARN_RATE = 0.25    # v4.1 Phase 9: reinforcement step — each proc_executed nudges
                          # outcome_score toward 1.0 (success) or 0.0 (failure); recency-weighted

# ── v4 Phase 6 attention dynamics (ARCHITECTURE_V4.md §8) ────────────────────
# (1) Resonance: concepts whose evidence co-occurs across >= MIN_COOC threads "wire
# together"; weight grows with co-occurrence (saturating at NORM) and is capped at
# MAX_PER edges per concept so the graph never saturates.
RESONANCE_MIN_COOC = 2
RESONANCE_NORM = 4.0
RESONANCE_MAX_PER = 6
# (2) Incubation: a query-less low-gain diffusion that lifts dormant-but-well-connected
# nodes into a separate channel, folded back as a small seed next pass (resurfacing).
INCUBATE_GAMMA = 0.25
INCUBATE_ROUNDS = 2
INCUBATE_SEED = 0.30
INCUBATION_WEIGHT = 0.5   # how strongly the incubation channel seeds the next attention pass
# (3) Oscillation: anti-fixation focus rotation (OFF by default) — dampen the current
# winner cluster, lift the runner-up, surfacing neglected-but-relevant threads.
OSC_INHIBIT = 0.25
OSC_LIFT = 0.20

# ── multi-agent attention windows (ARCHITECTURE_V3.md §9) ────────────────────
# Each agent computes its OWN attention over the ONE shared graph: same engine,
# a role-biased SEED (not a separate store). A window is a constrained seed —
# spreading still runs over the full graph so cross-agent context stays reachable
# (soft bias, not a hard filter). `agent=None`/'' is the single-agent path today.
ROLE_SEED = 0.55          # weight applied to a node whose type matches the role's focus
ASSIGNED_THREAD_BOOST = 0.65   # seed lift for members of an agent's assigned threads
# role -> event types the agent's attention is biased toward (the rest still spread in).
ROLE_PROFILES = {
    "planner":    {"objective", "open_loop", "decision", "reflection", "topic_shift"},
    "coder":      {"artifact", "command", "output", "error", "api_call"},
    "researcher": {"observation", "hypothesis", "contradiction", "assumption"},
    "memory":     {"contradiction", "assumption", "assumption_invalidated",
                   "open_loop", "reflection"},
}

# ── v4 Phase 7 branching/versioning + multi-agent formalization (§3.3, §10) ────
# A branch is a labeled SUBSET of the one log, not a fork of the file. Events carry an
# optional `branch` (absent ⇒ "main", back-compatible); `branch_open`/`branch_merge`
# bracket a hypothetical line of cognition. Reconstructing branch B replays main + any
# merged branches + B, so the same log can render the real timeline OR a what-if without
# ever forking events.jsonl. The default build reconstructs "main".
DEFAULT_BRANCH = "main"
# Multi-agent arbitration (§10): the memory curator is the single consolidation authority.
CURATOR_AGENT = "memory"
CURATOR_TYPES = {
    "concept_formed", "procedure_learned", "concept_retired",
    "assumption_invalidated", "loop_closed", "snapshot",
    "pattern_detected", "procedure_retired",
}


def role_types(agent: str):
    """Focus event types for a role, or None for an unknown/blank agent (no bias)."""
    if not agent:
        return None
    return ROLE_PROFILES.get(agent.lower())


# ── reuse v2 base scoring; degrade locally if retrieval is unavailable ───────
try:
    import retrieval as _r
    IMPORTANCE = _r.IMPORTANCE
    DEFAULT_IMPORTANCE = _r.DEFAULT_IMPORTANCE
    _recency = _r.recency
except Exception:                                   # pragma: no cover
    IMPORTANCE, DEFAULT_IMPORTANCE = {}, 0.5
    def _recency(t):
        try:
            dt = datetime.fromisoformat(t)
        except (ValueError, TypeError):
            return 0.0
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return 0.5 ** (max(0.0, age) / HALF_LIFE_DAYS)


def now() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now().astimezone().replace(microsecond=0).isoformat()


def read_events() -> list:
    try:
        import runtime
        return runtime.read_events()
    except Exception:
        if not EVENTS.exists():
            return []
        out = []
        for raw in EVENTS.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                try:
                    out.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        return out


def _layer(e: dict) -> str:
    if e.get("layer"):
        return e["layer"]
    try:
        import runtime
        return runtime.infer_layer(e.get("type", ""))
    except Exception:
        return "meta"


def _body(e: dict) -> str:
    # Clean human text for display; `type` is a separate node column, so it is not
    # prefixed here (avoids "decision: decision ..." double-printing in the workspace).
    parts = []
    for k in ("msg", "title", "path", "reason", "resolution"):
        v = e.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
    return " ".join(parts) or e.get("type", "")


def _as_list(v) -> list:
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str) and v:
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


# ─── L2: graph build ─────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE nodes (
    event_id TEXT PRIMARY KEY, seq INTEGER, t TEXT, type TEXT, layer TEXT,
    body TEXT, importance REAL, unresolved INTEGER DEFAULT 0,
    reinforced INTEGER DEFAULT 0,
    cold INTEGER DEFAULT 0          -- v4 Phase 5: compacted (seq<=snapshot_seq, thread cold)
);
CREATE TABLE membership (event_id TEXT, thread_id TEXT);
CREATE TABLE edges (src TEXT, dst TEXT, kind TEXT, weight REAL);
CREATE TABLE threads (
    thread_id TEXT PRIMARY KEY, title TEXT, opened_t TEXT, last_activity_t TEXT,
    status TEXT, merged_into TEXT, n_events INTEGER DEFAULT 0
);
CREATE TABLE activation (
    agent TEXT DEFAULT '', event_id TEXT, base REAL, activation REAL,
    reinforced INTEGER DEFAULT 0, PRIMARY KEY (agent, event_id)
);
CREATE TABLE thread_activation (
    agent TEXT DEFAULT '', thread_id TEXT, activation REAL, resurfaced INTEGER DEFAULT 0,
    PRIMARY KEY (agent, thread_id)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- v4 Phase 2 semantic memory: durable concepts + their evidence back to episodes.
CREATE TABLE concepts (
    concept_id TEXT PRIMARY KEY, summary TEXT, salience REAL,
    support INTEGER DEFAULT 0, formed_t TEXT
);
CREATE TABLE concept_evidence (concept_id TEXT, event_id TEXT);
-- v4 Phase 3 procedural memory: reusable how-to workflows + their evidence episodes.
CREATE TABLE procedures (
    procedure_id TEXT PRIMARY KEY, label TEXT, steps TEXT, trigger TEXT,
    outcome_score REAL DEFAULT 0.5, uses INTEGER DEFAULT 0, last_used_t TEXT
);
CREATE TABLE procedure_evidence (procedure_id TEXT, event_id TEXT);
-- v4 Phase 4 persistent hybrid retrieval: one persisted embedding per node so query-time
-- candidate generation is a single query-embed + nearest-neighbour scan over `vec`
-- (instead of re-encoding the whole corpus every call). `model` tags the embedder that
-- produced the vectors; `vec` is JSON (dense float list, or sparse {token: weight} for
-- the stdlib TF-IDF default). Empty/absent ⇒ retrieval falls back to BM25 then lexical.
CREATE TABLE vectors (event_id TEXT PRIMARY KEY, model TEXT, dim INTEGER, vec TEXT);
-- v4 Phase 6 attention dynamics: a separate "subconscious" activation channel written by
-- the query-less incubation pass; folded back into the next attention seed (resurfacing).
CREATE TABLE incubation (event_id TEXT PRIMARY KEY, lift REAL);
-- v4.1 Phase 8 pattern recognition: recurring reasoning/failure/success/bottleneck patterns
-- mined from the projected graph and recorded as curator `pattern_detected` truth. Patterns
-- are first-class memory objects: each is also a node (layer='pattern') wired to its evidence
-- by `pattern` edges, so an active context surfaces the pattern it is repeating. `recommended`
-- is JSON (recommended_actions); `confidence`/`frequency` are reinforced by re-detection.
CREATE TABLE patterns (
    pattern_id TEXT PRIMARY KEY, kind TEXT, label TEXT,
    confidence REAL DEFAULT 0.5, frequency INTEGER DEFAULT 1,
    recommended TEXT, last_seen_t TEXT
);
CREATE TABLE pattern_evidence (pattern_id TEXT, ref_kind TEXT, ref_id TEXT);
CREATE INDEX ix_mem_thread ON membership(thread_id);
CREATE INDEX ix_edges_dst ON edges(dst);
CREATE INDEX ix_edges_src ON edges(src);
CREATE INDEX ix_cev_concept ON concept_evidence(concept_id);
CREATE INDEX ix_pev_procedure ON procedure_evidence(procedure_id);
CREATE INDEX ix_patev_pattern ON pattern_evidence(pattern_id);
"""

# Bump when the cache SCHEMA or its projection logic changes so an older cognition.db
# never silently satisfies the watermark skip (forces one rebuild). Phase 2 added the
# concepts/concept_evidence tables and the `concept` edge; Phase 3 added the
# procedures/procedure_evidence tables and the `procedure` edge; Phase 4 added the
# `vectors` table (persisted embeddings for hybrid retrieval); Phase 5 added the
# `nodes.cold` compaction flag, snapshot bookkeeping (meta.snapshot_seq), concept
# retirement (`concept_retired`) and recency-recalibrated concept salience; Phase 6 added
# `resonance` concept<->concept edges and the `incubation` activation channel; Phase 7
# added branch-filtered projection (the default mainline build excludes unmerged branches);
# Phase 8 added the patterns/pattern_evidence tables, pattern nodes (layer='pattern') and the
# `pattern` edge (recurring patterns as first-class memory objects).
SCHEMA_VERSION = 9


def _build_watermark(db_path: Path) -> int:
    """Event count the current cognition.db was built from, or -1 if unknown/stale.

    Returns -1 (forcing a rebuild) whenever the DB is missing, predates the v4
    `meta` table, or was built under an older SCHEMA_VERSION, so an older cache never
    silently satisfies the watermark check."""
    if not db_path.exists():
        return -1
    try:
        conn = sqlite3.connect(db_path)
        ver = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if not ver or int(ver[0]) != SCHEMA_VERSION:
            conn.close()
            return -1
        row = conn.execute(
            "SELECT value FROM meta WHERE key='event_count'").fetchone()
        conn.close()
        return int(row[0]) if row else -1
    except sqlite3.Error:
        return -1


# ── v4 Phase 7: branch filtering (a branch is a labeled subset of the one log) ──
def _branch_of(e: dict) -> str:
    return e.get("branch") or DEFAULT_BRANCH


def _merged_branches(events: list) -> set:
    """Branches folded back into the mainline via a `branch_merge` event."""
    merged = set()
    for e in events:
        if e.get("type") == "branch_merge":
            nm = e.get("branch") or e.get("from")
            if nm and nm != DEFAULT_BRANCH:
                merged.add(nm)
    return merged


def _events_for_branch(events: list, branch: str = DEFAULT_BRANCH) -> list:
    """Replay set for reconstructing `branch`: the mainline + every merged branch + the
    requested branch itself. An unmerged what-if branch is excluded from `main`, so a
    hypothetical never pollutes the real timeline — yet both come from the one log."""
    include = {DEFAULT_BRANCH} | _merged_branches(events)
    if branch:
        include.add(branch)
    return [e for e in events if _branch_of(e) in include]


def branches(events: list = None) -> list:
    """List every branch in the log with its status (main / merged / open) + event count."""
    if events is None:
        events = read_events()
    merged = _merged_branches(events)
    opened, counts = {}, {}
    for e in events:
        b = _branch_of(e)
        counts[b] = counts.get(b, 0) + 1
        if e.get("type") == "branch_open":
            nm = e.get("branch") or e.get("name")
            if nm:
                opened[nm] = {"opened_t": e.get("t"),
                              "base": e.get("from") or DEFAULT_BRANCH}
    names = set(counts) | set(opened) | merged | {DEFAULT_BRANCH}
    out = []
    for nm in sorted(names):
        status = ("main" if nm == DEFAULT_BRANCH
                  else "merged" if nm in merged else "open")
        out.append({"branch": nm, "status": status, "events": counts.get(nm, 0),
                    "base": opened.get(nm, {}).get("base", DEFAULT_BRANCH)})
    return out


def arbitration_violations(events: list = None) -> list:
    """Curator-arbitration audit (v4 Phase 7 / §10): consolidation-type events emitted by
    a named non-curator agent. Append-only means these are never rejected at write time;
    this surfaces them as an auditable list. Returns [{event_id, type, agent}]."""
    if events is None:
        events = read_events()
    out = []
    for seq, e in enumerate(events, 1):
        et, ag = e.get("type", ""), e.get("agent")
        if et in CURATOR_TYPES and ag and ag != CURATOR_AGENT:
            out.append({"event_id": e.get("event_id") or f"evt_{seq}",
                        "type": et, "agent": ag})
    return out


def build(db_path: Path = COG_DB, force: bool = False,
          branch: str = DEFAULT_BRANCH) -> dict:
    """Rebuild the cognitive graph from zero (drop-and-rebuild = pure projection).

    v4 Phase 0 — watermark skip: the graph is a pure function of the event log, so if
    the log hasn't grown since the last build the existing cognition.db is already a
    correct projection and we skip the O(N) rebuild. `force=True` always rebuilds
    (full-rebuild fallback for integrity / schema changes). Correctness still rests on
    the log; the watermark only avoids redundant work.

    v4 Phase 7 — `branch` selects which labeled subset of the log to reconstruct
    (default "main"; absent fields ⇒ main, so existing logs project byte-for-byte). The
    watermark/skip fast-path only applies to the default mainline build."""
    events = read_events()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if not force and branch == DEFAULT_BRANCH and _build_watermark(db_path) == len(events):
        conn = sqlite3.connect(db_path)
        summary = {
            "events": len(events),
            "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
            "threads": conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0],
            "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
            "skipped": True,
        }
        conn.close()
        return summary
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    summary = _project(conn, _events_for_branch(events, branch))

    # watermark: record the FULL event count (so log growth is detected) + schema version.
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('event_count',?)",
                 (str(len(events)),))
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                 (str(SCHEMA_VERSION),))
    if branch != DEFAULT_BRANCH:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('branch',?)",
                     (branch,))
    conn.commit()
    summary["events"] = len(events)
    summary["skipped"] = False
    conn.close()
    _export_threads(summary)
    return summary


def reconstruct(branch: str, db_path: Path = None) -> dict:
    """Reconstruct a hypothetical (or any) branch WITHOUT touching the mainline cache.

    Builds the graph for `branch` into an isolated db (in-memory by default) and returns
    its summary — so "what if we'd chosen Postgres?" can be replayed and measured against
    main without forking the log or clobbering runtime/cognition.db."""
    events = read_events()
    conn = sqlite3.connect(str(db_path) if db_path else ":memory:")
    conn.executescript(SCHEMA)
    summary = _project(conn, _events_for_branch(events, branch))
    summary["events"] = len(_events_for_branch(events, branch))
    summary["branch"] = branch
    conn.close()
    return summary


def _project(conn, events: list) -> dict:
    """Project a (branch-filtered) event list into the graph cache; returns the summary
    counts. Shared by build() (mainline, persisted) and reconstruct() (what-if, isolated)
    so both paths are byte-for-byte the same projection."""
    closed = {e.get("id") for e in events if e.get("type") == "loop_closed"}
    # reinforcement: count `attended` events per target event_id (§5.3, replayable)
    reinforced: dict = {}
    for e in events:
        if e.get("type") == "attended" and e.get("target"):
            reinforced[e["target"]] = reinforced.get(e["target"], 0) + 1
    threads: dict = {}        # thread_id -> meta
    cooc: dict = {}           # (a,b) sorted -> shared-event count

    def eid_of(e, seq):
        return e.get("event_id") or f"evt_{seq}"

    for seq, e in enumerate(events, 1):
        etype = e.get("type", "")
        eid = eid_of(e, seq)
        unresolved = 1 if (etype == "open_loop" and e.get("id") not in closed) else 0
        conn.execute(
            "INSERT OR REPLACE INTO nodes"
            "(event_id,seq,t,type,layer,body,importance,unresolved,reinforced)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (eid, seq, e.get("t"), etype, _layer(e), _body(e),
             IMPORTANCE.get(etype, DEFAULT_IMPORTANCE), unresolved,
             reinforced.get(eid, 0)))

        # thread lifecycle events
        if etype == "thread_open":
            tid = e.get("thread_id") or e.get("id")
            if tid:
                threads.setdefault(tid, _new_thread(tid))
                threads[tid]["title"] = e.get("title") or threads[tid]["title"]
                threads[tid]["opened_t"] = e.get("t")
        elif etype == "thread_merge":
            tid, into = e.get("thread_id"), e.get("into")
            if tid:
                threads.setdefault(tid, _new_thread(tid))
                threads[tid]["status"] = "merged"
                threads[tid]["merged_into"] = into
            if into:
                conn.execute("INSERT INTO edges VALUES (?,?,?,?)",
                             (tid, into, "thread_rel", 1.0))
        elif etype == "thread_split":
            spawn = e.get("spawn")
            if spawn:
                threads.setdefault(spawn, _new_thread(spawn))
                conn.execute("INSERT INTO edges VALUES (?,?,?,?)",
                             (e.get("thread_id"), spawn, "thread_rel", 1.0))

        # membership (any event may belong to threads)
        tids = _as_list(e.get("thread_ids"))
        for tid in tids:
            threads.setdefault(tid, _new_thread(tid))
            threads[tid]["n_events"] += 1
            lt = threads[tid]["last_activity_t"]
            if not lt or (e.get("t") or "") > lt:
                threads[tid]["last_activity_t"] = e.get("t")
            conn.execute("INSERT INTO membership VALUES (?,?)", (eid, tid))
            conn.execute("INSERT INTO edges VALUES (?,?,?,?)",
                         (eid, tid, "membership", EDGE_WEIGHT["membership"]))
        # derived thread_rel via co-membership
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                key = tuple(sorted((tids[i], tids[j])))
                cooc[key] = cooc.get(key, 0) + 1

        # causal edges (event -> event)
        for c in _as_list(e.get("causes")):
            conn.execute("INSERT INTO edges VALUES (?,?,?,?)",
                         (c, eid, "causal", EDGE_WEIGHT["causal"]))

        # contradiction: a `contradiction` event links the two rival hypotheses
        # via `causes` (or explicit `between`). Creates a mutual-inhibition edge.
        if etype == "contradiction":
            pair = _as_list(e.get("between")) or _as_list(e.get("causes"))
            if len(pair) >= 2:
                conn.execute("INSERT INTO edges VALUES (?,?,?,?)",
                             (pair[0], pair[1], "contradiction", 1.0))

    for key, w in cooc.items():
        conn.execute("INSERT INTO edges VALUES (?,?,?,?)",
                     (key[0], key[1], "thread_rel", EDGE_WEIGHT["thread_rel"] * w))

    # finalize thread status (dormant by inactivity unless merged)
    for tid, th in threads.items():
        if th["status"] != "merged":
            th["status"] = "dormant" if _is_dormant(th["last_activity_t"]) else "active"
        conn.execute(
            "INSERT OR REPLACE INTO threads"
            "(thread_id,title,opened_t,last_activity_t,status,merged_into,n_events)"
            " VALUES (?,?,?,?,?,?,?)",
            (tid, th["title"], th["opened_t"], th["last_activity_t"],
             th["status"], th["merged_into"], th["n_events"]))

    # v4 Phase 2: materialize semantic memory from concept_formed events (curator truth).
    _build_concepts(conn, events)
    # v4 Phase 3: materialize procedural memory from procedure_learned events.
    _build_procedures(conn, events)
    # v4.1 Phase 8: materialize recurring patterns from pattern_detected events.
    _build_patterns(conn, events)
    # v4 Phase 6: Hebbian concept<->concept resonance edges (associative spreading).
    _build_resonance(conn)
    # v4 Phase 4: persist one embedding per node so retrieval is a stored-vector scan.
    _build_vectors(conn)
    # v4 Phase 5: compact cold episodes behind the latest snapshot (keeps the active
    # graph small over unlimited time; the log stays whole, meaning persists as concepts).
    n_cold = _apply_compaction(conn, events)
    conn.commit()
    return {
        "events": len(events),
        "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "threads": len(threads),
        "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "concepts": conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
        "procedures": conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0],
        "patterns": conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0],
        "vectors": conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0],
        "cold": n_cold,
    }


def _concept_id_of(e: dict) -> str:
    return e.get("concept_id") or e.get("id") or ""


def _build_concepts(conn, events: list) -> None:
    """Project `concept_formed` events into semantic memory (v4 Phase 2 §5.1).

    Concepts are curator-emitted truth: each event names a concept, a one-line summary,
    and the evidence episodes it abstracts. Replaying in order accumulates support and
    evidence (a later `concept_formed` for the same id reinforces it). Each concept is
    ALSO materialized as a graph node (type='concept', layer='semantic') wired to its
    evidence by `concept` edges, so spreading activation flows episode<->meaning and the
    concept can seed/surface in working memory. Truth stays in the log; this is a pure
    projection — absent concept_formed events ⇒ empty semantic memory (current behavior).

    v4 Phase 5: a `concept_retired` event (curator-emitted) retires a stale concept — it is
    skipped on materialization (a later `concept_formed` revives it). Derived salience is
    recency-recalibrated against the freshest evidence, so the meaning of cold episodes
    fades while recurring concepts stay strong; an explicit event `salience` still wins."""
    acc: dict = {}   # concept_id -> {summary, salience, support, evidence:set, formed_t}
    for e in events:
        etype = e.get("type")
        if etype not in ("concept_formed", "concept_retired"):
            continue
        cid = _concept_id_of(e)
        if not cid:
            continue
        if etype == "concept_retired":
            if cid in acc:
                acc[cid]["retired"] = True   # terminal until a later concept_formed revives
            continue
        ev = _as_list(e.get("evidence"))
        c = acc.setdefault(cid, {"summary": "", "salience": 0.0, "support": 0,
                                 "evidence": set(), "formed_t": e.get("t"), "retired": False})
        c["retired"] = False                 # (re)forming revives a previously retired concept
        c["summary"] = e.get("summary") or e.get("msg") or c["summary"] or cid
        c["evidence"].update(ev)
        # support: explicit, else cumulative distinct evidence count; reinforces over time.
        c["support"] = int(e.get("support", max(len(c["evidence"]), c["support"] + 1)))
        if e.get("salience") is not None:
            c["salience"] = float(e["salience"])
    # freshest-evidence timestamp per concept drives salience recalibration (Phase 5).
    node_t = {r[0]: r[1] for r in conn.execute("SELECT event_id, t FROM nodes")}
    for cid, c in acc.items():
        if c.get("retired"):
            continue
        if c["salience"]:                    # explicit curator salience is authoritative
            sal = c["salience"]
        else:
            ev_ts = [node_t[e] for e in c["evidence"] if node_t.get(e)]
            rec = _recency(max(ev_ts)) if ev_ts else 1.0
            base = min(0.9, 0.4 + 0.1 * c["support"])
            sal = round(base * (0.5 + 0.5 * rec), 4)   # decay cold meaning, floor at half
        conn.execute(
            "INSERT OR REPLACE INTO concepts(concept_id,summary,salience,support,formed_t)"
            " VALUES (?,?,?,?,?)", (cid, c["summary"], sal, c["support"], c["formed_t"]))
        # concept node so it participates in attention / working memory
        conn.execute(
            "INSERT OR REPLACE INTO nodes"
            "(event_id,seq,t,type,layer,body,importance,unresolved,reinforced)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (cid, None, c["formed_t"], "concept", "semantic",
             c["summary"], sal, 0, 0))
        for eid in c["evidence"]:
            conn.execute("INSERT INTO concept_evidence VALUES (?,?)", (cid, eid))
            conn.execute("INSERT INTO edges VALUES (?,?,?,?)",
                         (cid, eid, "concept", EDGE_WEIGHT["concept"]))


def _procedure_id_of(e: dict) -> str:
    return e.get("procedure_id") or e.get("id") or ""


def _build_procedures(conn, events: list) -> None:
    """Project `procedure_learned` events into procedural memory (v4 Phase 3 §5.2).

    Procedures are curator-emitted truth: each event names a procedure, a label, the
    ordered `steps`, a `trigger` cue (when it applies), and the evidence episodes the
    workflow was distilled from. Replaying in order accumulates uses + evidence (a later
    `procedure_learned` for the same id reinforces it, lifting its outcome_score). Each
    procedure is ALSO materialized as a graph node (type='procedure', layer='procedural')
    wired to its evidence by `procedure` edges, so spreading activation flows
    episode<->how-to and the procedure can seed/surface in working memory once its trigger
    matches. Truth stays in the log; absent procedure_learned events ⇒ empty procedural
    memory (current behavior).

    v4.1 Phase 9 (§14): the procedure is self-improving. `proc_executed` events (episodic
    truth, anyone may emit) are folded chronologically into `outcome_score` by a bounded
    reinforcement delta-rule (`score += PROC_LEARN_RATE * (target - score)`, target 1.0 for
    success / 0.0 for failure), so consistently-succeeding workflows strengthen and failing
    ones decay with recency. Each execution also counts as a use and advances last_used_t. A
    curator `procedure_retired` event withdraws a chronically-failing procedure (skip
    materialization); a later `procedure_learned` revives it. The score is a pure
    deterministic replay of the execution log — no hidden state."""
    acc: dict = {}   # procedure_id -> {label, steps:list, trigger:list, score, uses, evidence:set, last_t}
    retired: set = set()
    for e in events:
        et = e.get("type")
        if et == "procedure_retired":
            pid = _procedure_id_of(e)
            if pid:
                retired.add(pid)
            continue
        if et == "proc_executed":
            pid = _procedure_id_of(e)
            if not pid or pid not in acc:
                continue                      # execution of an unknown/unlearned procedure: ignore
            p = acc[pid]
            outcome = str(e.get("outcome", "success")).lower()
            target = 1.0 if outcome not in ("failure", "fail", "error", "0") else 0.0
            base = p["score"] if p["score"] is not None else 0.5
            p["score"] = max(0.0, min(1.0, base + PROC_LEARN_RATE * (target - base)))
            p["uses"] += 1
            p["last_t"] = e.get("t") or p["last_t"]
            continue
        if et != "procedure_learned":
            continue
        pid = _procedure_id_of(e)
        if not pid:
            continue
        retired.discard(pid)                  # re-learning revives a previously retired procedure
        ev = _as_list(e.get("evidence"))
        p = acc.setdefault(pid, {"label": "", "steps": [], "trigger": [], "score": None,
                                 "uses": 0, "evidence": set(), "last_t": e.get("t")})
        p["label"] = e.get("label") or e.get("summary") or e.get("msg") or p["label"] or pid
        trig = _as_list(e.get("trigger"))
        if trig:
            p["trigger"] = trig
        steps = e.get("steps")
        if isinstance(steps, list) and steps:
            p["steps"] = [str(s) for s in steps]
        elif isinstance(steps, str) and steps:
            p["steps"] = [steps]
        p["evidence"].update(ev)
        p["uses"] += 1                       # each learned/reinforced emission = one use
        p["last_t"] = e.get("t") or p["last_t"]
        if e.get("outcome_score") is not None:
            p["score"] = float(e["outcome_score"])
    for pid, p in acc.items():
        if pid in retired:
            continue                          # chronically-failing procedure withdrawn by curator
        # outcome_score: explicit/reinforced, else a confidence that climbs with learn uses.
        score = p["score"] if p["score"] is not None else round(min(0.9, 0.5 + 0.1 * (p["uses"] - 1)), 4)
        conn.execute(
            "INSERT OR REPLACE INTO procedures"
            "(procedure_id,label,steps,trigger,outcome_score,uses,last_used_t)"
            " VALUES (?,?,?,?,?,?,?)",
            (pid, p["label"], "\n".join(p["steps"]), ",".join(p["trigger"]),
             score, p["uses"], p["last_t"]))
        # procedure node so it participates in attention / working memory
        conn.execute(
            "INSERT OR REPLACE INTO nodes"
            "(event_id,seq,t,type,layer,body,importance,unresolved,reinforced)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, None, p["last_t"], "procedure", "procedural",
             p["label"], score, 0, 0))
        for eid in p["evidence"]:
            conn.execute("INSERT INTO procedure_evidence VALUES (?,?)", (pid, eid))
            conn.execute("INSERT INTO edges VALUES (?,?,?,?)",
                         (pid, eid, "procedure", EDGE_WEIGHT["procedure"]))


def _pattern_id_of(e: dict) -> str:
    return e.get("pattern_id") or e.get("id") or ""


def _ref_kind(ref: str) -> str:
    """Classify a pattern-evidence reference by id convention (thread vs concept vs episode)."""
    if ref.startswith("thr_"):
        return "thread"
    if ref.startswith("cpt_"):
        return "concept"
    if ref.startswith("prc_"):
        return "procedure"
    return "event"


def _build_patterns(conn, events: list) -> None:
    """Project `pattern_detected` events into the pattern layer (v4.1 Phase 8 §13).

    Patterns are curator-emitted truth: each event names a pattern, its `kind`
    (reasoning/failure/success/temporal/bottleneck/similarity), a label, a `confidence`
    and `frequency`, the `recommended` actions, and the `evidence` it generalizes
    (thread/event/concept ids). Replaying in order reinforces a pattern (a later
    `pattern_detected` for the same id lifts confidence/frequency). Each pattern is ALSO
    materialized as a graph node (type='pattern', layer='pattern') wired to its evidence by
    `pattern` edges, so spreading activation flows context<->pattern and the pattern
    *surfaces the recurrence a live context is repeating*. Truth stays in the log; absent
    pattern_detected events ⇒ empty pattern memory (pre-Phase-8 behavior)."""
    acc: dict = {}   # pattern_id -> {kind,label,confidence,frequency,recommended,evidence:list,last_t}
    for e in events:
        if e.get("type") != "pattern_detected":
            continue
        pid = _pattern_id_of(e)
        if not pid:
            continue
        p = acc.setdefault(pid, {"kind": "", "label": "", "confidence": 0.0,
                                 "frequency": 0, "recommended": [], "evidence": [],
                                 "last_t": e.get("t")})
        p["kind"] = e.get("kind") or p["kind"] or "reasoning"
        p["label"] = e.get("label") or e.get("summary") or e.get("msg") or p["label"] or pid
        if e.get("confidence") is not None:
            p["confidence"] = float(e["confidence"])
        freq = e.get("frequency")
        p["frequency"] = int(freq) if freq is not None else p["frequency"] + 1
        rec = _as_list(e.get("recommended"))
        if rec:
            p["recommended"] = [str(x) for x in rec]
        for ev in _as_list(e.get("evidence")):
            if ev not in p["evidence"]:
                p["evidence"].append(ev)
        p["last_t"] = e.get("t") or p["last_t"]
    for pid, p in acc.items():
        # confidence: explicit, else climbs with re-detection frequency (bounded).
        conf = (p["confidence"] if p["confidence"]
                else round(min(0.95, 0.5 + 0.1 * (p["frequency"] - 1)), 4))
        conn.execute(
            "INSERT OR REPLACE INTO patterns"
            "(pattern_id,kind,label,confidence,frequency,recommended,last_seen_t)"
            " VALUES (?,?,?,?,?,?,?)",
            (pid, p["kind"], p["label"], conf, p["frequency"],
             json.dumps(p["recommended"]), p["last_t"]))
        # pattern node so it participates in attention / working memory
        conn.execute(
            "INSERT OR REPLACE INTO nodes"
            "(event_id,seq,t,type,layer,body,importance,unresolved,reinforced)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, None, p["last_t"], "pattern", "pattern", p["label"], conf, 0, 0))
        for ref in p["evidence"]:
            conn.execute("INSERT INTO pattern_evidence VALUES (?,?,?)",
                         (pid, _ref_kind(ref), ref))
            # wire the pattern to its evidence so activation flows context<->pattern
            # (harmless if the ref is not itself a node — spreading just won't propagate).
            conn.execute("INSERT INTO edges VALUES (?,?,?,?)",
                         (pid, ref, "pattern", EDGE_WEIGHT["pattern"]))


def _build_resonance(conn) -> int:
    """Hebbian concept<->concept resonance edges (v4 Phase 6 §8.1): concepts that "fire
    together" — their evidence episodes co-occur in the same thread — "wire together", so
    activation spreads associatively between related meanings, beyond causal/thread/concept
    proximity. Co-occurrence = number of shared threads; the edge weight saturates at
    RESONANCE_NORM, fires only past RESONANCE_MIN_COOC, and is capped at RESONANCE_MAX_PER
    edges per concept so the graph never saturates. A pure projection of the log (the
    evidence/membership it reads is itself derived), so resonance is fully replayable."""
    concept_threads: dict = {}   # concept_id -> set(thread_id) of its evidence episodes
    for cid, tid in conn.execute(
            "SELECT ce.concept_id, m.thread_id FROM concept_evidence ce "
            "JOIN membership m ON m.event_id = ce.event_id"):
        concept_threads.setdefault(cid, set()).add(tid)

    cids = sorted(concept_threads)
    cooc: dict = {}
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            shared = len(concept_threads[cids[i]] & concept_threads[cids[j]])
            if shared >= RESONANCE_MIN_COOC:
                cooc[(cids[i], cids[j])] = shared

    # bound the graph: keep only each concept's strongest RESONANCE_MAX_PER associations
    by_concept: dict = {}
    for (a, b), w in cooc.items():
        by_concept.setdefault(a, []).append((w, b))
        by_concept.setdefault(b, []).append((w, a))
    keep: set = set()
    for c, lst in by_concept.items():
        for w, other in sorted(lst, reverse=True)[:RESONANCE_MAX_PER]:
            keep.add(tuple(sorted((c, other))))

    base = EDGE_WEIGHT["resonance"]
    n = 0
    for (a, b) in sorted(keep):
        w = cooc.get((a, b), 0)
        weight = round(base * min(1.0, w / RESONANCE_NORM), 4)   # saturating, bounded
        conn.execute("INSERT INTO edges VALUES (?,?,?,?)", (a, b, "resonance", weight))
        n += 1
    return n


def incubate(persist: bool = True) -> dict:
    """Subconscious incubation (v4 Phase 6 §8.2): a query-LESS, low-gain diffusion that lets
    dormant-but-well-connected nodes accrue a small lift, so an unsolved problem can
    "resurface with an idea" next session. Seeds from dormant-thread members weighted by
    connectivity, diffuses a couple of low-gain rounds, and writes the result to the
    separate `incubation` channel that `_seed` folds into the next attention pass. Pure
    function of the current graph (a recomputable cache); empty channel ⇒ no effect."""
    build()  # ensure the graph reflects the current log
    conn = sqlite3.connect(COG_DB)
    conn.row_factory = sqlite3.Row
    nodes = [r["event_id"] for r in conn.execute("SELECT event_id FROM nodes")]

    out_adj: dict = {}
    degree: dict = {}
    for r in conn.execute("SELECT src,dst,kind,weight FROM edges WHERE kind!='contradiction'"):
        out_adj.setdefault(r["src"], []).append((r["dst"], r["weight"]))
        out_adj.setdefault(r["dst"], []).append((r["src"], r["weight"]))
        degree[r["src"]] = degree.get(r["src"], 0) + 1
        degree[r["dst"]] = degree.get(r["dst"], 0) + 1

    dormant = {r["thread_id"] for r in conn.execute(
        "SELECT thread_id FROM threads WHERE status='dormant'")}
    seed: dict = {}
    if dormant:
        q = ",".join("?" * len(dormant))
        for r in conn.execute(
                f"SELECT event_id FROM membership WHERE thread_id IN ({q})", list(dormant)):
            eid = r["event_id"]
            # well-connected dormant nodes incubate more strongly (diminishing returns)
            seed[eid] = INCUBATE_SEED * (1.0 - math.exp(-degree.get(eid, 0) / 5.0))

    act = dict(seed)
    for _ in range(INCUBATE_ROUNDS):
        nxt: dict = {}
        for v in nodes:
            spread = sum(w * act.get(u, 0.0) for (u, w) in out_adj.get(v, []))
            val = INCUBATE_GAMMA * spread + seed.get(v, 0.0)
            if val > 0:
                nxt[v] = val
        act = nxt
    lift = {eid: round(a, 4) for eid, a in act.items() if a > 0.01}

    if persist:
        conn.execute("DELETE FROM incubation")
        conn.executemany("INSERT INTO incubation(event_id,lift) VALUES (?,?)",
                         list(lift.items()))
        conn.commit()
    conn.close()
    return {"incubated": len(lift), "persisted": persist}


def _apply_oscillation(conn, act: dict) -> None:
    """Anti-fixation focus rotation (v4 Phase 6 §8.3, OFF by default): dampen the current
    winner thread-cluster and lift the runner-up, so a neglected-but-relevant thread gets a
    turn. In-place on the settled activation map; parameterized, never auto-on."""
    members_by_thread: dict = {}
    for r in conn.execute("SELECT thread_id, event_id FROM membership"):
        members_by_thread.setdefault(r["thread_id"], []).append(r["event_id"])
    tmax = {tid: max((act.get(m, 0.0) for m in mems), default=0.0)
            for tid, mems in members_by_thread.items()}
    ranked = sorted((t for t in tmax if tmax[t] > 0), key=tmax.get, reverse=True)
    if len(ranked) < 2:
        return
    winner, runner = ranked[0], ranked[1]
    lift = OSC_LIFT * tmax[winner]
    for m in members_by_thread[winner]:
        if m in act:
            act[m] *= (1 - OSC_INHIBIT)
    for m in members_by_thread[runner]:
        if m in act:
            act[m] += lift


def _new_thread(tid: str) -> dict:
    return {"title": tid, "opened_t": None, "last_activity_t": None,
            "status": "active", "merged_into": None, "n_events": 0}


def _is_dormant(last_t: str) -> bool:
    if not last_t:
        return True
    try:
        dt = datetime.fromisoformat(last_t)
    except (ValueError, TypeError):
        return True
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now() - dt).total_seconds() / 86400.0 > DORMANT_DAYS


# ─── L3: attention / spreading activation (§5) ───────────────────────────────

def _base_activation(row) -> float:
    """Emit-time/decaying base (reuses the v2 salience ingredients) plus a
    reinforcement floor: nodes attended in past cycles resist decay (§5.3)."""
    imp = row["importance"] or DEFAULT_IMPORTANCE
    rec = _recency(row["t"])
    unres = UNRESOLVED_SEED if row["unresolved"] else 0.0
    reinf = min(REINFORCE_CAP, REINFORCE_STEP * (row["reinforced"] or 0))
    return round(0.5 * imp + 0.3 * rec + 0.2 * unres + reinf, 4)


def _seed(conn, query: str, agent: str = "", threads=None) -> dict:
    """Seed activation: query matches (via retrieval BM25 if available) + every
    unresolved open loop + nodes in currently-active threads.

    Multi-agent (§9): when `agent` names a role, nodes whose type matches the role's
    focus get an extra ROLE_SEED lift, and members of the agent's `threads` (its
    assignment) get an ASSIGNED_THREAD_BOOST. This is a SOFT bias on the seed only —
    spreading still happens over the full shared graph. `agent=''` => unchanged."""
    seed = {}
    # unresolved loops always seeded (bias to finishing)
    for r in conn.execute("SELECT event_id FROM nodes WHERE unresolved=1"):
        seed[r[0]] = max(seed.get(r[0], 0.0), UNRESOLVED_SEED)
    # active-thread members
    active = [r[0] for r in conn.execute(
        "SELECT thread_id FROM threads WHERE status='active'")]
    if active:
        q = ",".join("?" * len(active))
        for r in conn.execute(
                f"SELECT event_id FROM membership WHERE thread_id IN ({q})", active):
            seed[r[0]] = max(seed.get(r[0], 0.0), 0.5)
    # in-scope semantic memory: concepts seed attention scaled by salience, so durable
    # meaning resurfaces even when the originating episodes have cooled (v4 Phase 2 §5.1).
    try:
        for r in conn.execute("SELECT concept_id, salience FROM concepts"):
            seed[r[0]] = max(seed.get(r[0], 0.0), CONCEPT_SEED * (r[1] or 0.0))
    except sqlite3.Error:
        pass  # pre-Phase-2 cache without the concepts table; next build() adds it
    # procedural memory: a procedure is injected ONLY when its trigger matches the current
    # situation (intent query + the live open loops), not always-on like a concept — "last
    # time this worked: …". Scaled by outcome_score (v4 Phase 3 §5.2).
    for pid, score in _matched_procedures(conn, query).items():
        seed[pid] = max(seed.get(pid, 0.0), PROCEDURE_SEED * score)
    # v4.1 Phase 8: recurring patterns seed attention scaled by confidence, so a live
    # context surfaces the pattern it is repeating (and the pattern's recommended actions).
    try:
        for r in conn.execute("SELECT pattern_id, confidence FROM patterns"):
            seed[r[0]] = max(seed.get(r[0], 0.0), PATTERN_SEED * (r[1] or 0.0))
    except sqlite3.Error:
        pass  # pre-Phase-8 cache without the patterns table; next build() adds it
    # role-biased seeding: lift nodes matching this agent's focus types. ADDITIVE
    # overlay (not max) so the role actually reweights the window above the shared
    # floors (active-thread 0.5, unresolved 0.9); spreading is still over the full graph.
    focus = role_types(agent)
    if focus:
        q = ",".join("?" * len(focus))
        for r in conn.execute(
                f"SELECT event_id FROM nodes WHERE type IN ({q})", list(focus)):
            seed[r[0]] = seed.get(r[0], 0.0) + ROLE_SEED
    # assigned-thread boost: this agent's lane gets lifted, but not walled off
    assigned = _as_list(threads)
    if assigned:
        q = ",".join("?" * len(assigned))
        for r in conn.execute(
                f"SELECT event_id FROM membership WHERE thread_id IN ({q})", assigned):
            seed[r[0]] = seed.get(r[0], 0.0) + ASSIGNED_THREAD_BOOST
    # query relevance: persisted vectors (v4 Phase 4) -> BM25 -> lexical LIKE. Graceful
    # 3-tier chain. The vector tier scans the embeddings stored at build() rather than
    # re-encoding the corpus, so it is cheap and the same path serves dense and stdlib.
    if query:
        hits = _vector_search(conn, query, k=VECTOR_TOPK)
        if hits:
            for eid, sim in hits:
                seed[eid] = max(seed.get(eid, 0.0), 0.6 + 0.4 * sim)
        else:
            try:
                for h in _r.search(query, k=8):
                    eid = _eid_for_seq(conn, h["seq"])
                    if eid:
                        seed[eid] = max(seed.get(eid, 0.0), 0.6 + 0.4 * h["relevance"])
            except Exception:
                for r in conn.execute(
                        "SELECT event_id FROM nodes WHERE body LIKE ?",
                        (f"%{query}%",)):
                    seed[r[0]] = max(seed.get(r[0], 0.0), 0.7)
    # v4 Phase 5: cold (compacted) episodes are dropped from the seed so the active working
    # set stays bounded; they remain in the graph and can still be reached via spreading.
    try:
        cold = {r[0] for r in conn.execute("SELECT event_id FROM nodes WHERE cold=1")}
        if cold:
            seed = {eid: v for eid, v in seed.items() if eid not in cold}
    except sqlite3.Error:
        pass  # pre-Phase-5 cache without the cold column; next build() adds it
    # v4 Phase 6: fold the incubation channel in AFTER cold-filtering, so a dormant
    # (even compacted) but well-connected node can still "resurface with an idea".
    try:
        for r in conn.execute("SELECT event_id, lift FROM incubation"):
            seed[r[0]] = max(seed.get(r[0], 0.0), INCUBATION_WEIGHT * (r[1] or 0.0))
    except sqlite3.Error:
        pass  # pre-Phase-6 cache without the incubation table; next build() adds it
    return seed


def _matched_procedures(conn, query: str = "") -> dict:
    """Procedures whose `trigger` matches the current situation (v4 Phase 3 §5.2).

    The situation is the intent query plus every live (unresolved) open loop's body.
    Returns {procedure_id: outcome_score} for the matches — the gate that makes procedural
    memory trigger-driven ("last time this worked: …") rather than always-on like semantic
    memory. Empty (and silent) on a pre-Phase-3 cache without the procedures table."""
    matched: dict = {}
    try:
        situation = (query or "").lower()
        for r in conn.execute("SELECT body FROM nodes WHERE unresolved=1"):
            situation += " " + (r[0] or "").lower()
        for r in conn.execute(
                "SELECT procedure_id, trigger, outcome_score FROM procedures"):
            terms = [t.strip().lower() for t in (r[1] or "").split(",") if t.strip()]
            if terms and any(t in situation for t in terms):
                matched[r[0]] = r[2] or 0.0
    except sqlite3.Error:
        pass  # pre-Phase-3 cache without the procedures table; next build() adds it
    return matched


def _eid_for_seq(conn, seq):
    r = conn.execute("SELECT event_id FROM nodes WHERE seq=?", (seq,)).fetchone()
    return r[0] if r else None


# ─── persistent hybrid retrieval (v4 Phase 4 §11) ───────────────────────────
#
# Embeddings are computed ONCE at build() and persisted in the `vectors` table, so a
# query only embeds itself and scans the stored vectors (nearest-neighbour) — the graph
# still decides, the vectors just accelerate candidate generation. Two backends share
# one persisted format and one query path:
#   • stdlib default — a deterministic TF-IDF sparse vector (no dependency); cosine over
#     it is associative term-overlap recall, strictly better than substring LIKE.
#   • dense upgrade  — sentence-transformers float vectors when CONTINUITY_DENSE is set
#     AND the package is installed.
# Either way, if no usable vectors exist (empty table, or dense vectors but the model is
# unavailable at query time) the chain degrades cleanly to BM25, then lexical LIKE.

DENSE_PREFIX = "st:"            # model tag prefix for dense (sentence-transformers) vectors
LEXICAL_MODEL = "lexical-tfidf"  # model tag for the stdlib default vectors
VECTOR_TOPK = 8                # candidates a vector scan contributes to the seed


class _Embedder:
    """Pluggable DENSE embedder (ARCHITECTURE_V3.md §5; v4 Phase 4).

    OFF by default — enabled only when CONTINUITY_DENSE is truthy AND
    sentence-transformers is installed. Anthropic has no embeddings endpoint, so dense is
    a local-model upgrade; when absent, _build_vectors persists stdlib TF-IDF vectors and
    retrieval still works. This keeps the stdlib-default invariant intact."""

    def __init__(self):
        self.model = None
        self.name = os.environ.get("CONTINUITY_EMBED_MODEL", "all-MiniLM-L6-v2")
        if os.environ.get("CONTINUITY_DENSE", "").lower() in ("1", "true", "yes"):
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.name)
            except Exception:
                self.model = None  # not installed / load failed -> stdlib TF-IDF vectors

    def available(self) -> bool:
        return self.model is not None

    def model_tag(self) -> str:
        return DENSE_PREFIX + self.name

    def encode(self, texts: list) -> list:
        """Return a normalized float vector (list) per text, or [] on any failure."""
        if not self.available() or not texts:
            return []
        try:
            vecs = self.model.encode(list(texts), normalize_embeddings=True)
            return [[round(float(x), 6) for x in row] for row in vecs]
        except Exception:
            return []


_EMBEDDER = _Embedder()  # singleton; cheap when disabled


def _vec_tokenize(text: str) -> list:
    """Tokenizer for the stdlib TF-IDF backend: reuse semantic_search.tokenize
    (lowercase, depunctuate, drop stopwords) with a minimal inline fallback."""
    try:
        from semantic_search import tokenize
        return tokenize(text or "")
    except Exception:
        return [w for w in "".join(c if c.isalnum() else " "
                for c in (text or "").lower()).split() if len(w) > 1]


def _lexical_tfidf(texts: list):
    """Build deterministic L2-normalized TF-IDF sparse vectors (pure stdlib).
    Returns (idf, vecs) where idf is {token: weight} and each vec is {token: weight}."""
    docs = [_vec_tokenize(t) for t in texts]
    n = max(len(docs), 1)
    df: dict = {}
    for toks in docs:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1
    idf = {tok: math.log(n / (c + 1)) + 1.0 for tok, c in df.items()}
    vecs = []
    for toks in docs:
        tf: dict = {}
        for tok in toks:
            tf[tok] = tf.get(tok, 0) + 1
        total = sum(tf.values()) or 1
        raw = {tok: (c / total) * idf[tok] for tok, c in tf.items()}
        norm = math.sqrt(sum(w * w for w in raw.values()))
        vecs.append({tok: round(w / norm, 6) for tok, w in raw.items()} if norm else {})
    return idf, vecs


def _build_vectors(conn) -> None:
    """Persist one embedding per node into `vectors` (v4 Phase 4). Prefers the dense
    backend when enabled+installed; otherwise stores stdlib TF-IDF vectors. The idf
    needed to embed a query against TF-IDF vectors is kept in meta so query-time embedding
    needs no corpus re-scan. A pure projection — re-derivable from the node bodies."""
    rows = conn.execute("SELECT event_id, body FROM nodes ORDER BY event_id").fetchall()
    if not rows:
        return
    ids = [r[0] for r in rows]
    texts = [r[1] or "" for r in rows]

    dense = _EMBEDDER.encode(texts) if _EMBEDDER.available() else []
    if dense and len(dense) == len(ids):
        tag = _EMBEDDER.model_tag()
        for eid, v in zip(ids, dense):
            conn.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                         (eid, tag, len(v), json.dumps(v)))
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('vector_model',?)", (tag,))
        conn.execute("DELETE FROM meta WHERE key='vector_idf'")
        return

    idf, vecs = _lexical_tfidf(texts)
    for eid, v in zip(ids, vecs):
        conn.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)",
                     (eid, LEXICAL_MODEL, len(v), json.dumps(v, separators=(",", ":"))))
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('vector_model',?)",
                 (LEXICAL_MODEL,))
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('vector_idf',?)",
                 (json.dumps(idf, separators=(",", ":")),))


def _vector_search(conn, query: str, k: int = VECTOR_TOPK) -> list:
    """Nearest-neighbour scan over persisted `vectors` (v4 Phase 4 candidate generation).

    Returns [(event_id, similarity in [0,1])] for the top matches, or [] to signal the
    caller to fall back to BM25/lexical. Returns [] when there are no vectors, when the
    stored vectors are dense but the matching model is unavailable now, or when the query
    shares no in-vocabulary terms — every degradation is silent."""
    if not query:
        return []
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='vector_model'").fetchone()
    except sqlite3.Error:
        return []  # pre-Phase-4 cache without the meta key; next build() populates it
    model = row[0] if row else None
    if not model:
        return []

    scored = []
    if model.startswith(DENSE_PREFIX):
        # dense vectors are only readable with the same model loaded at query time.
        if not _EMBEDDER.available() or _EMBEDDER.model_tag() != model:
            return []
        qv = _EMBEDDER.encode([query])
        if not qv:
            return []
        q = qv[0]
        for eid, vec in conn.execute("SELECT event_id, vec FROM vectors"):
            try:
                v = json.loads(vec)
            except (TypeError, ValueError):
                continue
            sim = sum(a * b for a, b in zip(q, v))  # cosine (vectors normalized)
            scored.append((eid, (sim + 1.0) / 2.0))  # map [-1,1] -> [0,1]
    else:
        irow = conn.execute("SELECT value FROM meta WHERE key='vector_idf'").fetchone()
        if not irow:
            return []
        idf = json.loads(irow[0])
        toks = _vec_tokenize(query)
        tf: dict = {}
        for tok in toks:
            if tok in idf:
                tf[tok] = tf.get(tok, 0) + 1
        total = sum(tf.values()) or 1
        raw = {tok: (c / total) * idf[tok] for tok, c in tf.items()}
        norm = math.sqrt(sum(w * w for w in raw.values()))
        if not norm:
            return []
        q = {tok: w / norm for tok, w in raw.items()}
        for eid, vec in conn.execute("SELECT event_id, vec FROM vectors"):
            try:
                v = json.loads(vec)
            except (TypeError, ValueError):
                continue
            sim = sum(q[t] * v[t] for t in q if t in v)  # cosine over shared terms
            if sim > 0:
                scored.append((eid, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [(eid, max(0.0, min(1.0, s))) for eid, s in scored[:k] if s > 0]


# ─── consolidation pipeline + snapshots (v4 Phase 5 §9) ──────────────────────

def _snapshot_seq(events: list) -> int:
    """The event count covered by the latest `snapshot` event (0 if none)."""
    snap = 0
    for e in events:
        if e.get("type") == "snapshot":
            try:
                snap = max(snap, int(e.get("snapshot_seq", e.get("seq", 0))))
            except (TypeError, ValueError):
                continue
    return snap


def _apply_compaction(conn, events: list) -> int:
    """Mark episodes COLD when they sit behind the latest snapshot and no longer belong
    to any active thread (v4 Phase 5). Cold = compacted: excluded from attention SEEDING
    (`_seed`) so the active working set stays bounded over unlimited time, while the nodes
    remain in the graph (spreading-reachable, fully replayable) and their meaning persists
    as concepts/procedures. Open loops (unresolved) are never compacted — finishing wins.
    Records `meta.snapshot_seq`. No snapshot ⇒ nothing cold ⇒ pre-Phase-5 behavior."""
    snap = _snapshot_seq(events)
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('snapshot_seq',?)",
                 (str(snap),))
    if snap <= 0:
        return 0
    active = [r[0] for r in conn.execute(
        "SELECT thread_id FROM threads WHERE status='active'")]
    if active:
        q = ",".join("?" * len(active))
        conn.execute(
            "UPDATE nodes SET cold=1 WHERE seq IS NOT NULL AND seq<=? AND unresolved=0 "
            f"AND event_id NOT IN (SELECT event_id FROM membership "
            f"WHERE thread_id IN ({q}))", [snap] + active)
    else:
        conn.execute("UPDATE nodes SET cold=1 WHERE seq IS NOT NULL AND seq<=? "
                     "AND unresolved=0", (snap,))
    return conn.execute("SELECT COUNT(*) FROM nodes WHERE cold=1").fetchone()[0]


def snapshot(apply: bool = False) -> dict:
    """T2 consolidation checkpoint (v4 Phase 5 §9): record a snapshot of derived state at
    the current event count, and (with apply) emit a `snapshot` event + a snapshot blob so
    later builds compact cold episodes behind it.

    The blob under runtime/snapshots/ is a CACHE — a full replay still reproduces the graph
    byte-for-byte; the `snapshot` event and `meta.snapshot_seq` are the bookkeeping that
    keeps the ACTIVE graph small. Curator-owned (agent='memory'); append-only, so it never
    edits truth. Dry-run by default."""
    build()  # ensure the graph reflects the current log before snapshotting
    events = read_events()
    seq = len(events)
    conn = sqlite3.connect(COG_DB)
    conn.row_factory = sqlite3.Row
    counts = {
        "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "threads": conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0],
    }
    blob = {
        "snapshot_seq": seq,
        "generated": now_iso(),
        "schema_version": SCHEMA_VERSION,
        "counts": counts,
        "concepts": [dict(r) for r in conn.execute(
            "SELECT concept_id,summary,salience,support FROM concepts ORDER BY concept_id")],
        "procedures": [dict(r) for r in conn.execute(
            "SELECT procedure_id,label,trigger,outcome_score FROM procedures "
            "ORDER BY procedure_id")],
        "patterns": [dict(r) for r in conn.execute(
            "SELECT pattern_id,kind,label,confidence,frequency FROM patterns "
            "ORDER BY pattern_id")],
        "threads": [dict(r) for r in conn.execute(
            "SELECT thread_id,title,status,n_events FROM threads ORDER BY thread_id")],
    }
    conn.close()

    written = None
    if apply:
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOTS / f"snap_{seq}.json"
        path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
        written = str(path.relative_to(ROOT))
        try:
            import runtime
            runtime.append("snapshot", agent="memory", snapshot_seq=seq,
                           nodes=counts["nodes"], concepts=len(blob["concepts"]),
                           procedures=len(blob["procedures"]))
        except Exception:
            pass
        build()  # rebuild so cold compaction applies behind the just-recorded snapshot
    return {"snapshot_seq": seq, "counts": counts, "concepts": len(blob["concepts"]),
            "procedures": len(blob["procedures"]), "written": written, "applied": apply}


def attend(query: str = "", persist: bool = True, agent: str = "", threads=None,
           oscillate: bool = False) -> dict:
    """Compute the attention state via spreading activation over the graph.

    `agent` (+ optional `threads` assignment) selects a role-biased attention window
    over the ONE shared graph (§9): the seed is constrained, the spread is not.
    Attention persistence is scoped per agent so concurrent windows never bleed.
    `agent=''` is the single-agent path (unchanged).

    `oscillate` (v4 Phase 6 §8.3, OFF by default) rotates focus off the current winner
    cluster onto the runner-up to break fixation; default False reproduces V3.1 behavior.

    Returns {"nodes": {eid: activation}, "threads": {tid: activation},
             "resurfaced": [tid], "contradictions": [(a,b,score_a,score_b)]}."""
    if not COG_DB.exists():
        build()
    conn = sqlite3.connect(COG_DB)
    conn.row_factory = sqlite3.Row

    nodes = {r["event_id"]: r for r in conn.execute("SELECT * FROM nodes")}
    base = {eid: _base_activation(r) for eid, r in nodes.items()}

    # adjacency for spread (causal flows both ways asymmetrically; membership/thread_rel
    # symmetric). contradiction edges are excluded from spread (handled as inhibition).
    out_adj: dict = {}
    for r in conn.execute("SELECT src,dst,kind,weight FROM edges"):
        if r["kind"] == "contradiction":
            continue
        w = r["weight"]
        out_adj.setdefault(r["src"], []).append((r["dst"], w))
        back = w if r["kind"] != "causal" else w * 0.6  # effect->cause weaker
        out_adj.setdefault(r["dst"], []).append((r["src"], back))

    # seed (role-biased when an agent window is requested)
    seed = _seed(conn, query, agent=agent, threads=threads)
    act = {eid: base[eid] + seed.get(eid, 0.0) for eid in nodes}

    # spreading activation, bounded hops
    for _ in range(HOP_LIMIT):
        nxt = {}
        for v in nodes:
            spread = 0.0
            for (u, w) in out_adj.get(v, []):
                if u in act:
                    spread += w * act[u] * HOP_DECAY
            nxt[v] = base[v] * BETA + GAMMA * spread + seed.get(v, 0.0)
        act = nxt

    # competing-hypothesis inhibition: iterative lateral inhibition. Each round both
    # rivals subtract a fraction of the other's CURRENT score (snapshot = simultaneous
    # update). Equal inputs stay equal (correct: no evidence to lean); any asymmetry
    # — e.g. a decision causally supporting one side lifts it via spread — is amplified
    # toward winner-take-most over the rounds.
    pairs = [(r["src"], r["dst"]) for r in conn.execute(
        "SELECT src,dst FROM edges WHERE kind='contradiction'")
        if r["src"] in act and r["dst"] in act]
    for _ in range(INHIBITION_ROUNDS):
        snap = dict(act)
        for a, b in pairs:
            act[a] = max(0.0, act[a] - INHIBITION * snap[b])
            act[b] = max(0.0, act[b] - INHIBITION * snap[a])
    contradictions = [(a, b, round(act[a], 4), round(act[b], 4)) for a, b in pairs]

    # attention persistence (blend with previous cycle) — scoped to this agent window
    if persist:
        prev = {r["event_id"]: r["activation"]
                for r in conn.execute(
                    "SELECT event_id,activation FROM activation WHERE agent=?", (agent,))}
        if prev:
            for eid in act:
                act[eid] = ALPHA * act[eid] + (1 - ALPHA) * prev.get(eid, 0.0)

    # v4 Phase 6 anti-fixation oscillation (opt-in): rotate focus to the runner-up cluster.
    if oscillate:
        _apply_oscillation(conn, act)

    # thread activation = max member activation; dormant resurfacing
    thread_act, resurfaced = {}, []
    for tr in conn.execute("SELECT * FROM threads"):
        tid = tr["thread_id"]
        members = [m[0] for m in conn.execute(
            "SELECT event_id FROM membership WHERE thread_id=?", (tid,))]
        score = max((act.get(m, 0.0) for m in members), default=0.0)
        if tr["status"] == "dormant" and score > 0.4:
            score += RESURFACE_BONUS
            resurfaced.append(tid)
            for m in members:                      # lift the resurfaced thread's events
                act[m] = act.get(m, 0.0) + RESURFACE_BONUS * 0.5
        thread_act[tid] = round(score, 4)

    # persist the new attention state (transient cache), scoped to this agent window
    conn.execute("DELETE FROM activation WHERE agent=?", (agent,))
    conn.executemany(
        "INSERT INTO activation(agent,event_id,base,activation) VALUES (?,?,?,?)",
        [(agent, eid, base[eid], round(act[eid], 4)) for eid in nodes])
    conn.execute("DELETE FROM thread_activation WHERE agent=?", (agent,))
    conn.executemany(
        "INSERT INTO thread_activation(agent,thread_id,activation,resurfaced) "
        "VALUES (?,?,?,?)",
        [(agent, tid, sc, 1 if tid in resurfaced else 0)
         for tid, sc in thread_act.items()])
    conn.commit()
    conn.close()

    return {"nodes": {e: round(a, 4) for e, a in act.items()},
            "threads": thread_act, "resurfaced": resurfaced,
            "contradictions": contradictions}


# ─── active subgraph extraction + workspace synthesis (§6, §7) ───────────────

def active_subgraph(query: str = "", top_n: int = 8, agent: str = "", threads=None,
                    oscillate: bool = False) -> dict:
    """Top-activation nodes expanded <=HOP_LIMIT hops = the active subgraph.

    `agent`/`threads` select a role-biased window (§9); defaults reproduce the
    single-agent subgraph exactly. `oscillate` (Phase 6) rotates focus off the winner."""
    state = attend(query, agent=agent, threads=threads, oscillate=oscillate)
    conn = sqlite3.connect(COG_DB)
    conn.row_factory = sqlite3.Row
    ranked = sorted(state["nodes"].items(), key=lambda kv: kv[1], reverse=True)
    core = [eid for eid, _ in ranked[:top_n]]

    # expand <=HOP_LIMIT hops along non-contradiction edges
    frontier, included = set(core), set(core)
    adj = {}
    for r in conn.execute("SELECT src,dst,kind FROM edges WHERE kind!='contradiction'"):
        adj.setdefault(r["src"], set()).add(r["dst"])
        adj.setdefault(r["dst"], set()).add(r["src"])
    for _ in range(HOP_LIMIT):
        nxt = set()
        for v in frontier:
            nxt |= adj.get(v, set())
        nxt -= included
        included |= nxt
        frontier = nxt

    rows = {r["event_id"]: r for r in conn.execute("SELECT * FROM nodes")}
    nodes = [{"event_id": eid, "type": rows[eid]["type"], "layer": rows[eid]["layer"],
              "body": rows[eid]["body"], "t": rows[eid]["t"],
              "activation": state["nodes"].get(eid, 0.0)}
             for eid in included if eid in rows]
    nodes.sort(key=lambda n: n["activation"], reverse=True)
    conn.close()
    return {"nodes": nodes, "threads": state["threads"],
            "resurfaced": state["resurfaced"],
            "contradictions": state["contradictions"], "core": core}


# ─── Phase 1: working memory as a runtime object (ARCHITECTURE_V4.md §4.2) ───

# Char budget for the rendered lens (~1,500-token guardrail). Mirrors runtime's
# WORKING_CONTEXT_BUDGET_CHARS; markdown is export only, so this bounds the export.
WORKING_MEMORY_BUDGET_CHARS = 6000


def _agent_threads(agent: str, limit: int = 4) -> list:
    """The threads in an agent's own scoped attention channel — its lane (v4 §10).

    Reads the per-agent `thread_activation` rows (V3.1 per-agent persistence) so an
    agent's working memory is seeded by where ITS attention already is, falling back to
    the globally-active threads for a cold agent. Returns None when nothing applies, so
    WorkingMemory.from_graph keeps its existing (assignment-free) behaviour."""
    if not COG_DB.exists():
        return None
    conn = sqlite3.connect(COG_DB)
    try:
        rows = conn.execute(
            "SELECT thread_id FROM thread_activation WHERE agent=? AND thread_id IS NOT NULL "
            "ORDER BY activation DESC LIMIT ?", (agent, limit)).fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT thread_id FROM threads WHERE status='active' LIMIT ?",
                (limit,)).fetchall()
    except sqlite3.Error:
        rows = []
    conn.close()
    return [r[0] for r in rows] or None


class WorkingMemory:
    """In-process attention lens over the active subgraph (L4).

    V3 synthesized this lens fresh every prompt and treated working_context.md as the
    artifact. V4 Phase 1 promotes it to a first-class runtime object: it is built once
    from the graph, then `mutate()`d incrementally as new events land (no full
    re-synthesis), `reprioritize()`d within a fixed budget, and `render()`ed to markdown
    only for export/debug. `concepts` (Phase 2) and `procedures` (Phase 3) carry the
    in-scope semantic/procedural memory; both stay empty for brains that have formed none,
    so absent ⇒ current behavior, per the backward-compat invariant."""

    def __init__(self, budget: int = WORKING_MEMORY_BUDGET_CHARS,
                 query: str = "", agent: str = ""):
        self.active: dict = {}        # event_id -> activation (current focus)
        self.threads: dict = {}       # thread_id -> activation
        self.concepts: list = []      # semantic context in scope  (Phase 2)
        self.procedures: list = []    # applicable how-to knowledge (Phase 3)
        self.budget = budget
        self.query = query
        self.agent = agent
        # detail backing render(), keyed for O(1) incremental mutation
        self._nodes: dict = {}        # event_id -> node detail row
        self._th_meta: dict = {}      # thread_id -> threads-table row
        self._concept_meta: dict = {} # concept_id -> concepts-table row
        self._procedure_meta: dict = {}  # procedure_id -> procedures-table row
        self._resurfaced: set = set()
        self._contradictions: list = []

    @classmethod
    def from_graph(cls, query: str = "", top_n: int = 8, agent: str = "",
                   threads=None, budget: int = WORKING_MEMORY_BUDGET_CHARS):
        """Synthesize working memory from the current active subgraph (initial seed)."""
        sg = active_subgraph(query, top_n, agent=agent, threads=threads)
        wm = cls(budget=budget, query=query, agent=agent)
        conn = sqlite3.connect(COG_DB)
        conn.row_factory = sqlite3.Row
        wm._th_meta = {r["thread_id"]: dict(r) for r in conn.execute("SELECT * FROM threads")}
        wm._concept_meta = {r["concept_id"]: dict(r)
                            for r in conn.execute("SELECT * FROM concepts")}
        try:
            wm._procedure_meta = {r["procedure_id"]: dict(r)
                                  for r in conn.execute("SELECT * FROM procedures")}
        except sqlite3.Error:
            wm._procedure_meta = {}   # pre-Phase-3 cache; rebuilt on next build()
        # procedural memory is trigger-gated: only procedures whose trigger matches the
        # current situation are eligible to surface (even if a procedure node spread-
        # activated from its evidence episodes). This is what distinguishes "how-to" recall
        # from always-on semantic recall.
        triggered = set(_matched_procedures(conn, query))
        conn.close()
        for n in sg["nodes"]:
            wm._nodes[n["event_id"]] = n
            wm.active[n["event_id"]] = n["activation"]
            # in-scope semantic memory: active concept nodes feed WorkingMemory.concepts
            if n["layer"] == "semantic" and n["event_id"] in wm._concept_meta:
                wm.concepts.append(n["event_id"])
            # in-scope procedural memory: a trigger-matched procedure node feeds .procedures
            if n["layer"] == "procedural" and n["event_id"] in triggered:
                wm.procedures.append(n["event_id"])
        wm.threads = dict(sg["threads"])
        wm._resurfaced = set(sg["resurfaced"])
        wm._contradictions = list(sg["contradictions"])
        return wm

    @classmethod
    def for_agent(cls, agent: str, query: str = "", top_n: int = 8,
                  budget: int = WORKING_MEMORY_BUDGET_CHARS) -> "WorkingMemory":
        """Agent-local working memory (v4 Phase 7 / §10): a first-class per-agent lens
        over the ONE shared graph, seeded by the agent's role focus (ROLE_PROFILES) and
        the threads currently in its own attention lane (_agent_threads). Each agent keeps
        a scoped focus without a separate store — coordination stays event-only, the graph
        stays shared. `agent=''` reduces to the single-agent from_graph path."""
        return cls.from_graph(query=query, top_n=top_n, agent=agent,
                              threads=_agent_threads(agent) if agent else None,
                              budget=budget)

    def mutate(self, event: dict) -> "WorkingMemory":
        """Fold one freshly-appended event into the lens WITHOUT rebuilding the graph.

        The new event is the most-recent signal, so it enters focus at the current peak
        activation (never below the existing top), and its threads are bumped likewise.
        Truth still lives in the log; this only keeps the in-process lens warm between
        builds. A subsequent build()+from_graph() reconciles to the exact projection."""
        eid = event.get("event_id")
        if not eid:
            return self
        peak = max(self.active.values(), default=0.0)
        act = max(peak, 1.0)
        self._nodes[eid] = {
            "event_id": eid, "type": event.get("type", ""),
            "layer": _layer(event), "body": _body(event),
            "t": event.get("t", now_iso()), "activation": act,
        }
        self.active[eid] = act
        for tid in _as_list(event.get("thread_ids")):
            self.threads[tid] = max(self.threads.get(tid, 0.0), act)
        return self

    def reprioritize(self) -> "WorkingMemory":
        """Re-rank focus by activation and drop the coldest nodes once render would
        exceed budget. Attention-driven, budget-bounded — the lens stays small."""
        ranked = sorted(self._nodes.values(), key=lambda n: n["activation"], reverse=True)
        # Greedy keep highest-activation nodes until the rendered size hits budget.
        while ranked and len(self._render(ranked)) > self.budget:
            dropped = ranked.pop()
            self.active.pop(dropped["event_id"], None)
            self._nodes.pop(dropped["event_id"], None)
        return self

    def _ranked_nodes(self) -> list:
        return sorted(self._nodes.values(), key=lambda n: n["activation"], reverse=True)

    def _render(self, nodes: list) -> str:
        threads_ranked = sorted(self.threads.items(), key=lambda kv: kv[1], reverse=True)
        cognitive = [n for n in nodes if n["layer"] == "cognitive"]
        loops = [n for n in cognitive if n["type"] == "open_loop"]
        L = []
        L.append(f"# Workspace @ {now_iso()} "
                 f"(active subgraph: {len(nodes)} nodes / "
                 f"{sum(1 for _, s in threads_ranked if s > 0)} threads)")
        L.append("")
        L.append("## Active threads")
        shown = [(t, s) for t, s in threads_ranked if s > 0][:5]
        if shown:
            for i, (tid, sc) in enumerate(shown, 1):
                meta = self._th_meta.get(tid)
                title = (meta["title"] if meta else tid) or tid
                flag = " ^resurfaced" if tid in self._resurfaced else ""
                state = f" [{meta['status']}]" if meta else ""
                L.append(f"{i}. [{sc:.2f}] {tid} - {title}{state}{flag}")
        else:
            L.append("(no threads yet - events are unthreaded)")
        L.append("")
        L.append("## Focus (highest activation)")
        for n in cognitive[:5]:
            L.append(f"- {n['type']} {n['event_id']}: {n['body'][:110]} [{n['activation']:.2f}]")
        if not cognitive:
            L.append("(no cognitive nodes active)")
        L.append("")
        # semantic memory in scope (v4 Phase 2). Section omitted when empty so the
        # rendered lens is byte-for-byte unchanged for pre-concept brains.
        active_concepts = [c for c in self.concepts if c in self._concept_meta]
        if active_concepts:
            ranked_c = sorted(active_concepts,
                              key=lambda c: self.active.get(c, 0.0), reverse=True)[:5]
            L.append("## Concepts in scope (semantic memory)")
            for cid in ranked_c:
                m = self._concept_meta[cid]
                L.append(f"- {cid}: {(m['summary'] or '')[:90]} "
                         f"(support {m['support']}, salience {m['salience']:.2f})")
            L.append("")
        # procedural memory in scope (v4 Phase 3). Trigger-matched how-to knowledge —
        # "last time this worked: …". Omitted when empty so pre-procedure brains render
        # byte-for-byte unchanged.
        active_procs = [p for p in self.procedures if p in self._procedure_meta]
        if active_procs:
            ranked_p = sorted(active_procs,
                              key=lambda p: self.active.get(p, 0.0), reverse=True)[:3]
            L.append("## Procedures (applicable how-to)")
            for pid in ranked_p:
                m = self._procedure_meta[pid]
                L.append(f"- {pid}: {(m['label'] or '')[:90]} "
                         f"(outcome {m['outcome_score']:.2f}, uses {m['uses']})")
                steps = [s for s in (m["steps"] or "").split("\n") if s]
                for s in steps[:5]:
                    L.append(f"    • {s[:100]}")
            L.append("")
        if self._contradictions:
            L.append("## Live tension (competing hypotheses)")
            for a, b, sa, sb in self._contradictions:
                lean = a if sa >= sb else b
                L.append(f"- {a} vs {b} - leaning {lean} ({sa:.2f} vs {sb:.2f})")
            L.append("")
        L.append("## Next concrete action")
        L.append(loops[0]["body"][:140] if loops else "(no open loops - define next objective)")
        L.append("")
        L.append("_Projection of the active subgraph; truth = runtime/events.jsonl_")
        return "\n".join(L) + "\n"

    def render(self) -> str:
        """Export/debug → working_context.md text."""
        return self._render(self._ranked_nodes())


def workspace(query: str = "", top_n: int = 8, reinforce: bool = False,
              agent: str = "", threads=None) -> str:
    """Synthesize the ephemeral attention lens over the active subgraph (L4).
    This REPLACES stored working_context; writing it to disk is debug/export only.

    V4 Phase 1: this is now a thin wrapper — `WorkingMemory.from_graph(...).render()`.
    reinforce=True emits an `attended` event for the top focus nodes (§5.3) so that
    what the workspace surfaces gains a replayable reinforcement floor next build.
    `agent`/`threads` select a role-biased window (§9); the reinforcement is tagged
    with the agent for provenance but its base-activation floor stays global (truth
    about the shared graph, not private to one window)."""
    wm = WorkingMemory.from_graph(query, top_n, agent=agent, threads=threads)
    if reinforce:
        cognitive = [n for n in wm._ranked_nodes() if n["layer"] == "cognitive"]
        if cognitive:
            _reinforce_attended([n["event_id"] for n in cognitive[:REINFORCE_TOP_N]], agent)
    return wm.render()


def _reinforce_attended(event_ids: list, agent: str = "") -> None:
    """Emit one `attended` event per focus node (reinforcement is replayable truth).
    Tag with `agent` for provenance when fired from an agent window. Best-effort:
    never block workspace synthesis on logging."""
    try:
        import runtime
        for eid in event_ids:
            if agent:
                runtime.append("attended", target=eid, agent=agent)
            else:
                runtime.append("attended", target=eid)
    except Exception as exc:
        print(f"  (reinforcement not logged: {exc})", file=sys.stderr)


def _export_threads(summary: dict) -> None:
    """Small JSON projection of thread state (runtime/threads.json)."""
    if not COG_DB.exists():
        return
    conn = sqlite3.connect(COG_DB)
    conn.row_factory = sqlite3.Row
    threads = [dict(r) for r in conn.execute(
        "SELECT * FROM threads ORDER BY last_activity_t DESC")]
    conn.close()
    THREADS_JSON.write_text(json.dumps(
        {"generated": now_iso(), "summary": summary, "threads": threads}, indent=2),
        encoding="utf-8")


# ─── Stage 4: adaptive refresh (Phase 4 autonomy hook driver) ────────────────

def refresh(quiet: bool = False) -> dict:
    """Mid-session adaptive refresh (ARCHITECTURE_V3.md Stage 4 / §6 continuous upkeep).

    Rebuild the graph from the latest events, run topic-drift detection, and when the
    work has drifted off its established topic: emit a replayable `topic_shift`,
    auto-checkpoint (keep the session open), and regenerate the workspace lens so the
    newly-relevant subgraph can be re-injected into context. No drift => no-op.

    Designed to be fired by a UserPromptSubmit hook: on drift its stdout (the fresh
    workspace) is added to the model's context; otherwise it stays silent."""
    if not EVENTS.exists():
        return {"drifted": False}  # uninitialized: silent
    build()
    try:
        import retrieval
        drift = retrieval.detect_drift(emit=True)   # emits topic_shift if drifted
    except Exception as exc:
        return {"drifted": False, "error": str(exc)}

    result = {"drifted": drift.get("drifted", False), "drift": drift.get("drift")}
    if drift.get("drifted"):
        try:
            import runtime
            runtime.checkpoint(reason="auto: topic drift", keep_open=True)
        except Exception as exc:
            result["checkpoint_error"] = str(exc)
        result["workspace"] = workspace()       # fresh lens over the new active subgraph
    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv):
    if not argv:
        print(__doc__); return 0
    cmd, rest = argv[0], argv[1:]

    if cmd == "refresh":
        # --hook: silent unless drift fires (safe to run every prompt). Plain: report.
        hook = "--hook" in rest
        res = refresh()
        if res.get("drifted"):
            if hook:
                print("[continuity] topic drift detected - refreshed working context:\n")
            print(res.get("workspace", ""))
        elif not hook:
            print(f"no drift (score={res.get('drift')}); workspace unchanged")
        return 0

    if cmd == "build":
        br = DEFAULT_BRANCH
        if "--branch" in rest:
            i = rest.index("--branch")
            br = rest[i + 1] if i + 1 < len(rest) else DEFAULT_BRANCH
        s = build(force="--force" in rest, branch=br)
        verb = "up-to-date (skipped rebuild)" if s.get("skipped") else "built"
        vecs = f", {s['vectors']} vectors" if "vectors" in s else ""
        pats = f", {s['patterns']} patterns" if s.get("patterns") else ""
        cold = f", {s['cold']} cold" if s.get("cold") else ""
        tag = "" if br == DEFAULT_BRANCH else f" [branch={br}]"
        print(f"Graph {verb} from {s['events']} events{tag}: {s['nodes']} nodes, "
              f"{s['threads']} threads, {s['edges']} edges{vecs}{pats}{cold} -> "
              f"{COG_DB.relative_to(ROOT)}")
        return 0

    if cmd == "branches":
        for b in branches():
            base = "" if b["base"] == DEFAULT_BRANCH else f" (from {b['base']})"
            print(f"  [{b['status']:<7}] {b['branch']:<20} "
                  f"events={b['events']}{base}")
        return 0

    if cmd == "patterns":
        if not COG_DB.exists():
            build()
        conn = sqlite3.connect(COG_DB); conn.row_factory = sqlite3.Row
        rows = list(conn.execute(
            "SELECT * FROM patterns ORDER BY confidence DESC, frequency DESC"))
        conn.close()
        if not rows:
            print("no patterns yet — run: reflect.py mine --apply  (then build)")
            return 0
        print("recognized patterns (v4.1 §13) — confidence x frequency:")
        for r in rows:
            rec = ", ".join(json.loads(r["recommended"] or "[]"))
            print(f"  [{r['confidence']:.2f} x{r['frequency']:<3}] ({r['kind']}) "
                  f"{r['pattern_id']}  {r['label'][:70]}")
            if rec:
                print(f"          ↳ recommended: {rec[:90]}")
        return 0

    if cmd == "reconstruct":
        if not rest:
            print("usage: reconstruct <branch>"); return 2
        br = rest[0]
        s = reconstruct(br)
        main = reconstruct(DEFAULT_BRANCH)
        print(f"Reconstructed branch '{br}' (isolated, mainline cache untouched):")
        print(f"  {s['events']} events -> {s['nodes']} nodes, {s['threads']} threads, "
              f"{s['edges']} edges, {s['concepts']} concepts")
        d = s["nodes"] - main["nodes"]
        print(f"  vs main: {d:+d} nodes "
              f"({'hypothetical adds context' if d > 0 else 'same as main'})")
        return 0

    if cmd == "audit":
        viol = arbitration_violations()
        if not viol:
            print(f"arbitration audit: clean — all curator-owned events under "
                  f"agent='{CURATOR_AGENT}' (or system).")
            return 0
        print(f"arbitration audit: {len(viol)} curator-owned event(s) emitted by a "
              f"non-curator agent (should PROPOSE, not consolidate):")
        for v in viol:
            print(f"  {v['event_id']:<14} {v['type']:<22} agent={v['agent']}")
        return 0

    if cmd == "snapshot":
        s = snapshot(apply="--apply" in rest)
        if s["applied"]:
            print(f"Snapshot taken at seq {s['snapshot_seq']}: {s['concepts']} concepts, "
                  f"{s['procedures']} procedures retained -> {s['written']}")
        else:
            print(f"  dry-run: snapshot would cover {s['snapshot_seq']} events "
                  f"({s['counts']['nodes']} nodes, {s['concepts']} concepts, "
                  f"{s['procedures']} procedures). Re-run with --apply to record it.")
        return 0

    def _opt(r, name):
        """Pull `--name value` out of arglist r; return (value_or_None, remaining)."""
        if name in r:
            i = r.index(name)
            val = r[i + 1] if i + 1 < len(r) else None
            return val, r[:i] + r[i + 2:]
        return None, r

    def _q_and_n(default_n):
        n = default_n
        r = rest
        if "-n" in r:
            i = r.index("-n"); n = int(r[i + 1]); r = r[:i] + r[i + 2:]
        agent, r = _opt(r, "--agent")
        threads, r = _opt(r, "--threads")
        r = [a for a in r if not a.startswith("--")]
        return " ".join(r), n, (agent or ""), threads

    if cmd == "roles":
        print("agent roles (ARCHITECTURE_V3.md §9) — focus types per role:")
        for role, types in ROLE_PROFILES.items():
            print(f"  {role:<11} {', '.join(sorted(types))}")
        print("\nusage: attend|context|subgraph --agent <role> [--threads t1,t2]")
        return 0

    if cmd == "incubate":
        info = incubate()
        print(f"Incubation pass: lifted {info['incubated']} dormant node(s) into the "
              f"subconscious channel (folded into the next attention seed).")
        return 0

    if cmd == "attend":
        query, n, agent, threads = _q_and_n(10)
        st = attend(query, agent=agent, threads=threads, oscillate="--oscillate" in rest)
        ranked = sorted(st["nodes"].items(), key=lambda kv: kv[1], reverse=True)[:n]
        conn = sqlite3.connect(COG_DB); conn.row_factory = sqlite3.Row
        bodies = {r["event_id"]: r["body"] for r in conn.execute(
            "SELECT event_id,body FROM nodes")}
        conn.close()
        print("top active nodes:")
        for eid, a in ranked:
            print(f"  [{a:.3f}] {eid}  {bodies.get(eid,'')[:90]}")
        print("\ntop threads:")
        for tid, a in sorted(st["threads"].items(), key=lambda k: k[1],
                             reverse=True)[:8]:
            tag = " (resurfaced)" if tid in st["resurfaced"] else ""
            print(f"  [{a:.3f}] {tid}{tag}")
        return 0

    if cmd == "context":
        query, _, agent, threads = _q_and_n(8)
        text = workspace(query, reinforce="--reinforce" in rest,
                         agent=agent, threads=threads)
        print(text)
        if "--write" in rest:
            WORKING_CONTEXT.write_text(text, encoding="utf-8")
            print(f"(written to {WORKING_CONTEXT.relative_to(ROOT)})")
        return 0

    if cmd == "subgraph":
        query, n, agent, threads = _q_and_n(8)
        sg = active_subgraph(query, n, agent=agent, threads=threads,
                             oscillate="--oscillate" in rest)
        for node in sg["nodes"]:
            print(f"  [{node['activation']:.3f}] ({node['layer']}/{node['type']}) "
                  f"{node['event_id']}  {node['body'][:80]}")
        return 0

    if cmd == "threads":
        if not COG_DB.exists():
            build()
        conn = sqlite3.connect(COG_DB); conn.row_factory = sqlite3.Row
        for r in conn.execute(
                "SELECT * FROM threads ORDER BY last_activity_t DESC"):
            into = f" -> {r['merged_into']}" if r["merged_into"] else ""
            print(f"  [{r['status']:<7}] {r['thread_id']:<22} "
                  f"n={r['n_events']:<3} last={r['last_activity_t']}{into}")
        conn.close()
        return 0

    print(f"unknown command: {cmd}"); print(__doc__); return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
