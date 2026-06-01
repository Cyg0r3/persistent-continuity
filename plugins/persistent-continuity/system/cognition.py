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
    python cognition.py attend [query] [-n N]  Compute attention; show top active nodes/threads
    python cognition.py context [query] [--write]  Synthesize the dynamic workspace lens
    python cognition.py threads                List threads with state (active/dormant/merged)
    python cognition.py subgraph [query] [-n N]    Show the extracted active subgraph
    python cognition.py roles                  List multi-agent roles and their focus types
    python cognition.py refresh [--hook]       Stage-4 adaptive refresh: on topic drift,
                                               checkpoint + re-inject fresh workspace

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
               "contradiction": 0.0}  # contradiction handled by inhibition, not spread
CONCEPT_SEED = 0.55       # v4 Phase 2: in-scope concepts seed attention (scaled by salience)

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
    reinforced INTEGER DEFAULT 0
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
CREATE INDEX ix_mem_thread ON membership(thread_id);
CREATE INDEX ix_edges_dst ON edges(dst);
CREATE INDEX ix_edges_src ON edges(src);
CREATE INDEX ix_cev_concept ON concept_evidence(concept_id);
"""

# Bump when the cache SCHEMA or its projection logic changes so an older cognition.db
# never silently satisfies the watermark skip (forces one rebuild). Phase 2 added the
# concepts/concept_evidence tables and the `concept` edge.
SCHEMA_VERSION = 2


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


def build(db_path: Path = COG_DB, force: bool = False) -> dict:
    """Rebuild the cognitive graph from zero (drop-and-rebuild = pure projection).

    v4 Phase 0 — watermark skip: the graph is a pure function of the event log, so if
    the log hasn't grown since the last build the existing cognition.db is already a
    correct projection and we skip the O(N) rebuild. `force=True` always rebuilds
    (full-rebuild fallback for integrity / schema changes). Correctness still rests on
    the log; the watermark only avoids redundant work."""
    events = read_events()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if not force and _build_watermark(db_path) == len(events):
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

    # watermark: record the event count + schema version this build is a projection of.
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('event_count',?)",
                 (str(len(events)),))
    conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                 (str(SCHEMA_VERSION),))
    conn.commit()
    summary = {
        "events": len(events),
        "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "threads": len(threads),
        "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "concepts": conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0],
        "skipped": False,
    }
    conn.close()
    _export_threads(summary)
    return summary


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
    projection — absent concept_formed events ⇒ empty semantic memory (current behavior)."""
    acc: dict = {}   # concept_id -> {summary, salience, support, evidence:set, formed_t}
    for e in events:
        if e.get("type") != "concept_formed":
            continue
        cid = _concept_id_of(e)
        if not cid:
            continue
        ev = _as_list(e.get("evidence"))
        c = acc.setdefault(cid, {"summary": "", "salience": 0.0, "support": 0,
                                 "evidence": set(), "formed_t": e.get("t")})
        c["summary"] = e.get("summary") or e.get("msg") or c["summary"] or cid
        c["evidence"].update(ev)
        # support: explicit, else cumulative distinct evidence count; reinforces over time.
        c["support"] = int(e.get("support", max(len(c["evidence"]), c["support"] + 1)))
        if e.get("salience") is not None:
            c["salience"] = float(e["salience"])
    for cid, c in acc.items():
        sal = c["salience"] or round(min(0.9, 0.4 + 0.1 * c["support"]), 4)
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
    # query relevance: dense (if enabled) -> BM25 -> lexical LIKE. Graceful 3-tier chain.
    if query:
        dense = []
        if _EMBEDDER.available():
            rows = conn.execute("SELECT event_id, body FROM nodes").fetchall()
            dense = _EMBEDDER.rank(query, [r[0] for r in rows], [r[1] or "" for r in rows])
        if dense:
            for eid, sim in dense:
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
    return seed


def _eid_for_seq(conn, seq):
    r = conn.execute("SELECT event_id FROM nodes WHERE seq=?", (seq,)).fetchone()
    return r[0] if r else None


# ─── optional dense embedder (off by default; pluggable Embedder interface) ──

class _Embedder:
    """Pluggable dense embedder for semantic seeding (ARCHITECTURE_V3.md §5).

    OFF by default — enabled only when CONTINUITY_DENSE is truthy AND
    sentence-transformers is installed. Anthropic has no embeddings endpoint, so
    dense is a local-model upgrade; when absent, attention seeding falls back to
    BM25 (then lexical LIKE). This keeps the stdlib-default invariant intact."""

    def __init__(self):
        self.model = None
        if os.environ.get("CONTINUITY_DENSE", "").lower() in ("1", "true", "yes"):
            try:
                from sentence_transformers import SentenceTransformer
                name = os.environ.get("CONTINUITY_EMBED_MODEL", "all-MiniLM-L6-v2")
                self.model = SentenceTransformer(name)
            except Exception:
                self.model = None  # not installed / load failed -> graceful BM25 fallback

    def available(self) -> bool:
        return self.model is not None

    def rank(self, query: str, ids: list, texts: list) -> list:
        """Return [(event_id, similarity in [0,1])] for the top matches, or []."""
        if not self.available() or not ids:
            return []
        try:
            import numpy as np
            vecs = self.model.encode([query] + texts, normalize_embeddings=True)
            q, mat = vecs[0], np.array(vecs[1:])
            sims = mat @ q  # cosine (vectors are normalized)
            order = sims.argsort()[::-1][:8]
            return [(ids[i], float((sims[i] + 1) / 2)) for i in order]
        except Exception:
            return []


_EMBEDDER = _Embedder()  # singleton; cheap when disabled


def attend(query: str = "", persist: bool = True, agent: str = "", threads=None) -> dict:
    """Compute the attention state via spreading activation over the graph.

    `agent` (+ optional `threads` assignment) selects a role-biased attention window
    over the ONE shared graph (§9): the seed is constrained, the spread is not.
    Attention persistence is scoped per agent so concurrent windows never bleed.
    `agent=''` is the single-agent path (unchanged).

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

def active_subgraph(query: str = "", top_n: int = 8, agent: str = "", threads=None) -> dict:
    """Top-activation nodes expanded <=HOP_LIMIT hops = the active subgraph.

    `agent`/`threads` select a role-biased window (§9); defaults reproduce the
    single-agent subgraph exactly."""
    state = attend(query, agent=agent, threads=threads)
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


class WorkingMemory:
    """In-process attention lens over the active subgraph (L4).

    V3 synthesized this lens fresh every prompt and treated working_context.md as the
    artifact. V4 Phase 1 promotes it to a first-class runtime object: it is built once
    from the graph, then `mutate()`d incrementally as new events land (no full
    re-synthesis), `reprioritize()`d within a fixed budget, and `render()`ed to markdown
    only for export/debug. `concepts`/`procedures` are reserved for Phases 2/3 and stay
    empty here (absent ⇒ current behavior, per the backward-compat invariant)."""

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
        conn.close()
        for n in sg["nodes"]:
            wm._nodes[n["event_id"]] = n
            wm.active[n["event_id"]] = n["activation"]
            # in-scope semantic memory: active concept nodes feed WorkingMemory.concepts
            if n["layer"] == "semantic" and n["event_id"] in wm._concept_meta:
                wm.concepts.append(n["event_id"])
        wm.threads = dict(sg["threads"])
        wm._resurfaced = set(sg["resurfaced"])
        wm._contradictions = list(sg["contradictions"])
        return wm

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
        s = build(force="--force" in rest)
        verb = "up-to-date (skipped rebuild)" if s.get("skipped") else "built"
        print(f"Graph {verb} from {s['events']} events: {s['nodes']} nodes, "
              f"{s['threads']} threads, {s['edges']} edges -> "
              f"{COG_DB.relative_to(ROOT)}")
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

    if cmd == "attend":
        query, n, agent, threads = _q_and_n(10)
        st = attend(query, agent=agent, threads=threads)
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
        sg = active_subgraph(query, n, agent=agent, threads=threads)
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
