#!/usr/bin/env python3
"""
retrieval.py — Persistent Continuity Architecture v2, Phase 3 (L3 retrieval).

Implements the retrieval layer that ARCHITECTURE_V2.md §5/§8 specify, operating on
the v2 truth store (runtime/events.jsonl), NOT the v1 markdown tree:

  - Incremental indexer  : index only events past index/cursor into SQLite FTS5
  - Salience model (§8)   : importance + recency-decay + relevance + unresolved + links
  - BM25 retrieval        : FTS5 candidate generation, blended with salience
  - Relationship graph    : artifact <-> session adjacency (index/*_graph.json)
  - Optional Claude rerank: off by default; real Anthropic-API rerank behind a flag
                            (cloud-first, 2026-05-31 decision) with stdlib fallback
  - Stage 4 drift         : lexical topic-drift detection over recent events

Everything in L3 is a CACHE — fully rebuildable from L1 (events.jsonl). Deleting
index/ and re-running `reindex` reproduces it deterministically.

Subcommands:
    python retrieval.py reindex              Incrementally index new events
    python retrieval.py reindex --full       Drop + rebuild the whole index
    python retrieval.py search <query> [-k N] [--rerank]   Top-K (BM25 x salience)
    python retrieval.py loops                Open loops ranked by salience (JSON)
    python retrieval.py drift [--window N] [--emit]        Stage 4 topic-drift check
    python retrieval.py stats                Index health / cursor position

Pure stdlib by default. Claude rerank uses the `anthropic` SDK only when enabled
(--rerank / CONTINUITY_RERANK=1) and present; otherwise it cleanly no-ops. Dense
embeddings remain a separate pluggable, off-by-default upgrade (not here).
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
OPEN_LOOPS = RUNTIME / "open_loops.json"
INDEX = ROOT / "index"
DB = INDEX / "index.db"
CURSOR = INDEX / "cursor"
ARTIFACT_GRAPH = INDEX / "artifact_graph.json"
TOPIC_GRAPH = INDEX / "topic_graph.json"

HALF_LIFE_DAYS = 7.0  # recency decay half-life (§8, tunable)

# Emit-time importance per event type (§8). Decisions/architecture high; routine low.
IMPORTANCE = {
    "objective": 0.90, "decision": 0.85, "reflection": 0.80, "open_loop": 0.80,
    "error": 0.75, "assumption_invalidated": 0.75, "artifact": 0.60,
    "assumption": 0.60, "loop_closed": 0.50, "topic_shift": 0.40,
    "checkpoint": 0.20, "session_start": 0.20, "session_end": 0.20,
}
DEFAULT_IMPORTANCE = 0.50  # unknown/forward-compatible types

# Salience composite weights (§8). Sum = 1.0.
W_IMPORTANCE = 0.30
W_RECENCY = 0.20
W_RELEVANCE = 0.25
W_UNRESOLVED = 0.15
W_LINKED = 0.10

# Events trivial enough to keep out of the hot retrieval set (salience-gated, §5).
GATE_OUT_TYPES = {"session_start", "session_end", "checkpoint"}

# Claude rerank (cloud-first, off by default). Reranking is how Claude contributes
# meaning since Anthropic has no embeddings endpoint (2026-05-31 decision).
RERANK_MODEL = os.environ.get("CONTINUITY_RERANK_MODEL", "claude-haiku-4-5-20251001")
RERANK_ENV = os.environ.get("CONTINUITY_RERANK", "").lower() in ("1", "true", "yes")
RERANK_CANDIDATES = 12   # cap candidates sent to the model (token budget)

# Stage 4 adaptive refresh: how many trailing events form the "recent" window, and
# the cosine-distance threshold above which we call it topic drift.
DRIFT_WINDOW = 5
DRIFT_THRESHOLD = 0.82   # tunable; lexical cosine is coarse, so require a strong turn
                         # (~complete vocab change) to fire — sub-task shifts stay quiet


# ─── time ──────────────────────────────────────────────────────────────────

def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(s: str) -> datetime:
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return now()
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def recency(t: str) -> float:
    """Exponential decay in [0,1]: 0.5 ** (age_days / half_life)."""
    age_days = (now() - parse_iso(t)).total_seconds() / 86400.0
    age_days = max(0.0, age_days)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


# ─── searchable text / link extraction ──────────────────────────────────────

def event_body(e: dict) -> str:
    """Flatten an event's human-meaningful fields into one searchable string."""
    parts = [e.get("type", "")]
    for k in ("msg", "path", "reason", "resolution", "objective", "branch",
              "project", "session", "id"):
        v = e.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
    # topics / links may be list or comma string
    for k in ("topics", "links"):
        v = e.get(k)
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif isinstance(v, str) and v:
            parts.append(v)
    return " ".join(parts)


def event_links(e: dict) -> list:
    """Normalize an event's `links` field to a list of 'kind:target' strings."""
    v = e.get("links")
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str) and v:
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


def event_topics(e: dict) -> list:
    v = e.get("topics")
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str) and v:
        return [s.strip() for s in v.split(",") if s.strip()]
    return []


# ─── L1 read (truth) ─────────────────────────────────────────────────────────

def read_events() -> list:
    if not EVENTS.exists():
        return []
    out = []
    for raw in EVENTS.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            print("  WARN: skipping malformed event during index", file=sys.stderr)
    return out


# ─── DB ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq        INTEGER PRIMARY KEY,   -- 1-based position in events.jsonl
    t          TEXT,
    type       TEXT,
    session    TEXT,
    eid        TEXT,                  -- event-level id (e.g. open_loop id)
    title      TEXT,                  -- clean human label (msg/path/reason)
    body       TEXT,
    importance REAL,
    unresolved INTEGER DEFAULT 0,     -- open_loop not yet closed
    linked     INTEGER DEFAULT 0      -- graph degree (filled by graph build)
);
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
    USING fts5(body, content='events', content_rowid='seq');
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS links (src TEXT, dst TEXT, kind TEXT);
"""


def connect() -> sqlite3.Connection:
    INDEX.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def get_cursor(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key='cursor'").fetchone()
    return int(row["value"]) if row else 0


def set_cursor(conn: sqlite3.Connection, n: int) -> None:
    conn.execute("INSERT INTO meta(key,value) VALUES('cursor',?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(n),))
    CURSOR.write_text(str(n), encoding="utf-8")  # human-inspectable mirror (§5)


# ─── incremental index ───────────────────────────────────────────────────────

def reindex(full: bool = False) -> dict:
    """Index events past the cursor. `full` drops and rebuilds everything."""
    conn = connect()
    if full:
        conn.executescript("DROP TABLE IF EXISTS events; DROP TABLE IF EXISTS events_fts;"
                           " DROP TABLE IF EXISTS links; DELETE FROM meta;")
        conn.executescript(SCHEMA)

    events = read_events()
    cursor = get_cursor(conn)
    new = events[cursor:]  # incremental: only unseen events (§3 Stage 0)

    # Track open-loop resolution across the whole stream so `unresolved` is correct
    # even when loop_closed arrives in a later batch.
    closed = {e.get("id") for e in events if e.get("type") == "loop_closed"}

    inserted = 0
    for i, e in enumerate(new, start=cursor + 1):
        etype = e.get("type", "")
        eid = e.get("id")
        unresolved = 1 if (etype == "open_loop" and eid not in closed) else 0
        body = event_body(e)
        title = e.get("msg") or e.get("path") or e.get("reason") or etype
        conn.execute(
            "INSERT OR REPLACE INTO events"
            "(seq,t,type,session,eid,title,body,importance,unresolved,linked)"
            " VALUES (?,?,?,?,?,?,?,?,?,0)",
            (i, e.get("t"), etype, e.get("session"), eid, title, body,
             IMPORTANCE.get(etype, DEFAULT_IMPORTANCE), unresolved),
        )
        conn.execute("INSERT INTO events_fts(rowid, body) VALUES (?,?)", (i, body))
        inserted += 1

    # A late loop_closed must clear `unresolved` on its open_loop row (idempotent).
    if closed:
        conn.execute(
            "UPDATE events SET unresolved=0 WHERE type='open_loop' AND eid IN (%s)"
            % ",".join("?" * len(closed)), tuple(closed))

    set_cursor(conn, len(events))
    conn.commit()
    build_graphs(conn, events)
    conn.commit()
    conn.close()
    return {"total": len(events), "new": inserted, "cursor": len(events)}


# ─── relationship graph (§5) ─────────────────────────────────────────────────

def build_graphs(conn: sqlite3.Connection, events: list) -> None:
    """Rebuild artifact<->session and topic<->topic adjacency, and fill `linked`
    degree per event. Cheap full rebuild — the graphs are small and derived."""
    conn.execute("DELETE FROM links")

    artifact_sessions: dict = {}   # path -> set(session)
    topic_cooc: dict = {}          # topic -> {topic: weight}
    session_of: dict = {}          # running session id

    cur_session = None
    for e in events:
        if e.get("type") == "session_start":
            cur_session = e.get("session")
        s = e.get("session") or cur_session

        if e.get("type") == "artifact" and e.get("path"):
            path = e["path"]
            artifact_sessions.setdefault(path, set())
            if s:
                artifact_sessions[path].add(s)
            conn.execute("INSERT INTO links(src,dst,kind) VALUES (?,?,?)",
                         (s or "?", path, "session->artifact"))

        for ln in event_links(e):
            conn.execute("INSERT INTO links(src,dst,kind) VALUES (?,?,?)",
                         (s or "?", ln, "explicit"))

        topics = event_topics(e)
        for a in topics:
            for b in topics:
                if a != b:
                    topic_cooc.setdefault(a, {})
                    topic_cooc[a][b] = topic_cooc[a].get(b, 0) + 1

    artifact_graph = {p: sorted(ss) for p, ss in artifact_sessions.items()}
    ARTIFACT_GRAPH.write_text(json.dumps(artifact_graph, indent=2), encoding="utf-8")
    TOPIC_GRAPH.write_text(json.dumps(topic_cooc, indent=2), encoding="utf-8")

    # linked degree: artifacts get the count of sessions touching them; sessions
    # get the number of distinct artifacts they touched. A cheap centrality proxy.
    sess_artifacts: dict = {}
    for p, ss in artifact_sessions.items():
        for s in ss:
            sess_artifacts.setdefault(s, set()).add(p)

    for p, ss in artifact_sessions.items():
        conn.execute("UPDATE events SET linked=? WHERE type='artifact' AND body LIKE ?",
                     (len(ss), f"%{p}%"))
    for s, arts in sess_artifacts.items():
        conn.execute("UPDATE events SET linked=? WHERE session=?", (len(arts), s))


# ─── salience (§8) ───────────────────────────────────────────────────────────

def _norm_linked(linked: int) -> float:
    """Squash graph degree into [0,1] with diminishing returns."""
    return 1.0 - math.exp(-linked / 5.0)


def salience(row, relevance: float = 0.0) -> float:
    """Composite retrieval-priority score (§8). relevance is query-dependent;
    pass 0.0 for intent-independent base salience (e.g. open-loop ordering)."""
    return round(
        W_IMPORTANCE * (row["importance"] or DEFAULT_IMPORTANCE)
        + W_RECENCY * recency(row["t"])
        + W_RELEVANCE * relevance
        + W_UNRESOLVED * (1.0 if row["unresolved"] else 0.0)
        + W_LINKED * _norm_linked(row["linked"] or 0),
        4,
    )


# ─── retrieval ───────────────────────────────────────────────────────────────

def _fts_query(query: str) -> str:
    """Make a forgiving FTS5 MATCH expression: OR the terms, prefix-match each.
    Avoids syntax errors from raw user punctuation."""
    terms = [t for t in "".join(c if c.isalnum() else " " for c in query).split() if t]
    if not terms:
        return ""
    return " OR ".join(f"{t}*" for t in terms)


def search(query: str, k: int = 6, gate: bool = True, rerank: bool = None) -> list:
    """BM25 candidate generation, reranked by salience (relevance from BM25).

    rerank=None defers to CONTINUITY_RERANK env; rerank=True forces the optional
    Claude API rerank pass over the top candidates (cloud-first; no-ops cleanly
    when the SDK/key are absent). Returns up to k result dicts."""
    if rerank is None:
        rerank = RERANK_ENV
    if not DB.exists():
        return []
    conn = connect()
    match = _fts_query(query)
    rows = []
    if match:
        # bm25() is lower = better; pull a wide candidate set then salience-rerank.
        sql = (
            "SELECT e.*, bm25(events_fts) AS rank FROM events_fts "
            "JOIN events e ON e.seq = events_fts.rowid "
            "WHERE events_fts MATCH ? ORDER BY rank LIMIT ?"
        )
        try:
            rows = conn.execute(sql, (match, max(k * 5, 25))).fetchall()
        except sqlite3.OperationalError:
            rows = []

    # Normalize bm25 -> relevance in [0,1] (best candidate = 1.0).
    results = []
    if rows:
        ranks = [r["rank"] for r in rows]
        lo, hi = min(ranks), max(ranks)
        span = (hi - lo) or 1.0
        for r in rows:
            if gate and r["type"] in GATE_OUT_TYPES:
                continue
            relevance = 1.0 - (r["rank"] - lo) / span  # lower bm25 -> higher relevance
            results.append({
                "seq": r["seq"], "type": r["type"], "t": r["t"],
                "session": r["session"], "title": r["title"], "body": r["body"],
                "importance": r["importance"], "unresolved": r["unresolved"],
                "linked": r["linked"], "bm25": round(r["rank"], 3),
                "relevance": round(relevance, 4),
                "reranked": False, "salience": salience(r, relevance),
            })
    conn.close()
    results.sort(key=lambda x: x["salience"], reverse=True)

    if rerank and results:
        # Rerank only the strongest candidates; tail keeps its BM25 ordering.
        head = claude_rerank(query, results[:RERANK_CANDIDATES])
        results = head + results[RERANK_CANDIDATES:]
        results.sort(key=lambda x: x["salience"], reverse=True)
    return results[:k]


def loops_ranked() -> list:
    """Open loops ranked by intent-independent salience (relevance=0).
    This is what restore() consumes to order open_loops.json."""
    if not DB.exists():
        return []
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM events WHERE type='open_loop' AND unresolved=1"
    ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["eid"], "msg": r["title"],
            "t": r["t"], "resolved": False, "salience": salience(r, 0.0),
        })
    conn.close()
    out.sort(key=lambda x: x["salience"], reverse=True)
    return out


# ─── optional Claude rerank (off by default) ─────────────────────────────────

def claude_rerank(query: str, candidates: list) -> list:
    """Cloud-first semantic rerank (2026-05-31 decision): ask Claude to score the
    BM25 candidates for relevance to `query`. Anthropic has no embeddings endpoint,
    so a scoring call is how Claude contributes *meaning* over keyword matching.

    The model returns a relevance score per candidate in [0,1]; we fold it into
    each candidate's `relevance` and recompute the §8 salience. Degrades cleanly:
    if the SDK or API key is absent, or the call/parse fails, candidates are
    returned unchanged (BM25 x salience order preserved)."""
    if not candidates:
        return candidates
    try:
        import anthropic  # optional dependency; only needed when rerank is enabled
    except ImportError:
        print("  (rerank skipped: `anthropic` SDK not installed - BM25 order kept)",
              file=sys.stderr)
        return candidates
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  (rerank skipped: ANTHROPIC_API_KEY not set - BM25 order kept)",
              file=sys.stderr)
        return candidates

    listing = "\n".join(f"{i}: [{c['type']}] {c['title']}"
                        for i, c in enumerate(candidates))
    prompt = (
        "You are reranking continuity-memory events for relevance to a query.\n"
        f"Query: {query}\n\nCandidates:\n{listing}\n\n"
        "Return ONLY a JSON object mapping each candidate index (as a string) to a "
        "relevance score in [0,1], e.g. {\"0\":0.9,\"1\":0.2}. No prose."
    )
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=RERANK_MODEL, max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        start, end = text.find("{"), text.rfind("}")
        scores = json.loads(text[start:end + 1]) if start != -1 else {}
    except Exception as exc:  # network/auth/parse — rerank is best-effort
        print(f"  (rerank failed: {exc}; BM25 order kept)", file=sys.stderr)
        return candidates

    for i, c in enumerate(candidates):
        s = scores.get(str(i))
        if isinstance(s, (int, float)):
            c["relevance"] = round(max(0.0, min(1.0, float(s))), 4)
            c["reranked"] = True
            c["salience"] = salience(c, c["relevance"])  # dict supports __getitem__
    return candidates


# ─── Stage 4: adaptive refresh / topic-drift detection (§3 Stage 4) ──────────

def _token_bag(texts) -> dict:
    """Frequency bag over tokens, reusing semantic_search.tokenize when available."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from semantic_search import tokenize  # reuse the v1 tokenizer (stopwords)
    except Exception:
        def tokenize(t):  # minimal fallback
            return [w for w in "".join(c if c.isalnum() else " "
                    for c in t.lower()).split() if len(w) > 1]
    bag: dict = {}
    for txt in texts:
        for tok in tokenize(txt or ""):
            bag[tok] = bag.get(tok, 0) + 1
    return bag


def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def detect_drift(window: int = DRIFT_WINDOW, emit: bool = False) -> dict:
    """Compare the recent event window against the session baseline (§3 Stage 4).

    Drift = 1 - cosine(recent_bag, baseline_bag). Above DRIFT_THRESHOLD signals the
    work has moved off its established topic — a cue to re-run retrieval / checkpoint.
    With emit=True, a `topic_shift` event is appended so the drift is itself replayable."""
    events = read_events()
    # restrict baseline to the current (latest) session for a fair comparison
    cur_session = None
    for e in events:
        if e.get("type") == "session_start":
            cur_session = e.get("session")
    sess = [e for e in events if (e.get("session") == cur_session) or not cur_session]
    # Drift must reflect SUBSTANTIVE work, not bookkeeping. Meta/low-content events
    # (reinforcement, checkpoints, session + thread markers) carry little topical text
    # and would otherwise swamp the recent window and trigger false drift.
    drift_meta = {"attended", "checkpoint", "session_start", "session_end",
                  "thread_open", "thread_merge", "thread_split"}
    sess = [e for e in sess if e.get("type") not in drift_meta]
    scope = sess if len(sess) > window else events

    if len(scope) <= window:
        return {"drift": 0.0, "drifted": False, "reason": "not enough events",
                "window": window, "session": cur_session}

    # Drift = a TURN, not accumulated novelty: compare the recent window against the
    # immediately preceding block (up to 2*window events), not all of history. A long
    # focused arc on one topic stays low-drift; a genuine change of direction spikes.
    recent = scope[-window:]
    baseline = scope[-(3 * window):-window] or scope[:-window]
    recent_bag = _token_bag(event_body(e) for e in recent)
    baseline_bag = _token_bag(event_body(e) for e in baseline)
    drift = round(1.0 - _cosine(recent_bag, baseline_bag), 4)
    drifted = drift >= DRIFT_THRESHOLD

    result = {"drift": drift, "drifted": drifted, "threshold": DRIFT_THRESHOLD,
              "window": window, "session": cur_session,
              "recommendation": ("re-run retrieval + checkpoint (topic shifted)"
                                 if drifted else "continue (on topic)")}
    if drifted and emit:
        _append_event("topic_shift", session=cur_session, drift=drift,
                      reason="Stage-4 adaptive refresh detected topic drift")
        result["emitted"] = "topic_shift"
    return result


def _append_event(etype: str, **fields) -> None:
    """Append one event to the L1 truth log (Stage-4 drift markers are replayable).
    Delegates to runtime.append when importable; falls back to a direct JSON line."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import runtime
        runtime.append(etype, **fields)
        return
    except Exception:
        pass
    event = {"t": now().astimezone().replace(microsecond=0).isoformat(), "type": etype}
    event.update(fields)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv):
    if not argv:
        print(__doc__); return 0
    cmd, rest = argv[0], argv[1:]

    if cmd == "reindex":
        info = reindex(full="--full" in rest)
        print(f"Indexed {info['new']} new event(s); cursor={info['cursor']} "
              f"(total {info['total']}). DB: {DB.relative_to(ROOT)}")
        return 0

    if cmd == "search":
        if not rest:
            print("usage: retrieval.py search <query> [-k N] [--rerank]"); return 2
        k = 6
        rerank = "--rerank" in rest
        rest = [a for a in rest if a != "--rerank"]
        if "-k" in rest:
            i = rest.index("-k"); k = int(rest[i + 1]); rest = rest[:i] + rest[i + 2:]
        query = " ".join(rest)
        hits = search(query, k=k, rerank=rerank or None)
        if not hits:
            print("(no results - try `reindex` first)"); return 0
        for h in hits:
            tag = " *reranked" if h.get("reranked") else ""
            print(f"[{h['salience']:.3f}] ({h['type']}) {h['t']}  "
                  f"rel={h['relevance']:.2f}{tag}")
            print(f"    {h['title'][:140]}")
        return 0

    if cmd == "loops":
        print(json.dumps(loops_ranked(), indent=2, ensure_ascii=False)); return 0

    if cmd == "drift":
        window = DRIFT_WINDOW
        if "--window" in rest:
            window = int(rest[rest.index("--window") + 1])
        result = detect_drift(window=window, emit="--emit" in rest)
        print(json.dumps(result, indent=2, ensure_ascii=False)); return 0

    if cmd == "stats":
        if not DB.exists():
            print("no index yet — run `reindex`"); return 0
        conn = connect()
        n = conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        loops = conn.execute("SELECT COUNT(*) c FROM events WHERE type='open_loop' "
                             "AND unresolved=1").fetchone()["c"]
        links = conn.execute("SELECT COUNT(*) c FROM links").fetchone()["c"]
        print(f"indexed events : {n}")
        print(f"cursor         : {get_cursor(conn)}")
        print(f"open loops     : {loops}")
        print(f"graph links    : {links}")
        conn.close()
        return 0

    print(f"unknown command: {cmd}"); print(__doc__); return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
