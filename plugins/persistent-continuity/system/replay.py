#!/usr/bin/env python3
"""
replay.py — Persistent Continuity Architecture v2, Phase 1 (L2 derived state).

Materializes the L1 truth (runtime/events.jsonl) into the L2 relational layer
(runtime/state.db) that ARCHITECTURE_V2.md §3 mandates: fast, queryable views of
sessions / decisions / artifacts / loops. This is the layer the architecture's
replay-determinism invariant (§"Determinism") is about — given the same event log,
`replay()` rebuilds the same state.db from zero, every time.

L2 is a CACHE. It is dropped and rebuilt on every run; truth is never written here.
Distinct from the Phase 3 FTS index (index/index.db, L3 retrieval). Reuses
runtime.read_events so the event-reading contract has exactly one implementation.

Subcommands:
    python replay.py                       Rebuild state.db from events.jsonl
    python replay.py rebuild               (same; explicit)
    python replay.py query <table> [-n N]  Dump rows from sessions|decisions|
                                           artifacts|loops (newest first)
    python replay.py verify                Rebuild twice; assert byte-identical
                                           (proves replay determinism)
    python replay.py stats                 Row counts + build metadata

Pure stdlib. Optional: salience on loops is filled from retrieval if its index
exists; otherwise left NULL (L2 does not depend on L3).
"""

import sys
import json
import hashlib
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
STATE_DB = RUNTIME / "state.db"

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _read_events() -> list:
    """Read events via runtime (single source of the reading contract); fall back
    to a local reader if runtime is unavailable."""
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
                    print("  WARN: skipping malformed event", file=sys.stderr)
        return out


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


SCHEMA = """
CREATE TABLE sessions (
    session  TEXT PRIMARY KEY,
    project  TEXT, branch TEXT,
    started  TEXT, ended TEXT, status TEXT,
    n_events INTEGER DEFAULT 0
);
CREATE TABLE decisions (
    seq INTEGER PRIMARY KEY, t TEXT, session TEXT, msg TEXT, links TEXT
);
CREATE TABLE artifacts (
    seq INTEGER PRIMARY KEY, t TEXT, session TEXT, path TEXT, action TEXT
);
CREATE TABLE loops (
    id TEXT PRIMARY KEY, opened_t TEXT, session TEXT, msg TEXT,
    resolved INTEGER DEFAULT 0, resolved_t TEXT, resolution TEXT, salience REAL
);
CREATE TABLE replay_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX ix_dec_session ON decisions(session);
CREATE INDEX ix_art_session ON artifacts(session);
CREATE INDEX ix_loops_resolved ON loops(resolved);
"""


def _links_str(e: dict) -> str:
    v = e.get("links")
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return v if isinstance(v, str) else ""


def _loop_salience() -> dict:
    """Optional: {loop_id: salience} from the L3 retrieval index, if present.
    L2 must not depend on L3, so any failure yields an empty map (NULL salience)."""
    try:
        import retrieval
        if not retrieval.DB.exists():
            return {}
        return {l["id"]: l["salience"] for l in retrieval.loops_ranked()}
    except Exception:
        return {}


def replay(db_path: Path = STATE_DB) -> dict:
    """Rebuild state.db from zero. Drop-and-rebuild is the point: it guarantees the
    DB is a pure function of events.jsonl with no carried-over residue."""
    events = _read_events()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # rebuild from zero (replay determinism)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    cur_session = None
    sessions: dict = {}        # session -> row dict
    salience = _loop_salience()

    for i, e in enumerate(events, 1):
        t = e.get("type")
        if t == "session_start":
            cur_session = e.get("session")
            sessions.setdefault(cur_session, {
                "session": cur_session, "project": e.get("project"),
                "branch": e.get("branch"), "started": e.get("t"),
                "ended": None, "status": "active", "n_events": 0,
            })
        sess = e.get("session") or cur_session
        if sess and sess in sessions:
            sessions[sess]["n_events"] += 1

        if t == "session_end":
            row = sessions.get(e.get("session") or cur_session)
            if row:
                row["ended"] = e.get("t")
                row["status"] = e.get("status", "paused")
        elif t == "decision":
            conn.execute("INSERT INTO decisions(seq,t,session,msg,links) "
                         "VALUES (?,?,?,?,?)",
                         (i, e.get("t"), sess, e.get("msg"), _links_str(e)))
        elif t == "artifact":
            conn.execute("INSERT INTO artifacts(seq,t,session,path,action) "
                         "VALUES (?,?,?,?,?)",
                         (i, e.get("t"), sess, e.get("path"),
                          e.get("action", "touched")))
        elif t == "open_loop":
            lid = e.get("id") or f"loop-{i}"
            conn.execute(
                "INSERT OR REPLACE INTO loops"
                "(id,opened_t,session,msg,resolved,salience) VALUES (?,?,?,?,0,?)",
                (lid, e.get("t"), sess, e.get("msg"), salience.get(lid)))
        elif t == "loop_closed":
            conn.execute("UPDATE loops SET resolved=1, resolved_t=?, resolution=? "
                         "WHERE id=?",
                         (e.get("t"), e.get("resolution"), e.get("id")))

    for row in sessions.values():
        conn.execute(
            "INSERT OR REPLACE INTO sessions"
            "(session,project,branch,started,ended,status,n_events) "
            "VALUES (?,?,?,?,?,?,?)",
            (row["session"], row["project"], row["branch"], row["started"],
             row["ended"], row["status"], row["n_events"]))

    meta = {
        "event_count": str(len(events)),
        "built_at": now_iso(),
        "source_hash": _events_hash(events),
        "last_event_t": events[-1]["t"] if events else "",
    }
    for k, v in meta.items():
        conn.execute("INSERT OR REPLACE INTO replay_meta(key,value) VALUES (?,?)",
                     (k, v))
    conn.commit()
    counts = _counts(conn)
    conn.close()
    return {"events": len(events), **counts, "db": str(db_path.relative_to(ROOT))}


def _events_hash(events: list) -> str:
    """Content hash of the event stream — the determinism anchor (independent of
    build_at wall-clock)."""
    blob = "\n".join(json.dumps(e, sort_keys=True, ensure_ascii=False)
                     for e in events)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _counts(conn: sqlite3.Connection) -> dict:
    out = {}
    for tbl in ("sessions", "decisions", "artifacts", "loops"):
        out[tbl] = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    out["open_loops"] = conn.execute(
        "SELECT COUNT(*) FROM loops WHERE resolved=0").fetchone()[0]
    return out


# ─── determinism check ───────────────────────────────────────────────────────

def _content_digest(db_path: Path) -> str:
    """Hash the logical contents of state.db (excludes the wall-clock built_at, so
    two replays of the same events compare equal)."""
    conn = sqlite3.connect(db_path)
    parts = []
    for tbl in ("sessions", "decisions", "artifacts", "loops"):
        for row in conn.execute(f"SELECT * FROM {tbl} ORDER BY 1"):
            parts.append(repr(row))
    parts.append(conn.execute(
        "SELECT value FROM replay_meta WHERE key='source_hash'").fetchone()[0])
    conn.close()
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def verify() -> bool:
    """Rebuild twice and confirm the logical content is identical — concretely
    demonstrates the replay-determinism invariant (§Determinism)."""
    replay()
    d1 = _content_digest(STATE_DB)
    replay()
    d2 = _content_digest(STATE_DB)
    ok = d1 == d2
    print(f"replay determinism: {'PASS' if ok else 'FAIL'}")
    print(f"  digest1={d1[:16]}  digest2={d2[:16]}")
    return ok


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv):
    if not argv or argv[0] == "rebuild":
        info = replay()
        print(f"Replayed {info['events']} events -> {info['db']}")
        print(f"  sessions={info['sessions']} decisions={info['decisions']} "
              f"artifacts={info['artifacts']} loops={info['loops']} "
              f"(open={info['open_loops']})")
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "verify":
        return 0 if verify() else 1

    if cmd == "query":
        if not rest or rest[0] not in ("sessions", "decisions", "artifacts", "loops"):
            print("usage: replay.py query sessions|decisions|artifacts|loops [-n N]")
            return 2
        tbl = rest[0]
        n = 10
        if "-n" in rest:
            n = int(rest[rest.index("-n") + 1])
        if not STATE_DB.exists():
            replay()
        conn = sqlite3.connect(STATE_DB)
        conn.row_factory = sqlite3.Row
        order = "started" if tbl == "sessions" else (
            "opened_t" if tbl == "loops" else "t")
        rows = conn.execute(
            f"SELECT * FROM {tbl} ORDER BY {order} DESC LIMIT ?", (n,)).fetchall()
        for r in rows:
            print(json.dumps({k: r[k] for k in r.keys()}, ensure_ascii=False))
        conn.close()
        return 0

    if cmd == "stats":
        if not STATE_DB.exists():
            print("no state.db yet - run `replay.py rebuild`"); return 0
        conn = sqlite3.connect(STATE_DB)
        c = _counts(conn)
        meta = dict(conn.execute("SELECT key,value FROM replay_meta").fetchall())
        conn.close()
        print(f"sessions   : {c['sessions']}")
        print(f"decisions  : {c['decisions']}")
        print(f"artifacts  : {c['artifacts']}")
        print(f"loops      : {c['loops']} (open {c['open_loops']})")
        print(f"events     : {meta.get('event_count')}")
        print(f"built_at   : {meta.get('built_at')}")
        print(f"source_hash: {meta.get('source_hash', '')[:16]}")
        return 0

    print(f"unknown command: {cmd}"); print(__doc__); return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
