#!/usr/bin/env python3
"""
reflect.py — Persistent Continuity Architecture v3, Phase 5 (reflection + compression).

Two periodic, autonomous-capable passes over the cognitive graph (ARCHITECTURE_V3.md
§7 Reflection, §9 Compression):

  reflect   — meta-cognition: scan threads/loops/contradictions, surface stale branches
              and aging unresolved work, and emit a replayable `reflection` event. The
              reflection itself becomes a high-importance node that resurfaces next build.

  compress  — roll-up projection: write per-thread DIGESTS to threads/ (these REPLACE v2
              session snapshots — the durable unit is the thread, not the session). Cold/
              dormant threads are summarized; originals stay in events.jsonl (truth is
              never destroyed) but the digest stands in for the hot retrieval set.

Both are pure projections of L1 truth via the L2 graph (cognition.db). Reflection appends
ONE event (replayable); compression writes only markdown under threads/.

Subcommands:
    python reflect.py reflect [--quiet]    Run reflection; append a `reflection` event
    python reflect.py compress             (Re)write thread digests under threads/
    python reflect.py digest <thread_id>   Print a single thread's digest to stdout
    python reflect.py all                  reflect + compress

Pure stdlib. Reuses cognition.build (graph) and runtime.append (truth log).
"""

import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import data_root
# Data root is decoupled from this code (see _paths.py): per-project .continuity/
# by default, or $CONTINUITY_HOME. Plugin code itself is shared/read-only.
ROOT = data_root()
RUNTIME = ROOT / "runtime"
COG_DB = RUNTIME / "cognition.db"
THREADS_DIR = ROOT / "threads"

sys.path.insert(0, str(Path(__file__).resolve().parent))

STALE_DAYS = 14.0          # thread untouched longer than this = stale branch (demote)
LOOP_AGING_DAYS = 7.0      # an open loop older than this is "aging" (nudged in reflection)


def now() -> datetime:
    return datetime.now(timezone.utc)


def now_iso() -> str:
    return now().astimezone().replace(microsecond=0).isoformat()


def _age_days(t: str) -> float:
    try:
        dt = datetime.fromisoformat(t)
    except (ValueError, TypeError):
        return 0.0
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (now() - dt).total_seconds() / 86400.0)


def _graph():
    """Ensure the cognitive graph is current, then open it read-only-ish."""
    import cognition
    cognition.build()
    conn = sqlite3.connect(COG_DB)
    conn.row_factory = sqlite3.Row
    return conn


# ─── reflection (§7) ─────────────────────────────────────────────────────────

def reflect(quiet: bool = False) -> dict:
    """Scan the graph for stale branches, aging unresolved work, and live tensions;
    emit a single replayable `reflection` event summarizing the findings."""
    conn = _graph()

    threads = [dict(r) for r in conn.execute("SELECT * FROM threads")]
    stale = [t for t in threads
             if t["status"] != "merged" and _age_days(t["last_activity_t"]) > STALE_DAYS]

    open_loops = [dict(r) for r in conn.execute(
        "SELECT * FROM nodes WHERE type='open_loop' AND unresolved=1")]
    aging = [l for l in open_loops if _age_days(l["t"]) > LOOP_AGING_DAYS]

    # live tensions: contradiction edges still present
    tensions = [(r["src"], r["dst"]) for r in conn.execute(
        "SELECT src,dst FROM edges WHERE kind='contradiction'")]

    counts = {
        "threads": len(threads),
        "active": sum(1 for t in threads if t["status"] == "active"),
        "dormant": sum(1 for t in threads if t["status"] == "dormant"),
        "open_loops": len(open_loops),
        "aging_loops": len(aging),
        "tensions": len(tensions),
    }
    conn.close()

    bits = [f"{counts['threads']} threads ({counts['active']} active, "
            f"{counts['dormant']} dormant)",
            f"{counts['open_loops']} open loops"]
    if aging:
        bits.append(f"{len(aging)} aging (>{LOOP_AGING_DAYS:.0f}d): "
                    + "; ".join(l["body"][:50] for l in aging[:3]))
    if stale:
        bits.append("stale branches: " + ", ".join(t["thread_id"] for t in stale))
    if tensions:
        bits.append(f"{len(tensions)} unresolved tension(s)")
    summary = "Reflection: " + " | ".join(bits)

    # the reflection is itself truth (replayable); it surfaces as a high-importance node
    try:
        import runtime
        runtime.append("reflection", msg=summary,
                       stale_threads=[t["thread_id"] for t in stale])
    except Exception as exc:                                  # pragma: no cover
        print(f"  (reflection not logged: {exc})", file=sys.stderr)

    result = {"summary": summary, "counts": counts,
              "stale": [t["thread_id"] for t in stale],
              "aging_loops": [l["body"] for l in aging]}
    if not quiet:
        print(summary)
    return result


# ─── compression: thread digests (§9; replaces session snapshots) ────────────

def _thread_digest(conn, tid: str) -> str:
    """Render one thread's digest from its member events (a projection, not truth)."""
    th = conn.execute("SELECT * FROM threads WHERE thread_id=?", (tid,)).fetchone()
    members = [dict(r) for r in conn.execute(
        "SELECT n.* FROM nodes n JOIN membership m ON n.event_id=m.event_id "
        "WHERE m.thread_id=? ORDER BY n.seq", (tid,))]

    def of(*types):
        return [m for m in members if m["type"] in types]

    objectives = of("objective")
    decisions = of("decision")
    reflections = of("reflection")
    artifacts = of("artifact")
    loops_open = [m for m in members if m["type"] == "open_loop" and m["unresolved"]]
    loops_done = [m for m in members if m["type"] == "loop_closed"]
    hyps = of("hypothesis", "contradiction")

    title = (th["title"] if th and th["title"] else tid)
    status = th["status"] if th else "active"
    last = th["last_activity_t"] if th else ""

    L = []
    L.append("---")
    L.append(f"thread_id: {tid}")
    L.append(f"title: {title}")
    L.append(f"status: {status}")
    L.append(f"n_events: {len(members)}")
    L.append(f"last_activity: {last}")
    L.append(f"generated: {now_iso()}")
    L.append("---")
    L.append(f"# Thread: {title}")
    L.append("")
    if objectives:
        L.append("## Objective")
        for m in objectives:
            L.append(f"- {m['body']}")
        L.append("")
    if decisions:
        L.append("## Key decisions")
        for m in decisions:
            L.append(f"- ({m['event_id']}) {m['body']}")
        L.append("")
    if hyps:
        L.append("## Hypotheses / tensions")
        for m in hyps:
            L.append(f"- [{m['type']}] {m['body']}")
        L.append("")
    if loops_open or loops_done:
        L.append("## Loops")
        for m in loops_open:
            L.append(f"- [open] {m['body']}")
        for m in loops_done:
            L.append(f"- [closed] {m['body']}")
        L.append("")
    if artifacts:
        L.append("## Artifacts")
        for m in artifacts:
            L.append(f"- {m['body']}")
        L.append("")
    if reflections:
        L.append("## Reflections")
        for m in reflections[-3:]:
            L.append(f"- {m['body'][:160]}")
        L.append("")
    L.append("_Digest: a projection of this thread's events; truth = runtime/events.jsonl_")
    return "\n".join(L) + "\n"


def compress() -> dict:
    """Write/refresh a digest per thread under threads/. Cold (dormant) threads are
    flagged so their digest stands in for the originals in the hot set."""
    conn = _graph()
    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    written, cold = [], []
    for r in conn.execute("SELECT thread_id, status FROM threads ORDER BY thread_id"):
        tid = r["thread_id"]
        digest = _thread_digest(conn, tid)
        path = THREADS_DIR / f"{tid}.md"
        path.write_text(digest, encoding="utf-8")
        written.append(tid)
        if r["status"] in ("dormant", "merged"):
            cold.append(tid)
    conn.close()
    print(f"Wrote {len(written)} thread digest(s) -> {THREADS_DIR.relative_to(ROOT)}/"
          + (f"  (cold/standin: {', '.join(cold)})" if cold else ""))
    return {"written": written, "cold": cold}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv):
    if not argv:
        print(__doc__); return 0
    cmd, rest = argv[0], argv[1:]

    if cmd == "reflect":
        reflect(quiet="--quiet" in rest); return 0
    if cmd == "compress":
        compress(); return 0
    if cmd == "digest":
        if not rest:
            print("usage: reflect.py digest <thread_id>"); return 2
        conn = _graph()
        print(_thread_digest(conn, rest[0]))
        conn.close(); return 0
    if cmd == "all":
        reflect(); compress(); return 0

    print(f"unknown command: {cmd}"); print(__doc__); return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
