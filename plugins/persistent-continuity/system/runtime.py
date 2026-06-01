#!/usr/bin/env python3
"""
runtime.py — Persistent Continuity Architecture v2, MVP cognitive runtime.

Implements the core inversion from ARCHITECTURE_V2.md in pure stdlib:
  events.jsonl is TRUTH (append-only) -> everything else is a derived projection.

Consolidates the three documented MVP responsibilities into one module:
  - eventlog   : append + validate + read the event stream
  - restore    : derive state from events -> regenerate runtime/working_context.md
  - checkpoint : autonomous snapshot (append checkpoint/session_end + refresh context)

Subcommands:
    python runtime.py append <type> [key=value ...]   Append one event
    python runtime.py restore                          Rebuild working_context.md
    python runtime.py checkpoint [--reason TEXT]       Checkpoint + refresh
    python runtime.py session-start [project=..]       Begin a session
    python runtime.py state                            Print derived state (JSON)
    python runtime.py tail [N]                         Show last N events
    python runtime.py demo                             Run the validation demo loop

Salience for the MVP is recency + unresolved only (no embeddings, no deps).
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import data_root
# Data root is decoupled from this code (see _paths.py): per-project .continuity/
# by default, or $CONTINUITY_HOME. Plugin code itself is shared/read-only.
ROOT = data_root()
RUNTIME = ROOT / "runtime"
EVENTS = RUNTIME / "events.jsonl"
SEQ_FILE = RUNTIME / "seq"          # persisted event counter (v4 Phase 0): kills the
                                    # O(N)-per-append log scan that made id assignment O(N²)
WORKING_CONTEXT = RUNTIME / "working_context.md"
ACTIVE_CONTEXT = RUNTIME / "active_context.json"
SESSION_STATE = RUNTIME / "session_state.json"
OPEN_LOOPS = RUNTIME / "open_loops.json"

# Code lives in the plugin (this dir's parent); templates ship alongside it. Distinct
# from ROOT, which is the per-project DATA root resolved by _paths.data_root().
PKG_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = PKG_ROOT / "templates"
IDENTITY_FILES = [
    "identity/PROJECT_VISION.md", "identity/NON_NEGOTIABLES.md",
    "identity/SYSTEM_PRINCIPLES.md", "procedural/USER_PROFILE.md",
]

# Event types we understand. Unknown types are stored and ignored by views
# that don't model them (forward-compatible per ARCHITECTURE_V2.md §3).
KNOWN_TYPES = {
    "session_start", "session_end", "objective", "decision", "artifact",
    "open_loop", "loop_closed", "checkpoint", "topic_shift", "reflection",
    "error", "assumption", "assumption_invalidated",
    # v3 cognition-native additions (ARCHITECTURE_V3.md)
    "observation", "hypothesis", "contradiction", "output", "api_call",
    "command", "thread_open", "thread_merge", "thread_split", "attended",
}

# v3: cognitive vs execution vs meta layer, inferred from type when not given (§4).
COGNITIVE_TYPES = {
    "objective", "decision", "observation", "reflection", "hypothesis",
    "contradiction", "assumption", "assumption_invalidated", "open_loop",
    "loop_closed", "topic_shift",
}
EXECUTION_TYPES = {"artifact", "output", "api_call", "error", "command"}


def infer_layer(event_type: str) -> str:
    if event_type in COGNITIVE_TYPES:
        return "cognitive"
    if event_type in EXECUTION_TYPES:
        return "execution"
    return "meta"

WORKING_CONTEXT_BUDGET_CHARS = 6000  # ~1,500 token guardrail


# ─── time ──────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


def parse_iso(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # tolerate trailing 'Z' or missing tz
        s = s.rstrip("Z")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ─── L1: event log (truth) ───────────────────────────────────────────────────

def _count_events() -> int:
    """Count events by streaming the log (no full parse). Used only to (re)seed the
    persisted counter — the slow path we are replacing for steady-state appends."""
    if not EVENTS.exists():
        return 0
    n = 0
    with EVENTS.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                n += 1
    return n


def _read_seq() -> int:
    """Current persisted event count, or None if absent/unreadable (caller reseeds)."""
    try:
        return int(SEQ_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def next_seq() -> int:
    """Allocate the next 1-based event sequence number in O(1) (v4 Phase 0).

    Truth is still the log; `seq` is a rebuildable cache. If it is missing or has
    drifted below the actual line count (e.g. the log was edited out-of-band), it
    self-heals by re-counting — so correctness never depends on the cache being right,
    only its speed does."""
    cur = _read_seq()
    if cur is None or cur < _count_events():
        cur = _count_events()
    nxt = cur + 1
    SEQ_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEQ_FILE.write_text(str(nxt), encoding="utf-8")
    return nxt


def append(event_type: str, **fields) -> dict:
    """Validate and append one event as a JSON line. Never edits existing lines.

    v3: assigns a stable `event_id` (evt_<seq>) and infers `layer` from type when
    not supplied; `thread_ids`/`causes` accept a list or comma-string. All v3 fields
    are optional — omitting them yields exactly a v2 event (backward compatible).
    v4 Phase 0: the `event_id` sequence comes from a persisted O(1) counter
    (`runtime/seq`) instead of re-reading the whole log on every append."""
    if not event_type or not isinstance(event_type, str):
        raise ValueError("event requires a non-empty string 'type'")
    event = {"t": fields.pop("t", now_iso()), "type": event_type}

    # v3 graph fields (kept out of the dict when empty to stay v2-clean).
    if "event_id" not in fields:
        fields["event_id"] = f"evt_{next_seq()}"
    if "layer" not in fields:
        fields["layer"] = infer_layer(event_type)
    for k in ("thread_ids", "causes"):
        v = fields.get(k)
        if isinstance(v, str):
            fields[k] = [s.strip() for s in v.split(",") if s.strip()]
        elif v in (None, ""):
            fields.pop(k, None)

    event.update(fields)
    # validate it round-trips as a single JSON line before committing
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    json.loads(line)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if event_type not in KNOWN_TYPES:
        print(f"  (note: '{event_type}' is an unknown type — stored, not modeled)")
    return event


def read_events() -> list:
    """Read all events. Skips malformed lines loudly rather than corrupting state."""
    if not EVENTS.exists():
        return []
    events = []
    for i, raw in enumerate(EVENTS.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            print(f"  WARN: skipping malformed event at line {i}", file=sys.stderr)
    return events


# ─── L2: derived state (replayable projection) ───────────────────────────────

def derive_state() -> dict:
    """Fold the event stream into the minimum viable cognitive state."""
    events = read_events()
    state = {
        "project": None, "branch": None, "session": None, "status": "unknown",
        "objective": None, "decisions": [], "artifacts": [],
        "open_loops": {}, "checkpoints": 0, "event_count": len(events),
        "last_event_t": events[-1]["t"] if events else None,
    }
    for e in events:
        t = e.get("type")
        if t == "session_start":
            state["session"] = e.get("session")
            state["status"] = "active"
            if e.get("project"):
                state["project"] = e["project"]
            if e.get("branch"):
                state["branch"] = e["branch"]
        elif t == "session_end":
            state["status"] = e.get("status", "paused")
        elif t == "objective":
            state["objective"] = {"msg": e.get("msg"), "t": e.get("t")}
        elif t == "decision":
            state["decisions"].append({"msg": e.get("msg"), "t": e.get("t"),
                                       "links": e.get("links", [])})
        elif t == "artifact":
            state["artifacts"].append({"path": e.get("path"),
                                       "action": e.get("action", "touched"),
                                       "t": e.get("t")})
        elif t == "open_loop":
            lid = e.get("id") or f"loop-{len(state['open_loops']) + 1}"
            state["open_loops"][lid] = {"id": lid, "msg": e.get("msg"),
                                        "t": e.get("t"), "resolved": False}
        elif t == "loop_closed":
            lid = e.get("id")
            if lid in state["open_loops"]:
                state["open_loops"][lid]["resolved"] = True
                state["open_loops"][lid]["resolution"] = e.get("resolution")
        elif t == "checkpoint":
            state["checkpoints"] += 1
    return state


def open_loops_ranked(state: dict) -> list:
    """Rank unresolved open loops by salience.

    Prefers the Phase 3 retrieval salience model (§8: importance + recency-decay
    + unresolved + linked) when an index exists; falls back to the MVP recency-only
    ranking when retrieval is unavailable (no index, import error). Either way the
    load path reads only derived state — never raw markdown."""
    loops = [l for l in state["open_loops"].values() if not l["resolved"]]

    ranked = _retrieval_loop_salience()
    if ranked is not None:
        by_id = {l["id"]: l["salience"] for l in ranked}
        for l in loops:
            l["salience"] = by_id.get(l["id"], 0.0)
        loops.sort(key=lambda l: l["salience"], reverse=True)
        return loops

    # MVP fallback: recency order, crude salience for display.
    loops.sort(key=lambda l: l.get("t") or "", reverse=True)
    n = len(loops)
    for i, l in enumerate(loops):
        l["salience"] = round(1.0 - (i / (n + 1)), 2) if n else 0.0
    return loops


def _retrieval_loop_salience():
    """Return Phase 3 salience-ranked loops, or None if retrieval is unavailable."""
    try:
        import retrieval  # same dir; L3 cache layer
    except ImportError:
        return None
    try:
        if not retrieval.DB.exists():
            return None
        retrieval.reindex()  # incremental: only new events since last cursor
        return retrieval.loops_ranked()
    except Exception as exc:  # retrieval is a cache; never block restore on it
        print(f"  (retrieval salience unavailable: {exc}; using recency fallback)",
              file=sys.stderr)
        return None


# ─── L4: runtime projections ─────────────────────────────────────────────────

def write_runtime_projections(state: dict, loops: list) -> None:
    ACTIVE_CONTEXT.write_text(json.dumps({
        "project": state["project"], "branch": state["branch"],
        "session": state["session"], "status": state["status"],
        "objective": (state["objective"] or {}).get("msg"),
        "regenerated": now_iso(),
    }, indent=2), encoding="utf-8")
    SESSION_STATE.write_text(json.dumps({
        "session": state["session"], "status": state["status"],
        "last_event_t": state["last_event_t"], "event_count": state["event_count"],
        "checkpoints": state["checkpoints"], "open_loop_count": len(loops),
    }, indent=2), encoding="utf-8")
    OPEN_LOOPS.write_text(json.dumps(loops, indent=2), encoding="utf-8")


def render_working_context(state: dict, loops: list) -> str:
    obj = (state["objective"] or {}).get("msg") or "(no objective recorded)"
    last_decision = state["decisions"][-1] if state["decisions"] else None
    last_artifact = state["artifacts"][-1] if state["artifacts"] else None
    next_action = loops[0]["msg"] if loops else "(no open loops — define next objective)"

    lines = []
    lines.append(f"# Working Context - regenerated {now_iso()}")
    lines.append("")
    lines.append("## You are continuing")
    lines.append(
        f"Project: {state['project'] or 'unknown'} | "
        f"Branch: {state['branch'] or '-'} | "
        f"Last status: {state['status']}"
    )
    lines.append("")
    lines.append("## Where work stopped")
    if last_decision:
        lines.append(f"Last decision: {last_decision['msg']} ({last_decision['t']})")
    if last_artifact:
        lines.append(f"Last artifact: {last_artifact['path']} ({last_artifact['action']})")
    if not last_decision and not last_artifact:
        lines.append("(no decisions or artifacts recorded yet)")
    lines.append("")
    lines.append("## Objective")
    lines.append(obj)
    lines.append("")
    lines.append("## Open loops (do these)")
    if loops:
        for i, l in enumerate(loops, 1):
            lines.append(f"{i}. [{l['salience']:.2f}] {l['msg']} - unresolved")
    else:
        lines.append("(none open)")
    lines.append("")
    lines.append("## Next concrete action")
    lines.append(next_action)
    lines.append("")
    lines.append(
        f"_Derived from {state['event_count']} events; "
        f"{state['checkpoints']} checkpoints. Truth: runtime/events.jsonl_"
    )
    text = "\n".join(lines) + "\n"
    if len(text) > WORKING_CONTEXT_BUDGET_CHARS:
        text = text[:WORKING_CONTEXT_BUDGET_CHARS] + "\n_[truncated to budget]_\n"
    return text


def restore(verbose: bool = True, print_context: bool = False) -> dict:
    """Stages 1–3: derive state -> regenerate working_context.md + projections.

    print_context=True also emits working_context.md to stdout — used by the
    SessionStart hook to inject continuity state into a new session."""
    # Uninitialized project (plugin enabled but /continuity-init not run): stay
    # silent and DO NOT create .continuity/ — hooks must not litter every project.
    if not EVENTS.exists():
        if verbose and not print_context:
            print("No continuity data here yet — run /continuity-init to start.")
        return {"event_count": 0, "status": "uninitialized", "project": None,
                "open_loops": {}, "checkpoints": 0}
    state = derive_state()
    loops = open_loops_ranked(state)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    rendered = render_working_context(state, loops)
    WORKING_CONTEXT.write_text(rendered, encoding="utf-8")
    write_runtime_projections(state, loops)
    if print_context:
        print(rendered)  # stdout becomes injected session context
    elif verbose:
        print(f"Restored from {state['event_count']} events -> {WORKING_CONTEXT}")
        print(f"  project={state['project']} status={state['status']} "
              f"open_loops={len(loops)}")
    return state


# ─── checkpoint (autonomous-capable) ─────────────────────────────────────────

def checkpoint(reason: str = "manual", end_status: str = "paused",
               keep_open: bool = False) -> dict:
    """Append a checkpoint event (+ session_end unless keep_open) and refresh.

    keep_open=True records the checkpoint marker WITHOUT ending the session —
    used by the PreCompact hook (context-pressure checkpoint mid-session)."""
    # Uninitialized project (plugin enabled but /continuity-init not run): stay
    # silent and DO NOT create .continuity/. The SessionEnd/PreCompact hooks call
    # this on every project; without this guard append() would mkdir + seed a
    # stray events.jsonl, littering projects that never opted in (mirrors restore()).
    if not EVENTS.exists():
        return {"event_count": 0, "status": "uninitialized", "project": None,
                "open_loops": {}, "checkpoints": 0, "session": None}
    state = derive_state()
    append("checkpoint", reason=reason, session=state["session"])
    if state["session"] and not keep_open:
        append("session_end", session=state["session"], status=end_status)
    state = restore(verbose=False)
    tail = "kept open" if keep_open else end_status
    print(f"Checkpoint ({reason}) -> session {state['session']} {tail}; "
          f"working_context refreshed.")
    return state


def init_data() -> dict:
    """Create the per-project data root and seed the stable identity layer from the
    plugin's templates (only files that don't already exist). Idempotent — safe to run
    on an existing brain; it never clobbers a filled-in identity file."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    created = []
    for rel in IDENTITY_FILES:
        dst = ROOT / rel
        if dst.exists():
            continue
        src = TEMPLATES / rel
        text = src.read_text(encoding="utf-8") if src.exists() else f"# {rel}\n"
        text = text.replace("__DATE__", today)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        created.append(rel)
    return {"home": str(ROOT), "created": created,
            "events_exists": EVENTS.exists()}


def session_start(**fields) -> dict:
    sid = fields.pop("session", None) or datetime.now().strftime("%Y-%m-%d_%H-%M")
    ev = append("session_start", session=sid, **fields)
    restore(verbose=False)
    print(f"Session {sid} started (project={fields.get('project')}).")
    return ev


# ─── demo: proves the inversion ──────────────────────────────────────────────

def demo() -> None:
    """Append a synthetic work session, checkpoint, then restore in a 'new session'
    and show working_context.md was rebuilt with NO markdown in the load path."""
    print("=== v2 runtime demo: events -> checkpoint -> restore ===\n")
    append("session_start", session="demo_session_1", project="continuity-v2",
           branch="event-sourced-runtime")
    append("objective", msg="Stand up the event-sourced runtime MVP")
    append("decision", msg="events.jsonl is source of truth; markdown is projection",
           links=["artifact:ARCHITECTURE_V2.md"])
    append("decision", msg="Default retrieval = FTS5/BM25 -> optional Claude rerank")
    append("artifact", path="system/runtime.py", action="created")
    append("open_loop", id="loop-1", msg="Wire autonomous checkpoint triggers via hooks")
    append("open_loop", id="loop-2", msg="Decide BM25 vs dense for episodic retrieval")
    append("loop_closed", id="loop-2", resolution="BM25 default, dense pluggable/off")
    print("\n-- checkpoint --")
    checkpoint(reason="demo: artifact generated")
    print("\n-- simulate NEW session: restore reads only events --")
    restore()
    print(f"\n--- {WORKING_CONTEXT.relative_to(ROOT)} ---")
    print(WORKING_CONTEXT.read_text(encoding="utf-8"))


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_kv(args):
    fields = {}
    for a in args:
        if "=" in a:
            k, _, v = a.partition("=")
            fields[k.strip()] = v.strip()
    return fields


def main(argv):
    if not argv:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "append":
        if not rest:
            print("usage: runtime.py append <type> [key=value ...]"); return 2
        ev = append(rest[0], **_parse_kv(rest[1:]))
        print(json.dumps(ev, ensure_ascii=False)); return 0
    if cmd == "restore":
        restore(print_context="--print" in rest); return 0
    if cmd == "checkpoint":
        reason = "manual"
        if "--reason" in rest:
            reason = rest[rest.index("--reason") + 1]
        checkpoint(reason=reason, keep_open="--keep-open" in rest); return 0
    if cmd == "session-start":
        session_start(**_parse_kv(rest)); return 0
    if cmd == "init":
        res = init_data()
        print(json.dumps(res, indent=2)); return 0
    if cmd == "home":
        # Print the resolved data root (.continuity/ or $CONTINUITY_HOME). Commands
        # use this to locate identity/ files regardless of the override.
        print(str(ROOT)); return 0
    if cmd == "state":
        print(json.dumps(derive_state(), indent=2, ensure_ascii=False)); return 0
    if cmd == "tail":
        n = int(rest[0]) if rest else 10
        for e in read_events()[-n:]:
            print(json.dumps(e, ensure_ascii=False))
        return 0
    if cmd == "demo":
        demo(); return 0
    print(f"unknown command: {cmd}"); print(__doc__); return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
