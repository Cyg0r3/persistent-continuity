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

  resolve   — the `memory` agent's conflict pass (§9): settle each contradiction via the
              same spreading-activation + lateral inhibition; when one rival dominates by
              a clear margin, propose an `assumption_invalidated` for the loser (and a
              `loop_closed` for any loop it owns). DRY-RUN by default — append only with
              --apply. Append-only means resolution never collides; it only adds
              interpretation, never edits truth.

Both are pure projections of L1 truth via the L2 graph (cognition.db). Reflection appends
ONE event (replayable); compression writes only markdown under threads/.

Subcommands:
    python reflect.py reflect [--quiet]    Run reflection; append a `reflection` event
    python reflect.py compress             (Re)write thread digests under threads/
    python reflect.py digest <thread_id>   Print a single thread's digest to stdout
    python reflect.py resolve [--apply]    memory-agent: resolve decisive contradictions
                                           (dry-run unless --apply)
    python reflect.py abstract [--apply]   T2: form concepts from recurring episodes
                                           (semantic memory; dry-run unless --apply)
    python reflect.py learn [--apply]      T2: learn procedures from recurring successful
                                           workflows (procedural memory; dry-run unless --apply)
    python reflect.py mine [--apply]       T2: detect recurring patterns (reasoning/failure/
                                           success/bottleneck) as first-class memory (§13)
    python reflect.py prune [--apply]      T2: retire stale concepts whose evidence went cold
                                           (semantic memory; dry-run unless --apply)
    python reflect.py retire-procedures [--apply]  T2: retire chronically-failing procedures by
                                           reinforced outcome_score (procedural memory; §14)
    python reflect.py consolidate [--apply]  T2 pipeline: abstract + learn + mine + prune +
                                           retire-procedures + snapshot (the "sleep cycle"; dry-run unless --apply)
    python reflect.py all                  reflect + compress

Pure stdlib. Reuses cognition.build (graph) and runtime.append (truth log).
"""

import re
import sys
import math
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
RESOLVE_MARGIN = 0.15      # min winner-loser activation gap to call a contradiction decided
                           # (after lateral inhibition has amplified the asymmetry, §5.5)


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


# ─── memory-agent conflict resolution (§9) ───────────────────────────────────

def resolve(apply: bool = False, quiet: bool = False) -> dict:
    """The `memory` agent's conflict pass. Settle each contradiction with the shared
    graph's spreading-activation + lateral inhibition (cognition.attend, memory window);
    for every DECISIVE one (winner-loser gap >= RESOLVE_MARGIN) propose the interpretive
    event that records the resolution:
      - loser is itself an open loop   -> `loop_closed`
      - otherwise (a hypothesis/etc.)  -> `assumption_invalidated` for the loser

    Dry-run by default (prints proposals); with apply=True it appends them, tagged
    agent="memory". Append-only => this never collides with other agents' writes — it
    only adds interpretation, never edits truth. Contradictions inside RESOLVE_MARGIN
    stay live (no evidence to lean)."""
    import cognition
    cognition.build()
    state = cognition.attend(agent="memory")   # inhibition-settled, memory-scoped window

    conn = sqlite3.connect(COG_DB)
    conn.row_factory = sqlite3.Row
    meta = {r["event_id"]: r for r in conn.execute(
        "SELECT event_id,type,body FROM nodes")}
    conn.close()
    import runtime
    by_eid = {e.get("event_id"): e for e in runtime.read_events()}

    proposals, undecided = [], []
    for a, b, sa, sb in state["contradictions"]:
        if abs(sa - sb) < RESOLVE_MARGIN:
            undecided.append((a, b, round(sa, 3), round(sb, 3)))
            continue
        winner, loser = (a, b) if sa >= sb else (b, a)
        lmeta = meta.get(loser)
        wbody = (meta.get(winner)["body"] if meta.get(winner) else winner)[:80]
        if lmeta and lmeta["type"] == "open_loop":
            loop_id = (by_eid.get(loser) or {}).get("id") or loser
            proposals.append({"type": "loop_closed", "id": loop_id, "causes": [winner],
                              "msg": f"superseded by {winner}: {wbody}",
                              "_loser": loser, "_winner": winner})
        else:
            proposals.append({"type": "assumption_invalidated", "target": loser,
                              "causes": [winner],
                              "msg": f"resolved in favor of {winner}: {wbody}",
                              "_loser": loser, "_winner": winner})

    applied = []
    for p in proposals:
        loser, winner = p.pop("_loser"), p.pop("_winner")
        verb = "APPLY " if apply else "would "
        if not quiet:
            print(f"  {verb}{p['type']:<22} loser={loser} winner={winner}")
        if apply:
            etype = p.pop("type")
            runtime.append(etype, agent="memory", **p)
            applied.append({"type": etype, "loser": loser, "winner": winner})

    if not quiet:
        if not proposals:
            print("  no decisive contradictions "
                  f"({len(undecided)} live, within margin {RESOLVE_MARGIN})")
        elif not apply:
            print(f"\n  dry-run: {len(proposals)} proposal(s). Re-run with --apply to write.")
    return {"proposals": len(proposals), "applied": applied,
            "undecided": undecided}


# ─── T2 consolidation: abstraction / concept formation (§9, §5.1) ────────────

ABSTRACT_MIN_SUPPORT = 3   # a term must recur across >= this many distinct episodes
                           # to be abstracted into a durable concept
ABSTRACT_MAX_NEW = 12      # cap concepts formed per pass (entropy bound, §9)
# v4.1 Phase 9 (§14): retire a procedure whose reinforced outcome_score has decayed below
# RETIRE_SCORE after at least RETIRE_MIN_USES recorded executions (enough evidence to judge).
PROC_RETIRE_SCORE = 0.25
PROC_RETIRE_MIN_USES = 3
PROC_RETIRE_MAX = 10       # cap retirements per pass (entropy bound, §9)
# Specificity gate (IDF). A term in almost every episode is generic noise ("fix",
# "update"), not a concept — drop it. But document frequency is only informative once
# the corpus is large enough, so the gate engages only at >= ABSTRACT_IDF_MIN_DOCS
# episodes; below that we fall back to support-only (a distinctive term legitimately
# appears in all of a handful of episodes). Candidates are then ranked by tf-idf so the
# MAX_NEW cap keeps the most distinctive concepts.
ABSTRACT_IDF_MIN_DOCS = 8
ABSTRACT_MAX_DF_RATIO = 0.6
# episodic/cognitive node types that carry abstractable meaning (skip pure plumbing)
ABSTRACT_TYPES = {"objective", "decision", "observation", "hypothesis",
                  "assumption", "error", "artifact", "reflection"}
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "have", "has",
    "was", "are", "were", "will", "would", "should", "could", "than", "then",
    "them", "they", "their", "what", "when", "where", "which", "while", "about",
    "after", "before", "over", "under", "your", "you", "our", "but", "not",
    "all", "any", "can", "out", "via", "use", "used", "using", "per", "its",
}
_WORD = re.compile(r"[a-z][a-z0-9_-]{2,}")


def _terms(body: str) -> set:
    """Significant lowercased terms in an episode body (stopword-filtered, len>=3)."""
    return {w for w in _WORD.findall((body or "").lower()) if w not in _STOPWORDS}


def _slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")


def abstract(apply: bool = False, quiet: bool = False) -> dict:
    """T2 abstraction: cluster recurring episodic content into durable concepts (§5.1).

    Deterministic, stdlib-only (no embeddings required): a term recurring across
    >= ABSTRACT_MIN_SUPPORT distinct episodes becomes a candidate concept, with those
    episodes as evidence. A specificity (IDF) gate drops generic terms that saturate the
    corpus and ranks the rest by tf-idf so the most distinctive concepts win the MAX_NEW
    cap. Idempotent — concepts already in the graph are skipped (a re-run forms only
    genuinely new abstractions). Dry-run by default; --apply emits `concept_formed`
    (agent="memory" — the curator owns semantic memory, §9). Append-only, so it never
    edits truth; `cognition._build_concepts` projects it into semantic memory."""
    conn = _graph()
    existing = {r["concept_id"] for r in conn.execute("SELECT concept_id FROM concepts")}
    rows = [dict(r) for r in conn.execute(
        f"SELECT event_id, body, type FROM nodes WHERE type IN "
        f"({','.join('?' * len(ABSTRACT_TYPES))})", list(ABSTRACT_TYPES))]
    conn.close()

    # document frequency: term -> set(event_id) it appears in
    df: dict = {}
    for r in rows:
        for term in _terms(r["body"]):
            df.setdefault(term, set()).add(r["event_id"])

    n_docs = len(rows)
    idf_reliable = n_docs >= ABSTRACT_IDF_MIN_DOCS

    # Score = support * idf (specificity). The hard ratio gate removes saturating-generic
    # terms outright; idf ranking demotes the merely-common. Both engage only once the
    # corpus is large enough for document frequency to mean anything.
    scored = []
    for term, ev in df.items():
        support = len(ev)
        if support < ABSTRACT_MIN_SUPPORT:
            continue
        if idf_reliable:
            if support / n_docs > ABSTRACT_MAX_DF_RATIO:
                continue  # generic noise: in most episodes, distinguishes nothing
            score = support * math.log(n_docs / support)
        else:
            score = float(support)  # tiny corpus: IDF degenerate, rank by support
        scored.append((term, ev, score))

    candidates = sorted(scored, key=lambda c: (-c[2], c[0]))

    proposals = []
    for term, ev, _score in candidates:
        cid = f"cpt_{_slug(term)}"
        if cid in existing:
            continue
        proposals.append({
            "concept_id": cid,
            "summary": f"{term} — recurring concept across {len(ev)} episodes",
            "evidence": sorted(ev),
            "support": len(ev),
        })
        if len(proposals) >= ABSTRACT_MAX_NEW:
            break

    applied = []
    if apply and proposals:
        import runtime
        for p in proposals:
            runtime.append("concept_formed", agent="memory", **p)
            applied.append(p["concept_id"])

    if not quiet:
        if not proposals:
            print(f"  no new concepts (min support {ABSTRACT_MIN_SUPPORT})")
        elif apply:
            print(f"  formed {len(applied)} concept(s): {', '.join(applied)}")
        else:
            print(f"  dry-run: {len(proposals)} candidate concept(s). "
                  "Re-run with --apply to form them:")
            for p in proposals:
                print(f"    {p['concept_id']}  (support {p['support']})")
    return {"proposals": proposals, "applied": applied}


# ─── T2 consolidation: procedure learning (§5.2, §10 T2) ─────────────────────

PROC_MIN_SUPPORT = 2   # a workflow must recur across >= this many distinct SUCCESSFUL
                       # threads (each closing a loop) to be learned as a procedure
PROC_MAX_NEW = 8       # cap procedures learned per pass (entropy bound, §9)
PROC_MIN_STEPS = 2     # a procedure is a sequence — at least two ordered steps
# execution/decision node types that constitute an actionable workflow step
PROC_STEP_TYPES = {"command", "artifact", "api_call", "decision"}
# node types describing the SITUATION a workflow addresses (its trigger cue)
PROC_TRIGGER_TYPES = {"objective", "open_loop", "error"}


def _lead(body: str) -> str:
    """Leading significant term of a step body (the step's signature token)."""
    for w in _WORD.findall((body or "").lower()):
        if w not in _STOPWORDS:
            return w
    return ""


def learn(apply: bool = False, quiet: bool = False) -> dict:
    """T2 procedure learning: distill recurring SUCCESSFUL workflows into procedures (§5.2).

    Deterministic, stdlib-only. A procedure is a workflow that *worked*: we consider only
    threads that reached a `loop_closed` (the success signal), take their ordered actionable
    steps (command/artifact/api_call/decision), and signature each thread by the sequence of
    its steps' leading terms. A signature recurring across >= PROC_MIN_SUPPORT distinct
    successful threads becomes a procedure — its `trigger` drawn from those threads'
    objective/open_loop/error cues, its `steps` from a representative run, its evidence the
    contributing episodes. Idempotent — procedures already in the graph are skipped. Dry-run
    by default; --apply emits `procedure_learned` (agent="memory" — the curator owns
    procedural memory, §9). Append-only; `cognition._build_procedures` projects it."""
    conn = _graph()
    existing = {r["procedure_id"] for r in conn.execute(
        "SELECT procedure_id FROM procedures")}
    success = {r["thread_id"] for r in conn.execute(
        "SELECT m.thread_id FROM nodes n JOIN membership m ON n.event_id=m.event_id "
        "WHERE n.type='loop_closed'")}
    step_ph = ",".join("?" * len(PROC_STEP_TYPES))
    steps = [dict(r) for r in conn.execute(
        f"SELECT n.event_id, n.seq, n.type, n.body, m.thread_id "
        f"FROM nodes n JOIN membership m ON n.event_id=m.event_id "
        f"WHERE n.type IN ({step_ph}) ORDER BY m.thread_id, n.seq",
        list(PROC_STEP_TYPES))]
    trig_ph = ",".join("?" * len(PROC_TRIGGER_TYPES))
    cues = [dict(r) for r in conn.execute(
        f"SELECT n.body, m.thread_id FROM nodes n "
        f"JOIN membership m ON n.event_id=m.event_id "
        f"WHERE n.type IN ({trig_ph})", list(PROC_TRIGGER_TYPES))]
    conn.close()

    by_thread, cues_by_thread = {}, {}
    for r in steps:
        if r["thread_id"] in success:
            by_thread.setdefault(r["thread_id"], []).append(r)
    for r in cues:
        cues_by_thread.setdefault(r["thread_id"], []).append(r["body"] or "")

    # group successful threads by their step-sequence signature
    groups: dict = {}
    for tid, ts in by_thread.items():
        if len(ts) < PROC_MIN_STEPS:
            continue
        sig = tuple(_lead(s["body"]) for s in ts)
        if not all(sig):
            continue
        g = groups.setdefault(sig, {"threads": set(), "evidence": set(),
                                    "repr": ts, "triggers": set()})
        g["threads"].add(tid)
        g["evidence"].update(s["event_id"] for s in ts)
        for body in cues_by_thread.get(tid, []):
            g["triggers"].update(_terms(body))

    ranked = sorted(
        [(sig, g) for sig, g in groups.items()
         if len(g["threads"]) >= PROC_MIN_SUPPORT],
        key=lambda kv: (-len(kv[1]["threads"]), kv[0]))

    proposals = []
    for sig, g in ranked:
        pid = f"prc_{_slug('-'.join(sig))}"
        if pid in existing:
            continue
        steps_desc = [f"{s['type']}: {(s['body'] or '')[:80]}" for s in g["repr"]]
        triggers = sorted(g["triggers"]) or [sig[0]]
        proposals.append({
            "procedure_id": pid,
            "label": "workflow: " + " → ".join(sig),
            "trigger": triggers,
            "steps": steps_desc,
            "evidence": sorted(g["evidence"]),
            "support": len(g["threads"]),
        })
        if len(proposals) >= PROC_MAX_NEW:
            break

    applied = []
    if apply and proposals:
        import runtime
        for p in proposals:
            runtime.append("procedure_learned", agent="memory", **p)
            applied.append(p["procedure_id"])

    if not quiet:
        if not proposals:
            print(f"  no new procedures (min support {PROC_MIN_SUPPORT} "
                  "successful threads)")
        elif apply:
            print(f"  learned {len(applied)} procedure(s): {', '.join(applied)}")
        else:
            print(f"  dry-run: {len(proposals)} candidate procedure(s). "
                  "Re-run with --apply to learn them:")
            for p in proposals:
                print(f"    {p['procedure_id']}  (support {p['support']}, "
                      f"{len(p['steps'])} steps)")
    return {"proposals": proposals, "applied": applied}


# ─── T2 consolidation: stale-belief pruning + pipeline (§9, §5.1) ────────────

PRUNE_MAX = 8   # entropy cap: retire at most this many stale concepts per pass (§9)


def prune(apply: bool = False, quiet: bool = False) -> dict:
    """T2 stale-belief pruning (§9): retire concepts whose evidence has ALL gone cold
    (compacted behind a snapshot) and which NO active thread still cites — meaning that did
    not persist through recurrence and is now dead weight on the active graph.

    Emits `concept_retired` (agent='memory' — the curator owns semantic memory);
    `cognition._build_concepts` then drops the concept from the projection (a later
    `concept_formed` would revive it). Dry-run by default; append-only, so it never edits
    truth. Idempotent: a retired concept is no longer materialized, so it is never re-proposed."""
    conn = _graph()
    active_members = {r["event_id"] for r in conn.execute(
        "SELECT m.event_id FROM membership m JOIN threads t ON m.thread_id=t.thread_id "
        "WHERE t.status='active'")}
    concepts = [dict(r) for r in conn.execute("SELECT concept_id FROM concepts")]
    proposals = []
    for c in concepts:
        ev = [dict(r) for r in conn.execute(
            "SELECT ce.event_id, COALESCE(n.cold,0) AS cold FROM concept_evidence ce "
            "LEFT JOIN nodes n ON n.event_id=ce.event_id WHERE ce.concept_id=?",
            (c["concept_id"],))]
        if not ev:
            continue   # no evidence in the graph yet — could be mid-formation; leave it
        all_cold = all(r["cold"] for r in ev)
        cited = any(r["event_id"] in active_members for r in ev)
        if all_cold and not cited:
            proposals.append({
                "concept_id": c["concept_id"],
                "reason": f"stale: all {len(ev)} evidence episodes cold, no active citation",
            })
        if len(proposals) >= PRUNE_MAX:
            break
    conn.close()

    applied = []
    if apply and proposals:
        import runtime
        for p in proposals:
            runtime.append("concept_retired", agent="memory", **p)
            applied.append(p["concept_id"])

    if not quiet:
        if not proposals:
            print("  no stale concepts to retire")
        elif apply:
            print(f"  retired {len(applied)} concept(s): {', '.join(applied)}")
        else:
            print(f"  dry-run: {len(proposals)} stale concept(s). "
                  "Re-run with --apply to retire:")
            for p in proposals:
                print(f"    {p['concept_id']}  ({p['reason']})")
    return {"proposals": proposals, "applied": applied}


def retire_procedures(apply: bool = False, quiet: bool = False) -> dict:
    """T2 adaptive-procedure retirement (§14, Phase 9): withdraw a procedure whose
    reinforcement-weighted `outcome_score` has decayed below PROC_RETIRE_SCORE after enough
    recorded executions (>= PROC_RETIRE_MIN_USES) — i.e. a workflow that has *consistently
    failed* in practice and should no longer be retrieved as "how-to".

    Emits `procedure_retired` (agent='memory' — the curator owns procedural memory);
    `cognition._build_procedures` then drops it from the projection (a later
    `procedure_learned` revives it). Dry-run by default; append-only; idempotent (a retired
    procedure is no longer materialized, so it is never re-proposed)."""
    conn = _graph()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT procedure_id, label, outcome_score, uses FROM procedures")]
    except Exception:
        rows = []   # pre-Phase-3 cache: no procedures table
    conn.close()
    proposals = []
    for r in rows:
        score = r["outcome_score"] if r["outcome_score"] is not None else 0.5
        uses = r["uses"] or 0
        if uses >= PROC_RETIRE_MIN_USES and score < PROC_RETIRE_SCORE:
            proposals.append({
                "procedure_id": r["procedure_id"],
                "reason": f"chronic failure: outcome_score {round(score, 3)} after {uses} executions",
            })
        if len(proposals) >= PROC_RETIRE_MAX:
            break

    applied = []
    if apply and proposals:
        import runtime
        for p in proposals:
            runtime.append("procedure_retired", agent="memory", **p)
            applied.append(p["procedure_id"])

    if not quiet:
        if not proposals:
            print("  no chronically-failing procedures to retire")
        elif apply:
            print(f"  retired {len(applied)} procedure(s): {', '.join(applied)}")
        else:
            print(f"  dry-run: {len(proposals)} failing procedure(s). "
                  "Re-run with --apply to retire:")
            for p in proposals:
                print(f"    {p['procedure_id']}  ({p['reason']})")
    return {"proposals": proposals, "applied": applied}


# ─── T2 pattern recognition (§13): recurring patterns as first-class memory ──────

PATTERN_MIN_SUPPORT = 2    # a pattern must recur across >= this many distinct threads
PATTERN_MAX_NEW = 10       # cap patterns emitted per pass (entropy bound, §9)
PATTERN_MOTIF_LEN = 3      # reasoning-motif length (contiguous type n-gram)
BOTTLENECK_MIN_DEPS = 3    # causal dependents at/above which a node is a candidate bottleneck
# cognitive types whose ORDER forms a recurring reasoning structure
REASON_TYPES = ("objective", "hypothesis", "observation", "contradiction",
                "decision", "reflection", "open_loop", "loop_closed", "assumption")
# signals that a thread hit a failure mode
FAILURE_TYPES = ("assumption_invalidated", "contradiction", "error")
# actionable step types whose leading-term sequence signatures a successful run (mirrors learn)
SUCCESS_STEP_TYPES = ("command", "artifact", "api_call", "decision")


def mine(apply: bool = False, quiet: bool = False) -> dict:
    """T2 pattern mining (§13): detect recurring patterns over the projected graph and
    record them as first-class memory objects via `pattern_detected` (curator truth).

    Deterministic, stdlib-only — a pure projection of the L2 graph. Four detectors:
      • reasoning  — a contiguous n-gram of cognitive event TYPES recurring across threads
                     (a repeated reasoning structure, e.g. hypothesis→observation→decision);
      • failure    — a failure signal (assumption_invalidated / contradiction / error) whose
                     leading term recurs across threads (a repeated failure mode);
      • success    — a successful run (thread reaching loop_closed) whose step leading-term
                     sequence recurs across threads (a reusable execution path → §16 promote);
      • bottleneck — a single node with many causal dependents inside a still-unresolved
                     thread (a prerequisite that stalls progress).
    Idempotent: pattern_ids already in the graph are skipped. Dry-run by default; --apply
    emits `pattern_detected` (agent='memory' — the curator owns the pattern layer, §10).
    Append-only; `cognition._build_patterns` projects it."""
    conn = _graph()
    existing = {r["pattern_id"] for r in conn.execute("SELECT pattern_id FROM patterns")}
    rows = [dict(r) for r in conn.execute(
        "SELECT n.event_id, n.seq, n.type, n.body, n.unresolved, m.thread_id "
        "FROM nodes n JOIN membership m ON n.event_id=m.event_id "
        "WHERE n.seq IS NOT NULL ORDER BY m.thread_id, n.seq")]
    # causal dependents = how many events list this node as a cause (out-degree of src):
    # a node many others depend on is a prerequisite that can stall progress.
    deps: dict = {}
    for r in conn.execute("SELECT src, COUNT(*) c FROM edges WHERE kind='causal' GROUP BY src"):
        deps[r["src"]] = r["c"]
    conn.close()

    by_thread: dict = {}
    for r in rows:
        by_thread.setdefault(r["thread_id"], []).append(r)

    proposals: list = []

    def _add(pid, kind, label, support, evidence, recommended):
        if pid in existing or any(p["pattern_id"] == pid for p in proposals):
            return
        proposals.append({
            "pattern_id": pid, "kind": kind, "label": label,
            "confidence": round(min(0.95, 0.5 + 0.1 * (support - 1)), 4),
            "frequency": support, "evidence": evidence,
            "recommended": recommended,
        })

    # (1) reasoning motifs: recurring contiguous n-gram of cognitive TYPES across threads
    motifs: dict = {}
    for tid, evs in by_thread.items():
        seq = [e["type"] for e in evs if e["type"] in REASON_TYPES]
        for i in range(len(seq) - PATTERN_MOTIF_LEN + 1):
            gram = tuple(seq[i:i + PATTERN_MOTIF_LEN])
            motifs.setdefault(gram, set()).add(tid)
    for gram, tids in sorted(motifs.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(tids) < PATTERN_MIN_SUPPORT:
            continue
        _add(f"pat_reason_{_slug('-'.join(gram))}", "reasoning",
             "reasoning structure: " + " → ".join(gram),
             len(tids), sorted(tids),
             [f"recurring reasoning path {' → '.join(gram)} — reuse or examine it"])

    # (2) failure modes: a failure signal whose leading term recurs across threads
    fail_groups: dict = {}
    for tid, evs in by_thread.items():
        for e in evs:
            is_fail = e["type"] in FAILURE_TYPES or (
                e["type"] == "open_loop" and e["unresolved"])
            if not is_fail:
                continue
            term = _lead(e["body"])
            if not term:
                continue
            g = fail_groups.setdefault(term, {"threads": set(), "evidence": set()})
            g["threads"].add(tid)
            g["evidence"].add(e["event_id"])
    for term, g in sorted(fail_groups.items(), key=lambda kv: (-len(kv[1]["threads"]), kv[0])):
        if len(g["threads"]) < PATTERN_MIN_SUPPORT:
            continue
        _add(f"pat_fail_{_slug(term)}", "failure",
             f"recurring failure mode: '{term}'",
             len(g["threads"]), sorted(g["evidence"]),
             [f"'{term}' recurs as a failure — add a guard / check earlier"])

    # (3) success paths: a successful run's step leading-term sequence recurring across threads
    success = {tid for tid, evs in by_thread.items()
               if any(e["type"] == "loop_closed" for e in evs)}
    succ_groups: dict = {}
    for tid in success:
        steps = [e for e in by_thread[tid] if e["type"] in SUCCESS_STEP_TYPES]
        if len(steps) < 2:
            continue
        sig = tuple(_lead(s["body"]) for s in steps)
        if not all(sig):
            continue
        g = succ_groups.setdefault(sig, {"threads": set(), "evidence": set()})
        g["threads"].add(tid)
        g["evidence"].update(s["event_id"] for s in steps)
    for sig, g in sorted(succ_groups.items(), key=lambda kv: (-len(kv[1]["threads"]), kv[0])):
        if len(g["threads"]) < PATTERN_MIN_SUPPORT:
            continue
        _add(f"pat_success_{_slug('-'.join(sig))}", "success",
             "successful path: " + " → ".join(sig),
             len(g["threads"]), sorted(g["evidence"]) + sorted(g["threads"]),
             ["recurring successful execution path — promote to a procedure (§16)"])

    # (4) bottlenecks: a high causal in-degree node inside a still-unresolved thread
    unresolved_threads = {tid for tid, evs in by_thread.items()
                          if any(e["unresolved"] for e in evs)}
    seen_bottleneck: set = set()
    for r in rows:
        eid = r["event_id"]
        if eid in seen_bottleneck or deps.get(eid, 0) < BOTTLENECK_MIN_DEPS:
            continue
        if r["thread_id"] not in unresolved_threads:
            continue
        seen_bottleneck.add(eid)
        _add(f"pat_bottleneck_{_slug(eid)}", "bottleneck",
             f"bottleneck: {(r['body'] or eid)[:60]}",
             deps[eid], [eid],
             [f"{deps[eid]} causal dependents and the thread is unresolved — unblock first"])

    proposals = proposals[:PATTERN_MAX_NEW]

    applied = []
    if apply and proposals:
        import runtime
        for p in proposals:
            runtime.append("pattern_detected", agent="memory", **p)
            applied.append(p["pattern_id"])

    if not quiet:
        if not proposals:
            print(f"  no new patterns (min support {PATTERN_MIN_SUPPORT} threads)")
        elif apply:
            print(f"  detected {len(applied)} pattern(s): {', '.join(applied)}")
        else:
            print(f"  dry-run: {len(proposals)} candidate pattern(s). "
                  "Re-run with --apply to record them:")
            for p in proposals:
                print(f"    {p['pattern_id']}  ({p['kind']}, conf {p['confidence']}, "
                      f"freq {p['frequency']})")
    return {"proposals": proposals, "applied": applied}


def consolidate(apply: bool = False, quiet: bool = False) -> dict:
    """T2 consolidation pipeline (§9, §10): the offline "sleep cycle" run in order —
    abstraction → procedure learning → stale-belief pruning → snapshot/compaction.

    Each stage is append-only and dry-run unless --apply, so this is the single entry point
    for opportunistic (idle / N-events / PreCompact) consolidation that never blocks an
    interactive turn. Pruning acts on episodes a PRIOR snapshot compacted, so the cycle
    converges over successive runs (form/learn now; compact + prune what stays cold later)."""
    import cognition
    if not quiet:
        print("T2 consolidation pipeline" + (" (--apply)" if apply else " (dry-run)") + ":")
        print(" [1/6] abstraction")
    a = abstract(apply=apply, quiet=quiet)
    if not quiet:
        print(" [2/6] procedure learning")
    l = learn(apply=apply, quiet=quiet)
    if not quiet:
        print(" [3/6] pattern mining")
    m = mine(apply=apply, quiet=quiet)
    if not quiet:
        print(" [4/7] stale-belief pruning")
    p = prune(apply=apply, quiet=quiet)
    if not quiet:
        print(" [5/7] procedure retirement")
    rp = retire_procedures(apply=apply, quiet=quiet)
    if not quiet:
        print(" [6/7] snapshot / compaction")
    s = cognition.snapshot(apply=apply)
    if not quiet:
        if apply:
            print(f"   snapshot at seq {s['snapshot_seq']} -> {s['written']}")
        else:
            print(f"   dry-run: snapshot would cover {s['snapshot_seq']} events")
    # v4 Phase 6: subconscious incubation refreshes a cache (no truth side effect), so it
    # runs on apply as the offline pass that lets dormant ideas resurface next session.
    inc = None
    if not quiet:
        print(" [7/7] incubation")
    if apply:
        inc = cognition.incubate()
        if not quiet:
            print(f"   incubated {inc['incubated']} dormant node(s)")
    elif not quiet:
        print("   dry-run: incubation would refresh the subconscious channel")
    return {"abstract": a, "learn": l, "mine": m, "prune": p,
            "retire_procedures": rp, "snapshot": s, "incubate": inc}


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
    if cmd == "resolve":
        resolve(apply="--apply" in rest, quiet="--quiet" in rest); return 0
    if cmd == "abstract":
        abstract(apply="--apply" in rest, quiet="--quiet" in rest); return 0
    if cmd == "learn":
        learn(apply="--apply" in rest, quiet="--quiet" in rest); return 0
    if cmd == "mine":
        mine(apply="--apply" in rest, quiet="--quiet" in rest); return 0
    if cmd == "prune":
        prune(apply="--apply" in rest, quiet="--quiet" in rest); return 0
    if cmd == "retire-procedures":
        retire_procedures(apply="--apply" in rest, quiet="--quiet" in rest); return 0
    if cmd == "consolidate":
        consolidate(apply="--apply" in rest, quiet="--quiet" in rest); return 0
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
