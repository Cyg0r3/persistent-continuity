---
description: Snapshot this session into the continuity event log before context exhausts or work pauses
argument-hint: [reason, e.g. "context pressure" / "switching focus"]
---

You are running the **Continuity Checkpoint Agent** — snapshot this session into the event log. A checkpoint is **cheap** (it appends events and regenerates the projection) — checkpoint often.

**Engine:** `${CLAUDE_PLUGIN_ROOT}/system/runtime.py`. **Truth:** `<data root>/runtime/events.jsonl`. Run from the project directory so the data root resolves correctly.

## Argument Handling
`$ARGUMENTS` is the trigger reason (e.g., "context pressure", "switching focus", "end of session"). If empty, use "manual".

## Step 1 — Reflect on this session
Review what actually happened **this session** and extract concrete, true events:
- **Decisions** made (architecture/approach). One `decision` event each.
- **Artifacts** created or modified. One `artifact` event each.
- **Open loops** opened (new unresolved threads) and **closed** (resolved).
- **Assumptions** invalidated, if any.

If nothing meaningful happened, say so and skip to Step 3 (a bare checkpoint marker is still valid).

## Step 2 — Append the events
One call per item. Quote values containing spaces:
```
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" append decision msg="Chose X over Y because Z" links="artifact:path/to/file"
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" append artifact path="relative/path.ext" action=created
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" append open_loop id=loop-NN msg="Unresolved thing to do next"
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" append loop_closed id=loop-NN resolution="How it was resolved"
```
Give open loops short stable ids (`loop-<slug>` or `loop-NN`); close by the same id.

Optionally tag cognitive events with a thread to strengthen long-horizon coherence:
`thread_ids=salience-ranking` or `causes=evt_44` (both accept comma-separated lists).

## Step 3 — Checkpoint
```
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" checkpoint --reason "[reason from $ARGUMENTS or 'manual']"
```
Appends `checkpoint` + `session_end` and regenerates `working_context.md`, `open_loops.json`, `active_context.json`, `session_state.json` from truth.

## Step 4 — Confirm
Read back the regenerated `working_context.md` and present:
```
╔══════════════════════════════════════════════════════════╗
║              CHECKPOINT WRITTEN TO EVENT LOG            ║
╚══════════════════════════════════════════════════════════╝

Reason:     [reason]
Session:    [session id]   Status: paused
Appended:   [N decisions, M artifacts, P loops opened, Q closed]
Totals:     [event_count] events · [checkpoints] checkpoints

Next concrete action (from working_context):
  [top open loop / next action]

To resume in a new session:  /continuity-resume
══════════════════════════════════════════════════════════
```

Truth is the event log; the markdown projections regenerate themselves — there is no manual state editing.
