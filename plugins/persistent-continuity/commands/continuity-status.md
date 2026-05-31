---
description: Fast, structured read of project continuity state without loading full context
argument-hint: [events | full]
---

You are running the **Continuity Status Agent** — give a fast, structured read of project state without loading full context.

**Engine:** `${CLAUDE_PLUGIN_ROOT}/system/runtime.py`. **Truth:** `<data root>/runtime/events.jsonl`.

## Argument Handling
- empty: standard status dashboard.
- `events`: show the last 20 raw events.
- `full`: dashboard + recent decisions + artifacts from derived state.

## Standard Status (default)
Run (fast, no full-context load):
```
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" state
```
Prints derived state as JSON (project, status, objective, decisions, artifacts, open_loops, event_count, checkpoints). If it reports `"status": "uninitialized"`, tell the user to run `/continuity-init` and stop.

Optionally also surface the cognitive view:
```
python "${CLAUDE_PLUGIN_ROOT}/system/cognition.py" threads
```

Then print:
```
╔══════════════════════════════════════════════════════════╗
║                   CONTINUITY — STATUS                   ║
╚══════════════════════════════════════════════════════════╝

PROJECT   [project] · branch [branch] · status [status]
TRUTH     [event_count] events · [checkpoints] checkpoints · last event [t]

OBJECTIVE
  [objective msg]

OPEN LOOPS ([N] unresolved, ranked)
  1. [score] [msg]
  2. [score] [msg]

LAST DECISION
  [most recent decision msg + date]

LAST ARTIFACT
  [most recent artifact path + action]

DRIFT RISK
  [heuristic: many unresolved high-salience loops? long gap since last event?
   stale objective vs recent work? report low / medium / high + one-line reason]
══════════════════════════════════════════════════════════

  /continuity-resume       Restore full working state
  /continuity-checkpoint   Snapshot session into the event log
  /continuity-status full  Add decisions + artifacts detail
```

## Events Mode (`$ARGUMENTS` = `events`)
Run `python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" tail 20` and print the stream directly.

## Full Mode (`$ARGUMENTS` = `full`)
From the `state` JSON, additionally list the last 5 decisions and last 5 artifacts with timestamps, and quote the working_context "Where work stopped" section.
