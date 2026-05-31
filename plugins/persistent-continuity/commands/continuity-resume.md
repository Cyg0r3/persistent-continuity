---
description: Restore the minimum viable working state for this project before any other work
argument-hint: [deep]
---

You are running the **Continuity Resume Agent** — reconstruct the minimum viable cognitive working state for the active project before any other work begins.

**Engine:** `${CLAUDE_PLUGIN_ROOT}/system/runtime.py`. **Truth:** `<data root>/runtime/events.jsonl`. The restoration is **synthesized**, not a manual file load — `runtime/working_context.md` is the bootstrap. (The SessionStart hook usually injects this automatically; this command does it on demand and adds the identity layer.)

## Argument Handling
- Empty: standard restore (working context + stable identity).
- `deep`: also show the last 15 raw events for fuller episodic context.

## Step 0 — Locate the data root
```
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" home
```
Use this `<home>` for the identity paths below. If it doesn't exist or has no events, tell the user to run `/continuity-init` and stop.

## Step 1 — Regenerate working context from truth
```
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" restore
```
Folds `events.jsonl` into `working_context.md` (+ active_context.json, session_state.json, open_loops.json). If it errors, report it and stop — do not guess state.

## Step 2 — Load the synthesized working context
Read `<home>/runtime/working_context.md` completely: objective, where work stopped, ranked open loops, next action.

## Step 3 — Load the stable identity layer
Read (small, stable):
- `<home>/identity/NON_NEGOTIABLES.md`
- `<home>/identity/SYSTEM_PRINCIPLES.md` (if present)
- `<home>/procedural/USER_PROFILE.md` (if present)

Read `<home>/identity/PROJECT_VISION.md` only if the objective is unclear without it.

## Step 4 — (if `$ARGUMENTS` = `deep`) Recent episodic detail
```
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" tail 15
```

## Step 5 — Reconstruction verification
Confirm you can answer each; if any is unknown, retrieve it before proceeding:

| Question | Source |
|----------|--------|
| Active objective? | working_context "Objective" |
| Non-negotiable rules? | identity/NON_NEGOTIABLES.md |
| Top open loop / next action? | working_context "Open loops" / "Next concrete action" |
| Last decision/artifact? | working_context "Where work stopped" |
| User's behavioral rules? | procedural/USER_PROFILE.md |

## Output Format
```
╔══════════════════════════════════════════════════════════╗
║            CONTINUITY — WORKING STATE RESTORED           ║
╚══════════════════════════════════════════════════════════╝

Project:    [project]   Branch: [branch]
Status:     [status]    Restored: [now]
Truth:      [event_count] events · [checkpoints] checkpoints

OBJECTIVE
  [objective one-liner]

OPEN LOOPS (ranked by salience)
  1. [score] [loop msg]
  2. [score] [loop msg]

WHERE WORK STOPPED
  Last decision: [...]
  Last artifact: [...]

NEXT CONCRETE ACTION
  [top open loop / next action]

NON-NEGOTIABLES ACTIVE
  • [top rules from NON_NEGOTIABLES.md]
══════════════════════════════════════════════════════════

Ready. What would you like to work on?
```

Then wait. Do not begin work autonomously — the user leads after restoration.
