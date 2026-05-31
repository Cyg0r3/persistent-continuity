---
description: Initialize the persistent-continuity event log + identity layer for the current project
argument-hint: [project name or short description]
---

You are running the **Continuity System Initializer** — set up the event-sourced runtime for THIS project so the first `/continuity-resume` reconstructs real working state.

**Engine:** `${CLAUDE_PLUGIN_ROOT}/system/runtime.py` (run with `python`).
**Data root:** a per-project `.continuity/` in the current working directory (or `$CONTINUITY_HOME` if set). The plugin code is shared/read-only; your memory is local to this project. Truth = `<data root>/runtime/events.jsonl`. Identity = `<data root>/identity/` (stable). Work is tracked as **open loops** (events), not task files.

## Argument Handling
If `$ARGUMENTS` is a project name/description, seed Section 1 with it. Otherwise ask first.

## Step 0 — Scaffold the data root
Run once up front (creates `.continuity/` and seeds identity templates; idempotent):
```
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" init
```
This prints the resolved `home` path and which identity files it created. Note the `home` path — the identity files to edit live under it (`<home>/identity/…`, `<home>/procedural/USER_PROFILE.md`).

## Interview Protocol
Work one section at a time. Ask, wait, then continue. Do not ask multiple sections at once.

### Section 1 — Project Identity
> "What is this project, and what problem does it solve? 2–4 sentences."
Capture: a short **slug** for the event `project` field, the problem, who benefits.

### Section 2 — Goals
> "What are 3–5 goals that define success? Each should be objectively 'done'-able."

### Section 3 — Constraints / Non-Negotiables
> "What hard constraints and non-negotiable rules govern this project?"

### Section 4 — First Objective + Open Loops
> "What is the single most important thing to do next, and what concrete unresolved threads (open loops) does it break into?"

## Writing the Identity Layer
Edit the seeded templates under `<home>` (preserve their YAML frontmatter), filling in real content:
- `identity/PROJECT_VISION.md` — purpose, goals, constraints, success criteria, non-goals, users.
- `identity/NON_NEGOTIABLES.md` — the invariant rules from Section 3 (keep RULE-001/002).
- `identity/SYSTEM_PRINCIPLES.md` — only if the user states operating principles; else leave as-is or delete.
- `procedural/USER_PROFILE.md` — only if the user shares working-style preferences.

Keep these short and durable — they are loaded on every resume.

## Seeding the Event Log (truth)
Run from the project directory (do NOT `cd` elsewhere — data resolves from cwd):
```
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" session-start project=<slug> branch=main
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" append objective msg="<first objective from Section 4>"
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" append open_loop id=loop-1 msg="<first concrete unresolved thread>"
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" append open_loop id=loop-2 msg="<second, if any>"
python "${CLAUDE_PLUGIN_ROOT}/system/runtime.py" restore
```
(Add one `open_loop` per concrete thread. Quote values with spaces.)

## Completion
Read back `<home>/runtime/working_context.md` and print:
```
Continuity initialized for: [Project Name]   (slug: [slug])
Data root: [home]

Identity written:
  ✓ identity/PROJECT_VISION.md
  ✓ identity/NON_NEGOTIABLES.md
Event log seeded:
  ✓ session_start + objective + [N] open loops → runtime/events.jsonl
  ✓ runtime/working_context.md generated

First objective: [objective]
Top open loop:   [loop-1 msg]

Resume any future session:  /continuity-resume
Checkpoint your work:        /continuity-checkpoint
```

Begin now. If `$ARGUMENTS` has a project name/description use it for Section 1; else ask: "What project are we initializing?"
