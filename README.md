# Persistent Continuity

**Cognition-native long-horizon memory for [Claude Code](https://claude.com/claude-code).**

Claude forgets between sessions, when the context window fills, and after compaction.
Persistent Continuity fixes that — not by replaying a chat log, but by **reconstructing
the minimum viable cognitive state** needed to continue your work accurately.

> An append-only event log is the only source of truth. A self-maintaining cognitive
> graph, focused by spreading-activation **attention**, rebuilds your working state on
> demand. Sessions are just a time-window view — not the unit of memory.

Pure Python **standard library**. Local-first. No API keys, no cloud, no dependencies.

---

## How it works (30 seconds)

```
events.jsonl   →   cognitive graph   →   attention   →   working context
(the only truth)   (nodes + threads      (what's relevant     (a tiny lens, ≤~1.5k
 append-only        + causal edges)       right now)          tokens, regenerated)
```

- **Truth is an event log.** Every decision, artifact, open loop, and observation is one
  append-only line in `runtime/events.jsonl`. It is the *only* thing that can't be rebuilt.
- **Everything else is derived.** The graph DBs, the ranked open loops, the working-context
  markdown, the per-thread digests — all regenerate from the log. Delete them and they
  come back.
- **Attention, not keyword search.** Relevance spreads through the graph from your current
  intent + unresolved loops + active threads, with temporal decay, reinforcement-on-revisit,
  competing-hypothesis inhibition, and dormant-thread resurfacing.
- **It maintains itself.** Hooks restore state when a session starts, checkpoint under
  context pressure and at session end, and run a drift check on every prompt.

See [`docs/ARCHITECTURE_V3.md`](docs/ARCHITECTURE_V3.md) for the full design (v2 lineage in
[`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md)).

---

## Requirements

- **Claude Code**
- **Python 3.10+** available on your `PATH` as `python`
  (the lifecycle hooks call `python`; on a system where only `python3` exists, either
  alias it or edit `plugins/persistent-continuity/hooks/hooks.json`).

No pip installs. Everything is stdlib.

---

## Install (plugin marketplace)

Add the marketplace and install the plugin from inside Claude Code:

```
/plugin marketplace add Cyg0r3/persistent-continuity
/plugin install persistent-continuity@persistent-continuity
```

Then **restart Claude Code** so the hooks load.

### Manual install (no marketplace)

```
git clone https://github.com/Cyg0r3/persistent-continuity.git
```
Then in Claude Code: `/plugin marketplace add /path/to/persistent-continuity` and
`/plugin install persistent-continuity@persistent-continuity`.

---

## Quick start

In any project directory:

```
/continuity-init        # interview → writes identity + seeds the event log
/continuity-resume      # restore working state (also runs automatically on session start)
/continuity-checkpoint  # snapshot the session before you stop or compact
/continuity-status      # fast dashboard: objective, open loops, drift risk
```

That's it. After `/continuity-init`, every new session auto-restores where you left off.

---

## Where your memory lives

Your data is **per-project and local** — never bundled with the plugin code:

- **Default:** a hidden `.continuity/` folder in the project's working directory.
- **Override:** set `CONTINUITY_HOME` to an absolute path to relocate it (e.g. one shared
  brain across shells, or keeping memory outside the repo).

```
.continuity/
├── runtime/
│   ├── events.jsonl        # THE TRUTH (append-only). Back this up; everything else rebuilds.
│   ├── cognition.db        # graph + attention state (cache)
│   ├── state.db            # relational projection (cache)
│   └── working_context.md  # the regenerated working lens
├── identity/               # stable: PROJECT_VISION, NON_NEGOTIABLES, SYSTEM_PRINCIPLES
├── procedural/             # USER_PROFILE (optional)
└── threads/                # per-thread digests
```

The plugin **never creates `.continuity/` until you run `/continuity-init`** — having it
installed won't litter folders into projects you haven't initialized.

> **Tip:** `.continuity/` is git-ignorable. Commit `events.jsonl` if you want your project's
> memory versioned and shared; ignore the `*.db` caches.

---

## What gets automated (hooks)

| Hook | When | What it does |
|------|------|--------------|
| `SessionStart` | new session | Injects your restored working context |
| `UserPromptSubmit` | every prompt | Drift check; re-injects a fresh lens only if the topic turned |
| `PreCompact` | context pressure | Checkpoints (keeps the session open) so nothing is lost to compaction |
| `SessionEnd` | session ends | Final checkpoint |

All hooks are silent no-ops in projects that haven't been initialized.

---

## The engine (for the curious)

Everything is driven by `plugins/persistent-continuity/system/runtime.py` and
`cognition.py`. You can call them directly:

```bash
python system/runtime.py init                       # scaffold + seed identity
python system/runtime.py append decision msg="..."  # write one event
python system/runtime.py restore                    # rebuild working_context.md
python system/runtime.py state                       # derived state as JSON
python system/cognition.py build                     # rebuild the graph
python system/cognition.py attend "query" -n 10      # show top active nodes/threads
python system/cognition.py context "query"           # synthesize the working lens
python system/cognition.py threads                   # list threads (active/dormant/merged)
```

Data root resolves the same way for the CLI as for the hooks (`CONTINUITY_HOME` or
`<cwd>/.continuity`), so run them from your project directory.
