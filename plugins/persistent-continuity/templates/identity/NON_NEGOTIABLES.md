---
id: foundation-non-negotiables
status: immutable
created: __DATE__
updated: __DATE__
tags: [foundation, rules, constraints]
---

# Non-Negotiables

> INVARIANT — these rules govern the project. Any session that cannot comply must
> halt and flag rather than proceed. Loaded on every resume.

## Memory Integrity

### RULE-001: The event log is the only source of truth
`runtime/events.jsonl` is append-only and irreplaceable. Everything else (the graph
DBs, working context, thread digests, projections) is derived and can be rebuilt from
it. Never hand-edit the event log to revise history.

### RULE-002: Summaries supplement, never replace
A reflection or thread digest may reference a decision. It may not substitute for one.
Source events stay; compression layers sit on top.

## Project Rules

<!-- Add the hard constraints specific to THIS project below. Examples:
### RULE-010: <invariant>
<what it means, and the consequence of violating it>
-->
