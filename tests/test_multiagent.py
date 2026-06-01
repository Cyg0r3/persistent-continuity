#!/usr/bin/env python3
"""
Smoke tests for the multi-agent attention runtime (ARCHITECTURE_V3.md §9).

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Each test runs the real CLIs against a throwaway $CONTINUITY_HOME so it exercises the
same path the hooks use. Run: python tests/test_multiagent.py

Asserts the three §9 guarantees:
  1. agent='' (single-agent) attention is unchanged & stable across runs.
  2. two role windows over the SAME graph produce different focus orderings.
  3. the memory-agent resolver fires only on a DECISIVE contradiction, stays dry-run
     by default (truth untouched), and appends an interpretation only with --apply.
"""

import os
import sys
import json
import unittest
import subprocess
import tempfile
from pathlib import Path

SYS_DIR = (Path(__file__).resolve().parent.parent
           / "plugins" / "persistent-continuity" / "system")
COGNITION = str(SYS_DIR / "cognition.py")
REFLECT = str(SYS_DIR / "reflect.py")

# --- synthetic event log -----------------------------------------------------
# auth thread: a normal line of work. infra thread: a contradiction (Redis vs
# Postgres) with a CAUSAL chain of support behind Redis (evt_5) — so lateral
# inhibition has an asymmetry to amplify and the resolver can decide it.
EVENTS = [
    {"type": "objective", "event_id": "evt_1", "layer": "cognitive",
     "msg": "Build the auth system", "thread_ids": ["auth"]},
    {"type": "decision", "event_id": "evt_2", "layer": "cognitive",
     "msg": "Use JWT tokens", "thread_ids": ["auth"], "causes": ["evt_1"]},
    {"type": "artifact", "event_id": "evt_3", "layer": "execution",
     "path": "auth/jwt.py", "action": "created", "thread_ids": ["auth"]},
    {"type": "open_loop", "event_id": "evt_4", "layer": "cognitive", "id": "loop-1",
     "msg": "Refresh token rotation not implemented", "thread_ids": ["auth"]},
    {"type": "hypothesis", "event_id": "evt_5", "layer": "cognitive",
     "msg": "Redis is best for the session store", "thread_ids": ["infra"]},
    {"type": "hypothesis", "event_id": "evt_6", "layer": "cognitive",
     "msg": "Postgres is best for the session store", "thread_ids": ["infra"]},
    {"type": "contradiction", "event_id": "evt_7", "layer": "cognitive",
     "msg": "Redis vs Postgres for sessions", "between": ["evt_5", "evt_6"]},
    {"type": "command", "event_id": "evt_8", "layer": "execution",
     "msg": "pytest auth/", "thread_ids": ["auth"]},
    # asymmetric support for evt_5 (Redis): a decision + artifact causally behind it
    {"type": "decision", "event_id": "evt_9", "layer": "cognitive",
     "msg": "Benchmarks favor Redis latency", "thread_ids": ["infra"],
     "causes": ["evt_5"]},
    {"type": "artifact", "event_id": "evt_10", "layer": "execution",
     "path": "infra/redis_bench.md", "action": "created", "thread_ids": ["infra"],
     "causes": ["evt_9"]},
]


def _ts(i):
    return f"2026-05-{20 + i // 4:02d}T{9 + i % 4:02d}:00:00+00:00"


def write_log(home: Path):
    rt = home / "runtime"
    rt.mkdir(parents=True, exist_ok=True)
    with (rt / "events.jsonl").open("w", encoding="utf-8") as fh:
        for i, e in enumerate(EVENTS):
            line = dict(e)
            line["t"] = _ts(i)
            fh.write(json.dumps({"t": line.pop("t"), **line}) + "\n")


def run(script, *args, home):
    env = dict(os.environ, CONTINUITY_HOME=str(home))
    return subprocess.run([sys.executable, script, *args], env=env,
                          capture_output=True, text=True, check=True).stdout


def attend_order(home, *flags):
    """Parse `attend` stdout into an ordered list of event_ids (focus ranking)."""
    out = run(COGNITION, "attend", "-n", "10", *flags, home=home)
    ids = []
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith("[") and "evt_" in ln:
            ids.append(ln.split("]", 1)[1].split()[0])
    return ids


class MultiAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        write_log(self.home)
        run(COGNITION, "build", home=self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roles_lists_four(self):
        out = run(COGNITION, "roles", home=self.home)
        for role in ("planner", "coder", "researcher", "memory"):
            self.assertIn(role, out)

    def test_single_agent_stable(self):
        # the default (agent='') window must be deterministic and not corrupted by
        # an interleaved agent window writing to its own persistence scope.
        a = attend_order(self.home)
        run(COGNITION, "attend", "--agent", "coder", home=self.home)
        b = attend_order(self.home)
        self.assertEqual(a, b, "single-agent focus changed after an agent window ran")
        self.assertTrue(a, "expected a non-empty focus ordering")

    def test_windows_differ(self):
        coder = attend_order(self.home, "--agent", "coder")
        planner = attend_order(self.home, "--agent", "planner")
        self.assertNotEqual(coder, planner,
                            "coder and planner windows produced identical focus")
        # Role bias should FLIP a same-thread pair: evt_9 is a decision (planner focus),
        # evt_10 an artifact (coder focus), both in the infra thread. A coder ranks the
        # artifact above the decision; a planner ranks the decision above the artifact.
        self.assertLess(coder.index("evt_10"), coder.index("evt_9"),
                        "coder window did not favor the artifact over the decision")
        self.assertLess(planner.index("evt_9"), planner.index("evt_10"),
                        "planner window did not favor the decision over the artifact")

    def test_assigned_thread_boost(self):
        base = attend_order(self.home, "--agent", "researcher")
        infra = attend_order(self.home, "--agent", "researcher",
                             "--threads", "infra")
        self.assertNotEqual(base, infra,
                            "assigned-thread boost had no effect on the window")

    def test_resolver_decisive_dryrun_then_apply(self):
        log = self.home / "runtime" / "events.jsonl"
        before = log.read_text(encoding="utf-8")

        dry = run(REFLECT, "resolve", home=self.home)
        self.assertIn("would", dry, "dry-run should describe a proposal")
        self.assertEqual(before, log.read_text(encoding="utf-8"),
                         "dry-run must not write to the event log")

        run(REFLECT, "resolve", "--apply", home=self.home)
        after = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l]
        verdicts = [e for e in after
                    if e["type"] in ("assumption_invalidated", "loop_closed")]
        self.assertTrue(verdicts, "resolver --apply emitted no verdict")
        v = verdicts[-1]
        self.assertEqual(v.get("agent"), "memory",
                         "resolver verdict must be tagged agent=memory")
        # Redis (evt_5) had the causal support; Postgres (evt_6) should be the loser.
        self.assertEqual(v.get("target"), "evt_6")


if __name__ == "__main__":
    unittest.main(verbosity=2)
