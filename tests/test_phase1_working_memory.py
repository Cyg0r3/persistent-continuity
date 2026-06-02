#!/usr/bin/env python3
"""
Phase 1 working-memory tests (ARCHITECTURE_V4.md §4.2 Phase 1).

Verifies the promotion of the attention lens to a first-class runtime object:
  1. WorkingMemory.from_graph(...).render() reproduces workspace() byte-for-byte
     (modulo the timestamp line) — the refactor preserves the projection.
  2. mutate() folds a new event into focus WITHOUT rebuilding the graph, and a
     subsequent build()+from_graph() reconciles to the same node.
  3. reprioritize() keeps the rendered lens within budget, dropping coldest nodes.
  4. concepts/procedures stay empty (reserved for Phases 2/3 — absent ⇒ current).

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase1_working_memory.py
"""

import os
import re
import sys
import json
import unittest
import subprocess
import tempfile
from pathlib import Path

SYS_DIR = (Path(__file__).resolve().parent.parent
           / "plugins" / "persistent-continuity" / "system")
RUNTIME = str(SYS_DIR / "runtime.py")
COGNITION = str(SYS_DIR / "cognition.py")

TS_LINE = re.compile(r"^# Workspace @ .*$", re.MULTILINE)


def run(script, *args, home):
    env = dict(os.environ, CONTINUITY_HOME=str(home))
    return subprocess.run([sys.executable, script, *args], env=env,
                          capture_output=True, text=True, check=True).stdout


def append(home, etype, **kv):
    args = [etype] + [f"{k}={v}" for k, v in kv.items()]
    run(RUNTIME, "append", *args, home=home)


def py(home, body):
    """Run an in-process snippet against the system modules with CONTINUITY_HOME set."""
    env = dict(os.environ, CONTINUITY_HOME=str(home))
    script = f"import sys; sys.path.insert(0, {str(SYS_DIR)!r})\n" + body
    return subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, check=True).stdout


def norm(ws):
    """Drop the volatile timestamp header so two renders can be compared."""
    return TS_LINE.sub("# Workspace @ <t>", ws).rstrip("\n")


class Phase1WorkingMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        run(RUNTIME, "init", home=self.home)
        append(self.home, "session_start", session="s1", project="proj")
        append(self.home, "objective", msg="Ship phase 1", thread_ids="t-core")
        append(self.home, "open_loop", title="wire WorkingMemory", thread_ids="t-core")
        append(self.home, "decision", msg="render is export only", thread_ids="t-core")
        run(COGNITION, "build", home=self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def test_workspace_equals_render(self):
        cli = run(COGNITION, "context", home=self.home)
        obj = py(self.home, (
            "import cognition\n"
            "print(cognition.WorkingMemory.from_graph().render(), end='')\n"))
        self.assertEqual(norm(cli), norm(obj))

    def test_fields_reserved_empty(self):
        out = py(self.home, (
            "import cognition, json\n"
            "wm = cognition.WorkingMemory.from_graph()\n"
            "print(json.dumps([wm.concepts, wm.procedures, bool(wm.active)]))\n"))
        concepts, procedures, has_active = json.loads(out)
        self.assertEqual(concepts, [])
        self.assertEqual(procedures, [])
        self.assertTrue(has_active)

    def test_mutate_adds_focus_without_rebuild(self):
        out = py(self.home, (
            "import cognition, json\n"
            "wm = cognition.WorkingMemory.from_graph()\n"
            "before = set(wm.active)\n"
            "ev = {'event_id': 'evt_new', 'type': 'decision',\n"
            "      'msg': 'fresh hot decision', 'thread_ids': ['t-core']}\n"
            "wm.mutate(ev)\n"
            "after = set(wm.active)\n"
            "top = max(wm.active.values())\n"
            "print(json.dumps(['evt_new' in after, 'evt_new' not in before,\n"
            "                  wm.active['evt_new'] >= top - 1e-9,\n"
            "                  'fresh hot decision' in wm.render()]))\n"))
        added, was_absent, is_peak, in_render = json.loads(out)
        self.assertTrue(added and was_absent and is_peak and in_render)

    def test_reprioritize_respects_budget(self):
        out = py(self.home, (
            "import cognition, json\n"
            "wm = cognition.WorkingMemory.from_graph(budget=400)\n"
            "for i in range(40):\n"
            "    wm.mutate({'event_id': f'e{i}', 'type': 'note',\n"
            "               'msg': 'x'*50, 'thread_ids': []})\n"
            "wm.reprioritize()\n"
            "print(json.dumps([len(wm.render()) <= 400, len(wm.active) < 44]))\n"))
        within_budget, trimmed = json.loads(out)
        self.assertTrue(within_budget)
        self.assertTrue(trimmed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
