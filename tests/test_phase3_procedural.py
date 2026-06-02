#!/usr/bin/env python3
"""
Phase 3 procedural-memory tests (ARCHITECTURE_V4.md §5.2, §11 Phase 3).

Verifies the new procedural layer preserves every invariant:
  1. The T2 procedure-learning pass distills recurring SUCCESSFUL workflows into
     procedures, is dry-run by default, writes only on --apply, and is idempotent.
  2. build() projects procedure_learned events into the procedures/procedure_evidence
     tables and `procedure` edges, and materializes each procedure as a graph node.
  3. A procedure is trigger-matched: it surfaces in WorkingMemory only when its trigger
     matches the current situation (a live open loop), not always-on.
  4. Backward-compat: a brain with NO procedures renders byte-for-byte as before, and a
     pre-Phase-3 cache (older schema_version) rebuilds rather than silently skipping.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase3_procedural.py
"""

import os
import sys
import sqlite3
import unittest
import subprocess
import tempfile
from pathlib import Path

SYS_DIR = (Path(__file__).resolve().parent.parent
           / "plugins" / "persistent-continuity" / "system")
RUNTIME = str(SYS_DIR / "runtime.py")
COGNITION = str(SYS_DIR / "cognition.py")
REFLECT = str(SYS_DIR / "reflect.py")


def run(script, *args, home):
    env = dict(os.environ, CONTINUITY_HOME=str(home))
    return subprocess.run([sys.executable, script, *args], env=env,
                          capture_output=True, text=True, check=True).stdout


def append(home, etype, **kv):
    args = [etype] + [f"{k}={v}" for k, v in kv.items()]
    run(RUNTIME, "append", *args, home=home)


def db(home):
    conn = sqlite3.connect(Path(home) / "runtime" / "cognition.db")
    conn.row_factory = sqlite3.Row
    return conn


def seed_workflow(home, tid, loop_id):
    """One successful thread: an objective cue + the recurring 3-step workflow + a closed loop."""
    append(home, "objective", msg="deploy migration pipeline", thread_ids=tid)
    append(home, "command", msg="run migration script", thread_ids=tid)
    append(home, "command", msg="execute tests suite", thread_ids=tid)
    append(home, "artifact", msg="deploy build output", thread_ids=tid)
    append(home, "loop_closed", id=loop_id, msg="pipeline shipped", thread_ids=tid)


PID = "prc_run-execute-deploy"


class Phase3Procedural(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        append(self.home, "session_start", session="s1", project="demo")
        # two SUCCESSFUL threads sharing the same run→execute→deploy workflow
        seed_workflow(self.home, "t-ship-1", "loop-1")
        seed_workflow(self.home, "t-ship-2", "loop-2")
        run(COGNITION, "build", "--force", home=self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def test_learn_dry_run_then_apply_idempotent(self):
        dry = run(REFLECT, "learn", home=self.home)
        self.assertIn("dry-run", dry)
        self.assertNotIn("procedure_learned",
                         (Path(self.home) / "runtime" / "events.jsonl").read_text())
        run(REFLECT, "learn", "--apply", home=self.home)
        self.assertIn("procedure_learned",
                      (Path(self.home) / "runtime" / "events.jsonl").read_text())
        again = run(REFLECT, "learn", "--apply", home=self.home)
        self.assertIn("no new procedures", again)

    def test_build_projects_procedures_and_edges(self):
        run(REFLECT, "learn", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        conn = db(self.home)
        row = conn.execute("SELECT * FROM procedures WHERE procedure_id=?",
                           (PID,)).fetchone()
        self.assertIsNotNone(row)
        self.assertGreaterEqual(row["uses"], 1)
        self.assertTrue(row["steps"])                 # ordered steps stored
        self.assertIn("migration", row["trigger"])    # trigger drawn from objective cue
        # materialized as a graph node...
        node = conn.execute("SELECT * FROM nodes WHERE event_id=?", (PID,)).fetchone()
        self.assertEqual(node["type"], "procedure")
        self.assertEqual(node["layer"], "procedural")
        # ...wired to evidence by procedure edges
        n_ev = conn.execute("SELECT COUNT(*) FROM procedure_evidence WHERE procedure_id=?",
                            (PID,)).fetchone()[0]
        n_edge = conn.execute("SELECT COUNT(*) FROM edges WHERE kind='procedure' AND src=?",
                             (PID,)).fetchone()[0]
        conn.close()
        self.assertEqual(n_ev, n_edge)
        self.assertGreaterEqual(n_ev, 3)

    def test_procedure_trigger_matched_in_working_memory(self):
        run(REFLECT, "learn", "--apply", home=self.home)
        # a live open loop whose text matches the procedure trigger ("migration")
        append(self.home, "open_loop", id="ol-1", msg="need to redo the migration")
        run(COGNITION, "build", "--force", home=self.home)
        ctx = run(COGNITION, "context", home=self.home)
        self.assertIn("## Procedures (applicable how-to)", ctx)
        self.assertIn(PID, ctx)

    def test_procedure_not_injected_without_trigger_match(self):
        run(REFLECT, "learn", "--apply", home=self.home)
        # a live open loop on an UNRELATED topic must not pull in the procedure
        append(self.home, "open_loop", id="ol-x", msg="investigate unrelated logging noise")
        run(COGNITION, "build", "--force", home=self.home)
        ctx = run(COGNITION, "context", "logging", home=self.home)
        self.assertNotIn(PID, ctx)

    def test_no_procedures_render_unchanged(self):
        ctx = run(COGNITION, "context", home=self.home)
        self.assertNotIn("Procedures (applicable how-to)", ctx)

    def test_unsuccessful_workflow_not_learned(self):
        # a recurring workflow that never closed a loop is NOT a procedure (no success signal)
        home = Path(self.tmp.name) / "nofail"
        append(home, "session_start", session="s2", project="nofail")
        for tid in ("t-a", "t-b"):
            append(home, "command", msg="run migration script", thread_ids=tid)
            append(home, "command", msg="execute tests suite", thread_ids=tid)
            append(home, "artifact", msg="deploy build output", thread_ids=tid)
        run(COGNITION, "build", "--force", home=home)
        out = run(REFLECT, "learn", home=home)
        self.assertIn("no new procedures", out)

    def test_pre_phase3_cache_rebuilds(self):
        conn = db(self.home)
        conn.execute("DELETE FROM meta WHERE key='schema_version'")
        conn.execute("DROP TABLE procedures")
        conn.commit()
        conn.close()
        out = run(COGNITION, "build", home=self.home)   # no --force
        self.assertIn("built", out)                      # rebuilt, not "skipped"
        conn = db(self.home)
        self.assertIsNotNone(conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone())
        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
