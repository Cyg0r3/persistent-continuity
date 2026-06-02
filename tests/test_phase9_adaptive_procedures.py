#!/usr/bin/env python3
"""
Phase 9 adaptive procedural learning tests (ARCHITECTURE_V4.md §14, migration Phase 9).

Phase 3 stores procedures statically; V4.1 makes them self-improving. This verifies every
invariant of that upgrade while preserving full back-compat:
  1. `proc_executed` is a KNOWN, episodic truth event — anyone may emit it (NOT curator-owned),
     so a named non-curator agent emitting it is NOT an arbitration violation.
  2. `cognition._build_procedures` folds executions into a reinforcement-weighted `outcome_score`
     via a bounded delta-rule: successes raise it, failures lower it, recency dominates. The
     score is a deterministic replay of the execution log.
  3. Each `proc_executed` counts as a use and advances last_used_t (procedure_learned still
     guarantees uses >= 1, the Phase 3 invariant).
  4. `procedure_retired` (curator-owned) drops a procedure from the projection; a later
     `procedure_learned` revives it.
  5. `reflect.py retire-procedures` proposes retirement for chronically-failing procedures
     (outcome_score below threshold after enough executions); dry-run writes nothing, --apply
     emits a curator `procedure_retired`. Idempotent.
  6. Back-compat: absent `proc_executed`/`procedure_retired` events ⇒ identical Phase 3
     projection; a pre-Phase-9 cache (schema_version=8) rebuilds rather than silently skips.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase9_adaptive_procedures.py
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

sys.path.insert(0, str(SYS_DIR))
import cognition  # noqa: E402
import runtime    # noqa: E402


def run(script, *args, home):
    env = dict(os.environ, CONTINUITY_HOME=str(home))
    return subprocess.run([sys.executable, script, *args], env=env,
                          capture_output=True, text=True, check=True)


def append(home, etype, **kv):
    args = [etype] + [f"{k}={v}" for k, v in kv.items()]
    return run(RUNTIME, "append", *args, home=home)


def db(home):
    conn = sqlite3.connect(Path(home) / "runtime" / "cognition.db")
    conn.row_factory = sqlite3.Row
    return conn


def proc(home, pid="prc_deploy"):
    conn = db(home)
    row = conn.execute(
        "SELECT * FROM procedures WHERE procedure_id=?", (pid,)).fetchone()
    conn.close()
    return dict(row) if row else None


class Phase9Adaptive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self._orig = (cognition.RUNTIME, cognition.EVENTS, cognition.COG_DB,
                      runtime.RUNTIME, runtime.EVENTS)
        cognition.RUNTIME = self.home / "runtime"
        cognition.EVENTS = cognition.RUNTIME / "events.jsonl"
        cognition.COG_DB = cognition.RUNTIME / "cognition.db"
        runtime.RUNTIME = self.home / "runtime"
        runtime.EVENTS = runtime.RUNTIME / "events.jsonl"
        append(self.home, "session_start", session="s1", project="demo")
        # curator distills a procedure (Phase 3 truth)
        append(self.home, "procedure_learned", agent="memory",
               procedure_id="prc_deploy", label="deploy flow",
               trigger="deploy", steps="run build", evidence="evt-1")
        run(COGNITION, "build", "--force", home=self.home)

    def tearDown(self):
        (cognition.RUNTIME, cognition.EVENTS, cognition.COG_DB,
         runtime.RUNTIME, runtime.EVENTS) = self._orig
        self.tmp.cleanup()

    # ── 1. proc_executed is known + episodic (not curator-owned) ─────────────
    def test_proc_executed_known_no_unknown_warning(self):
        out = append(self.home, "proc_executed",
                     procedure_id="prc_deploy", outcome="success").stderr
        self.assertNotIn("unknown", out.lower())

    def test_proc_executed_not_curator_owned(self):
        self.assertFalse(runtime.is_arbitration_violation("proc_executed", "coder"))
        append(self.home, "proc_executed", agent="coder",
               procedure_id="prc_deploy", outcome="success")
        self.assertEqual(cognition.arbitration_violations(), [])

    def test_procedure_retired_is_curator_owned(self):
        self.assertTrue(runtime.is_arbitration_violation("procedure_retired", "coder"))
        self.assertFalse(runtime.is_arbitration_violation("procedure_retired", "memory"))

    # ── 2. reinforcement weighting ───────────────────────────────────────────
    def test_success_raises_failure_lowers(self):
        base = proc(self.home)["outcome_score"]
        append(self.home, "proc_executed", procedure_id="prc_deploy", outcome="success")
        run(COGNITION, "build", "--force", home=self.home)
        up = proc(self.home)["outcome_score"]
        self.assertGreater(up, base)
        for _ in range(4):
            append(self.home, "proc_executed", procedure_id="prc_deploy", outcome="failure")
        run(COGNITION, "build", "--force", home=self.home)
        down = proc(self.home)["outcome_score"]
        self.assertLess(down, up)
        self.assertGreaterEqual(down, 0.0)

    def test_score_is_deterministic_replay(self):
        append(self.home, "proc_executed", procedure_id="prc_deploy", outcome="success")
        append(self.home, "proc_executed", procedure_id="prc_deploy", outcome="failure")
        run(COGNITION, "build", "--force", home=self.home)
        a = proc(self.home)["outcome_score"]
        (self.home / "runtime" / "cognition.db").unlink()
        run(COGNITION, "build", "--force", home=self.home)
        b = proc(self.home)["outcome_score"]
        self.assertEqual(a, b)

    def test_execution_of_unknown_procedure_ignored(self):
        append(self.home, "proc_executed", procedure_id="prc_ghost", outcome="success")
        run(COGNITION, "build", "--force", home=self.home)
        self.assertIsNone(proc(self.home, "prc_ghost"))

    # ── 3. executions count as uses; Phase 3 invariant preserved ─────────────
    def test_executions_increment_uses(self):
        before = proc(self.home)["uses"]
        self.assertGreaterEqual(before, 1)          # Phase 3 invariant
        append(self.home, "proc_executed", procedure_id="prc_deploy", outcome="success")
        append(self.home, "proc_executed", procedure_id="prc_deploy", outcome="failure")
        run(COGNITION, "build", "--force", home=self.home)
        self.assertEqual(proc(self.home)["uses"], before + 2)

    # ── 4. retire / revive ───────────────────────────────────────────────────
    def test_retire_drops_then_relearn_revives(self):
        append(self.home, "procedure_retired", agent="memory", procedure_id="prc_deploy")
        run(COGNITION, "build", "--force", home=self.home)
        self.assertIsNone(proc(self.home))
        append(self.home, "procedure_learned", agent="memory",
               procedure_id="prc_deploy", label="deploy flow v2",
               trigger="deploy", steps="run build", evidence="evt-2")
        run(COGNITION, "build", "--force", home=self.home)
        self.assertIsNotNone(proc(self.home))

    # ── 5. consolidation retire-procedures pass ──────────────────────────────
    def test_retire_pass_dry_run_then_apply(self):
        for _ in range(6):
            append(self.home, "proc_executed", procedure_id="prc_deploy", outcome="failure")
        run(COGNITION, "build", "--force", home=self.home)
        events_before = (self.home / "runtime" / "events.jsonl").read_text()
        out = run(REFLECT, "retire-procedures", home=self.home).stdout
        self.assertIn("prc_deploy", out)
        # dry-run must not write truth
        self.assertEqual(events_before,
                         (self.home / "runtime" / "events.jsonl").read_text())
        run(REFLECT, "retire-procedures", "--apply", home=self.home)
        last = (self.home / "runtime" / "events.jsonl").read_text().strip().splitlines()[-1]
        self.assertIn("procedure_retired", last)
        self.assertIn("memory", last)               # curator-owned
        run(COGNITION, "build", "--force", home=self.home)
        self.assertIsNone(proc(self.home))          # dropped from projection

    def test_healthy_procedure_not_retired(self):
        for _ in range(6):
            append(self.home, "proc_executed", procedure_id="prc_deploy", outcome="success")
        run(COGNITION, "build", "--force", home=self.home)
        out = run(REFLECT, "retire-procedures", home=self.home).stdout
        self.assertIn("no chronically-failing procedures", out)

    # ── 6. back-compat ───────────────────────────────────────────────────────
    def test_no_executions_matches_phase3(self):
        # the baseline procedure (no executions) projects exactly as Phase 3 did
        p = proc(self.home)
        self.assertEqual(p["uses"], 1)
        self.assertIsNotNone(p["outcome_score"])

    def test_pre_phase9_cache_rebuilds(self):
        conn = db(self.home)
        conn.execute("UPDATE meta SET value='8' WHERE key='schema_version'")
        conn.commit()
        conn.close()
        out = run(COGNITION, "build", home=self.home).stdout   # no --force
        self.assertIn("built", out)
        conn = db(self.home)
        self.assertEqual(conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],
            str(cognition.SCHEMA_VERSION))
        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
