#!/usr/bin/env python3
"""
Phase 8 pattern-recognition tests (ARCHITECTURE_V4.md §13, §11 Phase 8).

Verifies patterns become first-class memory objects derived from the one log, preserving
every V4 invariant:
  1. The `reflect.py mine` pass detects recurring patterns from the projected graph —
     reasoning motifs, failure modes, successful execution paths, and bottlenecks — and is
     a dry-run until --apply (append-only; emits curator-owned `pattern_detected`).
  2. `cognition._build_patterns` projects `pattern_detected` into the patterns/pattern_evidence
     tables AND materializes each as a graph node (layer='pattern') wired to its evidence by
     `pattern` edges, so a live context surfaces the pattern it is repeating (seeded by
     PATTERN_SEED x confidence).
  3. Mining is idempotent (existing pattern_ids skipped) and entropy-capped.
  4. Back-compat: a log with no pattern_detected events ⇒ empty pattern layer and an otherwise
     identical projection; an absent patterns table on a pre-Phase-8 cache forces a rebuild
     (SCHEMA_VERSION bump), never a silent skip.
  5. Curator arbitration: `pattern_detected` is curator-owned, so emitting it under a named
     non-curator agent is flagged by `cognition audit` (append-only ⇒ stored, never rejected).
  6. The pattern projection is deterministic (same log ⇒ same patterns).

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase8_patterns.py
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


def run(script, *args, home, check=True):
    env = dict(os.environ, CONTINUITY_HOME=str(home))
    return subprocess.run([sys.executable, script, *args], env=env,
                          capture_output=True, text=True, check=check)


def append(home, etype, **kv):
    args = [etype] + [f"{k}={v}" for k, v in kv.items()]
    return run(RUNTIME, "append", *args, home=home)


def db(home):
    conn = sqlite3.connect(Path(home) / "runtime" / "cognition.db")
    conn.row_factory = sqlite3.Row
    return conn


class Phase8Patterns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        # in-process helpers (arbitration_violations, _build_patterns) read the log via
        # runtime, so BOTH modules' paths must target the per-test home (mirrors Phase 7).
        self._orig = (cognition.RUNTIME, cognition.EVENTS, cognition.COG_DB,
                      runtime.RUNTIME, runtime.EVENTS)
        cognition.RUNTIME = self.home / "runtime"
        cognition.EVENTS = cognition.RUNTIME / "events.jsonl"
        cognition.COG_DB = cognition.RUNTIME / "cognition.db"
        runtime.RUNTIME = self.home / "runtime"
        runtime.EVENTS = runtime.RUNTIME / "events.jsonl"

        append(self.home, "session_start", session="s1", project="demo")
        # two SUCCESSFUL threads sharing one step signature -> success pattern + reasoning motif
        for t in ("thr_a", "thr_b"):
            append(self.home, "objective", msg="fix the auth bug", thread_ids=t)
            append(self.home, "command", msg="run pytest suite", thread_ids=t)
            append(self.home, "artifact", msg="patch auth module", thread_ids=t)
            append(self.home, "loop_closed", msg="auth fixed", thread_ids=t)
        # two threads sharing one failure term -> failure pattern
        append(self.home, "error", msg="timeout connecting database", thread_ids="thr_c")
        append(self.home, "error", msg="timeout connecting database", thread_ids="thr_d")
        run(COGNITION, "build", "--force", home=self.home)

    def tearDown(self):
        (cognition.RUNTIME, cognition.EVENTS, cognition.COG_DB,
         runtime.RUNTIME, runtime.EVENTS) = self._orig
        self.tmp.cleanup()

    # ── 1. mining: dry-run vs apply ──────────────────────────────────────────
    def test_mine_dry_run_writes_nothing(self):
        out = run(REFLECT, "mine", home=self.home).stdout
        self.assertIn("dry-run", out)
        # no pattern_detected event was appended
        log = (self.home / "runtime" / "events.jsonl").read_text()
        self.assertNotIn("pattern_detected", log)

    def test_mine_apply_emits_curator_patterns(self):
        out = run(REFLECT, "mine", "--apply", home=self.home).stdout
        self.assertIn("detected", out)
        log = (self.home / "runtime" / "events.jsonl").read_text()
        self.assertIn("pattern_detected", log)
        self.assertIn("pat_success_", log)
        self.assertIn("pat_fail_", log)
        # curator-owned: emitted under agent="memory"
        self.assertIn('"agent":"memory"', log)

    # ── 2. projection: tables, nodes, edges, seeding ─────────────────────────
    def test_build_projects_patterns(self):
        run(REFLECT, "mine", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        conn = db(self.home)
        kinds = {r["kind"] for r in conn.execute("SELECT kind FROM patterns")}
        self.assertIn("success", kinds)
        self.assertIn("failure", kinds)
        # each pattern is a graph node (layer='pattern') wired by `pattern` edges
        npat = conn.execute("SELECT COUNT(*) FROM nodes WHERE layer='pattern'").fetchone()[0]
        self.assertGreaterEqual(npat, 2)
        edges = conn.execute("SELECT COUNT(*) FROM edges WHERE kind='pattern'").fetchone()[0]
        self.assertGreater(edges, 0)
        ev = conn.execute("SELECT COUNT(*) FROM pattern_evidence").fetchone()[0]
        self.assertGreater(ev, 0)
        conn.close()

    def test_pattern_surfaces_in_attention(self):
        run(REFLECT, "mine", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        out = run(COGNITION, "attend", "timeout database", "-n", "12",
                  home=self.home).stdout
        # the failure pattern the live context is repeating is lifted into the window
        self.assertIn("pat_fail_", out)

    # ── 3. idempotency ───────────────────────────────────────────────────────
    def test_mine_is_idempotent(self):
        run(REFLECT, "mine", "--apply", home=self.home)
        out = run(REFLECT, "mine", "--apply", home=self.home).stdout
        self.assertIn("no new patterns", out)

    # ── 4. back-compat ───────────────────────────────────────────────────────
    def test_no_patterns_means_empty_layer(self):
        # setUp never mined, so the pattern layer is empty but the graph is otherwise normal
        conn = db(self.home)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0], 0)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM nodes WHERE layer='pattern'").fetchone()[0], 0)
        # episodic nodes still present
        self.assertGreater(conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0], 0)
        conn.close()

    def test_pre_phase8_cache_rebuilds(self):
        # simulate an older cache by stamping an older schema_version: the watermark must
        # treat it as stale and rebuild (never silently skip), so the patterns table exists.
        conn = db(self.home)
        conn.execute("UPDATE meta SET value='7' WHERE key='schema_version'")
        conn.execute("DROP TABLE patterns")
        conn.commit()
        conn.close()
        run(COGNITION, "build", home=self.home)   # no --force: relies on schema gate
        conn = db(self.home)
        # rebuilt under the current schema => patterns table is back and version current
        self.assertEqual(
            conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],
            str(cognition.SCHEMA_VERSION))
        conn.execute("SELECT COUNT(*) FROM patterns")  # table exists again (no error)
        conn.close()

    # ── 5. curator arbitration ───────────────────────────────────────────────
    def test_pattern_detected_is_curator_owned(self):
        self.assertIn("pattern_detected", runtime.CURATOR_TYPES)
        self.assertIn("pattern_detected", cognition.CURATOR_TYPES)
        # a pattern_detected under a NAMED non-curator agent is an arbitration violation
        self.assertTrue(runtime.is_arbitration_violation("pattern_detected", "coder"))
        self.assertFalse(runtime.is_arbitration_violation("pattern_detected", "memory"))
        self.assertFalse(runtime.is_arbitration_violation("pattern_detected", ""))

    def test_audit_flags_non_curator_pattern(self):
        append(self.home, "pattern_detected", agent="coder",
               pattern_id="pat_rogue", kind="reasoning", confidence="0.9")
        out = run(COGNITION, "audit", home=self.home).stdout
        self.assertIn("pattern_detected", out)

    # ── 6. determinism ───────────────────────────────────────────────────────
    def test_pattern_projection_is_deterministic(self):
        run(REFLECT, "mine", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        a = sorted(r["pattern_id"] for r in db(self.home).execute(
            "SELECT pattern_id FROM patterns"))
        run(COGNITION, "build", "--force", home=self.home)
        b = sorted(r["pattern_id"] for r in db(self.home).execute(
            "SELECT pattern_id FROM patterns"))
        self.assertEqual(a, b)
        self.assertTrue(a)

    def test_known_type_no_unknown_warning(self):
        self.assertIn("pattern_detected", runtime.KNOWN_TYPES)
        res = append(self.home, "pattern_detected", agent="memory",
                     pattern_id="pat_x", kind="success", confidence="0.8")
        self.assertNotIn("unknown type", res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
