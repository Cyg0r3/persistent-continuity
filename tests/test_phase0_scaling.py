#!/usr/bin/env python3
"""
Phase 0 scaling-foundation tests (ARCHITECTURE_V4.md §11 Phase 0).

Verifies the two performance fixes preserve correctness:
  1. Persisted sequence counter (runtime/seq) gives sequential, gap-free event_ids
     without re-reading the whole log — and self-heals if the counter is lost.
  2. Incremental build watermark skips the rebuild when the log hasn't grown, rebuilds
     when it has, --force always rebuilds, and a pre-v4 cache (no meta table) rebuilds.
  3. The late-loop_closed correctness invariant still holds across a watermark rebuild.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase0_scaling.py
"""

import os
import sys
import json
import sqlite3
import unittest
import subprocess
import tempfile
from pathlib import Path

SYS_DIR = (Path(__file__).resolve().parent.parent
           / "plugins" / "persistent-continuity" / "system")
RUNTIME = str(SYS_DIR / "runtime.py")
COGNITION = str(SYS_DIR / "cognition.py")


def run(script, *args, home):
    env = dict(os.environ, CONTINUITY_HOME=str(home))
    return subprocess.run([sys.executable, script, *args], env=env,
                          capture_output=True, text=True, check=True).stdout


def append(home, etype, **kv):
    args = [etype] + [f"{k}={v}" for k, v in kv.items()]
    out = run(RUNTIME, "append", *args, home=home)
    # the JSON line is the last non-note line of stdout
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"no event JSON in output: {out!r}")


class Phase0Scaling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.seq = self.home / "runtime" / "seq"
        self.db = self.home / "runtime" / "cognition.db"

    def tearDown(self):
        self.tmp.cleanup()

    # ── sequence counter ─────────────────────────────────────────────────────
    def test_sequential_ids_and_seq_file(self):
        self.assertEqual(append(self.home, "objective", msg="a")["event_id"], "evt_1")
        self.assertEqual(append(self.home, "decision", msg="b")["event_id"], "evt_2")
        self.assertEqual(append(self.home, "artifact", path="x.py")["event_id"], "evt_3")
        self.assertEqual(self.seq.read_text().strip(), "3")

    def test_seq_self_heals_when_deleted(self):
        append(self.home, "objective", msg="a")
        append(self.home, "decision", msg="b")
        self.seq.unlink()                       # lose the cache
        # next id must still be 3 (recovered by counting the log), not 1
        self.assertEqual(append(self.home, "artifact", path="x")["event_id"], "evt_3")

    def test_seq_self_heals_when_stale_low(self):
        append(self.home, "objective", msg="a")
        append(self.home, "decision", msg="b")
        self.seq.write_text("1")                # corrupt: below true count
        # must not collide with evt_2 — self-heal pushes past the real count
        self.assertEqual(append(self.home, "artifact", path="x")["event_id"], "evt_3")

    # ── watermark build ──────────────────────────────────────────────────────
    def test_watermark_skip_and_rebuild(self):
        append(self.home, "objective", msg="a")
        self.assertIn("built", run(COGNITION, "build", home=self.home))
        self.assertIn("skipped", run(COGNITION, "build", home=self.home))
        # log grows -> must rebuild, not skip
        append(self.home, "decision", msg="b")
        self.assertNotIn("skipped", run(COGNITION, "build", home=self.home))

    def test_force_always_rebuilds(self):
        append(self.home, "objective", msg="a")
        run(COGNITION, "build", home=self.home)
        self.assertNotIn("skipped",
                         run(COGNITION, "build", "--force", home=self.home))

    def test_pre_v4_cache_rebuilds(self):
        append(self.home, "objective", msg="a")
        run(COGNITION, "build", home=self.home)
        conn = sqlite3.connect(self.db)         # simulate an old cache: drop meta
        conn.execute("DROP TABLE meta")
        conn.commit(); conn.close()
        self.assertNotIn("skipped", run(COGNITION, "build", home=self.home))

    # ── correctness across a watermark rebuild ───────────────────────────────
    def test_late_loop_closed_flips_unresolved(self):
        append(self.home, "open_loop", id="L1", msg="do thing")
        run(COGNITION, "build", home=self.home)             # built with loop open
        append(self.home, "loop_closed", id="L1", resolution="done")
        run(COGNITION, "build", home=self.home)             # log grew -> rebuild
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            "SELECT unresolved FROM nodes WHERE type='open_loop'").fetchone()
        conn.close()
        self.assertEqual(row[0], 0, "late loop_closed must clear unresolved on rebuild")


if __name__ == "__main__":
    unittest.main(verbosity=2)
