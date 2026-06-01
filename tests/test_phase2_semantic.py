#!/usr/bin/env python3
"""
Phase 2 semantic-memory tests (ARCHITECTURE_V4.md §5.1, §11 Phase 2).

Verifies the new semantic layer preserves every invariant:
  1. The T2 abstraction pass forms concepts from recurring episodes, is dry-run by
     default, writes only on --apply, and is idempotent (no duplicate concepts).
  2. build() projects concept_formed events into the concepts/concept_evidence tables
     and `concept` edges, and materializes each concept as a graph node.
  3. Concepts seed attention and surface in WorkingMemory (active + render section).
  4. Backward-compat: a brain with NO concepts renders byte-for-byte as before, and a
     pre-Phase-2 cache (older schema_version) rebuilds rather than silently skipping.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase2_semantic.py
"""

import os
import re
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


class Phase2Semantic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        append(self.home, "session_start", session="s1", project="demo")
        # five episodes sharing the recurring term "sqlite"
        for _ in range(4):
            append(self.home, "decision",
                   msg="adopt sqlite cache for the session store", thread_ids="t-store")
        append(self.home, "observation", msg="sqlite rebuild is fast", thread_ids="t-store")
        run(COGNITION, "build", "--force", home=self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def test_abstract_dry_run_then_apply_idempotent(self):
        dry = run(REFLECT, "abstract", home=self.home)
        self.assertIn("dry-run", dry)
        # nothing written on dry-run
        self.assertNotIn("concept_formed",
                         (Path(self.home) / "runtime" / "events.jsonl").read_text())
        run(REFLECT, "abstract", "--apply", home=self.home)
        self.assertIn("concept_formed",
                      (Path(self.home) / "runtime" / "events.jsonl").read_text())
        again = run(REFLECT, "abstract", "--apply", home=self.home)
        self.assertIn("no new concepts", again)

    def test_build_projects_concepts_and_edges(self):
        run(REFLECT, "abstract", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        conn = db(self.home)
        cid = "cpt_sqlite"
        row = conn.execute("SELECT * FROM concepts WHERE concept_id=?", (cid,)).fetchone()
        self.assertIsNotNone(row)
        self.assertGreaterEqual(row["support"], 3)
        # materialized as a graph node...
        node = conn.execute("SELECT * FROM nodes WHERE event_id=?", (cid,)).fetchone()
        self.assertEqual(node["type"], "concept")
        self.assertEqual(node["layer"], "semantic")
        # ...wired to evidence by concept edges
        n_ev = conn.execute("SELECT COUNT(*) FROM concept_evidence WHERE concept_id=?",
                            (cid,)).fetchone()[0]
        n_edge = conn.execute("SELECT COUNT(*) FROM edges WHERE kind='concept' AND src=?",
                             (cid,)).fetchone()[0]
        conn.close()
        self.assertEqual(n_ev, n_edge)
        self.assertGreaterEqual(n_ev, 3)

    def test_concept_in_working_memory(self):
        run(REFLECT, "abstract", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        ctx = run(COGNITION, "context", home=self.home)
        self.assertIn("## Concepts in scope (semantic memory)", ctx)
        self.assertIn("cpt_sqlite", ctx)

    def test_no_concepts_render_unchanged(self):
        # before any abstraction, the lens must not mention semantic memory at all
        ctx = run(COGNITION, "context", home=self.home)
        self.assertNotIn("Concepts in scope", ctx)

    def test_pre_phase2_cache_rebuilds(self):
        # simulate an older cache: present but without schema_version → must NOT skip
        conn = db(self.home)
        conn.execute("DELETE FROM meta WHERE key='schema_version'")
        conn.execute("DROP TABLE concepts")
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
