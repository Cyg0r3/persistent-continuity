#!/usr/bin/env python3
"""
Phase 5 consolidation-pipeline + snapshot tests (ARCHITECTURE_V4.md §9, §11 Phase 5).

Verifies the consolidation layer preserves every invariant:
  1. `cognition snapshot` is dry-run by default; --apply emits a `snapshot` event, writes a
     snapshot blob, and records meta.snapshot_seq.
  2. Compaction: episodes behind the snapshot that no active thread cites go COLD; open
     loops and active-thread members stay hot. Cold nodes are dropped from attention SEEDING
     (the active working set stays bounded) but remain in the graph (replayable).
  3. Stale-belief pruning emits `concept_retired` for concepts whose evidence all went cold
     and which no active thread cites; the rebuild drops them (a later form revives).
  4. Salience is recency-recalibrated: a concept on fresh evidence outranks one on cold
     evidence (derived path), and an explicit event salience still wins.
  5. The snapshot blob is a CACHE — deleting cognition.db and rebuilding reproduces the same
     cold set; and a pre-Phase-5 cache (older schema_version) rebuilds rather than skipping.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase5_consolidation.py
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
import cognition  # noqa: E402  (pure helpers operate on an explicit conn / events list)


def run(script, *args, home):
    env = dict(os.environ, CONTINUITY_HOME=str(home))
    return subprocess.run([sys.executable, script, *args], env=env,
                          capture_output=True, text=True, check=True).stdout


def append(home, etype, **kv):
    args = [etype] + [f"{k}={v}" for k, v in kv.items()]
    run(RUNTIME, "append", *args, home=home)


def events_text(home):
    return (Path(home) / "runtime" / "events.jsonl").read_text()


def db(home):
    conn = sqlite3.connect(Path(home) / "runtime" / "cognition.db")
    conn.row_factory = sqlite3.Row
    return conn


def node_cold(home, eid):
    conn = db(home)
    row = conn.execute("SELECT cold FROM nodes WHERE event_id=?", (eid,)).fetchone()
    conn.close()
    return row["cold"] if row else None


class Phase5Consolidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        append(self.home, "session_start", session="s1", project="demo")
        # a loose, threadless episode (compacts behind a snapshot) ...
        append(self.home, "decision", event_id="d1", msg="alpha orphan widget retry logic")
        # ... an active thread (stays hot) ... and an open loop (never compacted)
        append(self.home, "objective", msg="ship the beta feature", thread_ids="t1")
        append(self.home, "open_loop", id="ol1", msg="finish gamma migration")
        run(COGNITION, "build", "--force", home=self.home)

    def tearDown(self):
        self.tmp.cleanup()

    # ── 1. snapshot dry-run vs apply ────────────────────────────────────────
    def test_snapshot_dry_run_then_apply(self):
        dry = run(COGNITION, "snapshot", home=self.home)
        self.assertIn("dry-run", dry)
        self.assertNotIn('"type":"snapshot"', events_text(self.home).replace(" ", ""))
        out = run(COGNITION, "snapshot", "--apply", home=self.home)
        self.assertIn("Snapshot taken", out)
        self.assertIn("snapshot", events_text(self.home))
        # blob written + meta.snapshot_seq recorded
        snaps = list((self.home / "runtime" / "snapshots").glob("snap_*.json"))
        self.assertTrue(snaps)
        conn = db(self.home)
        seq = conn.execute("SELECT value FROM meta WHERE key='snapshot_seq'").fetchone()[0]
        conn.close()
        self.assertEqual(int(seq), 4)   # the 4 events that existed before the snapshot event

    # ── 2. compaction (cold flags + seed exclusion) ─────────────────────────
    def test_compaction_marks_cold_and_preserves_hot(self):
        run(COGNITION, "snapshot", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        self.assertEqual(node_cold(self.home, "d1"), 1)        # loose old episode -> cold
        # the active-thread objective and the open loop stay hot
        conn = db(self.home)
        objective = conn.execute(
            "SELECT cold FROM nodes WHERE type='objective'").fetchone()["cold"]
        loop = conn.execute(
            "SELECT cold FROM nodes WHERE type='open_loop'").fetchone()["cold"]
        conn.close()
        self.assertEqual(objective, 0)
        self.assertEqual(loop, 0)

    def test_cold_excluded_from_seed_but_in_graph(self):
        conn = db(self.home)
        before = cognition._seed(conn, "alpha orphan widget")
        conn.close()
        self.assertIn("d1", before)                            # seeded before compaction
        run(COGNITION, "snapshot", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        conn = db(self.home)
        after = cognition._seed(conn, "alpha orphan widget")
        still_a_node = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE event_id='d1'").fetchone()[0]
        conn.close()
        self.assertNotIn("d1", after)                          # dropped from the seed
        self.assertEqual(still_a_node, 1)                      # but still in the graph

    def test_no_snapshot_nothing_cold(self):
        conn = db(self.home)
        n_cold = conn.execute("SELECT COUNT(*) FROM nodes WHERE cold=1").fetchone()[0]
        seq = conn.execute("SELECT value FROM meta WHERE key='snapshot_seq'").fetchone()[0]
        conn.close()
        self.assertEqual(n_cold, 0)                            # backward-compatible default
        self.assertEqual(int(seq), 0)

    # ── 3. stale-belief pruning (concept_retired) ───────────────────────────
    def test_prune_retires_stale_concept(self):
        # a concept whose only evidence is the loose episode d1 (it will go cold)
        append(self.home, "concept_formed", agent="memory", concept_id="cpt_alpha",
               summary="alpha widget pattern", evidence="d1", support=3)
        run(COGNITION, "snapshot", "--apply", home=self.home)   # d1 -> cold
        run(COGNITION, "build", "--force", home=self.home)
        # before pruning the concept is materialized
        conn = db(self.home)
        self.assertIsNotNone(conn.execute(
            "SELECT 1 FROM concepts WHERE concept_id='cpt_alpha'").fetchone())
        conn.close()
        dry = run(REFLECT, "prune", home=self.home)
        self.assertIn("cpt_alpha", dry)
        run(REFLECT, "prune", "--apply", home=self.home)
        self.assertIn("concept_retired", events_text(self.home))
        run(COGNITION, "build", "--force", home=self.home)
        conn = db(self.home)
        gone = conn.execute(
            "SELECT 1 FROM concepts WHERE concept_id='cpt_alpha'").fetchone()
        node = conn.execute(
            "SELECT 1 FROM nodes WHERE event_id='cpt_alpha'").fetchone()
        conn.close()
        self.assertIsNone(gone)                                 # dropped from projection
        self.assertIsNone(node)

    def test_retired_concept_revived_by_later_formation(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(cognition.SCHEMA)
        events = [
            {"type": "concept_formed", "concept_id": "cpt_x", "summary": "x", "support": 3},
            {"type": "concept_retired", "concept_id": "cpt_x"},
            {"type": "concept_formed", "concept_id": "cpt_x", "summary": "x again",
             "support": 3},
        ]
        cognition._build_concepts(conn, events)
        self.assertIsNotNone(conn.execute(
            "SELECT 1 FROM concepts WHERE concept_id='cpt_x'").fetchone())
        # retirement with no later formation drops it
        conn2 = sqlite3.connect(":memory:")
        conn2.executescript(cognition.SCHEMA)
        cognition._build_concepts(conn2, events[:2])
        self.assertIsNone(conn2.execute(
            "SELECT 1 FROM concepts WHERE concept_id='cpt_x'").fetchone())

    # ── 4. salience recency recalibration ───────────────────────────────────
    def test_salience_recalibrated_by_recency(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(cognition.SCHEMA)
        for eid, t in (("e-fresh", cognition.now_iso()),
                       ("e-old", "2000-01-01T00:00:00+00:00")):
            conn.execute(
                "INSERT INTO nodes(event_id,seq,t,type,layer,body,importance,unresolved,"
                "reinforced) VALUES (?,?,?,?,?,?,?,?,?)",
                (eid, 1, t, "decision", "episodic", "x", 0.5, 0, 0))
        events = [
            {"type": "concept_formed", "concept_id": "cpt_fresh", "summary": "f",
             "evidence": ["e-fresh"], "support": 3},
            {"type": "concept_formed", "concept_id": "cpt_old", "summary": "o",
             "evidence": ["e-old"], "support": 3},
        ]
        cognition._build_concepts(conn, events)
        fresh = conn.execute(
            "SELECT salience FROM concepts WHERE concept_id='cpt_fresh'").fetchone()[0]
        old = conn.execute(
            "SELECT salience FROM concepts WHERE concept_id='cpt_old'").fetchone()[0]
        conn.close()
        self.assertGreater(fresh, old)                          # cold meaning faded

    def test_explicit_salience_wins(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(cognition.SCHEMA)
        cognition._build_concepts(conn, [
            {"type": "concept_formed", "concept_id": "cpt_e", "summary": "e",
             "support": 3, "salience": 0.99}])
        self.assertAlmostEqual(conn.execute(
            "SELECT salience FROM concepts WHERE concept_id='cpt_e'").fetchone()[0], 0.99)
        conn.close()

    # ── 5. snapshot blob is a cache; pre-Phase-5 cache rebuilds ─────────────
    def test_snapshot_is_cache_rebuild_reproduces_cold(self):
        run(COGNITION, "snapshot", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        conn = db(self.home)
        cold1 = sorted(r[0] for r in conn.execute(
            "SELECT event_id FROM nodes WHERE cold=1"))
        conn.close()
        (self.home / "runtime" / "cognition.db").unlink()       # drop the cache
        run(COGNITION, "build", "--force", home=self.home)      # rebuild from the log
        conn = db(self.home)
        cold2 = sorted(r[0] for r in conn.execute(
            "SELECT event_id FROM nodes WHERE cold=1"))
        conn.close()
        self.assertEqual(cold1, cold2)                          # deterministic projection

    def test_pre_phase5_cache_rebuilds(self):
        conn = db(self.home)
        conn.execute("DELETE FROM meta WHERE key='schema_version'")
        conn.execute("ALTER TABLE nodes DROP COLUMN cold")      # simulate older schema
        conn.commit()
        conn.close()
        out = run(COGNITION, "build", home=self.home)           # no --force
        self.assertIn("built", out)                             # rebuilt, not "skipped"
        conn = db(self.home)
        self.assertEqual(conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],
            str(cognition.SCHEMA_VERSION))
        conn.close()

    def test_consolidate_pipeline_dry_run(self):
        out = run(REFLECT, "consolidate", home=self.home)
        for stage in ("abstraction", "procedure learning", "stale-belief pruning",
                      "snapshot / compaction"):
            self.assertIn(stage, out)
        self.assertNotIn("snapshot", events_text(self.home))    # dry-run writes nothing


if __name__ == "__main__":
    unittest.main(verbosity=2)
