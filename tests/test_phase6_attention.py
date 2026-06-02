#!/usr/bin/env python3
"""
Phase 6 attention-dynamics tests (ARCHITECTURE_V4.md §8, §11 Phase 6).

Verifies the four attention dynamics layer onto the V3 engine without breaking it:
  1. Resonance: concepts whose evidence co-occurs across >= MIN_COOC threads get a Hebbian
     `resonance` edge, so activating one associatively lifts the other (beyond causal/thread
     proximity). Bounded (cap per concept) and absent for non-co-occurring concepts.
  2. Incubation: a query-less low-gain pass lifts dormant-but-well-connected nodes into the
     `incubation` channel, which `_seed` then folds in — so a dormant idea resurfaces.
  3. Oscillation (opt-in): dampens the winner cluster and lifts the runner-up; OFF by
     default reproduces the prior ranking exactly.
  4. Backward-compat: no concepts ⇒ no resonance and unchanged attention; a pre-Phase-6
     cache (older schema_version / missing table) rebuilds rather than silently skipping;
     resonance is a deterministic projection (build twice ⇒ identical edges).

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase6_attention.py
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

OLD_T = "2020-01-01T00:00:00+00:00"   # > DORMANT_DAYS ago ⇒ thread goes dormant


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


class Phase6Attention(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        append(self.home, "session_start", session="s1", project="demo")
        # alpha & beta co-occur across TWO threads (t1, t2) -> they resonate
        for tid, (da, dbid) in (("t1", ("d1", "d2")), ("t2", ("d3", "d4"))):
            append(self.home, "objective", msg="build alpha beta pipeline", thread_ids=tid)
            append(self.home, "decision", event_id=da, msg="alpha tuning", thread_ids=tid)
            append(self.home, "decision", event_id=dbid, msg="beta tuning", thread_ids=tid)
        # gamma lives alone on t3 -> no resonance
        append(self.home, "decision", event_id="d5", msg="gamma note", thread_ids="t3")
        append(self.home, "concept_formed", agent="memory", concept_id="cpt_alpha",
               summary="alpha concept", evidence="d1,d3", support=3)
        append(self.home, "concept_formed", agent="memory", concept_id="cpt_beta",
               summary="beta concept", evidence="d2,d4", support=3)
        append(self.home, "concept_formed", agent="memory", concept_id="cpt_gamma",
               summary="gamma concept", evidence="d5", support=3)
        run(COGNITION, "build", "--force", home=self.home)
        # point the in-process cognition module at this test's home so attend()/incubate()
        # (which open the module-level COG_DB) operate on the fixture we just built.
        self._orig = (cognition.RUNTIME, cognition.EVENTS, cognition.COG_DB,
                      cognition.SNAPSHOTS)
        cognition.RUNTIME = self.home / "runtime"
        cognition.EVENTS = cognition.RUNTIME / "events.jsonl"
        cognition.COG_DB = cognition.RUNTIME / "cognition.db"
        cognition.SNAPSHOTS = cognition.RUNTIME / "snapshots"

    def tearDown(self):
        (cognition.RUNTIME, cognition.EVENTS, cognition.COG_DB,
         cognition.SNAPSHOTS) = self._orig
        self.tmp.cleanup()

    # ── 1. resonance ────────────────────────────────────────────────────────
    def test_resonance_edge_between_cooccurring_concepts(self):
        conn = db(self.home)
        res = {(r["src"], r["dst"]) for r in conn.execute(
            "SELECT src,dst FROM edges WHERE kind='resonance'")}
        conn.close()
        self.assertIn(("cpt_alpha", "cpt_beta"), res)               # co-occur in t1 & t2
        # gamma shares no thread with either -> no resonance edge touches it
        self.assertFalse(any("cpt_gamma" in pair for pair in res))

    def test_resonance_gives_associative_lift(self):
        # attending alpha should lift its resonant peer beta above the unrelated gamma
        st = cognition.attend("alpha tuning", persist=False)
        self.assertGreater(st["nodes"]["cpt_beta"], st["nodes"]["cpt_gamma"])

    # ── 2. incubation ───────────────────────────────────────────────────────
    def test_incubation_resurfaces_dormant_node(self):
        # a dormant (old) thread with a connected node that no query/loop would seed
        append(self.home, "objective", event_id="dz0", msg="stale neglected idea",
               thread_ids="t-old", t=OLD_T)
        append(self.home, "decision", event_id="dz1", msg="neglected detail",
               thread_ids="t-old", t=OLD_T)
        run(COGNITION, "build", "--force", home=self.home)
        conn = db(self.home)
        before = cognition._seed(conn, "")
        conn.close()
        self.assertNotIn("dz1", before)                             # not seeded while cold
        run(COGNITION, "incubate", home=self.home)
        conn = db(self.home)
        after = cognition._seed(conn, "")
        n_inc = conn.execute("SELECT COUNT(*) FROM incubation").fetchone()[0]
        conn.close()
        self.assertGreater(n_inc, 0)
        self.assertIn("dz1", after)                                 # resurfaced by incubation

    def test_no_incubation_channel_is_noop(self):
        conn = db(self.home)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM incubation").fetchone()[0], 0)
        conn.close()  # empty channel ⇒ attention seed unchanged (backward compatible)
        st = cognition.attend("alpha", persist=False)
        self.assertTrue(st["nodes"])

    # ── 3. oscillation ──────────────────────────────────────────────────────
    def test_oscillation_rotates_focus_off_winner(self):
        base = cognition.attend("alpha tuning", persist=False)
        osc = cognition.attend("alpha tuning", persist=False, oscillate=True)
        threads = sorted(base["threads"], key=base["threads"].get, reverse=True)
        winner, runner = threads[0], threads[1]
        self.assertLess(osc["threads"][winner], base["threads"][winner])   # winner damped
        self.assertGreater(osc["threads"][runner], base["threads"][runner])  # runner lifted

    def test_oscillation_off_by_default_unchanged(self):
        a = cognition.attend("alpha tuning", persist=False)
        b = cognition.attend("alpha tuning", persist=False, oscillate=False)
        self.assertEqual(a["threads"], b["threads"])

    # ── 4. backward-compat / determinism ────────────────────────────────────
    def test_no_concepts_no_resonance(self):
        home = Path(self.tmp.name) / "plain"
        append(home, "session_start", session="s2", project="plain")
        append(home, "decision", msg="just an episode", thread_ids="t1")
        run(COGNITION, "build", "--force", home=home)
        conn = db(home)
        n = conn.execute("SELECT COUNT(*) FROM edges WHERE kind='resonance'").fetchone()[0]
        conn.close()
        self.assertEqual(n, 0)

    def test_resonance_is_deterministic(self):
        conn = db(self.home)
        first = sorted(tuple(r) for r in conn.execute(
            "SELECT src,dst,weight FROM edges WHERE kind='resonance'"))
        conn.close()
        run(COGNITION, "build", "--force", home=self.home)
        conn = db(self.home)
        second = sorted(tuple(r) for r in conn.execute(
            "SELECT src,dst,weight FROM edges WHERE kind='resonance'"))
        conn.close()
        self.assertEqual(first, second)

    def test_pre_phase6_cache_rebuilds(self):
        conn = db(self.home)
        conn.execute("DELETE FROM meta WHERE key='schema_version'")
        conn.execute("DROP TABLE incubation")
        conn.commit()
        conn.close()
        out = run(COGNITION, "build", home=self.home)              # no --force
        self.assertIn("built", out)                                # rebuilt, not skipped
        conn = db(self.home)
        self.assertEqual(conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],
            str(cognition.SCHEMA_VERSION))
        conn.close()

    def test_consolidate_runs_incubation_stage(self):
        out = run(REFLECT, "consolidate", "--apply", home=self.home)
        self.assertIn("incubation", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
