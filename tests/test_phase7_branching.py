#!/usr/bin/env python3
"""
Phase 7 branching/versioning + multi-agent-formalization tests (ARCHITECTURE_V4.md §3.3,
§10, §11 Phase 7).

Verifies the final V4 phase preserves every invariant:
  1. Branching is a labeled SUBSET of the one log, never a fork: an unmerged branch's
     events are excluded from the default `main` reconstruction (a what-if never pollutes
     the real timeline), yet `reconstruct(branch)` replays main + that branch in isolation
     without touching runtime/cognition.db.
  2. `branch_merge` folds a branch back into the mainline, so a later `build()` includes it.
  3. Absent `branch` ⇒ main (back-compat): a log with no branch fields projects exactly as
     before, and `runtime.append` drops a redundant branch="main".
  4. Agent-local working memory (`WorkingMemory.for_agent`) is a first-class per-agent lens
     over the ONE shared graph, seeded by the agent's role + its own attention lane.
  5. Curator arbitration: a consolidation-type event under a named non-curator agent is
     flagged by the audit (append-only ⇒ stored, never rejected); under the curator (or
     system, no agent) it is clean.
  6. The branch-filtered projection is deterministic and the schema_version bump forces a
     pre-Phase-7 cache to rebuild rather than silently skip.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase7_branching.py
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


def events_text(home):
    return (Path(home) / "runtime" / "events.jsonl").read_text()


def db(home):
    conn = sqlite3.connect(Path(home) / "runtime" / "cognition.db")
    conn.row_factory = sqlite3.Row
    return conn


class Phase7Branching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        # point the in-process cognition + runtime modules at this test home: in-process
        # helpers (reconstruct/branches/audit/WorkingMemory) call read_events(), which
        # delegates to runtime, so BOTH modules' paths must target the per-test home.
        self._orig = (cognition.RUNTIME, cognition.EVENTS, cognition.COG_DB,
                      runtime.RUNTIME, runtime.EVENTS)
        cognition.RUNTIME = self.home / "runtime"
        cognition.EVENTS = cognition.RUNTIME / "events.jsonl"
        cognition.COG_DB = cognition.RUNTIME / "cognition.db"
        runtime.RUNTIME = self.home / "runtime"
        runtime.EVENTS = runtime.RUNTIME / "events.jsonl"
        append(self.home, "session_start", session="s1", project="demo")
        append(self.home, "decision", msg="use sqlite for the cache layer")
        # a hypothetical line of cognition, bracketed by branch_open / (later) branch_merge
        append(self.home, "branch_open", branch="pg-exp", **{"from": "main"})
        append(self.home, "decision", branch="pg-exp", msg="what if we used postgres")
        append(self.home, "objective", branch="pg-exp", msg="benchmark postgres throughput")
        run(COGNITION, "build", "--force", home=self.home)

    def tearDown(self):
        (cognition.RUNTIME, cognition.EVENTS, cognition.COG_DB,
         runtime.RUNTIME, runtime.EVENTS) = self._orig
        self.tmp.cleanup()

    # ── 1. unmerged branch is excluded from main; reconstruct sees it ────────
    def test_main_excludes_unmerged_branch(self):
        conn = db(self.home)
        bodies = [r[0] for r in conn.execute("SELECT body FROM nodes")]
        conn.close()
        self.assertTrue(any("sqlite" in b for b in bodies))      # main content present
        self.assertFalse(any("postgres" in b for b in bodies))   # what-if excluded from main

    def test_reconstruct_branch_in_isolation(self):
        before = (self.home / "runtime" / "cognition.db").read_bytes()
        s = cognition.reconstruct("pg-exp")
        main = cognition.reconstruct("main")
        self.assertGreater(s["nodes"], main["nodes"])             # branch adds context
        self.assertEqual(s["branch"], "pg-exp")
        after = (self.home / "runtime" / "cognition.db").read_bytes()
        self.assertEqual(before, after)                           # mainline cache untouched

    def test_reconstruct_branch_has_branch_content(self):
        conn = sqlite3.connect(":memory:")
        # reconstruct into an explicit on-disk db and inspect its nodes
        tmpdb = self.home / "runtime" / "branch.db"
        cognition.reconstruct("pg-exp", db_path=tmpdb)
        c = sqlite3.connect(tmpdb)
        bodies = [r[0] for r in c.execute("SELECT body FROM nodes")]
        c.close()
        self.assertTrue(any("postgres" in b for b in bodies))
        self.assertTrue(any("sqlite" in b for b in bodies))       # main base + branch

    # ── 2. merge folds the branch back into the mainline ────────────────────
    def test_merge_folds_branch_into_main(self):
        append(self.home, "branch_merge", branch="pg-exp", into="main")
        run(COGNITION, "build", "--force", home=self.home)
        conn = db(self.home)
        bodies = [r[0] for r in conn.execute("SELECT body FROM nodes")]
        conn.close()
        self.assertTrue(any("postgres" in b for b in bodies))     # now part of main
        # and the branch is reported as merged
        names = {b["branch"]: b["status"] for b in cognition.branches()}
        self.assertEqual(names["pg-exp"], "merged")
        self.assertEqual(names["main"], "main")

    # ── 3. back-compat: absent branch ⇒ main; redundant main dropped ────────
    def test_absent_branch_is_main_and_redundant_dropped(self):
        ev = runtime  # noqa
        out = append(self.home, "reflection", branch="main", msg="explicit main label")
        # the stored line must NOT carry a branch field (absent ⇒ main, v2-clean)
        last = events_text(self.home).strip().splitlines()[-1]
        self.assertNotIn('"branch"', last)

    def test_branchless_log_projects_as_before(self):
        # a brain with no branch fields at all reconstructs identically through both paths
        home2 = self.home / "plain"
        append(home2, "session_start", session="s0", project="plain")
        append(home2, "decision", msg="ship it")
        run(COGNITION, "build", "--force", home=home2)
        c = sqlite3.connect(home2 / "runtime" / "cognition.db")
        n = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        c.close()
        self.assertEqual(n, 2)                                    # session_start + decision

    def test_branches_listing(self):
        bs = {b["branch"]: b for b in cognition.branches()}
        self.assertEqual(bs["main"]["status"], "main")
        self.assertEqual(bs["pg-exp"]["status"], "open")
        self.assertEqual(bs["pg-exp"]["base"], "main")
        self.assertEqual(bs["pg-exp"]["events"], 3)               # 2 decisions/objective + open

    # ── 4. agent-local working memory ───────────────────────────────────────
    def test_working_memory_for_agent(self):
        run(COGNITION, "build", "--force", home=self.home)
        wm = cognition.WorkingMemory.for_agent("planner", query="cache")
        self.assertEqual(wm.agent, "planner")
        self.assertIsInstance(wm.render(), str)
        # planner focus includes 'decision'/'objective'; its lens is non-empty on this graph
        self.assertTrue(wm.active)

    def test_for_agent_empty_reduces_to_single_agent(self):
        run(COGNITION, "build", "--force", home=self.home)
        wm = cognition.WorkingMemory.for_agent("")
        base = cognition.WorkingMemory.from_graph("")
        self.assertEqual(set(wm.active), set(base.active))        # same lens, no role bias

    # ── 5. curator arbitration audit ────────────────────────────────────────
    def test_audit_flags_non_curator_consolidation(self):
        append(self.home, "concept_formed", agent="coder",
               concept_id="cpt_x", summary="x")
        viol = cognition.arbitration_violations()
        self.assertEqual(len(viol), 1)
        self.assertEqual(viol[0]["type"], "concept_formed")
        self.assertEqual(viol[0]["agent"], "coder")
        out = run(COGNITION, "audit", home=self.home).stdout
        self.assertIn("concept_formed", out)
        self.assertIn("coder", out)

    def test_audit_clean_for_curator_and_system(self):
        append(self.home, "concept_formed", agent="memory",
               concept_id="cpt_ok", summary="ok")          # curator: allowed
        append(self.home, "concept_formed",
               concept_id="cpt_sys", summary="sys")         # no agent (system/CLI): allowed
        self.assertEqual(cognition.arbitration_violations(), [])
        self.assertTrue(runtime.is_arbitration_violation("concept_formed", "researcher"))
        self.assertFalse(runtime.is_arbitration_violation("concept_formed", "memory"))
        self.assertFalse(runtime.is_arbitration_violation("hypothesis", "researcher"))

    # ── 6. determinism + schema-version rebuild ─────────────────────────────
    def test_branch_projection_deterministic(self):
        a = cognition.reconstruct("pg-exp")
        (self.home / "runtime" / "cognition.db").unlink()
        run(COGNITION, "build", "--force", home=self.home)
        b = cognition.reconstruct("pg-exp")
        self.assertEqual(a["nodes"], b["nodes"])
        self.assertEqual(a["edges"], b["edges"])

    def test_pre_phase7_cache_rebuilds(self):
        conn = db(self.home)
        conn.execute("UPDATE meta SET value='6' WHERE key='schema_version'")
        conn.commit()
        conn.close()
        out = run(COGNITION, "build", home=self.home).stdout   # no --force
        self.assertIn("built", out)                            # rebuilt, not "skipped"
        conn = db(self.home)
        self.assertEqual(conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],
            str(cognition.SCHEMA_VERSION))
        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
