#!/usr/bin/env python3
"""
Phase 11 pattern-to-procedure pipeline tests (ARCHITECTURE_V4.md §16, migration Phase 11).

The capstone loop: observation → pattern (§13) → procedure (§14) → optimized strategy.
`reflect.py promote` takes a mined `kind="success"` pattern whose confidence clears a floor
and emits ONE curator `pattern_promoted` event linking `pattern_id → procedure_id`;
`cognition._build_procedures` then materializes the procedure, seeding its initial
`outcome_score` from the pattern confidence and its trigger from the recurring entry context.
Subsequent `proc_executed` reinforcement optimizes it. This verifies every invariant:

  1. `pattern_promoted` is a KNOWN, curator-owned consolidation conclusion (a named non-curator
     agent emitting it is an arbitration violation; the curator path is clean).
  2. promote is a projection of the current pattern layer: dry-run writes nothing; --apply
     appends exactly one pattern_promoted per promotable pattern, linking pattern → procedure.
  3. Only `success` patterns above the confidence floor promote (failure/low-confidence skip).
  4. The promoted procedure materializes with outcome_score seeded from the pattern confidence
     and trigger/steps derived from the pattern — and then `proc_executed` reinforcement (§14)
     optimizes it (the loop closes).
  5. Promotion is idempotent (one procedure per pattern); a promotion §14-retired is NOT revived
     by a later promote pass — the self-correction is sticky.
  6. Replayable truth: the link projects as a node; the curator audit stays clean.
  7. The T2 consolidation "sleep cycle" runs promotion as its [4/9] stage.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase11_promotion.py
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
REFLECT = str(SYS_DIR / "reflect.py")

sys.path.insert(0, str(SYS_DIR))
import runtime    # noqa: E402


def run(script, *args, home):
    env = dict(os.environ, CONTINUITY_HOME=str(home))
    return subprocess.run([sys.executable, script, *args], env=env,
                          capture_output=True, text=True, check=True)


def append(home, etype, **kv):
    args = [etype] + [f"{k}={v}" for k, v in kv.items()]
    return run(RUNTIME, "append", *args, home=home)


def events(home):
    p = Path(home) / "runtime" / "events.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def promotions(home):
    return [e for e in events(home) if e.get("type") == "pattern_promoted"]


def procedures(home):
    run(COGNITION, "build", "--force", home=home)
    conn = sqlite3.connect(Path(home) / "runtime" / "cognition.db")
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT procedure_id, outcome_score, uses, trigger, steps FROM procedures")]
    conn.close()
    return {r["procedure_id"]: r for r in rows}


def seed_success_pattern(home, pid="pat_success_run-tests-then-patch", conf=0.85):
    """Record a confident success pattern directly (mining it from threads is exercised by the
    Phase 8 suite; here we test the promotion edge that consumes a pattern)."""
    append(home, "pattern_detected", agent="memory", pattern_id=pid, kind="success",
           label="successful path: run-tests → patch → verify",
           confidence=conf, frequency="4", evidence="thr_1,evt_2",
           recommended="reuse this path")


class Phase11Promotion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        append(self.home, "session_start", session="s1", project="demo")
        append(self.home, "objective", msg="fix the failing parser tests for tokens")

    def tearDown(self):
        self.tmp.cleanup()

    # ── 1. pattern_promoted is known + curator-owned ─────────────────────────
    def test_pattern_promoted_known_no_unknown_warning(self):
        out = append(self.home, "pattern_promoted", agent="memory",
                     pattern_id="pat_success_x", procedure_id="prc_x")
        self.assertNotIn("unknown", (out.stdout + out.stderr).lower())

    def test_pattern_promoted_is_curator_owned(self):
        self.assertTrue(runtime.is_arbitration_violation("pattern_promoted", "coder"))
        self.assertFalse(runtime.is_arbitration_violation("pattern_promoted", "memory"))

    # ── 2. dry-run vs apply ──────────────────────────────────────────────────
    def test_dry_run_writes_nothing(self):
        seed_success_pattern(self.home)
        before = (self.home / "runtime" / "events.jsonl").read_text()
        run(REFLECT, "promote", home=self.home)
        self.assertEqual(before, (self.home / "runtime" / "events.jsonl").read_text())

    def test_apply_emits_one_link_per_pattern(self):
        seed_success_pattern(self.home)
        run(REFLECT, "promote", "--apply", home=self.home)
        proms = promotions(self.home)
        self.assertEqual(len(proms), 1)
        m = proms[0]
        self.assertEqual(m["agent"], "memory")                    # curator-owned
        self.assertEqual(m["pattern_id"], "pat_success_run-tests-then-patch")
        self.assertEqual(m["procedure_id"], "prc_run-tests-then-patch")

    # ── 3. only confident success patterns promote ───────────────────────────
    def test_failure_pattern_does_not_promote(self):
        append(self.home, "pattern_detected", agent="memory",
               pattern_id="pat_fail_timeout", kind="failure",
               label="recurring failure mode: 'timeout'", confidence="0.9",
               frequency="5", evidence="thr_1")
        run(REFLECT, "promote", "--apply", home=self.home)
        self.assertEqual(promotions(self.home), [])

    def test_low_confidence_success_does_not_promote(self):
        seed_success_pattern(self.home, pid="pat_success_weak", conf=0.55)
        run(REFLECT, "promote", "--apply", home=self.home)
        self.assertEqual(promotions(self.home), [])

    # ── 4. materialization + reinforcement (the loop closes) ─────────────────
    def test_promoted_procedure_seeds_score_from_confidence(self):
        seed_success_pattern(self.home, conf=0.85)
        run(REFLECT, "promote", "--apply", home=self.home)
        proc = procedures(self.home)["prc_run-tests-then-patch"]
        self.assertAlmostEqual(proc["outcome_score"], 0.85, places=4)
        self.assertTrue(proc["steps"])           # steps derived from the pattern signature
        self.assertTrue(proc["trigger"])         # trigger derived from the entry context

    def test_proc_executed_reinforcement_optimizes_promoted_procedure(self):
        # seed below 1.0 so a success execution has room to lift the score, but above the floor
        seed_success_pattern(self.home, conf=0.75)
        run(REFLECT, "promote", "--apply", home=self.home)
        base = procedures(self.home)["prc_run-tests-then-patch"]["outcome_score"]
        append(self.home, "proc_executed",
               procedure_id="prc_run-tests-then-patch", outcome="success")
        after = procedures(self.home)["prc_run-tests-then-patch"]
        self.assertGreater(after["outcome_score"], base)   # success reinforcement lifts it
        self.assertEqual(after["uses"], 1)                 # the execution counts as a use

    # ── 5. idempotency + sticky retirement ───────────────────────────────────
    def test_promotion_is_idempotent(self):
        seed_success_pattern(self.home)
        run(REFLECT, "promote", "--apply", home=self.home)
        run(REFLECT, "promote", "--apply", home=self.home)
        self.assertEqual(len(promotions(self.home)), 1)

    def test_retired_promotion_is_not_revived(self):
        seed_success_pattern(self.home)
        run(REFLECT, "promote", "--apply", home=self.home)
        # the curator retires the promoted procedure (§14 self-correction)
        append(self.home, "procedure_retired", agent="memory",
               procedure_id="prc_run-tests-then-patch", reason="chronically failing")
        self.assertNotIn("prc_run-tests-then-patch", procedures(self.home))
        # a later promote pass must NOT re-emit a link (so the retirement stays sticky)
        run(REFLECT, "promote", "--apply", home=self.home)
        self.assertEqual(len(promotions(self.home)), 1)
        self.assertNotIn("prc_run-tests-then-patch", procedures(self.home))

    # ── 6. replayable truth: projects as a node, audit clean ─────────────────
    def test_link_projects_as_node_and_audit_clean(self):
        seed_success_pattern(self.home)
        run(REFLECT, "promote", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        conn = sqlite3.connect(self.home / "runtime" / "cognition.db")
        row = conn.execute(
            "SELECT layer FROM nodes WHERE type='pattern_promoted'").fetchone()
        conn.close()
        self.assertIsNotNone(row)
        out = run(COGNITION, "audit", home=self.home).stdout
        self.assertIn("clean", out.lower())

    # ── 7. consolidation pipeline integration ────────────────────────────────
    def test_consolidate_runs_promotion_stage(self):
        seed_success_pattern(self.home)
        out = run(REFLECT, "consolidate", "--apply", home=self.home).stdout
        self.assertIn("[4/9] pattern promotion", out)
        # the sleep cycle promoted the seeded pattern into a procedure
        self.assertTrue(promotions(self.home))
        self.assertIn("prc_run-tests-then-patch", procedures(self.home))

    # ── back-compat: a log with no patterns promotes nothing, no crash ───────
    def test_no_patterns_promotes_nothing(self):
        run(REFLECT, "promote", "--apply", home=self.home)
        self.assertEqual(promotions(self.home), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
