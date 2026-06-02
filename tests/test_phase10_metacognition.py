#!/usr/bin/env python3
"""
Phase 10 meta-cognition tests (ARCHITECTURE_V4.md §15, migration Phase 10).

A bounded, single-pass self-evaluation: `reflect.py introspect` reads five already-derived
signals (retrieval effectiveness, procedural success, cognitive drift, context pollution,
attention instability), folds them into a `confidence` scalar plus declarative strategy
`nudges`, and emits ONE curator `meta_assessment` event. This verifies every invariant:

  1. `meta_assessment` is a KNOWN, curator-owned consolidation conclusion (a named non-curator
     agent emitting it is an arbitration violation; the curator path is clean).
  2. introspect is a projection of the *current* graph/activation state: dry-run writes nothing,
     --apply appends exactly one event carrying confidence + per-metric readings + nudges.
  3. The five metric keys are always present; when activation is current they read real values,
     and a poor reading deterministically raises the matching nudge (metric↔nudge consistency).
  4. Back-compat / graceful degradation: with no activation snapshot the retrieval/pollution
     metrics read null and are simply excluded from the fold — no crash, confidence still derived.
  5. Non-recursion: a stored `meta_assessment` is NOT itself a meta-cognition input — introspecting
     a log that already contains assessments reports the same metric set and never feeds itself.
  6. The assessment is replayable truth: it projects as a layer='meta' node and the curator audit
     stays clean.
  7. The T2 consolidation "sleep cycle" runs meta-cognition as its final stage.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase10_metacognition.py
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


def last_meta(home):
    metas = [e for e in events(home) if e.get("type") == "meta_assessment"]
    return metas[-1] if metas else None


def warm(home, query="parser tokens lexer"):
    """Establish a current activation snapshot (the only state where retrieval-effectiveness
    and context-pollution are measurable): build, reinforce a focus, re-sync the watermark, then
    persist activation without growing the log."""
    run(COGNITION, "build", "--force", home=home)
    run(COGNITION, "context", query, "--reinforce", home=home)
    run(COGNITION, "build", home=home)          # fold the attended events, sync watermark
    run(COGNITION, "context", query, home=home)  # persist activation, no new truth


class Phase10Meta(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        append(self.home, "session_start", session="s1", project="demo")
        append(self.home, "objective", msg="build the parser for tokens and grammar")
        append(self.home, "observation", msg="parser tokens lexer grammar rules")
        append(self.home, "procedure_learned", agent="memory", procedure_id="prc_x",
               label="x", trigger="t", steps="s", evidence="e1")
        append(self.home, "proc_executed", procedure_id="prc_x", outcome="success")

    def tearDown(self):
        self.tmp.cleanup()

    # ── 1. meta_assessment is known + curator-owned ──────────────────────────
    def test_meta_assessment_known_no_unknown_warning(self):
        out = append(self.home, "meta_assessment", agent="memory", confidence="0.5")
        self.assertNotIn("unknown", (out.stdout + out.stderr).lower())

    def test_meta_assessment_is_curator_owned(self):
        self.assertTrue(runtime.is_arbitration_violation("meta_assessment", "coder"))
        self.assertFalse(runtime.is_arbitration_violation("meta_assessment", "memory"))

    # ── 2. dry-run vs apply ──────────────────────────────────────────────────
    def test_dry_run_writes_nothing(self):
        warm(self.home)
        before = (self.home / "runtime" / "events.jsonl").read_text()
        run(REFLECT, "introspect", home=self.home)
        self.assertEqual(before, (self.home / "runtime" / "events.jsonl").read_text())

    def test_apply_emits_one_curator_meta_assessment(self):
        warm(self.home)
        run(REFLECT, "introspect", "--apply", home=self.home)
        metas = [e for e in events(self.home) if e.get("type") == "meta_assessment"]
        self.assertEqual(len(metas), 1)
        m = metas[0]
        self.assertEqual(m["agent"], "memory")            # curator-owned
        self.assertIn("confidence", m)
        self.assertIn("metrics", m)
        self.assertIn("nudges", m)

    # ── 3. metrics + metric↔nudge consistency ────────────────────────────────
    def test_all_five_metric_keys_present(self):
        warm(self.home)
        run(REFLECT, "introspect", "--apply", home=self.home)
        metrics = last_meta(self.home)["metrics"]
        self.assertEqual(set(metrics), {
            "retrieval_effectiveness", "procedural_success", "cognitive_drift",
            "context_pollution", "attention_instability"})

    def test_warm_state_measures_retrieval_and_pollution(self):
        warm(self.home)
        run(REFLECT, "introspect", "--apply", home=self.home)
        metrics = last_meta(self.home)["metrics"]
        self.assertIsNotNone(metrics["retrieval_effectiveness"])
        self.assertIsNotNone(metrics["context_pollution"])

    def test_metric_threshold_drives_nudges(self):
        # whatever the readings are, each nudge must be present iff its metric crossed threshold
        import reflect
        warm(self.home)
        run(REFLECT, "introspect", "--apply", home=self.home)
        m = last_meta(self.home)
        metrics, nudges = m["metrics"], set(m["nudges"])
        re_ = metrics["retrieval_effectiveness"]
        ps = metrics["procedural_success"]
        dr = metrics["cognitive_drift"]
        po = metrics["context_pollution"]
        ins = metrics["attention_instability"]
        self.assertEqual("widen_retrieval" in nudges,
                         re_ is not None and re_ < reflect.META_RECALL_FLOOR)
        self.assertEqual("review_procedures" in nudges,
                         ps is not None and ps < reflect.META_PROC_FLOOR)
        self.assertEqual("trigger_compaction" in nudges,
                         po is not None and po >= reflect.META_POLLUTION_CEIL)
        self.assertEqual("enable_incubation" in nudges,
                         ins is not None and ins >= reflect.META_INSTABILITY_CEIL)

    # ── 4. graceful degradation: no activation snapshot ──────────────────────
    def test_no_activation_yields_null_metrics_no_crash(self):
        run(COGNITION, "build", "--force", home=self.home)   # no context ⇒ no activation
        run(REFLECT, "introspect", "--apply", home=self.home)
        metrics = last_meta(self.home)["metrics"]
        self.assertIsNone(metrics["retrieval_effectiveness"])
        self.assertIsNone(metrics["context_pollution"])
        # procedural success is still measurable (a procedure was executed)
        self.assertIsNotNone(metrics["procedural_success"])

    def test_empty_graph_self_assesses_without_proposing(self):
        # a home with only a session marker: nothing to assess, no nudges, no crash
        tmp = tempfile.TemporaryDirectory()
        home = Path(tmp.name)
        append(home, "session_start", session="s0", project="empty")
        run(COGNITION, "build", "--force", home=home)
        run(REFLECT, "introspect", "--apply", home=home)
        self.assertEqual(last_meta(home)["nudges"], [])
        tmp.cleanup()

    # ── 5. determinism + non-recursion ───────────────────────────────────────
    def test_confidence_is_deterministic(self):
        warm(self.home)
        a = run(REFLECT, "introspect", home=self.home).stdout
        b = run(REFLECT, "introspect", home=self.home).stdout
        self.assertEqual(a, b)

    def test_meta_assessment_is_not_its_own_input(self):
        # introspecting a log that ALREADY contains meta_assessments must report the same metric
        # set and not error — assessments are never inputs to meta-cognition (no recursion).
        warm(self.home)
        run(REFLECT, "introspect", "--apply", home=self.home)
        m1 = last_meta(self.home)["metrics"]
        warm(self.home)                                       # re-establish activation
        run(REFLECT, "introspect", "--apply", home=self.home)
        m2 = last_meta(self.home)["metrics"]
        self.assertEqual(set(m1), set(m2))
        # the procedural-success signal is unchanged by the intervening assessment
        self.assertEqual(m1["procedural_success"], m2["procedural_success"])

    # ── 6. replayable truth: projects as a meta node, audit clean ────────────
    def test_meta_assessment_projects_as_meta_node(self):
        warm(self.home)
        run(REFLECT, "introspect", "--apply", home=self.home)
        run(COGNITION, "build", "--force", home=self.home)
        conn = sqlite3.connect(self.home / "runtime" / "cognition.db")
        row = conn.execute(
            "SELECT layer FROM nodes WHERE type='meta_assessment'").fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "meta")

    def test_curator_audit_clean(self):
        warm(self.home)
        run(REFLECT, "introspect", "--apply", home=self.home)
        out = run(COGNITION, "audit", home=self.home).stdout
        self.assertIn("clean", out.lower())

    # ── 7. consolidation pipeline integration ────────────────────────────────
    def test_consolidate_runs_metacognition_stage(self):
        warm(self.home)
        out = run(REFLECT, "consolidate", "--apply", home=self.home).stdout
        self.assertIn("[9/9] meta-cognition", out)
        # the sleep cycle recorded a meta_assessment as its final act
        self.assertIsNotNone(last_meta(self.home))


if __name__ == "__main__":
    unittest.main(verbosity=2)
