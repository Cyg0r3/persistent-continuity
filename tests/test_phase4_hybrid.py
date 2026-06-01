#!/usr/bin/env python3
"""
Phase 4 persistent-hybrid-retrieval tests (ARCHITECTURE_V4.md §11 Phase 4).

Verifies the new persisted-vector layer preserves every invariant:
  1. build() persists one embedding per node into `vectors` (stdlib TF-IDF by default,
     no dependency), tags the backend in meta, and stores the idf needed to embed a query.
  2. _vector_search is a nearest-neighbour scan over the stored vectors: a query surfaces
     the term-overlapping node ahead of an unrelated one (associative recall > LIKE).
  3. Graceful fallback: no vectors ⇒ [] (caller drops to BM25/lexical); dense-tagged
     vectors with the model unavailable at query time ⇒ [] (no crash).
  4. Backward-compat: a pre-Phase-4 cache (older schema_version / missing table) rebuilds
     rather than silently skipping, and an empty brain still builds + renders.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_phase4_hybrid.py
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

sys.path.insert(0, str(SYS_DIR))
import cognition  # noqa: E402  (pure helpers operate on an explicit conn, no ROOT needed)


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


def mem_graph(node_bodies):
    """An in-memory cognition.db seeded with the given {event_id: body} nodes."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(cognition.SCHEMA)
    for i, (eid, body) in enumerate(node_bodies.items(), 1):
        conn.execute(
            "INSERT INTO nodes(event_id,seq,t,type,layer,body,importance,unresolved,"
            "reinforced) VALUES (?,?,?,?,?,?,?,?,?)",
            (eid, i, "2026-01-01T00:00:00+00:00", "decision", "episodic", body, 0.5, 0, 0))
    return conn


class Phase4Hybrid(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        append(self.home, "session_start", session="s1", project="demo")
        append(self.home, "decision", msg="adopt sqlite connection pooling for throughput")
        append(self.home, "artifact", msg="auth token refresh handler module")
        run(COGNITION, "build", "--force", home=self.home)

    def tearDown(self):
        self.tmp.cleanup()

    def test_vectors_persisted_by_default(self):
        conn = db(self.home)
        n_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_vecs = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        self.assertEqual(n_vecs, n_nodes)         # one persisted embedding per node
        self.assertGreater(n_vecs, 0)
        model = conn.execute(
            "SELECT value FROM meta WHERE key='vector_model'").fetchone()[0]
        self.assertEqual(model, cognition.LEXICAL_MODEL)   # stdlib default, no dependency
        idf = conn.execute(
            "SELECT value FROM meta WHERE key='vector_idf'").fetchone()
        self.assertIsNotNone(idf)                  # query-time idf persisted
        # stored vec is a non-trivial sparse TF-IDF map
        row = conn.execute("SELECT vec FROM vectors LIMIT 1").fetchone()
        conn.close()
        self.assertIsInstance(json.loads(row[0]), dict)

    def test_build_cli_reports_vectors(self):
        out = run(COGNITION, "build", "--force", home=self.home)
        self.assertIn("vectors", out)

    def test_vector_search_surfaces_relevant_node(self):
        conn = mem_graph({
            "d-pool": "adopt sqlite connection pooling for throughput performance",
            "d-auth": "auth token refresh handler module",
            "d-ui": "redesign the dashboard color palette",
        })
        cognition._build_vectors(conn)
        hits = cognition._vector_search(conn, "database pooling throughput")
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "d-pool")            # best match by shared terms
        self.assertTrue(0.0 < hits[0][1] <= 1.0)          # similarity normalized to [0,1]
        ids = [eid for eid, _ in hits]
        self.assertNotIn("d-ui", ids)                     # unrelated node not pulled in

    def test_no_vectors_returns_empty_for_fallback(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(cognition.SCHEMA)              # nodes + vectors empty, no meta
        self.assertEqual(cognition._vector_search(conn, "anything"), [])

    def test_dense_tag_without_model_degrades(self):
        conn = mem_graph({"d-pool": "connection pooling throughput"})
        cognition._build_vectors(conn)
        # Pretend the vectors were built by a dense model that isn't loaded now.
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('vector_model',?)",
                     (cognition.DENSE_PREFIX + "all-MiniLM-L6-v2",))
        self.assertEqual(cognition._vector_search(conn, "pooling"), [])  # silent fallback

    def test_empty_query_returns_empty(self):
        conn = mem_graph({"d-pool": "connection pooling"})
        cognition._build_vectors(conn)
        self.assertEqual(cognition._vector_search(conn, ""), [])

    def test_pre_phase4_cache_rebuilds(self):
        conn = db(self.home)
        conn.execute("DELETE FROM meta WHERE key='schema_version'")
        conn.execute("DROP TABLE vectors")
        conn.commit()
        conn.close()
        out = run(COGNITION, "build", home=self.home)     # no --force
        self.assertIn("built", out)                       # rebuilt, not "skipped"
        conn = db(self.home)
        self.assertEqual(
            conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0],
            str(cognition.SCHEMA_VERSION))
        self.assertGreater(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 0)
        conn.close()

    def test_empty_brain_builds_and_renders(self):
        home = Path(self.tmp.name) / "empty"
        append(home, "session_start", session="s0", project="empty")
        run(COGNITION, "build", "--force", home=home)
        ctx = run(COGNITION, "context", home=home)        # must not crash on no vectors
        self.assertIsInstance(ctx, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
