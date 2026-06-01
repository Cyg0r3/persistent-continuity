#!/usr/bin/env python3
"""
Uninitialized-project guard tests.

Invariant (hooks.json): all lifecycle hooks must no-op silently in projects that
never ran /continuity-init, and must NOT create .continuity/. The SessionEnd and
PreCompact hooks invoke `runtime.py checkpoint`, so checkpoint() must honor the
same guard restore() does — otherwise append() mkdirs + seeds a stray events.jsonl.

Pure stdlib (unittest + subprocess), matching the project's no-dependency invariant.
Run: python tests/test_uninitialized_guard.py
"""

import os
import sys
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


class UninitializedGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _assert_clean(self):
        # No data root contents should have been created.
        self.assertFalse((self.home / "runtime").exists(),
                         "hook created .continuity/runtime in an uninitialized project")
        self.assertFalse((self.home / "runtime" / "events.jsonl").exists())

    def test_session_end_checkpoint_no_litter(self):
        run(RUNTIME, "checkpoint", "--reason", "auto: session-end", home=self.home)
        self._assert_clean()

    def test_pre_compact_checkpoint_no_litter(self):
        run(RUNTIME, "checkpoint", "--reason", "auto: pre-compact",
            "--keep-open", home=self.home)
        self._assert_clean()

    def test_session_start_restore_no_litter(self):
        run(RUNTIME, "restore", "--print", home=self.home)
        self._assert_clean()

    def test_prompt_refresh_no_litter(self):
        run(COGNITION, "refresh", "--hook", home=self.home)
        self._assert_clean()


if __name__ == "__main__":
    unittest.main(verbosity=2)
