"""
_paths.py — resolve the continuity *data* root (separate from this code).

The engine code ships inside the plugin (read-only, shared across every project).
A user's actual memory — events.jsonl, the graph/state DBs, identity files,
thread digests — must live per-project, NOT next to the code. This module is the
single place that decides where that data lives.

Resolution order:
  1. $CONTINUITY_HOME  — explicit override (absolute path). Use this to keep one
                         shared store, or to point several shells at the same brain.
  2. <cwd>/.continuity — default: a hidden folder in the current project. Claude Code
                         runs hooks/commands from the project root, so this lands the
                         memory beside the work it is about.

Nothing here creates the directory; only write paths (append/init) do that, so merely
having the plugin enabled never litters `.continuity/` into an untouched project.
"""

import os
from pathlib import Path


def data_root() -> Path:
    home = os.environ.get("CONTINUITY_HOME")
    base = Path(home).expanduser() if home else (Path.cwd() / ".continuity")
    return base.resolve()
