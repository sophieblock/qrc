"""
graphviz_check.py
=================

Drop-in, platform-agnostic Graphviz availability check.

The Python ``graphviz`` package (and ``pydot``, ``pygraphviz``, ``networkx``,
``qiskit``, ``torch.fx``, etc.) all shell out to the **system** ``dot``
executable to render graphs. That binary is a separate OS-level install and
is the #1 source of "works on my machine" bugs.

This module:
  * Verifies the ``dot`` executable is on PATH (or at ``$GRAPHVIZ_DOT``)
  * Verifies it actually runs (catches broken installs / arch mismatches)
  * Verifies the Python wrapper ``graphviz`` is importable
  * Returns a structured status dict you can log, assert on, or surface in CI
  * Raises a single typed exception with platform-specific install hints

Zero third-party dependencies in this file — pure stdlib. Safe to copy into
any project (GitHub, GitLab CI, conda envs, Docker, locked-down work boxes).

Usage
-----

    from graphviz_check import check_graphviz, require_graphviz, GraphvizNotFound

    # Soft check — never raises
    status = check_graphviz()
    if not status["ok"]:
        log.warning("Graphviz unavailable: %s", status["reason"])

    # Hard check — raise at import time of your viz module
    require_graphviz()

    # CLI smoke test
    #   $ python -m graphviz_check
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Optional, TypedDict

__all__ = [
    "GraphvizNotFound",
    "GraphvizStatus",
    "check_graphviz",
    "require_graphviz",
    "get_dot_executable",
    "install_hint",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class GraphvizStatus(TypedDict):
    ok: bool
    dot_path: Optional[str]
    dot_version: Optional[str]
    python_pkg: Optional[str]   # version string of `graphviz` python pkg, or None
    reason: Optional[str]       # human-readable failure reason if ok is False
    platform: str               # 'Darwin' | 'Linux' | 'Windows'


class GraphvizNotFound(RuntimeError):
    """Raised by ``require_graphviz()`` when Graphviz isn't usable."""


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def get_dot_executable() -> Optional[str]:
    """Resolve the path to the ``dot`` executable.

    Resolution order (matches the PlantUML / qiskit-terra convention):
      1. ``$GRAPHVIZ_DOT`` env var if set and the file exists
      2. ``shutil.which("dot")`` — first match on PATH
    """
    env_override = os.environ.get("GRAPHVIZ_DOT")
    if env_override and os.path.isfile(env_override):
        return env_override
    return shutil.which("dot")


@lru_cache(maxsize=1)
def check_graphviz() -> GraphvizStatus:
    """Return a structured status dict. Never raises. Result is cached.

    The cache means it's safe to call from hot paths (e.g. inside a render
    loop) — only the first call pays the subprocess cost.
    """
    status: GraphvizStatus = {
        "ok": False,
        "dot_path": None,
        "dot_version": None,
        "python_pkg": None,
        "reason": None,
        "platform": platform.system(),
    }

    # 1) Locate the binary
    dot_path = get_dot_executable()
    status["dot_path"] = dot_path
    if dot_path is None:
        status["reason"] = (
            "Graphviz `dot` executable not found on PATH and "
            "$GRAPHVIZ_DOT is not set."
        )
        return status

    # 2) Verify it actually runs (catches broken installs, wrong arch on
    #    Apple Silicon, missing dylib deps, sandbox blocks, etc.)
    try:
        result = subprocess.run(
            [dot_path, "-V"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        status["reason"] = f"Found {dot_path} but it failed to execute: {exc!r}"
        return status

    if result.returncode != 0:
        status["reason"] = (
            f"`{dot_path} -V` exited with code {result.returncode}. "
            f"stderr: {result.stderr.strip()!r}"
        )
        return status

    # `dot -V` writes its version banner to stderr (historical quirk)
    version_line = (result.stderr or result.stdout).strip()
    status["dot_version"] = version_line or "unknown"

    # 3) Check the Python wrapper. Not strictly required (you can drive
    #    `dot` over subprocess yourself), but most callers want it.
    try:
        import graphviz as _graphviz_pkg  # type: ignore[import-untyped]
        status["python_pkg"] = getattr(_graphviz_pkg, "__version__", "unknown")
    except ImportError:
        status["python_pkg"] = None
        # Not a hard failure — system Graphviz is the thing that matters.

    status["ok"] = True
    return status


def require_graphviz(*, need_python_pkg: bool = True) -> GraphvizStatus:
    """Hard check. Raises :class:`GraphvizNotFound` with install hints.

    Parameters
    ----------
    need_python_pkg:
        If True (default), also require the Python ``graphviz`` package.
        Set to False if you only shell out to ``dot`` directly.
    """
    status = check_graphviz()
    if not status["ok"]:
        raise GraphvizNotFound(
            f"{status['reason']}\n\n{install_hint(status['platform'])}"
        )
    if need_python_pkg and status["python_pkg"] is None:
        raise GraphvizNotFound(
            "System Graphviz is installed, but the Python `graphviz` package "
            "is missing. Run:\n\n    pip install graphviz\n"
        )
    return status


# ---------------------------------------------------------------------------
# Install hints
# ---------------------------------------------------------------------------

_HINTS = {
    "Darwin": (
        "Install Graphviz on macOS:\n"
        "    brew install graphviz          # Homebrew (Intel or Apple Silicon)\n"
        "    conda install -c conda-forge graphviz   # conda/mamba/miniforge\n"
        "Then verify:\n"
        "    dot -V"
    ),
    "Linux": (
        "Install Graphviz on Linux:\n"
        "    sudo apt-get install -y graphviz     # Debian/Ubuntu\n"
        "    sudo dnf install -y graphviz         # Fedora/RHEL\n"
        "    conda install -c conda-forge graphviz\n"
        "Then verify:\n"
        "    dot -V"
    ),
    "Windows": (
        "Install Graphviz on Windows:\n"
        "    winget install graphviz              # Windows Package Manager\n"
        "    choco install graphviz               # Chocolatey\n"
        "    conda install -c conda-forge graphviz\n"
        "Or download the installer from https://graphviz.org/download/\n"
        "IMPORTANT: tick 'Add Graphviz to the system PATH' during install,\n"
        "or set the environment variable GRAPHVIZ_DOT to the full path of dot.exe.\n"
        "Then verify in a NEW shell:\n"
        "    dot -V"
    ),
}


def install_hint(plat: Optional[str] = None) -> str:
    """Return a platform-appropriate install instruction block."""
    plat = plat or platform.system()
    return _HINTS.get(plat, _HINTS["Linux"])


# ---------------------------------------------------------------------------
# CLI: `python -m graphviz_check`
# ---------------------------------------------------------------------------

def _main() -> int:
    status = check_graphviz()
    print(f"Platform     : {status['platform']}")
    print(f"dot path     : {status['dot_path']}")
    print(f"dot version  : {status['dot_version']}")
    print(f"python pkg   : {status['python_pkg']}")
    print(f"OK           : {status['ok']}")
    if not status["ok"]:
        print(f"\nReason       : {status['reason']}")
        print()
        print(install_hint(status["platform"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
