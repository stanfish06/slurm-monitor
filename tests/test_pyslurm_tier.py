"""Tier two: real pyslurm and libslurm present, no controller needed.

Runs in the nix devShell on Linux (CI) and on the cluster. libslurm needs a slurm.conf at import
time; CI points SLURM_CONF at a two-line stub.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.pyslurm

EXT_DIR = Path(__file__).resolve().parents[1] / "src" / "slurm_monitor" / "ext"


def test_pyslurm_is_built_against_slurm_25_11():
    pyslurm = pytest.importorskip("pyslurm")
    assert pyslurm.__version__ == "25.11.2"
    assert pyslurm.slurm_api_version()[:2] == (25, 11)


def test_extension_compiles_and_matches_pyslurm_api_version(tmp_path: Path):
    pyslurm = pytest.importorskip("pyslurm")
    build = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--build-lib", str(tmp_path)],
        cwd=EXT_DIR,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr[-2000:]
    probe = subprocess.run(
        [sys.executable, "-c", "import slurm_monitor_ext as e; print(*e.extension_version())"],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert tuple(int(x) for x in probe.stdout.split()) == tuple(pyslurm.slurm_api_version())
