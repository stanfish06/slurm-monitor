"""Test tiers: fixture tests run anywhere; `pyslurm`-marked tests need an importable pyslurm;
`cluster`-marked tests need a live controller and SLURM_MONITOR_CLUSTER_TESTS=1."""

from __future__ import annotations

import importlib.util
import os

import pytest


def pytest_collection_modifyitems(config, items):
    have_pyslurm = importlib.util.find_spec("pyslurm") is not None
    cluster_enabled = os.environ.get("SLURM_MONITOR_CLUSTER_TESTS") == "1"
    skip_pyslurm = pytest.mark.skip(reason="pyslurm is not importable on this host")
    skip_cluster = pytest.mark.skip(reason="set SLURM_MONITOR_CLUSTER_TESTS=1 on a login node")
    for item in items:
        if "cluster" in item.keywords and not (cluster_enabled and have_pyslurm):
            item.add_marker(skip_cluster)
        elif "pyslurm" in item.keywords and not have_pyslurm:
            item.add_marker(skip_pyslurm)
