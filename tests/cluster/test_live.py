"""Live tier: PyslurmCore against the real controller and accounting database.

Runs only on a login node with SLURM_MONITOR_CLUSTER_TESTS=1 (see tests/conftest.py). Each test
issues a handful of cheap RPCs (Job.load, filtered db.Jobs.load); the one whole-cluster load
happens only when the filtered extension is missing, inside active_jobs().
"""

from __future__ import annotations

import getpass
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from slurm_monitor.core.queries import PyslurmCore
from slurm_monitor.models import versions_compatible

pytestmark = [pytest.mark.cluster, pytest.mark.pyslurm]

RECORDED = Path(__file__).resolve().parents[1] / "fixtures" / "recorded"


@pytest.fixture(scope="module")
def core() -> PyslurmCore:
    return PyslurmCore()


@pytest.fixture(scope="module")
def history(core: PyslurmCore):
    return core.history_jobs(datetime.now(UTC) - timedelta(days=7))


def test_hello_reports_25_11(core: PyslurmCore):
    hs = core.hello()
    assert hs.pyslurm_version.startswith("25.11.")
    assert hs.slurm_built_version.startswith("25.11.")
    assert hs.slurm_running_version.startswith("25.11.")
    assert versions_compatible(hs.slurm_built_version, hs.slurm_running_version)
    assert hs.user == getpass.getuser()
    assert hs.sampling_interval_seconds == 30


def test_active_jobs_are_mine_and_not_terminal(core: PyslurmCore):
    result = core.active_jobs()
    me = getpass.getuser()
    for job in result.jobs:
        assert job.user == me
        assert job.is_active and not job.is_terminal
    if result.degraded:
        assert core.last_degraded_error


def test_history_window_holds_only_terminal_jobs(history):
    # until defaults to the agent's clock at call time, a few microseconds after `since` was built
    span = history.window_end - history.window_start
    assert abs((span - timedelta(days=7)).total_seconds()) < 5
    for job in history.jobs:
        assert job.is_terminal
        assert job.end_time is not None
        assert history.window_start <= job.end_time < history.window_end
    ends = [j.end_time for j in history.jobs]
    assert ends == sorted(ends, reverse=True)


def test_detail_of_recent_job_has_output_paths(core: PyslurmCore, history):
    if not history.jobs:
        pytest.skip("no jobs in the last 7 days for this user")
    detail = core.job_detail(history.jobs[0].job_id)
    assert detail.stdout_path and detail.stderr_path
    assert detail.stdout_path.startswith("/")
    assert detail.working_directory


def test_usage_of_recorded_short_job_is_unsampled(core: PyslurmCore):
    usage_dir = RECORDED / "short-job" / "job_usage"
    files = sorted(usage_dir.glob("*.json")) if usage_dir.exists() else []
    if not files:
        pytest.skip("short-job fixture not recorded")
    job_id = int(files[0].stem)
    usage = core.job_usage(job_id)
    assert usage.sampled is False
    assert usage.cpu.consumed is None and usage.memory.consumed is None
    assert usage.state == "COMPLETED"


def test_cancel_of_finished_job_is_refused(core: PyslurmCore, history):
    from slurm_monitor.core import CoreError
    from slurm_monitor.protocol import ERR_NOT_CANCELLABLE

    if not history.jobs:
        pytest.skip("no jobs in the last 7 days for this user")
    with pytest.raises(CoreError) as e:
        core.cancel(history.jobs[0].job_id)
    assert e.value.code == ERR_NOT_CANCELLABLE


def test_marker_gate():
    assert os.environ.get("SLURM_MONITOR_CLUSTER_TESTS") == "1"
