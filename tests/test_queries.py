"""PyslurmCore against injected loaders (no pyslurm needed) and RecordingCore's file layout."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from slurm_monitor.core import CoreError, dispatch
from slurm_monitor.core.queries import PyslurmCore
from slurm_monitor.core.recording import RecordingCore
from slurm_monitor.models import ActiveJobsResult, JobUsage
from slurm_monitor.protocol import ERR_NOT_CANCELLABLE, ERR_NOT_FOUND, ERR_RPC

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
T0 = int(datetime(2026, 9, 5, 11, 0, tzinfo=UTC).timestamp())


def ctrl(job_id: int, state: str, **extra):
    """A controller record as pyslurm.Job exposes it (attribute names, epoch ints)."""
    base = dict(
        id=job_id,
        name=f"job{job_id}",
        state=state,
        state_reason="Priority" if state == "PENDING" else "None",
        user_name="me",
        user_id=1000,
        partition="standard",
        account="acct",
        submit_time=T0,
        start_time=0 if state == "PENDING" else T0 + 60,
        end_time=T0 + 360 if state != "PENDING" else 0,
        run_time=0 if state == "PENDING" else 120,
        num_nodes=1,
        cpus=2,
        array_id=None,
        array_task_id=None,
        array_tasks_waiting=None,
        time_limit=5,
        memory=200,
        gpus={},
        allocated_nodes=None if state == "PENDING" else "gl1",
        working_directory="/w",
        standard_output="/w/slurm-%j.out",
        standard_error="/w/slurm-%j.out",
    )
    base.update(extra)
    return SimpleNamespace(**base)


def acct(job_id: int, state: str, end: int, **extra):
    base = dict(
        id=job_id,
        name=f"job{job_id}",
        state=state,
        user_name="me",
        partition="standard",
        account="acct",
        submit_time=T0,
        start_time=end - 90,
        end_time=end,
        elapsed_time=90,
        num_nodes=1,
        cpus=2,
        array_id=None,
        array_task_id=None,
        array_tasks_waiting=None,
        exit_code=0,
        exit_code_signal=0,
        cancelled_by=None,
        nodelist="gl1",
        working_directory="/w",
        submit_command="sbatch --wrap=true",
        standard_output=None,
        standard_error=None,
        script=None,
        time_limit=5,
        memory=200,
        gpus={},
        steps={},
        stats=SimpleNamespace(total_cpu_time=0, resident_memory=0),
    )
    base.update(extra)
    return SimpleNamespace(**base)


def step(**stats):
    return SimpleNamespace(stats=SimpleNamespace(**stats))


def make_core(**kw) -> PyslurmCore:
    kw.setdefault("clock", lambda: NOW)
    kw.setdefault("username", "me")
    kw.setdefault("uid", 1000)
    kw.setdefault("sampling_interval_seconds", 30)
    return PyslurmCore(**kw)


# --- active --------------------------------------------------------------------------------


def test_active_excludes_terminal_jobs_still_held_by_controller():
    held = [
        ctrl(1, "PENDING"),
        ctrl(2, "RUNNING"),
        ctrl(3, "COMPLETED"),  # within MinJobAge: the controller still returns it
        ctrl(4, "CANCELLED"),
        ctrl(5, "COMPLETING"),
    ]
    result = make_core(job_loader=lambda: held).active_jobs()
    assert [j.job_id for j in result.jobs] == [1, 2, 5]
    assert result.degraded is False and result.fetched_at == NOW
    pending = result.jobs[0]
    assert pending.state_reason == "Priority" and pending.start_time is None
    running = result.jobs[1]
    assert running.end_time is None and running.elapsed_seconds == 120


def test_active_groups_array_records():
    held = [
        ctrl(11, "RUNNING", array_id=10, array_task_id=1),
        ctrl(10, "PENDING", array_id=10, array_tasks_waiting="3-6"),
        ctrl(12, "RUNNING", array_id=10, array_task_id=2),
        ctrl(7, "RUNNING"),
    ]
    result = make_core(job_loader=lambda: held).active_jobs()
    assert [j.display_id for j in result.jobs] == ["7", "10_[3-6]", "10_1", "10_2"]


# --- history -------------------------------------------------------------------------------


def _history_store():
    end = int(NOW.timestamp())
    return [
        acct(100, "COMPLETED", end - 60),
        acct(101, "FAILED", end - 3600, exit_code=3),
        acct(102, "RUNNING", 0),  # still running: never in history
        acct(103, "CANCELLED", end - 2 * 86400, cancelled_by="me"),
        acct(104, "TIMEOUT", end - 8 * 86400),
        acct(105, "COMPLETED", end - 7 * 86400),  # exactly on the 7-day boundary
    ]


def test_history_default_window_and_ordering():
    calls = []

    def loader(user, since, until):
        calls.append((user, since, until))
        return _history_store()

    core = make_core(history_loader=loader)
    result = core.history_jobs(NOW - timedelta(days=7))
    assert calls == [("me", NOW - timedelta(days=7), NOW)]
    assert [j.job_id for j in result.jobs] == [100, 101, 103, 105]
    assert all(j.is_terminal for j in result.jobs)
    assert result.jobs[2].cancelled_by == "me" and result.jobs[1].exit_code == 3
    assert result.window_start == NOW - timedelta(days=7) and result.window_end == NOW


def test_history_extension_is_disjoint_from_first_window():
    core = make_core(history_loader=lambda u, s, e: _history_store())
    first = core.history_jobs(NOW - timedelta(days=7))
    older = core.history_jobs(NOW - timedelta(days=30), NOW - timedelta(days=7))
    assert [j.job_id for j in older.jobs] == [104]
    assert not {j.job_id for j in first.jobs} & {j.job_id for j in older.jobs}


def test_history_naive_datetimes_are_utc_and_dispatch_parses_iso():
    core = make_core(history_loader=lambda u, s, e: _history_store())
    result = dispatch(
        core,
        "history_jobs",
        {"since": (NOW - timedelta(days=1)).isoformat(), "until": None},
    )
    assert [j.job_id for j in result.jobs] == [100, 101]
    naive = core.history_jobs(datetime(2026, 9, 4, 12, 0))
    assert naive.window_start.tzinfo is UTC


# --- detail --------------------------------------------------------------------------------


def _controller_with(records: dict[int, object]):
    def load(job_id: int):
        if job_id in records:
            return records[job_id]
        raise CoreError(ERR_NOT_FOUND, "Invalid job id specified")

    return load


def test_detail_prefers_controller_then_accounting():
    core = make_core(
        controller_loader=_controller_with({1: ctrl(1, "RUNNING")}),
        accounting_loader=lambda jid, with_script: (
            [acct(2, "COMPLETED", T0 + 500)] if jid == 2 else []
        ),
    )
    live = core.job_detail(1)
    assert live.source == "controller" and live.stdout_path == "/w/slurm-1.out"
    assert live.nodelist == "gl1" and live.requested.cpus == 2
    done = core.job_detail(2)
    assert done.source == "accounting" and done.stdout_path == "/w/slurm-2.out"
    with pytest.raises(CoreError) as e:
        core.job_detail(3)
    assert e.value.code == ERR_NOT_FOUND


def test_detail_picks_exact_record_from_array_siblings():
    siblings = [
        acct(20, "COMPLETED", T0 + 500, array_id=20, array_task_id=3),
        acct(21, "COMPLETED", T0 + 500, array_id=20, array_task_id=None),  # task 0
        acct(22, "FAILED", T0 + 500, array_id=20, array_task_id=2, exit_code=1),
    ]
    core = make_core(
        controller_loader=_controller_with({}),
        accounting_loader=lambda jid, with_script: siblings,
    )
    assert core.job_detail(22).display_id == "20_2"
    assert core.job_detail(21).display_id == "20_0"
    assert core.job_detail(22).stdout_path == "/w/slurm-20_2.out"


# --- usage ---------------------------------------------------------------------------------


class RunningJob(SimpleNamespace):
    """Controller record whose load_stats() fills in sstat-style step statistics."""

    def load_stats(self):
        self.steps = {
            "batch": step(
                total_cpu_time=100, max_resident_memory=50 * 1024 * 1024, avg_cpu_time=100
            ),
            "extern": step(total_cpu_time=0, max_resident_memory=0, avg_cpu_time=0),
        }
        self.stats = SimpleNamespace(total_cpu_time=100, resident_memory=50 * 1024 * 1024)
        return self.stats


class UnsampledRunningJob(RunningJob):
    def load_stats(self):
        self.steps = {"batch": step(total_cpu_time=0, max_resident_memory=0, avg_cpu_time=0)}
        self.stats = SimpleNamespace(total_cpu_time=0, resident_memory=0)
        return self.stats


def test_usage_running_job_from_live_stats():
    job = RunningJob(**vars(ctrl(1, "RUNNING")))
    core = make_core(controller_loader=_controller_with({1: job}))
    usage = core.job_usage(1)
    assert usage.sampled is True and usage.state == "RUNNING"
    assert usage.cpu.consumed == 100.0 and usage.cpu.requested == 240.0
    assert usage.memory.consumed == 50.0 and usage.memory.requested == 200.0
    assert usage.walltime.consumed == 120.0 and usage.walltime.requested == 300.0
    assert usage.gpus_allocated == 0 and usage.gpu_utilization is None


def test_usage_running_job_younger_than_sampling_interval():
    job = UnsampledRunningJob(**vars(ctrl(1, "RUNNING", run_time=12)))
    usage = make_core(controller_loader=_controller_with({1: job})).job_usage(1)
    assert usage.sampled is False
    assert usage.cpu.consumed is None and usage.memory.consumed is None
    assert usage.cpu.percent is None
    assert usage.walltime.consumed == 12.0


def test_usage_pending_job_has_no_samples():
    usage = make_core(controller_loader=_controller_with({1: ctrl(1, "PENDING")})).job_usage(1)
    assert usage.sampled is False and usage.cpu.requested is None
    assert usage.walltime.consumed == 0.0 and usage.walltime.requested == 300.0


def test_usage_terminal_job_prefers_accounting_over_controller_copy():
    record = acct(
        5,
        "COMPLETED",
        T0 + 500,
        steps={"batch": step(total_cpu_time=80, max_resident_memory=1024**2, avg_cpu_time=80)},
        stats=SimpleNamespace(total_cpu_time=80, resident_memory=1024**2),
    )
    core = make_core(
        controller_loader=_controller_with({5: ctrl(5, "COMPLETED")}),
        accounting_loader=lambda jid, ws: [record],
    )
    usage = core.job_usage(5)
    assert usage.sampled and usage.cpu.consumed == 80.0 and usage.memory.consumed == 1.0
    assert usage.cpu.requested == 180.0  # 2 cpus * 90 s elapsed from accounting


def test_usage_short_job_zero_stats_is_absent_not_zero():
    record = acct(
        6,
        "COMPLETED",
        T0 + 100,
        elapsed_time=1,
        steps={
            # jobacct_gather's teardown poll: 2 MB RSS from a job that ran for 1 s
            "batch": step(total_cpu_time=0, max_resident_memory=2072576, avg_cpu_time=0),
            "extern": step(total_cpu_time=0, max_resident_memory=0, avg_cpu_time=0),
        },
    )
    core = make_core(
        controller_loader=_controller_with({}), accounting_loader=lambda j, w: [record]
    )
    usage = core.job_usage(6)
    assert usage.sampled is False
    assert usage.cpu.consumed is None and usage.memory.consumed is None
    assert usage.walltime.consumed == 1.0


def test_usage_gpu_job_reads_tres_usage_and_keeps_recorded_zero():
    gpu_record = acct(
        7,
        "COMPLETED",
        T0 + 500,
        gpus={"gpu": SimpleNamespace(count=1)},
        steps={"batch": step(total_cpu_time=50, max_resident_memory=2048, avg_cpu_time=50)},
        stats=SimpleNamespace(total_cpu_time=50, resident_memory=2048),
    )
    tres = [
        {"tres_usage_in_ave": "1=50000,2=2048,1035=0", "tres_usage_in_max": "2=2048,1036=1048576"},
        {"tres_usage_in_ave": "3=0", "tres_usage_in_max": "3=0"},
    ]
    core = make_core(
        controller_loader=_controller_with({}),
        accounting_loader=lambda j, w: [gpu_record],
        tres_usage_loader=lambda j: tres,
        gpu_tres_ids=lambda: (1035, 1036),
    )
    usage = core.job_usage(7)
    assert usage.gpus_allocated == 1 and usage.gpu_accounting_available is True
    assert usage.gpu_utilization.consumed == 0.0 and usage.gpu_utilization.percent == 0.0
    assert usage.gpu_memory.consumed == 1.0

    # Same job, but no step carries the GPU TRES: the sample is absent, not zero.
    core = make_core(
        controller_loader=_controller_with({}),
        accounting_loader=lambda j, w: [gpu_record],
        tres_usage_loader=lambda j: [{"tres_usage_in_ave": "1=50000", "tres_usage_in_max": ""}],
        gpu_tres_ids=lambda: (1035, 1036),
    )
    assert core.job_usage(7).gpu_utilization.consumed is None

    # Cluster without gres/gpuutil in its TRES table: unavailable.
    core = make_core(
        controller_loader=_controller_with({}),
        accounting_loader=lambda j, w: [gpu_record],
        tres_usage_loader=lambda j: tres,
        gpu_tres_ids=lambda: (None, None),
    )
    usage = core.job_usage(7)
    assert usage.gpu_accounting_available is False and usage.gpu_utilization.consumed is None


def test_usage_unknown_job():
    core = make_core(controller_loader=_controller_with({}), accounting_loader=lambda j, w: [])
    with pytest.raises(CoreError) as e:
        core.job_usage(99)
    assert e.value.code == ERR_NOT_FOUND


# --- cancel --------------------------------------------------------------------------------


def test_cancel_targets_and_refusals():
    killed = []

    def canceller(target: str):
        if target == "9":
            raise CoreError(ERR_RPC, "Access/permission denied")
        killed.append(target)

    core = make_core(
        controller_loader=_controller_with(
            {
                1: ctrl(1, "RUNNING"),
                2: ctrl(2, "COMPLETED"),
                9: ctrl(9, "PENDING"),
                10: ctrl(10, "PENDING", array_id=10, array_tasks_waiting="1-6"),
            }
        ),
        accounting_loader=lambda j, w: [acct(3, "COMPLETED", T0 + 100)] if j == 3 else [],
        canceller=canceller,
    )
    assert core.cancel(1).model_dump() == {
        "job_id": 1,
        "array_task_id": None,
        "cancelled": True,
        "message": None,
    }
    with pytest.raises(CoreError) as e:
        core.cancel(2)  # terminal but still held by the controller
    assert e.value.code == ERR_NOT_CANCELLABLE
    with pytest.raises(CoreError) as e:
        core.cancel(3)  # purged from the controller, present in accounting
    assert e.value.code == ERR_NOT_CANCELLABLE
    with pytest.raises(CoreError) as e:
        core.cancel(4)
    assert e.value.code == ERR_NOT_FOUND
    with pytest.raises(CoreError) as e:
        core.cancel(9)
    assert e.value.code == ERR_RPC and "denied" in e.value.message
    # Whole array by base id; one task as "<array>_<task>" without a controller pre-check.
    core.cancel(10)
    result = core.cancel(10, 4)
    assert result.array_task_id == 4 and result.cancelled
    assert killed == ["1", "10", "10_4"]


# --- recording -----------------------------------------------------------------------------


def test_recording_core_layout(tmp_path):
    inner = make_core(
        job_loader=lambda: [ctrl(1, "RUNNING")],
        history_loader=lambda u, s, e: _history_store(),
        controller_loader=_controller_with({1: RunningJob(**vars(ctrl(1, "RUNNING")))}),
        accounting_loader=lambda j, w: [],
        canceller=lambda t: None,
    )
    rec = RecordingCore(inner, tmp_path / "scenario")
    rec.active_jobs()
    rec.active_jobs()
    rec.history_jobs(NOW - timedelta(days=7))
    rec.job_detail(1)
    rec.job_usage(1)
    rec.job_usage(1)
    rec.cancel(1)
    rec.cancel(10, 4)
    rec.record_raw("controller_1", {"id": 1, "gpus": {"gpu": SimpleNamespace(count=1)}})
    base = tmp_path / "scenario"
    snaps = json.loads((base / "active_jobs.json").read_text())
    assert len(snaps) == 2 and ActiveJobsResult.model_validate(snaps[0]).jobs[0].job_id == 1
    assert len(json.loads((base / "history_jobs.json").read_text())) == 1
    assert (base / "job_detail" / "1.json").exists()
    usages = json.loads((base / "job_usage" / "1.json").read_text())
    assert isinstance(usages, list) and len(usages) == 2
    JobUsage.model_validate(usages[0])
    assert (base / "cancel" / "1.json").exists() and (base / "cancel" / "10_4.json").exists()
    raw = json.loads((base / "raw" / "controller_1.json").read_text())
    assert raw["gpus"]["gpu"].startswith("namespace(")  # non-model objects become strings
