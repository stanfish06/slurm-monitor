"""Normalization: controller and accounting records onto the shared models.

The first half runs against plain dicts; the second half walks tests/fixtures/recorded, validates
every fixture file against its model, and checks that the raw controller and accounting dumps of
the same job normalize to identical shared fields.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from slurm_monitor.core.normalize import (
    accounting_detail,
    accounting_summary,
    build_usage,
    controller_detail,
    controller_summary,
    expand_stdio,
    gpu_count,
    gpu_usage_from_tres,
    parse_tres_usage,
    steps_sampled,
    task_range_from_bitmask,
)
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobUsage,
    Source,
)

RECORDED = Path(__file__).parent / "fixtures" / "recorded"

CONTROLLER_DONE = {
    "id": 4242,
    "name": "train",
    "state": "COMPLETED",
    "user_name": "alice",
    "partition": "standard",
    "account": "acct",
    "qos": "normal",
    "submit_time": 1_800_000_000,
    "start_time": 1_800_000_100,
    "end_time": 1_800_000_190,
    "run_time": 90,
    "num_nodes": 1,
    "cpus": 2,
    "array_id": None,
    "array_task_id": None,
    "array_tasks_waiting": None,
    "state_reason": "None",
    "exit_code": 0,
    "exit_code_signal": 0,
    "allocated_nodes": "gl1234",
    "working_directory": "/home/alice/run",
    "command": "/home/alice/run/job.sh",
    "standard_output": "/home/alice/run/slurm-4242.out",
    "standard_error": "/home/alice/run/slurm-4242.out",
    "time_limit": 5,
    "memory": 200,
    "gpus": {},
    "priority": 1000,
    "dependencies": {"afterok": [4000], "afterany": []},
    "batch_host": "gl1234",
}

ACCOUNTING_DONE = {
    "id": 4242,
    "name": "train",
    "state": "COMPLETED",
    "user_name": "alice",
    "partition": "standard",
    "account": "acct",
    "qos": "normal",
    "submit_time": 1_800_000_000,
    "start_time": 1_800_000_100,
    "end_time": 1_800_000_190,
    "elapsed_time": 90,
    "num_nodes": 1,
    "cpus": 2,
    "array_id": None,
    "array_task_id": None,
    "array_tasks_waiting": None,
    "exit_code": 0,
    "exit_code_signal": 0,
    "cancelled_by": None,
    "nodelist": "gl1234",
    "working_directory": "/home/alice/run",
    "submit_command": "sbatch --time=5 job.sh",
    "standard_output": None,
    "standard_error": None,
    "script": "#!/bin/bash\n#SBATCH --job-name=train\nsleep 90\n",
    "time_limit": 5,
    "memory": 200,
    "gpus": {},
    "priority": 1000,
}

# Fields both sources populate; the two dumps of one job must agree on every one of them.
SHARED_SUMMARY_FIELDS = (
    "job_id",
    "name",
    "state",
    "user",
    "partition",
    "account",
    "submit_time",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "num_nodes",
    "num_cpus",
    "array_job_id",
    "array_task_id",
    "array_task_range",
)
SHARED_DETAIL_FIELDS = ("qos", "nodelist", "working_directory", "time_limit_minutes")


# slurmdbd has no TRES or nodes for a job that has not started; the controller reports the request.
PENDING_ONLY_DIFFERENCES = {"num_nodes", "num_cpus", "nodelist"}


def _shared_equal(controller: JobDetail, accounting: JobDetail, *, time_tolerance: int = 1):
    for field in SHARED_SUMMARY_FIELDS + SHARED_DETAIL_FIELDS:
        if controller.state == "PENDING" and field in PENDING_ONLY_DIFFERENCES:
            continue
        a, b = getattr(controller, field), getattr(accounting, field)
        if isinstance(a, datetime) and isinstance(b, datetime):
            assert abs((a - b).total_seconds()) <= time_tolerance, field
        elif field == "elapsed_seconds":
            assert abs(a - b) <= time_tolerance, field
        else:
            assert a == b, f"{field}: controller={a!r} accounting={b!r}"


# --- synthetic records ---------------------------------------------------------------------


def test_renamed_pairs_normalize_identically():
    c = controller_detail(CONTROLLER_DONE)
    a = accounting_detail(ACCOUNTING_DONE)
    assert c.source == Source.CONTROLLER and a.source == Source.ACCOUNTING
    _shared_equal(c, a)
    # run_time / elapsed_time and allocated_nodes / nodelist land in the same fields.
    assert c.elapsed_seconds == a.elapsed_seconds == 90
    assert c.nodelist == a.nodelist == "gl1234"
    assert c.dependencies == "afterok:4000"
    assert c.requested.time_limit_minutes == 5 and a.requested.memory_mb == 200


def test_controller_running_job_hides_deadline_and_reason():
    rec = {
        **CONTROLLER_DONE,
        "state": "RUNNING",
        "end_time": 1_800_000_400,  # controller's predicted deadline, not an end
        "exit_code": 0,
    }
    s = controller_summary(rec)
    assert s.end_time is None
    assert s.exit_code is None and s.state_reason is None
    assert s.is_active


def test_controller_pending_job_reason_and_no_start():
    rec = {
        **CONTROLLER_DONE,
        "state": "PENDING",
        "state_reason": "Priority",
        "start_time": 1_900_000_000,
        "run_time": 0,
    }
    s = controller_summary(rec)
    assert s.state_reason == "Priority" and s.start_time is None and s.elapsed_seconds == 0


def test_array_range_and_task_zero():
    pending = {**CONTROLLER_DONE, "id": 500, "state": "PENDING", "array_id": 500}
    pending["array_task_id"] = None
    pending["array_tasks_waiting"] = "3-6"
    r = controller_summary(pending)
    assert (r.array_job_id, r.array_task_id, r.array_task_range) == (500, None, "3-6")
    assert r.display_id == "500_[3-6]"
    # pyslurm reports task 0 as None (u32_parse treats 0 as no value); a taskless array record
    # without a pending range is task 0.
    zero = {**ACCOUNTING_DONE, "id": 501, "array_id": 500, "array_task_id": None}
    assert accounting_summary(zero).display_id == "500_0"
    seven = {**ACCOUNTING_DONE, "id": 507, "array_id": 500, "array_task_id": 7}
    assert accounting_summary(seven).display_id == "500_7"


def test_cancelled_by_only_for_cancelled():
    rec = {**ACCOUNTING_DONE, "state": "CANCELLED", "cancelled_by": "bob"}
    assert accounting_summary(rec).cancelled_by == "bob"
    assert accounting_summary({**ACCOUNTING_DONE, "cancelled_by": "bob"}).cancelled_by is None


def test_stdio_paths_from_submit_line_then_default():
    submitted = {
        **ACCOUNTING_DONE,
        "submit_command": "sbatch --output=logs/%x-%A_%a.out --error=logs/err-%5j.txt job.sh",
        "array_id": 4200,
        "array_task_id": 3,
    }
    d = accounting_detail(submitted)
    assert d.stdout_path == "/home/alice/run/logs/train-4200_3.out"
    assert d.stderr_path == "/home/alice/run/logs/err-04242.txt"
    # Nothing recorded anywhere: Slurm's default name, stderr merged into stdout.
    bare = {**ACCOUNTING_DONE, "submit_command": None, "script": None}
    d = accounting_detail(bare)
    assert d.stdout_path == "/home/alice/run/slurm-4242.out" == d.stderr_path
    array_bare = {**bare, "array_id": 4200, "array_task_id": 3}
    assert accounting_detail(array_bare).stdout_path == "/home/alice/run/slurm-4200_3.out"
    # #SBATCH lines win over the command line.
    scripted = {**submitted, "script": "#!/bin/bash\n#SBATCH -o /scratch/out.%j\n"}
    assert accounting_detail(scripted).stdout_path == "/scratch/out.4242"


def test_expand_stdio_leaves_unknown_node_pattern():
    path = expand_stdio(
        "%N-%j.out",
        job_id=1,
        array_job_id=None,
        array_task_id=None,
        user="u",
        name="n",
        first_node=None,
        working_directory="/w",
    )
    assert path == "/w/%N-1.out"


def test_gpu_count_from_dict_and_string():
    assert gpu_count({"gpus": {"gpu": {"count": 2}}}) == 2
    assert (
        gpu_count({"gpus": {}, "allocated_tres": "cpu=8,mem=120G,gres/gpu=1,gres/gpu:a40=1"}) == 2
    )
    assert gpu_count({"gpus": {}, "allocated_tres": "cpu=8,mem=120G"}) == 0


def test_parse_tres_usage_and_gpu_usage():
    assert parse_tres_usage("1=10405266,2=19803729920,1035=42") == {
        1: 10405266,
        2: 19803729920,
        1035: 42,
    }
    steps = [
        {"tres_usage_in_ave": "1=5,1035=0", "tres_usage_in_max": "2=100,1036=2097152"},
        {"tres_usage_in_ave": "3=0", "tres_usage_in_max": "3=0"},
    ]
    util, mem = gpu_usage_from_tres(steps, 1035, 1036)
    assert util == 0.0 and mem == 2.0  # a recorded zero utilization stays 0.0, not None
    assert gpu_usage_from_tres(steps, 9999, 9998) == (None, None)
    # pyslurm's legacy API exposes max/min/tot but not ave: fall back to the per-task peak.
    legacy = [{"tres_usage_in_max": "1=75,1036=0,1035=40", "tres_usage_in_tot": "1035=40"}]
    assert gpu_usage_from_tres(legacy, 1035, 1036) == (40.0, 0.0)


def test_steps_sampled_distinguishes_zero_record_from_data():
    zero_step = {"total_cpu_time": 0, "max_resident_memory": 0, "avg_cpu_time": 0}
    assert steps_sampled([zero_step, None]) is False
    assert steps_sampled([]) is False
    assert steps_sampled([zero_step, {"max_resident_memory": 4096}]) is True
    # Shorter than one sampling interval: the teardown poll's figures are not a sample.
    teardown = {"max_resident_memory": 2072576, "total_cpu_time": 0}
    assert steps_sampled([teardown], elapsed_seconds=1, interval=30) is False
    assert steps_sampled([teardown], elapsed_seconds=44, interval=30) is True


def test_task_range_from_bitmask():
    assert task_range_from_bitmask("0x1FFFFFFE000") == "13-40"
    assert task_range_from_bitmask("0x5") == "0,2"
    assert task_range_from_bitmask("0xE") == "1-3"
    pending = {**ACCOUNTING_DONE, "state": "PENDING", "array_id": 4242}
    pending["array_tasks_waiting"] = "0x1FFFFFFE000"
    assert accounting_summary(pending).display_id == "4242_[13-40]"


def test_build_usage_absent_vs_zero():
    common = dict(
        job_id=1,
        state="COMPLETED",
        cpus=2,
        elapsed_seconds=1,
        time_limit_minutes=5,
        memory_mb=200,
        gpus_allocated=0,
        total_cpu_seconds=0,
        peak_rss_bytes=0,
    )
    absent = build_usage(sampled=False, **common)
    assert absent.cpu.consumed is None and absent.memory.consumed is None
    assert absent.cpu.percent is None
    assert absent.walltime.consumed == 1.0 and absent.walltime.requested == 300.0
    zero_elapsed = build_usage(sampled=False, **{**common, "elapsed_seconds": 0})
    assert zero_elapsed.walltime.consumed == 0.0  # a clock reading, never an accounting sample
    assert absent.gpu_utilization is None and absent.gpu_accounting_available is None
    zero = build_usage(sampled=True, **common)
    assert zero.cpu.consumed == 0.0 and zero.cpu.percent == 0.0
    # GPU job on a cluster without GPU TRES: figures present but unavailable, never zero.
    gpu = build_usage(
        sampled=True, gpu_accounting_available=False, **{**common, "gpus_allocated": 1}
    )
    assert gpu.gpu_utilization is not None and gpu.gpu_utilization.consumed is None
    assert gpu.gpu_accounting_available is False
    sampled_gpu = build_usage(
        sampled=True,
        gpu_accounting_available=True,
        gpu_utilization=37.0,
        gpu_memory_mb=512.0,
        **{**common, "gpus_allocated": 1},
    )
    assert sampled_gpu.gpu_utilization.percent == 37.0 and sampled_gpu.gpu_memory.consumed == 512.0


# --- recorded fixtures ---------------------------------------------------------------------


def _scenarios() -> list[Path]:
    if not RECORDED.exists():
        return []
    return sorted(p for p in RECORDED.iterdir() if p.is_dir() and (p / "hello.json").exists())


def _load(path: Path):
    return json.loads(path.read_text())


def _as_list(payload):
    return payload if isinstance(payload, list) else [payload]


@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda p: p.name)
def test_recorded_fixture_files_validate(scenario: Path):
    Handshake.model_validate(_load(scenario / "hello.json"))
    if (scenario / "active_jobs.json").exists():
        snapshots = _load(scenario / "active_jobs.json")
        assert isinstance(snapshots, list) and snapshots
        for snap in snapshots:
            result = ActiveJobsResult.model_validate(snap)
            assert all(j.is_active for j in result.jobs)
    if (scenario / "history_jobs.json").exists():
        for snap in _load(scenario / "history_jobs.json"):
            result = HistoryJobsResult.model_validate(snap)
            for j in result.jobs:
                assert j.is_terminal and j.end_time is not None
                assert result.window_start <= j.end_time < result.window_end
    for path in sorted((scenario / "job_detail").glob("*.json")):
        detail = JobDetail.model_validate(_load(path))
        assert detail.job_id == int(path.stem)
        assert detail.stdout_path and detail.stderr_path
    for path in sorted((scenario / "job_usage").glob("*.json")):
        for payload in _as_list(_load(path)):
            usage = JobUsage.model_validate(payload)
            assert usage.job_id == int(path.stem)
            if not usage.sampled:
                assert usage.cpu.consumed is None and usage.memory.consumed is None
    for path in sorted((scenario / "cancel").glob("*.json")):
        result = CancelResult.model_validate(_load(path))
        assert result.cancelled


def _raw_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for scenario in _scenarios():
        raw = scenario / "raw"
        if not raw.exists():
            continue
        for ctrl in sorted(raw.glob("controller_*.json")):
            acct = raw / ctrl.name.replace("controller_", "accounting_", 1)
            if acct.exists():
                pairs.append((ctrl, acct))
    return pairs


@pytest.mark.parametrize(
    "pair", _raw_pairs(), ids=lambda p: f"{p[0].parent.parent.name}/{p[0].stem}"
)
def test_recorded_controller_and_accounting_agree(pair: tuple[Path, Path]):
    """Both dumps of one job, captured at the same moment, normalize to the same shared fields.
    State is compared only when both recordings saw the same state (a dump taken while the job
    was mid-transition differs by design)."""
    ctrl_raw, acct_raw = _load(pair[0]), _load(pair[1])
    c = controller_detail(ctrl_raw)
    a = accounting_detail(acct_raw)
    assert c.job_id == a.job_id
    if c.state != a.state:
        pytest.skip(f"state differs by design: controller={c.state} accounting={a.state}")
    _shared_equal(c, a)


def test_recorded_short_job_reports_no_samples():
    path = RECORDED / "short-job" / "job_usage"
    if not path.exists():
        pytest.skip("short-job fixture not recorded")
    for usage_file in path.glob("*.json"):
        for payload in _as_list(_load(usage_file)):
            usage = JobUsage.model_validate(payload)
            assert usage.sampled is False
            assert usage.cpu.consumed is None and usage.memory.consumed is None
            assert usage.walltime.requested == 300.0


def test_recorded_failed_job_exit_code():
    path = RECORDED / "failed-job" / "history_jobs.json"
    if not path.exists():
        pytest.skip("failed-job fixture not recorded")
    detail_ids = {int(p.stem) for p in (RECORDED / "failed-job" / "job_detail").glob("*.json")}
    failed = [
        j
        for snap in _load(path)
        for j in HistoryJobsResult.model_validate(snap).jobs
        if j.job_id in detail_ids
    ]
    assert failed and all(j.state == "FAILED" and j.exit_code == 3 for j in failed)


def test_recorded_cancelled_job_attribution():
    path = RECORDED / "cancelled-job" / "history_jobs.json"
    if not path.exists():
        pytest.skip("cancelled-job fixture not recorded")
    cancelled_ids = {int(p.stem) for p in (RECORDED / "cancelled-job" / "cancel").glob("*.json")}
    hits = [
        j
        for snap in _load(path)
        for j in HistoryJobsResult.model_validate(snap).jobs
        if j.job_id in cancelled_ids
    ]
    assert hits and all(j.state.startswith("CANCELLED") and j.cancelled_by for j in hits)


def test_recorded_completion_handoff_snapshots():
    """cancelled-job records the active set before and after the job left it."""
    path = RECORDED / "cancelled-job" / "active_jobs.json"
    if not path.exists():
        pytest.skip("cancelled-job fixture not recorded")
    snaps = [ActiveJobsResult.model_validate(s) for s in _load(path)]
    cancelled_ids = {int(p.stem) for p in (RECORDED / "cancelled-job" / "cancel").glob("*.json")}
    before = {j.job_id for j in snaps[0].jobs}
    after = {j.job_id for j in snaps[-1].jobs}
    assert cancelled_ids <= before and not (cancelled_ids & after)


def test_recorded_array_range_coexists_with_running_tasks():
    path = RECORDED / "array-range" / "active_jobs.json"
    if not path.exists():
        pytest.skip("array-range fixture not recorded")
    first = ActiveJobsResult.model_validate(_load(path)[0])
    ranges = [j for j in first.jobs if j.array_task_range is not None]
    tasks = [j for j in first.jobs if j.array_task_id is not None and j.state == "RUNNING"]
    assert ranges and tasks
    assert {r.array_job_id for r in ranges} & {t.array_job_id for t in tasks}
