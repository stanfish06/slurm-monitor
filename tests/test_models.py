from datetime import UTC, datetime

from slurm_monitor.models import (
    Handshake,
    JobDetail,
    JobSummary,
    JobUsage,
    Measure,
    ResourceRequest,
    Source,
    versions_compatible,
)


def _roundtrip(model):
    cls = type(model)
    again = cls.model_validate_json(model.model_dump_json())
    assert again == model
    again2 = cls.model_validate(model.model_dump(mode="json"))
    assert again2 == model
    return again


def test_job_summary_roundtrip_and_ids():
    j = JobSummary(
        job_id=101,
        name="n",
        state="PENDING",
        source=Source.CONTROLLER,
        submit_time=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
        array_job_id=100,
        array_task_range="1-10",
        state_reason="Priority",
    )
    _roundtrip(j)
    assert j.display_id == "100_[1-10]"
    assert j.is_active and not j.is_terminal
    t = j.model_copy(update={"array_task_range": None, "array_task_id": 4, "state": "RUNNING"})
    assert t.display_id == "100_4"
    plain = JobSummary(job_id=7, name="x", state="CANCELLED by 1000", source=Source.ACCOUNTING)
    assert plain.display_id == "7"
    assert plain.is_terminal and not plain.is_active


def test_job_detail_roundtrip():
    d = JobDetail(
        job_id=5,
        name="train",
        state="RUNNING",
        source=Source.CONTROLLER,
        account="acct",
        qos="normal",
        nodelist="gl1234",
        partition="gpu",
        requested=ResourceRequest(
            cpus=4, nodes=1, memory_mb=16000, gpus=1, gres="gres/gpu:1", time_limit_minutes=120
        ),
        working_directory="/home/u",
        stdout_path="/home/u/slurm-5.out",
        stderr_path="/home/u/slurm-5.out",
        time_limit_minutes=120,
    )
    _roundtrip(d)
    # A detail is also a summary: the table row can be built from it.
    JobSummary.model_validate(d.model_dump(include=set(JobSummary.model_fields)))


def test_usage_roundtrip_and_absent_sample():
    u = JobUsage(
        job_id=5,
        state="RUNNING",
        sampled=False,
        cpu=Measure(unit="cpu-seconds", requested=400.0),
        memory=Measure(unit="MB", requested=16000.0),
        walltime=Measure(unit="seconds", consumed=100.0, requested=7200.0),
    )
    _roundtrip(u)
    assert u.cpu.percent is None  # absent, never 0
    assert u.walltime.percent == 100.0 * 100 / 7200
    zero = Measure(unit="percent", consumed=0.0, requested=100.0)
    assert zero.percent == 0.0  # a recorded zero is a real zero


def test_handshake_roundtrip_and_version_rule():
    h = Handshake(
        protocol_version=1,
        agent_version="0.1.0",
        user="u",
        hostname="h",
        pyslurm_version="25.11.2",
        slurm_built_version="25.11.5",
        slurm_running_version="25.11.6",
        extension_available=False,
        extension_error="ImportError: x",
    )
    _roundtrip(h)
    assert versions_compatible("25.11.5", "25.11.6")
    assert not versions_compatible("25.11.5", "25.5.0")
    assert not versions_compatible("24.11.0", "25.11.5")
