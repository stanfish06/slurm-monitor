"""In-memory CoreAPI for agent tests. Logs and prints on purpose to prove stdout stays clean."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from slurm_monitor import PROTOCOL_VERSION
from slurm_monitor.core import CoreAPI, CoreError
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobSummary,
    JobUsage,
    Measure,
    Source,
)
from slurm_monitor.protocol import ERR_NOT_FOUND

log = logging.getLogger("tests.fake_core")

LOG_MARKER = "fake-core-log-line"
PRINT_MARKER = "fake-core-stray-print"


class FakeCore(CoreAPI):
    def __init__(self, built: str = "25.11.5", running: str = "25.11.5") -> None:
        self.built = built
        self.running = running
        self.calls: list[str] = []

    def hello(self) -> Handshake:
        self.calls.append("hello")
        return Handshake(
            protocol_version=PROTOCOL_VERSION,
            agent_version="0.0-test",
            user="tester",
            hostname="fakehost",
            pyslurm_version="25.11.2",
            slurm_built_version=self.built,
            slurm_running_version=self.running,
            extension_available=True,
        )

    def active_jobs(self) -> ActiveJobsResult:
        self.calls.append("active_jobs")
        log.info(LOG_MARKER)
        print(PRINT_MARKER)  # a stray print inside the core must not reach the frame stream
        job = JobSummary(job_id=42, name="fake", state="RUNNING", source=Source.CONTROLLER)
        return ActiveJobsResult(jobs=[job], fetched_at=datetime(2026, 1, 1, tzinfo=UTC))

    def history_jobs(self, since: datetime, until: datetime | None = None) -> HistoryJobsResult:
        self.calls.append("history_jobs")
        return HistoryJobsResult(
            jobs=[], window_start=since, window_end=until or since, fetched_at=since
        )

    def job_detail(self, job_id: int) -> JobDetail:
        self.calls.append("job_detail")
        if job_id == 999:
            raise CoreError(ERR_NOT_FOUND, f"job {job_id} unknown", {"job_id": job_id})
        return JobDetail(job_id=job_id, name="fake", state="RUNNING", source=Source.CONTROLLER)

    def job_usage(self, job_id: int) -> JobUsage:
        self.calls.append("job_usage")
        if job_id == 500:
            raise RuntimeError("boom")
        m = Measure(unit="seconds")
        return JobUsage(job_id=job_id, state="RUNNING", sampled=False, cpu=m, memory=m, walltime=m)

    def cancel(self, job_id: int, array_task_id: int | None = None) -> CancelResult:
        self.calls.append("cancel")
        return CancelResult(job_id=job_id, array_task_id=array_task_id, cancelled=True)
