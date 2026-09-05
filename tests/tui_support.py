"""Shared scaffolding for the TUI tests: a scriptable FakeSession and hand-written job data.

FakeSession implements the ClientSession interface (see slurm_monitor.session) with snapshot
queues: each call to active_jobs()/history_jobs() pops the next queued result and keeps
returning the last one once the queue is exhausted.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from slurm_monitor.config import ClusterConfig, HistoryConfig, Intervals, Resolved
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    ConnectionState,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobSummary,
    JobUsage,
    Measure,
    ResourceRequest,
    Source,
)
from slurm_monitor.transport import AgentError, TransportError
from slurm_monitor.tui.app import SlurmMonitorApp

TZ = timezone(timedelta(hours=-4))
T0 = datetime(2026, 9, 5, 12, 0, 0, tzinfo=TZ)


class FakeClock:
    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def handshake(extension_available: bool = True) -> Handshake:
    return Handshake(
        protocol_version=1,
        agent_version="0.1.0",
        user="tester",
        hostname="login-node",
        pyslurm_version="25.11.2",
        slurm_built_version="25.11.5",
        slurm_running_version="25.11.5",
        extension_available=extension_available,
        extension_error=None if extension_available else "ImportError: libslurmfull.so",
    )


def active_result(jobs: list[JobSummary], degraded: bool = False) -> ActiveJobsResult:
    return ActiveJobsResult(jobs=jobs, degraded=degraded, fetched_at=T0)


def history_result(jobs: list[JobSummary], since: datetime, until: datetime) -> HistoryJobsResult:
    return HistoryJobsResult(jobs=jobs, window_start=since, window_end=until, fetched_at=T0)


def job(job_id: int, state: str = "RUNNING", **kw) -> JobSummary:
    kw.setdefault("name", f"job{job_id}")
    kw.setdefault("source", Source.CONTROLLER)
    kw.setdefault("partition", "standard")
    return JobSummary(job_id=job_id, state=state, **kw)


def hist_job(job_id: int, state: str = "COMPLETED", **kw) -> JobSummary:
    kw.setdefault("source", Source.ACCOUNTING)
    kw.setdefault("end_time", T0 - timedelta(hours=1))
    return job(job_id, state, **kw)


# --- Sample data ----------------------------------------------------------------------------

ACTIVE_JOBS: list[JobSummary] = [
    job(
        1001,
        "RUNNING",
        name="train",
        partition="gpu",
        elapsed_seconds=3725,
        num_nodes=1,
        num_cpus=4,
        submit_time=T0 - timedelta(hours=2),
        start_time=T0 - timedelta(seconds=3725),
    ),
    job(
        1002,
        "PENDING",
        name="preprocess",
        partition="standard",
        elapsed_seconds=0,
        num_nodes=2,
        state_reason="Priority",
        submit_time=T0 - timedelta(minutes=10),
    ),
    job(
        1003,
        "RUNNING",
        name="long",
        partition="largemem",
        elapsed_seconds=90061,  # 1 day, 1h 1m 1s
        num_nodes=1,
    ),
]

MIXED_ARRAY_JOBS: list[JobSummary] = [
    job(123, "PENDING", name="sweep", array_job_id=123, array_task_range="3-6"),
    job(124, "RUNNING", name="sweep", array_job_id=123, array_task_id=1, elapsed_seconds=120),
    job(125, "RUNNING", name="sweep", array_job_id=123, array_task_id=2, elapsed_seconds=90),
]

HISTORY_JOBS: list[JobSummary] = [
    hist_job(
        900,
        "COMPLETED",
        name="done",
        partition="standard",
        elapsed_seconds=600,
        exit_code=0,
        exit_signal=0,
    ),
    hist_job(
        901,
        "CANCELLED",
        name="oops",
        partition="gpu",
        elapsed_seconds=42,
        exit_code=0,
        exit_signal=15,
        cancelled_by="tester",
    ),
    hist_job(
        902,
        "NODE_FAIL",
        name="crash",
        partition="standard",
        elapsed_seconds=7200,
        exit_code=1,
        failed_node="gl3021",
    ),
]

OLDER_HISTORY_JOBS: list[JobSummary] = [
    hist_job(
        800,
        "COMPLETED",
        name="ancient",
        elapsed_seconds=10,
        exit_code=0,
        end_time=T0 - timedelta(days=9),
    ),
]


def detail_for(summary: JobSummary) -> JobDetail:
    base = summary.model_dump()
    base["account"] = "lab-account"
    return JobDetail(
        **base,
        qos="normal",
        nodelist="gl1520" if summary.is_active and summary.state == "RUNNING" else None,
        requested=ResourceRequest(
            cpus=4,
            nodes=summary.num_nodes or 1,
            memory_mb=16000,
            gpus=1 if summary.partition == "gpu" else None,
            gres="gres/gpu:a100=1" if summary.partition == "gpu" else None,
            time_limit_minutes=120,
        ),
        working_directory=f"/home/tester/run{summary.job_id}",
        stdout_path=f"/home/tester/run{summary.job_id}/slurm-{summary.job_id}.out",
        stderr_path=f"/home/tester/run{summary.job_id}/slurm-{summary.job_id}.err",
        time_limit_minutes=120,
    )


# cpu: 1800 cpu-s of 4 cpus * 900 s = 3600 -> 50.0%; mem 4000/16000 -> 25.0%; wall 900/7200.
CPU_USAGE = JobUsage(
    job_id=1001,
    state="RUNNING",
    sampled=True,
    cpu=Measure(unit="cpu-seconds", consumed=1800.0, requested=3600.0),
    memory=Measure(unit="MB", consumed=4000.0, requested=16000.0),
    walltime=Measure(unit="seconds", consumed=900.0, requested=7200.0),
)

GPU_USAGE = CPU_USAGE.model_copy(
    update={
        "gpus_allocated": 1,
        "gpu_utilization": Measure(unit="percent", consumed=87.5, requested=100.0),
        "gpu_memory": Measure(unit="MB", consumed=20480.0, requested=40960.0),
        "gpu_accounting_available": True,
    }
)

GPU_NO_ACCOUNTING_USAGE = GPU_USAGE.model_copy(
    update={"gpu_utilization": None, "gpu_memory": None, "gpu_accounting_available": False}
)

SUB_INTERVAL_USAGE = JobUsage(
    job_id=1003,
    state="RUNNING",
    sampled=False,
    cpu=Measure(unit="cpu-seconds", requested=80.0),
    memory=Measure(unit="MB", requested=4000.0),
    walltime=Measure(unit="seconds", consumed=20.0, requested=3600.0),
)


class FakeSession:
    """ClientSession stand-in with scriptable results and call recording."""

    def __init__(
        self,
        active: list[ActiveJobsResult] | None = None,
        history: list[HistoryJobsResult] | None = None,
        details: dict[int, JobDetail] | None = None,
        usages: dict[int, JobUsage] | None = None,
        hs: Handshake | None = None,
    ) -> None:
        self.state = ConnectionState.CONNECTING
        self.handshake: Handshake | None = hs or handshake()
        self.degraded = not self.handshake.extension_available
        self.last_error: str | None = None
        self.last_success: datetime | None = None
        self.clock: Callable[[], datetime] = FakeClock()
        self._callbacks: list[Callable[[ConnectionState], None]] = []
        self.active_queue = list(active or [])
        self.history_queue = list(history or [])
        self.details = details or {}
        self.usages = usages or {}
        self.calls: dict[str, list] = {
            "active_jobs": [],
            "history_jobs": [],
            "job_detail": [],
            "job_usage": [],
            "cancel": [],
            "reconnect_now": [],
        }
        self.cancel_error: Exception | None = None
        self._last_active: ActiveJobsResult | None = None
        self._last_history: HistoryJobsResult | None = None
        self.started = False
        self.stopped = False

    # -- interface -----------------------------------------------------------------------------

    def on_state_change(self, callback: Callable[[ConnectionState], None]) -> None:
        self._callbacks.append(callback)

    async def start(self) -> Handshake:
        self.started = True
        self.set_state(ConnectionState.CONNECTED)
        assert self.handshake is not None
        return self.handshake

    async def stop(self) -> None:
        self.stopped = True

    async def reconnect_now(self) -> None:
        self.calls["reconnect_now"].append(None)

    async def active_jobs(self) -> ActiveJobsResult:
        self.calls["active_jobs"].append(None)
        self._check_connected()
        if self.active_queue:
            self._last_active = self.active_queue.pop(0)
        if self._last_active is None:
            self._last_active = active_result([])
        self._touch()
        return self._last_active

    async def history_jobs(
        self, since: datetime, until: datetime | None = None
    ) -> HistoryJobsResult:
        self.calls["history_jobs"].append((since, until))
        self._check_connected()
        if self.history_queue:
            self._last_history = self.history_queue.pop(0)
        if self._last_history is None:
            self._last_history = history_result([], since, until or since)
        self._touch()
        return self._last_history

    async def job_detail(self, job_id: int) -> JobDetail:
        self.calls["job_detail"].append(job_id)
        self._check_connected()
        if job_id not in self.details:
            raise AgentError("not_found", f"job {job_id} not found")
        self._touch()
        return self.details[job_id]

    async def job_usage(self, job_id: int) -> JobUsage:
        self.calls["job_usage"].append(job_id)
        self._check_connected()
        if job_id not in self.usages:
            raise AgentError("not_found", f"job {job_id} not found")
        self._touch()
        return self.usages[job_id]

    async def cancel(self, job_id: int, array_task_id: int | None = None) -> CancelResult:
        self.calls["cancel"].append((job_id, array_task_id))
        self._check_connected()
        if self.cancel_error is not None:
            raise self.cancel_error
        self._touch()
        return CancelResult(job_id=job_id, array_task_id=array_task_id, cancelled=True)

    # -- test controls -------------------------------------------------------------------------

    def set_state(self, state: ConnectionState, error: str | None = None) -> None:
        self.state = state
        self.last_error = error
        for cb in self._callbacks:
            cb(state)

    def _check_connected(self) -> None:
        if self.state != ConnectionState.CONNECTED:
            raise TransportError(self.last_error or f"session is {self.state.value}")

    def _touch(self) -> None:
        self.last_success = self.clock()


# --- App construction -------------------------------------------------------------------------


def make_settings(**kw) -> Resolved:
    intervals = Intervals(**{k: v for k, v in kw.items() if k.endswith("_seconds")})
    history = HistoryConfig(window_days=kw.get("window_days", 7))
    return Resolved(
        cluster_name=kw.get("cluster_name", "testcluster"),
        cluster=ClusterConfig(),
        intervals=intervals,
        history=history,
    )


def make_app(session: FakeSession, clock: FakeClock | None = None, **kw) -> SlurmMonitorApp:
    """App wired to `session` with a shared fake clock and timers off unless asked."""
    clock = clock or FakeClock()
    session.clock = clock
    start_timers = kw.pop("start_timers", False)
    return SlurmMonitorApp(session, make_settings(**kw), clock=clock, start_timers=start_timers)


def standard_session(**kw) -> FakeSession:
    """Three plain jobs plus the mixed array in Active; three terminal jobs in History."""
    active_jobs = kw.pop("active_jobs", ACTIVE_JOBS + MIXED_ARRAY_JOBS)
    details = {j.job_id: detail_for(j) for j in ACTIVE_JOBS + HISTORY_JOBS + MIXED_ARRAY_JOBS}
    usages = {
        1001: GPU_USAGE,
        1002: CPU_USAGE.model_copy(update={"job_id": 1002, "state": "PENDING"}),
        1003: SUB_INTERVAL_USAGE,
        900: CPU_USAGE.model_copy(update={"job_id": 900, "state": "COMPLETED"}),
        123: CPU_USAGE.model_copy(update={"job_id": 123}),
    }
    return FakeSession(
        active=[active_result(active_jobs)],
        history=[history_result(HISTORY_JOBS, T0 - timedelta(days=7), T0)],
        details=details,
        usages=usages,
        **kw,
    )
