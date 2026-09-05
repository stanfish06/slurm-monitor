"""Shared data models.

Every payload that crosses the client/agent boundary is one of these pydantic models, so the
in-process transport, the SSH transport, and the fixture transport all speak the same shapes.
Controller records (pyslurm.Job) and accounting records (pyslurm.db.Job) both normalize onto
JobSummary / JobDetail; see core/normalize.py.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Slurm job states as pyslurm reports them (Job.state / db.Job.state strings).
ACTIVE_STATES: frozenset[str] = frozenset(
    {
        "PENDING",
        "RUNNING",
        "SUSPENDED",
        "COMPLETING",
        "CONFIGURING",
        "REQUEUED",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
    }
)
TERMINAL_STATES: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "CANCELLED",
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
        "SPECIAL_EXIT",
        "LAUNCH_FAILED",
    }
)


def is_active_state(state: str) -> bool:
    """True for states the Active tab shows (pending/running and their transitional forms)."""
    return state.split()[0].upper() in ACTIVE_STATES


def is_terminal_state(state: str) -> bool:
    """True once a job can no longer be cancelled. "CANCELLED by 123" counts as CANCELLED."""
    return state.split()[0].upper() in TERMINAL_STATES


class Source(StrEnum):
    """Where a record came from. Controller = slurmctld (live), accounting = slurmdbd."""

    CONTROLLER = "controller"
    ACCOUNTING = "accounting"


class ResourceRequest(BaseModel):
    """What the job asked for. memory_mb is the total across all nodes when known."""

    model_config = ConfigDict(extra="forbid")

    cpus: int | None = None
    nodes: int | None = None
    memory_mb: int | None = None
    gpus: int | None = None
    gres: str | None = None  # raw TRES/GRES string, e.g. "gres/gpu:a100=2"
    time_limit_minutes: int | None = None


class JobSummary(BaseModel):
    """One row in either table. Shared fields are populated from both sources."""

    model_config = ConfigDict(extra="forbid")

    job_id: int  # raw Slurm job id (for array tasks, the task's own id)
    name: str
    state: str  # e.g. "PENDING", "RUNNING", "CANCELLED"
    user: str | None = None
    partition: str | None = None
    account: str | None = None
    source: Source
    submit_time: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    elapsed_seconds: int = 0
    num_nodes: int | None = None
    num_cpus: int | None = None
    # Array membership. A pending array shows up as one record with task_range set
    # (e.g. "1-10,12"); once tasks run, each becomes its own record with array_task_id set.
    array_job_id: int | None = None
    array_task_id: int | None = None
    array_task_range: str | None = None
    # Controller-only
    state_reason: str | None = None  # pending reason, e.g. "Priority", "Resources"
    # Accounting-only
    exit_code: int | None = None
    exit_signal: int | None = None
    cancelled_by: str | None = None
    failed_node: str | None = None

    @property
    def is_array_member(self) -> bool:
        return self.array_job_id is not None and (
            self.array_task_id is not None or self.array_task_range is not None
        )

    @property
    def display_id(self) -> str:
        """Slurm-style id: 123 / 123_4 / 123_[1-10]."""
        if self.array_job_id is not None:
            if self.array_task_id is not None:
                return f"{self.array_job_id}_{self.array_task_id}"
            if self.array_task_range is not None:
                return f"{self.array_job_id}_[{self.array_task_range}]"
        return str(self.job_id)

    @property
    def is_active(self) -> bool:
        return is_active_state(self.state)

    @property
    def is_terminal(self) -> bool:
        return is_terminal_state(self.state)


class JobDetail(JobSummary):
    """Everything the detail panel shows, for a job from either source."""

    qos: str | None = None
    nodelist: str | None = None  # allocated nodes (controller: allocated_nodes; db: nodelist)
    requested: ResourceRequest = Field(default_factory=ResourceRequest)
    working_directory: str | None = None
    command: str | None = None
    stdout_path: str | None = None  # resolved (%j etc. expanded) where Slurm reports it
    stderr_path: str | None = None
    priority: int | None = None
    time_limit_minutes: int | None = None
    dependencies: str | None = None
    batch_host: str | None = None


class Measure(BaseModel):
    """A consumed-vs-requested pair. consumed=None means no accounting sample exists yet."""

    model_config = ConfigDict(extra="forbid")

    unit: str  # "cpu-seconds", "MB", "seconds", "percent"
    consumed: float | None = None
    requested: float | None = None

    @property
    def percent(self) -> float | None:
        if self.consumed is None or not self.requested:
            return None
        return 100.0 * self.consumed / self.requested


class JobUsage(BaseModel):
    """Efficiency figures for one job. GPU fields are None unless the job holds a GPU."""

    model_config = ConfigDict(extra="forbid")

    job_id: int
    state: str
    sampled: bool  # False when accounting has recorded no samples for this job
    sampled_at: datetime | None = None
    cpu: Measure  # cpu-seconds consumed vs allocated cpus * elapsed
    memory: Measure  # peak RSS MB vs requested MB
    walltime: Measure  # elapsed seconds vs time limit seconds
    gpus_allocated: int = 0
    gpu_utilization: Measure | None = None  # percent; None when no GPU allocated
    gpu_memory: Measure | None = None  # MB; None when no GPU allocated
    gpu_accounting_available: bool | None = None  # False when the cluster records no GPU TRES


class ArrayTaskCounts(BaseModel):
    """Per-state task histogram for a collapsed array row."""

    model_config = ConfigDict(extra="forbid")

    counts: dict[str, int] = Field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class ConnectionState(StrEnum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    AUTH_REQUIRED = "auth_required"


class Handshake(BaseModel):
    """First frame the agent sends. The client refuses to proceed on a version mismatch."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: int
    agent_version: str
    user: str
    hostname: str
    pyslurm_version: str
    slurm_built_version: str  # what pyslurm was compiled against, e.g. "25.11.5"
    slurm_running_version: str  # what slurmctld reports, e.g. "25.11.5"
    extension_available: bool  # slurm_load_job_user() binding importable
    extension_error: str | None = None
    sampling_interval_seconds: int = 30  # JobAcctGatherFrequency task=


def versions_compatible(built: str, running: str) -> bool:
    """pyslurm masks the patch component, so 25.11.5 and 25.11.6 are interchangeable."""
    b = built.split(".")[:2]
    r = running.split(".")[:2]
    return b == r


# --- Request / response payloads --------------------------------------------------------------

Method = Literal["hello", "active_jobs", "history_jobs", "job_detail", "job_usage", "cancel"]


class ActiveJobsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[JobSummary]
    degraded: bool = False  # True when served by the unfiltered fallback
    fetched_at: datetime


class HistoryJobsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[JobSummary]
    window_start: datetime
    window_end: datetime
    fetched_at: datetime


class CancelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: int
    array_task_id: int | None = None
    cancelled: bool
    message: str | None = None
