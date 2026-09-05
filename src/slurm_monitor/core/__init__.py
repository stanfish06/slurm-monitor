"""Agent core: every query the agent can answer.

Importing this package must not require pyslurm. The pyslurm-backed implementation lives in
core.queries (PyslurmCore) and is imported lazily by the agent, so the client package installs
anywhere. CoreAPI is the seam shared by the agent process (agent.py), the in-process transport
(transport/local.py), and the fixture recorder (core/recording.py).

All methods are synchronous and may block on RPCs; callers run them in a worker thread.
Errors: raise CoreError(code, message) with a code from slurm_monitor.protocol (ERR_*).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobUsage,
)


class CoreError(Exception):
    def __init__(self, code: str, message: str, data: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}


class CoreAPI(ABC):
    @abstractmethod
    def hello(self) -> Handshake:
        """Environment and version facts; sent unsolicited as the first frame."""

    @abstractmethod
    def active_jobs(self) -> ActiveJobsResult:
        """The invoking user's pending/running jobs from slurmctld. Terminal jobs the
        controller still holds are excluded. degraded=True when served by the unfiltered path."""

    @abstractmethod
    def history_jobs(self, since: datetime, until: datetime | None = None) -> HistoryJobsResult:
        """The user's terminal jobs from slurmdbd with end_time in [since, until]."""

    @abstractmethod
    def job_detail(self, job_id: int) -> JobDetail:
        """Full detail for one job, from the controller if it still holds it, else accounting."""

    @abstractmethod
    def job_usage(self, job_id: int) -> JobUsage:
        """Efficiency figures. sampled=False when accounting has no samples yet."""

    @abstractmethod
    def cancel(self, job_id: int, array_task_id: int | None = None) -> CancelResult:
        """Cancel one job, one array task (job_id = array job id, array_task_id set), or a whole
        array (job_id = array job id, array_task_id None). Raises CoreError(ERR_NOT_CANCELLABLE)
        if the target is already terminal, CoreError(ERR_RPC) if slurmctld refuses."""


# Method name -> (CoreAPI method, param names) for dispatch in agent.py and transport/local.py.
METHODS: dict[str, tuple[str, tuple[str, ...]]] = {
    "hello": ("hello", ()),
    "active_jobs": ("active_jobs", ()),
    "history_jobs": ("history_jobs", ("since", "until")),
    "job_detail": ("job_detail", ("job_id",)),
    "job_usage": ("job_usage", ("job_id",)),
    "cancel": ("cancel", ("job_id", "array_task_id")),
}


def dispatch(core: CoreAPI, method: str, params: dict[str, Any]) -> Any:
    """Call a CoreAPI method by protocol name. Datetime params arrive as ISO strings."""
    if method not in METHODS:
        raise CoreError("bad_request", f"unknown method {method!r}")
    attr, names = METHODS[method]
    unknown = set(params) - set(names)
    if unknown:
        raise CoreError("bad_request", f"unexpected params for {method}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name in names:
        if name in params:
            value = params[name]
            if name in ("since", "until") and isinstance(value, str):
                value = datetime.fromisoformat(value)
            kwargs[name] = value
    return getattr(core, attr)(**kwargs)
