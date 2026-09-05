"""Client-side session: one transport wrapped with connection-state tracking and retry policy.

The TUI talks only to ClientSession. It never sees the transport, ssh, or pyslurm.

Behavior contract (see specs/cluster-connection):
- start() opens the transport and completes the handshake. A protocol/Slurm version mismatch
  raises VersionMismatch (message names `slurm-monitor bootstrap` as the remedy) and the session
  does not serve requests. BootstrapRequired and AuthError propagate unchanged: neither retries.
- Any request that fails with TransportError marks the session RECONNECTING and schedules a
  reconnect with increasing delay (backoff sequence is configurable for tests). While reconnecting,
  requests raise TransportError immediately so the caller keeps showing its last data.
- AuthError during reconnect moves the state to AUTH_REQUIRED and stops retrying; `last_error`
  carries the instruction to authenticate in a separate terminal.
- `degraded` mirrors the handshake's extension_available flag.
- `on_state_change(callback)` registers a callback invoked with the new ConnectionState.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

from slurm_monitor.config import Resolved
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    ConnectionState,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobUsage,
)
from slurm_monitor.transport import Transport

DEFAULT_BACKOFF: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)


class ClientSession:
    """Implemented in this module by the transports work package. This stub fixes the
    interface so the TUI can be written against it."""

    def __init__(
        self,
        transport_factory: Callable[[], Transport],
        backoff: Sequence[float] = DEFAULT_BACKOFF,
    ) -> None:
        raise NotImplementedError

    state: ConnectionState
    handshake: Handshake | None
    degraded: bool
    last_error: str | None
    last_success: datetime | None  # wall-clock time of the last successful response

    def on_state_change(self, callback: Callable[[ConnectionState], None]) -> None: ...

    async def start(self) -> Handshake: ...

    async def stop(self) -> None: ...

    async def active_jobs(self) -> ActiveJobsResult: ...

    async def history_jobs(
        self, since: datetime, until: datetime | None = None
    ) -> HistoryJobsResult: ...

    async def job_detail(self, job_id: int) -> JobDetail: ...

    async def job_usage(self, job_id: int) -> JobUsage: ...

    async def cancel(self, job_id: int, array_task_id: int | None = None) -> CancelResult: ...


def make_transport(settings: Resolved) -> Transport:
    """Pick LocalTransport when settings.cluster.host is None, else SshTransport."""
    raise NotImplementedError


def run_check(settings: Resolved) -> int:
    """`slurm-monitor check`: connect, print the handshake as key/value lines, exit 0; on
    BootstrapRequired / VersionMismatch / AuthError print the instruction and exit 1."""
    raise NotImplementedError
