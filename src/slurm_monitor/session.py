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
- `last_success` is the aware-UTC wall-clock time of the last successful response; the UI
  subtracts it from datetime.now(UTC) to show data age.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from slurm_monitor import PROTOCOL_VERSION
from slurm_monitor.config import Resolved
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    ConnectionState,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobUsage,
    versions_compatible,
)
from slurm_monitor.protocol import ERR_VERSION_MISMATCH
from slurm_monitor.transport import (
    AgentError,
    AuthError,
    BootstrapRequired,
    Transport,
    TransportError,
    VersionMismatch,
)

logger = logging.getLogger("slurm_monitor.session")

DEFAULT_BACKOFF: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
BOOTSTRAP_HINT = "re-run `slurm-monitor bootstrap --force`"


def auth_instruction(host: str | None) -> str:
    target = host or "<host>"
    return (
        f"authentication required: run `ssh {target}` in a separate terminal, complete "
        "authentication, then press r to reconnect"
    )


def check_handshake(h: Handshake) -> None:
    """Raise VersionMismatch unless this client can trust the agent's data."""
    if h.protocol_version != PROTOCOL_VERSION:
        raise VersionMismatch(
            f"agent speaks protocol version {h.protocol_version}, this client speaks "
            f"{PROTOCOL_VERSION}; {BOOTSTRAP_HINT} to reinstall the agent"
        )
    if not versions_compatible(h.slurm_built_version, h.slurm_running_version):
        raise VersionMismatch(
            f"agent was built against Slurm {h.slurm_built_version} but the cluster runs "
            f"{h.slurm_running_version}; {BOOTSTRAP_HINT} to rebuild the agent"
        )


class ClientSession:
    def __init__(
        self,
        transport_factory: Callable[[], Transport],
        backoff: Sequence[float] = DEFAULT_BACKOFF,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        host: str | None = None,
    ) -> None:
        if not backoff:
            raise ValueError("backoff needs at least one delay")
        self._factory = transport_factory
        self._backoff = tuple(backoff)
        self._sleep = sleep
        self._now = now
        self._host = host
        self._transport: Transport | None = None
        self._callbacks: list[Callable[[ConnectionState], None]] = []
        self._reconnect_task: asyncio.Task[None] | None = None
        self._stopped = False
        self.state: ConnectionState = ConnectionState.DISCONNECTED
        self.handshake: Handshake | None = None
        self.degraded: bool = False
        self.last_error: str | None = None
        self.last_success: datetime | None = None

    # --- state ----------------------------------------------------------------------------------

    def on_state_change(self, callback: Callable[[ConnectionState], None]) -> None:
        self._callbacks.append(callback)

    def _set_state(self, state: ConnectionState) -> None:
        if state == self.state:
            return
        self.state = state
        for cb in list(self._callbacks):
            try:
                cb(state)
            except Exception:  # a UI callback bug must not take the session down
                logger.exception("state callback failed")

    # --- lifecycle ------------------------------------------------------------------------------

    async def start(self) -> Handshake:
        self._stopped = False
        self._set_state(ConnectionState.CONNECTING)
        try:
            return await self._connect()
        except AuthError:
            self.last_error = auth_instruction(self._host)
            self._set_state(ConnectionState.AUTH_REQUIRED)
            raise
        except TransportError as e:
            self.last_error = str(e)
            self._set_state(ConnectionState.DISCONNECTED)
            raise

    async def stop(self) -> None:
        self._stopped = True
        await self._cancel_reconnect()
        await self._close_transport()
        self._set_state(ConnectionState.DISCONNECTED)

    async def _connect(self) -> Handshake:
        """Open a fresh transport and validate its handshake; on any failure it is closed."""
        transport = self._factory()
        try:
            h = await transport.start()
            check_handshake(h)
        except BaseException:
            await _safe_close(transport)
            raise
        self._transport = transport
        self.handshake = h
        self.degraded = not h.extension_available
        self.last_error = None
        self.last_success = self._now()
        self._set_state(ConnectionState.CONNECTED)
        return h

    async def _close_transport(self) -> None:
        transport, self._transport = self._transport, None
        if transport is not None:
            await _safe_close(transport)

    async def _cancel_reconnect(self) -> None:
        task, self._reconnect_task = self._reconnect_task, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # --- retry policy ---------------------------------------------------------------------------

    def _on_transport_failure(self, exc: TransportError) -> None:
        """Called on a retryable failure; the first caller starts the reconnect loop."""
        if self._stopped or self.state == ConnectionState.RECONNECTING:
            return
        self.last_error = str(exc)
        self._set_state(ConnectionState.RECONNECTING)
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        await self._close_transport()
        attempt = 0
        while not self._stopped:
            delay = self._backoff[min(attempt, len(self._backoff) - 1)]
            await self._sleep(delay)
            if self._stopped:
                return
            if await self._try_connect():
                return
            attempt += 1

    async def _try_connect(self) -> bool:
        """One connection attempt. Returns True when connected or when retrying must stop."""
        try:
            await self._connect()
            return True
        except AuthError:
            self.last_error = auth_instruction(self._host)
            self._set_state(ConnectionState.AUTH_REQUIRED)
            return True
        except (BootstrapRequired, VersionMismatch) as e:
            self.last_error = str(e)
            self._set_state(ConnectionState.DISCONNECTED)
            return True
        except TransportError as e:
            self.last_error = str(e)
            logger.info("reconnect failed: %s", e)
            return False

    async def reconnect_now(self) -> bool:
        """Manual retry (the `r` key). Returns True when the session is connected afterwards."""
        if self._stopped:
            return False
        await self._cancel_reconnect()
        await self._close_transport()
        self._set_state(ConnectionState.RECONNECTING)
        if not await self._try_connect():
            # A transport failure resumes the backed-off loop from the first delay.
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
        return self.state == ConnectionState.CONNECTED

    # --- requests -------------------------------------------------------------------------------

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        transport = self._transport
        if self.state != ConnectionState.CONNECTED or transport is None:
            raise TransportError(self.last_error or f"not connected ({self.state.value})")
        try:
            result = await transport.request(method, params)
        except AgentError as e:
            if e.code == ERR_VERSION_MISMATCH:
                await self._fail_permanently(VersionMismatch(f"{e.message}; {BOOTSTRAP_HINT}"))
            raise
        except AuthError:
            self.last_error = auth_instruction(self._host)
            await self._close_transport()
            self._set_state(ConnectionState.AUTH_REQUIRED)
            raise
        except (BootstrapRequired, VersionMismatch) as e:
            await self._fail_permanently(e)
        except TransportError as e:
            self._on_transport_failure(e)
            raise
        self.last_success = self._now()
        return result

    async def _fail_permanently(self, exc: TransportError) -> None:
        self.last_error = str(exc)
        await self._close_transport()
        self._set_state(ConnectionState.DISCONNECTED)
        raise exc

    async def active_jobs(self) -> ActiveJobsResult:
        return ActiveJobsResult.model_validate(await self._call("active_jobs"))

    async def history_jobs(
        self, since: datetime, until: datetime | None = None
    ) -> HistoryJobsResult:
        params: dict[str, Any] = {"since": since.isoformat()}
        if until is not None:
            params["until"] = until.isoformat()
        return HistoryJobsResult.model_validate(await self._call("history_jobs", params))

    async def job_detail(self, job_id: int) -> JobDetail:
        return JobDetail.model_validate(await self._call("job_detail", {"job_id": job_id}))

    async def job_usage(self, job_id: int) -> JobUsage:
        return JobUsage.model_validate(await self._call("job_usage", {"job_id": job_id}))

    async def cancel(self, job_id: int, array_task_id: int | None = None) -> CancelResult:
        params: dict[str, Any] = {"job_id": job_id}
        if array_task_id is not None:
            params["array_task_id"] = array_task_id
        return CancelResult.model_validate(await self._call("cancel", params))


async def _safe_close(transport: Transport) -> None:
    try:
        await transport.close()
    except Exception:
        logger.exception("closing transport failed")


def make_transport(settings: Resolved) -> Transport:
    """Pick LocalTransport when settings.cluster.host is None, else SshTransport."""
    if settings.cluster.host is None:
        from slurm_monitor.transport.local import LocalTransport

        return LocalTransport()
    from slurm_monitor.transport.ssh import SshTransport

    return SshTransport(settings)


def run_check(settings: Resolved) -> int:
    """`slurm-monitor check`: connect, print the handshake as key/value lines, exit 0; on
    BootstrapRequired / VersionMismatch / AuthError print the instruction and exit 1."""
    return asyncio.run(_check(settings))


async def _check(settings: Resolved) -> int:
    host = settings.cluster.host
    session = ClientSession(lambda: make_transport(settings), host=host)
    mode = "local (in-process)" if host is None else f"ssh {host}"
    try:
        h = await session.start()
    except TransportError as e:
        # AuthError, BootstrapRequired, VersionMismatch and plain failures all end here; none
        # of them retries during a check, and nothing is written to stdout.
        print(f"slurm-monitor: {session.last_error or e}", file=sys.stderr)
        return 1
    finally:
        await session.stop()
    print(f"cluster: {settings.cluster_name} via {mode}")
    for key, value in h.model_dump().items():
        print(f"{key}: {value}")
    print(f"degraded: {'yes' if not h.extension_available else 'no'}")
    return 0
