"""Transport over a child process speaking NDJSON on its stdin/stdout.

One reader task resolves pending futures by frame id, so any number of coroutines may await
request() concurrently on the same process. The child's stderr is drained into a logger and a
short tail buffer used only to classify why the process died; it never reaches the frame stream.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from typing import Any

from slurm_monitor.models import Handshake
from slurm_monitor.protocol import (
    ERR_VERSION_MISMATCH,
    ProtocolError,
    Request,
    decode_response,
    encode,
)
from slurm_monitor.transport import (
    AgentError,
    AuthError,
    BootstrapRequired,
    Transport,
    TransportError,
    VersionMismatch,
)

logger = logging.getLogger("slurm_monitor.transport.subprocess")

# A frame can carry a few hundred KB of jobs; asyncio's default 64 KiB line limit is too small.
LINE_LIMIT = 64 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_START_TIMEOUT = 120.0  # ssh + a 6.7 s login shell + python startup

# ssh/PAM/Duo output that means the user must authenticate out-of-band.
AUTH_PATTERNS = (
    "Permission denied",
    "Authentication failed",
    "Host key verification failed",
    "Too many authentication failures",
    "verification code",
    "Duo",
    "passcode",
)
# ssh exits 255 for these too, but they are network failures worth retrying.
NETWORK_PATTERNS = (
    "Could not resolve hostname",
    "Connection refused",
    "Connection timed out",
    "Connection reset",
    "No route to host",
    "Network is unreachable",
    "closed by remote host",
    "Broken pipe",
    "Timeout, server",
)
MISSING_FILE = "No such file or directory"
BOOTSTRAP_HINT = "run `slurm-monitor bootstrap` to install the agent on the cluster"


def classify_exit(
    returncode: int | None,
    stderr: str,
    *,
    agent_path: str | None = None,
    connected: bool = False,
    context: str = "agent exited",
) -> TransportError:
    """Map a dead process to the exception the session should see.

    connected=True means the handshake had completed, so an ssh exit 255 without an
    authentication message is a dropped link (retryable), not an auth failure."""
    last = stderr.strip().splitlines()[-1] if stderr.strip() else ""
    detail = f"{context} (exit {returncode})" + (f": {last}" if last else "")

    if any(p in stderr for p in AUTH_PATTERNS):
        return AuthError(detail)

    # The remote shell reports a missing interpreter with the expanded path, so match without
    # the leading "~/" that the configured agent_dir may carry.
    needle = agent_path.removeprefix("~/").removeprefix("~") if agent_path else None
    if MISSING_FILE in stderr and (needle is None or needle in stderr):
        return BootstrapRequired(f"{detail}; {BOOTSTRAP_HINT}")
    if returncode == 127:
        return BootstrapRequired(f"{detail}; {BOOTSTRAP_HINT}")

    if returncode == 255 and not connected and not any(p in stderr for p in NETWORK_PATTERNS):
        return AuthError(detail)
    return TransportError(detail)


class SubprocessTransport(Transport):
    def __init__(
        self,
        argv: list[str],
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        start_timeout: float = DEFAULT_START_TIMEOUT,
        agent_path: str | None = None,
        env: dict[str, str] | None = None,
        stderr_tail_lines: int = 50,
    ) -> None:
        self.argv = list(argv)
        self.request_timeout = request_timeout
        self.start_timeout = start_timeout
        self.agent_path = agent_path
        self.env = env
        self.handshake = None
        self.connection_count = 0  # processes spawned over this transport's lifetime
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=stderr_tail_lines)
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_id = 1
        self._exit_error: TransportError | None = None

    @property
    def opens_network_connection(self) -> bool:
        return False

    # --- lifecycle ------------------------------------------------------------------------------

    async def start(self) -> Handshake:
        if self._proc is not None:
            raise TransportError("transport already started")
        self._exit_error = None
        self._stderr_tail.clear()
        env = None if self.env is None else {**os.environ, **self.env}
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=LINE_LIMIT,
            )
        except OSError as e:
            raise TransportError(f"cannot spawn {self.argv[0]!r}: {e}") from e
        self.connection_count += 1
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            self.handshake = await asyncio.wait_for(self._read_handshake(), self.start_timeout)
        except TimeoutError:
            await self.close()
            raise TransportError(
                f"no handshake from agent within {self.start_timeout:g} s"
            ) from None
        except BaseException:
            await self.close()
            raise
        self._reader_task = asyncio.create_task(self._read_loop())
        return self.handshake

    async def _read_handshake(self) -> Handshake:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                raise await self._exit_error_after_eof("agent exited before handshake")
            try:
                resp = decode_response(line)
            except ProtocolError as e:
                # Login shells may print banners before the agent starts; skip them.
                logger.info("skipping non-frame line before handshake (%s): %r", e, line[:200])
                continue
            if resp.id != 0:
                logger.warning("frame id %d before handshake; ignoring", resp.id)
                continue
            if resp.error is not None:
                if resp.error.code == ERR_VERSION_MISMATCH:
                    raise VersionMismatch(resp.error.message)
                raise TransportError(f"agent refused handshake: {resp.error.message}")
            return Handshake.model_validate(resp.result or {})

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    resp = decode_response(line)
                except ProtocolError as e:
                    logger.warning("dropping non-frame line from agent (%s): %r", e, line[:200])
                    continue
                fut = self._pending.pop(resp.id, None)
                if fut is None or fut.done():
                    logger.warning("response id %d has no waiting request", resp.id)
                    continue
                if resp.error is not None:
                    fut.set_exception(
                        AgentError(resp.error.code, resp.error.message, resp.error.data)
                    )
                else:
                    fut.set_result(resp.result or {})
        except asyncio.CancelledError:
            self._fail_pending(TransportError("transport closed"))
            raise
        except Exception as e:  # reader must never die silently with futures outstanding
            logger.exception("reader task failed")
            self._fail_pending(TransportError(f"reader failed: {e}"))
            return
        err = await self._exit_error_after_eof("agent exited", connected=True)
        self._exit_error = err
        self._fail_pending(err)

    async def _drain_stderr(self) -> None:
        # Hold the stream locally: close() clears self._proc while this task may still be reading.
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        stderr = proc.stderr
        while True:
            line = await stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            self._stderr_tail.append(text)
            logger.info("agent stderr: %s", text)

    async def _exit_error_after_eof(
        self, context: str, *, connected: bool = False
    ) -> TransportError:
        """After stdout EOF: wait for exit and the stderr tail, then classify."""
        proc = self._proc
        if proc is None:
            return TransportError(context)
        try:
            await asyncio.wait_for(proc.wait(), 5.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
        if self._stderr_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._stderr_task), 1.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
        return classify_exit(
            proc.returncode,
            "\n".join(self._stderr_tail),
            agent_path=self.agent_path,
            connected=connected,
            context=context,
        )

    def _fail_pending(self, err: BaseException) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(err)
        self._pending.clear()

    async def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        self._reader_task = None
        if proc.returncode is None:
            # EOF on stdin lets the agent exit on its own; kill if it does not.
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), 2.0)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        if self._stderr_task is not None and not self._stderr_task.done():
            try:
                await asyncio.wait_for(self._stderr_task, 1.0)
            except (TimeoutError, Exception):
                self._stderr_task.cancel()
        self._stderr_task = None
        self._fail_pending(TransportError("transport closed"))

    # --- requests -------------------------------------------------------------------------------

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self._exit_error is not None:
            raise type(self._exit_error)(str(self._exit_error))
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise TransportError("transport not started")
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        try:
            proc.stdin.write(encode(Request(id=req_id, method=method, params=params or {})))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            self._pending.pop(req_id, None)
            raise TransportError(f"write to agent failed: {e}") from e
        limit = self.request_timeout if timeout is None else timeout
        try:
            return await asyncio.wait_for(fut, limit)
        except TimeoutError:
            self._pending.pop(req_id, None)
            raise TransportError(f"{method} timed out after {limit:g} s") from None
