"""ClientSession: handshake checks, retry policy, auth/bootstrap stops."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from slurm_monitor.models import ConnectionState, Handshake
from slurm_monitor.protocol import ERR_VERSION_MISMATCH
from slurm_monitor.session import ClientSession, make_transport
from slurm_monitor.transport import (
    AgentError,
    AuthError,
    BootstrapRequired,
    Transport,
    TransportError,
    VersionMismatch,
)
from slurm_monitor.transport.fixture import FixtureTransport
from tests.fake_agent import ACTIVE, DETAILS, USAGES, FakeCore, make_handshake

BACKOFF = (1.0, 2.0, 4.0)


class BrokenTransport(Transport):
    """start() raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.closed = False
        self.handshake = None

    async def start(self) -> Handshake:
        raise self.exc

    async def request(self, method, params=None):
        raise AssertionError("must not be called")

    async def close(self) -> None:
        self.closed = True


def good_transport(handshake: Handshake | None = None) -> FixtureTransport:
    return FixtureTransport.from_results(
        hello=handshake or make_handshake(), active=[ACTIVE], details=DETAILS, usages=USAGES
    )


class Factory:
    """Hands out transports in order; records them. The last one repeats forever."""

    def __init__(self, *transports: Transport) -> None:
        self.queue = list(transports)
        self.made: list[Transport] = []

    def __call__(self) -> Transport:
        t = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        self.made.append(t)
        return t


class FakeClock:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)
        await asyncio.sleep(0)  # yield so the test can observe intermediate states


def make_session(factory: Callable[[], Transport], clock: FakeClock, **kw) -> ClientSession:
    return ClientSession(factory, BACKOFF, sleep=clock.sleep, host="u@login.example", **kw)


async def settle(session: ClientSession, *, rounds: int = 50) -> None:
    """Let the reconnect task run to completion (it only awaits the fake sleep)."""
    for _ in range(rounds):
        await asyncio.sleep(0)
        if session.state != ConnectionState.RECONNECTING:
            return
    raise AssertionError(f"session still reconnecting: {session.last_error}")


# --- 6.2: handshake checks -----------------------------------------------------------------------


async def test_slurm_version_mismatch_refuses_to_serve():
    t = good_transport(make_handshake(mismatch=True))
    clock = FakeClock()
    states: list[ConnectionState] = []
    session = make_session(Factory(t), clock)
    session.on_state_change(states.append)
    with pytest.raises(VersionMismatch) as info:
        await session.start()
    msg = str(info.value)
    assert "25.11.5" in msg and "24.11.0" in msg and "slurm-monitor bootstrap" in msg
    assert session.state == ConnectionState.DISCONNECTED
    assert session.last_error == msg
    assert states == [ConnectionState.CONNECTING, ConnectionState.DISCONNECTED]
    assert t.requests == []  # no request was served
    with pytest.raises(TransportError):
        await session.active_jobs()
    assert t.requests == []
    assert clock.delays == []  # no retry


async def test_protocol_version_mismatch_refuses_to_serve():
    h = make_handshake().model_copy(update={"protocol_version": 99})
    session = make_session(Factory(good_transport(h)), FakeClock())
    with pytest.raises(VersionMismatch, match="protocol version 99"):
        await session.start()
    assert "slurm-monitor bootstrap" in session.last_error


async def test_version_mismatch_error_frame_disconnects_without_retry():
    t = good_transport()
    clock = FakeClock()
    session = make_session(Factory(t), clock)
    await session.start()
    t.fail_next(AgentError(ERR_VERSION_MISMATCH, "agent build mismatch"))
    with pytest.raises(VersionMismatch, match="slurm-monitor bootstrap"):
        await session.active_jobs()
    assert session.state == ConnectionState.DISCONNECTED
    assert clock.delays == []


async def test_connected_session_serves_typed_results_and_tracks_success():
    t0 = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
    ticks = iter(range(10))
    session = ClientSession(
        Factory(good_transport()),
        BACKOFF,
        sleep=FakeClock().sleep,
        now=lambda: t0.replace(second=next(ticks)),
    )
    h = await session.start()
    assert h.extension_available and session.degraded is False
    assert session.state == ConnectionState.CONNECTED
    assert session.last_success == t0.replace(second=0)
    assert await session.active_jobs() == ACTIVE
    assert session.last_success == t0.replace(second=1)
    assert (await session.job_detail(1002)) == DETAILS[1002]
    assert (await session.job_usage(1002)) == USAGES[1002]
    cancel = await session.cancel(1001)
    assert cancel.cancelled and cancel.job_id == 1001
    history = await session.history_jobs(datetime(2026, 9, 1, tzinfo=UTC))
    assert history.jobs == []
    await session.stop()
    assert session.state == ConnectionState.DISCONNECTED


# --- 11.3: retry policy -----------------------------------------------------------------------


async def test_transport_failure_reconnects_with_backoff():
    first = good_transport()
    second = good_transport(make_handshake(extension_available=False))
    factory = Factory(
        first,
        BrokenTransport(TransportError("ssh exited")),
        BrokenTransport(TransportError("ssh exited again")),
        second,
    )
    clock = FakeClock()
    states: list[ConnectionState] = []
    session = make_session(factory, clock)
    session.on_state_change(states.append)
    await session.start()
    assert session.degraded is False

    first.fail_next(TransportError("pipe closed"))
    with pytest.raises(TransportError, match="pipe closed"):
        await session.active_jobs()
    assert session.state == ConnectionState.RECONNECTING
    assert session.last_error == "pipe closed"
    # While reconnecting, requests fail immediately so the caller keeps its last data.
    with pytest.raises(TransportError):
        await session.active_jobs()

    await settle(session)
    assert session.state == ConnectionState.CONNECTED
    assert clock.delays == [1.0, 2.0, 4.0]
    assert states == [
        ConnectionState.CONNECTING,
        ConnectionState.CONNECTED,
        ConnectionState.RECONNECTING,
        ConnectionState.CONNECTED,
    ]
    # The new handshake is re-read: the second agent lacks the extension.
    assert session.degraded is True
    assert session.handshake is second.handshake
    assert await session.active_jobs() == ACTIVE
    assert len(second.requests) == 1
    await session.stop()


async def test_backoff_caps_at_last_value():
    factory = Factory(good_transport(), BrokenTransport(TransportError("down")))
    clock = FakeClock()
    session = make_session(factory, clock)
    await session.start()
    factory.made[0].fail_next(TransportError("drop"))
    with pytest.raises(TransportError):
        await session.active_jobs()
    for _ in range(40):
        await asyncio.sleep(0)
    assert session.state == ConnectionState.RECONNECTING
    assert clock.delays[:5] == [1.0, 2.0, 4.0, 4.0, 4.0]
    await session.stop()
    n = len(clock.delays)
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(clock.delays) == n  # stop() ends the loop


async def test_auth_error_stops_retrying_with_instruction():
    first = good_transport()
    factory = Factory(
        first,
        BrokenTransport(TransportError("blip")),
        BrokenTransport(AuthError("Permission denied (publickey,keyboard-interactive)")),
    )
    clock = FakeClock()
    states: list[ConnectionState] = []
    session = make_session(factory, clock)
    session.on_state_change(states.append)
    await session.start()
    first.fail_next(TransportError("pipe closed"))
    with pytest.raises(TransportError):
        await session.active_jobs()
    await settle(session)
    assert session.state == ConnectionState.AUTH_REQUIRED
    assert clock.delays == [1.0, 2.0]
    assert session.last_error == (
        "authentication required: run `ssh u@login.example` in a separate terminal, "
        "complete authentication, then press r to reconnect"
    )
    assert states[-2:] == [ConnectionState.RECONNECTING, ConnectionState.AUTH_REQUIRED]
    for _ in range(20):
        await asyncio.sleep(0)
    assert clock.delays == [1.0, 2.0]  # sleep is never called again
    with pytest.raises(TransportError, match="authentication required"):
        await session.active_jobs()
    await session.stop()


async def test_auth_error_at_start_does_not_retry():
    clock = FakeClock()
    session = make_session(Factory(BrokenTransport(AuthError("Permission denied"))), clock)
    with pytest.raises(AuthError):
        await session.start()
    assert session.state == ConnectionState.AUTH_REQUIRED
    assert "ssh u@login.example" in session.last_error
    assert clock.delays == []


async def test_manual_reconnect_after_auth_required():
    factory = Factory(BrokenTransport(AuthError("Permission denied")), good_transport())
    clock = FakeClock()
    session = make_session(factory, clock)
    with pytest.raises(AuthError):
        await session.start()
    assert await session.reconnect_now() is True
    assert session.state == ConnectionState.CONNECTED
    assert await session.active_jobs() == ACTIVE
    assert clock.delays == []
    await session.stop()


async def test_manual_reconnect_failure_resumes_backoff():
    factory = Factory(
        BrokenTransport(AuthError("Permission denied")), BrokenTransport(TransportError("down"))
    )
    clock = FakeClock()
    session = make_session(factory, clock)
    with pytest.raises(AuthError):
        await session.start()
    assert await session.reconnect_now() is False
    assert session.state == ConnectionState.RECONNECTING
    for _ in range(20):
        await asyncio.sleep(0)
    assert clock.delays[:3] == [1.0, 2.0, 4.0]
    await session.stop()


# --- 6.3 client side: bootstrap required ---------------------------------------------------------


async def test_bootstrap_required_at_start_propagates_without_retry():
    clock = FakeClock()
    exc = BootstrapRequired("agent exited (exit 127); run `slurm-monitor bootstrap`")
    session = make_session(Factory(BrokenTransport(exc)), clock)
    with pytest.raises(BootstrapRequired, match="slurm-monitor bootstrap"):
        await session.start()
    assert session.state == ConnectionState.DISCONNECTED
    assert "slurm-monitor bootstrap" in session.last_error
    assert clock.delays == []


async def test_bootstrap_required_during_reconnect_stops_retrying():
    first = good_transport()
    factory = Factory(
        first, BrokenTransport(BootstrapRequired("venv gone; slurm-monitor bootstrap"))
    )
    clock = FakeClock()
    session = make_session(factory, clock)
    await session.start()
    first.fail_next(TransportError("pipe closed"))
    with pytest.raises(TransportError):
        await session.active_jobs()
    await settle(session)
    assert session.state == ConnectionState.DISCONNECTED
    assert clock.delays == [1.0]
    assert "slurm-monitor bootstrap" in session.last_error
    await session.stop()


# --- wiring -----------------------------------------------------------------------------------


def test_make_transport_picks_by_host():
    from slurm_monitor.config import Config, Overrides, resolve
    from slurm_monitor.transport.local import LocalTransport
    from slurm_monitor.transport.ssh import SshTransport

    local = resolve(Config(), Overrides(local=True))
    assert isinstance(make_transport(local), LocalTransport)
    remote = resolve(Config(), Overrides(host="u@somewhere"))
    t = make_transport(remote)
    assert isinstance(t, SshTransport) and t.host == "u@somewhere"


async def test_session_over_local_transport_end_to_end():
    from slurm_monitor.transport.local import LocalTransport

    session = ClientSession(lambda: LocalTransport(FakeCore()))
    await session.start()
    assert await session.active_jobs() == ACTIVE
    await session.stop()
