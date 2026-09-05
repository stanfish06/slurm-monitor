"""One suite run against every Transport, plus subprocess-specific failure modes."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from slurm_monitor import PROTOCOL_VERSION
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    HistoryJobsResult,
    JobDetail,
    JobUsage,
)
from slurm_monitor.protocol import ERR_NOT_CANCELLABLE, ERR_NOT_FOUND
from slurm_monitor.transport import (
    AgentError,
    AuthError,
    BootstrapRequired,
    Transport,
    TransportError,
    VersionMismatch,
)
from slurm_monitor.transport.fixture import FixtureTransport
from slurm_monitor.transport.local import LocalTransport
from slurm_monitor.transport.subprocess import SubprocessTransport, classify_exit
from tests import fake_agent
from tests.fake_agent import ACTIVE, DETAILS, FINISHED, USAGES, FakeCore

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {"PYTHONPATH": str(REPO_ROOT)}
SINCE = datetime(2026, 9, 1, tzinfo=UTC)


def fake_agent_transport(*args: str, **kwargs) -> SubprocessTransport:
    argv = [sys.executable, "-m", "tests.fake_agent", *args]
    return SubprocessTransport(argv, env=SUBPROCESS_ENV, **kwargs)


def fixture_from_fake_core() -> FixtureTransport:
    core = FakeCore()
    return FixtureTransport.from_results(
        hello=core.hello(),
        active=[ACTIVE],
        history=[core.history_jobs(SINCE)],
        details=DETAILS,
        usages=USAGES,
        cancels={"1001": CancelResult(job_id=1001, cancelled=True)},
    )


@pytest.fixture(params=["local", "subprocess", "fixture"])
async def transport(request) -> AsyncIterator[Transport]:
    if request.param == "local":
        t: Transport = LocalTransport(FakeCore())
    elif request.param == "subprocess":
        t = fake_agent_transport()
    else:
        t = fixture_from_fake_core()
    await t.start()
    try:
        yield t
    finally:
        await t.close()


# --- shared suite -----------------------------------------------------------------------------


async def test_handshake_fields(transport: Transport):
    h = transport.handshake
    assert h is not None
    assert h.protocol_version == PROTOCOL_VERSION
    assert h.slurm_built_version == "25.11.5"
    assert h.slurm_running_version == "25.11.6"
    assert h.extension_available is True
    assert h.user == "fake"


async def test_active_jobs_roundtrip_equals_core_model(transport: Transport):
    raw = await transport.request("active_jobs")
    assert raw == ACTIVE.model_dump(mode="json")
    assert ActiveJobsResult.model_validate(raw) == ACTIVE


async def test_history_jobs_carries_window(transport: Transport):
    raw = await transport.request("history_jobs", {"since": SINCE.isoformat()})
    result = HistoryJobsResult.model_validate(raw)
    assert result.window_start == SINCE
    assert [j.job_id for j in result.jobs] == [FINISHED.job_id]


async def test_error_frame_becomes_agent_error_with_code(transport: Transport):
    with pytest.raises(AgentError) as info:
        await transport.request("job_detail", {"job_id": 999999})
    assert info.value.code == ERR_NOT_FOUND
    assert "999999" in info.value.message
    # The transport keeps working after an error frame.
    assert JobDetail.model_validate(await transport.request("job_detail", {"job_id": 1002}))


async def test_concurrent_requests_correlate_by_id(transport: Transport):
    results = await asyncio.gather(
        transport.request("job_usage", {"job_id": 1002}),
        transport.request("active_jobs"),
        transport.request("job_detail", {"job_id": 1002}),
        transport.request("job_detail", {"job_id": 1001}),
        transport.request("cancel", {"job_id": 1001}),
    )
    assert JobUsage.model_validate(results[0]) == USAGES[1002]
    assert ActiveJobsResult.model_validate(results[1]) == ACTIVE
    assert JobDetail.model_validate(results[2]) == DETAILS[1002]
    assert JobDetail.model_validate(results[3]) == DETAILS[1001]
    cancel = CancelResult.model_validate(results[4])
    assert cancel.job_id == 1001 and cancel.cancelled is True


async def test_concurrent_mix_of_errors_and_results(transport: Transport):
    results = await asyncio.gather(
        transport.request("job_usage", {"job_id": 424242}),
        transport.request("job_usage", {"job_id": 1002}),
        transport.request("active_jobs"),
        return_exceptions=True,
    )
    assert isinstance(results[0], AgentError) and results[0].code == ERR_NOT_FOUND
    assert JobUsage.model_validate(results[1]) == USAGES[1002]
    assert ActiveJobsResult.model_validate(results[2]) == ACTIVE


# --- local / fixture specifics -------------------------------------------------------------------


async def test_local_transport_maps_core_error_and_opens_no_connection():
    t = LocalTransport(FakeCore())
    assert t.opens_network_connection is False
    await t.start()
    with pytest.raises(AgentError) as info:
        await t.request("cancel", {"job_id": FINISHED.job_id})
    assert info.value.code == ERR_NOT_CANCELLABLE
    await t.close()


async def test_local_transport_lazy_core_factory():
    calls = []

    def factory():
        calls.append(1)
        return FakeCore()

    t = LocalTransport(core_factory=factory)
    assert calls == []
    await t.start()
    assert calls == [1]
    await t.close()


async def test_fixture_directory_replays_snapshots_and_sticks_on_last():
    t = FixtureTransport(REPO_ROOT / "tests" / "fixtures" / "synthetic" / "basic")
    h = await t.start()
    assert h.protocol_version == PROTOCOL_VERSION
    first = ActiveJobsResult.model_validate(await t.request("active_jobs"))
    second = ActiveJobsResult.model_validate(await t.request("active_jobs"))
    third = ActiveJobsResult.model_validate(await t.request("active_jobs"))
    assert {j.state for j in first.jobs} == {"PENDING", "RUNNING"}
    assert all(j.state == "RUNNING" for j in second.jobs)
    assert third == second  # sticks on the last snapshot
    history = HistoryJobsResult.model_validate(
        await t.request("history_jobs", {"since": SINCE.isoformat()})
    )
    assert len(history.jobs) == 1 and history.jobs[0].is_terminal
    detail = JobDetail.model_validate(await t.request("job_detail", {"job_id": 2002}))
    assert detail.stdout_path
    usage = JobUsage.model_validate(await t.request("job_usage", {"job_id": 2002}))
    assert usage.sampled is True
    with pytest.raises(AgentError) as info:
        await t.request("job_detail", {"job_id": 2001})
    assert info.value.code == ERR_NOT_FOUND
    with pytest.raises(AgentError) as info:
        await t.request("job_usage", {"job_id": 5555})
    assert info.value.code == ERR_NOT_FOUND
    await t.close()


async def test_fixture_fail_next_and_set_active():
    t = fixture_from_fake_core()
    await t.start()
    t.fail_next(TransportError("simulated drop"))
    with pytest.raises(TransportError, match="simulated drop"):
        await t.request("active_jobs")
    # Only the one request fails; the next is served.
    assert ActiveJobsResult.model_validate(await t.request("active_jobs")) == ACTIVE
    t.set_active([ActiveJobsResult(jobs=[], fetched_at=datetime.now(UTC))])
    assert ActiveJobsResult.model_validate(await t.request("active_jobs")).jobs == []
    t.fail_next(AuthError("Permission denied"))
    with pytest.raises(AuthError):
        await t.request("active_jobs")
    assert [m for m, _ in t.requests].count("active_jobs") == 4


# --- subprocess specifics ------------------------------------------------------------------------


async def test_subprocess_tolerates_banner_lines(caplog):
    caplog.set_level(logging.INFO, logger="slurm_monitor.transport.subprocess")
    t = fake_agent_transport("--banner")
    h = await t.start()
    assert h.user == "fake"
    assert ActiveJobsResult.model_validate(await t.request("active_jobs")) == ACTIVE
    assert any("skipping non-frame line" in r.message for r in caplog.records)
    await t.close()


async def test_subprocess_stderr_goes_to_log_not_results(caplog):
    caplog.set_level(logging.INFO, logger="slurm_monitor.transport.subprocess")
    t = fake_agent_transport()
    await t.start()
    raw = await t.request("active_jobs")
    assert raw == ACTIVE.model_dump(mode="json")
    await t.close()
    stderr_lines = [r.message for r in caplog.records if "agent stderr" in r.message]
    assert any("fake_agent: active_jobs" in m for m in stderr_lines)
    assert not any("fake_agent" in str(v) for v in raw.values())


async def test_subprocess_death_mid_request_is_transport_error():
    t = fake_agent_transport("--die-after", "2")
    await t.start()
    assert ActiveJobsResult.model_validate(await t.request("active_jobs")) == ACTIVE
    with pytest.raises(TransportError) as info:
        await t.request("active_jobs")
    assert type(info.value) is TransportError  # not Auth/Bootstrap: exit 3, no auth message
    assert "exit 3" in str(info.value)
    # Later requests fail fast with the same classification.
    with pytest.raises(TransportError):
        await t.request("active_jobs")
    await t.close()


async def test_subprocess_request_timeout():
    t = fake_agent_transport("--hang", request_timeout=0.2)
    await t.start()
    with pytest.raises(TransportError, match="timed out"):
        await t.request("active_jobs")
    with pytest.raises(TransportError, match="timed out"):
        await t.request("active_jobs", timeout=0.05)
    await t.close()


async def test_subprocess_auth_failure_maps_to_auth_error():
    t = fake_agent_transport(
        "--exit-code", "255", "--stderr-msg", "Permission denied (publickey,keyboard-interactive)"
    )
    with pytest.raises(AuthError, match="Permission denied"):
        await t.start()
    assert t.connection_count == 1


async def test_subprocess_duo_prompt_maps_to_auth_error():
    t = fake_agent_transport("--exit-code", "1", "--stderr-msg", "Duo two-factor login for u")
    with pytest.raises(AuthError):
        await t.start()


async def test_subprocess_missing_agent_maps_to_bootstrap_required():
    t = fake_agent_transport(
        "--exit-code",
        "127",
        "--stderr-msg",
        "bash: /home/u/.slurm-monitor/venv/bin/python: No such file or directory",
    )
    with pytest.raises(BootstrapRequired, match="slurm-monitor bootstrap"):
        await t.start()


async def test_subprocess_version_mismatch_error_frame_surfaces_as_agent_error():
    t = fake_agent_transport("--mismatch")
    h = await t.start()  # the transport does not judge versions; the session does
    assert h.slurm_running_version == "24.11.0"
    with pytest.raises(AgentError) as info:
        await t.request("active_jobs")
    assert info.value.code == "version_mismatch"
    await t.close()


async def test_subprocess_twenty_requests_use_one_process():
    t = fake_agent_transport()
    await t.start()
    for _ in range(20):
        assert ActiveJobsResult.model_validate(await t.request("active_jobs")) == ACTIVE
    assert t.connection_count == 1
    await t.close()
    assert t.connection_count == 1


async def test_subprocess_spawn_failure_is_transport_error():
    t = SubprocessTransport(["/definitely/not/a/binary"])
    with pytest.raises(TransportError, match="cannot spawn"):
        await t.start()


def test_classify_exit_rules():
    assert isinstance(classify_exit(255, "Permission denied (publickey)."), AuthError)
    assert isinstance(classify_exit(255, ""), AuthError)  # pre-handshake ssh 255: auth prompt
    dropped = classify_exit(255, "Connection to host closed by remote host.", connected=True)
    assert type(dropped) is TransportError
    unresolved = classify_exit(255, "ssh: Could not resolve hostname x: nodename nor servname")
    assert type(unresolved) is TransportError
    assert isinstance(classify_exit(127, ""), BootstrapRequired)
    missing = classify_exit(
        1,
        "bash: /home/u/.slurm-monitor/venv/bin/python: No such file or directory",
        agent_path="~/.slurm-monitor/venv/bin/python",
    )
    assert isinstance(missing, BootstrapRequired) and "slurm-monitor bootstrap" in str(missing)
    other_missing = classify_exit(
        1,
        "cat: /etc/nothing: No such file or directory",
        agent_path="~/.slurm-monitor/venv/bin/python",
    )
    assert type(other_missing) is TransportError
    assert type(classify_exit(3, "boom")) is TransportError
    assert not isinstance(classify_exit(3, "boom"), VersionMismatch)


@pytest.mark.skipif(os.name != "posix", reason="ssh argv is posix-only")
def test_ssh_argv_keeps_tilde_and_batch_mode():
    from slurm_monitor.config import Config, Overrides, resolve
    from slurm_monitor.transport.ssh import SshTransport

    cfg = Config.model_validate(
        {
            "clusters": {
                "c": {"host": "u@login.example", "ssh_options": ["-o", "ServerAliveInterval=30"]}
            }
        }
    )
    settings = resolve(cfg, Overrides(cluster="c"))
    t = SshTransport(settings)
    assert t.argv == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=30",
        "u@login.example",
        "~/.slurm-monitor/venv/bin/python -m slurm_monitor.agent",
    ]
    assert t.opens_network_connection is True
    assert t.connection_count == 0
    assert t.agent_path == "~/.slurm-monitor/venv/bin/python"
    assert fake_agent.__doc__  # module imported for the subprocess tests above
