"""Transports carry protocol frames between the client and the agent's core.

Three implementations share one interface so the TUI never knows where the agent runs:

- transport.local.LocalTransport   : agent core runs in this process (on the cluster)
- transport.ssh.SshTransport       : agent runs on the login node over one ssh subprocess
- transport.fixture.FixtureTransport: replays recorded JSON fixtures (tests, --fixtures)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from slurm_monitor.models import Handshake


class TransportError(Exception):
    """The link failed for a transport reason (process died, pipe closed, timeout). Retryable."""


class AuthError(TransportError):
    """ssh could not authenticate. Not retryable: the user must authenticate out-of-band."""


class BootstrapRequired(TransportError):
    """The cluster has no agent environment. Not retryable: the user must run bootstrap."""


class VersionMismatch(TransportError):
    """Agent build does not match the running Slurm. Not retryable: re-run bootstrap."""


class AgentError(Exception):
    """The agent answered with an error frame."""

    def __init__(self, code: str, message: str, data: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}


class Transport(ABC):
    """One session's link to the agent. start() completes the handshake; request() is
    awaited per call; close() tears the link down. Implementations must be safe to
    call request() on concurrently from one event loop (queue or lock internally)."""

    handshake: Handshake | None = None

    @abstractmethod
    async def start(self) -> Handshake: ...

    @abstractmethod
    async def request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @property
    def opens_network_connection(self) -> bool:
        return False
