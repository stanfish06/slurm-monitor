"""In-process transport: the agent core runs inside the client process (on the cluster).

Results are serialized with model_dump(mode="json") so the payload the TUI sees is byte-for-byte
what the SSH path would carry; the two modes differ only in where the core executes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from slurm_monitor.core import CoreAPI, CoreError, dispatch
from slurm_monitor.models import Handshake
from slurm_monitor.transport import AgentError, Transport, TransportError

logger = logging.getLogger("slurm_monitor.transport.local")


def _default_core_factory() -> CoreAPI:
    # Imported lazily: pyslurm links libslurmfull, which exists only on Slurm hosts.
    try:
        from slurm_monitor.core.queries import PyslurmCore
    except ImportError as e:
        raise TransportError(
            f"cannot run in-process on this host: {e}; use --host to reach a cluster over ssh"
        ) from e
    return PyslurmCore()


class LocalTransport(Transport):
    def __init__(
        self,
        core: CoreAPI | None = None,
        core_factory: Callable[[], CoreAPI] | None = None,
    ) -> None:
        self._core = core
        self._core_factory = core_factory or _default_core_factory
        self.handshake = None

    async def start(self) -> Handshake:
        if self._core is None:
            self._core = self._core_factory()
        # Blocking pyslurm RPCs run in a worker thread so the event loop keeps rendering.
        self.handshake = await asyncio.to_thread(self._core.hello)
        return self.handshake

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._core is None:
            raise TransportError("transport not started")
        try:
            result = await asyncio.to_thread(dispatch, self._core, method, params or {})
        except CoreError as e:
            raise AgentError(e.code, e.message, e.data) from e
        if isinstance(result, BaseModel):
            return result.model_dump(mode="json")
        return dict(result)

    async def close(self) -> None:
        self._core = None

    @property
    def opens_network_connection(self) -> bool:
        return False
