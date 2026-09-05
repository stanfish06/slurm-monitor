"""Fixture transport: replays recorded or hand-written JSON responses.

Directory layout (every file is Model.model_dump(mode="json")):

    hello.json                  Handshake
    active_jobs.json            list[ActiveJobsResult]   one snapshot per request, sticks on last
    history_jobs.json           list[HistoryJobsResult]  same
    job_detail/<job_id>.json    JobDetail
    job_usage/<job_id>.json     JobUsage, or a list served one per request
    cancel/<job_id>[_<task>].json  CancelResult

A single object is accepted wherever a list is expected. Missing detail/usage/cancel files
answer with AgentError(not_found); a missing list file answers with an empty result so the TUI
can start against a partial fixture set.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from slurm_monitor import PROTOCOL_VERSION, __version__
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobUsage,
)
from slurm_monitor.protocol import ERR_BAD_REQUEST, ERR_NOT_FOUND
from slurm_monitor.transport import AgentError, Transport, TransportError

Payload = BaseModel | Mapping[str, Any]


def default_handshake(**overrides: Any) -> Handshake:
    base: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "agent_version": __version__,
        "user": "fixture",
        "hostname": "fixture",
        "pyslurm_version": "25.11.2",
        "slurm_built_version": "25.11.5",
        "slurm_running_version": "25.11.5",
        "extension_available": True,
    }
    return Handshake(**{**base, **overrides})


def _dump(model_cls: type[BaseModel], payload: Payload) -> dict[str, Any]:
    # Validate so a bad fixture fails at load, and normalize to the wire representation.
    model = payload if isinstance(payload, model_cls) else model_cls.model_validate(payload)
    return model.model_dump(mode="json")


def _as_list(payload: Any) -> list[Any]:
    return list(payload) if isinstance(payload, list) else [payload]


class _Queue:
    """Snapshots served one per request, sticking on the last."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.index = 0

    def next(self) -> dict[str, Any] | None:
        if not self.items:
            return None
        item = self.items[min(self.index, len(self.items) - 1)]
        self.index += 1
        return item


class FixtureTransport(Transport):
    def __init__(self, directory: Path | str | None = None) -> None:
        self.handshake = None
        self._hello: dict[str, Any] = default_handshake().model_dump(mode="json")
        self._active = _Queue([])
        self._history = _Queue([])
        self._details: dict[int, dict[str, Any]] = {}
        self._usages: dict[int, _Queue] = {}
        self._cancels: dict[str, dict[str, Any]] = {}
        self._fail_next: Exception | None = None
        self._started = False
        self.requests: list[tuple[str, dict[str, Any]]] = []  # every request seen, for tests
        if directory is not None:
            self._load(Path(directory).expanduser())

    # --- construction ---------------------------------------------------------------------------

    @classmethod
    def from_results(
        cls,
        *,
        hello: Payload | None = None,
        active: Iterable[Payload] = (),
        history: Iterable[Payload] = (),
        details: Mapping[int, Payload] | None = None,
        usages: Mapping[int, Payload | Iterable[Payload]] | None = None,
        cancels: Mapping[str | int, Payload] | None = None,
    ) -> FixtureTransport:
        t = cls()
        if hello is not None:
            t._hello = _dump(Handshake, hello)
        t.set_active(active)
        t.set_history(history)
        for job_id, d in (details or {}).items():
            t._details[int(job_id)] = _dump(JobDetail, d)
        for job_id, u in (usages or {}).items():
            items = [u] if isinstance(u, BaseModel | Mapping) else list(u)
            t._usages[int(job_id)] = _Queue([_dump(JobUsage, x) for x in items])
        for key, c in (cancels or {}).items():
            t._cancels[str(key)] = _dump(CancelResult, c)
        return t

    def _load(self, directory: Path) -> None:
        if not directory.is_dir():
            raise TransportError(f"fixture directory not found: {directory}")
        hello = directory / "hello.json"
        if hello.exists():
            self._hello = _dump(Handshake, json.loads(hello.read_text()))
        active = directory / "active_jobs.json"
        if active.exists():
            self.set_active(_as_list(json.loads(active.read_text())))
        history = directory / "history_jobs.json"
        if history.exists():
            self.set_history(_as_list(json.loads(history.read_text())))
        for f in sorted((directory / "job_detail").glob("*.json")):
            self._details[int(f.stem)] = _dump(JobDetail, json.loads(f.read_text()))
        for f in sorted((directory / "job_usage").glob("*.json")):
            items = _as_list(json.loads(f.read_text()))
            self._usages[int(f.stem)] = _Queue([_dump(JobUsage, x) for x in items])
        for f in sorted((directory / "cancel").glob("*.json")):
            self._cancels[f.stem] = _dump(CancelResult, json.loads(f.read_text()))

    # --- scripting hooks for tests --------------------------------------------------------------

    def set_active(self, snapshots: Iterable[Payload]) -> None:
        """Replace the active_jobs snapshot queue; the next request serves the first item."""
        self._active = _Queue([_dump(ActiveJobsResult, s) for s in snapshots])

    def set_history(self, snapshots: Iterable[Payload]) -> None:
        self._history = _Queue([_dump(HistoryJobsResult, s) for s in snapshots])

    def fail_next(self, exc: Exception) -> None:
        """Raise exc from the next request() call instead of answering it."""
        self._fail_next = exc

    # --- Transport ------------------------------------------------------------------------------

    async def start(self) -> Handshake:
        self._started = True
        self.handshake = Handshake.model_validate(self._hello)
        return self.handshake

    async def close(self) -> None:
        self._started = False

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        self.requests.append((method, params))
        if self._fail_next is not None:
            exc, self._fail_next = self._fail_next, None
            raise exc
        if not self._started:
            raise TransportError("transport not started")
        if method == "hello":
            return dict(self._hello)
        if method == "active_jobs":
            snap = self._active.next()
            if snap is None:
                snap = ActiveJobsResult(jobs=[], fetched_at=datetime.now(UTC)).model_dump(
                    mode="json"
                )
            return snap
        if method == "history_jobs":
            snap = self._history.next()
            if snap is None:
                now = datetime.now(UTC)
                since = params.get("since")
                snap = HistoryJobsResult(
                    jobs=[],
                    window_start=datetime.fromisoformat(since) if since else now,
                    window_end=now,
                    fetched_at=now,
                ).model_dump(mode="json")
            return snap
        if method == "job_detail":
            job_id = int(params["job_id"])
            if job_id not in self._details:
                raise AgentError(ERR_NOT_FOUND, f"no fixture for job {job_id}")
            return self._details[job_id]
        if method == "job_usage":
            job_id = int(params["job_id"])
            queue = self._usages.get(job_id)
            item = queue.next() if queue is not None else None
            if item is None:
                raise AgentError(ERR_NOT_FOUND, f"no usage fixture for job {job_id}")
            return item
        if method == "cancel":
            job_id = int(params["job_id"])
            task = params.get("array_task_id")
            key = f"{job_id}_{task}" if task is not None else str(job_id)
            if key in self._cancels:
                return self._cancels[key]
            # No recorded answer: a cancel of a known job succeeds, anything else is unknown.
            if job_id in self._details or self._job_listed(job_id):
                return CancelResult(job_id=job_id, array_task_id=task, cancelled=True).model_dump(
                    mode="json"
                )
            raise AgentError(ERR_NOT_FOUND, f"no cancel fixture for {key}")
        raise AgentError(ERR_BAD_REQUEST, f"unknown method {method!r}")

    def _job_listed(self, job_id: int) -> bool:
        for snap in self._active.items:
            for job in snap["jobs"]:
                if job["job_id"] == job_id or job.get("array_job_id") == job_id:
                    return True
        return False
