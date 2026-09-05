"""RecordingCore: a CoreAPI wrapper that writes every response to a fixture directory.

Layout (all files hold Model.model_dump(mode="json") payloads, i.e. protocol `result` bodies):

    hello.json                 one Handshake (overwritten)
    active_jobs.json           list of ActiveJobsResult, in call order
    history_jobs.json          list of HistoryJobsResult, in call order
    job_detail/<job_id>.json   one JobDetail (overwritten)
    job_usage/<job_id>.json    one JobUsage; becomes a list once recorded more than once
    cancel/<job_id>[_<task>].json  one CancelResult
    raw/<name>.json            optional raw pyslurm dumps (record_raw), for normalization tests

The replaying transport serves list entries one per request and sticks on the last. Files are
read-modify-written on every call; fixture sizes are a few hundred kilobytes at most.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from slurm_monitor.core import CoreAPI
from slurm_monitor.core.normalize import jsonable
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobUsage,
)


class RecordingCore(CoreAPI):
    def __init__(self, inner: CoreAPI, directory: Path | str):
        self.inner = inner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # --- CoreAPI ------------------------------------------------------------------------------

    def hello(self) -> Handshake:
        result = self.inner.hello()
        self._write(self.directory / "hello.json", _dump(result))
        return result

    def active_jobs(self) -> ActiveJobsResult:
        result = self.inner.active_jobs()
        self._append(self.directory / "active_jobs.json", _dump(result))
        return result

    def history_jobs(self, since: datetime, until: datetime | None = None) -> HistoryJobsResult:
        result = self.inner.history_jobs(since, until)
        self._append(self.directory / "history_jobs.json", _dump(result))
        return result

    def job_detail(self, job_id: int) -> JobDetail:
        result = self.inner.job_detail(job_id)
        self._write(self.directory / "job_detail" / f"{job_id}.json", _dump(result))
        return result

    def job_usage(self, job_id: int) -> JobUsage:
        result = self.inner.job_usage(job_id)
        self._append_or_single(self.directory / "job_usage" / f"{job_id}.json", _dump(result))
        return result

    def cancel(self, job_id: int, array_task_id: int | None = None) -> CancelResult:
        result = self.inner.cancel(job_id, array_task_id)
        name = f"{job_id}.json" if array_task_id is None else f"{job_id}_{array_task_id}.json"
        self._write(self.directory / "cancel" / name, _dump(result))
        return result

    # --- extras -------------------------------------------------------------------------------

    def record_raw(self, name: str, payload: Any) -> None:
        """Store a raw pyslurm dump (to_dict()) under raw/<name>.json."""
        self._write(self.directory / "raw" / f"{name}.json", jsonable(payload))

    # --- file helpers -------------------------------------------------------------------------

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _append(self, path: Path, payload: Any) -> None:
        existing = self._read(path)
        items = existing if isinstance(existing, list) else ([] if existing is None else [existing])
        items.append(payload)
        self._write(path, items)

    def _append_or_single(self, path: Path, payload: Any) -> None:
        existing = self._read(path)
        if existing is None:
            self._write(path, payload)
        else:
            self._append(path, payload)

    @staticmethod
    def _read(path: Path) -> Any:
        if not path.exists():
            return None
        return json.loads(path.read_text())


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
