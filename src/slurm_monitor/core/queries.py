"""PyslurmCore: CoreAPI backed by pyslurm.

Data paths:
- active_jobs   : slurm_monitor_ext.load_user_jobs() (slurm_load_job_user) when the extension
                  imports; otherwise pyslurm.Jobs.load() filtered by uid, reported degraded.
                  Terminal jobs the controller still holds (MinJobAge) are dropped afterwards.
- history_jobs  : pyslurm.db.Jobs.load(JobFilter(users=[me], start_time, end_time)), then
                  terminal states with since <= end_time < until, newest first.
- job_detail    : pyslurm.Job.load(id) while the controller holds the job, else accounting.
- job_usage     : running jobs: Job.load(id).load_stats() (sstat path); terminal jobs: the
                  accounting record's step statistics. GPU TRES usage (gres/gpuutil, gres/gpumem)
                  is read through pyslurm's legacy slurmdb_jobs().get(), the only pyslurm API that
                  exposes the raw tres_usage_in_* strings.
- cancel        : slurm_kill_job2(job_id_str, SIGKILL, 0) through ctypes on the libslurmfull that
                  pyslurm already loaded, because pyslurm has no string-id kill and pending array
                  tasks are only addressable as "<array>_<task>".

Every pyslurm touch point is a constructor-injectable callable so the fixture tier can run the
same code with plain objects on a host without pyslurm.
"""

from __future__ import annotations

import ctypes
import getpass
import os
import re
import signal
import socket
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from slurm_monitor import PROTOCOL_VERSION, __version__
from slurm_monitor.core import CoreAPI, CoreError
from slurm_monitor.core.normalize import (
    _get,
    _int,
    accounting_detail,
    accounting_summary,
    build_usage,
    controller_detail,
    controller_summary,
    gpu_count,
    gpu_usage_from_tres,
    steps_sampled,
)
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobUsage,
    is_active_state,
    is_terminal_state,
)
from slurm_monitor.protocol import ERR_NOT_CANCELLABLE, ERR_NOT_FOUND, ERR_RPC

# Slurm errno values (slurm_errno.h) that cancel and load map to protocol codes.
ESLURM_INVALID_JOB_ID = 2017
ESLURM_ALREADY_DONE = 2021

JobLoader = Callable[[], Iterable[Any]]
HistoryLoader = Callable[[str, datetime, datetime], Iterable[Any]]
ControllerLoader = Callable[[int], Any]
AccountingLoader = Callable[[int, bool], Iterable[Any]]
TresUsageLoader = Callable[[int], list[dict[str, str | None]] | None]
Canceller = Callable[[str], None]


def _rpc_message(exc: BaseException) -> str:
    msg = getattr(exc, "msg", None)
    return str(msg) if msg else str(exc)


def _is_invalid_job_id(exc: BaseException) -> bool:
    if getattr(exc, "errno", None) == ESLURM_INVALID_JOB_ID:
        return True
    text = _rpc_message(exc).lower()
    return "invalid job id" in text or "does not exist" in text


class PyslurmCore(CoreAPI):
    def __init__(
        self,
        *,
        job_loader: JobLoader | None = None,
        history_loader: HistoryLoader | None = None,
        controller_loader: ControllerLoader | None = None,
        accounting_loader: AccountingLoader | None = None,
        tres_usage_loader: TresUsageLoader | None = None,
        gpu_tres_ids: Callable[[], tuple[int | None, int | None]] | None = None,
        canceller: Canceller | None = None,
        username: str | None = None,
        uid: int | None = None,
        clock: Callable[[], datetime] | None = None,
        sampling_interval_seconds: int | None = None,
    ):
        self._job_loader = job_loader
        self._history_loader = history_loader
        self._controller_loader = controller_loader
        self._accounting_loader = accounting_loader
        self._tres_usage_loader = tres_usage_loader
        self._gpu_tres_ids_fn = gpu_tres_ids
        self._canceller = canceller
        self._username = username or getpass.getuser()
        self._uid = uid if uid is not None else os.getuid()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sampling_interval = sampling_interval_seconds
        self._gpu_tres_ids: tuple[int | None, int | None] | None = None
        self._libslurm: ctypes.CDLL | None = None
        self.last_degraded_error: str | None = None

    # --- handshake --------------------------------------------------------------------------

    def hello(self) -> Handshake:
        try:
            from slurm_monitor.core.environment import gather_handshake, probe_extension
        except ImportError:
            return self._fallback_handshake()
        ok, error = probe_extension()
        return gather_handshake(ok, error)

    def _fallback_handshake(self) -> Handshake:
        """Minimal handshake used only when core.environment is not present."""
        import pyslurm
        from pyslurm.core import slurmctld

        ext_ok, ext_err = True, None
        try:
            import slurm_monitor_ext  # noqa: F401
        except Exception as e:  # ImportError or a broken build
            ext_ok, ext_err = False, f"{type(e).__name__}: {e}"
        built = ".".join(str(p) for p in pyslurm.slurm_api_version())
        cfg = slurmctld.Config.load()
        # pyslurm renders JobAcctGatherFrequency as {"task": 30}; older builds return "task=30".
        freq = _get(cfg, "job_accounting_gather_frequency")
        sampling = 30
        if isinstance(freq, dict) and _int(freq.get("task")) is not None:
            sampling = _int(freq["task"])
        else:
            m = re.search(r"task=(\d+)", str(freq)) if freq else None
            if m:
                sampling = int(m.group(1))
        return Handshake(
            protocol_version=PROTOCOL_VERSION,
            agent_version=__version__,
            user=self._username,
            hostname=socket.gethostname(),
            pyslurm_version=pyslurm.__version__,
            slurm_built_version=built,
            slurm_running_version=str(cfg.version),
            extension_available=ext_ok,
            extension_error=ext_err,
            sampling_interval_seconds=sampling,
        )

    # --- active --------------------------------------------------------------------------------

    def active_jobs(self) -> ActiveJobsResult:
        jobs, degraded = self._load_user_jobs()
        summaries = [controller_summary(j) for j in jobs]
        # Narrow after the RPC: slurm_load_job_user() also returns terminal jobs within MinJobAge.
        active = [s for s in summaries if is_active_state(s.state)]
        active.sort(key=lambda s: (s.array_job_id or s.job_id, s.array_task_id or -1, s.job_id))
        return ActiveJobsResult(jobs=active, degraded=degraded, fetched_at=self._clock())

    def _load_user_jobs(self) -> tuple[list[Any], bool]:
        if self._job_loader is not None:
            return list(self._job_loader()), False
        try:
            import slurm_monitor_ext

            loaded = slurm_monitor_ext.load_user_jobs()
            self.last_degraded_error = None
            return list(_values(loaded)), False
        except Exception as e:  # missing module, build mismatch, or RPC failure inside it
            self.last_degraded_error = f"{type(e).__name__}: {e}"
        import pyslurm

        try:
            everything = pyslurm.Jobs.load()
        except pyslurm.RPCError as e:
            raise CoreError(ERR_RPC, _rpc_message(e)) from e
        mine = [j for j in everything.values() if _get(j, "user_id") == self._uid]
        return mine, True

    # --- history -------------------------------------------------------------------------------

    def history_jobs(self, since: datetime, until: datetime | None = None) -> HistoryJobsResult:
        """Terminal jobs with since <= end_time < until (half-open, so adjacent windows do not
        repeat a job). Naive datetimes are taken as UTC."""
        since = _aware(since)
        until = _aware(until) if until is not None else self._clock()
        records = list(self._load_history(since, until))
        jobs = []
        for rec in records:
            summary = accounting_summary(rec)
            if not is_terminal_state(summary.state) or summary.end_time is None:
                continue
            if since <= summary.end_time < until:
                jobs.append(summary)
        jobs.sort(key=lambda s: (s.end_time or since, s.job_id), reverse=True)
        return HistoryJobsResult(
            jobs=jobs, window_start=since, window_end=until, fetched_at=self._clock()
        )

    def _load_history(self, since: datetime, until: datetime) -> Iterable[Any]:
        if self._history_loader is not None:
            return self._history_loader(self._username, since, until)
        import pyslurm

        # slurmdbd matches jobs that overlapped [start_time, end_time] in any state; the
        # end_time/terminal filter above narrows that to jobs that finished inside the window.
        db_filter = pyslurm.db.JobFilter(
            users=[self._username],
            start_time=int(since.timestamp()),
            end_time=int(until.timestamp()),
        )
        try:
            return list(pyslurm.db.Jobs.load(db_filter).values())
        except pyslurm.RPCError as e:
            raise CoreError(ERR_RPC, _rpc_message(e)) from e

    # --- detail --------------------------------------------------------------------------------

    def job_detail(self, job_id: int) -> JobDetail:
        job = self._controller_job(job_id)
        if job is not None:
            return controller_detail(job)
        record = self._accounting_job(job_id, with_script=True)
        if record is None:
            raise CoreError(ERR_NOT_FOUND, f"job {job_id} is unknown to slurmctld and slurmdbd")
        return accounting_detail(record)

    def _controller_job(self, job_id: int) -> Any | None:
        """The controller's record for job_id, or None once the controller has purged it."""
        if self._controller_loader is not None:
            try:
                return self._controller_loader(job_id)
            except CoreError as e:
                if e.code == ERR_NOT_FOUND:
                    return None
                raise
        import pyslurm

        try:
            return pyslurm.Job.load(job_id)
        except pyslurm.RPCError as e:
            if _is_invalid_job_id(e):
                return None
            raise CoreError(ERR_RPC, _rpc_message(e)) from e
        except KeyError as e:
            # slurm_load_job(array_id) answers with every task of the array; pyslurm keeps the
            # first record but then looks its steps up under the requested id and fails.
            raise CoreError(
                ERR_RPC, f"job {job_id} is an array id; the controller holds several records"
            ) from e

    def _accounting_job(self, job_id: int, with_script: bool = False) -> Any | None:
        """The accounting record whose raw id is job_id. slurmdbd answers an id filter with every
        task of an array when the id is the array's base id, so pick the exact record."""
        records = list(self._load_accounting(job_id, with_script))
        for rec in records:
            if _int(_get(rec, "id")) == job_id:
                return rec
        return None

    def _load_accounting(self, job_id: int, with_script: bool) -> Iterable[Any]:
        if self._accounting_loader is not None:
            return self._accounting_loader(job_id, with_script)
        import pyslurm

        db_filter = pyslurm.db.JobFilter(ids=[job_id], with_script=with_script)
        try:
            return list(pyslurm.db.Jobs.load(db_filter).values())
        except pyslurm.RPCError as e:
            raise CoreError(ERR_RPC, _rpc_message(e)) from e

    # --- usage ---------------------------------------------------------------------------------

    def job_usage(self, job_id: int) -> JobUsage:
        job = self._controller_job(job_id)
        if job is not None:
            state = str(_get(job, "state") or "UNKNOWN")
            if is_active_state(state):
                return self._live_usage(job, state)
        record = self._accounting_job(job_id)
        if record is None:
            if job is not None:  # finished within MinJobAge but slurmdbd has not stored it yet
                return self._live_usage(job, str(_get(job, "state") or "UNKNOWN"), stats_ok=False)
            raise CoreError(ERR_NOT_FOUND, f"job {job_id} is unknown to slurmctld and slurmdbd")
        return self._accounting_usage(record)

    def _live_usage(self, job: Any, state: str, stats_ok: bool = True) -> JobUsage:
        """Running job: sstat-style statistics via Job.load_stats(). Pending jobs have no steps
        and report sampled=False. GPU TRES usage is not reachable through the live path."""
        step_stats: list[Any] = []
        stats = None
        if stats_ok and state.split()[0].upper() == "RUNNING":
            load_stats = getattr(job, "load_stats", None)
            if callable(load_stats):
                try:
                    stats = load_stats()
                except Exception:  # slurmstepd unreachable or step already gone
                    stats = None
            steps = _get(job, "steps") or {}
            step_stats = [_get(s, "stats") for s in _values(steps)]
        gpus = gpu_count(job)
        util_id, mem_id = self._gpu_tres() if gpus else (None, None)
        elapsed = _int(_get(job, "run_time")) or 0
        sampled = steps_sampled(
            step_stats, elapsed_seconds=elapsed, interval=self.sampling_interval()
        )
        return build_usage(
            job_id=int(_get(job, "id")),
            state=state,
            cpus=_int(_get(job, "cpus")),
            elapsed_seconds=elapsed,
            time_limit_minutes=_int(_get(job, "time_limit")),
            memory_mb=_int(_get(job, "memory")),
            gpus_allocated=gpus,
            sampled=sampled,
            total_cpu_seconds=_int(_get(stats, "total_cpu_time")) if stats else None,
            peak_rss_bytes=_int(_get(stats, "resident_memory")) if stats else None,
            gpu_accounting_available=(util_id is not None) if gpus else None,
            sampled_at=self._clock(),
        )

    def _accounting_usage(self, record: Any) -> JobUsage:
        job_id = int(_get(record, "id"))
        state = str(_get(record, "state") or "UNKNOWN")
        steps = _get(record, "steps") or {}
        step_stats = [_get(s, "stats") for s in _values(steps)]
        stats = _get(record, "stats")
        gpus = gpu_count(record)
        util = mem = None
        util_id, mem_id = self._gpu_tres() if gpus else (None, None)
        if gpus and util_id is not None:
            tres_steps = self._tres_usage(job_id)
            if tres_steps:
                util, mem = gpu_usage_from_tres(tres_steps, util_id, mem_id)
        elapsed = _int(_get(record, "elapsed_time")) or 0
        return build_usage(
            job_id=job_id,
            state=state,
            cpus=_int(_get(record, "cpus")),
            elapsed_seconds=elapsed,
            time_limit_minutes=_int(_get(record, "time_limit")),
            memory_mb=_int(_get(record, "memory")),
            gpus_allocated=gpus,
            sampled=steps_sampled(
                step_stats, elapsed_seconds=elapsed, interval=self.sampling_interval()
            ),
            total_cpu_seconds=_int(_get(stats, "total_cpu_time")) if stats else None,
            peak_rss_bytes=_int(_get(stats, "resident_memory")) if stats else None,
            gpu_utilization=util,
            gpu_memory_mb=mem,
            gpu_accounting_available=(util_id is not None) if gpus else None,
            sampled_at=self._clock(),
        )

    def sampling_interval(self) -> int:
        """JobAcctGatherFrequency task= seconds, from the handshake; 30 if it cannot be read."""
        if self._sampling_interval is None:
            try:
                self._sampling_interval = int(self.hello().sampling_interval_seconds)
            except Exception:
                self._sampling_interval = 30
        return self._sampling_interval

    def _gpu_tres(self) -> tuple[int | None, int | None]:
        """TRES ids of gres/gpuutil and gres/gpumem in slurmdbd; (None, None) when the cluster
        does not account GPU usage. Loaded once per core."""
        if self._gpu_tres_ids is None:
            if self._gpu_tres_ids_fn is not None:
                self._gpu_tres_ids = self._gpu_tres_ids_fn()
            else:
                self._gpu_tres_ids = self._load_gpu_tres_ids()
        return self._gpu_tres_ids

    @staticmethod
    def _load_gpu_tres_ids() -> tuple[int | None, int | None]:
        import pyslurm

        try:
            tres = pyslurm.db.TrackableResources.load()
        except pyslurm.RPCError:
            return None, None
        util = mem = None
        for tres_id, rec in tres._id_map.items():
            if _get(rec, "type") == "gres" and _get(rec, "name") == "gpuutil":
                util = int(tres_id)
            elif _get(rec, "type") == "gres" and _get(rec, "name") == "gpumem":
                mem = int(tres_id)
        return util, mem

    def _tres_usage(self, job_id: int) -> list[dict[str, str | None]] | None:
        if self._tres_usage_loader is not None:
            return self._tres_usage_loader(job_id)
        import pyslurm

        try:
            records = pyslurm.slurmdb_jobs().get(jobids=[job_id])
        except Exception:  # legacy API raises ValueError on RPC failure
            return None
        rec = records.get(job_id) if isinstance(records, dict) else None
        if not rec:
            return None
        out: list[dict[str, str | None]] = []
        for step in (rec.get("steps") or {}).values():
            stats = step.get("stats") or {}
            out.append(
                {
                    "tres_usage_in_ave": stats.get("tres_usage_in_ave"),
                    "tres_usage_in_max": stats.get("tres_usage_in_max"),
                    "tres_usage_in_tot": stats.get("tres_usage_in_tot"),
                }
            )
        return out

    # --- cancel --------------------------------------------------------------------------------

    def cancel(self, job_id: int, array_task_id: int | None = None) -> CancelResult:
        """Single job / whole array: the numeric id (Slurm kills every task when given the array's
        base id). One task: "<array>_<task>", which also addresses tasks still inside the pending
        range record. State is checked against the controller first for plain jobs; for array
        targets the kill RPC's errno is authoritative because Job.load(array_id) returns only
        one of the tasks."""
        target = str(job_id) if array_task_id is None else f"{job_id}_{array_task_id}"
        if array_task_id is None:
            try:
                job = self._controller_job(job_id)
            except CoreError as e:
                if e.code != ERR_RPC:
                    raise
                job = False  # array id with several live records: let the kill RPC decide
            if job is None:
                if self._accounting_job(job_id) is not None:
                    raise CoreError(ERR_NOT_CANCELLABLE, f"job {job_id} has already finished")
                raise CoreError(ERR_NOT_FOUND, f"job {job_id} is unknown to slurmctld")
            state = str(_get(job, "state") or "") if job else ""
            if job and _get(job, "array_id") is None and is_terminal_state(state):
                raise CoreError(ERR_NOT_CANCELLABLE, f"job {job_id} is already {state}")
        self._kill(target)
        return CancelResult(job_id=job_id, array_task_id=array_task_id, cancelled=True)

    def _kill(self, target: str) -> None:
        if self._canceller is not None:
            self._canceller(target)
            return
        lib = self._load_libslurm()
        if lib is None:
            self._kill_with_pyslurm(target)
            return
        ctypes.set_errno(0)
        rc = lib.slurm_kill_job2(target.encode(), int(signal.SIGKILL), 0, None)
        if rc == 0:
            return
        errno = ctypes.get_errno()
        message = lib.slurm_strerror(errno).decode(errors="replace") if errno else "unknown error"
        if errno == ESLURM_ALREADY_DONE or "already complet" in message.lower():
            raise CoreError(ERR_NOT_CANCELLABLE, f"job {target} has already finished: {message}")
        if errno == ESLURM_INVALID_JOB_ID or "invalid job id" in message.lower():
            raise CoreError(ERR_NOT_FOUND, f"job {target} is unknown to slurmctld: {message}")
        raise CoreError(ERR_RPC, f"cancel {target} refused: {message}", {"errno": errno})

    def _kill_with_pyslurm(self, target: str) -> None:
        """Fallback without libslurmfull: numeric ids only (pyslurm.Job.cancel swallows
        ESLURM_ALREADY_DONE, so the terminal check in cancel() is the only guard here)."""
        import pyslurm

        if "_" in target:
            raise CoreError(
                ERR_RPC,
                f"cannot address array task {target}: libslurmfull not found for slurm_kill_job2",
            )
        try:
            pyslurm.Job(int(target)).cancel()
        except pyslurm.RPCError as e:
            if _is_invalid_job_id(e):
                raise CoreError(ERR_NOT_FOUND, _rpc_message(e)) from e
            raise CoreError(ERR_RPC, _rpc_message(e)) from e

    def _load_libslurm(self) -> ctypes.CDLL | None:
        """The libslurmfull.so pyslurm mapped into this process, opened again via ctypes so
        slurm_kill_job2 (declared in pyslurm's headers but not wrapped) is callable."""
        if self._libslurm is not None:
            return self._libslurm
        import pyslurm  # noqa: F401  # ensures the library is mapped

        path = None
        try:
            with open("/proc/self/maps") as maps:
                for line in maps:
                    if "libslurmfull" in line:
                        path = line.split()[-1]
                        break
        except OSError:
            path = None
        if path is None:
            return None
        try:
            lib = ctypes.CDLL(path, use_errno=True)
            lib.slurm_kill_job2.argtypes = [
                ctypes.c_char_p,
                ctypes.c_uint16,
                ctypes.c_uint16,
                ctypes.c_char_p,
            ]
            lib.slurm_kill_job2.restype = ctypes.c_int
            lib.slurm_strerror.argtypes = [ctypes.c_int]
            lib.slurm_strerror.restype = ctypes.c_char_p
        except (OSError, AttributeError):
            return None
        self._libslurm = lib
        return lib


def _values(collection: Any) -> Iterable[Any]:
    """Iterate a pyslurm collection (MultiClusterMap / dict) or a plain iterable of records."""
    values = getattr(collection, "values", None)
    if callable(values):
        return values()
    return collection


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
