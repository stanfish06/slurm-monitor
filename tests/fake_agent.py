"""A stdio NDJSON agent over canned models, for transport and session tests.

    python -m tests.fake_agent [--banner] [--exit-code N] [--stderr-msg TEXT] [--mismatch]
                               [--die-after N] [--hang]

--banner      print two non-frame lines to stdout before the handshake (login-shell noise)
--exit-code N write --stderr-msg to stderr and exit N before the handshake (ssh failure modes)
--mismatch    handshake reports a running Slurm version the agent was not built against
--die-after N exit 3 without answering after reading the N-th request (mid-request death)
--hang        answer the handshake, then never answer any request (client timeout)

Requests are served on a small thread pool and job_usage is delayed, so responses to concurrent
requests arrive out of order and exercise id correlation.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from slurm_monitor import PROTOCOL_VERSION, __version__
from slurm_monitor.core import CoreAPI, CoreError, dispatch
from slurm_monitor.models import (
    ActiveJobsResult,
    CancelResult,
    Handshake,
    HistoryJobsResult,
    JobDetail,
    JobSummary,
    JobUsage,
    Measure,
    ResourceRequest,
    Source,
)
from slurm_monitor.protocol import (
    ERR_INTERNAL,
    ERR_NOT_CANCELLABLE,
    ERR_NOT_FOUND,
    ERR_VERSION_MISMATCH,
    ProtocolError,
    decode_request,
    encode,
    error_response,
    result_response,
)

NOW = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)

PENDING = JobSummary(
    job_id=1001,
    name="pending-job",
    state="PENDING",
    user="fake",
    partition="standard",
    account="acct",
    source=Source.CONTROLLER,
    submit_time=datetime(2026, 9, 5, 11, 50, tzinfo=UTC),
    num_nodes=1,
    num_cpus=4,
    state_reason="Priority",
)
RUNNING = JobSummary(
    job_id=1002,
    name="running-job",
    state="RUNNING",
    user="fake",
    partition="gpu",
    account="acct",
    source=Source.CONTROLLER,
    submit_time=datetime(2026, 9, 5, 11, 0, tzinfo=UTC),
    start_time=datetime(2026, 9, 5, 11, 30, tzinfo=UTC),
    elapsed_seconds=1800,
    num_nodes=1,
    num_cpus=8,
)
FINISHED = JobSummary(
    job_id=900,
    name="finished-job",
    state="COMPLETED",
    user="fake",
    partition="standard",
    account="acct",
    source=Source.ACCOUNTING,
    submit_time=datetime(2026, 9, 4, 9, 0, tzinfo=UTC),
    start_time=datetime(2026, 9, 4, 9, 5, tzinfo=UTC),
    end_time=datetime(2026, 9, 4, 10, 5, tzinfo=UTC),
    elapsed_seconds=3600,
    num_nodes=1,
    num_cpus=2,
    exit_code=0,
)

ACTIVE = ActiveJobsResult(jobs=[PENDING, RUNNING], fetched_at=NOW)
HISTORY_JOBS = [FINISHED]

DETAILS: dict[int, JobDetail] = {
    1002: JobDetail(
        **RUNNING.model_dump(),
        qos="normal",
        nodelist="gl1520",
        requested=ResourceRequest(
            cpus=8, nodes=1, memory_mb=32000, gpus=1, gres="gres/gpu:a100=1", time_limit_minutes=240
        ),
        working_directory="/home/fake/run",
        command="/home/fake/run/train.sh",
        stdout_path="/home/fake/run/slurm-1002.out",
        stderr_path="/home/fake/run/slurm-1002.out",
        priority=12345,
        time_limit_minutes=240,
        batch_host="gl1520",
    ),
    1001: JobDetail(
        **PENDING.model_dump(),
        qos="normal",
        requested=ResourceRequest(cpus=4, nodes=1, memory_mb=8000, time_limit_minutes=60),
        working_directory="/home/fake/pending",
        command="/home/fake/pending/job.sh",
        stdout_path="/home/fake/pending/slurm-1001.out",
        priority=100,
        time_limit_minutes=60,
    ),
    900: JobDetail(
        **FINISHED.model_dump(),
        qos="normal",
        nodelist="gl0007",
        requested=ResourceRequest(cpus=2, nodes=1, memory_mb=4000, time_limit_minutes=120),
        working_directory="/home/fake/old",
        stdout_path="/home/fake/old/slurm-900.out",
        time_limit_minutes=120,
    ),
}

USAGES: dict[int, JobUsage] = {
    1002: JobUsage(
        job_id=1002,
        state="RUNNING",
        sampled=True,
        sampled_at=NOW,
        cpu=Measure(unit="cpu-seconds", consumed=7200.0, requested=14400.0),
        memory=Measure(unit="MB", consumed=12000.0, requested=32000.0),
        walltime=Measure(unit="seconds", consumed=1800.0, requested=14400.0),
        gpus_allocated=1,
        gpu_utilization=Measure(unit="percent", consumed=85.0, requested=100.0),
        gpu_memory=Measure(unit="MB", consumed=20000.0, requested=40960.0),
        gpu_accounting_available=True,
    ),
    900: JobUsage(
        job_id=900,
        state="COMPLETED",
        sampled=True,
        sampled_at=datetime(2026, 9, 4, 10, 5, tzinfo=UTC),
        cpu=Measure(unit="cpu-seconds", consumed=6000.0, requested=7200.0),
        memory=Measure(unit="MB", consumed=1500.0, requested=4000.0),
        walltime=Measure(unit="seconds", consumed=3600.0, requested=7200.0),
    ),
}


def make_handshake(*, mismatch: bool = False, extension_available: bool = True) -> Handshake:
    return Handshake(
        protocol_version=PROTOCOL_VERSION,
        agent_version=__version__,
        user="fake",
        hostname="fake-login",
        pyslurm_version="25.11.2",
        slurm_built_version="25.11.5",
        slurm_running_version="24.11.0" if mismatch else "25.11.6",
        extension_available=extension_available,
        extension_error=None if extension_available else "ImportError: no module _filtered",
    )


class FakeCore(CoreAPI):
    def __init__(self, *, mismatch: bool = False, extension_available: bool = True) -> None:
        self._handshake = make_handshake(mismatch=mismatch, extension_available=extension_available)

    def hello(self) -> Handshake:
        return self._handshake

    def active_jobs(self) -> ActiveJobsResult:
        return ACTIVE

    def history_jobs(self, since: datetime, until: datetime | None = None) -> HistoryJobsResult:
        return HistoryJobsResult(
            jobs=HISTORY_JOBS, window_start=since, window_end=until or NOW, fetched_at=NOW
        )

    def job_detail(self, job_id: int) -> JobDetail:
        if job_id not in DETAILS:
            raise CoreError(ERR_NOT_FOUND, f"job {job_id} is unknown to the controller")
        return DETAILS[job_id]

    def job_usage(self, job_id: int) -> JobUsage:
        if job_id not in USAGES:
            raise CoreError(ERR_NOT_FOUND, f"no accounting record for job {job_id}")
        return USAGES[job_id]

    def cancel(self, job_id: int, array_task_id: int | None = None) -> CancelResult:
        if job_id == FINISHED.job_id:
            raise CoreError(ERR_NOT_CANCELLABLE, f"job {job_id} already finished")
        if job_id not in DETAILS:
            raise CoreError(ERR_NOT_FOUND, f"job {job_id} is unknown to the controller")
        return CancelResult(job_id=job_id, array_task_id=array_task_id, cancelled=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fake_agent")
    parser.add_argument("--banner", action="store_true")
    parser.add_argument("--exit-code", type=int, default=None)
    parser.add_argument("--stderr-msg", default=None)
    parser.add_argument("--mismatch", action="store_true")
    parser.add_argument("--no-extension", action="store_true")
    parser.add_argument("--die-after", type=int, default=None)
    parser.add_argument("--hang", action="store_true")
    args = parser.parse_args(argv)

    out = sys.stdout.buffer
    err = sys.stderr
    if args.exit_code is not None:
        if args.stderr_msg:
            print(args.stderr_msg, file=err, flush=True)
        return args.exit_code
    if args.banner:
        out.write(b"Welcome to fake-login. Modules restored.\n")
        out.write(b"not json at all { [\n")
        out.flush()

    core = FakeCore(mismatch=args.mismatch, extension_available=not args.no_extension)
    write_lock = threading.Lock()

    def send(frame) -> None:
        with write_lock:
            out.write(encode(frame))
            out.flush()

    send(result_response(0, core.hello()))
    print("fake_agent: handshake sent", file=err, flush=True)

    def handle(req_id: int, method: str, params: dict) -> None:
        print(f"fake_agent: {method} id={req_id}", file=err, flush=True)
        if method == "job_usage":
            time.sleep(0.05)  # reorder responses relative to the other methods
        try:
            result = dispatch(core, method, params)
            if args.mismatch:
                send(error_response(req_id, ERR_VERSION_MISMATCH, "agent build mismatch"))
                return
            send(result_response(req_id, result))
        except CoreError as e:
            send(error_response(req_id, e.code, e.message, e.data or None))
        except Exception as e:  # keep serving other requests
            send(error_response(req_id, ERR_INTERNAL, f"{type(e).__name__}: {e}"))

    seen = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for line in sys.stdin.buffer:
            try:
                req = decode_request(line)
            except ProtocolError as e:
                print(f"fake_agent: bad request ({e})", file=err, flush=True)
                continue
            seen += 1
            if args.die_after is not None and seen >= args.die_after:
                print("fake_agent: dying on purpose", file=err, flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                out.flush()
                sys.exit(3)
            if args.hang:
                continue
            pool.submit(handle, req.id, req.method, req.params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
