"""Record the fixture scenarios against a live cluster.

Run on a login node with pyslurm importable:

    PYTHONPATH=src python tests/cluster/record_fixtures.py all --out tests/fixtures/recorded
    PYTHONPATH=src python tests/cluster/record_fixtures.py array-mixed gpu-job

Each scenario submits tiny jobs (<= 5 min, 200M, 1 CPU) with sbatch, drives PyslurmCore wrapped
in RecordingCore at the interesting moments, and stores raw pyslurm dumps under raw/ for the
normalization test. Polling sleeps 10 s between checks; whole-cluster Jobs.load() happens once
(the mixed array snapshot); everything else goes through Job.load(id) for the ids this script
submitted, which yields the same controller records without loading 46k foreign jobs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from slurm_monitor.core import CoreError
from slurm_monitor.core.queries import PyslurmCore
from slurm_monitor.core.recording import RecordingCore

ACCOUNT = "iheemske0"
PARTITION = "standard"
POLL = 10
COMMON = ["--account", ACCOUNT, "--mem=200M", "--cpus-per-task=1", "--time=5", "--nodes=1"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sbatch(*args: str, partition: str = PARTITION) -> int:
    cmd = ["sbatch", "--parsable", "--partition", partition, *COMMON, *args]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    job_id = int(out.split(";")[0])
    log(f"submitted {job_id}: {' '.join(args)}")
    return job_id


def squeue_rows(job_id: int) -> list[tuple[int, str, str]]:
    """(raw_id, array_task_str, state) per row the controller shows for job_id / its array."""
    out = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%A|%K|%T"], capture_output=True, text=True
    ).stdout
    rows = []
    for line in out.splitlines():
        raw, task, state = line.split("|")
        rows.append((int(raw), task, state))
    return rows


def wait_until(job_id: int, predicate, timeout: float, what: str) -> list[tuple[int, str, str]]:
    deadline = time.monotonic() + timeout
    while True:
        rows = squeue_rows(job_id)
        if predicate(rows):
            log(f"{job_id}: {what} ({rows})")
            return rows
        if time.monotonic() > deadline:
            raise TimeoutError(f"{job_id}: timed out waiting for {what}; last rows {rows}")
        time.sleep(POLL)


def wait_gone(job_id: int, timeout: float = 900) -> None:
    wait_until(job_id, lambda rows: not rows, timeout, "left the queue")


class KnownJobsLoader:
    """job_loader for PyslurmCore: Job.load(id) for the ids this script submitted plus every raw
    id squeue lists for them (array tasks get their own ids once they start)."""

    def __init__(self, *job_ids: int):
        self.job_ids = set(job_ids)

    def __call__(self):
        import pyslurm

        raw_ids = set(self.job_ids)
        for jid in self.job_ids:
            raw_ids.update(r[0] for r in squeue_rows(jid))
        jobs = []
        for raw in sorted(raw_ids):
            try:
                jobs.append(pyslurm.Job.load(raw))
            except pyslurm.RPCError:
                continue  # purged from the controller
        return jobs


def recorder(out: Path, scenario: str, *job_ids: int) -> tuple[RecordingCore, PyslurmCore]:
    target = out / scenario
    if target.exists():
        shutil.rmtree(target)
    core = PyslurmCore(job_loader=KnownJobsLoader(*job_ids) if job_ids else None)
    rec = RecordingCore(core, target)
    rec.hello()
    return rec, core


def dump_raw(rec: RecordingCore, tag: str, job_id: int) -> None:
    """Store controller (if still held) and accounting dumps of one job."""
    import pyslurm

    try:
        job = pyslurm.Job.load(job_id)
        rec.record_raw(f"controller_{tag}_{job_id}", job.to_dict())
    except pyslurm.RPCError as e:
        log(f"controller has no {job_id}: {e}")
    try:
        records = pyslurm.db.Jobs.load(pyslurm.db.JobFilter(ids=[job_id], with_script=True))
        for rec_job in records.values():
            if rec_job.id == job_id:
                payload = rec_job.to_dict()
                payload["steps"] = {
                    str(sid): {**step.to_dict(), "stats": step.stats.to_dict()}
                    for sid, step in rec_job.steps.items()
                }
                payload["stats"] = rec_job.stats.to_dict()
                rec.record_raw(f"accounting_{tag}_{job_id}", payload)
    except pyslurm.RPCError as e:
        log(f"accounting has no {job_id}: {e}")


def history_since(hours: float = 1) -> datetime:
    return datetime.now(UTC) - timedelta(hours=hours)


# --- scenarios ------------------------------------------------------------------------------


def array_mixed(out: Path) -> None:
    aid = sbatch("--array=1-6%2", "--job-name=arrmix", "--wrap=sleep 90")
    rec, _ = recorder(out, "array-mixed", aid)

    def mixed(rows):
        states = {r[2] for r in rows}
        return "RUNNING" in states and "PENDING" in states

    rows = wait_until(aid, mixed, 900, "mixed pending and running")
    running = sorted(r[0] for r in rows if r[2] == "RUNNING")
    task = running[0]
    # The one whole-cluster load in this script: the real degraded path for the mixed snapshot.
    full = RecordingCore(PyslurmCore(), out / "array-mixed")
    result = full.active_jobs()
    log(f"full-load snapshot: {len(result.jobs)} active jobs, degraded={result.degraded}")
    dump_raw(rec, "mixed", aid)
    dump_raw(rec, "mixed", task)
    rec.job_detail(aid)
    rec.job_detail(task)
    time.sleep(40)  # one accounting sample (task=30) before asking for live usage
    rec.job_usage(task)
    rec.active_jobs()  # same instant as the usage, via Job.load per known id
    wait_gone(aid)
    rec.active_jobs()  # tasks left the active set: completion-handoff "after" snapshot
    time.sleep(30)  # let slurmdbd store the final records
    dump_raw(rec, "done", task)
    dump_raw(rec, "done", aid)
    rec.history_jobs(history_since())
    rec.job_usage(task)
    rec.job_detail(task)


def cancelled_job(out: Path) -> None:
    jid = sbatch("--job-name=cancelme", "--wrap=sleep 300")
    rec, core = recorder(out, "cancelled-job", jid)
    try:
        wait_until(jid, lambda rows: any(r[2] == "RUNNING" for r in rows), 300, "running")
    except TimeoutError:
        log("still pending after 5 min; cancelling while pending")
    rec.active_jobs()
    rec.job_detail(jid)
    dump_raw(rec, "before", jid)
    rec.cancel(jid)
    time.sleep(POLL)
    rec.active_jobs()
    try:
        core.cancel(jid)
    except CoreError as e:
        log(f"second cancel refused as expected: {e.code}: {e.message}")
    time.sleep(30)
    dump_raw(rec, "after", jid)
    rec.history_jobs(history_since())
    rec.job_usage(jid)
    rec.job_detail(jid)


def short_job(out: Path) -> None:
    jid = sbatch("--job-name=shortjob", "--wrap=true")
    rec, _ = recorder(out, "short-job", jid)
    wait_gone(jid)
    time.sleep(30)
    dump_raw(rec, "done", jid)
    rec.job_detail(jid)
    rec.job_usage(jid)
    rec.history_jobs(history_since())


def failed_job(out: Path) -> None:
    jid = sbatch("--job-name=failjob", "--wrap=exit 3")
    rec, _ = recorder(out, "failed-job", jid)
    wait_gone(jid)
    time.sleep(30)
    dump_raw(rec, "done", jid)
    rec.job_detail(jid)
    rec.job_usage(jid)
    rec.history_jobs(history_since())


def gpu_job(out: Path) -> None:
    args = [
        "--gres=gpu:1",
        "--mem=2G",
        "--time=3",
        "--job-name=gpujob",
        "--wrap=nvidia-smi; sleep 75",
    ]
    jid = None
    for partition in ("gpu", "spgpu"):
        jid = sbatch(*args, partition=partition)
        try:
            wait_until(jid, lambda rows: any(r[2] == "RUNNING" for r in rows), 600, "running")
            break
        except TimeoutError:
            log(f"{jid} did not start on {partition} in 10 min; cancelling")
            subprocess.run(["scancel", str(jid)], check=False)
            jid = None
    if jid is None:
        raise SystemExit("gpu-job: no GPU partition started the job; fixture not recorded")
    rec, _ = recorder(out, "gpu-job", jid)
    time.sleep(40)
    rec.active_jobs()
    rec.job_detail(jid)
    rec.job_usage(jid)
    dump_raw(rec, "running", jid)
    wait_gone(jid)
    time.sleep(30)
    dump_raw(rec, "done", jid)
    rec.job_usage(jid)
    rec.history_jobs(history_since())


def array_cancel(out: Path) -> None:
    """Verify task-level and whole-array cancellation semantics of slurm_kill_job2."""
    aid = sbatch("--array=1-3", "--job-name=arrcancel", "--wrap=sleep 120")
    rec, _ = recorder(out, "array-cancel", aid)
    time.sleep(POLL)
    rows = squeue_rows(aid)
    log(f"before task cancel: {rows}")
    rec.active_jobs()
    rec.cancel(aid, 3)
    time.sleep(POLL)
    rows = squeue_rows(aid)
    log(f"after cancelling task 3: {rows}")
    assert all(r[1] != "3" for r in rows), "task 3 still queued"
    rec.cancel(aid)
    time.sleep(POLL)
    rows = squeue_rows(aid)
    log(f"after cancelling the array: {rows}")
    assert not rows, "array tasks still queued"
    rec.active_jobs()
    time.sleep(30)
    rec.history_jobs(history_since())


def array_range(out: Path) -> None:
    """An array large enough that backfill leaves tasks inside the pending range record
    (bf_max_job_array_resv splits at most ~20), so the snapshot holds a "<aid>_[range]" record
    next to running task records. Cancelled as a whole once captured."""
    aid = sbatch("--array=1-40%2", "--job-name=arrrange", "--wrap=sleep 120")
    rec, _ = recorder(out, "array-range", aid)

    def mixed(rows):
        return any(r[2] == "RUNNING" for r in rows) and any("-" in r[1] for r in rows)

    rows = wait_until(aid, mixed, 900, "running tasks plus a pending range")
    full = RecordingCore(PyslurmCore(), out / "array-range")
    result = full.active_jobs()
    log(f"full-load snapshot: {[j.display_id for j in result.jobs]}")
    dump_raw(rec, "range", aid)
    running = sorted(r[0] for r in rows if r[2] == "RUNNING")
    dump_raw(rec, "range", running[0])
    try:
        rec.job_detail(aid)
    except CoreError as e:
        log(f"job_detail({aid}) -> {e.code}: {e.message}")
    rec.job_detail(running[0])
    rec.cancel(aid)
    time.sleep(POLL)
    log(f"after cancelling the array: {squeue_rows(aid)}")
    rec.active_jobs()
    time.sleep(30)
    rec.history_jobs(history_since())


SCENARIOS = {
    "array-range": array_range,
    "array-mixed": array_mixed,
    "gpu-job": gpu_job,
    "cancelled-job": cancelled_job,
    "short-job": short_job,
    "failed-job": failed_job,
    "array-cancel": array_cancel,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scenarios", nargs="+", choices=[*SCENARIOS, "all"])
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/recorded"))
    ns = parser.parse_args(argv)
    names = list(SCENARIOS) if "all" in ns.scenarios else ns.scenarios
    failures = 0
    for name in names:
        log(f"=== {name}")
        try:
            SCENARIOS[name](ns.out)
        except Exception as e:  # keep recording the other scenarios
            failures += 1
            log(f"!!! {name} failed: {type(e).__name__}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
