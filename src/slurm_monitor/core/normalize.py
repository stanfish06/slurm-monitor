"""Map pyslurm records onto the shared models.

Two record shapes arrive here: controller jobs (pyslurm.Job, from slurmctld) and accounting jobs
(pyslurm.db.Job, from slurmdbd). They name the same concept differently (run_time vs elapsed_time,
allocated_nodes vs nodelist, user_name on both, array_id on both) and each has fields the other
lacks. Every accessor below is duck-typed through getattr so the fixture-tier tests can feed plain
objects on hosts where pyslurm cannot be imported.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from slurm_monitor.models import (
    JobDetail,
    JobSummary,
    JobUsage,
    Measure,
    ResourceRequest,
    Source,
    is_terminal_state,
)

# Slurm's default output file name when the job asked for none (see sbatch(1) "--output").
DEFAULT_STDOUT = "slurm-%j.out"
DEFAULT_ARRAY_STDOUT = "slurm-%A_%a.out"

# Built-in TRES ids that the controller/accounting usage strings always use.
TRES_CPU = 1
TRES_MEM = 2

_STDIO_PATTERN = re.compile(r"%(\d*)([%jJAauxNnts])")
_SBATCH_OUTPUT = re.compile(r"^\s*#SBATCH\s+(?:--output[= ]|-o\s*)(\S+)", re.MULTILINE)
_SBATCH_ERROR = re.compile(r"^\s*#SBATCH\s+(?:--error[= ]|-e\s*)(\S+)", re.MULTILINE)
_CLI_OUTPUT = re.compile(r"(?:^|\s)(?:--output[= ]|-o\s*)(\S+)")
_CLI_ERROR = re.compile(r"(?:^|\s)(?:--error[= ]|-e\s*)(\S+)")


# --- duck-typed accessors -------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute on a record object, or key on a dict (raw to_dict() fixtures)."""
    if isinstance(obj, dict):
        value = obj.get(name, default)
        return default if value is None else value
    try:
        value = getattr(obj, name, default)
    except Exception:  # pyslurm properties can raise on partially populated records
        return default
    return default if value is None else value


def _str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    # "None assigned" is slurmdbd's placeholder nodelist for jobs that never started.
    return text if text and text not in ("(null)", "None assigned") else None


def _int(value: Any) -> int | None:
    """int for numeric values; None for None, bools, and sentinel strings like "UNLIMITED"."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def to_datetime(value: Any) -> datetime | None:
    """Epoch seconds (pyslurm's _raw_time) -> aware UTC datetime. 0/None -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float) and value > 0:
        return datetime.fromtimestamp(value, UTC)
    return None


# --- GPU / TRES helpers --------------------------------------------------------------------


def gpu_count(job: Any) -> int:
    """Number of GPUs allocated (or requested while pending), from the job's gres/TRES."""
    gpus = _get(job, "gpus")
    if isinstance(gpus, dict) and gpus:
        total = 0
        for entry in gpus.values():
            count = _int(_get(entry, "count", entry))
            total += count if count is not None else 0
        return total
    for attr in ("allocated_tres", "requested_tres", "tres", "gres"):
        total = _gpus_from_tres(_get(job, attr))
        if total:
            return total
    return 0


def _gpus_from_tres(tres: Any) -> int:
    if tres is None:
        return 0
    if isinstance(tres, str):
        return sum(int(m) for m in re.findall(r"gres/gpu(?::[^=,]+)?=(\d+)", tres))
    gres = _get(tres, "gres", tres if isinstance(tres, dict) else None)
    if isinstance(gres, dict):
        total = 0
        for name, entry in gres.items():
            if str(name) == "gpu" or str(name).startswith("gpu:"):
                count = _int(_get(entry, "count", entry))
                total += count if count is not None else 0
        return total
    raw = _str(_get(tres, "raw_str"))
    return _gpus_from_tres(raw) if raw else 0


def gres_string(job: Any) -> str | None:
    """Raw-ish GRES description: "gres/gpu=2,gres/gpu:a100=2"."""
    gpus = _get(job, "gpus")
    if isinstance(gpus, dict) and gpus:
        parts = []
        for name, entry in gpus.items():
            count = _int(_get(entry, "count", entry))
            parts.append(f"gres/{name}={count if count is not None else 1}")
        return ",".join(parts)
    for attr in ("allocated_tres", "requested_tres", "tres"):
        tres = _get(job, attr)
        if isinstance(tres, str) and "gres/" in tres:
            return ",".join(p for p in tres.split(",") if p.startswith("gres/")) or None
    return None


def parse_tres_usage(text: str | None) -> dict[int, int]:
    """Parse a Slurm TRES usage string "1=10405266,2=19803729920,1035=42" to {id: value}."""
    out: dict[int, int] = {}
    if not text:
        return out
    for item in text.split(","):
        key, sep, value = item.partition("=")
        if sep and key.strip().isdigit() and value.strip().lstrip("-").isdigit():
            out[int(key)] = int(value)
    return out


# --- stdout/stderr paths -------------------------------------------------------------------


def _first_node(nodelist: str | None) -> str | None:
    """ "gl[1001-1002,1010]" -> "gl1001"; "gl1507" -> "gl1507"."""
    if not nodelist:
        return None
    m = re.match(r"^([^\[,]*)\[(\d+)", nodelist)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return nodelist.split(",")[0]


def expand_stdio(
    pattern: str | None,
    *,
    job_id: int,
    array_job_id: int | None,
    array_task_id: int | None,
    user: str | None,
    name: str | None,
    first_node: str | None,
    working_directory: str | None,
) -> str | None:
    """Expand sbatch filename patterns (%j %A %a %u %x %N %n %t %s %%, with optional
    zero-padding width) and make relative paths absolute against the working directory."""
    if not pattern:
        return None

    def repl(m: re.Match[str]) -> str:
        width, code = m.group(1), m.group(2)
        value: str | None
        if code == "%":
            return "%"
        if code == "j":
            value = str(job_id)
        elif code == "J":
            value = str(job_id)
        elif code == "A":
            value = str(array_job_id if array_job_id is not None else job_id)
        elif code == "a":
            value = str(array_task_id) if array_task_id is not None else "4294967294"
        elif code == "u":
            value = user
        elif code == "x":
            value = name
        elif code == "N":
            value = first_node
        elif code in ("n", "t"):
            value = "0"
        else:  # %s: step id, unknown at job level
            value = "batch"
        if value is None:
            return m.group(0)
        return value.zfill(int(width)) if width and value.isdigit() else value

    path = _STDIO_PATTERN.sub(repl, pattern)
    if not path.startswith("/") and working_directory:
        path = f"{working_directory.rstrip('/')}/{path}"
    return path


def _sbatch_paths(script: str | None, submit_line: str | None) -> tuple[str | None, str | None]:
    """--output/--error from the batch script's #SBATCH lines, else from the sbatch command line."""
    out = err = None
    if script:
        m = _SBATCH_OUTPUT.search(script)
        out = m.group(1) if m else None
        m = _SBATCH_ERROR.search(script)
        err = m.group(1) if m else None
    if submit_line:
        if out is None:
            m = _CLI_OUTPUT.search(submit_line)
            out = m.group(1) if m else None
        if err is None:
            m = _CLI_ERROR.search(submit_line)
            err = m.group(1) if m else None
    return out, err


def resolve_stdio_paths(
    summary: JobSummary,
    *,
    stdout: str | None,
    stderr: str | None,
    script: str | None,
    submit_line: str | None,
    working_directory: str | None,
    first_node: str | None,
) -> tuple[str | None, str | None]:
    """Resolved stdout/stderr paths. Precedence: what Slurm reports; then --output/--error from
    the script or submit line; then Slurm's default slurm-%j.out (slurm-%A_%a.out for array
    tasks). The default is a guess based on Slurm's naming rule, not a recorded value."""
    if not stdout or not stderr:
        s_out, s_err = _sbatch_paths(script, submit_line)
        stdout = stdout or s_out
        stderr = stderr or s_err
    if not stdout:
        stdout = DEFAULT_ARRAY_STDOUT if summary.array_task_id is not None else DEFAULT_STDOUT
    if not stderr:
        stderr = stdout  # sbatch sends stderr to the stdout file unless --error is given
    kwargs = dict(
        job_id=summary.job_id,
        array_job_id=summary.array_job_id,
        array_task_id=summary.array_task_id,
        user=summary.user,
        name=summary.name,
        first_node=first_node,
        working_directory=working_directory,
    )
    return expand_stdio(stdout, **kwargs), expand_stdio(stderr, **kwargs)


# --- summaries -----------------------------------------------------------------------------


def task_range_from_bitmask(text: str) -> str:
    """slurmdbd stores a pending array's task set as a hex bitmask ("0x1FFFFFFE000"); render it
    the way the controller does ("13-40")."""
    bits = int(text, 16)
    tasks = [i for i in range(bits.bit_length()) if bits >> i & 1]
    ranges: list[str] = []
    start = prev = None
    for t in tasks:
        if start is None:
            start = prev = t
        elif t == prev + 1:
            prev = t
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = t
    if start is not None:
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def _array_fields(job: Any) -> tuple[int | None, int | None, str | None]:
    """(array_job_id, array_task_id, array_task_range). A pending array arrives as one record
    with the task range in array_tasks_waiting; running/finished tasks carry array_task_id."""
    array_job_id = _int(_get(job, "array_id"))
    if array_job_id is None:
        return None, None, None
    task_range = _str(_get(job, "array_tasks_waiting"))
    if task_range and task_range.lower().startswith("0x"):
        task_range = task_range_from_bitmask(task_range)
    task_id = _int(_get(job, "array_task_id"))
    if task_range:
        return array_job_id, None, task_range
    # pyslurm's u32_parse treats 0 as "no value", so task 0 arrives as None on both sources;
    # an array record without a pending range can only be a task, hence task 0.
    return array_job_id, 0 if task_id is None else task_id, None


def _exit_fields(job: Any, state: str) -> tuple[int | None, int | None]:
    if not is_terminal_state(state):
        return None, None
    return _int(_get(job, "exit_code")), _int(_get(job, "exit_code_signal"))


def controller_summary(job: Any) -> JobSummary:
    """pyslurm.Job (slurmctld) -> JobSummary."""
    state = _str(_get(job, "state")) or "UNKNOWN"
    pending = state.split()[0].upper() == "PENDING"
    terminal = is_terminal_state(state)
    array_job_id, array_task_id, task_range = _array_fields(job)
    reason = _str(_get(job, "state_reason"))
    exit_code, exit_signal = _exit_fields(job, state)
    return JobSummary(
        job_id=int(_get(job, "id")),
        name=_str(_get(job, "name")) or "",
        state=state,
        user=_str(_get(job, "user_name")),
        partition=_str(_get(job, "partition")),
        account=_str(_get(job, "account")),
        source=Source.CONTROLLER,
        submit_time=to_datetime(_get(job, "submit_time")),
        # Pending jobs carry a scheduler estimate in start_time; only report real starts.
        start_time=None if pending else to_datetime(_get(job, "start_time")),
        # For running jobs the controller's end_time is the time-limit deadline, not an end.
        end_time=to_datetime(_get(job, "end_time")) if terminal else None,
        elapsed_seconds=_int(_get(job, "run_time")) or 0,
        num_nodes=_int(_get(job, "num_nodes")),
        num_cpus=_int(_get(job, "cpus")),
        array_job_id=array_job_id,
        array_task_id=array_task_id,
        array_task_range=task_range,
        state_reason=reason if pending and reason not in (None, "None") else None,
        exit_code=exit_code,
        exit_signal=exit_signal,
        failed_node=_str(_get(job, "failed_node")),
    )


def accounting_summary(job: Any) -> JobSummary:
    """pyslurm.db.Job (slurmdbd) -> JobSummary."""
    state = _str(_get(job, "state")) or "UNKNOWN"
    array_job_id, array_task_id, task_range = _array_fields(job)
    exit_code, exit_signal = _exit_fields(job, state)
    start = to_datetime(_get(job, "start_time"))
    return JobSummary(
        job_id=int(_get(job, "id")),
        name=_str(_get(job, "name")) or "",
        state=state,
        user=_str(_get(job, "user_name")),
        partition=_str(_get(job, "partition")),
        account=_str(_get(job, "account")),
        source=Source.ACCOUNTING,
        submit_time=to_datetime(_get(job, "submit_time")),
        start_time=start,
        end_time=to_datetime(_get(job, "end_time")) if is_terminal_state(state) else None,
        elapsed_seconds=_int(_get(job, "elapsed_time")) or 0,
        num_nodes=_int(_get(job, "num_nodes")),
        num_cpus=_int(_get(job, "cpus")),
        array_job_id=array_job_id,
        array_task_id=array_task_id,
        array_task_range=task_range,
        exit_code=exit_code,
        exit_signal=exit_signal,
        cancelled_by=_str(_get(job, "cancelled_by")) if state.startswith("CANCELLED") else None,
        failed_node=_str(_get(job, "failed_node")),
    )


# --- details -------------------------------------------------------------------------------


def _dependencies_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        parts = []
        for kind, ids in value.items():
            if isinstance(ids, list | tuple | set):
                if not ids:
                    continue
                parts.append(f"{kind}:{':'.join(str(i) for i in ids)}")
            elif ids:
                parts.append(f"{kind}:{ids}")
        return ",".join(parts) or None
    return str(value)


def _requested(job: Any, summary: JobSummary, time_limit: int | None) -> ResourceRequest:
    return ResourceRequest(
        cpus=summary.num_cpus,
        nodes=summary.num_nodes,
        memory_mb=_int(_get(job, "memory")),
        gpus=gpu_count(job) or None,
        gres=gres_string(job),
        time_limit_minutes=time_limit,
    )


def controller_detail(job: Any) -> JobDetail:
    """pyslurm.Job -> JobDetail. standard_output/standard_error come from slurm_get_job_stdout(),
    which expands most patterns already; expand_stdio covers what it left (%N while pending)."""
    summary = controller_summary(job)
    time_limit = _int(_get(job, "time_limit"))
    nodelist = _str(_get(job, "allocated_nodes"))
    workdir = _str(_get(job, "working_directory"))
    stdout, stderr = resolve_stdio_paths(
        summary,
        stdout=_str(_get(job, "standard_output")),
        stderr=_str(_get(job, "standard_error")),
        script=None,
        submit_line=_str(_get(job, "submit_line")),
        working_directory=workdir,
        first_node=_first_node(nodelist) or _str(_get(job, "batch_host")),
    )
    return JobDetail(
        **summary.model_dump(),
        qos=_str(_get(job, "qos")),
        nodelist=nodelist,
        requested=_requested(job, summary, time_limit),
        working_directory=workdir,
        command=_str(_get(job, "command")),
        stdout_path=stdout,
        stderr_path=stderr,
        priority=_int(_get(job, "priority")),
        time_limit_minutes=time_limit,
        dependencies=_dependencies_str(_get(job, "dependencies")),
        batch_host=_str(_get(job, "batch_host")),
    )


def accounting_detail(job: Any) -> JobDetail:
    """pyslurm.db.Job -> JobDetail. Accounting stores std_out/std_err only when the submission
    named them; otherwise resolve_stdio_paths falls back to #SBATCH lines and Slurm's default."""
    summary = accounting_summary(job)
    time_limit = _int(_get(job, "time_limit"))
    nodelist = _str(_get(job, "nodelist"))
    workdir = _str(_get(job, "working_directory"))
    stdout, stderr = resolve_stdio_paths(
        summary,
        stdout=_str(_get(job, "standard_output")),
        stderr=_str(_get(job, "standard_error")),
        script=_str(_get(job, "script")),
        submit_line=_str(_get(job, "submit_command")),
        working_directory=workdir,
        first_node=_first_node(nodelist),
    )
    return JobDetail(
        **summary.model_dump(),
        qos=_str(_get(job, "qos")),
        nodelist=nodelist,
        requested=_requested(job, summary, time_limit),
        working_directory=workdir,
        command=_str(_get(job, "submit_command")),
        stdout_path=stdout,
        stderr_path=stderr,
        priority=_int(_get(job, "priority")),
        time_limit_minutes=time_limit,
    )


# --- usage ---------------------------------------------------------------------------------

_SAMPLE_FIELDS = (
    "total_cpu_time",
    "avg_cpu_time",
    "max_resident_memory",
    "avg_resident_memory",
    "max_virtual_memory",
    "avg_disk_read",
    "avg_disk_write",
    "avg_page_faults",
)


def steps_sampled(
    step_stats: Iterable[Any], *, elapsed_seconds: int | None = None, interval: int | None = None
) -> bool:
    """True when the job has run at least one JobAcctGatherFrequency interval and some step
    carries a nonzero CPU/memory/IO figure. jobacct_gather takes one poll at step teardown, so a
    sub-interval job still shows a few MB of RSS; that reading is not a sample of the job's
    behaviour and is reported as absent. An energy-only TRES string never counts."""
    if elapsed_seconds is not None and interval and elapsed_seconds < interval:
        return False
    for stats in step_stats:
        if stats is None:
            continue
        for field in _SAMPLE_FIELDS:
            if (_int(_get(stats, field)) or 0) > 0:
                return True
    return False


def gpu_usage_from_tres(
    step_tres: Iterable[dict[str, str | None]], util_id: int | None, mem_id: int | None
) -> tuple[float | None, float | None]:
    """(utilization percent, memory MB) from per-step TRES usage strings. Utilization takes the
    per-step average (tres_usage_in_ave) when the caller has it, else the per-task peak
    (tres_usage_in_max, identical for single-task steps), else the total; memory takes the peak.
    The largest step wins. None when no step recorded that TRES id."""
    util: float | None = None
    mem: float | None = None
    for stats in step_tres:
        ave = parse_tres_usage(stats.get("tres_usage_in_ave"))
        peak = parse_tres_usage(stats.get("tres_usage_in_max"))
        tot = parse_tres_usage(stats.get("tres_usage_in_tot"))
        if util_id is not None:
            for table in (ave, peak, tot):
                if util_id in table:
                    util = max(util or 0.0, float(table[util_id]))
                    break
        if mem_id is not None:
            for table in (peak, ave, tot):
                if mem_id in table:
                    mem = max(mem or 0.0, table[mem_id] / (1024 * 1024))
                    break
    return util, mem


def build_usage(
    *,
    job_id: int,
    state: str,
    cpus: int | None,
    elapsed_seconds: int,
    time_limit_minutes: int | None,
    memory_mb: int | None,
    gpus_allocated: int,
    sampled: bool,
    total_cpu_seconds: float | None,
    peak_rss_bytes: float | None,
    gpu_utilization: float | None = None,
    gpu_memory_mb: float | None = None,
    gpu_accounting_available: bool | None = None,
    sampled_at: datetime | None = None,
) -> JobUsage:
    """Assemble a JobUsage. When sampled is False the CPU and memory consumed values are None
    regardless of what the stats object holds (they would be zeros standing in for no data);
    wall time is a clock reading, not an accounting sample, so it is reported either way."""
    cpu_req = float(cpus * elapsed_seconds) if cpus and elapsed_seconds else None
    wall_req = float(time_limit_minutes * 60) if time_limit_minutes else None
    usage = JobUsage(
        job_id=job_id,
        state=state,
        sampled=sampled,
        sampled_at=sampled_at,
        cpu=Measure(
            unit="cpu-seconds",
            consumed=float(total_cpu_seconds)
            if sampled and total_cpu_seconds is not None
            else None,
            requested=cpu_req,
        ),
        memory=Measure(
            unit="MB",
            consumed=peak_rss_bytes / (1024 * 1024)
            if sampled and peak_rss_bytes is not None
            else None,
            requested=float(memory_mb) if memory_mb else None,
        ),
        walltime=Measure(unit="seconds", consumed=float(elapsed_seconds), requested=wall_req),
        gpus_allocated=gpus_allocated,
    )
    if gpus_allocated > 0:
        usage.gpu_accounting_available = gpu_accounting_available
        usage.gpu_utilization = Measure(
            unit="percent",
            consumed=gpu_utilization if gpu_accounting_available else None,
            requested=100.0,
        )
        usage.gpu_memory = Measure(
            unit="MB",
            consumed=gpu_memory_mb if gpu_accounting_available else None,
            requested=None,
        )
    return usage


# --- raw dumps for fixtures ----------------------------------------------------------------


def jsonable(value: Any, depth: int = 0) -> Any:
    """Recursively convert a pyslurm to_dict() result to JSON-compatible data."""
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v, depth + 1) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return jsonable(to_dict(), depth + 1)
        except Exception:
            return str(value)
    return str(value)
