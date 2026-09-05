"""Array collapse: fold Slurm's two array representations into one parent row per array.

The controller reports a pending array as ONE record carrying `array_task_range` ("3-6") and
each running task as its own record carrying `array_task_id`. Both coexist in the Active list
while an array is starting, so the parent row Slurm never sends is synthesized here: its per-state
counts add every id in each range record under that record's state, plus one per task record.

Pure functions on JobSummary lists; no Textual or transport imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from slurm_monitor.models import ArrayTaskCounts, JobSummary


@dataclass(frozen=True)
class CancelTarget:
    """What a cancel request addresses. Resolved by id, never by row position."""

    job_id: int  # plain job id, or the array job id for array/task targets
    array_task_id: int | None = None  # set only for a single task
    task_count: int = 1  # tasks affected: 1 for jobs and tasks, histogram total for arrays
    label: str = ""  # display id shown in the confirmation


@dataclass
class PlainRow:
    job: JobSummary

    @property
    def key(self) -> str:
        return str(self.job.job_id)

    @property
    def cancel_target(self) -> CancelTarget:
        return CancelTarget(job_id=self.job.job_id, label=self.job.display_id)


@dataclass
class TaskRow:
    """One array task, either a real record or one synthesized from a pending range."""

    job: JobSummary  # array_job_id and array_task_id always set
    synthetic: bool = False  # True when expanded from a range record

    @property
    def array_job_id(self) -> int:
        assert self.job.array_job_id is not None
        return self.job.array_job_id

    @property
    def task_id(self) -> int:
        assert self.job.array_task_id is not None
        return self.job.array_task_id

    @property
    def key(self) -> str:
        return f"{self.array_job_id}_{self.task_id}"

    @property
    def cancel_target(self) -> CancelTarget:
        return CancelTarget(job_id=self.array_job_id, array_task_id=self.task_id, label=self.key)


@dataclass
class ArrayRow:
    """Synthesized parent row for one array."""

    array_job_id: int
    name: str
    partition: str | None
    counts: ArrayTaskCounts
    tasks: list[TaskRow] = field(default_factory=list)  # sorted by task id
    expanded: bool = False

    @property
    def key(self) -> str:
        return str(self.array_job_id)

    @property
    def display_id(self) -> str:
        return f"{self.array_job_id}_[*]"

    @property
    def summary(self) -> str:
        """State histogram, e.g. `6 tasks: 2 RUNNING, 4 PENDING` (running states first)."""
        total = self.counts.total
        parts = ", ".join(f"{n} {state}" for state, n in _ordered_counts(self.counts))
        noun = "task" if total == 1 else "tasks"
        return f"{total} {noun}: {parts}"

    @property
    def running_count(self) -> int:
        return sum(n for state, n in self.counts.counts.items() if state.startswith("RUNNING"))

    @property
    def elapsed_seconds(self) -> int:
        """Longest-running task, so the parent shows how long the array has been going."""
        return max((t.job.elapsed_seconds for t in self.tasks), default=0)

    @property
    def num_nodes(self) -> int | None:
        nodes = [t.job.num_nodes for t in self.tasks if t.job.num_nodes and t.job.is_active]
        return sum(nodes) if nodes else None

    @property
    def cancel_target(self) -> CancelTarget:
        return CancelTarget(
            job_id=self.array_job_id, task_count=self.counts.total, label=str(self.array_job_id)
        )


Row = PlainRow | ArrayRow | TaskRow

# Display order for histogram states: what is happening now before what is waiting.
_STATE_ORDER = ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "PENDING", "REQUEUED")


def _ordered_counts(counts: ArrayTaskCounts) -> list[tuple[str, int]]:
    def rank(item: tuple[str, int]) -> tuple[int, str]:
        state = item[0].split()[0]
        return (_STATE_ORDER.index(state) if state in _STATE_ORDER else len(_STATE_ORDER), state)

    return sorted(counts.counts.items(), key=rank)


def expand_task_range(spec: str) -> list[int]:
    """Expand a Slurm task range spec into sorted task ids.

    Handles comma lists ("1,3,5"), ranges ("1-10"), steps ("1-10:2") and the throttle suffix
    ("1-6%2" = at most 2 running at once), which does not change membership.
    """
    body = spec.split("%", 1)[0].strip()
    if not body:
        return []
    ids: set[int] = set()
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, step_str = part.split(":", 1)
            step = int(step_str)
            if step < 1:
                raise ValueError(f"invalid step in task range {spec!r}")
        if "-" in part:
            lo_str, hi_str = part.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            if hi < lo:
                raise ValueError(f"descending task range {part!r} in {spec!r}")
            ids.update(range(lo, hi + 1, step))
        else:
            ids.add(int(part))
    return sorted(ids)


def collapse_arrays(jobs: list[JobSummary], expanded: set[int] | frozenset[int] = frozenset()):
    """Fold array members into one ArrayRow each, keeping input order for first appearance.

    Returns a flat list of rows; an ArrayRow whose array_job_id is in `expanded` is followed by
    its TaskRows.
    """
    rows: list[Row] = []
    parents: dict[int, ArrayRow] = {}

    for job in jobs:
        if not job.is_array_member:
            rows.append(PlainRow(job=job))
            continue
        aid = job.array_job_id
        assert aid is not None
        parent = parents.get(aid)
        if parent is None:
            parent = ArrayRow(
                array_job_id=aid,
                name=job.name,
                partition=job.partition,
                counts=ArrayTaskCounts(),
                expanded=aid in expanded,
            )
            parents[aid] = parent
            rows.append(parent)
        _add_member(parent, job)

    out: list[Row] = []
    for row in rows:
        if isinstance(row, ArrayRow):
            row.tasks.sort(key=lambda t: t.task_id)
            out.append(row)
            if row.expanded:
                out.extend(row.tasks)
        else:
            out.append(row)
    return out


def _add_member(parent: ArrayRow, job: JobSummary) -> None:
    counts = parent.counts.counts
    if job.array_task_id is not None:
        counts[job.state] = counts.get(job.state, 0) + 1
        parent.tasks.append(TaskRow(job=job))
        return
    assert job.array_task_range is not None
    task_ids = expand_task_range(job.array_task_range)
    counts[job.state] = counts.get(job.state, 0) + len(task_ids)
    for tid in task_ids:
        # One synthetic record per pending id; it shares the range record's job_id and state.
        task = job.model_copy(update={"array_task_id": tid, "array_task_range": None})
        parent.tasks.append(TaskRow(job=task, synthetic=True))


def find_row(rows: list[Row], key: str) -> Row | None:
    for row in rows:
        if row.key == key:
            return row
    return None


def target_is_cancellable(jobs: list[JobSummary], target: CancelTarget) -> bool:
    """True when the latest snapshot still holds the target in a non-terminal state."""
    for job in jobs:
        if job.is_terminal:
            continue
        if target.array_task_id is None:
            if job.job_id == target.job_id or job.array_job_id == target.job_id:
                return True
            continue
        if job.array_job_id != target.job_id:
            continue
        if job.array_task_id == target.array_task_id:
            return True
        if job.array_task_range and target.array_task_id in expand_task_range(job.array_task_range):
            return True
    return False
