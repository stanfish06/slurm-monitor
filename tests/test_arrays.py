import pytest

from slurm_monitor.core.arrays import (
    ArrayRow,
    CancelTarget,
    PlainRow,
    TaskRow,
    collapse_arrays,
    expand_task_range,
    target_is_cancellable,
)
from slurm_monitor.models import JobSummary, Source


def _job(job_id: int, state: str = "RUNNING", **kw) -> JobSummary:
    kw.setdefault("name", f"job{job_id}")
    return JobSummary(job_id=job_id, state=state, source=Source.CONTROLLER, **kw)


def mixed_array() -> list[JobSummary]:
    """Array 123: tasks 1 and 2 running as separate records, 3-6 pending as one range record."""
    return [
        _job(500, name="solo"),
        _job(123, "PENDING", name="arr", array_job_id=123, array_task_range="3-6"),
        _job(124, "RUNNING", name="arr", array_job_id=123, array_task_id=1, elapsed_seconds=60),
        _job(125, "RUNNING", name="arr", array_job_id=123, array_task_id=2, elapsed_seconds=30),
        _job(600, "PENDING", name="other", state_reason="Priority"),
    ]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("3-6", [3, 4, 5, 6]),
        ("1-3,7,9-10", [1, 2, 3, 7, 9, 10]),
        ("1-6%2", [1, 2, 3, 4, 5, 6]),
        ("0-8:4", [0, 4, 8]),
        ("5", [5]),
        ("2,1,2", [1, 2]),
        ("", []),
    ],
)
def test_expand_task_range(spec, expected):
    assert expand_task_range(spec) == expected


def test_expand_task_range_rejects_descending():
    with pytest.raises(ValueError):
        expand_task_range("6-3")


def test_mixed_array_collapses_to_one_parent_with_combined_counts():
    rows = collapse_arrays(mixed_array())
    assert [type(r) for r in rows] == [PlainRow, ArrayRow, PlainRow]
    parent = rows[1]
    assert isinstance(parent, ArrayRow)
    assert parent.array_job_id == 123
    assert parent.counts.counts == {"PENDING": 4, "RUNNING": 2}
    assert parent.counts.total == 6
    assert parent.summary == "6 tasks: 2 RUNNING, 4 PENDING"
    assert parent.elapsed_seconds == 60
    assert parent.key == "123"
    assert parent.cancel_target == CancelTarget(job_id=123, task_count=6, label="123")


def test_expansion_yields_per_task_rows_in_task_order():
    rows = collapse_arrays(mixed_array(), expanded={123})
    tasks = [r for r in rows if isinstance(r, TaskRow)]
    assert [t.task_id for t in tasks] == [1, 2, 3, 4, 5, 6]
    assert [t.job.display_id for t in tasks] == [f"123_{i}" for i in range(1, 7)]
    assert [t.job.state for t in tasks] == ["RUNNING"] * 2 + ["PENDING"] * 4
    assert [t.synthetic for t in tasks] == [False, False, True, True, True, True]
    # Task rows sit directly beneath their parent.
    assert rows.index(tasks[0]) == 2
    assert tasks[3].cancel_target == CancelTarget(job_id=123, array_task_id=4, label="123_4")


def test_plain_job_cancel_target():
    rows = collapse_arrays([_job(7, name="x")])
    assert rows[0].cancel_target == CancelTarget(job_id=7, label="7")


def test_target_is_cancellable_checks_latest_snapshot():
    jobs = mixed_array()
    assert target_is_cancellable(jobs, CancelTarget(job_id=500))
    assert target_is_cancellable(jobs, CancelTarget(job_id=123, task_count=6))
    assert target_is_cancellable(jobs, CancelTarget(job_id=123, array_task_id=1))
    # Pending task inside the range record.
    assert target_is_cancellable(jobs, CancelTarget(job_id=123, array_task_id=5))
    assert not target_is_cancellable(jobs, CancelTarget(job_id=123, array_task_id=9))
    assert not target_is_cancellable(jobs, CancelTarget(job_id=999))
    # A terminal record no longer counts.
    done = [_job(500, "COMPLETED", name="solo")]
    assert not target_is_cancellable(done, CancelTarget(job_id=500))
