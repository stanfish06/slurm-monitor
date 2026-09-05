"""Active and History DataTables. Both render from lists the app owns and keep a key -> record
map so the app resolves the selected row by id, never by index."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import DataTable

from slurm_monitor.core.arrays import ArrayRow, PlainRow, Row, TaskRow
from slurm_monitor.models import JobSummary
from slurm_monitor.tui.format import format_elapsed, format_exit, format_state


def _cell(value: object) -> Text:
    """Plain Text so ids like 123_[3-6] are never parsed as markup."""
    return Text("" if value is None else str(value))


class _JobTable(DataTable):
    """DataTable with row cursor, restoring the cursor to the same key across re-renders."""

    def __init__(self, **kwargs) -> None:
        super().__init__(cursor_type="row", zebra_stripes=True, **kwargs)

    def current_key(self) -> str | None:
        if self.row_count == 0:
            return None
        try:
            return self.coordinate_to_cell_key(self.cursor_coordinate).row_key.value
        except Exception:
            return None

    def restore_cursor(self, key: str | None) -> None:
        if key is None or self.row_count == 0:
            return
        try:
            index = self.get_row_index(key)
        except Exception:
            return
        self.move_cursor(row=index, scroll=False)

    def cells(self, key: str) -> list[str]:
        return [str(c) for c in self.get_row(key)]


class ActiveTable(_JobTable):
    COLUMNS = ("id", "name", "state", "partition", "elapsed", "nodes", "reason")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rows_by_key: dict[str, Row] = {}

    def on_mount(self) -> None:
        for name in self.COLUMNS:
            self.add_column(name, key=name)

    def render_rows(self, rows: list[Row]) -> None:
        key = self.current_key()
        self.clear()
        self.rows_by_key = {}
        for row in rows:
            self.rows_by_key[row.key] = row
            self.add_row(*self._cells(row), key=row.key)
        self.restore_cursor(key)

    def current_row(self) -> Row | None:
        key = self.current_key()
        return None if key is None else self.rows_by_key.get(key)

    @staticmethod
    def _cells(row: Row) -> list[Text]:
        if isinstance(row, PlainRow):
            j = row.job
            return [
                _cell(j.display_id),
                _cell(j.name),
                _cell(format_state(j)),
                _cell(j.partition),
                _cell(format_elapsed(j.elapsed_seconds)),
                _cell(j.num_nodes),
                _cell(j.state_reason if j.state.startswith("PENDING") else None),
            ]
        if isinstance(row, ArrayRow):
            marker = "▾" if row.expanded else "▸"  # ▾ / ▸
            return [
                _cell(f"{marker} {row.array_job_id}"),
                _cell(row.name),
                _cell(row.summary),
                _cell(row.partition),
                _cell(format_elapsed(row.elapsed_seconds)),
                _cell(row.num_nodes),
                _cell(None),
            ]
        assert isinstance(row, TaskRow)
        j = row.job
        return [
            _cell(f"  {j.display_id}"),
            _cell(j.name),
            _cell(format_state(j)),
            _cell(j.partition),
            _cell(format_elapsed(j.elapsed_seconds)),
            _cell(j.num_nodes),
            _cell(j.state_reason if j.state.startswith("PENDING") else None),
        ]


class HistoryTable(_JobTable):
    COLUMNS = ("id", "name", "state", "partition", "elapsed", "exit", "failed node")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.jobs_by_key: dict[str, JobSummary] = {}

    def on_mount(self) -> None:
        for name in self.COLUMNS:
            self.add_column(name, key=name)

    def render_jobs(self, jobs: list[JobSummary]) -> None:
        key = self.current_key()
        self.clear()
        self.jobs_by_key = {}
        for j in jobs:
            row_key = j.display_id
            # Accounting can hold the same display id twice (requeued jobs); keep both rows.
            n = 1
            while row_key in self.jobs_by_key:
                n += 1
                row_key = f"{j.display_id}#{n}"
            self.jobs_by_key[row_key] = j
            self.add_row(
                _cell(j.display_id),
                _cell(j.name),
                _cell(format_state(j)),
                _cell(j.partition),
                _cell(format_elapsed(j.elapsed_seconds)),
                _cell(format_exit(j.exit_code, j.exit_signal)),
                _cell(j.failed_node),
                key=row_key,
            )
        self.restore_cursor(key)

    def current_job(self) -> JobSummary | None:
        key = self.current_key()
        return None if key is None else self.jobs_by_key.get(key)

    def last_key(self) -> str | None:
        if not self.jobs_by_key:
            return None
        return next(reversed(self.jobs_by_key))
