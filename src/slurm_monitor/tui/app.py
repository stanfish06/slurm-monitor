"""SlurmMonitorApp: two job tables, side panels, and the refresh/cancel logic around them.

The app talks only to a ClientSession-shaped object (see slurm_monitor.session). Every refresh
is an ordinary coroutine (`refresh_active`, `refresh_history`, `refresh_usage`, `tick_age`) so
tests drive them directly with a fake clock; the interval timers merely call them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, TabbedContent, TabPane

from slurm_monitor.config import Config, Overrides, Resolved, resolve
from slurm_monitor.core.arrays import (
    ArrayRow,
    CancelTarget,
    PlainRow,
    Row,
    TaskRow,
    collapse_arrays,
    target_is_cancellable,
)
from slurm_monitor.models import ConnectionState, JobSummary, JobUsage
from slurm_monitor.transport import (
    AgentError,
    AuthError,
    BootstrapRequired,
    TransportError,
    VersionMismatch,
)
from slurm_monitor.tui.widgets import (
    ActiveTable,
    ConfirmCancel,
    DetailPanel,
    HelpScreen,
    HistoryTable,
    MessageLine,
    StatusBar,
    UsagePanel,
)
from slurm_monitor.tui.widgets.status import STALE_STATES

# Stop extending the history window past this many days back.
MAX_HISTORY_DAYS = 366


def _local_now() -> datetime:
    return datetime.now().astimezone()


class SlurmMonitorApp(App[int]):
    CSS_PATH = "app.tcss"
    TITLE = "slurm-monitor"

    BINDINGS = [
        Binding("a", "show_tab('active')", "Active"),
        Binding("h", "show_tab('history')", "History"),
        Binding("tab", "next_tab", "Switch tab", show=False, priority=True),
        Binding("r", "refresh", "Refresh"),
        Binding("space", "toggle_expand", "Expand", show=False),
        Binding("c", "cancel_job", "Cancel"),
        Binding("d", "toggle_panel", "Detail"),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        session,
        settings: Resolved,
        *,
        clock: Callable[[], datetime] = _local_now,
        start_timers: bool = True,
    ) -> None:
        super().__init__()
        self.session = session
        self.settings = settings
        self.clock = clock
        self.start_timers = start_timers

        self.active_snapshot: list[JobSummary] = []  # latest pending/running records
        self.expanded: set[int] = set()  # array_job_ids shown per task
        self.displayed_ids: set[int] = set()  # job ids currently rendered in Active
        self._handoff_done: set[int] = set()  # ids whose disappearance already fired a refresh
        self.frozen = False  # True while a cancel confirmation is open
        self._render_pending = False
        self._active_inflight = False

        self.history_base: list[JobSummary] = []  # default window, replaced on each refresh
        self.history_older: list[JobSummary] = []  # extension windows, appended
        self.history_oldest: datetime | None = None  # `since` of the oldest window loaded
        self._history_inflight = False
        self._extending = False
        self._last_history_key: str | None = None

        self.selected_job_id: int | None = None
        self.selected_running = False
        self.usage: JobUsage | None = None
        self._usage_for: int | None = None
        self.degraded = bool(getattr(session, "degraded", False))
        self.pending_target: CancelTarget | None = None
        self.last_message = ""
        self.ready = asyncio.Event()  # set once the session started and both tables loaded
        self._mounted = False

    # -- layout --------------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield StatusBar(self.settings.cluster_name, id="status")
        with Horizontal(id="body"):
            with TabbedContent(initial="active", id="tabs"):
                with TabPane("Active", id="active"):
                    yield ActiveTable(id="active-table")
                with TabPane("History", id="history"):
                    yield HistoryTable(id="history-table")
            with Vertical(id="side"):
                yield DetailPanel(id="detail")
                yield UsagePanel(id="usage")
        yield MessageLine(id="message")
        yield Footer()

    @property
    def active_table(self) -> ActiveTable:
        return self.query_one("#active-table", ActiveTable)

    @property
    def history_table(self) -> HistoryTable:
        return self.query_one("#history-table", HistoryTable)

    @property
    def status_bar(self) -> StatusBar:
        return self.query_one("#status", StatusBar)

    @property
    def detail_panel(self) -> DetailPanel:
        return self.query_one("#detail", DetailPanel)

    @property
    def usage_panel(self) -> UsagePanel:
        return self.query_one("#usage", UsagePanel)

    @property
    def current_tab(self) -> str:
        return self.query_one("#tabs", TabbedContent).active or "active"

    def on_mount(self) -> None:
        self._mounted = True
        self.detail_panel.show_empty()
        self.usage_panel.show_empty()
        self.session.on_state_change(self._on_session_state)
        self._update_status()
        self.active_table.focus()
        if self.start_timers:
            iv = self.settings.intervals
            self.set_interval(iv.active_seconds, self._timer_active)
            self.set_interval(iv.history_seconds, self._timer_history)
            self.set_interval(iv.usage_seconds, self._timer_usage)
            self.set_interval(1.0, self.tick_age)
        self.run_worker(self._start_session(), exit_on_error=False)

    async def _start_session(self) -> None:
        try:
            await self.session.start()
        except (BootstrapRequired, VersionMismatch, AuthError) as e:
            # Not retryable: leave the screen and print the instruction on the plain terminal.
            self.exit(return_code=1, message=f"slurm-monitor: {e}")
            return
        except TransportError as e:
            self.set_message(f"connection failed: {e}")
        self.degraded = self.degraded or bool(getattr(self.session, "degraded", False))
        self._update_status()
        await self.refresh_active()
        await self.refresh_history()
        self.ready.set()

    async def action_quit(self) -> None:
        try:
            await self.session.stop()
        finally:
            self.exit(return_code=0)

    # -- timers --------------------------------------------------------------------------------

    def _timer_active(self) -> None:
        self.run_worker(self.refresh_active(), exit_on_error=False, group="active")

    def _timer_history(self) -> None:
        self.run_worker(self.refresh_history(), exit_on_error=False, group="history")

    def _timer_usage(self) -> None:
        self.run_worker(self.refresh_usage(), exit_on_error=False, group="usage")

    def tick_age(self) -> None:
        """Once a second: keep the `data from Ns ago` annotation current while stale."""
        if self.session.state in STALE_STATES:
            self._update_status()

    # -- status / messages ---------------------------------------------------------------------

    def set_message(self, text: str) -> None:
        self.last_message = text
        if self._mounted:
            self.query_one("#message", MessageLine).show(text)
            self.notify(text)

    def _data_age(self) -> float | None:
        last = self.session.last_success
        if last is None:
            return None
        now = self.clock()
        if last.tzinfo is None and now.tzinfo is not None:
            last = last.astimezone()
        return (now - last).total_seconds()

    def _update_status(self) -> None:
        if not self._mounted:
            return
        state = self.session.state
        self.status_bar.render_state(
            state, self.session.last_error, self.degraded, self._data_age()
        )
        stale = state in STALE_STATES
        for table in self.query(DataTable):
            table.set_class(stale, "stale")

    def _on_session_state(self, state: ConnectionState) -> None:
        self._update_status()
        if state == ConnectionState.AUTH_REQUIRED and self.session.last_error:
            self.set_message(self.session.last_error)
        if state == ConnectionState.CONNECTED and self._mounted and self.ready.is_set():
            # Back online: replace stale data right away rather than waiting for the timers.
            self.run_worker(self._refresh_all(), exit_on_error=False)

    async def _refresh_all(self) -> None:
        await self.refresh_active()
        await self.refresh_history()
        await self.refresh_usage(force=True)

    # -- active tab ----------------------------------------------------------------------------

    async def refresh_active(self) -> None:
        """Fetch the active list; render unless a confirmation is open; fire the completion
        handoff for ids that were displayed and are now absent."""
        if self._active_inflight:
            return
        self._active_inflight = True
        try:
            try:
                result = await self.session.active_jobs()
            except (TransportError, AgentError) as e:
                self.set_message(f"active refresh failed: {e}")
                self._update_status()
                return
            jobs = [j for j in result.jobs if j.is_active]
            if result.degraded != self.degraded:
                self.degraded = result.degraded or bool(getattr(self.session, "degraded", False))
            self.active_snapshot = jobs
            new_ids = {j.job_id for j in jobs}
            gone = (self.displayed_ids - new_ids) - self._handoff_done
            if self.frozen:
                self._render_pending = True
            else:
                self._render_active()
            self._update_status()
        finally:
            self._active_inflight = False
        if gone:
            self._handoff_done |= gone
            await self.refresh_history()

    def _render_active(self) -> None:
        rows = collapse_arrays(self.active_snapshot, self.expanded)
        self.active_table.render_rows(rows)
        self.displayed_ids = {j.job_id for j in self.active_snapshot}
        self._handoff_done = set()
        self._render_pending = False
        self._sync_selection()

    def active_rows(self) -> list[Row]:
        return list(self.active_table.rows_by_key.values())

    def action_toggle_expand(self) -> None:
        if self.current_tab != "active" or self.frozen:
            return
        row = self.active_table.current_row()
        if isinstance(row, ArrayRow):
            self.expanded ^= {row.array_job_id}
        elif isinstance(row, TaskRow):
            self.expanded.discard(row.array_job_id)
        else:
            return
        self._render_active()
        if isinstance(row, TaskRow):
            self.active_table.restore_cursor(str(row.array_job_id))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "active-table":
            self.action_toggle_expand()

    # -- history tab ---------------------------------------------------------------------------

    def _window_width(self) -> timedelta:
        return timedelta(days=self.settings.history.window_days)

    async def refresh_history(self) -> None:
        """Re-query the default window (last window_days up to now) and re-render."""
        if self._history_inflight:
            return
        self._history_inflight = True
        try:
            since = self.clock() - self._window_width()
            try:
                result = await self.session.history_jobs(since, None)
            except (TransportError, AgentError) as e:
                self.set_message(f"history refresh failed: {e}")
                self._update_status()
                return
            self.history_base = [j for j in result.jobs if j.is_terminal or not j.is_active]
            if self.history_oldest is None or self.history_oldest > since:
                self.history_oldest = since
            self._render_history()
            self._update_status()
        finally:
            self._history_inflight = False

    async def extend_history(self) -> None:
        """Load the window preceding the oldest one loaded (same width) and append its jobs."""
        if self._extending or self.history_oldest is None:
            return
        if self.clock() - self.history_oldest > timedelta(days=MAX_HISTORY_DAYS):
            return
        self._extending = True
        try:
            until = self.history_oldest
            since = until - self._window_width()
            try:
                result = await self.session.history_jobs(since, until)
            except (TransportError, AgentError) as e:
                self.set_message(f"history extension failed: {e}")
                return
            self.history_oldest = since
            self.history_older.extend(result.jobs)
            self._render_history()
            self.set_message(
                f"history now covers {(self.clock() - since).days} days"
                if result.jobs
                else f"no jobs between {since:%Y-%m-%d} and {until:%Y-%m-%d}"
            )
        finally:
            self._extending = False

    def _render_history(self) -> None:
        def sort_key(j: JobSummary) -> tuple[int, float]:
            # Newest first; jobs without an end time sort by id at the top.
            if j.end_time is None:
                return (0, -j.job_id)
            return (1, -j.end_time.timestamp())

        jobs = sorted(self.history_base + self.history_older, key=sort_key)
        self.history_table.render_jobs(jobs)
        # The cursor is back on the row it was on; only a later move onto the last row extends.
        self._last_history_key = self.history_table.current_key()
        self._sync_selection()

    def history_jobs(self) -> list[JobSummary]:
        return list(self.history_table.jobs_by_key.values())

    # -- selection, detail and usage -----------------------------------------------------------

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = event.data_table
        if table.id == "history-table" and event.row_key is not None:
            key = event.row_key.value
            if key == self.history_table.last_key() and self._last_history_key not in (None, key):
                self.run_worker(self.extend_history(), exit_on_error=False, group="extend")
            self._last_history_key = key
        self._sync_selection()

    def _current_selection(self) -> tuple[int | None, bool]:
        if self.current_tab == "active":
            row = self.active_table.current_row()
            if row is None:
                return None, False
            if isinstance(row, ArrayRow):
                return row.array_job_id, row.running_count > 0
            if isinstance(row, PlainRow):
                return row.job.job_id, row.job.state.startswith("RUNNING")
            return row.job.job_id, row.job.state.startswith("RUNNING")
        job = self.history_table.current_job()
        return (None, False) if job is None else (job.job_id, False)

    def _sync_selection(self) -> None:
        job_id, running = self._current_selection()
        self.selected_running = running
        if job_id == self.selected_job_id:
            return
        self.selected_job_id = job_id
        self.usage = None
        self._usage_for = None
        if job_id is None:
            self.detail_panel.show_empty()
            self.usage_panel.show_empty()
            return
        self.run_worker(self.load_detail(job_id), exit_on_error=False, group="detail")

    async def load_detail(self, job_id: int) -> None:
        try:
            detail = await self.session.job_detail(job_id)
        except (TransportError, AgentError) as e:
            if self.selected_job_id == job_id:
                self.detail_panel.show_empty(f"detail unavailable: {e}")
            return
        if self.selected_job_id != job_id:
            return
        self.detail_panel.show(detail)
        await self.refresh_usage(force=True)

    async def refresh_usage(self, *, force: bool = False) -> None:
        """Fetch usage for the selected job. On the timer, only running jobs are re-polled:
        terminal figures never change."""
        job_id = self.selected_job_id
        if job_id is None:
            return
        if not force and self._usage_for == job_id and not self.selected_running:
            return
        try:
            usage = await self.session.job_usage(job_id)
        except (TransportError, AgentError) as e:
            if self.selected_job_id == job_id:
                self.usage_panel.show_empty(f"usage unavailable: {e}")
            return
        if self.selected_job_id != job_id:
            return
        self.usage = usage
        self._usage_for = job_id
        self.usage_panel.show(usage)

    # -- cancellation --------------------------------------------------------------------------

    @staticmethod
    def confirmation_text(row: Row) -> str:
        if isinstance(row, ArrayRow):
            n = row.counts.total
            noun = "task" if n == 1 else "tasks"
            return f"Cancel array {row.array_job_id} and all its {n} {noun}?"
        if isinstance(row, TaskRow):
            return f"Cancel array task {row.key}?"
        return f"Cancel job {row.job.job_id} ({row.job.name})?"

    def action_cancel_job(self) -> None:
        if self.frozen:
            return
        if self.current_tab != "active":
            self.set_message("history rows cannot be cancelled")
            return
        row = self.active_table.current_row()
        if row is None:
            self.set_message("no job selected")
            return
        target = row.cancel_target
        self.pending_target = target
        self.frozen = True  # keep rows still under the modal; refreshes queue a re-render
        self.push_screen(ConfirmCancel(self.confirmation_text(row)), self._confirm_done)

    def _confirm_done(self, confirmed: bool | None) -> None:
        target = self.pending_target
        self.pending_target = None
        self.frozen = False
        if self._render_pending:
            self._render_active()
        if confirmed and target is not None:
            self.run_worker(self.perform_cancel(target), exit_on_error=False, group="cancel")

    async def perform_cancel(self, target: CancelTarget) -> None:
        label = f"array task {target.label}" if target.array_task_id is not None else "job"
        if target.array_task_id is None and target.task_count > 1:
            label = "array"
        if not target_is_cancellable(self.active_snapshot, target):
            self.set_message(f"{label} {target.label} is no longer cancellable")
            return
        try:
            result = await self.session.cancel(target.job_id, target.array_task_id)
        except AgentError as e:
            self.set_message(f"cancel {target.label} refused: {e.message}")
            return
        except TransportError as e:
            self.set_message(f"cancel {target.label} failed: {e}")
            return
        if result.cancelled:
            self.set_message(f"cancel requested for {label} {target.label}")
        else:
            self.set_message(f"cancel {target.label} not performed: {result.message or 'refused'}")
        await self.refresh_active()

    # -- navigation ----------------------------------------------------------------------------

    def action_show_tab(self, name: str) -> None:
        self.query_one("#tabs", TabbedContent).active = name

    def action_next_tab(self) -> None:
        self.action_show_tab("history" if self.current_tab == "active" else "active")

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        table = self.active_table if event.pane.id == "active" else self.history_table
        table.focus()
        self._sync_selection()

    async def action_refresh(self) -> None:
        if self.current_tab == "active":
            await self.refresh_active()
        else:
            await self.refresh_history()
        await self.refresh_usage(force=True)

    def action_toggle_panel(self) -> None:
        side = self.query_one("#side")
        side.display = not side.display

    def action_help(self) -> None:
        self.push_screen(HelpScreen())


# -- entry points ------------------------------------------------------------------------------


def run_tui(settings: Resolved) -> int:
    from slurm_monitor.session import ClientSession, make_transport

    session = ClientSession(lambda: make_transport(settings))
    app = SlurmMonitorApp(session, settings)
    app.run()
    return app.return_code or 0


def run_tui_with_fixtures(directory: Path, config: Config, overrides: Overrides) -> int:
    """Replay a recorded fixture directory; needs no cluster and no ssh."""
    from slurm_monitor.session import ClientSession
    from slurm_monitor.transport.fixture import FixtureTransport

    settings = fixture_settings(directory, config, overrides)
    session = ClientSession(lambda: FixtureTransport(directory))
    app = SlurmMonitorApp(session, settings)
    app.run()
    return app.return_code or 0


def fixture_settings(directory: Path, config: Config, overrides: Overrides) -> Resolved:
    """Intervals and window from config/overrides; the cluster entry is a local placeholder."""
    local = overrides.model_copy(
        update={"local": True, "cluster": f"fixtures:{directory.name}", "host": None}
    )
    return resolve(config, local)
