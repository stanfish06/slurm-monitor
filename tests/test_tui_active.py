"""Tasks 7.1-7.4: app shell, Active table cells, refresh, array collapse/expand."""

from textual.widgets import TabbedContent, TabPane

from slurm_monitor.tui.widgets import ConfirmCancel, HelpScreen
from tests.tui_support import (
    ACTIVE_JOBS,
    MIXED_ARRAY_JOBS,
    active_result,
    job,
    make_app,
    standard_session,
)


async def test_app_boots_with_two_tabs_and_keys_switch_them():
    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        panes = [p.id for p in app.query(TabPane)]
        assert panes == ["active", "history"]
        assert app.current_tab == "active"
        assert session.started
        assert app.active_table.row_count == 4
        assert app.history_table.row_count == 3

        await pilot.press("h")
        assert app.current_tab == "history"
        await pilot.press("a")
        assert app.current_tab == "active"
        await pilot.press("tab")
        assert app.current_tab == "history"
        await pilot.press("tab")
        assert app.current_tab == "active"
        assert app.query_one(TabbedContent).active == "active"

        await pilot.press("question_mark")
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        assert not isinstance(app.screen, (HelpScreen, ConfirmCancel))

        await pilot.press("d")
        assert app.query_one("#side").display is False
        await pilot.press("d")
        assert app.query_one("#side").display is True


async def test_quit_key_stops_session_and_exits():
    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.ready.wait()
        await pilot.press("q")
        await pilot.pause()
    assert session.stopped
    assert app.return_code == 0


async def test_active_table_columns_and_cells():
    app = make_app(standard_session())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        table = app.active_table
        labels = [str(c.label) for c in table.columns.values()]
        assert labels == ["id", "name", "state", "partition", "elapsed", "nodes", "reason"]
        assert table.cells("1001") == ["1001", "train", "RUNNING", "gpu", "1:02:05", "1", ""]
        assert table.cells("1002") == [
            "1002",
            "preprocess",
            "PENDING",
            "standard",
            "0:00:00",
            "2",
            "Priority",
        ]
        # >= 1 day switches to D-HH:MM:SS.
        assert table.cells("1003")[4] == "1-01:01:01"


async def test_terminal_jobs_from_controller_are_not_listed():
    finished = job(1500, "COMPLETED", name="stale-in-controller")
    session = standard_session(active_jobs=[*ACTIVE_JOBS, finished])
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        assert "1500" not in app.active_table.rows_by_key
        assert app.active_table.row_count == 3


async def test_state_change_shows_after_one_refresh():
    first = active_result(ACTIVE_JOBS)
    second = active_result(
        [ACTIVE_JOBS[0], ACTIVE_JOBS[1].model_copy(update={"state": "RUNNING"}), ACTIVE_JOBS[2]]
    )
    session = standard_session()
    session.active_queue = [first, second]
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        assert app.active_table.cells("1002")[2] == "PENDING"
        await app.refresh_active()
        assert app.active_table.cells("1002")[2] == "RUNNING"


async def test_interval_timer_drives_refresh():
    first = active_result(ACTIVE_JOBS)
    second = active_result([ACTIVE_JOBS[1].model_copy(update={"state": "RUNNING"})])
    session = standard_session()
    session.active_queue = [first, second]
    app = make_app(session, start_timers=True, active_seconds=0.05, history_seconds=60)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause(0.3)
        assert app.active_table.cells("1002")[2] == "RUNNING"
        assert app.active_table.row_count == 1
        assert len(session.calls["active_jobs"]) >= 2


async def test_manual_refresh_key_queries_current_tab():
    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        n_active = len(session.calls["active_jobs"])
        n_history = len(session.calls["history_jobs"])
        await pilot.press("r")
        await pilot.pause()
        assert len(session.calls["active_jobs"]) == n_active + 1
        assert len(session.calls["history_jobs"]) == n_history
        await pilot.press("h")
        await pilot.press("r")
        await pilot.pause()
        assert len(session.calls["history_jobs"]) == n_history + 1


async def test_array_collapses_with_histogram_and_expands_to_tasks():
    session = standard_session(active_jobs=MIXED_ARRAY_JOBS)
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        table = app.active_table
        assert list(table.rows_by_key) == ["123"]
        cells = table.cells("123")
        assert cells[0].endswith("123")
        assert cells[1] == "sweep"
        # Counts include the four pending ids from "3-6" and the two running task records.
        assert cells[2] == "6 tasks: 2 RUNNING, 4 PENDING"
        assert cells[4] == "0:02:00"

        await pilot.press("space")
        await pilot.pause()
        assert list(table.rows_by_key) == ["123"] + [f"123_{i}" for i in range(1, 7)]
        assert table.cells("123_1")[:3] == ["  123_1", "sweep", "RUNNING"]
        assert table.cells("123_4")[:3] == ["  123_4", "sweep", "PENDING"]
        assert table.cells("123")[0].startswith("▾")

        # Enter on a task row collapses its parent again and leaves the cursor on the parent.
        table.move_cursor(row=table.get_row_index("123_4"))
        await pilot.press("enter")
        await pilot.pause()
        assert list(table.rows_by_key) == ["123"]
        assert table.current_key() == "123"


async def test_expansion_survives_refresh():
    session = standard_session(active_jobs=MIXED_ARRAY_JOBS)
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        await app.refresh_active()
        assert app.active_table.row_count == 7
