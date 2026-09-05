"""Tasks 8.1-8.2: History table, default window, lazy extension, completion handoff."""

from datetime import timedelta

from tests.tui_support import (
    ACTIVE_JOBS,
    HISTORY_JOBS,
    OLDER_HISTORY_JOBS,
    T0,
    FakeClock,
    active_result,
    history_result,
    job,
    make_app,
    standard_session,
)


async def test_history_table_columns_and_cells():
    app = make_app(standard_session())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        table = app.history_table
        labels = [str(c.label) for c in table.columns.values()]
        assert labels == ["id", "name", "state", "partition", "elapsed", "exit", "failed node"]
        assert table.cells("900") == ["900", "done", "COMPLETED", "standard", "0:10:00", "0", ""]
        # Who cancelled it lives in the state cell; a nonzero signal renders as code:signal.
        assert table.cells("901")[2] == "CANCELLED by tester"
        assert table.cells("901")[5] == "0:15"
        assert table.cells("902")[2] == "NODE_FAIL"
        assert table.cells("902")[6] == "gl3021"


async def test_default_window_is_window_days_up_to_now():
    clock = FakeClock()
    session = standard_session()
    app = make_app(session, clock=clock)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        since, until = session.calls["history_jobs"][0]
        assert since == T0 - timedelta(days=7)
        assert until is None

    session = standard_session()
    app = make_app(session, clock=FakeClock(), window_days=3)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        since, _ = session.calls["history_jobs"][0]
        assert since == T0 - timedelta(days=3)


async def test_cursor_on_oldest_row_loads_previous_window_and_appends():
    clock = FakeClock()
    width = timedelta(days=7)
    session = standard_session()
    session.history_queue = [
        history_result(HISTORY_JOBS, T0 - width, T0),
        history_result(OLDER_HISTORY_JOBS, T0 - 2 * width, T0 - width),
    ]
    app = make_app(session, clock=clock)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        await pilot.press("h")
        await pilot.pause()
        table = app.history_table
        assert table.row_count == 3
        assert len(session.calls["history_jobs"]) == 1

        table.move_cursor(row=table.row_count - 1)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert len(session.calls["history_jobs"]) == 2
        since, until = session.calls["history_jobs"][1]
        assert until == T0 - width
        assert since == T0 - 2 * width
        assert table.row_count == 4
        # Older jobs sort below the newer ones, so the oldest stays at the bottom.
        assert list(table.jobs_by_key)[-1] == "800"
        assert table.cells("800")[1] == "ancient"


async def test_history_refresh_failure_keeps_rows():
    from slurm_monitor.models import ConnectionState

    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        session.set_state(ConnectionState.RECONNECTING)
        await app.refresh_history()
        assert app.history_table.row_count == 3
        assert "history refresh failed" in app.last_message


async def test_completion_handoff_fires_only_for_displayed_jobs():
    session = standard_session()
    session.active_queue = [
        active_result(ACTIVE_JOBS),  # 1001 1002 1003 displayed
        active_result([ACTIVE_JOBS[0], ACTIVE_JOBS[2]]),  # 1002 left -> handoff
        active_result([ACTIVE_JOBS[0], ACTIVE_JOBS[2], job(1500, "COMPLETED")]),  # filtered
        active_result([ACTIVE_JOBS[0], ACTIVE_JOBS[2]]),  # 1500 gone but was never displayed
    ]
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        assert len(session.calls["history_jobs"]) == 1

        await app.refresh_active()
        assert "1002" not in app.active_table.rows_by_key
        assert len(session.calls["history_jobs"]) == 2

        await app.refresh_active()
        assert "1500" not in app.active_table.rows_by_key
        assert len(session.calls["history_jobs"]) == 2

        await app.refresh_active()
        assert len(session.calls["history_jobs"]) == 2


async def test_handoff_fires_once_while_frozen():
    session = standard_session()
    session.active_queue = [
        active_result(ACTIVE_JOBS),
        active_result([ACTIVE_JOBS[0]]),
        active_result([ACTIVE_JOBS[0]]),
    ]
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        await pilot.press("c")  # opens the confirmation: table frozen
        await app.refresh_active()
        await app.refresh_active()
        assert len(session.calls["history_jobs"]) == 2
        await pilot.press("n")
        await pilot.pause()
        assert app.active_table.row_count == 1
