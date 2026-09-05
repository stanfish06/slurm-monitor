"""Tasks 10.1-10.4: confirmation text, frozen table + id-stable target, refusals, history rows."""

from slurm_monitor.transport import AgentError
from slurm_monitor.tui.widgets import ConfirmCancel
from tests.tui_support import (
    ACTIVE_JOBS,
    MIXED_ARRAY_JOBS,
    active_result,
    make_app,
    standard_session,
)


async def _open_confirm(app, pilot, key):
    table = app.active_table
    table.move_cursor(row=table.get_row_index(key))
    await pilot.pause()
    await pilot.press("c")
    await pilot.pause()
    assert isinstance(app.screen, ConfirmCancel), type(app.screen)
    return app.screen.prompt


async def _settle(app, pilot):
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_confirmation_names_job_array_or_task():
    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        assert await _open_confirm(app, pilot, "1001") == "Cancel job 1001 (train)?"
        await pilot.press("n")
        await pilot.pause()

        assert await _open_confirm(app, pilot, "123") == "Cancel array 123 and all its 6 tasks?"
        await pilot.press("escape")
        await pilot.pause()

        app.active_table.move_cursor(row=app.active_table.get_row_index("123"))
        await pilot.press("space")
        await pilot.pause()
        assert await _open_confirm(app, pilot, "123_4") == "Cancel array task 123_4?"
        await pilot.press("n")
        await pilot.pause()

        assert session.calls["cancel"] == []
        assert not isinstance(app.screen, ConfirmCancel)


async def test_confirm_targets_job_and_task_and_array_by_id():
    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        await _open_confirm(app, pilot, "1001")
        await pilot.press("y")
        await _settle(app, pilot)
        assert session.calls["cancel"] == [(1001, None)]
        assert "cancel requested for job 1001" in app.last_message

        await _open_confirm(app, pilot, "123")
        await pilot.press("enter")
        await _settle(app, pilot)
        assert session.calls["cancel"][-1] == (123, None)

        app.active_table.move_cursor(row=app.active_table.get_row_index("123"))
        await pilot.press("space")
        await pilot.pause()
        await _open_confirm(app, pilot, "123_5")
        await pilot.press("y")
        await _settle(app, pilot)
        assert session.calls["cancel"][-1] == (123, 5)


async def test_refresh_during_confirmation_freezes_table_and_keeps_target():
    session = standard_session()
    reordered = [ACTIVE_JOBS[2], ACTIVE_JOBS[1], ACTIVE_JOBS[0]]
    session.active_queue = [active_result(ACTIVE_JOBS), active_result(reordered)]
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        table = app.active_table
        assert list(table.rows_by_key) == ["1001", "1002", "1003"]
        table.move_cursor(row=0)
        await pilot.press("c")
        await pilot.pause()
        n_calls = len(session.calls["active_jobs"])

        await app.refresh_active()  # fetches, but must not re-render under the modal
        assert len(session.calls["active_jobs"]) == n_calls + 1
        assert list(table.rows_by_key) == ["1001", "1002", "1003"]
        assert app.frozen

        await pilot.press("y")
        await _settle(app, pilot)
        # Row 0 now holds 1003, but the target was captured by id.
        assert session.calls["cancel"] == [(1001, None)]
        assert not app.frozen
        assert list(table.rows_by_key)[0] == "1003"


async def test_target_gone_before_confirm_is_reported_and_not_sent():
    session = standard_session()
    finished = ACTIVE_JOBS[0].model_copy(update={"state": "COMPLETED"})
    session.active_queue = [
        active_result(ACTIVE_JOBS),
        active_result([finished, ACTIVE_JOBS[1], ACTIVE_JOBS[2]]),
    ]
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        await _open_confirm(app, pilot, "1001")
        await app.refresh_active()
        await pilot.press("y")
        await _settle(app, pilot)
        assert session.calls["cancel"] == []
        assert app.last_message == "job 1001 is no longer cancellable"
        assert "1001" not in app.active_table.rows_by_key


async def test_refusal_is_reported_and_state_untouched():
    session = standard_session()
    session.cancel_error = AgentError("rpc_error", "Access/permission denied")
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        await _open_confirm(app, pilot, "1001")
        await pilot.press("y")
        await _settle(app, pilot)
        assert session.calls["cancel"] == [(1001, None)]
        assert "Access/permission denied" in app.last_message
        assert app.active_table.cells("1001")[2] == "RUNNING"


async def test_history_rows_cannot_be_cancelled():
    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        await pilot.press("h")
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmCancel)
        assert app.last_message == "history rows cannot be cancelled"
        assert session.calls["cancel"] == []


async def test_array_confirmation_count_matches_histogram_total():
    session = standard_session(active_jobs=MIXED_ARRAY_JOBS)
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        prompt = await _open_confirm(app, pilot, "123")
        assert prompt == "Cancel array 123 and all its 6 tasks?"
        assert app.pending_target is not None
        assert app.pending_target.task_count == 6
        await pilot.press("n")
        await pilot.pause()
        assert app.pending_target is None
