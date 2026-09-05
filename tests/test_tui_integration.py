"""The app over the real ClientSession and FixtureTransport, replaying a synthetic directory.
Exercises the same wiring as `slurm-monitor tui --fixtures DIR` minus the terminal."""

from pathlib import Path

from slurm_monitor.config import Config, Overrides
from slurm_monitor.session import ClientSession
from slurm_monitor.transport.fixture import FixtureTransport
from slurm_monitor.tui.app import SlurmMonitorApp, fixture_settings
from slurm_monitor.tui.widgets import ConfirmCancel

FIXTURES = Path(__file__).parent / "fixtures" / "synthetic" / "mixed_array"


def _app(directory: Path = FIXTURES) -> tuple[SlurmMonitorApp, ClientSession]:
    settings = fixture_settings(directory, Config(), Overrides(active_seconds=0.5))
    session = ClientSession(lambda: FixtureTransport(directory))
    return SlurmMonitorApp(session, settings, start_timers=False), session


async def test_app_renders_fixture_directory_through_client_session():
    app, session = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.state.value == "connected"
        assert app.settings.cluster_name == "fixtures:mixed_array"
        assert "CONNECTED" in app.status_bar.text

        table = app.active_table
        assert list(table.rows_by_key) == ["1001", "1002", "1003", "123"]
        assert table.cells("123")[2] == "6 tasks: 2 RUNNING, 4 PENDING"
        assert app.history_table.cells("901")[2] == "CANCELLED by tester"
        assert "stdout      /home/tester/run1001/slurm-1001.out" in app.detail_panel.text
        assert "gpu util    87.5%" in app.usage_panel.text

        # Second snapshot: task 3 started, so the histogram shifts but the total holds.
        await app.refresh_active()
        assert table.cells("123")[2] == "6 tasks: 3 RUNNING, 3 PENDING"
        table.move_cursor(row=table.get_row_index("123"))
        await pilot.press("space")
        await pilot.pause()
        assert "123_3" in table.rows_by_key

        # Cancel a task through the real session; the fixture answers cancel/123_4.json.
        table.move_cursor(row=table.get_row_index("123_4"))
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmCancel)
        assert app.screen.prompt == "Cancel array task 123_4?"
        await pilot.press("y")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "cancel requested for array task 123_4" in app.last_message

        await pilot.press("q")
        await pilot.pause()
    assert app.return_code == 0


async def test_basic_fixture_directory_also_boots():
    """The transport work package's own fixture set loads through the same path."""
    basic = FIXTURES.parent / "basic"
    app, _ = _app(basic)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        assert app.active_table.row_count >= 1
        assert "CONNECTED" in app.status_bar.text
