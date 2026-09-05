"""Tasks 11.1, 11.2, 11.4: connection state display, stale data with age, degraded badge."""

import pytest

from slurm_monitor.models import ConnectionState
from slurm_monitor.tui.widgets.status import DEGRADED_BADGE
from tests.tui_support import FakeClock, handshake, make_app, standard_session


@pytest.mark.parametrize(
    ("state", "error", "label"),
    [
        (ConnectionState.CONNECTING, None, "CONNECTING..."),
        (ConnectionState.CONNECTED, None, "CONNECTED"),
        (ConnectionState.RECONNECTING, "pipe closed", "RECONNECTING"),
        (ConnectionState.DISCONNECTED, "ssh exited 255", "DISCONNECTED: ssh exited 255"),
        (
            ConnectionState.AUTH_REQUIRED,
            "authenticate in a separate terminal",
            "AUTH REQUIRED: authenticate in a separate terminal",
        ),
    ],
)
async def test_each_connection_state_renders_distinctly(state, error, label):
    session = standard_session()
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        session.set_state(state, error)
        await pilot.pause()
        bar = app.status_bar
        assert label in bar.text
        assert "testcluster" in bar.text
        assert bar.has_class(state.value)
        others = {s.value for s in ConnectionState} - {state.value}
        assert not any(bar.has_class(o) for o in others)
        if state == ConnectionState.AUTH_REQUIRED:
            assert app.last_message == error


async def test_degraded_badge_is_shown_while_data_still_served():
    session = standard_session(hs=handshake(extension_available=False))
    app = make_app(session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        assert DEGRADED_BADGE in app.status_bar.text
        assert app.active_table.row_count == 4
        # Badge persists across refreshes.
        await app.refresh_active()
        assert DEGRADED_BADGE in app.status_bar.text


async def test_no_degraded_badge_with_extension():
    app = make_app(standard_session())
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        assert DEGRADED_BADGE not in app.status_bar.text


async def test_disconnect_keeps_rows_dims_them_and_age_advances():
    clock = FakeClock()
    session = standard_session()
    app = make_app(session, clock=clock)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.ready.wait()
        await pilot.pause()
        assert app.active_table.row_count == 4
        assert not app.active_table.has_class("stale")

        session.set_state(ConnectionState.RECONNECTING, "pipe closed")
        await app.refresh_active()  # raises TransportError inside the session
        await pilot.pause()
        assert app.active_table.row_count == 4
        assert app.history_table.row_count == 3
        assert app.active_table.has_class("stale")
        assert app.history_table.has_class("stale")
        assert "active refresh failed" in app.last_message

        clock.advance(5)
        app.tick_age()
        assert "data from 5s ago" in app.status_bar.text
        clock.advance(1)
        app.tick_age()
        assert "data from 6s ago" in app.status_bar.text

        # Reconnection refreshes immediately and clears the stale marking.
        session.set_state(ConnectionState.CONNECTED)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not app.active_table.has_class("stale")
        assert "data from" not in app.status_bar.text
        assert app.active_table.row_count == 4
