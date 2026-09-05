"""Top status bar (cluster, connection state, degraded badge, data age) and bottom message line."""

from __future__ import annotations

from textual.widgets import Static

from slurm_monitor.models import ConnectionState

DEGRADED_BADGE = "DEGRADED: unfiltered job query"

STATE_LABELS = {
    ConnectionState.CONNECTING: "CONNECTING...",
    ConnectionState.CONNECTED: "CONNECTED",
    ConnectionState.RECONNECTING: "RECONNECTING",
    ConnectionState.DISCONNECTED: "DISCONNECTED",
    ConnectionState.AUTH_REQUIRED: "AUTH REQUIRED",
}

# States in which the tables show data that is no longer being refreshed.
STALE_STATES = frozenset(
    {ConnectionState.RECONNECTING, ConnectionState.DISCONNECTED, ConnectionState.AUTH_REQUIRED}
)


class StatusBar(Static):
    def __init__(self, cluster: str, **kwargs) -> None:
        super().__init__("", markup=False, **kwargs)
        self.cluster = cluster
        self.text = ""

    def render_state(
        self,
        state: ConnectionState,
        error: str | None,
        degraded: bool,
        age_seconds: float | None,
    ) -> None:
        parts = [f"slurm-monitor  {self.cluster}", STATE_LABELS[state]]
        if error and state in (ConnectionState.DISCONNECTED, ConnectionState.AUTH_REQUIRED):
            parts[-1] += f": {error}"
        if degraded:
            parts.append(DEGRADED_BADGE)
        if age_seconds is not None and state in STALE_STATES:
            from slurm_monitor.tui.format import format_age

            parts.append(f"data from {format_age(age_seconds)} ago")
        self.text = "  |  ".join(parts)
        for s in ConnectionState:
            self.set_class(s == state, s.value)
        self.update(self.text)


class MessageLine(Static):
    def __init__(self, **kwargs) -> None:
        super().__init__("", markup=False, **kwargs)
        self.text = ""

    def show(self, text: str) -> None:
        self.text = text
        self.update(text)
