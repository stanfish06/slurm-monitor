"""Widgets composing the app: tables, side panels, status/message bars, modal screens."""

from slurm_monitor.tui.widgets.modals import ConfirmCancel, HelpScreen
from slurm_monitor.tui.widgets.panels import DetailPanel, UsagePanel
from slurm_monitor.tui.widgets.status import MessageLine, StatusBar
from slurm_monitor.tui.widgets.tables import ActiveTable, HistoryTable

__all__ = [
    "ActiveTable",
    "ConfirmCancel",
    "DetailPanel",
    "HelpScreen",
    "HistoryTable",
    "MessageLine",
    "StatusBar",
    "UsagePanel",
]
