"""Keyboard-driven modal screens: cancel confirmation and key help."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class ConfirmCancel(ModalScreen[bool]):
    """Yes/no prompt. Dismisses with True on y/Enter and False on n/Escape."""

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("enter", "confirm", "Yes", show=False),
        Binding("n", "abandon", "No"),
        Binding("escape", "abandon", "No", show=False),
    ]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Static(self.prompt, id="confirm-prompt", markup=False)
            yield Static("y / Enter: cancel it      n / Esc: keep it", id="confirm-hint")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_abandon(self) -> None:
        self.dismiss(False)


HELP_TEXT = """\
a / h / Tab    switch between Active and History
Up / Down      move the cursor; the side panels follow it
Enter / Space  expand or collapse an array row
r              refresh the current tab now
c              cancel the selected job, task or array (asks first)
d              show or hide the detail/usage panels
?              this help
q              quit

Active refreshes every few seconds, History once a minute plus whenever a job
leaves Active. Scroll onto the oldest History row to load an older window.
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close", show=False),
        Binding("question_mark", "close", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            yield Static("keys", id="help-title")
            yield Static(HELP_TEXT, markup=False)

    def action_close(self) -> None:
        self.dismiss(None)
