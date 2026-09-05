"""Cell and panel text formatting. Pure functions so tests can hand-check the strings."""

from __future__ import annotations

from datetime import datetime

from slurm_monitor.models import JobSummary, Measure

NO_SAMPLES = "no samples yet"
UNAVAILABLE = "unavailable"


def format_elapsed(seconds: int | float | None) -> str:
    """H:MM:SS, or D-HH:MM:SS once a day has passed (Slurm's own notation)."""
    if seconds is None:
        return "-"
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_minutes(minutes: int | None) -> str:
    return "-" if minutes is None else format_elapsed(minutes * 60)


def format_local(ts: datetime | None) -> str:
    """Local wall-clock time; naive datetimes are taken as already local."""
    if ts is None:
        return "-"
    if ts.tzinfo is not None:
        ts = ts.astimezone()
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def format_state(job: JobSummary) -> str:
    """State cell: `CANCELLED by <user>` when accounting names who cancelled it."""
    if job.cancelled_by and job.state.split()[0].upper() == "CANCELLED":
        return f"CANCELLED by {job.cancelled_by}"
    return job.state


def format_exit(code: int | None, signal: int | None) -> str:
    """`code`, or `code:signal` when the job died from a signal."""
    if code is None:
        return "-"
    if signal:
        return f"{code}:{signal}"
    return str(code)


def format_memory_mb(mb: float | None) -> str:
    if mb is None:
        return "-"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _format_quantity(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    if unit == "MB":
        return format_memory_mb(value)
    if unit == "seconds":
        return format_elapsed(value)
    if unit == "cpu-seconds":
        return f"{value:.0f} cpu-s"
    if unit == "percent":
        return f"{value:.1f}%"
    return f"{value:g} {unit}"


def format_measure(m: Measure | None, sampled: bool = True) -> str:
    """`consumed / requested (NN.N%)`; `no samples yet` whenever the consumed side is unknown."""
    if m is None:
        return UNAVAILABLE
    if not sampled or m.consumed is None:
        requested = _format_quantity(m.requested, m.unit)
        return f"{NO_SAMPLES} / {requested}" if m.requested is not None else NO_SAMPLES
    if m.unit == "percent":
        return f"{m.consumed:.1f}%"
    consumed = _format_quantity(m.consumed, m.unit)
    requested = _format_quantity(m.requested, m.unit)
    pct = m.percent
    if pct is None:
        return f"{consumed} / {requested}"
    return f"{consumed} / {requested} ({pct:.1f}%)"


def format_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 5400:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"
