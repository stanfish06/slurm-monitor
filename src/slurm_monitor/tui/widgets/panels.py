"""Detail and usage panels for the selected job. Each keeps `.text`, the plain string it shows,
so tests assert on content rather than on rendering."""

from __future__ import annotations

from textual.widgets import Static

from slurm_monitor.models import JobDetail, JobUsage
from slurm_monitor.tui.format import (
    NO_SAMPLES,
    UNAVAILABLE,
    format_local,
    format_measure,
    format_memory_mb,
    format_minutes,
)


class _TextPanel(Static):
    def __init__(self, title: str, **kwargs) -> None:
        super().__init__("", markup=False, **kwargs)
        self.border_title = title
        self.text = ""

    def set_text(self, text: str) -> None:
        self.text = text
        self.update(text)


def _kv(pairs: list[tuple[str, object]]) -> str:
    width = max(len(k) for k, _ in pairs)
    return "\n".join(f"{k:<{width}}  {'-' if v in (None, '') else v}" for k, v in pairs)


class DetailPanel(_TextPanel):
    def __init__(self, **kwargs) -> None:
        super().__init__("detail", **kwargs)

    def show_empty(self, note: str = "no job selected") -> None:
        self.set_text(note)

    def show(self, d: JobDetail) -> None:
        req = d.requested
        resources = ", ".join(
            part
            for part in (
                f"{req.cpus} cpus" if req.cpus is not None else None,
                f"{req.nodes} nodes" if req.nodes is not None else None,
                format_memory_mb(req.memory_mb) if req.memory_mb is not None else None,
                f"{req.gpus} gpus" if req.gpus else None,
                req.gres,
            )
            if part
        )
        limit = d.time_limit_minutes if d.time_limit_minutes is not None else req.time_limit_minutes
        self.set_text(
            _kv(
                [
                    ("job", d.display_id),
                    ("name", d.name),
                    ("state", d.state),
                    ("account", d.account),
                    ("qos", d.qos),
                    ("partition", d.partition),
                    ("nodes", d.nodelist),
                    ("requested", resources),
                    ("time limit", format_minutes(limit)),
                    ("submitted", format_local(d.submit_time)),
                    ("started", format_local(d.start_time)),
                    ("workdir", d.working_directory),
                    ("stdout", d.stdout_path),
                    ("stderr", d.stderr_path),
                ]
            )
        )


class UsagePanel(_TextPanel):
    def __init__(self, **kwargs) -> None:
        super().__init__("usage", **kwargs)

    def show_empty(self, note: str = "no job selected") -> None:
        self.set_text(note)

    def show(self, u: JobUsage) -> None:
        pairs: list[tuple[str, object]] = [
            ("cpu", format_measure(u.cpu, u.sampled)),
            ("memory", format_measure(u.memory, u.sampled)),
            ("wall time", format_measure(u.walltime, u.sampled)),
        ]
        if u.gpus_allocated > 0:
            if u.gpu_accounting_available is False:
                pairs.append(("gpu util", UNAVAILABLE))
                pairs.append(("gpu memory", UNAVAILABLE))
            else:
                pairs.append(("gpu util", format_measure(u.gpu_utilization, u.sampled)))
                pairs.append(("gpu memory", format_measure(u.gpu_memory, u.sampled)))
        if not u.sampled:
            pairs.append(("samples", NO_SAMPLES))
        elif u.sampled_at is not None:
            pairs.append(("sampled", format_local(u.sampled_at)))
        self.set_text(_kv(pairs))
