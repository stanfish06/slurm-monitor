"""SSH transport: one `ssh` subprocess per session running the agent on the login node.

Using the ssh binary inherits the user's config, ControlMaster multiplexing, and any existing
authentication state. BatchMode=yes turns an interactive auth prompt into an immediate failure,
which classify_exit maps to AuthError so the TUI never hangs on a hidden password prompt.
"""

from __future__ import annotations

from slurm_monitor.config import Resolved
from slurm_monitor.transport.subprocess import SubprocessTransport


def agent_python(agent_dir: str) -> str:
    """Interpreter inside the bootstrapped venv. `~` stays literal for the remote shell."""
    return f"{agent_dir.rstrip('/')}/venv/bin/python"


def ssh_argv(settings: Resolved) -> list[str]:
    cluster = settings.cluster
    if cluster.host is None:
        raise ValueError("ssh transport needs a host")
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        *cluster.ssh_options,
        cluster.host,
        f"{agent_python(cluster.agent_dir)} -m slurm_monitor.agent",
    ]


class SshTransport(SubprocessTransport):
    def __init__(self, settings: Resolved, **kwargs: object) -> None:
        self.host = settings.cluster.host
        super().__init__(
            ssh_argv(settings),
            agent_path=agent_python(settings.cluster.agent_dir),
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def opens_network_connection(self) -> bool:
        return True
