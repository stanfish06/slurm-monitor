"""Environment facts for the handshake: versions, user, host, extension availability.

pyslurm is imported only inside the functions that need it, so this module imports on any host
and probe_extension() / parse_sampling_interval() are unit-testable without Slurm.
"""

from __future__ import annotations

import os
import pwd
import socket
from typing import Any

from slurm_monitor import PROTOCOL_VERSION, __version__
from slurm_monitor.models import Handshake

EXTENSION_MODULE = "slurm_monitor_ext"
DEFAULT_SAMPLING_INTERVAL = 30


def probe_extension() -> tuple[bool, str | None]:
    """Try to import the slurm_load_job_user() binding. Returns (ok, error text)."""
    try:
        __import__(EXTENSION_MODULE)
    except Exception as e:  # ImportError, or a libslurm load failure surfacing as OSError
        return False, f"{type(e).__name__}: {e}"
    return True, None


def parse_sampling_interval(freq: Any) -> int:
    """JobAcctGatherFrequency as pyslurm gives it ({"task": 30}) or raw ("task=30") -> seconds."""
    if isinstance(freq, dict):
        value = freq.get("task")
    elif isinstance(freq, str):
        value = None
        for part in freq.split(","):
            key, sep, val = part.partition("=")
            if sep and key.strip() == "task":
                value = val.strip()
    else:
        value = None
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_SAMPLING_INTERVAL
    return parsed if parsed > 0 else DEFAULT_SAMPLING_INTERVAL


def built_slurm_version() -> str:
    """Slurm version pyslurm was compiled against, e.g. "25.11.5"."""
    import pyslurm

    return ".".join(str(p) for p in pyslurm.slurm_api_version())


def gather_handshake(extension_ok: bool, extension_error: str | None) -> Handshake:
    """Query slurmctld's config once for the running version and sampling interval."""
    import pyslurm

    conf = pyslurm.slurmctld.Config.load()
    return Handshake(
        protocol_version=PROTOCOL_VERSION,
        agent_version=__version__,
        user=pwd.getpwuid(os.getuid()).pw_name,
        hostname=socket.gethostname(),
        pyslurm_version=pyslurm.__version__,
        slurm_built_version=built_slurm_version(),
        slurm_running_version=conf.version or "unknown",
        extension_available=extension_ok,
        extension_error=extension_error,
        sampling_interval_seconds=parse_sampling_interval(conf.job_accounting_gather_frequency),
    )
