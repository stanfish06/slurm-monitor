"""User configuration: TOML file plus command-line overrides.

File: $XDG_CONFIG_HOME/slurm-monitor/config.toml, default ~/.config/slurm-monitor/config.toml.

    default_cluster = "greatlakes"

    [intervals]
    active_seconds = 2
    history_seconds = 60
    usage_seconds = 30

    [history]
    window_days = 7

    [clusters.greatlakes]
    host = "me@login.example.edu"    # ssh destination, as accepted by ssh(1)
    agent_dir = "~/.slurm-monitor"   # remote directory holding the agent venv
    python = "3.12"                  # python version uv installs for the agent
    ssh_options = ["-o", "ServerAliveInterval=30"]

Nothing here is hard-coded to a user, account, or host; defaults come from this module only.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_AGENT_DIR = "~/.slurm-monitor"
DEFAULT_AGENT_PYTHON = "3.12"


class ClusterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str | None = None  # None means "run in-process on this host"
    agent_dir: str = DEFAULT_AGENT_DIR
    python: str = DEFAULT_AGENT_PYTHON
    ssh_options: list[str] = Field(default_factory=list)


class Intervals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_seconds: float = Field(2.0, gt=0)
    history_seconds: float = Field(60.0, gt=0)
    usage_seconds: float = Field(30.0, gt=0)


class HistoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_days: int = Field(7, ge=1)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_cluster: str | None = None
    intervals: Intervals = Field(default_factory=Intervals)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    clusters: dict[str, ClusterConfig] = Field(default_factory=dict)

    def cluster(self, name: str | None) -> tuple[str, ClusterConfig]:
        """Resolve a cluster by name, falling back to default_cluster, then to a sole entry."""
        if name is None:
            name = self.default_cluster
        if name is None and len(self.clusters) == 1:
            name = next(iter(self.clusters))
        if name is None:
            raise ConfigError(
                "no cluster selected: pass --cluster NAME or set default_cluster in "
                f"{config_path()}"
            )
        if name not in self.clusters:
            known = ", ".join(sorted(self.clusters)) or "(none configured)"
            raise ConfigError(f"unknown cluster {name!r}; configured clusters: {known}")
        return name, self.clusters[name]


class ConfigError(Exception):
    pass


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "slurm-monitor" / "config.toml"


def load_config(path: Path | None = None) -> Config:
    """Load the TOML file; a missing file yields all defaults."""
    path = path or config_path()
    if not path.exists():
        return Config()
    try:
        data = tomllib.loads(path.read_text())
        return Config.model_validate(data)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        raise ConfigError(f"{path}: {loc}: {first['msg']}") from e


class Overrides(BaseModel):
    """Values from the command line. Any non-None field wins over the file."""

    model_config = ConfigDict(extra="forbid")

    cluster: str | None = None
    host: str | None = None
    local: bool = False  # force in-process mode
    agent_dir: str | None = None
    active_seconds: float | None = None
    history_seconds: float | None = None
    usage_seconds: float | None = None
    window_days: int | None = None


class Resolved(BaseModel):
    """Effective settings for one session."""

    model_config = ConfigDict(extra="forbid")

    cluster_name: str
    cluster: ClusterConfig
    intervals: Intervals
    history: HistoryConfig


def resolve(config: Config, overrides: Overrides) -> Resolved:
    """Apply command-line overrides on top of the file. --host with no --cluster creates an
    ad-hoc cluster entry so the file is not required at all."""
    intervals = config.intervals.model_copy(
        update={
            k: v
            for k, v in {
                "active_seconds": overrides.active_seconds,
                "history_seconds": overrides.history_seconds,
                "usage_seconds": overrides.usage_seconds,
            }.items()
            if v is not None
        }
    )
    history = config.history.model_copy(
        update={} if overrides.window_days is None else {"window_days": overrides.window_days}
    )

    if overrides.local:
        name = overrides.cluster or "local"
        base = config.clusters.get(name, ClusterConfig())
        cluster = base.model_copy(update={"host": None})
    elif overrides.cluster is None and overrides.host is not None:
        name = overrides.host
        cluster = ClusterConfig(host=overrides.host)
    else:
        name, cluster = config.cluster(overrides.cluster)

    updates: dict[str, Any] = {}
    if overrides.host is not None and not overrides.local:
        updates["host"] = overrides.host
    if overrides.agent_dir is not None:
        updates["agent_dir"] = overrides.agent_dir
    if updates:
        cluster = cluster.model_copy(update=updates)

    return Resolved(cluster_name=name, cluster=cluster, intervals=intervals, history=history)
