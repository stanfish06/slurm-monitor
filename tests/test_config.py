from pathlib import Path

import pytest

from slurm_monitor.cli import overrides_from, parse_args
from slurm_monitor.config import Config, ConfigError, Overrides, load_config, resolve


def test_missing_file_yields_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.intervals.active_seconds == 2
    assert cfg.intervals.history_seconds == 60
    assert cfg.intervals.usage_seconds == 30
    assert cfg.history.window_days == 7
    assert cfg.clusters == {}
    with pytest.raises(ConfigError, match="no cluster selected"):
        cfg.cluster(None)


def test_multi_cluster_file(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text(
        """
default_cluster = "a"
[intervals]
active_seconds = 5
[history]
window_days = 3
[clusters.a]
host = "me@a.example"
ssh_options = ["-o", "ServerAliveInterval=30"]
[clusters.b]
host = "me@b.example"
agent_dir = "/scratch/me/agent"
python = "3.11"
"""
    )
    cfg = load_config(p)
    assert set(cfg.clusters) == {"a", "b"}
    assert cfg.cluster(None) == ("a", cfg.clusters["a"])
    assert cfg.cluster("b")[1].agent_dir == "/scratch/me/agent"
    assert cfg.clusters["a"].ssh_options == ["-o", "ServerAliveInterval=30"]
    assert cfg.intervals.active_seconds == 5
    assert cfg.intervals.history_seconds == 60
    assert cfg.history.window_days == 3
    with pytest.raises(ConfigError, match="unknown cluster 'zzz'"):
        cfg.cluster("zzz")


def test_sole_cluster_is_default(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text('[clusters.only]\nhost = "x@y"\n')
    assert load_config(p).cluster(None)[0] == "only"


def test_bad_key_is_reported_with_location(tmp_path: Path):
    p = tmp_path / "config.toml"
    p.write_text("[intervals]\nactive_seconds = -1\n")
    with pytest.raises(ConfigError, match="intervals.active_seconds"):
        load_config(p)


def test_cli_overrides_win_over_file():
    cfg = Config.model_validate(
        {
            "default_cluster": "a",
            "intervals": {"active_seconds": 5, "history_seconds": 120},
            "history": {"window_days": 3},
            "clusters": {"a": {"host": "me@a", "agent_dir": "~/x"}, "b": {"host": "me@b"}},
        }
    )
    ns = parse_args(
        [
            "--cluster",
            "b",
            "--host",
            "other@b",
            "--active-interval",
            "1",
            "--window-days",
            "30",
            "--agent-dir",
            "/tmp/agent",
        ]
    )
    r = resolve(cfg, overrides_from(ns))
    assert r.cluster_name == "b"
    assert r.cluster.host == "other@b"
    assert r.cluster.agent_dir == "/tmp/agent"
    assert r.intervals.active_seconds == 1
    assert r.intervals.history_seconds == 120  # untouched file value survives
    assert r.history.window_days == 30


def test_host_alone_needs_no_file():
    r = resolve(Config(), Overrides(host="me@somewhere"))
    assert r.cluster_name == "me@somewhere"
    assert r.cluster.host == "me@somewhere"


def test_local_forces_in_process():
    cfg = Config.model_validate({"clusters": {"a": {"host": "me@a"}}})
    r = resolve(cfg, Overrides(cluster="a", local=True))
    assert r.cluster.host is None


def test_default_subcommand_is_tui():
    assert parse_args([]).command == "tui"
    assert parse_args(["--cluster", "a"]).command == "tui"
    assert parse_args(["--config", "/x/y.toml", "--cluster", "a"]).command == "tui"
    assert parse_args(["bootstrap", "--cluster", "a"]).command == "bootstrap"
    assert parse_args(["agent"]).command == "agent"
