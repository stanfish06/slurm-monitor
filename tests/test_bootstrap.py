import io
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from slurm_monitor import bootstrap
from slurm_monitor.config import ClusterConfig, Config, HistoryConfig, Intervals, Resolved


def make_settings(host: str | None, agent_dir: str, **cluster) -> Resolved:
    return Resolved(
        cluster_name="t",
        cluster=ClusterConfig(host=host, agent_dir=agent_dir, **cluster),
        intervals=Intervals(),
        history=HistoryConfig(),
    )


def test_agent_requirements_pin_pyslurm():
    reqs = bootstrap.agent_requirements()
    assert any(r.startswith("pyslurm==") for r in reqs)
    assert any(r.startswith("pydantic") for r in reqs)
    assert any(r.lower().startswith("cython") for r in reqs)
    assert any(r.startswith("setuptools") for r in reqs)


def test_package_tarball_ships_sources_and_ext_but_no_caches(tmp_path):
    tarball = bootstrap.package_tarball()
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tar:
        names = tar.getnames()
    assert "slurm_monitor/agent.py" in names
    assert "slurm_monitor/ext/slurm_monitor_ext.pyx" in names
    assert "slurm_monitor/ext/setup.py" in names
    assert "slurm_monitor/ext/__init__.py" in names
    assert all("__pycache__" not in n and not n.endswith((".pyc", ".so", ".c")) for n in names)
    assert all(n.startswith("slurm_monitor/") for n in names)


def test_remote_agent_command_keeps_tilde_literal():
    settings = make_settings("me@login.example", "~/.slurm-monitor", ssh_options=["-o", "X=1"])
    assert bootstrap.remote_agent_command(settings) == [
        "ssh",
        "-o",
        "X=1",
        "me@login.example",
        "~/.slurm-monitor/venv/bin/python -m slurm_monitor.agent",
    ]
    assert bootstrap.bootstrap_command(settings) == [
        "ssh",
        "-o",
        "X=1",
        "me@login.example",
        "bash -s",
    ]
    assert "-t" not in bootstrap.bootstrap_command(settings)
    with pytest.raises(ValueError):
        bootstrap.remote_agent_command(make_settings(None, "~/.slurm-monitor"))
    assert bootstrap.bootstrap_command(make_settings(None, "/x")) == ["bash", "-s"]


def test_local_agent_available(tmp_path):
    assert not bootstrap.local_agent_available(str(tmp_path / "agent"))
    (tmp_path / "agent" / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "agent" / "venv" / "bin" / "python").touch()
    assert bootstrap.local_agent_available(str(tmp_path / "agent"))


def test_render_script_embeds_settings_and_env():
    settings = make_settings("h", "~/.slurm-monitor", python="3.11")
    script = bootstrap.render_script(settings, force=True, tarball=b"x")
    assert (
        script.startswith("#!/usr/bin/env bash\nset -euo pipefail") or "set -euo pipefail" in script
    )
    assert "AGENT_DIR='~/.slurm-monitor'\n" in script
    assert "FORCE=1\n" in script
    assert "PYTHON_VERSION=3.11\n" in script
    assert 'CC="${CC:-gcc}"' in script
    assert 'SLURM_INCLUDE_DIR="${SLURM_INCLUDE_DIR:-/usr/include}"' in script
    assert 'SLURM_LIB_DIR="${SLURM_LIB_DIR:-/usr/lib64}"' in script
    assert "pyslurm==" in script
    assert "setup.py build_ext --build-lib" in script
    assert "slurm_monitor.agent --check" in script
    assert "trap cleanup EXIT" in script
    assert "eA==" in script  # base64 of b"x"
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    """A `uv` on PATH that fails loudly, standing in for a broken remote build."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\necho "fake uv: gcc: error: libslurmfull.so not found" >&2\nexit 1\n')
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return uv


def test_failed_build_leaves_no_partial_environment(tmp_path, fake_uv, capfd):
    agent_dir = tmp_path / "agent"
    settings = make_settings(None, str(agent_dir))
    rc = bootstrap.run_bootstrap(settings, force=False)
    captured = capfd.readouterr()
    assert rc == 1
    assert "fake uv: gcc: error: libslurmfull.so not found" in captured.err
    assert "bootstrap: FAILED" in captured.err
    assert "failed on localhost" in captured.err
    assert not agent_dir.exists()
    assert not Path(str(agent_dir) + ".tmp").exists()


def test_failed_build_streams_stderr_and_returns_tail(tmp_path, fake_uv):
    agent_dir = tmp_path / "agent"
    settings = make_settings(None, str(agent_dir))
    out, err = io.StringIO(), io.StringIO()
    rc, tail = bootstrap.run_script(
        bootstrap.bootstrap_command(settings),
        bootstrap.render_script(settings, force=False, tarball=b""),
        stdout=out,
        stderr=err,
    )
    assert rc != 0
    assert any("fake uv" in line for line in tail)
    assert "fake uv" in err.getvalue()
    assert not agent_dir.exists() and not Path(str(agent_dir) + ".tmp").exists()


def test_existing_environment_short_circuits_without_force(tmp_path, fake_uv, capfd):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "manifest.json").write_text(json.dumps({"pyslurm_version": "25.11.2"}))
    settings = make_settings(None, str(agent_dir))
    assert bootstrap.run_bootstrap(settings, force=False) == 0
    captured = capfd.readouterr()
    assert "already bootstrapped" in captured.err
    assert '"pyslurm_version": "25.11.2"' in captured.out
    assert agent_dir.exists()
    # --force goes through the (failing) build and must leave the existing dir untouched.
    assert bootstrap.run_bootstrap(settings, force=True) == 1
    assert (agent_dir / "manifest.json").exists()
    assert not Path(str(agent_dir) + ".tmp").exists()


def test_tilde_agent_dir_expands_in_script(tmp_path, fake_uv, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = make_settings(None, "~/agent-under-tilde")
    out, err = io.StringIO(), io.StringIO()
    rc, tail = bootstrap.run_script(
        ["bash", "-s"], bootstrap.render_script(settings, force=False, tarball=b""), out, err
    )
    assert rc != 0
    assert f"removing {tmp_path}/agent-under-tilde.tmp" in err.getvalue()
    assert not (tmp_path / "agent-under-tilde.tmp").exists()


def test_tui_and_check_paths_never_import_bootstrap(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[clusters.a]\nhost = "user@example.invalid"\n')
    script = f"""
import sys
from slurm_monitor import cli
from slurm_monitor.config import load_config, resolve
for argv in (["--cluster", "a"], ["check", "--cluster", "a"], ["tui", "--host", "x@y"]):
    ns = cli.parse_args(["--config", {str(cfg)!r}, *argv])
    resolve(load_config(ns.config), cli.overrides_from(ns))
try:
    cli.main(["--config", {str(cfg)!r}, "check", "--cluster", "a"])
except Exception as e:  # session/transport may be unimplemented or unreachable here
    print("check raised", type(e).__name__, file=sys.stderr)
assert "slurm_monitor.bootstrap" not in sys.modules, "TUI/check path imported bootstrap"
assert "slurm_monitor.agent" not in sys.modules, "TUI/check path imported the agent"
print("clean")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "clean"


def test_bootstrap_only_runs_from_explicit_subcommand():
    from slurm_monitor import cli

    assert cli.parse_args(["bootstrap", "--host", "x@y"]).command == "bootstrap"
    assert cli.parse_args(["--host", "x@y"]).command == "tui"
    assert Config().clusters == {}
