"""Command-line entry point.

    slurm-monitor [tui] [--cluster NAME] [--host DEST] [--local] [interval overrides]
    slurm-monitor bootstrap [--cluster NAME] [--host DEST]
    slurm-monitor agent [--record DIR]          # runs on the cluster; speaks NDJSON on stdio
    slurm-monitor check [--cluster NAME]        # handshake only, print versions and exit

Heavy imports (textual, pyslurm) happen inside the subcommand functions so that `agent` starts
without textual and `tui` starts without pyslurm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slurm_monitor import __version__
from slurm_monitor.config import ConfigError, Overrides, load_config, resolve

SUBCOMMANDS = ("tui", "bootstrap", "agent", "check")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slurm-monitor", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config", type=Path, help="config file (default: ~/.config/slurm-monitor/config.toml)"
    )
    sub = parser.add_subparsers(dest="command")

    def add_connection_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--cluster", help="named cluster from the config file")
        p.add_argument("--host", help="ssh destination, overrides the cluster's host")
        p.add_argument("--local", action="store_true", help="run in-process on this host (no ssh)")
        p.add_argument("--agent-dir", help="remote directory for the agent environment")

    def add_interval_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--active-interval", type=float, metavar="SEC", dest="active_seconds")
        p.add_argument("--history-interval", type=float, metavar="SEC", dest="history_seconds")
        p.add_argument("--usage-interval", type=float, metavar="SEC", dest="usage_seconds")
        p.add_argument("--window-days", type=int, metavar="DAYS")

    tui = sub.add_parser("tui", help="open the interface (default)")
    add_connection_args(tui)
    add_interval_args(tui)
    tui.add_argument(
        "--fixtures", type=Path, metavar="DIR", help="replay recorded fixtures instead of a cluster"
    )

    boot = sub.add_parser("bootstrap", help="build the agent environment on the cluster")
    add_connection_args(boot)
    boot.add_argument("--force", action="store_true", help="rebuild even if an environment exists")

    agent = sub.add_parser("agent", help="run the cluster-side agent on stdin/stdout")
    agent.add_argument(
        "--record", type=Path, metavar="DIR", help="also write every response to DIR as fixtures"
    )

    check = sub.add_parser("check", help="connect, print handshake, exit")
    add_connection_args(check)
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    # Default subcommand is tui: insert it unless the first non-option token is a subcommand.
    if (
        not any(tok in SUBCOMMANDS for tok in argv)
        and "--version" not in argv
        and "-h" not in argv
        and "--help" not in argv
    ):
        # Global options (--config PATH) may precede the implied subcommand.
        insert_at = 0
        i = 0
        while i < len(argv):
            if argv[i] == "--config":
                i += 2
                insert_at = i
                continue
            break
        argv = [*argv[:insert_at], "tui", *argv[insert_at:]]
    return build_parser().parse_args(argv)


def overrides_from(ns: argparse.Namespace) -> Overrides:
    return Overrides(
        cluster=getattr(ns, "cluster", None),
        host=getattr(ns, "host", None),
        local=bool(getattr(ns, "local", False)),
        agent_dir=getattr(ns, "agent_dir", None),
        active_seconds=getattr(ns, "active_seconds", None),
        history_seconds=getattr(ns, "history_seconds", None),
        usage_seconds=getattr(ns, "usage_seconds", None),
        window_days=getattr(ns, "window_days", None),
    )


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if ns.command == "agent":
            from slurm_monitor.agent import run_agent

            return run_agent(record_dir=ns.record)

        config = load_config(ns.config)
        if ns.command == "tui" and ns.fixtures is not None:
            from slurm_monitor.tui.app import run_tui_with_fixtures

            return run_tui_with_fixtures(ns.fixtures, config, overrides_from(ns))

        settings = resolve(config, overrides_from(ns))
        if ns.command == "bootstrap":
            from slurm_monitor.bootstrap import run_bootstrap

            return run_bootstrap(settings, force=ns.force)
        if ns.command == "check":
            from slurm_monitor.session import run_check

            return run_check(settings)
        from slurm_monitor.tui.app import run_tui

        return run_tui(settings)
    except ConfigError as e:
        print(f"slurm-monitor: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
