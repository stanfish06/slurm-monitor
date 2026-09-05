"""Cluster-side agent: NDJSON requests on stdin, one response per line on stdout, logs on stderr.

    python -m slurm_monitor.agent [--record DIR]   # serve until EOF on stdin
    python -m slurm_monitor.agent --check          # print the handshake as JSON; exit 1 on mismatch

Startup sends the unsolicited handshake frame {"id": 0, "result": <Handshake>} before reading any
request. If pyslurm's compiled Slurm version does not match the running slurmctld, the handshake is
still sent (so the client can show both versions) and every request is refused with
ERR_VERSION_MISMATCH; pyslurm queries are never called in that state.

Nothing but frames may reach stdout: sys.stdout is rebound to stderr while serving so a stray
print() inside pyslurm or the core lands in the log stream, and logging is configured on stderr.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import IO

from slurm_monitor.core import CoreAPI, CoreError, dispatch
from slurm_monitor.models import Handshake, versions_compatible
from slurm_monitor.protocol import (
    ERR_BAD_REQUEST,
    ERR_INTERNAL,
    ERR_VERSION_MISMATCH,
    ProtocolError,
    Response,
    decode_request,
    encode,
    error_response,
    result_response,
)

log = logging.getLogger("slurm_monitor.agent")

BOOTSTRAP_HINT = "re-run `slurm-monitor bootstrap --force` to rebuild the agent"


def mismatch_message(handshake: Handshake) -> str | None:
    """None when the agent may serve, else the refusal text naming bootstrap as the remedy."""
    if versions_compatible(handshake.slurm_built_version, handshake.slurm_running_version):
        return None
    return (
        f"agent built against Slurm {handshake.slurm_built_version} but slurmctld runs "
        f"{handshake.slurm_running_version}; {BOOTSTRAP_HINT}"
    )


def build_core(record_dir: Path | None) -> CoreAPI:
    """PyslurmCore, wrapped in RecordingCore when --record is given and the recorder exists."""
    from slurm_monitor.core.queries import PyslurmCore

    core: CoreAPI = PyslurmCore()
    if record_dir is not None:
        try:
            from slurm_monitor.core.recording import RecordingCore
        except ImportError as e:
            log.warning("--record ignored, recorder unavailable: %s", e)
        else:
            core = RecordingCore(core, record_dir)
            log.info("recording responses to %s", record_dir)
    return core


def _write(stdout: IO[str], frame: Response) -> None:
    stdout.write(encode(frame).decode("utf-8"))
    stdout.flush()


def serve(core: CoreAPI, stdin: IO[str], stdout: IO[str]) -> int:
    """Handshake, then the request loop until EOF. Returns the process exit code."""
    try:
        handshake = core.hello()
    except Exception as e:
        log.exception("handshake failed")
        _write(stdout, error_response(0, ERR_INTERNAL, f"handshake failed: {e}"))
        return 1
    _write(stdout, Response(id=0, result=handshake.model_dump(mode="json")))

    refusal = mismatch_message(handshake)
    if refusal is not None:
        log.error("refusing all requests: %s", refusal)
    mismatch_data = {
        "slurm_built_version": handshake.slurm_built_version,
        "slurm_running_version": handshake.slurm_running_version,
    }

    while True:
        line = stdin.readline()
        if line == "":
            log.info("stdin closed, exiting")
            return 0
        if not line.strip():
            continue
        try:
            request = decode_request(line)
        except ProtocolError as e:
            log.warning("bad request line: %s", e)
            _write(stdout, error_response(-1, ERR_BAD_REQUEST, str(e)))
            continue

        if refusal is not None:
            _write(stdout, error_response(request.id, ERR_VERSION_MISMATCH, refusal, mismatch_data))
            continue

        try:
            result = dispatch(core, request.method, request.params)
            response = result_response(request.id, result)
        except CoreError as e:
            response = error_response(request.id, e.code, e.message, e.data or None)
        except Exception as e:
            log.exception("%s failed", request.method)
            response = error_response(request.id, ERR_INTERNAL, f"{type(e).__name__}: {e}")
        _write(stdout, response)


def run_agent(
    record_dir: Path | None = None,
    core: CoreAPI | None = None,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Entry point used by `slurm-monitor agent` and `python -m slurm_monitor.agent`.

    core/stdin/stdout/stderr are injectable for tests; defaults are PyslurmCore and the process
    streams. Exit 0 on EOF, 1 when the core cannot be built or the handshake fails.
    """
    stdin = sys.stdin if stdin is None else stdin
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    handler = logging.StreamHandler(err)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)

    # Keep the frame stream clean: anything printing to sys.stdout goes to the log stream instead.
    saved_stdout = sys.stdout
    sys.stdout = err
    try:
        if core is None:
            try:
                core = build_core(record_dir)
            except Exception as e:
                log.exception("cannot start agent core")
                _write(out, error_response(0, ERR_INTERNAL, f"cannot start agent core: {e}"))
                return 1
        try:
            return serve(core, stdin, out)
        except BrokenPipeError:
            log.info("stdout closed by client, exiting")
            return 0
    finally:
        sys.stdout = saved_stdout
        root.removeHandler(handler)


def check(core: CoreAPI | None = None, stdout: IO[str] | None = None) -> int:
    """`--check`: print the handshake as one JSON line; exit 1 on a Slurm version mismatch.

    Without a core this gathers the facts directly from pyslurm (no query layer needed), which is
    what bootstrap runs to verify a fresh environment."""
    out = sys.stdout if stdout is None else stdout
    if core is None:
        from slurm_monitor.core.environment import gather_handshake, probe_extension

        handshake = gather_handshake(*probe_extension())
    else:
        handshake = core.hello()
    out.write(handshake.model_dump_json() + "\n")
    out.flush()
    refusal = mismatch_message(handshake)
    if refusal is not None:
        print(f"slurm-monitor agent: {refusal}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m slurm_monitor.agent", description=__doc__)
    parser.add_argument("--record", type=Path, metavar="DIR", help="write responses as fixtures")
    parser.add_argument("--check", action="store_true", help="print handshake JSON and exit")
    ns = parser.parse_args(argv)
    if ns.check:
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
        return check()
    return run_agent(record_dir=ns.record)


if __name__ == "__main__":
    sys.exit(main())
