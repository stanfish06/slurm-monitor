import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from slurm_monitor import agent
from slurm_monitor.protocol import (
    ERR_BAD_REQUEST,
    ERR_INTERNAL,
    ERR_NOT_FOUND,
    ERR_VERSION_MISMATCH,
    decode_response,
)
from tests.fake_core import LOG_MARKER, PRINT_MARKER, FakeCore

REPO_ROOT = Path(__file__).resolve().parent.parent


def drive(core: FakeCore, lines: list[str]) -> tuple[int, list[dict], str]:
    stdin = io.StringIO("".join(line + "\n" for line in lines))
    stdout, stderr = io.StringIO(), io.StringIO()
    rc = agent.run_agent(core=core, stdin=stdin, stdout=stdout, stderr=stderr)
    frames = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return rc, frames, stderr.getvalue()


def test_handshake_is_first_frame_and_ids_correlate():
    core = FakeCore()
    rc, frames, err = drive(
        core,
        [
            '{"id": 7, "method": "active_jobs", "params": {}}',
            '{"id": 3, "method": "job_detail", "params": {"job_id": 1}}',
            '{"id": 9, "method": "cancel", "params": {"job_id": 5, "array_task_id": 2}}',
        ],
    )
    assert rc == 0
    assert frames[0]["id"] == 0
    assert frames[0]["result"]["slurm_running_version"] == "25.11.5"
    assert frames[0]["result"]["protocol_version"] == 1
    assert [f["id"] for f in frames[1:]] == [7, 3, 9]
    assert frames[1]["result"]["jobs"][0]["job_id"] == 42
    assert frames[2]["result"]["job_id"] == 1
    assert frames[3]["result"] == {
        "job_id": 5,
        "array_task_id": 2,
        "cancelled": True,
        "message": None,
    }
    for frame in frames:
        decode_response(json.dumps(frame))  # every line is a valid protocol frame


def test_logs_and_prints_go_to_stderr_only():
    _, frames, err = drive(FakeCore(), ['{"id": 1, "method": "active_jobs", "params": {}}'])
    stdout_text = json.dumps(frames)
    assert LOG_MARKER in err
    assert PRINT_MARKER in err
    assert LOG_MARKER not in stdout_text
    assert PRINT_MARKER not in stdout_text
    assert len(frames) == 2


def test_malformed_lines_yield_bad_request_and_loop_continues():
    rc, frames, _ = drive(
        FakeCore(),
        [
            "this is not json",
            '{"id": 1, "method": "active_jobs", "params": {}}',
            '{"id": "x", "method": "active_jobs"}',
            "",
            '["a", "list"]',
            '{"id": 2, "method": "no_such_method", "params": {}}',
            '{"id": 4, "method": "job_detail", "params": {"job_id": 1, "bogus": 1}}',
        ],
    )
    assert rc == 0
    assert frames[1]["id"] == -1 and frames[1]["error"]["code"] == ERR_BAD_REQUEST
    assert frames[2]["id"] == 1 and "result" in frames[2]
    assert frames[3]["id"] == -1 and frames[3]["error"]["code"] == ERR_BAD_REQUEST
    assert frames[4]["id"] == -1 and frames[4]["error"]["code"] == ERR_BAD_REQUEST
    assert frames[5]["id"] == 2 and frames[5]["error"]["code"] == ERR_BAD_REQUEST
    assert "no_such_method" in frames[5]["error"]["message"]
    assert frames[6]["id"] == 4 and frames[6]["error"]["code"] == ERR_BAD_REQUEST
    assert len(frames) == 7


def test_core_error_and_unexpected_exception_become_error_frames():
    rc, frames, err = drive(
        FakeCore(),
        [
            '{"id": 1, "method": "job_detail", "params": {"job_id": 999}}',
            '{"id": 2, "method": "job_usage", "params": {"job_id": 500}}',
            '{"id": 3, "method": "job_usage", "params": {"job_id": 1}}',
        ],
    )
    assert rc == 0
    assert frames[1]["error"]["code"] == ERR_NOT_FOUND
    assert frames[1]["error"]["data"] == {"job_id": 999}
    assert frames[2]["error"]["code"] == ERR_INTERNAL
    assert "boom" in frames[2]["error"]["message"]
    assert "boom" in err  # traceback logged on stderr
    assert frames[3]["result"]["job_id"] == 1  # loop survived the exception


def test_version_mismatch_sends_handshake_then_refuses_everything():
    core = FakeCore(built="25.11.5", running="26.05.1")
    rc, frames, err = drive(
        core,
        [
            '{"id": 1, "method": "active_jobs", "params": {}}',
            '{"id": 2, "method": "hello", "params": {}}',
        ],
    )
    assert rc == 0
    assert frames[0]["id"] == 0
    assert frames[0]["result"]["slurm_built_version"] == "25.11.5"
    assert frames[0]["result"]["slurm_running_version"] == "26.05.1"
    for frame in frames[1:]:
        assert frame["error"]["code"] == ERR_VERSION_MISMATCH
        assert "slurm-monitor bootstrap" in frame["error"]["message"]
        assert "25.11.5" in frame["error"]["message"] and "26.05.1" in frame["error"]["message"]
        assert frame["error"]["data"]["slurm_running_version"] == "26.05.1"
    assert core.calls == ["hello"]  # no query ever reached the core


def test_patch_level_difference_is_compatible():
    core = FakeCore(built="25.11.5", running="25.11.6")
    _, frames, _ = drive(core, ['{"id": 1, "method": "active_jobs", "params": {}}'])
    assert "result" in frames[1]


def test_check_prints_handshake_and_exit_code():
    out = io.StringIO()
    assert agent.check(core=FakeCore(), stdout=out) == 0
    assert json.loads(out.getvalue())["hostname"] == "fakehost"
    out = io.StringIO()
    assert agent.check(core=FakeCore(running="24.05.1"), stdout=out) == 1
    assert json.loads(out.getvalue())["slurm_running_version"] == "24.05.1"


def test_record_flag_tolerates_missing_recorder(monkeypatch, tmp_path, caplog):
    import types

    fake_queries = types.ModuleType("slurm_monitor.core.queries")
    fake_queries.PyslurmCore = FakeCore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slurm_monitor.core.queries", fake_queries)
    try:
        import slurm_monitor.core.recording  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("core.recording exists; the ImportError fallback is not reachable")
    monkeypatch.setitem(sys.modules, "slurm_monitor.core.recording", None)
    with caplog.at_level("WARNING"):
        core = agent.build_core(tmp_path)
    assert isinstance(core, FakeCore)
    assert "--record ignored" in caplog.text


def test_subprocess_over_pipes_keeps_stdout_clean():
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from tests.fake_core import FakeCore
from slurm_monitor.agent import run_agent
sys.exit(run_agent(core=FakeCore()))
"""
    requests = (
        '{"id": 1, "method": "active_jobs", "params": {}}\n'
        "garbage\n"
        '{"id": 2, "method": "job_detail", "params": {"job_id": 7}}\n'
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=requests,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert len(lines) == 4
    frames = [json.loads(line) for line in lines]
    assert [f["id"] for f in frames] == [0, 1, -1, 2]
    assert LOG_MARKER in proc.stderr and PRINT_MARKER in proc.stderr
    assert LOG_MARKER not in proc.stdout and PRINT_MARKER not in proc.stdout


def test_module_help_runs_without_pyslurm():
    proc = subprocess.run(
        [sys.executable, "-m", "slurm_monitor.agent", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--check" in proc.stdout and "--record" in proc.stdout
