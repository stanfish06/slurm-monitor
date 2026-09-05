"""Newline-delimited JSON protocol between client and agent.

One request object per line on the agent's stdin, one response per line on its stdout,
correlated by integer id. The agent never writes anything but frames to stdout; logs go to
stderr. Frames:

    request:  {"id": 1, "method": "active_jobs", "params": {...}}
    response: {"id": 1, "result": {...}}
    error:    {"id": 1, "error": {"code": "rpc_error", "message": "...", "data": {...}}}

The handshake is the agent's first unsolicited frame: {"id": 0, "result": {<Handshake>}}.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

# Error codes the agent emits. The client maps these to user-facing behavior.
ERR_BAD_REQUEST = "bad_request"  # malformed frame or unknown method
ERR_RPC = "rpc_error"  # slurmctld/slurmdbd refused or failed
ERR_NOT_FOUND = "not_found"  # job id unknown to the controller
ERR_NOT_CANCELLABLE = "not_cancellable"  # job reached a terminal state before cancel
ERR_VERSION_MISMATCH = "version_mismatch"  # agent build does not match running Slurm
ERR_INTERNAL = "internal"


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    method: str
    params: dict[str, Any] = {}


class ErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    data: dict[str, Any] | None = None


class Response(BaseModel):
    """Exactly one of result / error is set."""

    model_config = ConfigDict(extra="forbid")

    id: int
    result: dict[str, Any] | None = None
    error: ErrorInfo | None = None


class ProtocolError(Exception):
    """A line that is not a valid frame."""


def encode(frame: BaseModel) -> bytes:
    """Serialize one frame to a single line (no embedded newlines) plus terminator."""
    line = frame.model_dump_json(exclude_none=True)
    return line.encode("utf-8") + b"\n"


def decode_request(line: bytes | str) -> Request:
    try:
        return Request.model_validate(_load_object(line))
    except ValidationError as e:
        raise ProtocolError(f"invalid request frame: {e.errors()[0]['msg']}") from e


def decode_response(line: bytes | str) -> Response:
    try:
        return Response.model_validate(_load_object(line))
    except ValidationError as e:
        raise ProtocolError(f"invalid response frame: {e.errors()[0]['msg']}") from e


def _load_object(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line:
        raise ProtocolError("empty line")
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"not JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise ProtocolError("frame is not a JSON object")
    return obj


def error_response(
    req_id: int, code: str, message: str, data: dict[str, Any] | None = None
) -> Response:
    return Response(id=req_id, error=ErrorInfo(code=code, message=message, data=data))


def result_response(req_id: int, result: BaseModel | dict[str, Any]) -> Response:
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json")
    return Response(id=req_id, result=result)
