import json

import pytest

from slurm_monitor.protocol import (
    ERR_NOT_FOUND,
    ErrorInfo,
    ProtocolError,
    Request,
    Response,
    decode_request,
    decode_response,
    encode,
    error_response,
    result_response,
)
from tests.fake_agent import ACTIVE


def test_request_roundtrip():
    req = Request(id=7, method="job_detail", params={"job_id": 42})
    line = encode(req)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert decode_request(line) == req
    assert decode_request(line.decode()) == req  # str input accepted too


def test_result_response_roundtrip_carries_model_payload():
    resp = result_response(3, ACTIVE)
    again = decode_response(encode(resp))
    assert again.id == 3 and again.error is None
    assert again.result == ACTIVE.model_dump(mode="json")
    assert type(ACTIVE).model_validate(again.result) == ACTIVE


def test_error_response_roundtrip_and_none_fields_omitted():
    resp = error_response(9, ERR_NOT_FOUND, "job 1 unknown", {"job_id": 1})
    raw = json.loads(encode(resp))
    assert raw == {
        "id": 9,
        "error": {"code": ERR_NOT_FOUND, "message": "job 1 unknown", "data": {"job_id": 1}},
    }
    again = decode_response(encode(resp))
    assert again.error == ErrorInfo(code=ERR_NOT_FOUND, message="job 1 unknown", data={"job_id": 1})
    assert again.result is None


def test_frames_never_contain_embedded_newlines():
    resp = error_response(1, "rpc_error", "line one\nline two\r\nline three")
    line = encode(resp)
    assert line.count(b"\n") == 1 and line.endswith(b"\n")
    assert decode_response(line).error.message == "line one\nline two\r\nline three"
    req = Request(id=2, method="cancel", params={"note": "a\nb"})
    assert encode(req).count(b"\n") == 1


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        (b"", "empty line"),
        (b"   \n", "empty line"),
        (b"Welcome to the login node\n", "not JSON"),
        (b"[1, 2, 3]\n", "not a JSON object"),
        (b'"just a string"\n', "not a JSON object"),
        (b'{"method": "hello"}\n', "required"),
        (b'{"id": "one", "method": "hello"}\n', "integer"),
        (b'{"id": 1, "method": 5}\n', "string"),
        (b'{"id": 1, "method": "hello", "params": []}\n', "dict"),
        (b'{"id": 1, "method": "hello", "extra": true}\n', "extra"),
    ],
)
def test_malformed_request_raises_protocol_error(line: bytes, fragment: str):
    with pytest.raises(ProtocolError) as info:
        decode_request(line)
    assert fragment.lower() in str(info.value).lower()


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        (b"", "empty line"),
        (b"garbage\n", "not JSON"),
        (b"[]\n", "not a JSON object"),
        (b'{"result": {}}\n', "required"),
        (b'{"id": 1.5, "result": {}}\n', "integer"),
        (b'{"id": 1, "result": [1]}\n', "dict"),
        (b'{"id": 1, "error": "boom"}\n', "dict"),
        (b'{"id": 1, "error": {"code": "x"}}\n', "required"),
    ],
)
def test_malformed_response_raises_protocol_error(line: bytes, fragment: str):
    with pytest.raises(ProtocolError) as info:
        decode_response(line)
    assert fragment.lower() in str(info.value).lower()


def test_response_model_accepts_only_known_fields():
    with pytest.raises(ProtocolError, match="(?i)extra"):
        decode_response(b'{"id": 1, "result": {}, "bogus": 1}')
    assert Response(id=0, result={}).error is None
