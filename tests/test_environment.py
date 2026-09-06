import os
import sys

import pytest

from slurm_monitor.core import environment
from slurm_monitor.models import Handshake, versions_compatible


def test_probe_extension_reports_failure_when_module_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, environment.EXTENSION_MODULE, None)
    ok, err = environment.probe_extension()
    assert ok is False
    assert err is not None and environment.EXTENSION_MODULE in err


def test_probe_extension_reports_success_when_importable(monkeypatch):
    import types

    monkeypatch.setitem(sys.modules, environment.EXTENSION_MODULE, types.ModuleType("fake_ext"))
    assert environment.probe_extension() == (True, None)


def test_probe_extension_captures_loader_errors(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == environment.EXTENSION_MODULE:
            raise OSError("libslurmfull.so: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, environment.EXTENSION_MODULE, raising=False)
    monkeypatch.setattr(builtins, "__import__", failing_import)
    ok, err = environment.probe_extension()
    assert ok is False
    assert err is not None and "libslurmfull" in err and err.startswith("OSError")


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ({"task": 30}, 30),
        ({"task": 15, "energy": 60}, 15),
        ("task=30", 30),
        ("energy=60,task=45", 45),
        ("task", 30),
        ({}, 30),
        (None, 30),
        ("", 30),
        ({"task": "abc"}, 30),
        ({"task": 0}, 30),
    ],
)
def test_parse_sampling_interval(freq, expected):
    assert environment.parse_sampling_interval(freq) == expected


def test_versions_compatible_masks_patch():
    assert versions_compatible("25.11.5", "25.11.6")
    assert not versions_compatible("25.11.5", "25.05.5")
    assert not versions_compatible("25.11.5", "26.11.5")


@pytest.mark.cluster
def test_gather_handshake_on_slurm_host():
    pytest.importorskip("pyslurm")
    hs = environment.gather_handshake(*environment.probe_extension())
    assert isinstance(hs, Handshake)
    assert hs.user == os.environ.get("USER", hs.user)
    assert hs.slurm_built_version.count(".") == 2
    assert versions_compatible(hs.slurm_built_version, hs.slurm_running_version)
    assert hs.sampling_interval_seconds > 0
