from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from benchpilot.storage import Storage


# ---------- storage ----------
@pytest.fixture
def temp_db(tmp_path: Path):
    s = Storage(tmp_path / "test.db")
    yield s
    s.close()


# ---------- hwmon fake tree ----------
@pytest.fixture
def fake_hwmon(tmp_path: Path, monkeypatch):
    """Build a fake /sys/class/hwmon tree under tmp_path and point HwmonProbe at it.

    Usage:
        fake_hwmon({
            "hwmon0": {
                "name": "coretemp",
                "temp1_input": "42000",      # millideg C  -> 42.0 C
                "temp1_label": "Package",
            },
            "hwmon1": {
                "name": "nct6798",
                "fan1_input": "1200",
                "fan1_label": "CPU_FAN",
                "in1_input": "1234",         # millivolts -> 1.234 V
            },
        })
    """
    root = tmp_path / "sys" / "class" / "hwmon"
    root.mkdir(parents=True)

    def build(spec: dict[str, dict[str, str]]):
        for chip, entries in spec.items():
            d = root / chip
            d.mkdir()
            for fname, content in entries.items():
                (d / fname).write_text(content)
        # Patch the module-level _HWMON pointer
        from benchpilot.monitor import hwmon as hwmod
        monkeypatch.setattr(hwmod, "_HWMON", root)
        return root

    return build


# ---------- _shell.run capture ----------
@dataclass
class _CapturedShell:
    cmdline: list[str] | None = None
    timeout: float | None = None
    check: bool | None = None


@pytest.fixture
def mock_shell(monkeypatch):
    """Patch benchpilot.benchmarks._shell.run AND its already-imported aliases
    so callers can assert what cmdline was constructed. Returns a function
    set_result(stdout="", stderr="", returncode=0) plus a `captured` record.
    """
    from benchpilot.benchmarks import _shell

    captured = _CapturedShell()
    result = {"stdout": "", "stderr": "", "returncode": 0}

    def set_result(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        result.update(stdout=stdout, stderr=stderr, returncode=returncode)

    def fake_run(cmd, *, timeout=None, check=True):
        captured.cmdline = list(cmd)
        captured.timeout = timeout
        captured.check = check
        return _shell.ShellResult(result["returncode"], result["stdout"], result["stderr"])

    # Patch the module attribute *and* every benchmark module that already imported `run` by name.
    monkeypatch.setattr(_shell, "run", fake_run)
    for mod_name in (
        "benchpilot.benchmarks.cpu",
        "benchpilot.benchmarks.ram",
        "benchpilot.benchmarks.ssd",
        "benchpilot.benchmarks.stress",
    ):
        import importlib
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "run"):
            monkeypatch.setattr(mod, "run", fake_run)

    # `require` always succeeds during tests (the real tool isn't installed in CI).
    def fake_require(tool):
        return f"/usr/bin/{tool}"
    monkeypatch.setattr(_shell, "require", fake_require)
    for mod_name in (
        "benchpilot.benchmarks.cpu",
        "benchpilot.benchmarks.ram",
        "benchpilot.benchmarks.ssd",
        "benchpilot.benchmarks.stress",
    ):
        import importlib
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "require"):
            monkeypatch.setattr(mod, "require", fake_require)

    return captured, set_result


# ---------- subprocess.check_output stub ----------
@pytest.fixture
def mock_check_output(monkeypatch):
    """Patches subprocess.check_output globally. Yields a setter taking a callable
    `cmd -> str` OR a flat string."""
    import subprocess as sp

    state = {"handler": lambda cmd: ""}

    def fake_check_output(cmd, **_kwargs):
        out = state["handler"](cmd)
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr(sp, "check_output", fake_check_output)

    def set_handler(handler):
        if isinstance(handler, str):
            payload = handler
            state["handler"] = lambda _cmd: payload
        elif isinstance(handler, BaseException) or (
            isinstance(handler, type) and issubclass(handler, BaseException)
        ):
            err = handler
            def _raise(_cmd):
                raise err if not isinstance(err, type) else err()
            state["handler"] = _raise
        else:
            state["handler"] = handler

    return set_handler


# ---------- GPU gate ----------
def _cuda_torch_available() -> bool:
    try:
        import torch  # type: ignore
    except ImportError:
        return False
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(not _cuda_torch_available(), reason="needs CUDA torch")
