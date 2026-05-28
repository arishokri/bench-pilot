import pytest

from benchpilot.benchmarks._shell import ShellResult, ToolMissing, require, run


def test_require_returns_path_when_present(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    assert require("ls") == "/usr/bin/ls"


def test_require_raises_when_missing(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(ToolMissing):
        require("does-not-exist")


def test_run_returns_shellresult_on_success():
    out = run(["true"], timeout=5, check=True)
    assert isinstance(out, ShellResult)
    assert out.returncode == 0


def test_run_raises_on_failure_when_check_true():
    with pytest.raises(RuntimeError):
        run(["false"], timeout=5, check=True)


def test_run_returns_nonzero_when_check_false():
    out = run(["false"], timeout=5, check=False)
    assert out.returncode != 0
