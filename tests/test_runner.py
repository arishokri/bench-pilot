from dataclasses import dataclass

import pytest
from rich.console import Console

from benchpress import runner as runner_mod
from benchpress.benchmarks.base import BenchmarkResult
from benchpress.config import COMPONENTS, RunConfig
from benchpress.runner import (
    _components_suffix,
    _execute,
    _gpu_cleanup,
    _prepare_scratch_dirs,
    _system_info,
)

# _execute_concurrent ships with feature/concurrent; on main this symbol doesn't
# exist yet. We skip the concurrent-executor tests below rather than fail at
# collection time.
try:
    from benchpress.runner import _execute_concurrent  # noqa: F401
    HAS_CONCURRENT_EXECUTOR = True
except ImportError:
    HAS_CONCURRENT_EXECUTOR = False

requires_concurrent_executor = pytest.mark.skipif(
    not HAS_CONCURRENT_EXECUTOR,
    reason="_execute_concurrent lands with feature/concurrent merge",
)


# ---------- _components_suffix ----------
@pytest.mark.parametrize("components,expected", [
    (COMPONENTS, "all"),
    (("cpu",), "cpu"),
    (("gpu",), "gpu"),
    (("cpu", "ram", "gpu"), "cpu_ram_gpu"),
    (("gpu", "cpu"), "cpu_gpu"),               # canonical order regardless of input
    (("ssd", "ram", "cpu"), "cpu_ram_ssd"),
])
def test_components_suffix_canonical(components, expected):
    assert _components_suffix(components) == expected


# ---------- _prepare_scratch_dirs ----------
def test_prepare_scratch_dirs_creates_and_sets_env(tmp_path, monkeypatch):
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "DIFFUSERS_CACHE",
                "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(key, raising=False)
    fio = tmp_path / "fio"
    hf = tmp_path / "hf"
    _prepare_scratch_dirs(fio_dir=fio, hf_dir=hf)
    assert fio.is_dir()
    assert hf.is_dir()
    import os
    assert os.environ["HF_HOME"] == str(hf.resolve())
    assert os.environ["HF_HUB_CACHE"].startswith(str(hf.resolve()))
    assert os.environ["TRANSFORMERS_CACHE"] == str(hf.resolve())
    assert os.environ["DIFFUSERS_CACHE"] == str(hf.resolve())
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_prepare_scratch_dirs_skips_hf_when_none(tmp_path, monkeypatch):
    for key in ("HF_HOME",):
        monkeypatch.delenv(key, raising=False)
    fio = tmp_path / "fio"
    _prepare_scratch_dirs(fio_dir=fio, hf_dir=None)
    assert fio.is_dir()
    import os
    assert "HF_HOME" not in os.environ  # untouched


# ---------- _gpu_cleanup ----------
def test_gpu_cleanup_is_noop_when_torch_missing(monkeypatch):
    """If `import torch` raises ImportError, _gpu_cleanup returns silently."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _gpu_cleanup()  # must not raise


# ---------- _system_info ----------
def test_system_info_includes_gpu_when_smi_present(mock_check_output):
    mock_check_output("NVIDIA GeForce RTX 5080, 595.71.05, 16303 MiB\n")
    info = _system_info()
    assert info["gpus"] == ["NVIDIA GeForce RTX 5080, 595.71.05, 16303 MiB"]
    assert "python" in info
    assert "platform" in info


def test_system_info_gpus_empty_when_smi_fails(mock_check_output):
    mock_check_output(FileNotFoundError("nvidia-smi"))
    info = _system_info()
    assert info["gpus"] == []


# ---------- _execute ----------
@dataclass
class _MockBench:
    name: str
    component: str
    _outcome: str = "ok"   # "ok" | "skipped" | "raise"

    def params(self) -> dict:
        return {"mocked": True}

    def run(self, cancel=None) -> BenchmarkResult:
        if self._outcome == "raise":
            raise RuntimeError("simulated failure")
        if self._outcome == "skipped":
            return BenchmarkResult(results={"skipped": "stubbed"})
        return BenchmarkResult(results={"answer": 42})


def test_execute_records_ok_skipped_and_error(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    plan = [
        _MockBench(name="m.ok", component="cpu", _outcome="ok"),
        _MockBench(name="m.skip", component="ram", _outcome="skipped"),
        _MockBench(name="m.err", component="ssd", _outcome="raise"),
    ]
    summary = _execute(plan, storage=temp_db, run_id=run_id,
                       console=Console(record=False, force_terminal=False))
    assert (summary.benchmarks_ok, summary.benchmarks_skipped, summary.benchmarks_failed) == (1, 1, 1)
    rows = {r["name"]: r for r in temp_db.benchmarks_for_run(run_id)}
    assert rows["m.ok"]["status"] == "ok"
    assert rows["m.ok"]["results"] == {"answer": 42}
    assert rows["m.skip"]["status"] == "skipped"
    assert rows["m.err"]["status"] == "error"
    assert "simulated failure" in rows["m.err"]["error"]


def test_execute_empty_plan_returns_zero_counts(temp_db):
    run_id = temp_db.start_run(mode="bench", label="t", hostname="h", system_info={})
    summary = _execute([], storage=temp_db, run_id=run_id,
                       console=Console(force_terminal=False))
    assert summary.benchmarks_ok == 0
    assert summary.benchmarks_failed == 0
    assert summary.benchmarks_skipped == 0


# ---------- _execute_concurrent (stress-mode executor, feature/concurrent only) ----------
@requires_concurrent_executor
def test_execute_concurrent_runs_all_in_parallel(temp_db):
    """Three benchmarks, each sleeping 100ms. Sequentially they'd take ~300ms;
    concurrently they should fit well under 250ms."""
    import time
    from benchpress.benchmarks.base import BenchmarkResult
    from benchpress.runner import _execute_concurrent

    class _Sleeper:
        def __init__(self, name): self.name = name; self.component = "cpu"
        def params(self): return {}
        def run(self, cancel=None):
            time.sleep(0.1)
            return BenchmarkResult(results={"slept": 0.1})

    run_id = temp_db.start_run(mode="stress", label="t", hostname="h", system_info={})
    plan = [_Sleeper(f"sleep.{i}") for i in range(3)]
    start = time.monotonic()
    summary = _execute_concurrent(plan, storage=temp_db, run_id=run_id,
                                   console=Console(force_terminal=False))
    elapsed = time.monotonic() - start
    assert summary.benchmarks_ok == 3
    assert summary.benchmarks_failed == 0
    assert elapsed < 0.25, f"executor was not concurrent — took {elapsed:.3f}s"


@requires_concurrent_executor
def test_execute_concurrent_handles_mixed_outcomes(temp_db):
    """One ok, one skipped, one raising — all status fields recorded correctly."""
    from benchpress.runner import _execute_concurrent

    run_id = temp_db.start_run(mode="stress", label="t", hostname="h", system_info={})
    plan = [
        _MockBench(name="c.ok", component="cpu", _outcome="ok"),
        _MockBench(name="c.skip", component="ram", _outcome="skipped"),
        _MockBench(name="c.err", component="gpu", _outcome="raise"),
    ]
    summary = _execute_concurrent(plan, storage=temp_db, run_id=run_id,
                                   console=Console(force_terminal=False))
    assert (summary.benchmarks_ok, summary.benchmarks_skipped, summary.benchmarks_failed) == (1, 1, 1)
    rows = {r["name"]: r for r in temp_db.benchmarks_for_run(run_id)}
    assert rows["c.ok"]["status"] == "ok"
    assert rows["c.skip"]["status"] == "skipped"
    assert rows["c.err"]["status"] == "error"
    # Every row ended (no stuck 'running' state)
    assert all(r["ended_at"] is not None for r in rows.values())


@requires_concurrent_executor
def test_execute_concurrent_empty_plan(temp_db):
    from benchpress.runner import _execute_concurrent

    run_id = temp_db.start_run(mode="stress", label="t", hostname="h", system_info={})
    summary = _execute_concurrent([], storage=temp_db, run_id=run_id,
                                   console=Console(force_terminal=False))
    assert (summary.benchmarks_ok, summary.benchmarks_failed, summary.benchmarks_skipped) == (0, 0, 0)
