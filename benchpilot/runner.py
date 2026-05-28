from __future__ import annotations

import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from benchpilot.benchmarks import Benchmark
from benchpilot.benchmarks.registry import build_benchmark_plan, build_stress_plan
from benchpilot.benchmarks.ssd import cleanup_fio_files
from benchpilot.config import RunConfig, StressConfig
from benchpilot.monitor import Sampler
from benchpilot.storage import Storage


def _system_info() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "",
    }
    # CPU model
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    # Memory total
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["mem_total_kib"] = int(line.split()[1])
                    break
    except OSError:
        pass
    # GPU
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True, timeout=2.0,
        )
        info["gpus"] = [line.strip() for line in out.strip().splitlines()]
    except (subprocess.SubprocessError, FileNotFoundError):
        info["gpus"] = []
    return info


@dataclass
class RunSummary:
    run_id: int
    benchmarks_ok: int
    benchmarks_failed: int
    benchmarks_skipped: int


def _prepare_scratch_dirs(*, fio_dir: Path, hf_dir: Path) -> None:
    """Create local scratch dirs and steer HuggingFace + fio at them.

    Done before any benchmark .run() so transformers/diffusers pick up HF_HOME
    on their first import inside a benchmark.
    """
    fio_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    hf_abs = str(hf_dir.resolve())
    os.environ["HF_HOME"] = hf_abs
    # Belt-and-braces: some libs still consult the legacy names.
    os.environ.setdefault("HF_HUB_CACHE", str((hf_dir / "hub").resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_abs)
    os.environ.setdefault("DIFFUSERS_CACHE", hf_abs)


def _execute(plan: Iterable[Benchmark], *, storage: Storage, run_id: int, console: Console) -> RunSummary:
    ok = failed = skipped = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        items = list(plan)
        task = progress.add_task("running", total=len(items), status="")
        for bench in items:
            progress.update(task, description=f"[bold blue]{bench.name}", status="…")
            bench_id = storage.start_benchmark(
                run_id, name=bench.name, component=bench.component, params=bench.params(),
            )
            try:
                result = bench.run()
            except Exception as e:  # noqa: BLE001
                storage.finish_benchmark(bench_id, status="error", error=str(e))
                failed += 1
                progress.update(task, advance=1, status=f"[red]✗ {e}")
                continue
            res = result.results
            if isinstance(res, dict) and res.get("skipped"):
                storage.finish_benchmark(bench_id, status="skipped", results=res)
                skipped += 1
                progress.update(task, advance=1, status=f"[yellow]– {res['skipped']}")
            else:
                storage.finish_benchmark(bench_id, status="ok", results=res)
                ok += 1
                progress.update(task, advance=1, status="[green]✓")
    return RunSummary(run_id=run_id, benchmarks_ok=ok, benchmarks_failed=failed, benchmarks_skipped=skipped)


def run_benchmarks(cfg: RunConfig, console: Console | None = None) -> RunSummary:
    console = console or Console()
    _prepare_scratch_dirs(fio_dir=cfg.ssd_target_dir, hf_dir=cfg.hf_cache_dir)
    storage = Storage(cfg.db_path)
    sysinfo = _system_info()
    label = cfg.label or ("quick" if cfg.quick else "full")
    run_id = storage.start_run(mode="bench", label=label, hostname=socket.gethostname(), system_info=sysinfo)
    sampler = Sampler(storage, run_id, interval=cfg.sample_interval)
    sampler.start()
    try:
        _render_sysinfo(console, sysinfo)
        plan = build_benchmark_plan(cfg)
        summary = _execute(plan, storage=storage, run_id=run_id, console=console)
    finally:
        sampler.stop()
        cleanup_fio_files(cfg.ssd_target_dir)
        storage.end_run(run_id)
        storage.close()
    console.print(f"\nRun [bold cyan]{run_id}[/] finished — "
                  f"[green]{summary.benchmarks_ok} ok[/], "
                  f"[yellow]{summary.benchmarks_skipped} skipped[/], "
                  f"[red]{summary.benchmarks_failed} failed[/]")
    return summary


def run_stress(cfg: StressConfig, console: Console | None = None) -> RunSummary:
    console = console or Console()
    _prepare_scratch_dirs(fio_dir=cfg.ssd_target_dir, hf_dir=cfg.hf_cache_dir)
    storage = Storage(cfg.db_path)
    sysinfo = _system_info()
    label = cfg.label or f"stress-{cfg.duration_seconds}s"
    run_id = storage.start_run(mode="stress", label=label, hostname=socket.gethostname(), system_info=sysinfo)
    sampler = Sampler(storage, run_id, interval=cfg.sample_interval)
    sampler.start()
    try:
        _render_sysinfo(console, sysinfo)
        plan = build_stress_plan(cfg)
        summary = _execute(plan, storage=storage, run_id=run_id, console=console)
    finally:
        sampler.stop()
        storage.end_run(run_id)
        storage.close()
    console.print(f"\nStress run [bold cyan]{run_id}[/] finished.")
    return summary


def _render_sysinfo(console: Console, info: dict) -> None:
    t = Table(title="System", show_header=False, box=None)
    t.add_row("CPU", info.get("cpu_model", info.get("cpu", "?")))
    mem_kib = info.get("mem_total_kib")
    if mem_kib:
        t.add_row("RAM", f"{mem_kib / 1024 / 1024:.1f} GiB")
    for g in info.get("gpus") or ["(none detected)"]:
        t.add_row("GPU", g)
    t.add_row("Python", info.get("python", "?"))
    console.print(t)
