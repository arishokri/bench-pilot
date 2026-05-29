from __future__ import annotations

import os
import platform
import signal
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from benchpress.benchmarks import Benchmark
from benchpress.benchmarks.registry import build_benchmark_plan, build_stress_plan
from benchpress.benchmarks.ssd import cleanup_fio_files
from benchpress.config import COMPONENTS, RunConfig, StressConfig
from benchpress.monitor import Sampler
from benchpress.storage import Storage


def _components_suffix(components: tuple[str, ...]) -> str:
    """Render the component selection as a label suffix.

    All four components → 'all'. Subsets are joined with '_' in the canonical
    COMPONENTS order so labels are stable regardless of CLI input order.
    """
    if set(components) == set(COMPONENTS):
        return "all"
    return "_".join(c for c in COMPONENTS if c in components)


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
    benchmarks_interrupted: int = 0


def _prepare_scratch_dirs(*, fio_dir: Path, hf_dir: Path | None = None) -> None:
    """Create local scratch dirs and steer HuggingFace + fio at them.

    Done before any benchmark .run() so transformers/diffusers pick up HF_HOME
    on their first import inside a benchmark. Pass hf_dir=None when the run will
    never touch HuggingFace models (e.g. stress mode).
    """
    fio_dir.mkdir(parents=True, exist_ok=True)
    if hf_dir is not None:
        hf_dir.mkdir(parents=True, exist_ok=True)
        hf_abs = str(hf_dir.resolve())
        os.environ["HF_HOME"] = hf_abs
        # Belt-and-braces: some libs still consult the legacy names.
        os.environ.setdefault("HF_HUB_CACHE", str((hf_dir / "hub").resolve()))
        os.environ.setdefault("TRANSFORMERS_CACHE", hf_abs)
        os.environ.setdefault("DIFFUSERS_CACHE", hf_abs)
    # Reduce CUDA allocator fragmentation across back-to-back GPU benchmarks.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _gpu_cleanup() -> None:
    """Run between GPU benchmarks to reclaim VRAM that each test's `del` + `empty_cache`
    couldn't free on its own (cudnn workspace, autograd graph leftovers, ref cycles)."""
    import gc
    gc.collect()
    try:
        import torch  # type: ignore
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _execute(plan: Iterable[Benchmark], *, storage: Storage, run_id: int, console: Console) -> RunSummary:
    ok = failed = skipped = interrupted = 0
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
        # Track the benchmark whose row is open ("running") so Ctrl+C can finalize
        # it as "interrupted" instead of leaving a dangling record. KeyboardInterrupt
        # is a BaseException, so the inner `except Exception` doesn't swallow it.
        in_flight: tuple[int, Benchmark] | None = None
        try:
            for bench in items:
                progress.update(task, description=f"[bold blue]{bench.name}", status="…")
                bench_id = storage.start_benchmark(
                    run_id, name=bench.name, component=bench.component, params=bench.params(),
                )
                in_flight = (bench_id, bench)
                try:
                    result = bench.run()
                except Exception as e:  # noqa: BLE001
                    storage.finish_benchmark(bench_id, status="error", error=str(e))
                    failed += 1
                    progress.update(task, advance=1, status=f"[red]✗ {e}")
                    in_flight = None
                    if bench.component == "gpu":
                        _gpu_cleanup()
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
                in_flight = None
                if bench.component == "gpu":
                    _gpu_cleanup()
        except KeyboardInterrupt:
            if in_flight is not None:
                cid, bench = in_flight
                storage.finish_benchmark(cid, status="interrupted")
                interrupted += 1
                progress.update(task, advance=1, status="[yellow]⨯ interrupted")
            console.print("\n[yellow]⚠ Interrupt — stopping run…[/]")
    return RunSummary(
        run_id=run_id, benchmarks_ok=ok, benchmarks_failed=failed,
        benchmarks_skipped=skipped, benchmarks_interrupted=interrupted,
    )


def _execute_concurrent(
    plan: Iterable[Benchmark], *, storage: Storage, run_id: int, console: Console,
) -> RunSummary:
    """Run every benchmark in `plan` in its own thread so they overlap in time.

    Used by stress mode where the goal is *simultaneous* component load
    (worst-case thermals, PSU pull, shared-resource contention), not isolated
    measurement.
    """
    items = list(plan)
    if not items:
        return RunSummary(run_id=run_id, benchmarks_ok=0, benchmarks_failed=0, benchmarks_skipped=0)

    # Tallies are touched from worker threads; guard with a lock.
    counts = {"ok": 0, "failed": 0, "skipped": 0, "interrupted": 0}
    counts_lock = threading.Lock()
    # Cooperative cancellation: set on Ctrl+C so workers (subprocesses + the GPU
    # Python loop) wind down promptly instead of running out the full duration.
    cancel = threading.Event()

    def run_one(bench: Benchmark) -> tuple[str, str, str | None]:
        bench_id = storage.start_benchmark(
            run_id, name=bench.name, component=bench.component, params=bench.params(),
        )
        try:
            result = bench.run(cancel=cancel)
        except Exception as e:  # noqa: BLE001
            storage.finish_benchmark(bench_id, status="error", error=str(e))
            with counts_lock:
                counts["failed"] += 1
            return ("error", bench.name, str(e))
        res = result.results
        if cancel.is_set():
            storage.finish_benchmark(bench_id, status="interrupted", results=res)
            with counts_lock:
                counts["interrupted"] += 1
            return ("interrupted", bench.name, None)
        if isinstance(res, dict) and res.get("skipped"):
            storage.finish_benchmark(bench_id, status="skipped", results=res)
            with counts_lock:
                counts["skipped"] += 1
            return ("skipped", bench.name, str(res.get("skipped")))
        storage.finish_benchmark(bench_id, status="ok", results=res)
        with counts_lock:
            counts["ok"] += 1
        return ("ok", bench.name, None)

    def apply(fut) -> None:
        status, name, info = fut.result()
        if status == "ok":
            progress.update(bench_tasks[name], status="[green]✓", completed=1, total=1)
        elif status == "skipped":
            progress.update(bench_tasks[name], status=f"[yellow]– {info}", completed=1, total=1)
        elif status == "interrupted":
            progress.update(bench_tasks[name], status="[yellow]⨯ interrupted", completed=1, total=1)
        else:
            progress.update(bench_tasks[name], status=f"[red]✗ {info}", completed=1, total=1)

    # First Ctrl+C requests cooperative cancellation; a second one force-quits.
    # We install an explicit SIGINT handler rather than relying on KeyboardInterrupt
    # surfacing inside as_completed() — the main thread is blocked in a lock wait
    # there, so a handler that just sets the event is far more reliable (PEP 475).
    interrupts = {"n": 0}

    def _on_sigint(signum, frame):
        interrupts["n"] += 1
        if interrupts["n"] >= 2:
            os._exit(130)
        cancel.set()
        console.print(
            "\n[yellow]⚠ Interrupt — stopping stress run "
            "(press Ctrl+C again to force quit)…[/]"
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        bench_tasks = {b.name: progress.add_task(b.name, total=None, status="…") for b in items}
        try:
            prev_handler = signal.signal(signal.SIGINT, _on_sigint)
        except ValueError:
            prev_handler = None  # not on the main thread; fall back to default behaviour
        ex = ThreadPoolExecutor(max_workers=len(items))
        try:
            futures = {ex.submit(run_one, b): b for b in items}
            for fut in as_completed(futures):
                apply(fut)
        except KeyboardInterrupt:
            # Belt-and-braces if SIGINT still surfaced as an exception.
            cancel.set()
            for fut in as_completed(futures):
                apply(fut)
        finally:
            if prev_handler is not None:
                signal.signal(signal.SIGINT, prev_handler)
            ex.shutdown(wait=False, cancel_futures=True)
    return RunSummary(
        run_id=run_id,
        benchmarks_ok=counts["ok"],
        benchmarks_failed=counts["failed"],
        benchmarks_skipped=counts["skipped"],
        benchmarks_interrupted=counts["interrupted"],
    )


def run_benchmarks(cfg: RunConfig, console: Console | None = None) -> RunSummary:
    console = console or Console()
    _prepare_scratch_dirs(fio_dir=cfg.ssd_target_dir, hf_dir=cfg.hf_cache_dir)
    storage = Storage(cfg.db_path)
    sysinfo = _system_info()
    label = cfg.label or f"{'quick' if cfg.quick else 'full'}-{_components_suffix(cfg.components)}"
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
    verb = "interrupted" if summary.benchmarks_interrupted else "finished"
    tail = (f", [yellow]{summary.benchmarks_interrupted} interrupted[/]"
            if summary.benchmarks_interrupted else "")
    console.print(f"\nRun [bold cyan]{run_id}[/] {verb} — "
                  f"[green]{summary.benchmarks_ok} ok[/], "
                  f"[yellow]{summary.benchmarks_skipped} skipped[/], "
                  f"[red]{summary.benchmarks_failed} failed[/]" + tail)
    return summary


def run_stress(cfg: StressConfig, console: Console | None = None) -> RunSummary:
    console = console or Console()
    _prepare_scratch_dirs(fio_dir=cfg.ssd_target_dir)
    storage = Storage(cfg.db_path)
    sysinfo = _system_info()
    label = cfg.label or f"stress-{_components_suffix(cfg.components)}"
    run_id = storage.start_run(mode="stress", label=label, hostname=socket.gethostname(), system_info=sysinfo)
    sampler = Sampler(storage, run_id, interval=cfg.sample_interval)
    sampler.start()
    try:
        _render_sysinfo(console, sysinfo)
        plan = build_stress_plan(cfg)
        summary = _execute_concurrent(plan, storage=storage, run_id=run_id, console=console)
    finally:
        sampler.stop()
        storage.end_run(run_id)
        storage.close()
    if summary.benchmarks_interrupted:
        console.print(
            f"\nStress run [bold cyan]{run_id}[/] [yellow]interrupted[/] — partial data saved."
        )
    else:
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
