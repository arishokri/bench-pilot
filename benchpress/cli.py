from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from benchpress.config import COMPONENTS, RunConfig, StressConfig
from benchpress.report.generator import generate_report
from benchpress.runner import run_benchmarks, run_stress
from benchpress.storage import Storage

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Linux benchmarking suite: CPU / RAM / SSD / NVIDIA GPU.",
)
console = Console()


def _parse_components(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return COMPONENTS
    parts = [p.strip().lower() for p in value.split(",") if p.strip()]
    unknown = [p for p in parts if p not in COMPONENTS]
    if unknown:
        raise typer.BadParameter(f"unknown components: {unknown}. Valid: {COMPONENTS}")
    return tuple(parts)


_DURATION_RE = re.compile(r"^(\d+)([smh]?)$")


def _parse_duration(value: str) -> int:
    m = _DURATION_RE.match(value.strip().lower())
    if not m:
        raise typer.BadParameter("duration must look like 90, 90s, 5m, or 1h")
    n = int(m.group(1))
    unit = m.group(2) or "s"
    return n * {"s": 1, "m": 60, "h": 3600}[unit]


@app.command("run")
def cmd_run(
    components: Optional[str] = typer.Option(
        None, "--components", "-c", help="Comma list: cpu,ram,ssd,gpu (default: all)."
    ),
    quick: bool = typer.Option(
        True, "--quick/--full", help="Quick run (~3 min) vs full sweep."
    ),
    label: Optional[str] = typer.Option(
        None, "--label", "-l", help="Human label saved with the run."
    ),
    data_dir: Path = typer.Option(Path("./data"), "--data-dir"),
    ssd_dir: Optional[Path] = typer.Option(
        None,
        "--ssd-dir",
        help="Where fio creates its scratch file (defaults to ./fio_scratch).",
    ),
    hf_cache_dir: Optional[Path] = typer.Option(
        None,
        "--hf-cache-dir",
        help="HuggingFace download cache (defaults to ./hf_cache). Sets HF_HOME for the run.",
    ),
    no_image_gen: bool = typer.Option(
        False, "--no-image-gen", help="Skip the SD-turbo test."
    ),
    ollama_models: Optional[str] = typer.Option(
        None,
        "--ollama-models",
        help="Comma list of ollama model tags to bench (e.g. 'qwen3.5:2b,gemma4:e2b'). Default: auto-pick smallest 2.",
    ),
    sample_interval: float = typer.Option(
        1.0, "--sample-interval", help="Sensor sampling interval seconds."
    ),
    warmup: bool = typer.Option(
        True, "--warmup/--no-warmup",
        help="Warm up the system with a stress load before benchmarking (60s quick / 2m full).",
    ),
    warmup_duration: Optional[str] = typer.Option(
        None, "--warmup-duration",
        help="Override warmup duration, e.g. 90s or 2m. Default: 60s quick / 2m full.",
    ),
) -> None:
    """Run benchmarks against selected components, sampling sensors throughout."""
    cfg = RunConfig(
        components=_parse_components(components),
        quick=quick,
        label=label,
        data_dir=data_dir,
        include_image_gen=not no_image_gen,
        ollama_models=tuple(
            s.strip() for s in (ollama_models or "").split(",") if s.strip()
        ),
        sample_interval=sample_interval,
        warmup=warmup,
    )
    if ssd_dir is not None:
        cfg.ssd_target_dir = ssd_dir
    if hf_cache_dir is not None:
        cfg.hf_cache_dir = hf_cache_dir
    if warmup_duration is not None:
        cfg.warmup_seconds = _parse_duration(warmup_duration)
    run_benchmarks(cfg, console=console)


@app.command("stress")
def cmd_stress(
    duration: str = typer.Option(
        "2m", "--duration", "-d", help="How long to stress (e.g. 90s, 5m, 1h)."
    ),
    components: Optional[str] = typer.Option(
        "cpu", "--components", "-c", help="Comma list: cpu,ram,ssd,gpu."
    ),
    label: Optional[str] = typer.Option(None, "--label", "-l"),
    data_dir: Path = typer.Option(Path("./data"), "--data-dir"),
    ssd_dir: Optional[Path] = typer.Option(None, "--ssd-dir"),
    sample_interval: float = typer.Option(1.0, "--sample-interval"),
) -> None:
    """Time-based stress test for thermal characterization."""
    cfg = StressConfig(
        duration_seconds=_parse_duration(duration),
        components=_parse_components(components),
        label=label,
        data_dir=data_dir,
        sample_interval=sample_interval,
    )
    if ssd_dir is not None:
        cfg.ssd_target_dir = ssd_dir
    run_stress(cfg, console=console)


@app.command("list")
def cmd_list(
    data_dir: Path = typer.Option(Path("./data"), "--data-dir"),
) -> None:
    """List previous runs."""
    storage = Storage(data_dir / "benchpress.db")
    runs = storage.list_runs()
    t = Table(title="Runs")
    t.add_column("id", justify="right")
    t.add_column("started")
    t.add_column("ended")
    t.add_column("mode")
    t.add_column("label")
    t.add_column("host")
    for r in runs:
        t.add_row(
            str(r["id"]),
            r["started_at"] or "",
            r["ended_at"] or "—",
            r["mode"],
            r["label"] or "",
            r["hostname"] or "",
        )
    console.print(t)
    storage.close()


@app.command("report")
def cmd_report(
    data_dir: Path = typer.Option(Path("./data"), "--data-dir"),
    report_dir: Path = typer.Option(Path("./reports"), "--report-dir"),
    open_browser: bool = typer.Option(
        False, "--open/--no-open", help="Open the report in your default browser."
    ),
    only: Optional[str] = typer.Option(
        None, "--only", help="Comma list of run ids; default: all."
    ),
) -> None:
    """Generate a self-contained HTML dashboard from the recorded runs."""
    db = data_dir / "benchpress.db"
    if not db.exists():
        raise typer.BadParameter(
            f"no database at {db}; run `benchpress run --quick` first"
        )
    run_ids: list[int] | None = None
    if only:
        run_ids = [int(x) for x in only.split(",") if x.strip()]
    out = generate_report(db_path=db, out_dir=report_dir, only_run_ids=run_ids)
    console.print(f"Report written to [bold green]{out}[/]")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    app()
