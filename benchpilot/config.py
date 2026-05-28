from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path("./data")
DEFAULT_REPORT_DIR = Path("./reports")
DEFAULT_DB_NAME = "benchpilot.db"

# Components the runner knows about.
COMPONENTS = ("cpu", "ram", "ssd", "gpu")


@dataclass
class RunConfig:
    components: tuple[str, ...] = COMPONENTS
    quick: bool = True            # short comparable run
    label: str | None = None
    data_dir: Path = DEFAULT_DATA_DIR
    report_dir: Path = DEFAULT_REPORT_DIR
    sample_interval: float = 1.0  # seconds between sensor samples
    # SSD benchmarks need a writable directory on the device under test.
    ssd_target_dir: Path = field(default_factory=lambda: Path.home() / ".benchpilot_fio")
    # Optional caps for GPU/ML benchmarks to fit the 3-min target.
    gpu_quick_seconds_per_test: float = 8.0
    # Whether to attempt the heavy SD/SDXL test (requires model download).
    include_image_gen: bool = True
    # Ollama models to probe (must already be `ollama pull`-ed).
    ollama_models: tuple[str, ...] = ()

    @property
    def db_path(self) -> Path:
        return self.data_dir / DEFAULT_DB_NAME


@dataclass
class StressConfig:
    components: tuple[str, ...] = ("cpu",)
    duration_seconds: int = 120
    label: str | None = None
    data_dir: Path = DEFAULT_DATA_DIR
    report_dir: Path = DEFAULT_REPORT_DIR
    sample_interval: float = 1.0
    ssd_target_dir: Path = field(default_factory=lambda: Path.home() / ".benchpilot_fio")

    @property
    def db_path(self) -> Path:
        return self.data_dir / DEFAULT_DB_NAME
