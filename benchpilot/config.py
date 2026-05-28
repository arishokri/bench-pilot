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
    # Per-test GPU compute budget. None = derive from `quick` (8s quick / 20s full).
    gpu_seconds_per_test: float | None = None
    # Whether to attempt the heavy SD/SDXL test (requires model download).
    include_image_gen: bool = True
    # Ollama models to probe (must already be `ollama pull`-ed).
    ollama_models: tuple[str, ...] = ()

    @property
    def db_path(self) -> Path:
        return self.data_dir / DEFAULT_DB_NAME

    @property
    def gpu_budget(self) -> float:
        if self.gpu_seconds_per_test is not None:
            return self.gpu_seconds_per_test
        return 8.0 if self.quick else 20.0


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
