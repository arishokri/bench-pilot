from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from benchpilot.benchmarks._shell import require, run
from benchpilot.benchmarks.base import BenchmarkResult


@dataclass
class CpuStress:
    """stress-ng CPU stress for `duration` seconds. Saturates every thread with mixed FP/int kernels."""

    duration_seconds: int
    threads: int = 0
    name: str = "stress.cpu"
    component: str = "cpu"

    def __post_init__(self) -> None:
        if self.threads <= 0:
            self.threads = os.cpu_count() or 1

    def params(self) -> dict:
        return {"duration_seconds": self.duration_seconds, "threads": self.threads}

    def run(self) -> BenchmarkResult:
        cmd = [
            require("stress-ng"),
            f"--cpu={self.threads}",
            f"--timeout={self.duration_seconds}s",
            "--metrics-brief",
            "--cpu-method=all",
        ]
        sh = run(cmd, timeout=self.duration_seconds + 60, check=False)
        return BenchmarkResult(results={"stderr_tail": sh.stderr[-2000:]})


@dataclass
class RamStress:
    """stress-ng VM workers exercising malloc/free, page churn."""

    duration_seconds: int
    workers: int = 4
    bytes_per_worker: str = "2G"
    name: str = "stress.ram"
    component: str = "ram"

    def params(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "workers": self.workers,
            "bytes_per_worker": self.bytes_per_worker,
        }

    def run(self) -> BenchmarkResult:
        cmd = [
            require("stress-ng"),
            f"--vm={self.workers}",
            f"--vm-bytes={self.bytes_per_worker}",
            "--vm-keep",
            f"--timeout={self.duration_seconds}s",
            "--metrics-brief",
        ]
        sh = run(cmd, timeout=self.duration_seconds + 60, check=False)
        return BenchmarkResult(results={"stderr_tail": sh.stderr[-2000:]})


@dataclass
class SsdStress:
    """stress-ng HDD workers churning a target directory."""

    duration_seconds: int
    target_dir: Path
    workers: int = 2
    name: str = "stress.ssd"
    component: str = "ssd"

    def params(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "workers": self.workers,
            "target_dir": str(self.target_dir),
        }

    def run(self) -> BenchmarkResult:
        self.target_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            require("stress-ng"),
            f"--hdd={self.workers}",
            "--hdd-bytes=4G",
            f"--temp-path={self.target_dir}",
            f"--timeout={self.duration_seconds}s",
            "--metrics-brief",
        ]
        sh = run(cmd, timeout=self.duration_seconds + 60, check=False)
        # stress-ng leaves scratch behind on cancel — clean up
        for p in self.target_dir.glob("stress-ng-*"):
            try:
                p.unlink()
            except OSError:
                pass
        return BenchmarkResult(results={"stderr_tail": sh.stderr[-2000:]})


@dataclass
class GpuStress:
    """Sustained GPU load: repeated large matmul on the default CUDA device.

    Uses PyTorch if available so the test runs at peak FLOPs.
    """

    duration_seconds: int
    matrix: int = 8192
    dtype: str = "bfloat16"
    name: str = "stress.gpu"
    component: str = "gpu"

    def params(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "matrix": self.matrix,
            "dtype": self.dtype,
        }

    def run(self) -> BenchmarkResult:
        import time
        try:
            import torch  # type: ignore
        except ImportError:
            return BenchmarkResult(results={"skipped": "torch not installed"})
        if not torch.cuda.is_available():
            return BenchmarkResult(results={"skipped": "no CUDA device"})
        dt = getattr(torch, self.dtype)
        a = torch.randn(self.matrix, self.matrix, device="cuda", dtype=dt)
        b = torch.randn(self.matrix, self.matrix, device="cuda", dtype=dt)
        torch.cuda.synchronize()
        start = time.monotonic()
        iters = 0
        while (time.monotonic() - start) < self.duration_seconds:
            c = a @ b
            iters += 1
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        # 2*N^3 FLOPs per matmul
        tflops = (2 * self.matrix ** 3 * iters) / elapsed / 1e12
        del a, b, c
        torch.cuda.empty_cache()
        return BenchmarkResult(results={"iters": iters, "elapsed_s": elapsed, "tflops_avg": tflops})
