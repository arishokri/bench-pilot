from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass

from benchpilot.benchmarks._shell import require, run
from benchpilot.benchmarks.base import BenchmarkResult


@dataclass
class MbwBench:
    """mbw: simple memcpy bandwidth in three flavours (memcpy, dumb, MCBLOCK)."""

    name: str = "ram.mbw"
    component: str = "ram"
    size_mib: int = 1024
    iterations: int = 3

    def params(self) -> dict:
        return {"size_mib": self.size_mib, "iterations": self.iterations}

    def run(self) -> BenchmarkResult:
        cmd = [require("mbw"), "-n", str(self.iterations), str(self.size_mib)]
        sh = run(cmd, timeout=120)
        # mbw output lines look like: "AVG     Method: MEMCPY  Elapsed: 0.0xx Speed: 12345.6 MiB/s"
        method_speeds: dict[str, float] = {}
        for line in sh.stdout.splitlines():
            m = re.search(r"AVG\s+Method:\s+(\w+)\s+Elapsed:\s+[\d.]+\s+MiB:\s+\d+\s+Copy:\s+([\d.]+)\s*MiB/s", line)
            if m:
                method_speeds[m.group(1).lower()] = float(m.group(2))
                continue
            # Newer mbw variant
            m2 = re.search(r"AVG\s+Method:\s+(\w+).*?(\d+\.\d+)\s*MiB/s", line)
            if m2:
                method_speeds[m2.group(1).lower()] = float(m2.group(2))
        return BenchmarkResult(results={"bandwidth_mib_s": method_speeds})


@dataclass
class StreamBench:
    """John McCalpin's STREAM (Copy/Scale/Add/Triad) — if the system binary is present."""

    name: str = "ram.stream"
    component: str = "ram"

    def params(self) -> dict:
        return {}

    def run(self) -> BenchmarkResult:
        # Some distros ship `stream` at /usr/bin/stream
        path = shutil.which("stream")
        if path is None:
            return BenchmarkResult(results={"skipped": "stream binary not on PATH"})
        sh = run([path], timeout=120, check=False)
        results: dict[str, float] = {}
        # Standard STREAM output has lines like:
        # "Copy:         12345.6     0.001234     0.001000     0.001500"
        for line in sh.stdout.splitlines():
            m = re.match(r"\s*(Copy|Scale|Add|Triad):\s+([\d.]+)\s+", line)
            if m:
                results[m.group(1).lower() + "_mb_s"] = float(m.group(2))
        return BenchmarkResult(results=results)


@dataclass
class SysbenchMemory:
    name: str = "ram.sysbench"
    component: str = "ram"
    threads: int = 0
    total_gib: int = 16          # total throughput target
    block_size: str = "1M"
    operation: str = "read"      # 'read' or 'write'

    def __post_init__(self) -> None:
        if self.threads <= 0:
            self.threads = os.cpu_count() or 1

    def params(self) -> dict:
        return {
            "threads": self.threads,
            "total_gib": self.total_gib,
            "block_size": self.block_size,
            "operation": self.operation,
        }

    def run(self) -> BenchmarkResult:
        cmd = [
            require("sysbench"),
            "memory",
            f"--threads={self.threads}",
            f"--memory-block-size={self.block_size}",
            f"--memory-total-size={self.total_gib}G",
            f"--memory-oper={self.operation}",
            "run",
        ]
        sh = run(cmd, timeout=180)
        results: dict = {}
        m = re.search(r"([\d.]+)\s*MiB transferred\s+\(([\d.]+)\s*MiB/sec\)", sh.stdout)
        if m:
            results["transferred_mib"] = float(m.group(1))
            results["bandwidth_mib_s"] = float(m.group(2))
        m = re.search(r"total time:\s*([\d.]+)s", sh.stdout)
        if m:
            results["wall_seconds"] = float(m.group(1))
        return BenchmarkResult(results=results)
