from __future__ import annotations

import os
import re
from dataclasses import dataclass

from benchpress.benchmarks._shell import require, run
from benchpress.benchmarks.base import BenchmarkResult


def _parse_sysbench_cpu(stdout: str) -> dict:
    out: dict = {}
    m = re.search(r"events per second:\s*([\d.]+)", stdout)
    if m:
        out["events_per_second"] = float(m.group(1))
    m = re.search(r"total number of events:\s*(\d+)", stdout)
    if m:
        out["total_events"] = int(m.group(1))
    m = re.search(r"total time:\s*([\d.]+)s", stdout)
    if m:
        out["wall_seconds"] = float(m.group(1))
    m = re.search(r"avg:\s*([\d.]+)\s*\n", stdout)
    if m:
        out["latency_ms_avg"] = float(m.group(1))
    m = re.search(r"95th percentile:\s*([\d.]+)\s*\n", stdout)
    if m:
        out["latency_ms_p95"] = float(m.group(1))
    return out


@dataclass
class SysbenchCpu:
    name: str = "cpu.sysbench"
    component: str = "cpu"
    threads: int = 0           # 0 = all logical CPUs
    seconds: int = 15
    cpu_max_prime: int = 20000

    def __post_init__(self) -> None:
        if self.threads <= 0:
            self.threads = os.cpu_count() or 1

    def params(self) -> dict:
        return {
            "threads": self.threads,
            "seconds": self.seconds,
            "cpu_max_prime": self.cpu_max_prime,
        }

    def run(self) -> BenchmarkResult:
        cmd = [
            require("sysbench"),
            "cpu",
            f"--threads={self.threads}",
            f"--time={self.seconds}",
            f"--cpu-max-prime={self.cpu_max_prime}",
            "run",
        ]
        sh = run(cmd, timeout=self.seconds + 30)
        return BenchmarkResult(results=_parse_sysbench_cpu(sh.stdout))
