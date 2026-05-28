from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class BenchmarkError(RuntimeError):
    pass


@dataclass
class BenchmarkResult:
    results: dict


class Benchmark(Protocol):
    name: str
    component: str

    def params(self) -> dict: ...

    def run(self) -> BenchmarkResult: ...
