from __future__ import annotations

import time
from dataclasses import dataclass

from benchpress.benchmarks.base import BenchmarkResult
from benchpress.benchmarks.gpu._torch import cuda_or_skip


@dataclass
class VramStress:
    """Allocate in steps until just under total, run a few ops, then release.

    Useful in stress mode to characterise sustained memory thermals. In a quick run
    we stop ~80% to avoid OOM-killing other processes.
    """

    name: str = "gpu.vram_stress"
    component: str = "gpu"
    target_fraction: float = 0.8
    op_seconds: float = 4.0

    def params(self) -> dict:
        return {"target_fraction": self.target_fraction, "op_seconds": self.op_seconds}

    def run(self) -> BenchmarkResult:
        torch = cuda_or_skip()
        if torch is None:
            return BenchmarkResult(results={"skipped": "no CUDA torch"})

        total = torch.cuda.get_device_properties(0).total_memory
        # Reserve out current usage
        used = torch.cuda.memory_allocated(0)
        budget = int((total * self.target_fraction) - used)
        if budget <= 0:
            return BenchmarkResult(results={"skipped": "vram already full"})

        # Allocate as float32 buffers in chunks of 256 MiB
        chunk_bytes = 256 * 1024 * 1024
        chunks = []
        allocated = 0
        try:
            while allocated + chunk_bytes <= budget:
                t = torch.empty(chunk_bytes // 4, device="cuda", dtype=torch.float32)
                chunks.append(t)
                allocated += chunk_bytes
        except torch.cuda.OutOfMemoryError:
            pass

        # Do work on the last chunk so we see real bandwidth use
        a = chunks[-1] if chunks else torch.empty(1024 * 1024, device="cuda", dtype=torch.float32)
        start = time.monotonic()
        iters = 0
        while (time.monotonic() - start) < self.op_seconds:
            a.mul_(1.0001).add_(1e-6)
            iters += 1
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start

        peak_mib = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        chunks.clear()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return BenchmarkResult(results={
            "allocated_mib": allocated / (1024 ** 2),
            "peak_mib": peak_mib,
            "total_mib": total / (1024 ** 2),
            "ops_iters": iters,
            "ops_elapsed_s": elapsed,
        })
