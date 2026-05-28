from __future__ import annotations

import time
from dataclasses import dataclass

from benchpress.benchmarks.base import BenchmarkResult
from benchpress.benchmarks.gpu._torch import cuda_or_skip


def _bench_matmul(torch, n: int, dtype, seconds: float, allow_tf32: bool = False) -> tuple[int, float]:
    """Returns (iterations, elapsed)."""
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    # Warmup
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    start = time.monotonic()
    iters = 0
    while (time.monotonic() - start) < seconds:
        c = a @ b
        iters += 1
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    del a, b, c
    torch.cuda.empty_cache()
    return iters, elapsed


def _tflops(n: int, iters: int, elapsed: float) -> float:
    return (2 * n ** 3 * iters) / elapsed / 1e12


@dataclass
class MatmulSuite:
    """Runs square matmul at multiple precisions and reports TFLOP/s for each."""

    name: str = "gpu.tensor.matmul"
    component: str = "gpu"
    matrix: int = 8192
    seconds_per_dtype: float = 4.0

    def params(self) -> dict:
        return {"matrix": self.matrix, "seconds_per_dtype": self.seconds_per_dtype}

    def run(self) -> BenchmarkResult:
        torch = cuda_or_skip()
        if torch is None:
            return BenchmarkResult(results={"skipped": "no CUDA torch"})

        results: dict = {}
        # FP32 (no TF32)
        torch.backends.cuda.matmul.allow_tf32 = False
        i, e = _bench_matmul(torch, self.matrix, torch.float32, self.seconds_per_dtype)
        results["fp32"] = {"tflops": _tflops(self.matrix, i, e), "iters": i, "elapsed_s": e}

        # TF32 (Ampere+)
        torch.backends.cuda.matmul.allow_tf32 = True
        i, e = _bench_matmul(torch, self.matrix, torch.float32, self.seconds_per_dtype, allow_tf32=True)
        results["tf32"] = {"tflops": _tflops(self.matrix, i, e), "iters": i, "elapsed_s": e}

        # FP16
        i, e = _bench_matmul(torch, self.matrix, torch.float16, self.seconds_per_dtype)
        results["fp16"] = {"tflops": _tflops(self.matrix, i, e), "iters": i, "elapsed_s": e}

        # BF16
        i, e = _bench_matmul(torch, self.matrix, torch.bfloat16, self.seconds_per_dtype)
        results["bf16"] = {"tflops": _tflops(self.matrix, i, e), "iters": i, "elapsed_s": e}

        results["device"] = torch.cuda.get_device_name(0)
        return BenchmarkResult(results=results)


@dataclass
class Conv2dSuite:
    """conv2d throughput at a typical ResNet-ish shape."""

    name: str = "gpu.tensor.conv2d"
    component: str = "gpu"
    batch: int = 64
    channels: int = 64
    out_channels: int = 64
    spatial: int = 56
    kernel: int = 3
    seconds: float = 5.0

    def params(self) -> dict:
        return {
            "batch": self.batch, "channels": self.channels, "out_channels": self.out_channels,
            "spatial": self.spatial, "kernel": self.kernel, "seconds": self.seconds,
        }

    def run(self) -> BenchmarkResult:
        torch = cuda_or_skip()
        if torch is None:
            return BenchmarkResult(results={"skipped": "no CUDA torch"})
        torch.backends.cudnn.benchmark = True
        x = torch.randn(self.batch, self.channels, self.spatial, self.spatial, device="cuda", dtype=torch.float16)
        conv = torch.nn.Conv2d(self.channels, self.out_channels, self.kernel, padding=1).to("cuda").half()
        # warmup
        for _ in range(5):
            y = conv(x)
        torch.cuda.synchronize()
        start = time.monotonic()
        iters = 0
        while (time.monotonic() - start) < self.seconds:
            y = conv(x)
            iters += 1
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        ips = (self.batch * iters) / elapsed
        del x, y, conv
        torch.cuda.empty_cache()
        return BenchmarkResult(results={
            "iters": iters,
            "elapsed_s": elapsed,
            "images_per_second": ips,
        })
