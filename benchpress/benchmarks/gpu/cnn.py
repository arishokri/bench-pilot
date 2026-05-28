from __future__ import annotations

import time
from dataclasses import dataclass

from benchpress.benchmarks.base import BenchmarkResult
from benchpress.benchmarks.gpu._torch import cuda_or_skip


@dataclass
class ResNet50Bench:
    """ResNet-50 forward + backward step throughput in FP16/BF16.

    No real data — synthetic tensors so the timing isolates compute.
    """

    name: str = "gpu.cnn.resnet50"
    component: str = "gpu"
    batch: int = 64
    seconds: float = 6.0
    precision: str = "bfloat16"
    do_backward: bool = True

    def params(self) -> dict:
        return {"batch": self.batch, "seconds": self.seconds, "precision": self.precision, "backward": self.do_backward}

    def run(self) -> BenchmarkResult:
        torch = cuda_or_skip()
        if torch is None:
            return BenchmarkResult(results={"skipped": "no CUDA torch"})
        try:
            from torchvision.models import resnet50  # type: ignore
        except ImportError:
            return BenchmarkResult(results={"skipped": "torchvision not installed"})

        torch.backends.cudnn.benchmark = True
        dtype = getattr(torch, self.precision)
        model = resnet50().to("cuda", dtype=dtype)
        model.train(self.do_backward)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3) if self.do_backward else None
        x = torch.randn(self.batch, 3, 224, 224, device="cuda", dtype=dtype)
        y = torch.randint(0, 1000, (self.batch,), device="cuda")
        loss_fn = torch.nn.CrossEntropyLoss()

        # warmup
        for _ in range(2):
            out = model(x)
            if self.do_backward:
                loss = loss_fn(out.float(), y)
                loss.backward()
                opt.step()
                opt.zero_grad()
        torch.cuda.synchronize()

        start = time.monotonic()
        iters = 0
        while (time.monotonic() - start) < self.seconds:
            out = model(x)
            if self.do_backward:
                loss = loss_fn(out.float(), y)
                loss.backward()
                opt.step()
                opt.zero_grad()
            iters += 1
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        ips = (self.batch * iters) / elapsed
        del model, x, y, out
        if self.do_backward:
            del loss, opt
        torch.cuda.empty_cache()
        return BenchmarkResult(results={
            "iters": iters,
            "elapsed_s": elapsed,
            "images_per_second": ips,
            "mode": "train" if self.do_backward else "inference",
        })
