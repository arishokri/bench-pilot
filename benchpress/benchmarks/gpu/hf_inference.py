from __future__ import annotations

import time
from dataclasses import dataclass

from benchpress.benchmarks.base import BenchmarkResult
from benchpress.benchmarks.gpu._torch import cuda_or_skip


@dataclass
class BertInference:
    name: str = "gpu.hf.bert_base"
    component: str = "gpu"
    batch: int = 16
    seq_len: int = 128
    seconds: float = 5.0

    def params(self) -> dict:
        return {"batch": self.batch, "seq_len": self.seq_len, "seconds": self.seconds}

    def run(self) -> BenchmarkResult:
        torch = cuda_or_skip()
        if torch is None:
            return BenchmarkResult(results={"skipped": "no CUDA torch"})
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError:
            return BenchmarkResult(results={"skipped": "transformers not installed"})

        model_id = "bert-base-uncased"
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()
        prompt = "the quick brown fox jumps over the lazy dog " * (self.seq_len // 10)
        enc = tok([prompt] * self.batch, padding="max_length", truncation=True,
                  max_length=self.seq_len, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            for _ in range(3):
                model(**enc)
            torch.cuda.synchronize()
            start = time.monotonic()
            iters = 0
            while (time.monotonic() - start) < self.seconds:
                model(**enc)
                iters += 1
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        seqs = iters * self.batch
        del model, tok, enc
        torch.cuda.empty_cache()
        return BenchmarkResult(results={
            "model": model_id, "iters": iters, "elapsed_s": elapsed,
            "sequences_per_second": seqs / elapsed,
        })


@dataclass
class VitInference:
    name: str = "gpu.hf.vit_base"
    component: str = "gpu"
    batch: int = 32
    seconds: float = 5.0

    def params(self) -> dict:
        return {"batch": self.batch, "seconds": self.seconds}

    def run(self) -> BenchmarkResult:
        torch = cuda_or_skip()
        if torch is None:
            return BenchmarkResult(results={"skipped": "no CUDA torch"})
        try:
            from transformers import ViTModel  # type: ignore
        except ImportError:
            return BenchmarkResult(results={"skipped": "transformers not installed"})

        model_id = "google/vit-base-patch16-224"
        model = ViTModel.from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()
        x = torch.randn(self.batch, 3, 224, 224, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            for _ in range(3):
                model(pixel_values=x)
            torch.cuda.synchronize()
            start = time.monotonic()
            iters = 0
            while (time.monotonic() - start) < self.seconds:
                model(pixel_values=x)
                iters += 1
            torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        imgs = iters * self.batch
        del model, x
        torch.cuda.empty_cache()
        return BenchmarkResult(results={
            "model": model_id, "iters": iters, "elapsed_s": elapsed,
            "images_per_second": imgs / elapsed,
        })
