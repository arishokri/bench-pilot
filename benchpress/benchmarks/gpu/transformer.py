from __future__ import annotations

import time
from dataclasses import dataclass

from benchpilot.benchmarks.base import BenchmarkResult
from benchpilot.benchmarks.gpu._torch import cuda_or_skip


@dataclass
class TinyGPTTrain:
    """Throughput of a single transformer training step on a small decoder model.

    Mirrors nanoGPT shape so results are interpretable: tokens/sec & step/sec.
    """

    name: str = "gpu.transformer.train"
    component: str = "gpu"
    batch: int = 8
    seq_len: int = 1024
    d_model: int = 768
    n_head: int = 12
    n_layer: int = 6
    vocab: int = 50257
    seconds: float = 6.0
    precision: str = "bfloat16"

    def params(self) -> dict:
        return {
            "batch": self.batch, "seq_len": self.seq_len, "d_model": self.d_model,
            "n_head": self.n_head, "n_layer": self.n_layer, "vocab": self.vocab,
            "seconds": self.seconds, "precision": self.precision,
        }

    def _build_model(self, torch):
        import torch.nn as nn

        class Block(nn.Module):
            def __init__(self, d_model, n_head):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
                self.ln2 = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
                )

            def forward(self, x, mask):
                h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
                x = x + h
                x = x + self.mlp(self.ln2(x))
                return x

        class TinyGPT(nn.Module):
            def __init__(self, vocab, d_model, n_head, n_layer, seq_len):
                super().__init__()
                self.tok = nn.Embedding(vocab, d_model)
                self.pos = nn.Embedding(seq_len, d_model)
                self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
                self.ln_f = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab, bias=False)

            def forward(self, idx, mask):
                B, T = idx.shape
                pos = torch.arange(T, device=idx.device)
                x = self.tok(idx) + self.pos(pos)
                for b in self.blocks:
                    x = b(x, mask)
                return self.head(self.ln_f(x))

        return TinyGPT(self.vocab, self.d_model, self.n_head, self.n_layer, self.seq_len)

    def run(self) -> BenchmarkResult:
        torch = cuda_or_skip()
        if torch is None:
            return BenchmarkResult(results={"skipped": "no CUDA torch"})

        dtype = getattr(torch, self.precision)
        model = self._build_model(torch).to("cuda", dtype=dtype)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        idx = torch.randint(0, self.vocab, (self.batch, self.seq_len), device="cuda")
        tgt = torch.randint(0, self.vocab, (self.batch, self.seq_len), device="cuda")
        # Mask must match the model dtype; recent PyTorch attention impls reject mixed dtypes.
        mask = torch.triu(
            torch.full((self.seq_len, self.seq_len), float("-inf"), device="cuda", dtype=dtype),
            diagonal=1,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(2):  # warmup
            out = model(idx, mask)
            loss = loss_fn(out.reshape(-1, self.vocab), tgt.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        start = time.monotonic()
        steps = 0
        while (time.monotonic() - start) < self.seconds:
            out = model(idx, mask)
            loss = loss_fn(out.reshape(-1, self.vocab), tgt.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        tokens = steps * self.batch * self.seq_len
        del model, idx, tgt, mask, out, loss, opt
        torch.cuda.empty_cache()
        return BenchmarkResult(results={
            "steps": steps,
            "elapsed_s": elapsed,
            "steps_per_second": steps / elapsed,
            "tokens_per_second": tokens / elapsed,
        })
