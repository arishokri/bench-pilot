from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from benchpress.benchmarks.base import BenchmarkResult


_DEFAULT_HOST = "http://127.0.0.1:11434"


def _list_models(host: str) -> list[str]:
    try:
        r = httpx.get(f"{host}/api/tags", timeout=5.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except (httpx.HTTPError, KeyError, ValueError):
        return []


def _bench_one(host: str, model: str, prompt: str, num_predict: int) -> dict:
    """Single non-streaming generate call. ollama reports eval_count + eval_duration."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # keep_alive: 0 makes Ollama unload the model right after the response,
        # so subsequent GPU tests (notably SDXL) have full VRAM available.
        "keep_alive": 0,
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }
    start = time.monotonic()
    r = httpx.post(f"{host}/api/generate", json=payload, timeout=300.0)
    elapsed = time.monotonic() - start
    r.raise_for_status()
    j = r.json()
    # Ollama timings in nanoseconds
    eval_count = j.get("eval_count")  # tokens generated
    eval_dur_ns = j.get("eval_duration") or 0
    prompt_eval_count = j.get("prompt_eval_count") or 0
    prompt_eval_dur_ns = j.get("prompt_eval_duration") or 0
    out: dict = {
        "wall_seconds": elapsed,
        "tokens_generated": eval_count,
        "gen_tokens_per_second": (eval_count / (eval_dur_ns / 1e9)) if eval_count and eval_dur_ns else None,
        "prompt_tokens": prompt_eval_count,
        "prompt_tokens_per_second": (prompt_eval_count / (prompt_eval_dur_ns / 1e9))
            if prompt_eval_count and prompt_eval_dur_ns else None,
    }
    return out


@dataclass
class OllamaInference:
    """Measure prompt eval + token generation tok/s on local Ollama for one or more models."""

    name: str = "gpu.ollama"
    component: str = "gpu"
    host: str = _DEFAULT_HOST
    models: tuple[str, ...] = ()
    prompt: str = "Explain why GPUs are good at matrix multiplication. Be concise."
    num_predict: int = 96
    max_models: int = 2  # cap so the test fits the time budget

    def params(self) -> dict:
        return {"host": self.host, "models": list(self.models), "num_predict": self.num_predict}

    def run(self) -> BenchmarkResult:
        try:
            available = _list_models(self.host)
        except Exception:
            return BenchmarkResult(results={"skipped": "ollama server not reachable"})
        if not available:
            return BenchmarkResult(results={"skipped": "no models pulled in ollama"})

        # If user didn't specify, pick the smallest 1-2 models from those available
        # (heuristic: prefer names containing :2b/:3b/:e2b, then anything else).
        targets = list(self.models) if self.models else _pick_small(available, self.max_models)
        per_model: dict[str, dict] = {}
        for m in targets:
            try:
                per_model[m] = _bench_one(self.host, m, self.prompt, self.num_predict)
            except httpx.HTTPError as e:
                per_model[m] = {"error": str(e)}
        return BenchmarkResult(results={"models": per_model, "available": available})


def _pick_small(models: list[str], n: int) -> list[str]:
    def size_hint(name: str) -> int:
        for marker, score in (
            (":2b", 1), (":e2b", 1), (":1.5b", 1), (":1b", 1),
            (":3b", 2), (":e3b", 2),
            (":4b", 3), (":e4b", 3),
            (":7b", 4), (":8b", 4), (":9b", 4),
            (":13b", 5), (":14b", 5),
            (":20b", 6), (":27b", 6),
        ):
            if marker in name:
                return score
        return 10
    return sorted([m for m in models if "cloud" not in m], key=size_hint)[:n]
