# benchpress

End-to-end Linux benchmarking suite. CPU, RAM, SSD, NVIDIA GPU (CUDA / PyTorch / Transformers / Diffusers / Ollama). Records temperature, load, fan RPM, power, clocks and voltage throughout each run and emits a self-contained interactive HTML dashboard.

## Install

```bash
# System tools (sysbench, fio, mbw, stress-ng, sensors, smartctl)
./scripts/install-system-deps.sh

# One-time: detect sensors
sudo sensors-detect --auto

# Python deps incl. PyTorch + CUDA for RTX 50-series (Blackwell, cu128)
uv sync --extra gpu

# Optional: also install the dev extras (pytest + pytest-cov) for running tests
uv sync --extra gpu --extra dev
```

## Use

```bash
# Quick comparable run (~3 min on i9-12900K + RTX 5080)
uv run benchpress run --quick

# Pick components
uv run benchpress run --components cpu,gpu

# Thermal stress test (user-defined duration)
uv run benchpress stress --duration 5m --components cpu,gpu

# List past runs
uv run benchpress list

# Generate / open dashboard
uv run benchpress report --open
```

Data lives in `./data/benchpress.db` (SQLite). The report is a single self-contained HTML file in `./reports/`.

All scratch and downloaded artefacts are kept inside the project:
- `./fio_scratch/` — fio + stress-ng working files (deleted after each run)
- `./hf_cache/` — HuggingFace model downloads (BERT / ViT / SDXL-Turbo), reused across runs
- `./uv_cache/` — uv's wheel + source download cache (configured via `[tool.uv].cache-dir` in `pyproject.toml`)
- `./.venv/` — uv-managed Python venv
- `./data/` — SQLite database
- `./reports/` — generated HTML dashboards

`HF_HOME` is set automatically at run time to `./hf_cache/`. Override either dir with `--ssd-dir` or `--hf-cache-dir`. Ollama keeps its system default (`~/.ollama/`).

## Layout

```
benchpress/
  cli.py            Typer entrypoint
  runner.py         Orchestrates sampler + benchmarks
  storage.py        SQLite schema + writes
  monitor/          Background sensor sampler (~1 Hz)
    nvidia.py       nvidia-smi --query-gpu
    cpu.py          /proc/stat utilisation
    hwmon.py        /sys/class/hwmon (temps, fans, voltage)
    disk.py         smartctl NVMe temps
  benchmarks/
    cpu.py          sysbench
    ram.py          mbw, stream
    ssd.py          fio (seq/rand R/W, IOPS, latency)
    stress.py       stress-ng (time-based)
    gpu/
      tensor.py     matmul FP32/TF32/FP16/BF16, conv2d
      cnn.py        ResNet50 fwd+bwd
      transformer.py  nanoGPT-style train step
      hf_inference.py BERT-base, ViT-base
      ollama_bench.py tok/s on small models
      image_gen.py  SD 1.5 / SDXL turbo inference
      vram_stress.py  graduated allocation
  report/
    generator.py    Jinja2 + Plotly dashboard
```

## Tests

```bash
# Install the test deps (uses the in-project ./uv_cache and ./.venv)
uv sync --extra gpu --extra dev

# Default suite — no CUDA, no system tools required (mocked at boundaries)
uv run pytest

# With coverage
uv run pytest --cov=benchpress --cov-report=term-missing

# Opt-in GPU integration tests (real torch on the local CUDA device)
uv run pytest -m gpu
```

The suite is in `tests/` mirroring the source layout. Tests use these markers (declared in `pyproject.toml`):

- `gpu` — exercises real CUDA torch; skipped automatically when `torch.cuda.is_available()` is False.
- `system_tools` — would call `sysbench`/`fio`/`mbw`/`stress-ng` for real (most tests instead mock `benchpress.benchmarks._shell.run`).
- `slow` — long-running integration paths.

Default `pytest` (no `-m` flag) runs the fast hermetic ~120 tests in under 20 s — they mock `subprocess`, `httpx`, `nvidia-smi`, and use a temp-dir SQLite. Coverage on a default run is ~84 %; the remaining ~16 % is gated GPU/ML code (`benchmarks/gpu/{hf_inference,image_gen,vram_stress}.py`) which is exercised by `pytest -m gpu` on a CUDA-capable machine.

