# benchpilot

End-to-end Linux benchmarking suite. CPU, RAM, SSD, NVIDIA GPU (CUDA / PyTorch / Transformers / Diffusers / Ollama). Records temperature, load, fan RPM, power, clocks and voltage throughout each run and emits a self-contained interactive HTML dashboard.

## Install

```bash
# System tools (sysbench, fio, mbw, stress-ng, sensors, smartctl)
./scripts/install-system-deps.sh

# One-time: detect sensors
sudo sensors-detect --auto

# Python deps incl. PyTorch + CUDA for RTX 50-series (Blackwell, cu126)
uv sync --extra gpu
```

## Use

```bash
# Quick comparable run (~3 min on i9-12900K + RTX 5080)
uv run benchpilot run --quick

# Pick components
uv run benchpilot run --components cpu,gpu

# Thermal stress test (user-defined duration)
uv run benchpilot stress --duration 5m --components cpu,gpu

# List past runs
uv run benchpilot list

# Generate / open dashboard
uv run benchpilot report --open
```

Data lives in `./data/benchpilot.db` (SQLite). The report is a single self-contained HTML file in `./reports/`.

## Layout

```
benchpilot/
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
