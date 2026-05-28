from __future__ import annotations

from pathlib import Path

from benchpilot.benchmarks.base import Benchmark
from benchpilot.benchmarks.cpu import SysbenchCpu
from benchpilot.benchmarks.gpu.cnn import ResNet50Bench
from benchpilot.benchmarks.gpu.hf_inference import BertInference, VitInference
from benchpilot.benchmarks.gpu.image_gen import StableDiffusionInference
from benchpilot.benchmarks.gpu.ollama_bench import OllamaInference
from benchpilot.benchmarks.gpu.tensor import Conv2dSuite, MatmulSuite
from benchpilot.benchmarks.gpu.transformer import TinyGPTTrain
from benchpilot.benchmarks.gpu.vram_stress import VramStress
from benchpilot.benchmarks.ram import MbwBench, StreamBench, SysbenchMemory
from benchpilot.benchmarks.ssd import FioRandRead, FioRandWrite, FioSeqRead, FioSeqWrite
from benchpilot.benchmarks.stress import CpuStress, GpuStress, RamStress, SsdStress
from benchpilot.config import RunConfig, StressConfig


def build_benchmark_plan(cfg: RunConfig) -> list[Benchmark]:
    """Translate a RunConfig into an ordered list of benchmarks.

    Order: cpu -> ram -> ssd -> gpu. Within gpu: cheap tensor ops first, heavy SD last.
    """
    plan: list[Benchmark] = []

    if "cpu" in cfg.components:
        plan.append(SysbenchCpu(seconds=15 if cfg.quick else 30))

    if "ram" in cfg.components:
        plan.append(MbwBench(size_mib=1024, iterations=3))
        plan.append(StreamBench())
        plan.append(SysbenchMemory(total_gib=8 if cfg.quick else 32, operation="read"))
        plan.append(SysbenchMemory(total_gib=8 if cfg.quick else 32, operation="write"))

    if "ssd" in cfg.components:
        plan.append(FioSeqRead(target_dir=cfg.ssd_target_dir, runtime=10 if cfg.quick else 20))
        plan.append(FioSeqWrite(target_dir=cfg.ssd_target_dir, runtime=10 if cfg.quick else 20))
        plan.append(FioRandRead(target_dir=cfg.ssd_target_dir, runtime=15 if cfg.quick else 30))
        plan.append(FioRandWrite(target_dir=cfg.ssd_target_dir, runtime=15 if cfg.quick else 30))

    if "gpu" in cfg.components:
        s = cfg.gpu_quick_seconds_per_test
        plan.append(MatmulSuite(seconds_per_dtype=max(2.0, s / 2)))
        plan.append(Conv2dSuite(seconds=s))
        plan.append(ResNet50Bench(seconds=s, do_backward=True))
        plan.append(TinyGPTTrain(seconds=s))
        plan.append(BertInference(seconds=s))
        plan.append(VitInference(seconds=s))
        if cfg.ollama_models:
            plan.append(OllamaInference(models=cfg.ollama_models))
        else:
            plan.append(OllamaInference())
        if cfg.include_image_gen:
            plan.append(StableDiffusionInference())
        plan.append(VramStress(target_fraction=0.7, op_seconds=max(3.0, s / 2)))

    return plan


def build_stress_plan(cfg: StressConfig) -> list[Benchmark]:
    plan: list[Benchmark] = []
    d = cfg.duration_seconds
    if "cpu" in cfg.components:
        plan.append(CpuStress(duration_seconds=d))
    if "ram" in cfg.components:
        plan.append(RamStress(duration_seconds=d))
    if "ssd" in cfg.components:
        plan.append(SsdStress(duration_seconds=d, target_dir=Path(cfg.ssd_target_dir)))
    if "gpu" in cfg.components:
        plan.append(GpuStress(duration_seconds=d))
    return plan
