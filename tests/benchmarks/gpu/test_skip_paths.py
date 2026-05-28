"""Every GPU benchmark must return a `skipped` BenchmarkResult when CUDA torch
is unavailable. We force that by monkey-patching `cuda_or_skip` to return None."""

import pytest

from benchpilot.benchmarks.gpu import _torch as torch_helpers
from benchpilot.benchmarks.gpu.cnn import ResNet50Bench
from benchpilot.benchmarks.gpu.image_gen import StableDiffusionInference
from benchpilot.benchmarks.gpu.tensor import Conv2dSuite, MatmulSuite
from benchpilot.benchmarks.gpu.transformer import TinyGPTTrain
from benchpilot.benchmarks.gpu.vram_stress import VramStress


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch_helpers, "cuda_or_skip", lambda: None)
    # Each benchmark imports cuda_or_skip into its own namespace; patch those too.
    for mod_name in (
        "benchpilot.benchmarks.gpu.tensor",
        "benchpilot.benchmarks.gpu.cnn",
        "benchpilot.benchmarks.gpu.transformer",
        "benchpilot.benchmarks.gpu.hf_inference",
        "benchpilot.benchmarks.gpu.image_gen",
        "benchpilot.benchmarks.gpu.vram_stress",
    ):
        import importlib
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "cuda_or_skip"):
            monkeypatch.setattr(mod, "cuda_or_skip", lambda: None)


def test_matmul_skipped(no_cuda):
    assert "skipped" in MatmulSuite().run().results


def test_conv2d_skipped(no_cuda):
    assert "skipped" in Conv2dSuite().run().results


def test_resnet50_skipped(no_cuda):
    assert "skipped" in ResNet50Bench().run().results


def test_tinygpt_skipped(no_cuda):
    assert "skipped" in TinyGPTTrain().run().results


def test_sdxl_skipped(no_cuda):
    assert "skipped" in StableDiffusionInference().run().results


def test_vram_stress_skipped(no_cuda):
    assert "skipped" in VramStress().run().results


# HF benchmarks share the same gating; covered together with a parametrised test.
def test_bert_skipped(no_cuda):
    from benchpilot.benchmarks.gpu.hf_inference import BertInference
    assert "skipped" in BertInference().run().results


def test_vit_skipped(no_cuda):
    from benchpilot.benchmarks.gpu.hf_inference import VitInference
    assert "skipped" in VitInference().run().results
