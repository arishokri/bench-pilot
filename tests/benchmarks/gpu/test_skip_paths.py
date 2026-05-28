"""Every GPU benchmark must return a `skipped` BenchmarkResult when CUDA torch
is unavailable. We force that by monkey-patching `cuda_or_skip` to return None."""

import pytest

from benchpress.benchmarks.gpu import _torch as torch_helpers
from benchpress.benchmarks.gpu.cnn import ResNet50Bench
from benchpress.benchmarks.gpu.image_gen import StableDiffusionInference
from benchpress.benchmarks.gpu.tensor import Conv2dSuite, MatmulSuite
from benchpress.benchmarks.gpu.transformer import TinyGPTTrain
from benchpress.benchmarks.gpu.vram_stress import VramStress


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch_helpers, "cuda_or_skip", lambda: None)
    # Each benchmark imports cuda_or_skip into its own namespace; patch those too.
    for mod_name in (
        "benchpress.benchmarks.gpu.tensor",
        "benchpress.benchmarks.gpu.cnn",
        "benchpress.benchmarks.gpu.transformer",
        "benchpress.benchmarks.gpu.hf_inference",
        "benchpress.benchmarks.gpu.image_gen",
        "benchpress.benchmarks.gpu.vram_stress",
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
    from benchpress.benchmarks.gpu.hf_inference import BertInference
    assert "skipped" in BertInference().run().results


def test_vit_skipped(no_cuda):
    from benchpress.benchmarks.gpu.hf_inference import VitInference
    assert "skipped" in VitInference().run().results
