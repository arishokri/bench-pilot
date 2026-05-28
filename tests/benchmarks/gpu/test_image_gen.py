"""Real SDXL-Turbo inference on the local CUDA device. Gated behind @pytest.mark.gpu.

First run downloads SDXL-Turbo (~6.7 GB) into ./hf_cache/; subsequent runs reuse
the cache. We restrict the test to 1 image at the smallest viable resolution
and a single denoising step to keep wall time around a few seconds.
"""
import pytest

from tests.conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_sdxl_turbo_produces_image():
    from benchpilot.benchmarks.gpu.image_gen import StableDiffusionInference

    r = StableDiffusionInference(
        num_images=1, height=512, width=512, steps=1, guidance=0.0,
    ).run().results
    assert r["images"] == 1
    assert r["elapsed_s"] > 0
    assert r["images_per_second"] > 0
    assert r["seconds_per_image"] > 0
