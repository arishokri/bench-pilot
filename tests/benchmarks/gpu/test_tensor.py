import pytest

from tests.conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_matmul_suite_runs_each_dtype():
    from benchpress.benchmarks.gpu.tensor import MatmulSuite

    r = MatmulSuite(matrix=512, seconds_per_dtype=0.3).run().results
    for k in ("fp32", "tf32", "fp16", "bf16"):
        assert k in r
        assert r[k]["tflops"] > 0
        assert r[k]["iters"] > 0
    assert "device" in r


@requires_gpu
@pytest.mark.gpu
def test_conv2d_suite_returns_throughput():
    from benchpress.benchmarks.gpu.tensor import Conv2dSuite

    r = Conv2dSuite(batch=4, channels=16, out_channels=16, spatial=32,
                    kernel=3, seconds=0.3).run().results
    assert r["iters"] > 0
    assert r["images_per_second"] > 0
