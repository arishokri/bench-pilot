import pytest

from tests.conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_resnet50_train_returns_throughput():
    from benchpilot.benchmarks.gpu.cnn import ResNet50Bench

    r = ResNet50Bench(batch=2, seconds=0.5, do_backward=True).run().results
    assert r["iters"] > 0
    assert r["images_per_second"] > 0
    assert r["mode"] == "train"


@requires_gpu
@pytest.mark.gpu
def test_resnet50_inference_returns_throughput():
    from benchpilot.benchmarks.gpu.cnn import ResNet50Bench

    r = ResNet50Bench(batch=2, seconds=0.5, do_backward=False).run().results
    assert r["images_per_second"] > 0
    assert r["mode"] == "inference"
