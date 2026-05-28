import pytest

from tests.conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_tiny_gpt_train_step():
    from benchpilot.benchmarks.gpu.transformer import TinyGPTTrain

    r = TinyGPTTrain(batch=2, seq_len=64, d_model=64, n_head=4, n_layer=1,
                     seconds=0.5).run().results
    assert r["steps"] > 0
    assert r["tokens_per_second"] > 0
