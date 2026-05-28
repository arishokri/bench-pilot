"""Real BERT/ViT inference on the local CUDA device. Gated behind @pytest.mark.gpu.

First run downloads BERT-base (~440 MB) and ViT-base (~340 MB) into ./hf_cache/.
Subsequent runs reuse the cache and complete in well under a second each.
"""
import pytest

from tests.conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_bert_inference_returns_throughput():
    from benchpress.benchmarks.gpu.hf_inference import BertInference

    r = BertInference(batch=2, seq_len=32, seconds=0.3).run().results
    assert r["model"] == "bert-base-uncased"
    assert r["iters"] > 0
    assert r["sequences_per_second"] > 0
    assert r["elapsed_s"] > 0


@requires_gpu
@pytest.mark.gpu
def test_vit_inference_returns_throughput():
    from benchpress.benchmarks.gpu.hf_inference import VitInference

    r = VitInference(batch=2, seconds=0.3).run().results
    assert r["model"] == "google/vit-base-patch16-224"
    assert r["iters"] > 0
    assert r["images_per_second"] > 0
