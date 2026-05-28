"""Real VRAM-stress allocation loop on the local CUDA device. Gated behind
@pytest.mark.gpu. No external downloads — pure torch.empty + mul/add."""
import pytest

from tests.conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_vram_stress_allocates_and_runs_ops():
    from benchpress.benchmarks.gpu.vram_stress import VramStress

    # 30% of VRAM keeps allocation well under the 5080's 16 GB even if other
    # processes are resident, and op_seconds=0.5 finishes the work loop quickly.
    r = VramStress(target_fraction=0.3, op_seconds=0.5).run().results
    assert "skipped" not in r
    assert r["allocated_mib"] > 0
    assert r["peak_mib"] >= r["allocated_mib"]
    assert r["total_mib"] > r["allocated_mib"]
    assert r["ops_iters"] > 0
    assert r["ops_elapsed_s"] > 0
